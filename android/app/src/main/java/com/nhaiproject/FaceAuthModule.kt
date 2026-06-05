package com.nhaiproject

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.PointF
import android.media.ExifInterface
import android.media.FaceDetector
import android.util.Base64
import android.util.Log
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.ReadableArray
import org.json.JSONArray
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.ByteArrayInputStream
import java.io.File
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

class FaceAuthModule(reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    override fun getName() = "FaceAuthModule"

    private data class Identity(
        val name: String,
        val embedding: FloatArray,
    )

    private data class FaceQuality(
        val brightness: Double,
        val sharpness: Double,
        val faceConfidence: Double,
        val faceCoverage: Double,
        val yaw: Double,
        val roll: Double,
        val rejectReason: String?,
    ) {
        val passed: Boolean
            get() = rejectReason == null
    }

    private data class EmbeddingReport(
        val embedding: FloatArray,
        val quality: FaceQuality,
    )

    private var interpreter: Interpreter? = null
    private var embeddingDim = 128
    private val embeddingStore = mutableMapOf<String, Identity>()

    private val threshold = 0.72f
    private val minMargin = 0.04f

    @ReactMethod
    fun initialize(promise: Promise) {
        try {
            val assetFd = reactApplicationContext.assets.openFd("mobilefacenet.tflite")
            val inputStream = FileInputStream(assetFd.fileDescriptor)
            val channel = inputStream.channel
            val buffer: MappedByteBuffer = channel.map(
                FileChannel.MapMode.READ_ONLY,
                assetFd.startOffset,
                assetFd.declaredLength,
            )
            val options = Interpreter.Options().setNumThreads(4)
            interpreter = Interpreter(buffer, options)
            embeddingDim = interpreter?.getOutputTensor(0)?.shape()?.lastOrNull() ?: 128

            loadEmbeddings()
            promise.resolve("Loaded ${embeddingStore.size} identities, embedding dim $embeddingDim")
        } catch (e: Exception) {
            promise.reject("INIT_ERROR", e.message, e)
        }
    }

    @ReactMethod
    fun getEmbedding(base64Image: String, promise: Promise) {
        try {
            val report = extractEmbedding(base64Image)
            if (!report.quality.passed) {
                promise.reject("FACE_QUALITY_ERROR", report.quality.rejectReason)
                return
            }
            promise.resolve(floatArrayToWritable(report.embedding))
        } catch (e: Exception) {
            promise.reject("EMB_ERROR", e.message, e)
        }
    }

    @ReactMethod
    fun getEmbeddingReport(base64Image: String, promise: Promise) {
        try {
            val report = extractEmbedding(base64Image)
            val result = Arguments.createMap()
            result.putArray("embedding", floatArrayToWritable(report.embedding))
            result.putMap("quality", qualityToWritable(report.quality))
            promise.resolve(result)
        } catch (e: Exception) {
            promise.reject("EMB_REPORT_ERROR", e.message, e)
        }
    }

    @ReactMethod
    fun matchEmbedding(embeddingArray: ReadableArray, promise: Promise) {
        try {
            val query = normalize(readableToFloatArray(embeddingArray))
            val scores = Arguments.createArray()
            var bestScore = -1f
            var secondBestScore = -1f
            var bestUserId: String? = null

            embeddingStore.forEach { (userId, identity) ->
                val score = cosine(query, identity.embedding)
                val scoreMap = Arguments.createMap()
                scoreMap.putString("userId", userId)
                scoreMap.putString("name", identity.name)
                scoreMap.putDouble("similarity", score.toDouble())
                scores.pushMap(scoreMap)

                if (score > bestScore) {
                    secondBestScore = bestScore
                    bestScore = score
                    bestUserId = userId
                } else if (score > secondBestScore) {
                    secondBestScore = score
                }
            }

            val margin = if (secondBestScore < 0f) bestScore else bestScore - secondBestScore
            val matched = bestScore >= threshold && margin >= minMargin

            val result = Arguments.createMap()
            result.putDouble("similarity", bestScore.toDouble())
            result.putDouble("secondBestSimilarity", secondBestScore.toDouble())
            result.putDouble("margin", margin.toDouble())
            result.putDouble("threshold", threshold.toDouble())
            result.putDouble("minMargin", minMargin.toDouble())
            result.putBoolean("matched", matched)
            result.putString("matchedUserId", if (matched) bestUserId else null)
            result.putArray("scores", scores)
            promise.resolve(result)
        } catch (e: Exception) {
            promise.reject("MATCH_ERROR", e.message, e)
        }
    }

    @ReactMethod
    fun registerEmbedding(userId: String, name: String, embeddingArray: ReadableArray, promise: Promise) {
        try {
            val emb = normalize(readableToFloatArray(embeddingArray))
            embeddingStore[userId] = Identity(name = name, embedding = emb)
            saveEmbeddings()
            promise.resolve("Registered $userId")
        } catch (e: Exception) {
            promise.reject("REG_ERROR", e.message, e)
        }
    }

    private fun extractEmbedding(base64Image: String): EmbeddingReport {
        val bitmap = decodeBase64Bitmap(base64Image)
        val faceData = detectAndAlign(bitmap)
        val input = bitmapToModelInput(faceData.first)
        val output = Array(1) { FloatArray(embeddingDim) }
        interpreter?.run(input, output) ?: throw IllegalStateException("Interpreter is not initialized")
        return EmbeddingReport(normalize(output[0]), faceData.second)
    }

    private fun decodeBase64Bitmap(base64Image: String): Bitmap {
        val bytes = Base64.decode(base64Image, Base64.DEFAULT)
        val decoded = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
            ?: throw IllegalArgumentException("Could not decode image")
        val orientationDegrees = readExifRotation(bytes)
        val oriented = rotateBitmap(decoded.copy(Bitmap.Config.ARGB_8888, false), orientationDegrees)
        Log.d("FaceAuthModule", "Decoded bitmap ${decoded.width}x${decoded.height}, EXIF rotation=$orientationDegrees")
        return oriented
    }

    private fun detectAndAlign(bitmap: Bitmap): Pair<Bitmap, FaceQuality> {
        val candidate = findDetectableOrientation(bitmap)
        val scale = min(1.0f, 640.0f / max(candidate.width, candidate.height).toFloat())
        val detectWidth = makeEven(max(2, (candidate.width * scale).toInt()))
        val detectHeight = max(2, (candidate.height * scale).toInt())
        val detectionBitmap = Bitmap.createScaledBitmap(candidate, detectWidth, detectHeight, true)
            .copy(Bitmap.Config.RGB_565, false)

        @Suppress("UNCHECKED_CAST")
        val faces = arrayOfNulls<FaceDetector.Face>(5) as Array<FaceDetector.Face>
        val count = FaceDetector(detectionBitmap.width, detectionBitmap.height, faces.size)
            .findFaces(detectionBitmap, faces)

        if (count == 0) {
            throw IllegalArgumentException("No face detected")
        }
        if (count > 1) {
            throw IllegalArgumentException("Multiple faces detected")
        }

        val face = faces[0]
        val midpoint = PointF()
        face.getMidPoint(midpoint)
        val inverseScale = 1.0f / scale
        val midX = midpoint.x * inverseScale
        val midY = midpoint.y * inverseScale
        val eyesDistance = face.eyesDistance() * inverseScale
        val roll = face.pose(FaceDetector.Face.EULER_Z)
        val yaw = face.pose(FaceDetector.Face.EULER_Y)

        val canonicalEyeDistance = 73.5318f - 38.2946f
        val canonicalMidX = (38.2946f + 73.5318f) / 2f
        val canonicalMidY = (51.6963f + 51.5014f) / 2f
        val transform = Matrix().apply {
            postTranslate(-midX, -midY)
            postRotate(-roll)
            postScale(canonicalEyeDistance / eyesDistance, canonicalEyeDistance / eyesDistance)
            postTranslate(canonicalMidX, canonicalMidY)
        }

        val aligned = Bitmap.createBitmap(112, 112, Bitmap.Config.ARGB_8888)
        Canvas(aligned).drawBitmap(candidate, transform, null)
        val quality = evaluateQuality(candidate, aligned, midX, midY, eyesDistance, yaw, roll, face.confidence())
        return Pair(aligned, quality)
    }

    private fun readExifRotation(bytes: ByteArray): Int {
        return try {
            when (
                ExifInterface(ByteArrayInputStream(bytes)).getAttributeInt(
                    ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_NORMAL,
                )
            ) {
                ExifInterface.ORIENTATION_ROTATE_90 -> 90
                ExifInterface.ORIENTATION_ROTATE_180 -> 180
                ExifInterface.ORIENTATION_ROTATE_270 -> 270
                else -> 0
            }
        } catch (e: Exception) {
            Log.w("FaceAuthModule", "Could not read EXIF orientation", e)
            0
        }
    }

    private fun rotateBitmap(bitmap: Bitmap, degrees: Int): Bitmap {
        if (degrees == 0) return bitmap
        val matrix = Matrix().apply { postRotate(degrees.toFloat()) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            .copy(Bitmap.Config.ARGB_8888, false)
    }

    private fun findDetectableOrientation(bitmap: Bitmap): Bitmap {
        val candidates = listOf(
            Pair(0, bitmap),
            Pair(90, rotateBitmap(bitmap, 90)),
            Pair(180, rotateBitmap(bitmap, 180)),
            Pair(270, rotateBitmap(bitmap, 270)),
        )

        var foundMultipleFaces = false
        candidates.forEach { (rotation, candidate) ->
            val scale = min(1.0f, 640.0f / max(candidate.width, candidate.height).toFloat())
            val detectWidth = makeEven(max(2, (candidate.width * scale).toInt()))
            val detectHeight = max(2, (candidate.height * scale).toInt())
            val detectionBitmap = Bitmap.createScaledBitmap(candidate, detectWidth, detectHeight, true)
                .copy(Bitmap.Config.RGB_565, false)
            @Suppress("UNCHECKED_CAST")
            val faces = arrayOfNulls<FaceDetector.Face>(5) as Array<FaceDetector.Face>
            val count = FaceDetector(detectionBitmap.width, detectionBitmap.height, faces.size)
                .findFaces(detectionBitmap, faces)

            Log.d("FaceAuthModule", "Face detection rotation=$rotation count=$count size=${candidate.width}x${candidate.height}")
            if (count == 1) return candidate
            if (count > 1) foundMultipleFaces = true
        }

        if (foundMultipleFaces) {
            throw IllegalArgumentException("Multiple faces detected")
        }
        throw IllegalArgumentException("No face detected")
    }

    private fun evaluateQuality(
        fullBitmap: Bitmap,
        aligned: Bitmap,
        midX: Float,
        midY: Float,
        eyesDistance: Float,
        yaw: Float,
        roll: Float,
        confidence: Float,
    ): FaceQuality {
        val brightness = meanBrightness(aligned)
        val sharpness = laplacianVariance(aligned)
        val boxLeft = midX - eyesDistance * 2.2f
        val boxTop = midY - eyesDistance * 1.7f
        val boxRight = midX + eyesDistance * 2.2f
        val boxBottom = midY + eyesDistance * 3.1f
        val faceCoverage = ((boxRight - boxLeft) * (boxBottom - boxTop)) /
            (fullBitmap.width.toFloat() * fullBitmap.height.toFloat())

        val rejectReason = when {
            confidence < 0.35f -> "Face confidence is too low"
            boxLeft < 0 || boxTop < 0 || boxRight > fullBitmap.width || boxBottom > fullBitmap.height ->
                "Face is partially outside the frame"
            faceCoverage < 0.04f -> "Face is too small in the frame"
            faceCoverage > 0.85f -> "Face is too close to the camera"
            brightness < 55.0 -> "Face is too dark"
            brightness > 235.0 -> "Face is overexposed"
            sharpness < 45.0 -> "Face is blurry"
            abs(yaw) > 35.0f -> "Face angle is too extreme"
            abs(roll) > 20.0f -> "Face is too tilted"
            else -> null
        }

        return FaceQuality(
            brightness = brightness,
            sharpness = sharpness,
            faceConfidence = confidence.toDouble(),
            faceCoverage = faceCoverage.toDouble(),
            yaw = yaw.toDouble(),
            roll = roll.toDouble(),
            rejectReason = rejectReason,
        )
    }

    private fun bitmapToModelInput(bitmap: Bitmap): Array<Array<Array<FloatArray>>> {
        val input = Array(1) { Array(112) { Array(112) { FloatArray(3) } } }
        for (y in 0 until 112) {
            for (x in 0 until 112) {
                val pixel = bitmap.getPixel(x, y)
                input[0][y][x][0] = ((pixel shr 16 and 0xFF) / 255f - 0.5f) / 0.5f
                input[0][y][x][1] = ((pixel shr 8 and 0xFF) / 255f - 0.5f) / 0.5f
                input[0][y][x][2] = ((pixel and 0xFF) / 255f - 0.5f) / 0.5f
            }
        }
        return input
    }

    private fun loadEmbeddings() {
        embeddingStore.clear()
        val file = File(reactApplicationContext.filesDir, "embeddings.json")
        val raw = if (file.exists() && file.length() > 0) {
            file.readText()
        } else {
            reactApplicationContext.assets.open("embeddings.json").bufferedReader().readText()
        }
        val json = JSONObject(raw)
        json.keys().forEach { userId ->
            val obj = json.getJSONObject(userId)
            val arr = obj.getJSONArray("embedding")
            val floats = FloatArray(arr.length()) { arr.getDouble(it).toFloat() }
            val name = obj.optString("name", userId)
            embeddingStore[userId] = Identity(name = name, embedding = normalize(floats))
        }
    }

    private fun saveEmbeddings() {
        val json = JSONObject()
        embeddingStore.forEach { (id, identity) ->
            val obj = JSONObject()
            val arrJson = JSONArray()
            identity.embedding.forEach { arrJson.put(it) }
            obj.put("embedding", arrJson)
            obj.put("name", identity.name)
            json.put(id, obj)
        }
        File(reactApplicationContext.filesDir, "embeddings.json").writeText(json.toString())
    }

    private fun normalize(embedding: FloatArray): FloatArray {
        val norm = sqrt(embedding.fold(0.0) { acc, value -> acc + value * value }).toFloat()
        if (norm <= 0f) return embedding
        return FloatArray(embedding.size) { embedding[it] / norm }
    }

    private fun cosine(a: FloatArray, b: FloatArray): Float {
        val limit = min(a.size, b.size)
        var dot = 0f
        for (i in 0 until limit) {
            dot += a[i] * b[i]
        }
        return dot
    }

    private fun readableToFloatArray(array: ReadableArray): FloatArray =
        FloatArray(array.size()) { array.getDouble(it).toFloat() }

    private fun floatArrayToWritable(array: FloatArray) =
        Arguments.createArray().apply {
            array.forEach { pushDouble(it.toDouble()) }
        }

    private fun qualityToWritable(quality: FaceQuality) =
        Arguments.createMap().apply {
            putBoolean("passed", quality.passed)
            putString("rejectReason", quality.rejectReason)
            putDouble("brightness", quality.brightness)
            putDouble("sharpness", quality.sharpness)
            putDouble("faceConfidence", quality.faceConfidence)
            putDouble("faceCoverage", quality.faceCoverage)
            putDouble("yaw", quality.yaw)
            putDouble("roll", quality.roll)
        }

    private fun meanBrightness(bitmap: Bitmap): Double {
        var sum = 0.0
        for (y in 0 until bitmap.height) {
            for (x in 0 until bitmap.width) {
                val pixel = bitmap.getPixel(x, y)
                sum += 0.299 * (pixel shr 16 and 0xFF) +
                    0.587 * (pixel shr 8 and 0xFF) +
                    0.114 * (pixel and 0xFF)
            }
        }
        return sum / (bitmap.width * bitmap.height)
    }

    private fun laplacianVariance(bitmap: Bitmap): Double {
        val width = bitmap.width
        val height = bitmap.height
        val gray = Array(height) { DoubleArray(width) }
        for (y in 0 until height) {
            for (x in 0 until width) {
                val pixel = bitmap.getPixel(x, y)
                gray[y][x] = 0.299 * (pixel shr 16 and 0xFF) +
                    0.587 * (pixel shr 8 and 0xFF) +
                    0.114 * (pixel and 0xFF)
            }
        }

        val values = mutableListOf<Double>()
        for (y in 1 until height - 1) {
            for (x in 1 until width - 1) {
                values.add(
                    -4.0 * gray[y][x] +
                        gray[y - 1][x] +
                        gray[y + 1][x] +
                        gray[y][x - 1] +
                        gray[y][x + 1],
                )
            }
        }
        val mean = values.average()
        return values.fold(0.0) { acc, value -> acc + (value - mean) * (value - mean) } / values.size
    }

    private fun makeEven(value: Int): Int = if (value % 2 == 0) value else value - 1
}
