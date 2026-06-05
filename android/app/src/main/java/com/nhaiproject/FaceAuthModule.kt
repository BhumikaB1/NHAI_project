package com.nhaiproject

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import com.facebook.react.bridge.*
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import android.util.Base64
import org.json.JSONObject
import java.io.File

class FaceAuthModule(reactContext: ReactApplicationContext) 
    : ReactContextBaseJavaModule(reactContext) {

    override fun getName() = "FaceAuthModule"

    private var interpreter: Interpreter? = null
    private val embeddingStore = mutableMapOf<String, FloatArray>()

    @ReactMethod
    fun initialize(promise: Promise) {
        try {
            // Load model from assets
            val assetFd = reactApplicationContext.assets.openFd("mobilefacenet.tflite")
            val inputStream = FileInputStream(assetFd.fileDescriptor)
            val channel = inputStream.channel
            val buffer: MappedByteBuffer = channel.map(
                FileChannel.MapMode.READ_ONLY,
                assetFd.startOffset,
                assetFd.declaredLength
            )
            interpreter = Interpreter(buffer)

            // Load embeddings from assets
            val embJson = reactApplicationContext.assets
                .open("embeddings.json")
                .bufferedReader()
                .readText()
            val json = JSONObject(embJson)
            json.keys().forEach { userId ->
                val arr = json.getJSONObject(userId).getJSONArray("embedding")
                val floats = FloatArray(arr.length()) { arr.getDouble(it).toFloat() }
                embeddingStore[userId] = floats
            }

            promise.resolve("Loaded ${embeddingStore.size} identities")
        } catch (e: Exception) {
            promise.reject("INIT_ERROR", e.message)
        }
    }

    @ReactMethod
    fun getEmbedding(base64Image: String, promise: Promise) {
        try {
            val bytes = Base64.decode(base64Image, Base64.DEFAULT)
            val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
            val resized = Bitmap.createScaledBitmap(bitmap, 112, 112, true)

            // Normalize to [-1, 1]
            val input = Array(1) { Array(112) { Array(112) { FloatArray(3) } } }
            for (y in 0 until 112) {
                for (x in 0 until 112) {
                    val pixel = resized.getPixel(x, y)
                    input[0][y][x][0] = ((pixel shr 16 and 0xFF) / 255f - 0.5f) / 0.5f
                    input[0][y][x][1] = ((pixel shr 8  and 0xFF) / 255f - 0.5f) / 0.5f
                    input[0][y][x][2] = ((pixel        and 0xFF) / 255f - 0.5f) / 0.5f
                }
            }

            val output = Array(1) { FloatArray(128) }
            interpreter?.run(input, output)

            // L2 normalize
            val emb = output[0]
            val norm = Math.sqrt(emb.map { it * it }.sum().toDouble()).toFloat()
            val normalized = emb.map { it / norm }

            val result = Arguments.createArray()
            normalized.forEach { result.pushDouble(it.toDouble()) }
            promise.resolve(result)
        } catch (e: Exception) {
            promise.reject("EMB_ERROR", e.message)
        }
    }

    @ReactMethod
    fun matchEmbedding(embeddingArray: ReadableArray, promise: Promise) {
        try {
            val query = FloatArray(embeddingArray.size()) { 
                embeddingArray.getDouble(it).toFloat() 
            }

            var bestScore = -1f
            var bestUserId = ""

            embeddingStore.forEach { (userId, stored) ->
                var dot = 0f
                for (i in query.indices) dot += query[i] * stored[i]
                if (dot > bestScore) {
                    bestScore = dot
                    bestUserId = userId
                }
            }

            val result = Arguments.createMap()
            result.putDouble("similarity", bestScore.toDouble())
            result.putBoolean("matched", bestScore >= 0.60f)
            result.putString("matchedUserId", if (bestScore >= 0.60f) bestUserId else null)
            promise.resolve(result)
        } catch (e: Exception) {
            promise.reject("MATCH_ERROR", e.message)
        }
    }

    @ReactMethod
    fun registerEmbedding(userId: String, name: String, embeddingArray: ReadableArray, promise: Promise) {
        try {
            val emb = FloatArray(embeddingArray.size()) { 
                embeddingArray.getDouble(it).toFloat() 
            }
            embeddingStore[userId] = emb

            // Persist to internal storage
            val file = File(reactApplicationContext.filesDir, "embeddings.json")
            val json = JSONObject()
            embeddingStore.forEach { (id, arr) ->
                val obj = JSONObject()
                val arrJson = org.json.JSONArray()
                arr.forEach { arrJson.put(it) }
                obj.put("embedding", arrJson)
                obj.put("name", id)
                json.put(id, obj)
            }
            file.writeText(json.toString())
            promise.resolve("Registered $userId")
        } catch (e: Exception) {
            promise.reject("REG_ERROR", e.message)
        }
    }
}