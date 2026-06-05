import React, { useRef, useState, useEffect } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  SafeAreaView,
  StatusBar,
  Dimensions,
  ActivityIndicator,
  FlatList,
  Switch,
  Alert,
  TextInput,
} from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  usePhotoOutput,
  CameraRef,
} from 'react-native-vision-camera';
import NetInfo from '@react-native-community/netinfo';
import { MLService, LIVENESS_PROMPTS, LivenessPrompt, EmbeddingQuality } from './src/services/MLService';
import { StorageService, AttendanceLog, UserProfile } from './src/services/StorageService';
import { SyncService } from './src/services/SyncService';
import RNFS from 'react-native-fs';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const DUPLICATE_FACE_THRESHOLD = 0.70;

type ResultState = 'idle' | 'face_detected' | 'liveness_check' | 'liveness_result' | 'processing' | 'success' | 'failed' | 'error' | 'registration';
type SyncStatus = 'idle' | 'syncing' | 'synced' | 'failed';
type DemoErrorType = 'NONE' | 'NO_FACE' | 'MULTIPLE_FACES' | 'POOR_LIGHTING';
type RegistrationPose = 'frontal' | 'left' | 'right';

const REGISTRATION_POSES: { id: RegistrationPose; label: string; instruction: string }[] = [
  { id: 'frontal', label: 'Frontal', instruction: 'Look straight at the camera' },
  { id: 'left', label: 'Slight left', instruction: 'Turn your face slightly left' },
  { id: 'right', label: 'Slight right', instruction: 'Turn your face slightly right' },
];

const l2Normalize = (embedding: number[]) => {
  const norm = Math.sqrt(embedding.reduce((sum, value) => sum + value * value, 0));
  return norm > 0 ? embedding.map(value => value / norm) : embedding;
};

const averageEmbeddings = (embeddings: number[][]) => {
  const dim = embeddings[0]?.length || 0;
  const averaged = new Array(dim).fill(0).map((_, index) => {
    const sum = embeddings.reduce((acc, embedding) => acc + embedding[index], 0);
    return sum / embeddings.length;
  });
  return l2Normalize(averaged);
};

const formatQuality = (quality: EmbeddingQuality) =>
  `Brightness ${quality.brightness.toFixed(0)}, Sharpness ${quality.sharpness.toFixed(0)}, Coverage ${(quality.faceCoverage * 100).toFixed(1)}%`;

function App() {
  const { hasPermission, requestPermission } = useCameraPermission();
  const [resultState, setResultState] = useState<ResultState>('idle');
  const [statusMessage, setStatusMessage] = useState<string>('Detecting face...');
  const [isActive, setIsActive] = useState<boolean>(true);
  
  // Liveness States
  const [livenessPrompt, setLivenessPrompt] = useState<LivenessPrompt | null>(null);
  const [livenessProgress, setLivenessProgress] = useState<number>(0);
  const [livenessPassed, setLivenessPassed] = useState<boolean>(false);
  
  // Face Match / ML States
  const [confidenceScore, setConfidenceScore] = useState<number>(0);
  const [matchedUserId, setMatchedUserId] = useState<string>('');
  const [matchedUserName, setMatchedUserName] = useState<string>('');

  // Registration States
  const [registrationName, setRegistrationName] = useState<string>('');
  const [registrationStep, setRegistrationStep] = useState<'name' | 'capture' | 'processing'>('name');
  const [registrationPoseIndex, setRegistrationPoseIndex] = useState<number>(0);
  const [registrationEmbeddings, setRegistrationEmbeddings] = useState<number[][]>([]);
  const [registrationQualities, setRegistrationQualities] = useState<EmbeddingQuality[]>([]);

  // Error Screen States
  const [errorTitle, setErrorTitle] = useState<string>('');
  const [errorMessage, setErrorMessage] = useState<string>('');
  const [errorSuggestion, setErrorSuggestion] = useState<string>('');

  // Local Storage Data
  const [attendanceLogs, setAttendanceLogs] = useState<AttendanceLog[]>([]);
  const [registeredUsers, setRegisteredUsers] = useState<UserProfile[]>([]);
  
  // UI Panels Toggle
  const [showLogsPanel, setShowLogsPanel] = useState<boolean>(false);
  const [showDemoPanel, setShowDemoPanel] = useState<boolean>(false);

  // Sync and Connection States
  const [realIsOnline, setRealIsOnline] = useState<boolean>(true);
  const [simulateOffline, setSimulateOffline] = useState<boolean>(false);
  const [syncStatus, setSyncStatus] = useState<SyncStatus>('idle');

  // Demo panel controls
  const [demoLivenessPass, setDemoLivenessPass] = useState<boolean>(true);
  const [demoFaceMatch, setDemoFaceMatch] = useState<boolean>(true);
  const [demoErrorType, setDemoErrorType] = useState<DemoErrorType>('NONE');

  // Computed online state
  const isOnline = realIsOnline && !simulateOffline;

  const cameraRef = useRef<CameraRef>(null);
  const device = useCameraDevice('front');
  const photoOutput = usePhotoOutput();

  // Monitor connection status
  useEffect(() => {
    const unsubscribe = NetInfo.addEventListener((state) => {
      setRealIsOnline(state.isConnected === true);
    });
    return () => unsubscribe();
  }, []);

  // Initial data loading
  useEffect(() => {
    loadData();
    initializeML();
  }, []);

  // Initialize ML module on app start
  const initializeML = async () => {
    try {
      const result = await MLService.checkHealth();
      console.log('[App] ML Health check:', result);
      const users = await StorageService.getRegisteredUsers();
      await Promise.all(
        users.map(user => MLService.registerUser(user.userId, user.name, user.embedding))
      );
      console.log('[App] Synced local profiles to native matcher:', users.length);
    } catch (error) {
      console.error('[App] ML initialization failed:', error);
    }
  };

  // Automatic sync queue flush on network restoration
  useEffect(() => {
    if (isOnline) {
      triggerSync();
    }
  }, [isOnline]);

  const loadData = async () => {
    const logs = await StorageService.getAttendanceLogs();
    const users = await StorageService.getRegisteredUsers();
    setAttendanceLogs(logs);
    setRegisteredUsers(users);
  };

  const triggerSync = async () => {
    if (!isOnline || SyncService.getIsSyncing()) return;

    try {
      await SyncService.syncLogs(
        () => setSyncStatus('syncing'),
        (success) => {
          if (success) {
            setSyncStatus('synced');
            loadData();
            setTimeout(() => setSyncStatus('idle'), 2500);
          } else {
            setSyncStatus('failed');
            setTimeout(() => setSyncStatus('idle'), 2500);
          }
        }
      );
    } catch (err) {
      console.log('[Sync] Sync failed:', err);
      setSyncStatus('failed');
      setTimeout(() => setSyncStatus('idle'), 2500);
    }
  };

  // REAL FACE REGISTRATION - Start registration
  const startRealFaceRegistration = () => {
    if (!registrationName.trim()) {
      Alert.alert('Enter Name', 'Please enter your name to register');
      return;
    }
    
    setRegistrationPoseIndex(0);
    setRegistrationEmbeddings([]);
    setRegistrationQualities([]);
    setRegistrationStep('capture');
    setStatusMessage(REGISTRATION_POSES[0].instruction);
  };

  // REAL FACE REGISTRATION - Capture frontal, left, and right samples.
  const captureRegistrationFace = async () => {
    try {
      const pose = REGISTRATION_POSES[registrationPoseIndex];
      setRegistrationStep('processing');
      setStatusMessage(`Capturing ${pose.label.toLowerCase()} face...`);

      console.log('[Registration] STEP 1: Capturing face for pose:', pose.id);
      const photoFile = await photoOutput.capturePhotoToFile({}, {});
      console.log('[Registration] STEP 2: Face captured');

      setStatusMessage('Extracting face embedding...');

      console.log('[Registration] STEP 3: Converting to base64...');
      const base64Image = await RNFS.readFile(photoFile.filePath, 'base64');
      console.log('[Registration] STEP 4: Base64 ready');

      console.log('[Registration] STEP 5: Extracting quality-gated embedding...');
      const report = await MLService.getEmbeddingReport(base64Image);
      console.log('[Registration] STEP 6: Embedding report:', report.quality);

      if (!report.quality.passed) {
        Alert.alert('Capture Rejected', report.quality.rejectReason || 'Face quality check failed');
        setRegistrationStep('capture');
        setStatusMessage(pose.instruction);
        return;
      }

      const nextEmbeddings = [...registrationEmbeddings, report.embedding];
      const nextQualities = [...registrationQualities, report.quality];
      setRegistrationEmbeddings(nextEmbeddings);
      setRegistrationQualities(nextQualities);

      if (nextEmbeddings.length < REGISTRATION_POSES.length) {
        const nextIndex = nextEmbeddings.length;
        setRegistrationPoseIndex(nextIndex);
        setRegistrationStep('capture');
        setStatusMessage(REGISTRATION_POSES[nextIndex].instruction);
        return;
      }

      const averagedEmbedding = averageEmbeddings(nextEmbeddings);
      const duplicateCheck = await MLService.matchEmbedding(averagedEmbedding);
      const duplicateScore = (duplicateCheck.scores || [])
        .slice()
        .sort((a, b) => b.similarity - a.similarity)[0];

      if (duplicateScore && duplicateScore.similarity >= DUPLICATE_FACE_THRESHOLD) {
        const duplicatePercent = Math.round(duplicateScore.similarity * 100);
        Alert.alert(
          'Face Already Registered',
          `This face is already close to ${duplicateScore.name} (${duplicateScore.userId}) at ${duplicatePercent}% similarity.\n\nRegistration stopped to prevent duplicate identities.`
        );
        setRegistrationPoseIndex(0);
        setRegistrationEmbeddings([]);
        setRegistrationQualities([]);
        setRegistrationStep('name');
        setStatusMessage('Registration stopped: duplicate face');
        return;
      }

      const userId = `USR-${Date.now()}-${Math.floor(Math.random() * 10000)}`;

      console.log('[Registration] STEP 7: Saving averaged embedding...');
      await StorageService.registerUser(userId, registrationName, averagedEmbedding);
      await MLService.registerUser(userId, registrationName, averagedEmbedding);
      console.log('[Registration] STEP 8: Profile saved successfully!');

      const qualitySummary = nextQualities
        .map((quality, index) => `${REGISTRATION_POSES[index].label}: ${formatQuality(quality)}`)
        .join('\n');

      setStatusMessage('Registration Complete!');
      Alert.alert(
        'Registration Successful!',
        `Name: ${registrationName}\nUser ID: ${userId}\nSamples: ${nextEmbeddings.length}\n\n${qualitySummary}`,
        [
          {
            text: 'Done',
            onPress: () => {
              setRegistrationName('');
              setRegistrationPoseIndex(0);
              setRegistrationEmbeddings([]);
              setRegistrationQualities([]);
              setRegistrationStep('name');
              setResultState('idle');
              loadData();
            },
          },
        ]
      );
    } catch (err) {
      console.error('[Registration] Error:', err);
      Alert.alert('Registration Failed', 'Error: ' + (err as any).message);
      setRegistrationStep('capture');
      setStatusMessage(REGISTRATION_POSES[registrationPoseIndex]?.instruction || 'Ready to capture your face...');
    }
  };
  const startAuthFlow = async () => {
    if (resultState !== 'idle') return;
    if (registeredUsers.length === 0) {
      Alert.alert('No Users Registered', 'Please register your face first using ⚙️ Demo panel!');
      return;
    }

    // Step 1: FACE DETECTED State
    setResultState('face_detected');
    setStatusMessage('Detecting Face...');

    setTimeout(async () => {
      // Step 2: Check for Forced Error simulations
      if (demoErrorType !== 'NONE') {
        handleDemoError(demoErrorType);
        return;
      }

      // Step 3: LIVENESS CHECK State
      const randomPrompt = LIVENESS_PROMPTS[Math.floor(Math.random() * LIVENESS_PROMPTS.length)];
      setLivenessPrompt(randomPrompt);
      setLivenessProgress(0);
      setResultState('liveness_check');
      setStatusMessage(randomPrompt.instruction);

      // Run simulated liveness detection
      const passed = await MLService.simulateLivenessCheck(
        randomPrompt,
        (progress) => setLivenessProgress(progress),
        demoLivenessPass
      );

      setLivenessPassed(passed);
      setResultState('liveness_result');

      if (passed) {
        setStatusMessage('Liveness: PASS');
        
        // Auto-transition to PROCESSING state after 1 second
        setTimeout(async () => {
          console.log('[AUTH] STEP 1: Starting authentication process');
          setResultState('processing');
          setStatusMessage('Capturing your face...');
          
          try {
            // Capture frame to temporary file
            console.log('[AUTH] STEP 2: Capturing photo');
            const photoFile = await photoOutput.capturePhotoToFile({}, {});
            console.log('[AUTH] STEP 3: Photo captured');
            
            setStatusMessage('Extracting face embedding...');

            // Convert photo to base64
            console.log('[AUTH] STEP 4: Converting to base64');
            const base64Image = await RNFS.readFile(photoFile.filePath, 'base64');
            console.log('[AUTH] STEP 5: Base64 ready');

            setStatusMessage('Matching your face...');

            // REAL ML PROCESSING - Get quality-gated embedding from YOUR face NOW
            console.log('[AUTH] STEP 6: Extracting embedding report from your face RIGHT NOW');
            const embeddingReport = await MLService.getEmbeddingReport(base64Image);
            console.log('[AUTH] STEP 7: Current embedding extracted, dimension:', embeddingReport.embedding.length);
            console.log('[AUTH] STEP 7B: Capture quality:', embeddingReport.quality);

            if (!embeddingReport.quality.passed) {
              throw new Error(embeddingReport.quality.rejectReason || 'Face quality check failed');
            }

            // REAL FACE MATCHING - Match against YOUR stored embedding
            console.log('[AUTH] STEP 8: Matching against stored user embeddings');
            const matchResult = await MLService.matchEmbedding(embeddingReport.embedding);
            console.log('[AUTH] STEP 9: Match result:', matchResult);
            console.table(
              (matchResult.scores || [])
                .slice()
                .sort((a, b) => b.similarity - a.similarity)
                .map(score => ({
                  userId: score.userId,
                  name: score.name,
                  similarity: `${(score.similarity * 100).toFixed(2)}%`,
                }))
            );

            // Calculate confidence score
            const confidenceValue = Math.round((matchResult.similarity || 0) * 100);
            setConfidenceScore(confidenceValue);

            if (matchResult.matched && matchResult.matchedUserId) {
              // AUTHENTICATION SUCCESS
              const matchedId = matchResult.matchedUserId;
              
              // Find matching user profile
              const matchedProfile = registeredUsers.find(u => u.userId === matchedId);
              const nativeScore = matchResult.scores?.find(score => score.userId === matchedId);
              const displayName = matchedProfile ? matchedProfile.name : nativeScore?.name || matchedId;

              setMatchedUserId(matchedId);
              setMatchedUserName(displayName);
              setResultState('success');
              setStatusMessage('Authenticated!');

              // Save attendance log
              await StorageService.saveAttendanceLog(
                matchedId,
                'SUCCESS',
                false,
                confidenceValue,
                'PASS'
              );

              console.log('[AUTH] ✅ User authenticated:', matchedId);
            } else {
              // AUTHENTICATION FAILED
              setResultState('failed');
              setStatusMessage('Authentication Failed');

              await StorageService.saveAttendanceLog(
                'UNKNOWN',
                'FAILED',
                false,
                confidenceValue,
                'PASS'
              );

              console.log('[AUTH] ❌ Face did not match any registered user');
            }

            await loadData();
            if (isOnline) triggerSync();

          } catch (err) {
            console.error('[AUTH] Exception:', err);
            setErrorTitle('Camera/ML Failure');
            setErrorMessage('Error during authentication: ' + (err as any).message);
            setErrorSuggestion('Try again with better lighting');
            setResultState('error');
            
            await StorageService.saveAttendanceLog('UNKNOWN', 'FAILED', false);
            await loadData();
          }
        }, 1000);
      } else {
        setStatusMessage('Liveness: FAIL');
        setErrorTitle('Liveness Failure');
        setErrorMessage('Eye blink or head motion was not verified in the feed.');
        setErrorSuggestion('Please blink or turn your head slowly in front of the camera.');
        setResultState('error');
        
        await StorageService.saveAttendanceLog('LIVENESS_FAIL', 'FAILED', false);
        await loadData();
      }
    }, 800);
  };

  // Handle forced error overrides
  const handleDemoError = async (errorType: DemoErrorType) => {
    setResultState('processing');
    setStatusMessage('Processing...');
    
    setTimeout(async () => {
      switch (errorType) {
        case 'NO_FACE':
          setErrorTitle('No Face Detected');
          setErrorMessage('We couldn\'t find a face in the camera frame.');
          setErrorSuggestion('Position your face inside the dashed guide frame.');
          break;
        case 'MULTIPLE_FACES':
          setErrorTitle('Multiple Faces');
          setErrorMessage('More than one face was found in the frame.');
          setErrorSuggestion('Ensure only one person is in front of the camera.');
          break;
        case 'POOR_LIGHTING':
          setErrorTitle('Poor Lighting');
          setErrorMessage('The environment is too dark.');
          setErrorSuggestion('Move to a well-lit area.');
          break;
        default:
          break;
      }
      
      setResultState('error');
      setStatusMessage('Error');
      
      await StorageService.saveAttendanceLog(errorType + '_ERR', 'FAILED', false);
      await loadData();
      if (isOnline) triggerSync();
    }, 1200);
  };

  const handleReset = () => {
    setResultState('idle');
    setLivenessPrompt(null);
    setLivenessProgress(0);
    setConfidenceScore(0);
    setMatchedUserId('');
    setMatchedUserName('');
    setStatusMessage('Detecting face...');
  };

  const clearLogs = async () => {
    await StorageService.clearAllData();
    await loadData();
  };

  // Dynamic color for the bounding face box
  const getGuideBorderColor = () => {
    switch (resultState) {
      case 'face_detected':
        return '#06B6D4';
      case 'liveness_check':
        return '#3B82F6';
      case 'liveness_result':
        return livenessPassed ? '#22C55E' : '#EF4444';
      case 'processing':
        return '#EAB308';
      case 'success':
        return '#22C55E';
      case 'failed':
        return '#EF4444';
      case 'error':
        return '#EF4444';
      case 'registration':
        return '#F59E0B';
      default:
        return '#3B82F6';
    }
  };

  if (!hasPermission) {
    return (
      <SafeAreaView style={styles.permissionContainer}>
        <StatusBar barStyle="light-content" backgroundColor="#0F172A" />
        <View style={styles.permissionCard}>
          <Text style={styles.permissionTitle}>Camera Permission Required</Text>
          <Text style={styles.permissionDescription}>
            This application requires access to your front camera for offline facial authentication.
          </Text>
          <TouchableOpacity style={styles.permissionButton} onPress={requestPermission}>
            <Text style={styles.permissionButtonText}>Grant Access</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <View style={styles.container}>
      <StatusBar translucent backgroundColor="transparent" barStyle="light-content" />
      
      {device != null ? (
        <Camera
          ref={cameraRef}
          style={StyleSheet.absoluteFill}
          device={device}
          isActive={isActive && resultState !== 'success' && resultState !== 'failed' && resultState !== 'error'}
          outputs={[photoOutput]}
        />
      ) : (
        <View style={styles.errorContainer}>
          <Text style={styles.errorText}>No front camera found</Text>
        </View>
      )}

      {/* Main Overlay Elements */}
      <View style={styles.overlayContainer} pointerEvents="box-none">
        
        {/* Top Network & Sync Bar */}
        <View style={styles.topHeaderRow}>
          <View style={[
            styles.networkBadge,
            isOnline ? styles.networkOnlineBadge : styles.networkOfflineBadge
          ]}>
            <Text style={styles.networkBadgeText}>
              {isOnline ? '🌐 Online' : '⚠️ Offline'}
            </Text>
          </View>

          {syncStatus !== 'idle' && (
            <View style={[
              styles.syncNotification,
              syncStatus === 'syncing' ? styles.syncingNotif :
              syncStatus === 'synced' ? styles.syncedNotif : styles.syncFailedNotif
            ]}>
              {syncStatus === 'syncing' && <ActivityIndicator size="small" color="#FFF" style={{ marginRight: 6 }} />}
              <Text style={styles.syncNotifText}>
                {syncStatus === 'syncing' ? 'Syncing...' :
                 syncStatus === 'synced' ? '✅ Synced' : '❌ Sync failed'}
              </Text>
            </View>
          )}
        </View>

        {/* REGISTRATION FLOW */}
        {resultState === 'registration' && registrationStep === 'name' && (
          <View style={styles.registrationOverlay}>
            <View style={styles.registrationCard}>
              <Text style={styles.registrationTitle}>📸 Register Your Face</Text>
              <Text style={styles.registrationSubtitle}>
                Enter your name and we'll capture your face for authentication
              </Text>

              <TextInput
                style={styles.nameInput}
                placeholder="Enter your full name"
                placeholderTextColor="#64748B"
                value={registrationName}
                onChangeText={setRegistrationName}
                maxLength={50}
              />

              <TouchableOpacity
                style={styles.registerButton}
                onPress={startRealFaceRegistration}
              >
                <Text style={styles.registerButtonText}>Next: Capture Face</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.cancelButton}
                onPress={() => {
                  setResultState('idle');
                  setRegistrationName('');
                  setRegistrationPoseIndex(0);
                  setRegistrationEmbeddings([]);
                  setRegistrationQualities([]);
                  setRegistrationStep('name');
                }}
              >
                <Text style={styles.cancelButtonText}>Cancel</Text>
              </TouchableOpacity>
            </View>
          </View>
        )}

        {/* REGISTRATION CAPTURE SCREEN */}
        {resultState === 'registration' && registrationStep === 'capture' && (
          <View style={styles.overlayContainer}>
            <View style={styles.guideContainer} pointerEvents="none">
              <View style={[
                styles.faceGuideFrame,
                { borderColor: getGuideBorderColor() }
              ]} />
              <Text style={styles.guideText}>
                {REGISTRATION_POSES[registrationPoseIndex].instruction} ({registrationPoseIndex + 1}/{REGISTRATION_POSES.length})
              </Text>
            </View>
            
            {/* CAPTURE BUTTON */}
            <TouchableOpacity 
              style={styles.captureButton} 
              onPress={captureRegistrationFace}
            >
              <View style={styles.captureInnerCircle} />
            </TouchableOpacity>
          </View>
        )}

        {/* REGISTRATION PROCESSING */}
        {resultState === 'registration' && registrationStep === 'processing' && (
          <View style={styles.processingOverlay}>
            <ActivityIndicator size="large" color="#3B82F6" />
            <Text style={styles.processingText}>{statusMessage}</Text>
          </View>
        )}

        {/* Dynamic Bounding Box Overlay */}
        {resultState !== 'registration' && !showLogsPanel && !showDemoPanel && (resultState === 'idle' || resultState === 'face_detected' || resultState === 'liveness_check') && (
          <View style={styles.guideContainer} pointerEvents="none">
            <View style={[
              styles.faceGuideFrame,
              { borderColor: getGuideBorderColor() }
            ]} />
            
            {resultState === 'idle' && (
              <Text style={styles.guideText}>Position your face inside the frame</Text>
            )}
            {resultState === 'face_detected' && (
              <Text style={[styles.guideText, { color: '#06B6D4', fontWeight: 'bold' }]}>🟢 Face Detected!</Text>
            )}
          </View>
        )}

        {/* Liveness Check Overlay */}
        {resultState === 'liveness_check' && (
          <View style={styles.livenessCheckOverlay}>
            <View style={styles.instructionCard}>
              <Text style={styles.livenessLabel}>LIVENESS VERIFICATION</Text>
              <Text style={styles.livenessPromptText}>{livenessPrompt?.instruction}</Text>
              
              <View style={styles.progressBarContainer}>
                <View style={[styles.progressBar, { width: `${livenessProgress * 100}%` }]} />
              </View>
              
              <Text style={styles.progressPercentage}>
                {Math.round(livenessProgress * 100)}% analyzed
              </Text>
            </View>
          </View>
        )}

        {/* Liveness Result Notification Overlay */}
        {resultState === 'liveness_result' && (
          <View style={[
            styles.livenessResultOverlay,
            livenessPassed ? styles.livenessPassBG : styles.livenessFailBG
          ]}>
            <View style={styles.resultCard}>
              <Text style={styles.resultIcon}>{livenessPassed ? '✅' : '❌'}</Text>
              <Text style={styles.resultStatusText}>
                Liveness: {livenessPassed ? 'PASS' : 'FAIL'}
              </Text>
              <Text style={styles.resultSubtext}>
                {livenessPassed 
                  ? 'Verifying identity...' 
                  : 'Motion not detected.'}
              </Text>
            </View>
          </View>
        )}

        {/* Processing State View */}
        {resultState === 'processing' && (
          <View style={styles.processingOverlay}>
            <ActivityIndicator size="large" color="#3B82F6" />
            <Text style={styles.processingText}>{statusMessage}</Text>
          </View>
        )}

        {/* Success Feedback Screen */}
        {resultState === 'success' && (
          <View style={[styles.feedbackOverlay, styles.successOverlay]}>
            <View style={styles.feedbackCard}>
              <Text style={styles.feedbackIcon}>✅</Text>
              <Text style={styles.feedbackTitle}>Authenticated!</Text>
              
              <View style={styles.metricsBox}>
                <Text style={styles.metricsText}>
                  Match Score: <Text style={styles.highlightText}>{confidenceScore}%</Text>
                </Text>
                <Text style={styles.metricsText}>
                  Liveness: <Text style={styles.highlightText}>PASS</Text>
                </Text>
                <Text style={styles.metricsText}>
                  Status: <Text style={styles.highlightText}>✅ Matched</Text>
                </Text>
              </View>

              <Text style={styles.feedbackDescription}>
                Welcome back,{'\n'}
                <Text style={styles.userNameText}>{matchedUserName}</Text>{'\n'}
                <Text style={{ fontSize: 12, color: '#94A3B8' }}>({matchedUserId})</Text>
              </Text>
              
              <TouchableOpacity style={[styles.actionButton, styles.successButton]} onPress={handleReset}>
                <Text style={styles.actionButtonText}>Done</Text>
              </TouchableOpacity>
            </View>
          </View>
        )}

        {/* Failure Feedback Screen */}
        {resultState === 'failed' && (
          <View style={[styles.feedbackOverlay, styles.failureOverlay]}>
            <View style={styles.feedbackCard}>
              <Text style={styles.feedbackIcon}>❌</Text>
              <Text style={styles.feedbackTitle}>Not Matched</Text>
              
              <View style={styles.metricsBox}>
                <Text style={styles.metricsText}>
                  Match Score: <Text style={styles.failHighlightText}>{confidenceScore}%</Text>
                </Text>
                <Text style={styles.metricsText}>
                  Liveness: <Text style={styles.highlightText}>PASS</Text>
                </Text>
                <Text style={styles.metricsText}>
                  Status: <Text style={styles.failHighlightText}>❌ No Match</Text>
                </Text>
              </View>

              <Text style={styles.feedbackDescription}>
                Your face didn't match any registered profile.
              </Text>
              
              <TouchableOpacity style={[styles.actionButton, styles.failureButton]} onPress={handleReset}>
                <Text style={styles.actionButtonText}>Try Again</Text>
              </TouchableOpacity>
            </View>
          </View>
        )}

        {/* Error Handler Screen */}
        {resultState === 'error' && (
          <View style={[styles.feedbackOverlay, styles.failureOverlay]}>
            <View style={styles.feedbackCard}>
              <Text style={styles.feedbackIcon}>⚠️</Text>
              <Text style={styles.feedbackTitle}>{errorTitle}</Text>
              
              <Text style={styles.errorMessageText}>{errorMessage}</Text>
              
              <View style={styles.suggestionBox}>
                <Text style={styles.suggestionText}>{errorSuggestion}</Text>
              </View>
              
              <TouchableOpacity style={[styles.actionButton, styles.failureButton]} onPress={handleReset}>
                <Text style={styles.actionButtonText}>Try Again</Text>
              </TouchableOpacity>
            </View>
          </View>
        )}

        {/* Local Storage Logs Drawer */}
        {showLogsPanel && resultState === 'idle' && (
          <View style={styles.logsDrawer}>
            <View style={styles.logsHeader}>
              <Text style={styles.logsTitle}>📂 Logs ({attendanceLogs.length})</Text>
              <View style={styles.logsHeaderActions}>
                <TouchableOpacity style={styles.clearLogsButton} onPress={clearLogs}>
                  <Text style={styles.clearLogsButtonText}>Clear DB</Text>
                </TouchableOpacity>
                <TouchableOpacity style={styles.closeLogsButton} onPress={() => setShowLogsPanel(false)}>
                  <Text style={styles.closeLogsText}>✕</Text>
                </TouchableOpacity>
              </View>
            </View>

            <FlatList
              data={attendanceLogs}
              keyExtractor={(item) => item.id}
              contentContainerStyle={styles.logsList}
              renderItem={({ item }) => (
                <View style={styles.logItem}>
                  <View style={styles.logRowLeft}>
                    <Text style={styles.statusEmoji}>
                      {item.status === 'SUCCESS' ? '✅' : '❌'}
                    </Text>
                    <View style={styles.logInfo}>
                      <Text style={styles.logUserId}>{item.userId}</Text>
                      <Text style={styles.logTime}>
                        {new Date(item.timestamp).toLocaleString()}
                      </Text>
                    </View>
                  </View>
                  <View style={[
                    styles.syncBadge,
                    item.synced ? styles.syncedBadge : styles.unsyncedBadge
                  ]}>
                    <Text style={styles.syncBadgeText}>
                      {item.synced ? '✓ Synced' : '⌛ Local'}
                    </Text>
                  </View>
                </View>
              )}
              ListEmptyComponent={
                <View style={styles.emptyLogs}>
                  <Text style={styles.emptyLogsText}>No logs yet. Authenticate to create logs!</Text>
                </View>
              }
            />
          </View>
        )}

        {/* Demo Controller Drawer Panel */}
        {showDemoPanel && resultState === 'idle' && (
          <View style={styles.logsDrawer}>
            <View style={styles.logsHeader}>
              <Text style={styles.logsTitle}>⚙️ Demo Settings</Text>
              <TouchableOpacity style={styles.closeLogsButton} onPress={() => setShowDemoPanel(false)}>
                <Text style={styles.closeLogsText}>✕</Text>
              </TouchableOpacity>
            </View>

            <FlatList
              data={[1]}
              keyExtractor={(item) => String(item)}
              contentContainerStyle={styles.demoContentContainer}
              renderItem={() => (
                <View>
                  <Text style={styles.demoSectionTitle}>👤 REGISTERED USERS ({registeredUsers.length})</Text>
                  {registeredUsers.length === 0 ? (
                    <Text style={styles.demoItemSubtitle}>No users registered yet. Tap "Register Yourself" below!</Text>
                  ) : (
                    registeredUsers.map(user => (
                      <View key={user.userId} style={styles.userListItem}>
                        <Text style={styles.userListName}>{user.name}</Text>
                        <Text style={styles.userListId}>{user.userId}</Text>
                      </View>
                    ))
                  )}

                  <View style={styles.divider} />

                  <Text style={styles.demoSectionTitle}>📸 REGISTER YOUR FACE</Text>
                  <TouchableOpacity 
                    style={styles.demoActionButton}
                    onPress={() => {
                      setRegistrationName('');
                      setRegistrationStep('name');
                      setResultState('registration');
                      setShowDemoPanel(false);
                    }}
                  >
                    <Text style={styles.demoActionButtonText}>+ Register Yourself</Text>
                  </TouchableOpacity>

                  <View style={styles.divider} />

                  <Text style={styles.demoSectionTitle}>✅ LIVENESS TEST</Text>
                  
                  <View style={styles.demoSwitchRow}>
                    <View>
                      <Text style={styles.demoItemTitle}>Force Liveness Pass</Text>
                      <Text style={styles.demoItemSubtitle}>
                        {demoLivenessPass ? '✓ Always passes' : '✗ Always fails'}
                      </Text>
                    </View>
                    <Switch
                      value={demoLivenessPass}
                      onValueChange={setDemoLivenessPass}
                      trackColor={{ false: '#EF4444', true: '#22C55E' }}
                    />
                  </View>

                  <View style={styles.demoSwitchRow}>
                    <View>
                      <Text style={styles.demoItemTitle}>Force Face Match</Text>
                      <Text style={styles.demoItemSubtitle}>
                        {demoFaceMatch ? '✓ Always matches' : '✗ Always fails'}
                      </Text>
                    </View>
                    <Switch
                      value={demoFaceMatch}
                      onValueChange={setDemoFaceMatch}
                      trackColor={{ false: '#EF4444', true: '#22C55E' }}
                    />
                  </View>

                  <View style={styles.divider} />

                  <Text style={styles.demoSectionTitle}>⚠️ ERROR SIMULATIONS</Text>
                  <View style={styles.errorSelectorContainer}>
                    <View style={styles.errorButtonGrid}>
                      {(['NONE', 'NO_FACE', 'MULTIPLE_FACES', 'POOR_LIGHTING'] as DemoErrorType[]).map((type) => (
                        <TouchableOpacity
                          key={type}
                          style={[
                            styles.errorSelectorButton,
                            demoErrorType === type ? styles.errorSelectorActive : null
                          ]}
                          onPress={() => setDemoErrorType(type)}
                        >
                          <Text style={[
                            styles.errorSelectorText,
                            demoErrorType === type ? styles.errorSelectorTextActive : null
                          ]}>
                            {type === 'NONE' ? 'Standard' : type.replace(/_/g, ' ')}
                          </Text>
                        </TouchableOpacity>
                      ))}
                    </View>
                  </View>

                  <View style={styles.divider} />

                  <Text style={styles.demoSectionTitle}>🔄 OFFLINE & SYNC</Text>

                  <View style={styles.demoSwitchRow}>
                    <View>
                      <Text style={styles.demoItemTitle}>Simulate Offline</Text>
                      <Text style={styles.demoItemSubtitle}>
                        {simulateOffline ? '⚠️ OFFLINE MODE' : '🌐 ONLINE'}
                      </Text>
                    </View>
                    <Switch
                      value={simulateOffline}
                      onValueChange={setSimulateOffline}
                      trackColor={{ false: '#334155', true: '#F59E0B' }}
                      thumbColor={simulateOffline ? '#FFF' : '#F4F3F4'}
                    />
                  </View>

                  <TouchableOpacity 
                    style={[styles.demoActionButton, { width: '100%', paddingVertical: 12, alignItems: 'center', marginTop: 12 }]} 
                    onPress={triggerSync}
                    disabled={!isOnline}
                  >
                    <Text style={styles.demoActionButtonText}>🔄 Manual Sync</Text>
                  </TouchableOpacity>
                </View>
              )}
            />
          </View>
        )}

        {/* Bottom Control panel */}
        {resultState === 'idle' && !showLogsPanel && !showDemoPanel && (
          <View style={styles.controlPanel}>
            
            <View style={styles.panelRow}>
              <TouchableOpacity style={[styles.viewLogsButton, { marginRight: 12 }]} onPress={() => setShowLogsPanel(true)}>
                <Text style={styles.viewLogsText}>📂 Logs ({attendanceLogs.length})</Text>
              </TouchableOpacity>

              <TouchableOpacity style={styles.viewLogsButton} onPress={() => setShowDemoPanel(true)}>
                <Text style={styles.viewLogsText}>⚙️ Demo</Text>
              </TouchableOpacity>
            </View>

            <View style={styles.statusBadge}>
              <Text style={styles.statusText}>{statusMessage}</Text>
            </View>

            <TouchableOpacity style={styles.captureButton} onPress={startAuthFlow}>
              <View style={styles.captureInnerCircle} />
            </TouchableOpacity>
          </View>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  permissionContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#0F172A',
  },
  permissionCard: {
    width: '85%',
    padding: 24,
    borderRadius: 16,
    backgroundColor: '#1E293B',
    alignItems: 'center',
  },
  permissionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#F8FAFC',
    marginBottom: 12,
    textAlign: 'center',
  },
  permissionDescription: {
    fontSize: 14,
    color: '#94A3B8',
    textAlign: 'center',
    marginBottom: 24,
    lineHeight: 20,
  },
  permissionButton: {
    paddingVertical: 12,
    paddingHorizontal: 32,
    borderRadius: 8,
    backgroundColor: '#3B82F6',
  },
  permissionButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#FFFFFF',
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#1E293B',
    padding: 24,
  },
  errorText: {
    color: '#EF4444',
    fontSize: 16,
    textAlign: 'center',
    fontWeight: '500',
  },
  overlayContainer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'space-between',
    paddingBottom: 40,
  },
  topHeaderRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingTop: 60,
    paddingHorizontal: 20,
    width: '100%',
  },
  networkBadge: {
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 12,
    borderWidth: 1,
  },
  networkOnlineBadge: {
    backgroundColor: 'rgba(22, 163, 74, 0.85)',
    borderColor: 'rgba(255, 255, 255, 0.1)',
  },
  networkOfflineBadge: {
    backgroundColor: 'rgba(245, 158, 11, 0.85)',
    borderColor: 'rgba(255, 255, 255, 0.1)',
  },
  networkBadgeText: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: 'bold',
  },
  syncNotification: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 12,
    borderWidth: 1,
  },
  syncingNotif: {
    backgroundColor: 'rgba(59, 130, 246, 0.85)',
    borderColor: 'rgba(255, 255, 255, 0.1)',
  },
  syncedNotif: {
    backgroundColor: 'rgba(22, 163, 74, 0.85)',
    borderColor: 'rgba(255, 255, 255, 0.1)',
  },
  syncFailedNotif: {
    backgroundColor: 'rgba(220, 38, 38, 0.85)',
    borderColor: 'rgba(255, 255, 255, 0.1)',
  },
  syncNotifText: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: 'bold',
  },
  registrationOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(15, 23, 42, 0.95)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
  },
  registrationCard: {
    width: '90%',
    padding: 28,
    borderRadius: 24,
    backgroundColor: '#0F172A',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
  },
  registrationTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#F8FAFC',
    marginBottom: 8,
    textAlign: 'center',
  },
  registrationSubtitle: {
    fontSize: 14,
    color: '#94A3B8',
    textAlign: 'center',
    marginBottom: 24,
    lineHeight: 20,
  },
  nameInput: {
    width: '100%',
    height: 54,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(59, 130, 246, 0.3)',
    paddingHorizontal: 16,
    fontSize: 16,
    color: '#F8FAFC',
    backgroundColor: 'rgba(30, 41, 59, 0.5)',
    marginBottom: 24,
  },
  registerButton: {
    width: '100%',
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
    backgroundColor: '#3B82F6',
    marginBottom: 12,
  },
  registerButtonText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#FFFFFF',
  },
  cancelButton: {
    width: '100%',
    paddingVertical: 12,
    borderRadius: 12,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#64748B',
  },
  cancelButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#94A3B8',
  },
  userListItem: {
    paddingVertical: 12,
    paddingHorizontal: 12,
    marginVertical: 8,
    borderRadius: 8,
    backgroundColor: 'rgba(59, 130, 246, 0.1)',
    borderWidth: 1,
    borderColor: 'rgba(59, 130, 246, 0.3)',
  },
  userListName: {
    color: '#F8FAFC',
    fontSize: 14,
    fontWeight: 'bold',
  },
  userListId: {
    color: '#94A3B8',
    fontSize: 11,
    marginTop: 2,
  },
  guideContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingTop: 40,
  },
  faceGuideFrame: {
    width: SCREEN_WIDTH * 0.75,
    height: SCREEN_WIDTH * 0.95,
    borderWidth: 3,
    borderRadius: 150,
    borderStyle: 'dashed',
    backgroundColor: 'rgba(59, 130, 246, 0.05)',
  },
  guideText: {
    marginTop: 20,
    color: '#F8FAFC',
    fontSize: 16,
    fontWeight: '500',
    textShadowColor: 'rgba(0, 0, 0, 0.75)',
    textShadowOffset: { width: 1, height: 1 },
    textShadowRadius: 3,
  },
  livenessCheckOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(15, 23, 42, 0.4)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  instructionCard: {
    width: '90%',
    padding: 24,
    borderRadius: 20,
    backgroundColor: 'rgba(15, 23, 42, 0.9)',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(59, 130, 246, 0.3)',
  },
  livenessLabel: {
    fontSize: 12,
    fontWeight: '800',
    color: '#3B82F6',
    letterSpacing: 2,
    marginBottom: 10,
  },
  livenessPromptText: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#F8FAFC',
    textAlign: 'center',
    marginBottom: 20,
  },
  progressBarContainer: {
    width: '100%',
    height: 8,
    backgroundColor: '#334155',
    borderRadius: 4,
    overflow: 'hidden',
    marginBottom: 8,
  },
  progressBar: {
    height: '100%',
    backgroundColor: '#3B82F6',
  },
  progressPercentage: {
    fontSize: 12,
    color: '#94A3B8',
  },
  livenessResultOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
  },
  livenessPassBG: {
    backgroundColor: 'rgba(22, 163, 74, 0.4)',
  },
  livenessFailBG: {
    backgroundColor: 'rgba(220, 38, 38, 0.7)',
  },
  resultCard: {
    width: '85%',
    padding: 28,
    borderRadius: 20,
    backgroundColor: '#0F172A',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
  },
  resultIcon: {
    fontSize: 48,
    marginBottom: 12,
  },
  resultStatusText: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#F8FAFC',
    marginBottom: 8,
  },
  resultSubtext: {
    fontSize: 14,
    color: '#94A3B8',
    textAlign: 'center',
    lineHeight: 20,
  },
  processingOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(15, 23, 42, 0.85)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  processingText: {
    marginTop: 16,
    color: '#F8FAFC',
    fontSize: 18,
    fontWeight: '600',
  },
  feedbackOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
  },
  successOverlay: {
    backgroundColor: 'rgba(22, 163, 74, 0.9)',
  },
  failureOverlay: {
    backgroundColor: 'rgba(220, 38, 38, 0.9)',
  },
  feedbackCard: {
    width: '90%',
    padding: 28,
    borderRadius: 24,
    backgroundColor: '#0F172A',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
  },
  feedbackIcon: {
    fontSize: 54,
    marginBottom: 10,
  },
  feedbackTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#F8FAFC',
    marginBottom: 16,
    textAlign: 'center',
  },
  metricsBox: {
    width: '100%',
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.08)',
  },
  metricsText: {
    color: '#F8FAFC',
    fontSize: 14,
    fontWeight: '600',
    marginVertical: 4,
  },
  feedbackDescription: {
    fontSize: 15,
    color: '#94A3B8',
    textAlign: 'center',
    marginBottom: 24,
    lineHeight: 22,
  },
  userNameText: {
    color: '#FFF',
    fontWeight: 'bold',
    fontSize: 16,
  },
  highlightText: {
    color: '#22C55E',
    fontWeight: 'bold',
  },
  failHighlightText: {
    color: '#EF4444',
    fontWeight: 'bold',
  },
  errorMessageText: {
    color: '#EF4444',
    fontSize: 15,
    textAlign: 'center',
    fontWeight: '600',
    marginBottom: 16,
  },
  suggestionBox: {
    width: '100%',
    backgroundColor: 'rgba(234, 179, 8, 0.1)',
    borderRadius: 12,
    padding: 14,
    marginBottom: 24,
    borderWidth: 1,
    borderColor: 'rgba(234, 179, 8, 0.2)',
  },
  suggestionText: {
    color: '#EAB308',
    fontSize: 13,
    fontWeight: '600',
    lineHeight: 18,
    textAlign: 'center',
  },
  actionButton: {
    width: '100%',
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
  },
  successButton: {
    backgroundColor: '#16A34A',
  },
  failureButton: {
    backgroundColor: '#DC2626',
  },
  actionButtonText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#FFFFFF',
  },
  logsDrawer: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    height: '65%',
    backgroundColor: '#0F172A',
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.15)',
    paddingTop: 16,
  },
  logsHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingBottom: 16,
    borderBottomWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.05)',
  },
  logsTitle: {
    color: '#F8FAFC',
    fontSize: 16,
    fontWeight: 'bold',
  },
  logsHeaderActions: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  clearLogsButton: {
    marginRight: 16,
    paddingVertical: 4,
    paddingHorizontal: 8,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#EF4444',
  },
  clearLogsButtonText: {
    color: '#EF4444',
    fontSize: 11,
    fontWeight: '600',
  },
  closeLogsButton: {
    padding: 4,
  },
  closeLogsText: {
    color: '#94A3B8',
    fontSize: 20,
  },
  logsList: {
    padding: 20,
    paddingBottom: 40,
  },
  logItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.05)',
  },
  logRowLeft: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  statusEmoji: {
    fontSize: 18,
    marginRight: 12,
  },
  logInfo: {
    justifyContent: 'center',
  },
  logUserId: {
    color: '#F8FAFC',
    fontSize: 14,
    fontWeight: '600',
  },
  logTime: {
    color: '#64748B',
    fontSize: 11,
    marginTop: 2,
  },
  syncBadge: {
    paddingVertical: 4,
    paddingHorizontal: 8,
    borderRadius: 8,
  },
  syncedBadge: {
    backgroundColor: 'rgba(22, 163, 74, 0.15)',
    borderWidth: 1,
    borderColor: '#16A34A',
  },
  unsyncedBadge: {
    backgroundColor: 'rgba(245, 158, 11, 0.15)',
    borderWidth: 1,
    borderColor: '#F59E0B',
  },
  syncBadgeText: {
    fontSize: 10,
    fontWeight: 'bold',
    color: '#FFF',
  },
  emptyLogs: {
    alignItems: 'center',
    paddingVertical: 40,
  },
  emptyLogsText: {
    color: '#64748B',
    fontSize: 14,
  },
  demoContentContainer: {
    padding: 20,
    paddingBottom: 60,
  },
  demoSectionTitle: {
    color: '#3B82F6',
    fontSize: 11,
    fontWeight: '900',
    letterSpacing: 1.5,
    marginBottom: 12,
  },
  demoItemTitle: {
    color: '#F8FAFC',
    fontSize: 14,
    fontWeight: 'bold',
  },
  demoItemSubtitle: {
    color: '#64748B',
    fontSize: 11,
    marginTop: 2,
    lineHeight: 16,
  },
  demoActionButton: {
    backgroundColor: '#3B82F6',
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 8,
  },
  demoActionButtonText: {
    color: '#FFF',
    fontSize: 12,
    fontWeight: 'bold',
  },
  demoSwitchRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  demoItemRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  divider: {
    height: 1,
    backgroundColor: 'rgba(255, 255, 255, 0.08)',
    marginVertical: 16,
  },
  errorSelectorContainer: {
    width: '100%',
    marginBottom: 8,
  },
  errorButtonGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginTop: 10,
    justifyContent: 'space-between',
  },
  errorSelectorButton: {
    width: '48%',
    backgroundColor: '#1E293B',
    paddingVertical: 10,
    paddingHorizontal: 8,
    borderRadius: 8,
    marginVertical: 6,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.05)',
    alignItems: 'center',
  },
  errorSelectorActive: {
    backgroundColor: 'rgba(239, 68, 68, 0.15)',
    borderColor: '#EF4444',
  },
  errorSelectorText: {
    color: '#94A3B8',
    fontSize: 12,
    fontWeight: '600',
  },
  errorSelectorTextActive: {
    color: '#EF4444',
    fontWeight: 'bold',
  },
  controlPanel: {
    alignItems: 'center',
    width: '100%',
  },
  panelRow: {
    flexDirection: 'row',
    justifyContent: 'center',
    width: '100%',
    marginBottom: 20,
  },
  viewLogsButton: {
    backgroundColor: 'rgba(15, 23, 42, 0.75)',
    paddingVertical: 8,
    paddingHorizontal: 18,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
  },
  viewLogsText: {
    color: '#94A3B8',
    fontSize: 13,
    fontWeight: '600',
  },
  statusBadge: {
    backgroundColor: 'rgba(15, 23, 42, 0.75)',
    paddingVertical: 8,
    paddingHorizontal: 20,
    borderRadius: 20,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
  },
  statusText: {
    color: '#3B82F6',
    fontSize: 15,
    fontWeight: '600',
  },
  captureButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    borderWidth: 4,
    borderColor: '#FFFFFF',
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'transparent',
  },
  captureInnerCircle: {
    width: 66,
    height: 66,
    borderRadius: 33,
    backgroundColor: '#FFFFFF',
  },
});

export default App;
