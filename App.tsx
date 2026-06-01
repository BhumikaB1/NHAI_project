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
} from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  usePhotoOutput,
  CameraRef,
} from 'react-native-vision-camera';
import NetInfo from '@react-native-community/netinfo';
import { MLService, LIVENESS_PROMPTS, LivenessPrompt } from './src/services/MLService';
import { StorageService, AttendanceLog, UserProfile } from './src/services/StorageService';
import { SyncService } from './src/services/SyncService';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

type ResultState = 'idle' | 'face_detected' | 'liveness_check' | 'liveness_result' | 'processing' | 'success' | 'failed' | 'error';
type SyncStatus = 'idle' | 'syncing' | 'synced' | 'failed';
type DemoErrorType = 'NONE' | 'NO_FACE' | 'MULTIPLE_FACES' | 'POOR_LIGHTING';

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
  }, []);

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

  const startAuthFlow = async () => {
    if (resultState !== 'idle') return;

    // Step 1: FACE DETECTED State (Simulated initialization)
    setResultState('face_detected');
    setStatusMessage('Detecting Face...');

    setTimeout(async () => {
      // Step 2: Check for Forced Error simulations (Phase 9)
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
          console.log('[CAMERA FLOW] STEP 1: Starting capture process');
          setResultState('processing');
          setStatusMessage('Capturing face...');
          
          try {
            // Capture frame to temporary file
            console.log('[CAMERA FLOW] STEP 2: Vision Camera capturePhotoToFile initiated');
            const photoFile = await photoOutput.capturePhotoToFile({}, {});
            console.log('[CAMERA FLOW] STEP 3: Vision Camera photo captured successfully, path:', photoFile.filePath);
            
            setStatusMessage('Matching face...');
            
            // Step 4: ML matching (Phase 4 integration)
            console.log('[CAMERA FLOW] STEP 4: Mock ML face matching initiated');
            const matchResult = await MLService.simulateFaceMatch(photoFile.filePath, demoFaceMatch);
            console.log('[CAMERA FLOW] STEP 5: Face matching completed, success:', matchResult.success);
            
            // Console output matching backend expectations
            console.log('[ML Response]', {
              faceDetected: true,
              liveness: 'PASS',
              similarity: matchResult.success ? (matchResult.confidence / 100) : 0.45,
              authenticated: matchResult.success
            });

            setConfidenceScore(matchResult.confidence);
            
            if (matchResult.success) {
              // Match ID with a registered user if possible
              let matchedId = matchResult.userId || 'UNKNOWN';
              let name = 'Demo User';
              
              if (registeredUsers.length > 0) {
                const randomProfile = registeredUsers[Math.floor(Math.random() * registeredUsers.length)];
                matchedId = randomProfile.userId;
                name = randomProfile.name;
              }
              
              setMatchedUserId(matchedId);
              setMatchedUserName(name);
              setResultState('success');
              setStatusMessage('Authenticated');
              
              // Persist log locally
              await StorageService.saveAttendanceLog(matchedId, 'SUCCESS', false);
            } else {
              setResultState('failed');
              setStatusMessage('Authentication Failed');
              
              await StorageService.saveAttendanceLog('UNKNOWN', 'FAILED', false);
            }
            
            await loadData();
            if (isOnline) triggerSync();

          } catch (err) {
            console.error('[App] Photo capture/match exception:', err);
            setErrorTitle('Camera Failure');
            setErrorMessage('An error occurred while initializing or reading from the camera.');
            setErrorSuggestion('Suggested Action: Ensure no other applications are using the camera and retry.');
            setResultState('error');
            
            await StorageService.saveAttendanceLog('UNKNOWN', 'FAILED', false);
            await loadData();
          }
        }, 1000);
      } else {
        // Liveness Failure (Phase 9)
        setStatusMessage('Liveness: FAIL');
        setErrorTitle('Liveness Failure');
        setErrorMessage('Eye blink or head motion was not verified in the feed.');
        setErrorSuggestion('Suggested Action: Please blink or turn your head slowly in front of the front camera.');
        setResultState('error');
        
        await StorageService.saveAttendanceLog('LIVENESS_FAIL', 'FAILED', false);
        await loadData();
      }
    }, 800);
  };

  // Helper to handle forced error overrides (Phase 9)
  const handleDemoError = async (errorType: DemoErrorType) => {
    setResultState('processing');
    setStatusMessage('Processing...');
    
    setTimeout(async () => {
      switch (errorType) {
        case 'NO_FACE':
          setErrorTitle('No Face Detected');
          setErrorMessage('We couldn\'t find a face in the camera frame.');
          setErrorSuggestion('Suggested Action: Position your face inside the dashed guide frame and ensure good lighting.');
          break;
        case 'MULTIPLE_FACES':
          setErrorTitle('Multiple Faces');
          setErrorMessage('More than one face was found in the frame.');
          setErrorSuggestion('Suggested Action: Ensure only one person is in front of the camera for authentication.');
          break;
        case 'POOR_LIGHTING':
          setErrorTitle('Poor Lighting');
          setErrorMessage('The environment is too dark or has heavy shadows.');
          setErrorSuggestion('Suggested Action: Move to a well-lit area or turn on lights to proceed.');
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

  const handleRegisterUser = async () => {
    const names = ['Harsh Vardhan', 'Amit Kumar', 'Neha Sharma', 'Rohan Gupta', 'Pooja Patil'];
    const randomName = names[Math.floor(Math.random() * names.length)];
    const userId = `USR-${Math.floor(1000 + Math.random() * 9000)}`;

    try {
      await StorageService.registerUser(userId, randomName);
      await loadData();
      Alert.alert('Profile Registered', `Name: ${randomName}\nUser ID: ${userId}\nEmbedding: Extracted`);
    } catch (err) {
      console.error('Failed to register user:', err);
    }
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
        return '#06B6D4'; // Glowing Cyan
      case 'liveness_check':
        return '#3B82F6'; // Electric Blue
      case 'liveness_result':
        return livenessPassed ? '#22C55E' : '#EF4444'; // Green pass, Red fail
      case 'processing':
        return '#EAB308'; // Pulsing Yellow
      case 'success':
        return '#22C55E'; // Green Success
      case 'failed':
        return '#EF4444'; // Red Failure
      case 'error':
        return '#EF4444'; // Red Error
      default:
        return '#3B82F6'; // Seeking Blue
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
          <Text style={styles.errorText}>No front camera device found on this system.</Text>
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
              {isOnline ? '🌐 Online' : '⚠️ Offline Mode'}
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
                 syncStatus === 'synced' ? 'Synced successfully' : 'Sync failed'}
              </Text>
            </View>
          )}
        </View>

        {/* Dynamic Bounding Box Overlay */}
        {!showLogsPanel && !showDemoPanel && (resultState === 'idle' || resultState === 'face_detected' || resultState === 'liveness_check') && (
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
                  ? 'Verifying identity. Please hold still...' 
                  : 'Liveness test failed. Motion discrepancy detected.'
                }
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

        {/* Success Feedback Screen (Phase 4 / 11) */}
        {resultState === 'success' && (
          <View style={[styles.feedbackOverlay, styles.successOverlay]}>
            <View style={styles.feedbackCard}>
              <Text style={styles.feedbackIcon}>✅</Text>
              <Text style={styles.feedbackTitle}>Authenticated</Text>
              
              <View style={styles.metricsBox}>
                <Text style={styles.metricsText}>
                  Match Score: <Text style={styles.highlightText}>{confidenceScore}%</Text>
                </Text>
                <Text style={styles.metricsText}>
                  Liveness: <Text style={styles.highlightText}>PASS</Text>
                </Text>
                <Text style={styles.metricsText}>
                  Status: <Text style={styles.highlightText}>Authenticated</Text>
                </Text>
              </View>

              <Text style={styles.feedbackDescription}>
                Welcome back,{'\n'}
                <Text style={styles.userNameText}>{matchedUserName}</Text> ({matchedUserId})
              </Text>
              
              <TouchableOpacity style={[styles.actionButton, styles.successButton]} onPress={handleReset}>
                <Text style={styles.actionButtonText}>Done</Text>
              </TouchableOpacity>
            </View>
          </View>
        )}

        {/* Failure Feedback Screen (Phase 11) */}
        {resultState === 'failed' && (
          <View style={[styles.feedbackOverlay, styles.failureOverlay]}>
            <View style={styles.feedbackCard}>
              <Text style={styles.feedbackIcon}>❌</Text>
              <Text style={styles.feedbackTitle}>Authentication Failed</Text>
              
              <View style={styles.metricsBox}>
                <Text style={styles.metricsText}>
                  Match Score: <Text style={styles.failHighlightText}>{confidenceScore}%</Text>
                </Text>
                <Text style={styles.metricsText}>
                  Liveness: <Text style={styles.highlightText}>PASS</Text>
                </Text>
                <Text style={styles.metricsText}>
                  Status: <Text style={styles.failHighlightText}>Not Matched</Text>
                </Text>
              </View>

              <Text style={styles.feedbackDescription}>
                The captured face did not match any registered profiles.
              </Text>
              
              <TouchableOpacity style={[styles.actionButton, styles.failureButton]} onPress={handleReset}>
                <Text style={styles.actionButtonText}>Try Again</Text>
              </TouchableOpacity>
            </View>
          </View>
        )}

        {/* Phase 9 Error Handler Screen */}
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
              <Text style={styles.logsTitle}>Local Attendance Logs ({attendanceLogs.length})</Text>
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
                      {item.synced ? 'Synced' : 'Local'}
                    </Text>
                  </View>
                </View>
              )}
              ListEmptyComponent={
                <View style={styles.emptyLogs}>
                  <Text style={styles.emptyLogsText}>No attendance records saved locally.</Text>
                </View>
              }
            />
          </View>
        )}

        {/* Demo Controller Drawer Panel */}
        {showDemoPanel && resultState === 'idle' && (
          <View style={styles.logsDrawer}>
            <View style={styles.logsHeader}>
              <Text style={styles.logsTitle}>⚙️ Hackathon Demo Panel</Text>
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
                  <Text style={styles.demoSectionTitle}>REGISTRATION & STORAGE (PHASE 5)</Text>
                  <View style={styles.demoItemRow}>
                    <View style={{ flex: 1, paddingRight: 10 }}>
                      <Text style={styles.demoItemTitle}>Enroll New Mock User</Text>
                      <Text style={styles.demoItemSubtitle}>
                        Stores profile (userId, timestamp, embedding string) locally. Face DB: {registeredUsers.length} profiles.
                      </Text>
                    </View>
                    <TouchableOpacity style={styles.demoActionButton} onPress={handleRegisterUser}>
                      <Text style={styles.demoActionButtonText}>+ Register User</Text>
                    </TouchableOpacity>
                  </View>

                  <View style={styles.divider} />

                  <Text style={styles.demoSectionTitle}>OUTCOME OVERRIDES (PHASE 10)</Text>
                  
                  <View style={styles.demoSwitchRow}>
                    <View>
                      <Text style={styles.demoItemTitle}>Force Liveness Pass</Text>
                      <Text style={styles.demoItemSubtitle}>
                        {demoLivenessPass ? 'Blink/Turn matches successfully' : 'Motion mismatch/fails'}
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
                      <Text style={styles.demoItemTitle}>Force Match Success</Text>
                      <Text style={styles.demoItemSubtitle}>
                        {demoFaceMatch ? 'Returns Similarity score > 90%' : 'Returns Not Matched'}
                      </Text>
                    </View>
                    <Switch
                      value={demoFaceMatch}
                      onValueChange={setDemoFaceMatch}
                      trackColor={{ false: '#EF4444', true: '#22C55E' }}
                    />
                  </View>

                  <View style={styles.divider} />

                  <Text style={styles.demoSectionTitle}>ERROR SIMULATIONS (PHASE 9)</Text>
                  <View style={styles.errorSelectorContainer}>
                    <Text style={styles.demoItemSubtitle}>Select an issue to mock during authentication:</Text>
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
                            {type === 'NONE' ? 'Standard Match' : 
                             type === 'NO_FACE' ? 'No Face' : 
                             type === 'MULTIPLE_FACES' ? 'Multi-Face' : 'Poor Light'}
                          </Text>
                        </TouchableOpacity>
                      ))}
                    </View>
                  </View>

                  <View style={styles.divider} />

                  <Text style={styles.demoSectionTitle}>OFFLINE QUEUE & SYNC (PHASE 7/8)</Text>

                  <View style={styles.demoSwitchRow}>
                    <View>
                      <Text style={styles.demoItemTitle}>Simulate Offline (Airplane Mode)</Text>
                      <Text style={styles.demoItemSubtitle}>
                        {simulateOffline 
                          ? 'OFFLINE - Attendance saved local-only' 
                          : 'ONLINE - Auto-sync restored logs'}
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
                    <Text style={styles.demoActionButtonText}>🔄 Trigger Manual Sync Queue</Text>
                  </TouchableOpacity>
                </View>
              )}
            />
          </View>
        )}

        {/* Bottom Control panel (only when idle and panels are closed) */}
        {resultState === 'idle' && !showLogsPanel && !showDemoPanel && (
          <View style={styles.controlPanel}>
            
            <View style={styles.panelRow}>
              {/* Database Logs Button */}
              <TouchableOpacity style={[styles.viewLogsButton, { marginRight: 12 }]} onPress={() => setShowLogsPanel(true)}>
                <Text style={styles.viewLogsText}>📂 DB Logs ({attendanceLogs.length})</Text>
              </TouchableOpacity>

              {/* Demo Controller Button */}
              <TouchableOpacity style={styles.viewLogsButton} onPress={() => setShowDemoPanel(true)}>
                <Text style={styles.viewLogsText}>⚙️ Demo Panel</Text>
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
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 5,
  },
  permissionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#F8FAFC',
    marginBottom: 12,
    textAlign: 'center',
    fontFamily: 'System',
  },
  permissionDescription: {
    fontSize: 14,
    color: '#94A3B8',
    textAlign: 'center',
    marginBottom: 24,
    lineHeight: 20,
    fontFamily: 'System',
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
    fontFamily: 'System',
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
    fontFamily: 'System',
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
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3,
    elevation: 3,
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
    fontFamily: 'System',
  },
  syncNotification: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 12,
    borderWidth: 1,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3,
    elevation: 3,
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
    fontFamily: 'System',
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
    shadowColor: '#3B82F6',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.4,
    shadowRadius: 8,
  },
  guideText: {
    marginTop: 20,
    color: '#F8FAFC',
    fontSize: 16,
    fontWeight: '500',
    textShadowColor: 'rgba(0, 0, 0, 0.75)',
    textShadowOffset: { width: 1, height: 1 },
    textShadowRadius: 3,
    fontFamily: 'System',
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
    shadowColor: '#3B82F6',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.3,
    shadowRadius: 10,
  },
  livenessLabel: {
    fontSize: 12,
    fontWeight: '800',
    color: '#3B82F6',
    letterSpacing: 2,
    marginBottom: 10,
    fontFamily: 'System',
  },
  livenessPromptText: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#F8FAFC',
    textAlign: 'center',
    marginBottom: 20,
    fontFamily: 'System',
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
    fontFamily: 'System',
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
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.4,
    shadowRadius: 12,
    elevation: 8,
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
    fontFamily: 'System',
  },
  resultSubtext: {
    fontSize: 14,
    color: '#94A3B8',
    textAlign: 'center',
    lineHeight: 20,
    fontFamily: 'System',
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
    fontFamily: 'System',
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
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.5,
    shadowRadius: 15,
    elevation: 10,
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
    fontFamily: 'System',
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
    fontFamily: 'System',
  },
  feedbackDescription: {
    fontSize: 15,
    color: '#94A3B8',
    textAlign: 'center',
    marginBottom: 24,
    lineHeight: 22,
    fontFamily: 'System',
  },
  userNameText: {
    color: '#FFF',
    fontWeight: 'bold',
  },
  highlightText: {
    color: '#22C55E', // Green
    fontWeight: 'bold',
  },
  failHighlightText: {
    color: '#EF4444', // Red
    fontWeight: 'bold',
  },
  errorMessageText: {
    color: '#EF4444',
    fontSize: 15,
    textAlign: 'center',
    fontWeight: '600',
    marginBottom: 16,
    fontFamily: 'System',
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
    fontFamily: 'System',
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
    fontFamily: 'System',
  },
  // Logs Drawer Styles
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
    shadowColor: '#000',
    shadowOffset: { width: 0, height: -4 },
    shadowOpacity: 0.3,
    shadowRadius: 10,
    elevation: 15,
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
    fontFamily: 'System',
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
    fontFamily: 'System',
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
  successStatusText: {
    fontSize: 18,
    marginRight: 12,
  },
  failedStatusText: {
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
    fontFamily: 'System',
  },
  logTime: {
    color: '#64748B',
    fontSize: 11,
    marginTop: 2,
    fontFamily: 'System',
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
    fontFamily: 'System',
  },
  // Demo Drawer Styles
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
    fontFamily: 'System',
  },
  demoItemRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  demoItemTitle: {
    color: '#F8FAFC',
    fontSize: 14,
    fontWeight: 'bold',
    fontFamily: 'System',
  },
  demoItemSubtitle: {
    color: '#64748B',
    fontSize: 11,
    marginTop: 2,
    lineHeight: 16,
    fontFamily: 'System',
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
    fontFamily: 'System',
  },
  demoSwitchRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
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
    fontFamily: 'System',
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
    fontFamily: 'System',
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
    fontFamily: 'System',
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
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
    elevation: 4,
  },
  captureInnerCircle: {
    width: 66,
    height: 66,
    borderRadius: 33,
    backgroundColor: '#FFFFFF',
  },
});

export default App;
