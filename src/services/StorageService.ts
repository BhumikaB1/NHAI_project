import AsyncStorage from '@react-native-async-storage/async-storage';

export interface AttendanceLog {
  id: string; // unique UUID or timestamp-based ID
  userId: string;
  timestamp: string; // ISO timestamp
  status: 'SUCCESS' | 'FAILED';
  synced: boolean;
}

export interface UserProfile {
  userId: string;
  name: string;
  registeredAt: string;
  embedding: string;
}

const KEYS = {
  ATTENDANCE_LOGS: '@nhai_attendance_logs',
  REGISTERED_PROFILES: '@nhai_registered_profiles',
};

export class StorageService {
  /**
   * Saves a new attendance log locally.
   */
  static async saveAttendanceLog(
    userId: string,
    status: 'SUCCESS' | 'FAILED',
    synced: boolean = false
  ): Promise<AttendanceLog> {
    try {
      const logs = await this.getAttendanceLogs();
      const newLog: AttendanceLog = {
        id: `LOG-${Date.now()}-${Math.floor(Math.random() * 1000)}`,
        userId,
        timestamp: new Date().toISOString(),
        status,
        synced,
      };
      
      logs.unshift(newLog); // Prepend so new logs show at top
      await AsyncStorage.setItem(KEYS.ATTENDANCE_LOGS, JSON.stringify(logs));
      console.log('[StorageService] Saved attendance log:', newLog);
      return newLog;
    } catch (error) {
      console.error('[StorageService] Error saving attendance log:', error);
      throw error;
    }
  }

  /**
   * Retrieves all stored attendance logs.
   */
  static async getAttendanceLogs(): Promise<AttendanceLog[]> {
    try {
      const data = await AsyncStorage.getItem(KEYS.ATTENDANCE_LOGS);
      return data ? JSON.parse(data) : [];
    } catch (error) {
      console.error('[StorageService] Error getting attendance logs:', error);
      return [];
    }
  }

  /**
   * Marks a set of log IDs as synced.
   */
  static async markLogsAsSynced(logIds: string[]): Promise<void> {
    try {
      const logs = await this.getAttendanceLogs();
      const updatedLogs = logs.map((log) => {
        if (logIds.includes(log.id)) {
          return { ...log, synced: true };
        }
        return log;
      });
      await AsyncStorage.setItem(KEYS.ATTENDANCE_LOGS, JSON.stringify(updatedLogs));
      console.log('[StorageService] Marked logs as synced:', logIds);
    } catch (error) {
      console.error('[StorageService] Error marking logs as synced:', error);
    }
  }

  /**
   * Registers a new face profile.
   */
  static async registerUser(userId: string, name: string): Promise<UserProfile> {
    try {
      const profiles = await this.getRegisteredUsers();
      
      // Check if user already exists
      const existingIdx = profiles.findIndex((p) => p.userId.toLowerCase() === userId.toLowerCase());
      const newProfile: UserProfile = {
        userId,
        name,
        registeredAt: new Date().toISOString(),
        embedding: "EMB-" + Math.random().toString(36).substring(2, 10).toUpperCase(),
      };

      if (existingIdx >= 0) {
        profiles[existingIdx] = newProfile;
      } else {
        profiles.push(newProfile);
      }

      await AsyncStorage.setItem(KEYS.REGISTERED_PROFILES, JSON.stringify(profiles));
      console.log('[StorageService] Registered profile:', newProfile);
      return newProfile;
    } catch (error) {
      console.error('[StorageService] Error registering user:', error);
      throw error;
    }
  }

  /**
   * Gets all registered user profiles.
   */
  static async getRegisteredUsers(): Promise<UserProfile[]> {
    try {
      const data = await AsyncStorage.getItem(KEYS.REGISTERED_PROFILES);
      return data ? JSON.parse(data) : [];
    } catch (error) {
      console.error('[StorageService] Error getting registered users:', error);
      return [];
    }
  }

  /**
   * Resets and clears all storage (useful for demo resets).
   */
  static async clearAllData(): Promise<void> {
    try {
      await AsyncStorage.removeItem(KEYS.ATTENDANCE_LOGS);
      await AsyncStorage.removeItem(KEYS.REGISTERED_PROFILES);
      console.log('[StorageService] Storage cleared successfully');
    } catch (error) {
      console.error('[StorageService] Error clearing storage:', error);
    }
  }
}
