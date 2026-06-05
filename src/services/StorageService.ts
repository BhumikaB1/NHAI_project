import AsyncStorage from '@react-native-async-storage/async-storage';

export interface AttendanceLog {
  id: string;
  userId: string;
  timestamp: string;
  status: 'SUCCESS' | 'FAILED';
  synced: boolean;
  matchScore?: number;
  liveness?: string;
}

export interface UserProfile {
  userId: string;
  name: string;
  registeredAt: string;
  embedding: number[]; // Real 128-d face embedding
}

const KEYS = {
  ATTENDANCE_LOGS: '@nhai_attendance_logs',
  REGISTERED_PROFILES: '@nhai_registered_profiles',
  SYNC_QUEUE: '@nhai_sync_queue',
};

export class StorageService {
  /**
   * Saves a new attendance log locally.
   */
  static async saveAttendanceLog(
    userId: string,
    status: 'SUCCESS' | 'FAILED',
    synced: boolean = false,
    matchScore?: number,
    liveness?: string
  ): Promise<AttendanceLog> {
    try {
      const logs = await this.getAttendanceLogs();
      const newLog: AttendanceLog = {
        id: `LOG-${Date.now()}-${Math.floor(Math.random() * 1000)}`,
        userId,
        timestamp: new Date().toISOString(),
        status,
        synced,
        matchScore,
        liveness,
      };
      
      logs.unshift(newLog);
      await AsyncStorage.setItem(KEYS.ATTENDANCE_LOGS, JSON.stringify(logs));
      console.log('[StorageService] Saved attendance log:', newLog.id);

      // Add to sync queue if not already synced
      if (!synced) {
        await this.addToSyncQueue(newLog);
      }

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
   * Registers a new user with real face embedding.
   */
  static async registerUser(userId: string, name: string, embedding?: number[]): Promise<UserProfile> {
    try {
      const profiles = await this.getRegisteredUsers();
      
      const existingIdx = profiles.findIndex((p) => p.userId.toLowerCase() === userId.toLowerCase());
      
      // Use provided embedding or generate dummy for demo
      const realEmbedding = embedding || new Array(128).fill(0).map(() => Math.random());
      
      const newProfile: UserProfile = {
        userId,
        name,
        registeredAt: new Date().toISOString(),
        embedding: realEmbedding,
      };

      if (existingIdx >= 0) {
        profiles[existingIdx] = newProfile;
      } else {
        profiles.push(newProfile);
      }

      await AsyncStorage.setItem(KEYS.REGISTERED_PROFILES, JSON.stringify(profiles));
      console.log('[StorageService] Registered profile:', userId, 'with embedding dimension:', newProfile.embedding.length);
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
   * Gets sync queue (unsynced logs).
   */
  static async getSyncQueue(): Promise<AttendanceLog[]> {
    try {
      const data = await AsyncStorage.getItem(KEYS.SYNC_QUEUE);
      return data ? JSON.parse(data) : [];
    } catch (error) {
      console.error('[StorageService] Error getting sync queue:', error);
      return [];
    }
  }

  /**
   * Add log to sync queue.
   */
  private static async addToSyncQueue(log: AttendanceLog): Promise<void> {
    try {
      const queue = await this.getSyncQueue();
      queue.push(log);
      await AsyncStorage.setItem(KEYS.SYNC_QUEUE, JSON.stringify(queue));
      console.log('[StorageService] Added to sync queue:', log.id);
    } catch (error) {
      console.error('[StorageService] Error adding to sync queue:', error);
    }
  }

  /**
   * Clear sync queue after successful sync.
   */
  static async clearSyncQueue(): Promise<void> {
    try {
      await AsyncStorage.setItem(KEYS.SYNC_QUEUE, JSON.stringify([]));
      console.log('[StorageService] Sync queue cleared');
    } catch (error) {
      console.error('[StorageService] Error clearing sync queue:', error);
    }
  }

  /**
   * Resets and clears all storage.
   */
  static async clearAllData(): Promise<void> {
    try {
      await AsyncStorage.removeItem(KEYS.ATTENDANCE_LOGS);
      await AsyncStorage.removeItem(KEYS.REGISTERED_PROFILES);
      await AsyncStorage.removeItem(KEYS.SYNC_QUEUE);
      console.log('[StorageService] Storage cleared successfully');
    } catch (error) {
      console.error('[StorageService] Error clearing storage:', error);
    }
  }
}