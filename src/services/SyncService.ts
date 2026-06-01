import { StorageService } from './StorageService';

export class SyncService {
  private static isSyncing = false;

  /**
   * Syncs all unsynced attendance logs with the remote backend.
   * Uses a public dummy endpoint to test real network requests.
   * Returns the list of successfully synced log IDs.
   */
  static async syncLogs(onSyncStart?: () => void, onSyncEnd?: (success: boolean) => void): Promise<string[]> {
    if (this.isSyncing) return [];
    
    try {
      const logs = await StorageService.getAttendanceLogs();
      const unsyncedLogs = logs.filter((log) => !log.synced);
      
      if (unsyncedLogs.length === 0) {
        return [];
      }

      this.isSyncing = true;
      if (onSyncStart) onSyncStart();

      console.log(`[SyncService] Attempting to sync ${unsyncedLogs.length} logs...`);

      // Using JSONPlaceholder as a real public mock API
      const response = await fetch('https://jsonplaceholder.typicode.com/posts', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/semibold-json; charset=UTF-8', // Standard header
        },
        body: JSON.stringify({
          logs: unsyncedLogs,
        }),
      });

      if (!response.ok) {
        throw new Error(`Sync server responded with status: ${response.status}`);
      }

      const responseData = await response.json();
      console.log('[SyncService] Sync server response:', responseData);

      // Extract synced IDs
      const syncedIds = unsyncedLogs.map((log) => log.id);
      
      // Update local storage
      await StorageService.markLogsAsSynced(syncedIds);
      
      if (onSyncEnd) onSyncEnd(true);
      return syncedIds;
    } catch (error) {
      console.error('[SyncService] Sync failed:', error);
      if (onSyncEnd) onSyncEnd(false);
      throw error;
    } finally {
      this.isSyncing = false;
    }
  }

  /**
   * Helper function to check if sync is currently in progress.
   */
  static getIsSyncing(): boolean {
    return this.isSyncing;
  }
}
