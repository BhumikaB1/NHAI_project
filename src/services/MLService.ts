import { NativeModules } from 'react-native';

const { FaceAuthModule } = NativeModules;

// Types
export interface LivenessPrompt {
  id: string;
  instruction: string;
  action: 'blink' | 'turn_head_left' | 'turn_head_right' | 'nod';
  duration: number;
}

// Liveness prompts
export const LIVENESS_PROMPTS: LivenessPrompt[] = [
  { id: 'blink', instruction: 'Blink your eyes slowly', action: 'blink', duration: 3000 },
  { id: 'turn_left', instruction: 'Turn your head to the left', action: 'turn_head_left', duration: 3000 },
  { id: 'turn_right', instruction: 'Turn your head to the right', action: 'turn_head_right', duration: 3000 },
  { id: 'nod', instruction: 'Nod your head up and down', action: 'nod', duration: 3000 },
];

class MLServiceClass {
  private initialized = false;

  async checkHealth(): Promise<string> {
    try {
      if (!this.initialized) {
        const result = await FaceAuthModule.initialize();
        this.initialized = true;
        console.log('[MLService] Health check:', result);
        return result || 'ML initialized';
      }
      return 'ML already initialized';
    } catch (error) {
      console.error('[MLService] Health check failed:', error);
      throw error;
    }
  }

  async getEmbedding(base64Image: string): Promise<number[]> {
    return await FaceAuthModule.getEmbedding(base64Image);
  }

  async matchEmbedding(embedding: number[]): Promise<any> {
    return await FaceAuthModule.matchEmbedding(embedding);
  }

  async registerUser(userId: string, name: string, embedding: number[]): Promise<string> {
    return await FaceAuthModule.registerEmbedding(userId, name, embedding);
  }

  async simulateLivenessCheck(prompt: LivenessPrompt, onProgress: (p: number) => void, shouldPass: boolean = true): Promise<boolean> {
    return new Promise((resolve) => {
      let progress = 0;
      const interval = setInterval(() => {
        progress += Math.random() * 0.25;
        if (progress >= 1) {
          clearInterval(interval);
          resolve(shouldPass);
        }
        onProgress(progress);
      }, 300);
    });
  }

  async simulateFaceMatch(imagePath: string, shouldMatch: boolean = true): Promise<any> {
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve({
          success: shouldMatch,
          confidence: shouldMatch ? Math.floor(Math.random() * 15 + 85) : Math.floor(Math.random() * 40 + 20),
          userId: shouldMatch ? `USR-${Math.floor(1000 + Math.random() * 9000)}` : undefined,
        });
      }, 1500);
    });
  }
}

export const MLService = new MLServiceClass();