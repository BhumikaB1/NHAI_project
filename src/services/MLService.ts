export type LivenessPromptType = 'BLINK' | 'TURN_LEFT' | 'TURN_RIGHT';

export interface LivenessPrompt {
  type: LivenessPromptType;
  instruction: string;
}

export const LIVENESS_PROMPTS: LivenessPrompt[] = [
  { type: 'BLINK', instruction: 'Blink your eyes' },
  { type: 'TURN_LEFT', instruction: 'Turn your head left' },
  { type: 'TURN_RIGHT', instruction: 'Turn your head right' },
];

export interface FaceMatchResult {
  success: boolean;
  confidence: number;
  userId?: string;
  error?: string;
}

export class MLService {
  /**
   * Simulates a liveness detection check.
   * Calls the onProgress callback as it processes.
   * forcePass lets us control the outcome.
   */
  static simulateLivenessCheck(
    prompt: LivenessPrompt,
    onProgress: (progress: number) => void,
    forcePass: boolean = true
  ): Promise<boolean> {
    return new Promise((resolve) => {
      let progress = 0;
      const intervalTime = 120; // ms (slightly faster for a snappy demo)
      const totalSteps = 10;
      
      const interval = setInterval(() => {
        progress += 1 / totalSteps;
        onProgress(Math.min(progress, 1));
        
        if (progress >= 1) {
          clearInterval(interval);
          resolve(forcePass);
        }
      }, intervalTime);
    });
  }

  /**
   * Simulates matching a captured face against registered profiles.
   * forceSuccess lets us control the outcome.
   */
  static simulateFaceMatch(imagePath: string, forceSuccess: boolean = true): Promise<FaceMatchResult> {
    return new Promise((resolve) => {
      // Log the captured file path to console to simulate real intake
      console.log(`[MLService] Analyzing frame: ${imagePath}`);
      
      setTimeout(() => {
        if (forceSuccess) {
          const confidence = parseFloat((93.5 + Math.random() * 5.4).toFixed(2)); // 93.5% - 98.9%
          resolve({
            success: true,
            confidence,
            userId: `USR-${Math.floor(1000 + Math.random() * 9000)}`,
          });
        } else {
          const confidence = parseFloat((32.1 + Math.random() * 18.3).toFixed(2)); // 32.1% - 50.4%
          resolve({
            success: false,
            confidence,
          });
        }
      }, 1200);
    });
  }
}
