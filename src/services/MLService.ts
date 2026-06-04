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
const SERVER_URL = 'http://10.0.2.2:5000';

export interface AuthResponse {
  faceDetected: boolean;
  liveness: 'PENDING' | 'PASS' | 'FAIL';
  similarity: number;
  authenticated: boolean;
  matchedUserId: string | null;
  instruction: string;
}

export class MLService {
  static async startNewSession() {
    await fetch(`${SERVER_URL}/new_session`, {
      method: 'POST',
    });
  }

  static async authenticate(base64Image: string): Promise<AuthResponse> {
    const response = await fetch(`${SERVER_URL}/authenticate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image: base64Image,
      }),
    });

    return await response.json();
  }

  static async checkHealth() {
    const response = await fetch(`${SERVER_URL}/health`);
    return await response.json();
  }
}