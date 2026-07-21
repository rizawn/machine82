import { create } from "zustand";

interface TrainingJob {
  id: string;
  experiment_id: string;
  celery_task_id: string | null;
  job_type: string;
  status: string;
  started_at: string | null;
  completed_at: string | null;
  error_message: string | null;
  progress_pct: number;
}

interface TrainingState {
  logs: string[];
  progress: number;
  status: string;
  loading: boolean;
  error: string | null;
  ws: WebSocket | null;
  startTraining: (experimentId: string, jobType: "ml" | "rl") => Promise<TrainingJob>;
  killTraining: (jobId: string) => Promise<void>;
  subscribeToLogs: (jobId: string) => void;
  unsubscribeFromLogs: () => void;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";
const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws";

export const useTrainingStore = create<TrainingState>((set, get) => ({
  logs: [],
  progress: 0,
  status: "queued",
  loading: false,
  error: null,
  ws: null,

  startTraining: async (experimentId: string, jobType: "ml" | "rl") => {
    set({ loading: true, error: null, logs: [], progress: 0, status: "queued" });
    try {
      const res = await fetch(`${API_URL}/training/${experimentId}/start?job_type=${jobType}`, {
        method: "POST",
      });
      if (!res.ok) throw new Error("Failed to start training");
      const job = await res.json();
      set({ status: job.status });
      return job;
    } catch (err: any) {
      set({ error: err.message });
      throw err;
    } finally {
      set({ loading: false });
    }
  },

  killTraining: async (jobId: string) => {
    try {
      const res = await fetch(`${API_URL}/training/jobs/${jobId}/kill`, {
        method: "POST",
      });
      if (!res.ok) throw new Error("Failed to cancel job");
      const job = await res.json();
      set({ status: job.status, progress: job.progress_pct });
    } catch (err: any) {
      set({ error: err.message });
    }
  },

  subscribeToLogs: (jobId: string) => {
    // Clean up existing WebSocket if any
    get().unsubscribeFromLogs();

    const ws = new WebSocket(`${WS_URL}/training/${jobId}`);
    
    ws.onopen = () => {
      console.log(`[WS] Subscribed to logs for job: ${jobId}`);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        // Check if status update or log message
        if (data.status) {
          set({ 
            status: data.status, 
            progress: data.progress_pct ?? get().progress,
            error: data.error_message || get().error
          });
        }
        
        if (data.message) {
          set((state) => ({ logs: [...state.logs, data.message] }));
        }
      } catch (e) {
        // Fallback for raw text
        set((state) => ({ logs: [...state.logs, event.data] }));
      }
    };

    ws.onerror = (err) => {
      console.error("[WS] WebSocket error", err);
      set({ error: "WebSocket connection error" });
    };

    ws.onclose = () => {
      console.log(`[WS] Disconnected from job logs: ${jobId}`);
    };

    set({ ws });
  },

  unsubscribeFromLogs: () => {
    const { ws } = get();
    if (ws) {
      ws.close();
      set({ ws: null });
    }
  },
}));
