import { create } from "zustand";

export interface ExperimentConfig {
  target_horizon: number;
  target_method: string;
  train_ratio: number;
  embargo_bars: number;
  rl_algorithm: string;
  rl_timesteps: number;
  learning_rate: number;
  batch_size: number;
  gamma: number;
  gae_lambda: number;
  clip_range: number;
  ent_coef: number;
  lstm_hidden: number;
  initial_capital: number;
  fee_rate: number;
  spread_cost: number;
  slippage: number;
}

export interface Experiment {
  id: string;
  name: string;
  description: string | null;
  status: string;
  created_at: string;
  updated_at: string;
  config?: ExperimentConfig;
}

interface ExperimentState {
  experiments: Experiment[];
  activeExperiment: any | null;
  loading: boolean;
  error: string | null;
  fetchExperiments: () => Promise<void>;
  fetchExperiment: (id: string) => Promise<void>;
  createExperiment: (name: string, description: string, config: Partial<ExperimentConfig>) => Promise<Experiment>;
  cloneExperiment: (id: string) => Promise<void>;
  deleteExperiment: (id: string) => Promise<void>;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

export const useExperimentStore = create<ExperimentState>((set, get) => ({
  experiments: [],
  activeExperiment: null,
  loading: false,
  error: null,

  fetchExperiments: async () => {
    set({ loading: true, error: null });
    try {
      const res = await fetch(`${API_URL}/experiments`);
      if (!res.ok) throw new Error("Failed to fetch experiments");
      const data = await res.json();
      set({ experiments: data });
    } catch (err: any) {
      set({ error: err.message });
    } finally {
      set({ loading: false });
    }
  },

  fetchExperiment: async (id: string) => {
    set({ loading: true, error: null });
    try {
      const res = await fetch(`${API_URL}/metrics/experiment/${id}`);
      if (!res.ok) throw new Error("Failed to fetch experiment metrics");
      const data = await res.json();
      set({ activeExperiment: data });
    } catch (err: any) {
      set({ error: err.message });
    } finally {
      set({ loading: false });
    }
  },

  createExperiment: async (name, description, config) => {
    set({ loading: true, error: null });
    try {
      const res = await fetch(`${API_URL}/experiments`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, description, config }),
      });
      if (!res.ok) throw new Error("Failed to create experiment");
      const data = await res.json();
      set({ experiments: [data, ...get().experiments] });
      return data;
    } catch (err: any) {
      set({ error: err.message });
      throw err;
    } finally {
      set({ loading: false });
    }
  },

  cloneExperiment: async (id: string) => {
    set({ loading: true, error: null });
    try {
      const res = await fetch(`${API_URL}/experiments/${id}/clone`, {
        method: "POST",
      });
      if (!res.ok) throw new Error("Failed to clone experiment");
      const data = await res.json();
      set({ experiments: [data, ...get().experiments] });
    } catch (err: any) {
      set({ error: err.message });
    } finally {
      set({ loading: false });
    }
  },

  deleteExperiment: async (id: string) => {
    set({ loading: true, error: null });
    try {
      const res = await fetch(`${API_URL}/experiments/${id}`, {
        method: "DELETE",
      });
      if (!res.ok) throw new Error("Failed to delete experiment");
      set({
        experiments: get().experiments.filter((e) => e.id !== id),
        activeExperiment: get().activeExperiment?.experiment_id === id ? null : get().activeExperiment,
      });
    } catch (err: any) {
      set({ error: err.message });
    } finally {
      set({ loading: false });
    }
  },
}));
