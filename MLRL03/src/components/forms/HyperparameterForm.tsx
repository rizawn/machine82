"use client";

import React, { useState } from "react";
import { z } from "zod";
import { AlertCircle, Play, Sparkles } from "lucide-react";

// Strict validation matching python schemas
const configSchema = z.object({
  name: z.string().min(1, "Experiment name is required").max(255),
  description: z.string().optional(),
  target_horizon: z.number().int().min(1).max(20),
  target_method: z.enum(["triple_barrier", "threshold", "binary"]),
  train_ratio: z.number().min(0.5).max(0.95),
  embargo_bars: z.number().int().min(10).max(200),
  rl_algorithm: z.enum(["PPO", "RecurrentPPO"]),
  rl_timesteps: z.number().int().min(10000).max(10000000),
  learning_rate: z.number().min(1e-6).max(1e-2),
  batch_size: z.coerce.number().refine(val => [16, 32, 64, 128, 256].includes(val), {
    message: "Batch size must be 16, 32, 64, 128, or 256",
  }),
  gamma: z.number().min(0.9).max(0.999),
  gae_lambda: z.number().min(0.8).max(1.0),
  clip_range: z.number().min(0.1).max(0.4),
  ent_coef: z.number().min(0.0).max(0.1),
  lstm_hidden: z.coerce.number().refine(val => [64, 128, 256].includes(val), {
    message: "LSTM hidden size must be 64, 128, or 256",
  }),
  initial_capital: z.number().min(1000).max(10000000),
  fee_rate: z.number().min(0).max(0.01),
  spread_cost: z.number().min(0).max(0.01),
  slippage: z.number().min(0).max(0.01),
});

type FormInputs = z.infer<typeof configSchema>;

interface HyperparameterFormProps {
  onSubmit: (data: FormInputs) => void;
  loading?: boolean;
}

export default function HyperparameterForm({ onSubmit, loading = false }: HyperparameterFormProps) {
  const [inputs, setInputs] = useState<FormInputs>({
    name: "Gold Production Run V5",
    description: "Production trading model with optimized hyperparams",
    target_horizon: 5,
    target_method: "triple_barrier",
    train_ratio: 0.80,
    embargo_bars: 60,
    rl_algorithm: "RecurrentPPO",
    rl_timesteps: 10000,
    learning_rate: 0.0003,
    batch_size: 16,
    gamma: 0.99,
    gae_lambda: 0.95,
    clip_range: 0.2,
    ent_coef: 0.01,
    lstm_hidden: 64,
    initial_capital: 100000,
    fee_rate: 0.0001,
    spread_cost: 0.0003,
    slippage: 0.0002,
  });

  const [errors, setErrors] = useState<Partial<Record<keyof FormInputs, string>>>({});

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
    const { name, value, type } = e.target;
    
    // Auto-parse numbers
    let parsedValue: any = value;
    if (type === "number") {
      parsedValue = parseFloat(value);
      if (isNaN(parsedValue)) parsedValue = 0;
    }

    setInputs(prev => ({
      ...prev,
      [name]: parsedValue
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const result = configSchema.safeParse(inputs);
    
    if (!result.success) {
      const fieldErrors: Partial<Record<keyof FormInputs, string>> = {};
      result.error.issues.forEach(err => {
        if (err.path[0]) {
          fieldErrors[err.path[0] as keyof FormInputs] = err.message;
        }
      });
      setErrors(fieldErrors);
    } else {
      setErrors({});
      onSubmit(result.data);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Basic Info */}
      <div className="glass rounded-2xl p-6 space-y-4">
        <h3 className="text-md font-bold text-white mb-2 flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-blue-500" />
          General Experiment Information
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-xs font-semibold text-zinc-400 mb-1.5">Experiment Name</label>
            <input
              type="text"
              name="name"
              value={inputs.name}
              onChange={handleInputChange}
              className="w-full bg-zinc-900 border border-zinc-800 rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:border-blue-500 transition"
              placeholder="Enter name"
            />
            {errors.name && <p className="text-xs text-red-400 mt-1 flex items-center gap-1"><AlertCircle className="w-3.5 h-3.5" /> {errors.name}</p>}
          </div>

          <div>
            <label className="block text-xs font-semibold text-zinc-400 mb-1.5">Description (Optional)</label>
            <input
              type="text"
              name="description"
              value={inputs.description || ""}
              onChange={handleInputChange}
              className="w-full bg-zinc-900 border border-zinc-800 rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:border-blue-500 transition"
              placeholder="Enter description"
            />
          </div>
        </div>
      </div>

      {/* Target Engineering & Split */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="glass rounded-2xl p-6 space-y-4">
          <h3 className="text-md font-bold text-white mb-2">Target & Data Split</h3>
          
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-semibold text-zinc-400 mb-1.5">Horizon (Days)</label>
              <input
                type="number"
                name="target_horizon"
                value={inputs.target_horizon}
                onChange={handleInputChange}
                className="w-full bg-zinc-900 border border-zinc-800 rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:border-blue-500 transition"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-zinc-400 mb-1.5">Target Method</label>
              <select
                name="target_method"
                value={inputs.target_method}
                onChange={handleInputChange}
                className="w-full bg-zinc-900 border border-zinc-800 rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:border-blue-500 transition"
              >
                <option value="triple_barrier">Triple Barrier</option>
                <option value="threshold">Threshold</option>
                <option value="binary">Binary</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-semibold text-zinc-400 mb-1.5">Train Ratio</label>
              <input
                type="number"
                step="0.05"
                name="train_ratio"
                value={inputs.train_ratio}
                onChange={handleInputChange}
                className="w-full bg-zinc-900 border border-zinc-800 rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:border-blue-500 transition"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-zinc-400 mb-1.5">Embargo Bars</label>
              <input
                type="number"
                name="embargo_bars"
                value={inputs.embargo_bars}
                onChange={handleInputChange}
                className="w-full bg-zinc-900 border border-zinc-800 rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:border-blue-500 transition"
              />
            </div>
          </div>
        </div>

        {/* Backtest & Costs */}
        <div className="glass rounded-2xl p-6 space-y-4">
          <h3 className="text-md font-bold text-white mb-2">Portfolio & Backtest Costs</h3>
          
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-semibold text-zinc-400 mb-1.5">Initial Capital ($)</label>
              <input
                type="number"
                name="initial_capital"
                value={inputs.initial_capital}
                onChange={handleInputChange}
                className="w-full bg-zinc-900 border border-zinc-800 rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:border-blue-500 transition"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-zinc-400 mb-1.5">Fee Rate (Ratio)</label>
              <input
                type="number"
                step="0.0001"
                name="fee_rate"
                value={inputs.fee_rate}
                onChange={handleInputChange}
                className="w-full bg-zinc-900 border border-zinc-800 rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:border-blue-500 transition"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-zinc-400 mb-1.5">Spread Cost (Ratio)</label>
              <input
                type="number"
                step="0.0001"
                name="spread_cost"
                value={inputs.spread_cost}
                onChange={handleInputChange}
                className="w-full bg-zinc-900 border border-zinc-800 rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:border-blue-500 transition"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-zinc-400 mb-1.5">Slippage (Ratio)</label>
              <input
                type="number"
                step="0.0001"
                name="slippage"
                value={inputs.slippage}
                onChange={handleInputChange}
                className="w-full bg-zinc-900 border border-zinc-800 rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:border-blue-500 transition"
              />
            </div>
          </div>
        </div>
      </div>

      {/* RL Model Hyperparameters */}
      <div className="glass rounded-2xl p-6 space-y-4">
        <h3 className="text-md font-bold text-white mb-2">Reinforcement Learning Hyperparameters</h3>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-xs font-semibold text-zinc-400 mb-1.5">Algorithm</label>
            <select
              name="rl_algorithm"
              value={inputs.rl_algorithm}
              onChange={handleInputChange}
              className="w-full bg-zinc-900 border border-zinc-800 rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:border-blue-500 transition"
            >
              <option value="RecurrentPPO">RecurrentPPO (LSTM)</option>
              <option value="PPO">Standard PPO</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-semibold text-zinc-400 mb-1.5">Total Timesteps</label>
            <input
              type="number"
              name="rl_timesteps"
              value={inputs.rl_timesteps}
              onChange={handleInputChange}
              className="w-full bg-zinc-900 border border-zinc-800 rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:border-blue-500 transition"
            />
          </div>

          <div>
            <label className="block text-xs font-semibold text-zinc-400 mb-1.5">Learning Rate</label>
            <input
              type="number"
              step="0.00001"
              name="learning_rate"
              value={inputs.learning_rate}
              onChange={handleInputChange}
              className="w-full bg-zinc-900 border border-zinc-800 rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:border-blue-500 transition"
            />
          </div>

          <div>
            <label className="block text-xs font-semibold text-zinc-400 mb-1.5">Batch Size</label>
            <select name="batch_size" value={inputs.batch_size} onChange={handleInputChange} className="w-full bg-zinc-900 border border-zinc-800 rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:border-blue-500 transition">
              <option value={16}>16</option>
              <option value={32}>32</option>
              <option value={64}>64</option>
              <option value={128}>128</option>
              <option value={256}>256</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-semibold text-zinc-400 mb-1.5">Gamma (Discount)</label>
            <input
              type="number"
              step="0.005"
              name="gamma"
              value={inputs.gamma}
              onChange={handleInputChange}
              className="w-full bg-zinc-900 border border-zinc-800 rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:border-blue-500 transition"
            />
          </div>

          <div>
            <label className="block text-xs font-semibold text-zinc-400 mb-1.5">GAE Lambda</label>
            <input
              type="number"
              step="0.01"
              name="gae_lambda"
              value={inputs.gae_lambda}
              onChange={handleInputChange}
              className="w-full bg-zinc-900 border border-zinc-800 rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:border-blue-500 transition"
            />
          </div>

          <div>
            <label className="block text-xs font-semibold text-zinc-400 mb-1.5">Entropy Coef</label>
            <input
              type="number"
              step="0.005"
              name="ent_coef"
              value={inputs.ent_coef}
              onChange={handleInputChange}
              className="w-full bg-zinc-900 border border-zinc-800 rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:border-blue-500 transition"
            />
          </div>

          <div>
            <label className="block text-xs font-semibold text-zinc-400 mb-1.5">LSTM Hidden Size</label>
            <select
              name="lstm_hidden"
              value={inputs.lstm_hidden}
              onChange={handleInputChange}
              className="w-full bg-zinc-900 border border-zinc-800 rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:border-blue-500 transition"
            >
              <option value={64}>64</option>
              <option value={128}>128</option>
              <option value={256}>256</option>
            </select>
          </div>
        </div>
      </div>

      {/* Submit */}
      <div className="flex justify-end">
        <button
          type="submit"
          disabled={loading}
          className="flex items-center gap-2 bg-blue-600 hover:bg-blue-500 text-white font-semibold px-8 py-3.5 rounded-xl text-sm transition-all duration-200 cursor-pointer disabled:opacity-50"
        >
          <Play className="w-4 h-4" />
          {loading ? "Initializing..." : "Create & Start Experiment"}
        </button>
      </div>
    </form>
  );
}
