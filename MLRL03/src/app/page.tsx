"use client";

import React, { useEffect, useState } from "react";
import { useExperimentStore } from "@/stores/useExperimentStore";
import { useTrainingStore } from "@/stores/useTrainingStore";
import MetricCard from "@/components/dashboard/MetricCard";
import EquityCurve from "@/components/dashboard/EquityCurve";
import { 
  TrendingUp, 
  Percent, 
  Activity, 
  Terminal, 
  Server, 
  Database, 
  Cpu, 
  ShieldCheck 
} from "lucide-react";

export default function DashboardPage() {
  const { experiments, activeExperiment, fetchExperiments, fetchExperiment } = useExperimentStore();
  const { logs, progress, status: jobStatus, subscribeToLogs, unsubscribeFromLogs } = useTrainingStore();
  
  const [selectedExpId, setSelectedExpId] = useState<string>("");

  useEffect(() => {
    fetchExperiments();
  }, [fetchExperiments]);

  // Set default selected experiment
  useEffect(() => {
    if (experiments.length > 0 && !selectedExpId) {
      setSelectedExpId(experiments[0].id);
    }
  }, [experiments, selectedExpId]);

  // Fetch metrics when selected experiment changes
  useEffect(() => {
    if (selectedExpId) {
      fetchExperiment(selectedExpId);
    }
  }, [selectedExpId, fetchExperiment]);

  // Subscribe to live logs if any active job is running
  useEffect(() => {
    if (activeExperiment && activeExperiment.jobs) {
      const runningJob = activeExperiment.jobs.find((j: any) => j.status === "running" || j.status === "queued");
      if (runningJob) {
        subscribeToLogs(runningJob.id);
      } else {
        unsubscribeFromLogs();
      }
    }
    return () => unsubscribeFromLogs();
  }, [activeExperiment, subscribeToLogs, unsubscribeFromLogs]);

  const stats = activeExperiment?.rl_results?.[0] || {};
  const mlResults = activeExperiment?.ml_results || [];

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* Top Selector bar */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-white tracking-tight">System Performance Monitor</h2>
          <p className="text-zinc-500 text-sm mt-1">Real-time quantitative trading analysis & validation</p>
        </div>

        <div className="flex items-center gap-3">
          <span className="text-sm text-zinc-400 font-medium">Select Experiment:</span>
          <select
            value={selectedExpId}
            onChange={(e) => setSelectedExpId(e.target.value)}
            className="bg-zinc-900 border border-zinc-800 text-sm text-zinc-300 rounded-xl px-4 py-2.5 focus:outline-none focus:border-blue-500 transition"
          >
            {experiments.map((exp) => (
              <option key={exp.id} value={exp.id}>
                {exp.name} ({exp.status})
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <MetricCard
          title="Sharpe Ratio"
          value={stats.sharpe_ratio ? stats.sharpe_ratio.toFixed(3) : "0.000"}
          description="Annualized Differential Sharpe Ratio"
          change={stats.sharpe_ratio}
          changeSuffix=""
          icon={<TrendingUp className="w-5 h-5" />}
        />
        <MetricCard
          title="Total Return"
          value={stats.total_return ? `${(stats.total_return * 100).toFixed(2)}%` : "0.00%"}
          description="Strategy return over backtest period"
          change={stats.total_return ? stats.total_return * 100 : undefined}
          icon={<Percent className="w-5 h-5" />}
        />
        <MetricCard
          title="Max Drawdown"
          value={stats.max_drawdown ? `${(stats.max_drawdown * 100).toFixed(2)}%` : "0.00%"}
          description="Peak-to-trough maximum drawdown"
          change={stats.max_drawdown ? -stats.max_drawdown * 100 : undefined}
          icon={<Activity className="w-5 h-5" />}
        />
      </div>

      {/* Charts & Logs Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Equity Curve Chart */}
        <div className="lg:col-span-2">
          <EquityCurve 
            equityHistory={stats.equity_curve || []} 
            title={`${activeExperiment?.config?.rl_algorithm || "Strategy"} Equity Curve`}
          />
        </div>

        {/* Live Terminal Logs */}
        <div className="glass rounded-2xl p-6 flex flex-col h-[450px]">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-md font-bold text-white flex items-center gap-2">
              <Terminal className="w-4 h-4 text-blue-500" />
              Live Training Log Stream
            </h3>
            <span className={`text-xs px-2 py-0.5 rounded-full font-semibold ${
              jobStatus === "running" ? "bg-blue-500/10 text-blue-400 border border-blue-500/20" : "bg-zinc-800 text-zinc-400"
            }`}>
              {jobStatus}
            </span>
          </div>

          {/* Progress Bar */}
          {jobStatus === "running" && (
            <div className="mb-4">
              <div className="flex items-center justify-between text-xs text-zinc-500 mb-1">
                <span>Training Progress</span>
                <span>{progress.toFixed(1)}%</span>
              </div>
              <div className="w-full bg-zinc-800 h-2 rounded-full overflow-hidden">
                <div 
                  className="bg-blue-500 h-full rounded-full transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          )}

          {/* Console logger */}
          <div className="flex-1 bg-black/40 border border-zinc-800/60 rounded-xl p-4 font-mono text-xs overflow-y-auto space-y-2 text-zinc-300">
            {logs.length === 0 ? (
              <p className="text-zinc-600 italic">Logs will stream here in real-time when training is running...</p>
            ) : (
              logs.map((log, idx) => (
                <p key={idx} className="leading-relaxed border-l-2 border-blue-600/40 pl-2">
                  {log}
                </p>
              ))
            )}
          </div>
        </div>
      </div>

      {/* Model Comparisons Table */}
      <div className="glass rounded-2xl p-6">
        <h3 className="text-md font-bold text-white mb-4">Supervised Classifiers Performance</h3>
        
        {mlResults.length === 0 ? (
          <p className="text-sm text-zinc-500">No ML results computed for this experiment run.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm border-collapse">
              <thead>
                <tr className="border-b border-zinc-800 text-zinc-500 font-medium">
                  <th className="py-3 px-4">Classifier</th>
                  <th className="py-3 px-4">Accuracy</th>
                  <th className="py-3 px-4">Precision</th>
                  <th className="py-3 px-4">Recall</th>
                  <th className="py-3 px-4">Backtest Sharpe</th>
                  <th className="py-3 px-4">Max Drawdown</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-800/40 text-zinc-300 font-medium">
                {mlResults.map((ml: any, idx: number) => (
                  <tr key={idx} className="hover:bg-zinc-900/30 transition">
                    <td className="py-3.5 px-4 font-semibold text-white">{ml.model_name}</td>
                    <td className="py-3.5 px-4">{(ml.accuracy * 100).toFixed(2)}%</td>
                    <td className="py-3.5 px-4">{(ml.precision * 100).toFixed(2)}%</td>
                    <td className="py-3.5 px-4">{(ml.recall * 100).toFixed(2)}%</td>
                    <td className="py-3.5 px-4 text-blue-400">{ml.bt_sharpe ? ml.bt_sharpe.toFixed(3) : "0.000"}</td>
                    <td className="py-3.5 px-4 text-rose-400">{ml.bt_max_dd ? `${(ml.bt_max_dd * 100).toFixed(2)}%` : "0.00%"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
