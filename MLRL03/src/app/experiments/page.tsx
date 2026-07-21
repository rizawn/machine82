"use client";

import React, { useEffect } from "react";
import HyperparameterForm from "@/components/forms/HyperparameterForm";
import { useExperimentStore } from "@/stores/useExperimentStore";
import { useTrainingStore } from "@/stores/useTrainingStore";
import { Play, RotateCcw, Trash2, Cpu, CheckCircle, AlertCircle, RefreshCw } from "lucide-react";
import { useRouter } from "next/navigation";

export default function ExperimentsPage() {
  const { 
    experiments, 
    loading: expLoading, 
    fetchExperiments, 
    createExperiment, 
    cloneExperiment, 
    deleteExperiment 
  } = useExperimentStore();

  const { startTraining } = useTrainingStore();
  const router = useRouter();

  useEffect(() => {
    fetchExperiments();
  }, [fetchExperiments]);

  const handleCreateExperiment = async (formData: any) => {
    const { name, description, ...config } = formData;
    try {
      // 1. Create the experiment + configuration record
      const newExp = await createExperiment(name, description || "", config);
      
      // 2. Automatically queue the ML and RL training jobs
      await startTraining(newExp.id, "ml");
      await startTraining(newExp.id, "rl");
      
      // 3. Re-fetch list
      fetchExperiments();
      
      // 4. Navigate back to Dashboard to monitor logs
      router.push("/");
    } catch (err) {
      console.error("Failed to start experiment", err);
    }
  };

  const getStatusBadgeClass = (status: string) => {
    switch (status) {
      case "completed":
        return "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20";
      case "running":
        return "bg-blue-500/10 text-blue-400 border border-blue-500/20";
      case "failed":
        return "bg-rose-500/10 text-rose-400 border border-rose-500/20";
      default:
        return "bg-zinc-500/10 text-zinc-400 border border-zinc-500/20";
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle className="w-4 h-4 text-emerald-400" />;
      case "running":
        return <RefreshCw className="w-4 h-4 text-blue-400 animate-spin" />;
      case "failed":
        return <AlertCircle className="w-4 h-4 text-rose-400" />;
      default:
        return <Cpu className="w-4 h-4 text-zinc-400" />;
    }
  };

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      <div>
        <h2 className="text-2xl font-bold text-white tracking-tight">Run Quantitative Experiments</h2>
        <p className="text-zinc-500 text-sm mt-1">Configure hyperparameter models for MLRL01 muscle processing engine.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 items-start">
        {/* Form panel */}
        <div className="lg:col-span-2 space-y-6">
          <HyperparameterForm onSubmit={handleCreateExperiment} loading={expLoading} />
        </div>

        {/* Experiment List Panel */}
        <div className="glass rounded-2xl p-6 space-y-4">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-md font-bold text-white">Recent Experiments</h3>
            <button 
              onClick={() => fetchExperiments()}
              className="text-xs text-blue-500 hover:text-blue-400 font-semibold cursor-pointer"
            >
              Refresh
            </button>
          </div>

          {expLoading && experiments.length === 0 ? (
            <div className="text-center py-8 text-zinc-500 text-sm">Loading runs...</div>
          ) : experiments.length === 0 ? (
            <div className="text-center py-8 text-zinc-500 text-sm">No experiments yet.</div>
          ) : (
            <div className="space-y-3 max-h-[600px] overflow-y-auto pr-1">
              {experiments.map((exp) => (
                <div key={exp.id} className="p-4 rounded-xl border border-zinc-800 bg-zinc-900/40 hover:bg-zinc-900/60 transition-all">
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-semibold text-sm text-zinc-200 truncate max-w-[120px]" title={exp.name}>
                      {exp.name}
                    </span>
                    <span className={`flex items-center gap-1.5 text-xs font-semibold px-2 py-0.5 rounded-full ${getStatusBadgeClass(exp.status)}`}>
                      {getStatusIcon(exp.status)}
                      {exp.status}
                    </span>
                  </div>
                  
                  {exp.description && (
                    <p className="text-xs text-zinc-500 mt-1.5 line-clamp-1">{exp.description}</p>
                  )}

                  <div className="mt-4 flex items-center justify-between gap-2 border-t border-zinc-800/60 pt-3">
                    <span className="text-[10px] text-zinc-600 font-medium">
                      {new Date(exp.created_at).toLocaleDateString()}
                    </span>

                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => cloneExperiment(exp.id)}
                        title="Clone Configuration"
                        className="p-1.5 text-zinc-400 hover:text-blue-500 hover:bg-zinc-800 rounded-lg transition cursor-pointer"
                      >
                        <RotateCcw className="w-3.5 h-3.5" />
                      </button>
                      <button
                        onClick={() => deleteExperiment(exp.id)}
                        title="Delete Run"
                        className="p-1.5 text-zinc-400 hover:text-rose-500 hover:bg-zinc-800 rounded-lg transition cursor-pointer"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
