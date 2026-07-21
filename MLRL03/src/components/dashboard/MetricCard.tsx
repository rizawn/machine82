import React from "react";
import { ArrowUpRight, ArrowDownRight } from "lucide-react";

interface MetricCardProps {
  title: string;
  value: string | number;
  description: string;
  change?: number;
  changeSuffix?: string;
  icon?: React.ReactNode;
}

export default function MetricCard({
  title,
  value,
  description,
  change,
  changeSuffix = "%",
  icon,
}: MetricCardProps) {
  const isPositive = change !== undefined && change >= 0;

  return (
    <div className="glass rounded-2xl p-6 transition-all duration-300 hover:-translate-y-1 hover:border-zinc-700/50 flex flex-col justify-between">
      <div className="flex items-center justify-between">
        <span className="text-sm text-zinc-500 font-medium">{title}</span>
        {icon && <div className="text-blue-500">{icon}</div>}
      </div>

      <div className="mt-4 flex items-baseline gap-3">
        <span className="text-3xl font-bold text-white tracking-tight">{value}</span>
        {change !== undefined && (
          <span
            className={`flex items-center text-xs font-semibold px-2 py-0.5 rounded-full ${
              isPositive 
                ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20" 
                : "bg-rose-500/10 text-rose-400 border border-rose-500/20"
            }`}
          >
            {isPositive ? <ArrowUpRight className="w-3.5 h-3.5 mr-0.5" /> : <ArrowDownRight className="w-3.5 h-3.5 mr-0.5" />}
            {isPositive ? "+" : ""}
            {change.toFixed(2)}
            {changeSuffix}
          </span>
        )}
      </div>

      <div className="mt-2 text-xs text-zinc-500 font-medium">{description}</div>
    </div>
  );
}
