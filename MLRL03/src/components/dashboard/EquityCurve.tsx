"use client";

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

interface EquityCurveProps {
  equityHistory: number[];
  dates?: string[];
  title?: string;
}

export default function EquityCurve({
  equityHistory,
  dates,
  title = "Strategy Equity Curve",
}: EquityCurveProps) {
  if (!equityHistory || equityHistory.length === 0) {
    return (
      <div className="glass rounded-2xl p-6 h-[400px] flex items-center justify-center text-zinc-500 font-medium">
        No equity curve data available. Run training first.
      </div>
    );
  }

  const data = equityHistory.map((val, idx) => ({
    name: dates && dates[idx] ? new Date(dates[idx]).toLocaleDateString() : `Step ${idx}`,
    Equity: Math.round(val),
  }));

  // Min and max bounds to focus chart
  const minVal = Math.min(...equityHistory) * 0.98;
  const maxVal = Math.max(...equityHistory) * 1.02;

  return (
    <div className="glass rounded-2xl p-6 flex flex-col justify-between h-[450px]">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-md font-bold text-white tracking-wide">{title}</h3>
        <span className="text-xs text-zinc-400 font-medium">
          Start: ${Math.round(equityHistory[0]).toLocaleString()} | End: ${Math.round(equityHistory[equityHistory.length - 1]).toLocaleString()}
        </span>
      </div>

      <div className="flex-1 w-full min-h-[320px]">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
            <defs>
              <linearGradient id="colorEquity" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
            <XAxis
              dataKey="name"
              stroke="#71717a"
              fontSize={10}
              tickLine={false}
              axisLine={false}
              dy={10}
            />
            <YAxis
              domain={[minVal, maxVal]}
              stroke="#71717a"
              fontSize={10}
              tickLine={false}
              axisLine={false}
              dx={-10}
              tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#09090b",
                border: "1px solid #27272a",
                borderRadius: "12px",
              }}
              labelStyle={{ color: "#a1a1aa", fontSize: "11px", fontWeight: 600 }}
              itemStyle={{ color: "#ffffff", fontSize: "13px", fontWeight: 600 }}
            />
            <Area
              type="monotone"
              dataKey="Equity"
              stroke="#3b82f6"
              strokeWidth={2}
              fillOpacity={1}
              fill="url(#colorEquity)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
