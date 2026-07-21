"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { 
  LayoutDashboard, 
  FlaskConical, 
  MessageSquare, 
  Activity, 
  ShieldAlert, 
  Cpu 
} from "lucide-react";

export default function Sidebar() {
  const pathname = usePathname();

  const links = [
    { name: "Dashboard", href: "/", icon: LayoutDashboard },
    { name: "Experiments", href: "/experiments", icon: FlaskConical },
    { name: "Agentic Chat", href: "/chat", icon: MessageSquare },
  ];

  return (
    <aside className="w-64 bg-zinc-950 border-r border-zinc-800 flex flex-col h-screen shrink-0">
      {/* Brand Logo */}
      <div className="h-16 flex items-center px-6 border-b border-zinc-800 gap-3">
        <Cpu className="w-6 h-6 text-blue-500 animate-pulse" />
        <span className="font-bold text-lg tracking-wider text-white">THE TRINITY</span>
      </div>

      {/* Nav List */}
      <nav className="flex-1 px-4 py-6 space-y-1 overflow-y-auto">
        {links.map((link) => {
          const Icon = link.icon;
          const isActive = pathname === link.href;

          return (
            <Link
              key={link.href}
              href={link.href}
              className={`flex items-center px-4 py-3 rounded-xl text-sm font-medium transition-all duration-200 gap-3 ${
                isActive
                  ? "bg-blue-600/10 border border-blue-500/20 text-blue-400"
                  : "text-zinc-400 hover:bg-zinc-900 hover:text-zinc-200 border border-transparent"
              }`}
            >
              <Icon className="w-5 h-5" />
              {link.name}
            </Link>
          );
        })}
      </nav>

      {/* System Status Banner */}
      <div className="p-4 border-t border-zinc-800">
        <div className="glass rounded-xl p-4 flex flex-col gap-2">
          <div className="flex items-center justify-between text-xs">
            <span className="text-zinc-500">MLRL01 Worker</span>
            <span className="flex items-center gap-1.5 font-medium text-emerald-400">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
              Active
            </span>
          </div>
          <div className="flex items-center justify-between text-xs">
            <span className="text-zinc-500">MLRL02 LLM</span>
            <span className="flex items-center gap-1.5 font-medium text-emerald-400">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
              Active
            </span>
          </div>
        </div>
      </div>
    </aside>
  );
}
