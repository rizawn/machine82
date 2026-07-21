"use client";

import { usePathname } from "next/navigation";
import { User, Bell } from "lucide-react";

export default function Header() {
  const pathname = usePathname();

  const getPageTitle = () => {
    if (pathname === "/") return "Overview Dashboard";
    if (pathname === "/experiments") return "Experiments Manager";
    if (pathname === "/chat") return "Agentic AI Chat";
    return "The Trinity";
  };

  return (
    <header className="h-16 border-b border-zinc-800 bg-zinc-950/50 backdrop-blur-md flex items-center justify-between px-8 z-10 shrink-0">
      <div>
        <h1 className="text-xl font-semibold text-white tracking-wide">{getPageTitle()}</h1>
      </div>

      <div className="flex items-center gap-4">
        {/* Notifications */}
        <button className="p-2 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900 rounded-xl transition">
          <Bell className="w-5 h-5" />
        </button>

        {/* Profile Card */}
        <div className="flex items-center gap-3 pl-2 border-l border-zinc-800">
          <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-white font-medium text-sm">
            QA
          </div>
          <div className="hidden md:block">
            <p className="text-sm font-semibold text-zinc-200">Quant Architect</p>
            <p className="text-xs text-zinc-500">Administrator</p>
          </div>
        </div>
      </div>
    </header>
  );
}
