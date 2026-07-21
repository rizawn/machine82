"use client";

import React, { useState, useEffect, useRef } from "react";
import { useChatStore } from "@/stores/useChatStore";
import { useExperimentStore } from "@/stores/useExperimentStore";
import { Send, Bot, User, BrainCircuit, Sparkles } from "lucide-react";

export default function ChatWindow() {
  const { 
    sessions, 
    activeSessionId, 
    messages, 
    loading, 
    fetchSessions, 
    selectSession, 
    createSession, 
    sendMessage 
  } = useChatStore();

  const { experiments, fetchExperiments } = useExperimentStore();

  const [input, setInput] = useState("");
  const [selectedExperimentId, setSelectedExperimentId] = useState<string>("");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchSessions();
    fetchExperiments();
  }, [fetchSessions, fetchExperiments]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;
    
    const textToSend = input;
    setInput("");
    await sendMessage(textToSend, selectedExperimentId || null);
  };

  return (
    <div className="flex h-[calc(100vh-4rem)] bg-zinc-950">
      {/* Session List Sidebar */}
      <div className="w-80 border-r border-zinc-800 bg-zinc-950/40 p-4 flex flex-col gap-4">
        <button
          onClick={() => createSession()}
          className="flex items-center justify-center gap-2 w-full bg-blue-600/10 hover:bg-blue-600/20 text-blue-400 font-semibold px-4 py-3 rounded-xl text-sm border border-blue-500/20 transition cursor-pointer"
        >
          <Sparkles className="w-4 h-4" />
          New Chat Session
        </button>

        <div className="flex-1 overflow-y-auto space-y-1">
          {sessions.map((session) => {
            const isActive = session.id === activeSessionId;
            return (
              <button
                key={session.id}
                onClick={() => selectSession(session.id)}
                className={`w-full text-left px-4 py-3.5 rounded-xl text-sm transition-all duration-200 ${
                  isActive
                    ? "bg-zinc-800 text-white font-medium border border-zinc-700"
                    : "text-zinc-400 hover:bg-zinc-900 hover:text-zinc-200 border border-transparent"
                }`}
              >
                <p className="truncate font-semibold">{session.title || "Untitled Chat"}</p>
                <p className="text-[10px] text-zinc-500 mt-1">
                  {new Date(session.created_at).toLocaleDateString()}
                </p>
              </button>
            );
          })}
        </div>
      </div>

      {/* Main Chat Pane */}
      <div className="flex-1 flex flex-col h-full bg-zinc-900/10">
        {/* Top Info Bar */}
        <div className="h-14 border-b border-zinc-800 px-6 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <BrainCircuit className="w-5 h-5 text-blue-500" />
            <span className="text-sm font-semibold text-white">MLRL02 Reasoning Assistant</span>
          </div>

          {/* Context Ingestion Tooltip */}
          <div className="flex items-center gap-3">
            <span className="text-xs text-zinc-500 font-medium">Attach Experiment Context:</span>
            <select
              value={selectedExperimentId}
              onChange={(e) => setSelectedExperimentId(e.target.value)}
              className="bg-zinc-900 border border-zinc-800 text-xs text-zinc-300 rounded-lg px-2.5 py-1.5 focus:outline-none focus:border-blue-500 transition"
            >
              <option value="">No context attached</option>
              {experiments.map((e) => (
                <option key={e.id} value={e.id}>
                  {e.name} ({e.status})
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Message Container */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-zinc-500 gap-3">
              <Bot className="w-12 h-12 text-zinc-600 animate-bounce" />
              <p className="text-sm font-medium">Ask MLRL02 about trading indicators, hyperparameters, or results.</p>
            </div>
          ) : (
            messages.map((msg) => {
              const isAi = msg.role === "ai";
              return (
                <div key={msg.id} className={`flex gap-4 ${isAi ? "" : "justify-end"}`}>
                  {isAi && (
                    <div className="w-9 h-9 rounded-xl bg-blue-600/10 border border-blue-500/20 flex items-center justify-center shrink-0">
                      <Bot className="w-5 h-5 text-blue-500" />
                    </div>
                  )}
                  
                  <div
                    className={`max-w-[70%] rounded-2xl p-4 text-sm leading-relaxed border ${
                      isAi
                        ? "bg-zinc-900 border-zinc-800 text-zinc-200"
                        : "bg-blue-600 border-blue-500 text-white font-medium"
                    }`}
                  >
                    {/* Render content as paragraphs (simple parser) */}
                    <div className="whitespace-pre-wrap">
                      {msg.content}
                    </div>
                    
                    {msg.metadata && msg.metadata.experiment_id && (
                      <div className="mt-2 pt-2 border-t border-zinc-800 text-[10px] text-zinc-500 flex items-center gap-1.5 font-semibold">
                        <BrainCircuit className="w-3 h-3 text-blue-500" />
                        Context: Experiment {msg.metadata.experiment_id.substring(0, 8)}
                      </div>
                    )}
                  </div>

                  {!isAi && (
                    <div className="w-9 h-9 rounded-xl bg-zinc-800 border border-zinc-700 flex items-center justify-center shrink-0">
                      <User className="w-5 h-5 text-zinc-400" />
                    </div>
                  )}
                </div>
              );
            })
          )}
          
          {loading && (
            <div className="flex gap-4">
              <div className="w-9 h-9 rounded-xl bg-blue-600/10 border border-blue-500/20 flex items-center justify-center shrink-0">
                <Bot className="w-5 h-5 text-blue-500 animate-pulse" />
              </div>
              <div className="glass rounded-2xl p-4 text-sm text-zinc-400 flex items-center gap-2">
                <div className="flex gap-1">
                  <span className="w-2.5 h-2.5 rounded-full bg-zinc-600 animate-bounce" style={{ animationDelay: "0ms" }} />
                  <span className="w-2.5 h-2.5 rounded-full bg-zinc-600 animate-bounce" style={{ animationDelay: "150ms" }} />
                  <span className="w-2.5 h-2.5 rounded-full bg-zinc-600 animate-bounce" style={{ animationDelay: "300ms" }} />
                </div>
                Thinking...
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Bar */}
        <div className="p-4 border-t border-zinc-800">
          <form onSubmit={handleSend} className="flex gap-3 max-w-4xl mx-auto">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask MLRL02 a question..."
              className="flex-1 bg-zinc-900 border border-zinc-800 rounded-xl px-5 py-3.5 text-sm text-white focus:outline-none focus:border-blue-500 transition"
            />
            <button
              type="submit"
              disabled={loading}
              className="w-12 h-12 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white rounded-xl flex items-center justify-center transition cursor-pointer shrink-0"
            >
              <Send className="w-5 h-5" />
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
