import { create } from "zustand";

export interface ChatMessage {
  id: number;
  session_id: string;
  role: "user" | "ai";
  content: string;
  created_at: string;
  metadata?: any;
}

export interface ChatSession {
  id: string;
  title: string | null;
  created_at: string;
  messages: ChatMessage[];
}

interface ChatState {
  sessions: ChatSession[];
  activeSessionId: string | null;
  messages: ChatMessage[];
  loading: boolean;
  error: string | null;
  fetchSessions: () => Promise<void>;
  selectSession: (sessionId: string) => Promise<void>;
  createSession: () => Promise<ChatSession>;
  sendMessage: (content: string, experimentId?: string | null) => Promise<void>;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

export const useChatStore = create<ChatState>((set, get) => ({
  sessions: [],
  activeSessionId: null,
  messages: [],
  loading: false,
  error: null,

  fetchSessions: async () => {
    set({ loading: true, error: null });
    try {
      const res = await fetch(`${API_URL}/chat/sessions`);
      if (!res.ok) throw new Error("Failed to fetch sessions");
      const data = await res.json();
      set({ sessions: data });
      if (data.length > 0 && !get().activeSessionId) {
        // Auto-select first session
        get().selectSession(data[0].id);
      }
    } catch (err: any) {
      set({ error: err.message });
    } finally {
      set({ loading: false });
    }
  },

  selectSession: async (sessionId: string) => {
    set({ activeSessionId: sessionId, loading: true, error: null });
    try {
      const res = await fetch(`${API_URL}/chat/sessions/${sessionId}`);
      if (!res.ok) throw new Error("Failed to fetch session details");
      const data = await res.json();
      set({ messages: data.messages });
    } catch (err: any) {
      set({ error: err.message });
    } finally {
      set({ loading: false });
    }
  },

  createSession: async () => {
    set({ loading: true, error: null });
    try {
      const res = await fetch(`${API_URL}/chat/sessions`, {
        method: "POST",
      });
      if (!res.ok) throw new Error("Failed to create chat session");
      const session = await res.json();
      set({
        sessions: [session, ...get().sessions],
        activeSessionId: session.id,
        messages: [],
      });
      return session;
    } catch (err: any) {
      set({ error: err.message });
      throw err;
    } finally {
      set({ loading: false });
    }
  },

  sendMessage: async (content: string, experimentId?: string | null) => {
    let { activeSessionId } = get();
    if (!activeSessionId) {
      const session = await get().createSession();
      activeSessionId = session.id;
    }

    // Append user message immediately for snappy UI
    const tempUserMsg: ChatMessage = {
      id: Date.now(),
      session_id: activeSessionId!,
      role: "user",
      content,
      created_at: new Date().toISOString(),
    };
    
    set((state) => ({ 
      messages: [...state.messages, tempUserMsg] 
    }));

    set({ loading: true, error: null });
    try {
      let url = `${API_URL}/chat/sessions/${activeSessionId}/message`;
      if (experimentId) {
        url += `?experiment_id=${experimentId}`;
      }
      
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content }),
      });
      
      if (!res.ok) throw new Error("Failed to send message");
      const aiResponseMsg = await res.json();
      
      set((state) => ({
        messages: [...state.messages.filter(m => m.id !== tempUserMsg.id), tempUserMsg, aiResponseMsg]
      }));
      
      // Update session title in list if first message
      get().fetchSessions();
      
    } catch (err: any) {
      set({ error: err.message });
    } finally {
      set({ loading: false });
    }
  },
}));
