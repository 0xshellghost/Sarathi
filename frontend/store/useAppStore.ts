// store/useAppStore.ts
import { create } from 'zustand'

interface AppState {
  messages: Array<{ role: string, content: string }>;
  isStreaming: boolean;
  formSchema: any | null;       // Holds the JSON schema from the AI
  extractedData: any | null;    // Holds the user's form inputs
  setFormSchema: (schema: any) => void;
  addMessage: (msg: { role: string, content: string }) => void;
}

export const useAppStore = create<AppState>()((set) => ({
  messages: [],
  isStreaming: false,
  formSchema: null,
  extractedData: null,
  setFormSchema: (schema) => set({ formSchema: schema }),
  addMessage: (msg) => set((state) => ({ messages: [...state.messages, msg] })),
}))