export interface ChatRequest {
  question: string;
  session_id?: string;
}

export interface SourceItem {
  source: string | null;
  page: number | null;
  chunk: number | null;
  distance?: number | null;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  language: 'ar' | 'fr';
  sources?: SourceItem[];
  timing_seconds?: {
    total: number;
  };
  isError?: boolean;
}

export interface ChatResponse {
  session_id: string | null;
  question: string;
  answer: string;
  language: 'ar' | 'fr';
  sources: SourceItem[];
  retrieved_sources: SourceItem[];
  timing_seconds: {
    total: number;
  };
}

export interface ChatSession {
  id: string;
  title: string | null;
  created_at: string;
  updated_at: string;
}

export interface StoredChatMessage {
  id: string;
  session_id: string;
  role: 'user' | 'assistant';
  content: string;
  language: string | null;
  sources: SourceItem[] | null;
  created_at: string;
}
