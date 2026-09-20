export interface ChatRequest {
  question: string;
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
  question: string;
  answer: string;
  language: 'ar' | 'fr';
  sources: SourceItem[];
  retrieved_sources: SourceItem[];
  timing_seconds: {
    total: number;
  };
}
