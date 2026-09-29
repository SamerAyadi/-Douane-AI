import { HttpClient } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { Observable } from 'rxjs';

import {
  ChatRequest,
  ChatResponse,
  ChatSession,
  StoredChatMessage,
} from '../../shared/models/chat.models';


@Injectable({ providedIn: 'root' })
export class ChatService {
  private readonly http = inject(HttpClient);
  private readonly apiUrl = 'http://127.0.0.1:8000';

  askQuestion(
    question: string,
    sessionId: string | null = null,
  ): Observable<ChatResponse> {
    const request: ChatRequest = { question };
    if (sessionId) {
      request.session_id = sessionId;
    }

    return this.http.post<ChatResponse>(`${this.apiUrl}/chat`, request);
  }

  getSessions(): Observable<ChatSession[]> {
    return this.http.get<ChatSession[]>(`${this.apiUrl}/sessions`);
  }

  getSessionMessages(sessionId: string): Observable<StoredChatMessage[]> {
    return this.http.get<StoredChatMessage[]>(
      `${this.apiUrl}/sessions/${sessionId}/messages`,
    );
  }
}
