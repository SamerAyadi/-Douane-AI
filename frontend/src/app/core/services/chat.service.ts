import { HttpClient } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { Observable } from 'rxjs';

import {
  ChatRequest,
  ChatResponse,
} from '../../shared/models/chat.models';


@Injectable({ providedIn: 'root' })
export class ChatService {
  private readonly http = inject(HttpClient);
  private readonly apiUrl = 'http://127.0.0.1:8000';

  askQuestion(question: string): Observable<ChatResponse> {
    const request: ChatRequest = { question };
    return this.http.post<ChatResponse>(`${this.apiUrl}/chat`, request);
  }
}
