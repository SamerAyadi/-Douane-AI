import { CommonModule } from '@angular/common';
import { HttpErrorResponse } from '@angular/common/http';
import { Component, inject } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { finalize } from 'rxjs';

import { ChatService } from '../../core/services/chat.service';
import { ChatResponse } from '../../shared/models/chat.models';


@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './chat.component.html',
  styleUrl: './chat.component.css',
})
export class ChatComponent {
  private readonly chatService = inject(ChatService);

  question = '';
  response: ChatResponse | null = null;
  errorMessage = '';
  loading = false;

  sendQuestion(): void {
    const question = this.question.trim();
    if (!question || this.loading) {
      if (!question) {
        this.errorMessage = 'Veuillez saisir une question. / الرجاء إدخال سؤال.';
      }
      return;
    }

    this.loading = true;
    this.errorMessage = '';
    this.response = null;

    this.chatService
      .askQuestion(question)
      .pipe(finalize(() => (this.loading = false)))
      .subscribe({
        next: (response) => {
          this.response = response;
        },
        error: (error: HttpErrorResponse) => {
          this.errorMessage = this.getErrorMessage(error);
        },
      });
  }

  private getErrorMessage(error: HttpErrorResponse): string {
    if (error.status === 0) {
      return 'Le backend est inaccessible. Vérifiez que FastAPI est démarré.';
    }
    if (error.status === 503) {
      return "Le modèle local Ollama n'est pas disponible. Veuillez le démarrer.";
    }
    return error.error?.detail || 'Une erreur est survenue. Veuillez réessayer.';
  }
}
