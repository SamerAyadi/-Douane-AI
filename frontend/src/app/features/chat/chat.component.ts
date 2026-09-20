import { CommonModule } from '@angular/common';
import { HttpErrorResponse } from '@angular/common/http';
import { Component, ElementRef, ViewChild, inject } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { finalize } from 'rxjs';

import { ChatService } from '../../core/services/chat.service';
import { ChatMessage } from '../../shared/models/chat.models';


@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './chat.component.html',
  styleUrl: './chat.component.css',
})
export class ChatComponent {
  private readonly chatService = inject(ChatService);

  @ViewChild('conversation') conversation?: ElementRef<HTMLElement>;

  question = '';
  messages: ChatMessage[] = [];
  inputError = '';
  loading = false;

  displaySourceName(source: string | null): string {
    if (!source) {
      return '';
    }

    const fileName = source.replace(/\\/g, '/').split('/').pop() || source;
    const maxLength = 70;
    return fileName.length > maxLength
      ? `${fileName.slice(0, maxLength - 3)}...`
      : fileName;
  }

  cleanAnswerText(answer: string): string {
    const text = answer.replace(/\r\n/g, '\n').trim();
    const lines = text.split('\n');
    const sourceLabel = /^\s*(?:sources?|documents?|références?|المصدر|المصادر)\s*[:：]/i;
    const pageReference = /(?:\bpages?\s*[:：]?\s*\d+|(?:الصفحة|صفحة)\s*[:：]?\s*\d+)/i;
    const documentReference = /(?:\.pdf\b|\b\d{1,4}[_/-]\d{4}\b)/i;
    const firstFooterLine = Math.max(0, lines.length - 8);

    for (let index = lines.length - 1; index >= firstFooterLine; index -= 1) {
      if (!sourceLabel.test(lines[index])) {
        continue;
      }

      const footer = lines.slice(index).join('\n');
      if (pageReference.test(footer) || documentReference.test(footer)) {
        return lines.slice(0, index).join('\n').trimEnd();
      }
    }

    return text;
  }

  sendQuestion(): void {
    const question = this.question.trim();
    if (!question || this.loading) {
      if (!question) {
        this.inputError = 'Veuillez saisir une question. / الرجاء إدخال سؤال.';
      }
      return;
    }

    const language = this.detectMessageLanguage(question);
    this.messages.push({ role: 'user', content: question, language });
    this.question = '';
    this.loading = true;
    this.inputError = '';
    this.scrollToLatest();

    this.chatService
      .askQuestion(question)
      .pipe(
        finalize(() => {
          this.loading = false;
          this.scrollToLatest();
        }),
      )
      .subscribe({
        next: (response) => {
          this.messages.push({
            role: 'assistant',
            content: response.answer,
            language: response.language,
            sources: response.sources,
            timing_seconds: response.timing_seconds,
          });
          this.scrollToLatest();
        },
        error: (error: HttpErrorResponse) => {
          this.messages.push({
            role: 'assistant',
            content: this.getErrorMessage(error, language),
            language,
            sources: [],
            isError: true,
          });
          this.scrollToLatest();
        },
      });
  }

  clearChat(): void {
    if (!this.loading) {
      this.messages = [];
      this.inputError = '';
    }
  }

  private detectMessageLanguage(text: string): 'ar' | 'fr' {
    return /[\u0600-\u06ff]/.test(text) ? 'ar' : 'fr';
  }

  private scrollToLatest(): void {
    setTimeout(() => {
      const element = this.conversation?.nativeElement;
      if (element) {
        element.scrollTop = element.scrollHeight;
      }
    });
  }

  private getErrorMessage(
    error: HttpErrorResponse,
    language: 'ar' | 'fr',
  ): string {
    if (language === 'ar') {
      if (error.status === 0) {
        return 'تعذر الاتصال بالخادم. تأكد من تشغيل FastAPI.';
      }
      if (error.status === 503) {
        return 'نموذج Ollama المحلي غير متاح. يرجى تشغيله ثم المحاولة مجددًا.';
      }
      return error.error?.detail || 'حدث خطأ. يرجى المحاولة مرة أخرى.';
    }

    if (error.status === 0) {
      return 'Le backend est inaccessible. Vérifiez que FastAPI est démarré.';
    }
    if (error.status === 503) {
      return "Le modèle local Ollama n'est pas disponible. Veuillez le démarrer.";
    }
    return error.error?.detail || 'Une erreur est survenue. Veuillez réessayer.';
  }
}
