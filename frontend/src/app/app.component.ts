import { CommonModule } from '@angular/common';
import { Component, OnInit, inject } from '@angular/core';
import { finalize } from 'rxjs';

import { AuthService } from './core/services/auth.service';
import { AuthComponent } from './features/auth/auth.component';
import { ChatComponent } from './features/chat/chat.component';


@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, AuthComponent, ChatComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css',
})
export class AppComponent implements OnInit {
  private readonly authService = inject(AuthService);

  readonly currentUser$ = this.authService.currentUser$;
  authReady = false;

  ngOnInit(): void {
    if (!this.authService.isLoggedIn()) {
      this.authReady = true;
      return;
    }

    this.authService
      .loadMe()
      .pipe(finalize(() => (this.authReady = true)))
      .subscribe({ error: () => undefined });
  }

  logout(): void {
    this.authService.logout();
  }
}