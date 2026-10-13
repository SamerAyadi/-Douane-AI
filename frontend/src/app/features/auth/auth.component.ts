import { CommonModule } from '@angular/common';
import { HttpErrorResponse } from '@angular/common/http';
import { Component, inject } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { finalize } from 'rxjs';

import { AuthService } from '../../core/services/auth.service';


@Component({
  selector: 'app-auth',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './auth.component.html',
  styleUrl: './auth.component.css',
})
export class AuthComponent {
  private readonly authService = inject(AuthService);

  registerMode = false;
  fullName = '';
  email = '';
  password = '';
  loading = false;
  errorMessage = '';

  toggleMode(): void {
    if (this.loading) {
      return;
    }
    this.registerMode = !this.registerMode;
    this.password = '';
    this.errorMessage = '';
  }

  submit(): void {
    const email = this.email.trim();
    const fullName = this.fullName.trim();

    if (!email || !this.password) {
      this.errorMessage = 'Veuillez saisir votre e-mail et votre mot de passe.';
      return;
    }
    if (this.registerMode && !fullName) {
      this.errorMessage = 'Veuillez saisir votre nom complet.';
      return;
    }
    if (this.registerMode && this.password.length < 8) {
      this.errorMessage = 'Le mot de passe doit contenir au moins 8 caractères.';
      return;
    }

    this.loading = true;
    this.errorMessage = '';

    const request$ = this.registerMode
      ? this.authService.register({
          email,
          password: this.password,
          full_name: fullName,
        })
      : this.authService.login({ email, password: this.password });

    request$
      .pipe(finalize(() => (this.loading = false)))
      .subscribe({
        next: () => {
          this.password = '';
        },
        error: (error: HttpErrorResponse) => {
          this.errorMessage = this.getErrorMessage(error);
        },
      });
  }

  private getErrorMessage(error: HttpErrorResponse): string {
    if (error.status === 0) {
      return 'Le serveur est inaccessible. Vérifiez que FastAPI est démarré.';
    }
    if (error.status === 401) {
      return 'E-mail ou mot de passe incorrect.';
    }
    if (error.status === 400 && this.registerMode) {
      return 'Cette adresse e-mail est déjà utilisée.';
    }
    return error.error?.detail || 'Une erreur est survenue. Veuillez réessayer.';
  }
}