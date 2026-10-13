import { HttpClient } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { BehaviorSubject, catchError, Observable, switchMap, tap, throwError } from 'rxjs';

import {
  AuthResponse,
  LoginRequest,
  RegisterRequest,
  User,
} from '../../shared/models/auth.models';


export const AUTH_TOKEN_STORAGE_KEY = 'douane_ai_access_token';
const AUTH_USER_STORAGE_KEY = 'douane_ai_current_user';

export function readAuthToken(): string | null {
  return localStorage.getItem(AUTH_TOKEN_STORAGE_KEY);
}

@Injectable({ providedIn: 'root' })
export class AuthService {
  private readonly http = inject(HttpClient);
  private readonly apiUrl = 'http://127.0.0.1:8000';
  private readonly currentUserSubject = new BehaviorSubject<User | null>(
    this.readStoredUser(),
  );

  readonly currentUser$ = this.currentUserSubject.asObservable();

  register(data: RegisterRequest): Observable<AuthResponse> {
    return this.http.post<User>(`${this.apiUrl}/auth/register`, data).pipe(
      switchMap(() =>
        this.login({ email: data.email, password: data.password }),
      ),
    );
  }

  login(data: LoginRequest): Observable<AuthResponse> {
    return this.http
      .post<AuthResponse>(`${this.apiUrl}/auth/login`, data)
      .pipe(tap((response) => this.storeAuth(response)));
  }

  logout(): void {
    localStorage.removeItem(AUTH_TOKEN_STORAGE_KEY);
    localStorage.removeItem(AUTH_USER_STORAGE_KEY);
    this.currentUserSubject.next(null);
  }

  getCurrentUser(): User | null {
    return this.currentUserSubject.value;
  }

  getToken(): string | null {
    return readAuthToken();
  }

  isLoggedIn(): boolean {
    return Boolean(this.getToken());
  }

  loadMe(): Observable<User> {
    return this.http.get<User>(`${this.apiUrl}/auth/me`).pipe(
      tap((user) => this.storeUser(user)),
      catchError((error) => {
        this.logout();
        return throwError(() => error);
      }),
    );
  }

  private storeAuth(response: AuthResponse): void {
    // TODO: localStorage is acceptable for this MVP; use a hardened token
    // strategy (for example secure HTTP-only cookies) before production.
    localStorage.setItem(AUTH_TOKEN_STORAGE_KEY, response.access_token);
    this.storeUser(response.user);
  }

  private storeUser(user: User): void {
    localStorage.setItem(AUTH_USER_STORAGE_KEY, JSON.stringify(user));
    this.currentUserSubject.next(user);
  }

  private readStoredUser(): User | null {
    if (!readAuthToken()) {
      return null;
    }

    const storedUser = localStorage.getItem(AUTH_USER_STORAGE_KEY);
    if (!storedUser) {
      return null;
    }

    try {
      return JSON.parse(storedUser) as User;
    } catch {
      localStorage.removeItem(AUTH_USER_STORAGE_KEY);
      return null;
    }
  }
}