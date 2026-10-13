import { HttpInterceptorFn } from '@angular/common/http';

import { readAuthToken } from '../services/auth.service';


export const authInterceptor: HttpInterceptorFn = (request, next) => {
  const token = readAuthToken();
  if (!token) {
    return next(request);
  }

  return next(
    request.clone({
      setHeaders: { Authorization: `Bearer ${token}` },
    }),
  );
};