import { http, HttpResponse } from 'msw';

export const authHandlers = [
  // Mock Clerk session
  http.get('/api/auth/session', () => {
    return HttpResponse.json({
      user: {
        id: 'test-user-id',
        email: 'test@example.com',
      },
      session: {
        token: 'test-token',
      },
    });
  }),
];
