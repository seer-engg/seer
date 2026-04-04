import { backendApiClient } from './api-client';

export interface WhatsAppLinkStatus {
  linked: boolean;
  phone_number?: string;
}

export interface WhatsAppLinkResponse {
  message: string;
}

export interface WhatsAppVerifyResponse {
  message: string;
}

export async function getWhatsAppLinkStatus(): Promise<WhatsAppLinkStatus> {
  return backendApiClient.request('/api/users/me/whatsapp/link');
}

export async function linkWhatsAppPhone(phone_number: string): Promise<WhatsAppLinkResponse> {
  return backendApiClient.request('/api/users/me/whatsapp/link', {
    method: 'POST',
    body: { phone_number },
  });
}

export async function verifyWhatsAppPhone(phone_number: string, code: string): Promise<WhatsAppVerifyResponse> {
  return backendApiClient.request('/api/users/me/whatsapp/verify', {
    method: 'POST',
    body: { phone_number, code },
  });
}

export async function unlinkWhatsApp(): Promise<void> {
  return backendApiClient.request('/api/users/me/whatsapp/link', { method: 'DELETE' });
}
