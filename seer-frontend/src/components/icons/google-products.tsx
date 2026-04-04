import type { SVGProps } from 'react';

/**
 * Custom SVG icons for integrations
 * - Google products: Logo.dev can't distinguish sub-products (all return "G" logo)
 * - Other common integrations: Ensures consistent, reliable icons
 */

export function GoogleDriveIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 87.3 78" xmlns="http://www.w3.org/2000/svg" {...props}>
      <path d="M6.6 66.85l3.85 6.65c.8 1.4 1.95 2.5 3.3 3.3l13.75-23.8H0c0 1.55.4 3.1 1.2 4.5l5.4 9.35z" fill="#0066da" />
      <path d="M43.65 25L29.9 1.2c-1.35.8-2.5 1.9-3.3 3.3L1.2 47.5c-.8 1.4-1.2 2.95-1.2 4.5h27.5l16.15-27z" fill="#00ac47" />
      <path d="M73.55 76.8c1.35-.8 2.5-1.9 3.3-3.3l1.6-2.75L86.1 57c.8-1.4 1.2-2.95 1.2-4.5H59.85l6.15 12.2 7.55 12.1z" fill="#ea4335" />
      <path d="M43.65 25L57.4 1.2C56.05.4 54.5 0 52.95 0H34.35c-1.55 0-3.1.4-4.45 1.2L43.65 25z" fill="#00832d" />
      <path d="M59.85 52H87.3c0-1.55-.4-3.1-1.2-4.5L73.55 25.8c-.8-1.4-1.95-2.5-3.3-3.3L57.4 1.2 43.65 25l16.2 27z" fill="#2684fc" />
      <path d="M27.5 52L13.75 75.8c1.35.8 2.9 1.2 4.45 1.2h51.9c1.55 0 3.1-.4 4.45-1.2L59.85 52H27.5z" fill="#ffba00" />
    </svg>
  );
}

export function GoogleSheetsIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg" {...props}>
      <path fill="#43a047" d="M37 45H11c-1.657 0-3-1.343-3-3V6c0-1.657 1.343-3 3-3h19l10 10v29c0 1.657-1.343 3-3 3z" />
      <path fill="#c8e6c9" d="M40 13H30V3l10 10z" />
      <path fill="#fff" d="M30 13l10 10V13z" opacity=".2" />
      <path fill="#c8e6c9" d="M31 23H17c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V25c0-1.1-.9-2-2-2z" />
      <path fill="#43a047" d="M17 23h6v6h-6zM25 23h6v6h-6zM17 31h6v6h-6zM25 31h6v6h-6z" />
    </svg>
  );
}

export function GoogleDocsIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg" {...props}>
      <path fill="#2196f3" d="M37 45H11c-1.657 0-3-1.343-3-3V6c0-1.657 1.343-3 3-3h19l10 10v29c0 1.657-1.343 3-3 3z" />
      <path fill="#bbdefb" d="M40 13H30V3l10 10z" />
      <path fill="#fff" d="M30 13l10 10V13z" opacity=".2" />
      <path fill="#fff" d="M15 23h18v2H15zM15 28h18v2H15zM15 33h12v2H15z" />
    </svg>
  );
}

export function GmailIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg" {...props}>
      <path fill="#4caf50" d="M45 16.2l-5 2.75-5 4.75V40h7c1.657 0 3-1.343 3-3V16.2z" />
      <path fill="#1e88e5" d="M3 16.2l3.614 1.71L13 23.7V40H6c-1.657 0-3-1.343-3-3V16.2z" />
      <polygon fill="#e53935" points="35,11.2 24,19.45 13,11.2 12,17 13,23.7 24,31.95 35,23.7 36,17" />
      <path fill="#c62828" d="M3 12.298V16.2l10 7.5V11.2L9.876 8.859C9.132 8.301 8.228 8 7.298 8h0C4.924 8 3 9.924 3 12.298z" />
      <path fill="#fbc02d" d="M45 12.298V16.2l-10 7.5V11.2l3.124-2.341C38.868 8.301 39.772 8 40.702 8h0C43.076 8 45 9.924 45 12.298z" />
    </svg>
  );
}

export function GoogleCalendarIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg" {...props}>
      <path fill="#fff" d="M37 45H11c-1.657 0-3-1.343-3-3V10c0-1.657 1.343-3 3-3h26c1.657 0 3 1.343 3 3v32c0 1.657-1.343 3-3 3z" />
      <path fill="#1e88e5" d="M37 7H11c-1.657 0-3 1.343-3 3v3h32v-3c0-1.657-1.343-3-3-3z" />
      <path fill="#1e88e5" d="M16 16h4v4h-4zM22 16h4v4h-4zM28 16h4v4h-4zM16 22h4v4h-4zM22 22h4v4h-4zM28 22h4v4h-4zM16 28h4v4h-4zM22 28h4v4h-4zM28 28h4v4h-4zM16 34h4v4h-4zM22 34h4v4h-4z" />
      <circle cx="33" cy="8" r="2" fill="#1565c0" />
      <circle cx="15" cy="8" r="2" fill="#1565c0" />
    </svg>
  );
}

export function GoogleSlidesIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg" {...props}>
      <path fill="#ff9800" d="M37 45H11c-1.657 0-3-1.343-3-3V6c0-1.657 1.343-3 3-3h19l10 10v29c0 1.657-1.343 3-3 3z" />
      <path fill="#ffe0b2" d="M40 13H30V3l10 10z" />
      <path fill="#fff" d="M30 13l10 10V13z" opacity=".2" />
      <path fill="#ffe0b2" d="M31 23H17c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V25c0-1.1-.9-2-2-2z" />
    </svg>
  );
}

// ============================================================================
// Other Common Integration Icons
// ============================================================================

export function LinkedInIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg" {...props}>
      <path fill="#0288d1" d="M42 37c0 2.762-2.238 5-5 5H11c-2.761 0-5-2.238-5-5V11c0-2.762 2.239-5 5-5h26c2.762 0 5 2.238 5 5v26z" />
      <path fill="#fff" d="M12 19h5v17h-5zM14.485 17h-.028C12.965 17 12 15.888 12 14.499 12 13.08 12.995 12 14.514 12c1.521 0 2.458 1.08 2.486 2.499C17 15.887 16.035 17 14.485 17zM36 36h-5v-9.099c0-2.198-1.225-3.698-3.192-3.698-1.501 0-2.313 1.012-2.707 1.99-.144.35-.101.858-.101 1.367V36h-5V19h5v2.616C25.721 20.5 26.85 19 29.738 19c3.578 0 6.261 2.25 6.261 7.274L36 36z" />
    </svg>
  );
}

export function GitHubIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg" {...props}>
      <path fill="#24292f" d="M24 4C12.954 4 4 12.954 4 24c0 8.887 5.801 16.411 13.82 19.016.12.023.32-.08.32-.32v-2.72c-5.6 1.2-6.8-2.72-6.8-2.72-.92-2.32-2.24-2.96-2.24-2.96-1.84-1.28.16-1.28.16-1.28 2 .16 3.12 2.08 3.12 2.08 1.76 3.04 4.64 2.16 5.76 1.68.16-1.28.72-2.16 1.28-2.64-4.48-.48-9.12-2.24-9.12-9.92 0-2.24.8-4 2.08-5.44-.24-.56-.88-2.56.16-5.36 0 0 1.68-.56 5.52 2.08 1.6-.48 3.28-.72 4.96-.72s3.36.24 4.96.72c3.84-2.56 5.52-2.08 5.52-2.08 1.04 2.8.4 4.8.16 5.36 1.28 1.44 2.08 3.2 2.08 5.44 0 7.68-4.64 9.44-9.12 9.92.72.64 1.36 1.84 1.36 3.68v5.44c0 .24.2.4.4.32C38.199 40.411 44 32.887 44 24c0-11.046-8.954-20-20-20z" />
    </svg>
  );
}

export function DiscordIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg" {...props}>
      <path fill="#5865f2" d="M40 12c0 0-4.585-3.588-10-4l-.488.976C34.408 10.174 36.654 11.891 39 14c-4.045-2.065-8.039-4-15-4s-10.955 1.935-15 4c2.346-2.109 5.018-4.015 9.488-5.024L18 8c-5.681.537-10 4-10 4s-5.121 7.425-6 22c5.162 5.953 13 6 13 6l1.639-2.185C13.857 36.848 10.715 35.121 8 32c3.238 2.45 8.125 5 16 5s12.762-2.55 16-5c-2.715 3.121-5.857 4.848-8.639 5.815L33 40s7.838-.047 13-6C45.121 19.425 40 12 40 12zM17.5 30c-1.933 0-3.5-1.791-3.5-4s1.567-4 3.5-4 3.5 1.791 3.5 4-1.567 4-3.5 4zM30.5 30c-1.933 0-3.5-1.791-3.5-4s1.567-4 3.5-4 3.5 1.791 3.5 4-1.567 4-3.5 4z" />
    </svg>
  );
}

export function SlackIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg" {...props}>
      <path fill="#33d375" d="M33 8c0-2.209-1.791-4-4-4s-4 1.791-4 4v11c0 2.209 1.791 4 4 4s4-1.791 4-4V8z" />
      <path fill="#33d375" d="M43 19c0 2.209-1.791 4-4 4h-4v-4c0-2.209 1.791-4 4-4s4 1.791 4 4z" />
      <path fill="#40c4ff" d="M8 15c-2.209 0-4 1.791-4 4s1.791 4 4 4h11c2.209 0 4-1.791 4-4s-1.791-4-4-4H8z" />
      <path fill="#40c4ff" d="M19 5c-2.209 0-4 1.791-4 4v4h4c2.209 0 4-1.791 4-4s-1.791-4-4-4z" />
      <path fill="#e91e63" d="M15 40c0 2.209 1.791 4 4 4s4-1.791 4-4V29c0-2.209-1.791-4-4-4s-4 1.791-4 4v11z" />
      <path fill="#e91e63" d="M5 29c0-2.209 1.791-4 4-4h4v4c0 2.209-1.791 4-4 4s-4-1.791-4-4z" />
      <path fill="#ffc107" d="M40 33c2.209 0 4-1.791 4-4s-1.791-4-4-4H29c-2.209 0-4 1.791-4 4s1.791 4 4 4h11z" />
      <path fill="#ffc107" d="M29 43c2.209 0 4-1.791 4-4v-4h-4c-2.209 0-4 1.791-4 4s1.791 4 4 4z" />
    </svg>
  );
}
