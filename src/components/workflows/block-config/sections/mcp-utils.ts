import { McpBlockConfig } from '../types';

export function buildFlatServerConfig(config: McpBlockConfig): { server: string; server_type: string; auth?: Record<string, unknown> } {
  const serverType = config.server_type || 'http';

  if (serverType === 'stdio') {
    const parts = [config.stdio_command || '', ...(config.stdio_args || [])].filter(Boolean);
    const server = parts.join(' ');
    const env = config.stdio_env || {};
    return {
      server,
      server_type: 'stdio',
      auth: Object.keys(env).length > 0 ? { env } : undefined,
    };
  }

  const headers = config.http_headers || {};
  return {
    server: config.server || '',
    server_type: 'http',
    auth: Object.keys(headers).length > 0 ? { headers } : undefined,
  };
}
