import type { IntegrationResource } from '@/lib/api-client';
import { bindSupabaseProject } from '@/lib/api-client';

export async function bindOAuthProject(config: {
  projectRef: string;
  connectionId?: string;
}): Promise<{ resource: IntegrationResource }> {
  return bindSupabaseProject({
    projectRef: config.projectRef,
    connectionId: config.connectionId,
  });
}
