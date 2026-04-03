import { useEffect, useMemo } from 'react';
import type { Tool } from '@/components/workflows/buildtypes';
import type { JsonObject } from '@/types/workflow-spec';
import { useToolingData } from './useToolingData';

export function useWorkflowTools() {
  const { tools: rawTools, toolsLoading, toolsLoaded, refreshToolingData } = useToolingData();

  useEffect(() => {
    if (!toolsLoaded && !toolsLoading) {
      void refreshToolingData();
    }
  }, [toolsLoaded, toolsLoading, refreshToolingData]);

  const tools: Tool[] = useMemo(
    () =>
      rawTools.map((t) => ({
        name: t.name,
        description: t.description,
        provider: t.provider ?? undefined,
        integration_type: (t.integration_type as string | undefined) ?? undefined,
        output_schema: (t.output_schema as JsonObject | null) ?? null,
      })),
    [rawTools],
  );

  const isLoadingTools = toolsLoading || !toolsLoaded;

  return { tools, isLoadingTools };
}
