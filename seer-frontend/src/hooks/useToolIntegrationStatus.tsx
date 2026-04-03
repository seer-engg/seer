import { useMemo } from 'react';
import type { ToolIntegrationStatus } from '@/hooks/useToolIntegration';
import { Badge } from '@/components/ui/badge';
import { DynamicIntegrationIcon } from '@/components/ui/DynamicIntegrationIcon';
import { Wrench, CheckCircle2, AlertTriangle } from 'lucide-react';

export function useToolIntegrationStatus(
  toolName: string,
  status: ToolIntegrationStatus | undefined,
  isLoading: boolean
) {
  return useMemo(() => {
    if (!toolName || isLoading) {
      return {
        icon: <Wrench className="w-4 h-4 text-primary" />,
        statusBadge: null,
        needsAuth: false,
      };
    }

    if (!status) {
      return {
        icon: <Wrench className="w-4 h-4 text-primary" />,
        statusBadge: null,
        needsAuth: false,
      };
    }

    const intIcon = (
      <DynamicIntegrationIcon
        integrationType={status.integrationType}
        width={16}
        height={16}
      />
    );

    // No OAuth required
    if (!status.integrationType) {
      return {
        icon: intIcon,
        statusBadge: null,
        needsAuth: false,
      };
    }

    // Connected
    if (status.isConnected) {
      return {
        icon: intIcon,
        statusBadge: (
          <Badge
            variant="secondary"
            className="flex items-center gap-1 text-[10px] px-1.5 py-0 h-5 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20"
          >
            <CheckCircle2 className="w-3 h-3" />
          </Badge>
        ),
        needsAuth: false,
      };
    }

    // Needs authorization
    return {
      icon: intIcon,
      statusBadge: (
        <Badge
          variant="secondary"
          className="flex items-center gap-1 text-[10px] px-1.5 py-0 h-5 bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20"
        >
          <AlertTriangle className="w-3 h-3" />
          Not connected
        </Badge>
      ),
      needsAuth: true,
    };
  }, [toolName, status, isLoading]);
}
