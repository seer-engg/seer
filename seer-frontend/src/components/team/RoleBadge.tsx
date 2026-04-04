/**
 * RoleBadge Component
 *
 * Displays an organization role as a styled badge.
 */

import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import type { OrganizationRole } from '@/types/organization';
import { getRoleDisplayName } from '@/types/organization';

interface RoleBadgeProps {
  role: OrganizationRole;
  className?: string;
}

const roleStyles: Record<OrganizationRole, string> = {
  owner: 'bg-seer-500/10 text-seer-600 dark:text-seer-400 border-seer-500/20',
  admin: 'bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/20',
  user: 'bg-secondary text-secondary-foreground',
  consultant: 'bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20',
};

export function RoleBadge({ role, className }: RoleBadgeProps) {
  return (
    <Badge
      variant="outline"
      className={cn('text-xs font-medium', roleStyles[role], className)}
    >
      {getRoleDisplayName(role)}
    </Badge>
  );
}
