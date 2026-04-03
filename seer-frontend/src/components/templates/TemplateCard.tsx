import { Star } from 'lucide-react';

import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { DynamicIntegrationIcon } from '@/components/ui/DynamicIntegrationIcon';
import { cn } from '@/lib/utils';
import type { TemplateSummary } from '@/types/templates';

export interface TemplateCardProps {
  template: TemplateSummary;
  onClick: () => void;
}

/**
 * Card component for displaying a template in the gallery grid.
 * Shows template name, description, featured badge, and integration icons.
 */
export function TemplateCard({ template, onClick }: TemplateCardProps) {
  const { name, description, is_featured, icon } = template;
  const integrations = template.required_integrations ?? [];

  // Show first 3 integrations, with overflow count
  const displayedIntegrations = integrations.slice(0, 3);
  const overflowCount = integrations.length - 3;

  return (
    <Card
      className={cn(
        'cursor-pointer transition-all hover:shadow-md hover:border-primary/50',
        'group relative overflow-hidden',
      )}
      onClick={onClick}
    >
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-2">
            {icon ? (
              <span className="text-2xl">{icon}</span>
            ) : (
              <div className="w-8 h-8 rounded-md bg-primary/10 flex items-center justify-center">
                <span className="text-primary text-sm font-semibold">
                  {name.charAt(0).toUpperCase()}
                </span>
              </div>
            )}
            <h3 className="font-semibold text-base line-clamp-1">{name}</h3>
          </div>
          {is_featured && (
            <Badge variant="secondary" className="shrink-0 gap-1">
              <Star className="w-3 h-3 fill-amber-500 text-amber-500" />
              Featured
            </Badge>
          )}
          {template.visibility === 'private' && (
            <Badge variant="outline" className="shrink-0 text-xs">
              Private
            </Badge>
          )}
        </div>
      </CardHeader>

      <CardContent className="pt-0">
        <p className="text-sm text-muted-foreground line-clamp-2 mb-3">{description}</p>

        {integrations.length > 0 && (
          <div className="flex items-center gap-1.5">
            {displayedIntegrations.map((integration) => (
              <div
                key={`${integration.provider}-${integration.integration_type}`}
                className="flex items-center justify-center w-6 h-6 rounded-md bg-muted"
                title={integration.integration_type}
              >
                <DynamicIntegrationIcon
                  integrationType={integration.integration_type}
                  width={14}
                  height={14}
                />
              </div>
            ))}
            {overflowCount > 0 && (
              <div className="flex items-center justify-center w-6 h-6 rounded-md bg-muted text-xs text-muted-foreground font-medium">
                +{overflowCount}
              </div>
            )}
          </div>
        )}
      </CardContent>

      {/* Hover gradient effect */}
      <div className="pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-200 group-hover:opacity-100 bg-[radial-gradient(circle_at_50%_0%,rgba(99,102,241,0.08),transparent_50%)]" />
    </Card>
  );
}
