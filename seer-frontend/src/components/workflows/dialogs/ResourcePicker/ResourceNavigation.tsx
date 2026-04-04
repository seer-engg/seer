import React from 'react';
import { ChevronLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';

export interface ResourceNavigationProps {
  currentPath: Array<{ id: string; name: string }>;
  onBack: () => void;
  onNavigateToPath: (index: number) => void;
  supportsHierarchy: boolean;
}

export function ResourceNavigation({
  currentPath,
  onBack,
  onNavigateToPath,
  supportsHierarchy,
}: ResourceNavigationProps) {
  if (!supportsHierarchy || currentPath.length === 0) {
    return null;
  }

  return (
    <div className="flex items-center gap-1 text-sm text-muted-foreground px-1 py-2 border-b">
      <Button
        variant="ghost"
        size="sm"
        className="h-6 px-2"
        onClick={onBack}
      >
        <ChevronLeft className="h-4 w-4 mr-1" />
        Back
      </Button>
      <span className="text-muted-foreground">/</span>
      {currentPath.map((crumb, index) => (
        <React.Fragment key={crumb.id}>
          <span
            className="cursor-pointer hover:text-foreground truncate max-w-[100px]"
            onClick={() => onNavigateToPath(index)}
            title={crumb.name}
          >
            {crumb.name}
          </span>
          {index < currentPath.length - 1 && (
            <span className="text-muted-foreground">/</span>
          )}
        </React.Fragment>
      ))}
    </div>
  );
}
