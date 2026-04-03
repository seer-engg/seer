import * as React from 'react';
import { HelpCircle } from 'lucide-react';

import { Label } from '@/components/ui/label';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { cn } from '@/lib/utils';

export interface FormFieldProps {
  // Required
  label: string;
  children: React.ReactNode;

  // Optional metadata
  description?: string; // Shows help icon with tooltip
  defaultValue?: string | number; // Shows below label as "Default: {value}"
  required?: boolean; // Adds red asterisk to label

  // Layout options
  labelClassName?: string; // Custom classes for label column
  inputClassName?: string; // Custom classes for input column

  // Alignment
  alignItems?: 'start' | 'center'; // Default: 'start'

  // For proper label association
  htmlFor?: string;
}

export function FormField({
  label,
  children,
  description,
  defaultValue,
  required = false,
  labelClassName,
  inputClassName,
  alignItems = 'start',
  htmlFor,
}: FormFieldProps) {
  return (
    <div className="flex flex-col gap-1.5">
      {/* Label Section */}
      <div className={cn('space-y-1 text-left', labelClassName)}>
        <div className="flex items-center gap-1">
          <Label htmlFor={htmlFor} className="text-xs font-medium">
            {label}
            {required && <span className="text-destructive ml-1">*</span>}
          </Label>
          {description && (
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  type="button"
                  className="text-muted-foreground hover:text-foreground transition-colors"
                  aria-label={`About ${label}`}
                >
                  <HelpCircle className="h-3.5 w-3.5" />
                </button>
              </TooltipTrigger>
              <TooltipContent className="max-w-xs text-xs leading-relaxed">
                {description}
              </TooltipContent>
            </Tooltip>
          )}
        </div>
        {defaultValue !== undefined && (
          <p className="text-xs text-muted-foreground italic">
            Default:{' '}
            {typeof defaultValue === 'object'
              ? JSON.stringify(defaultValue)
              : String(defaultValue)}
          </p>
        )}
      </div>

      {/* Input Section */}
      <div className={cn('space-y-1 relative', inputClassName)}>{children}</div>
    </div>
  );
}
