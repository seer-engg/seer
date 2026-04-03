/**
 * DynamicIntegrationIcon Component
 *
 * Renders integration icons dynamically based on metadata from the backend.
 * Supports three icon types:
 * - 'url': External image URL (Logo.dev, CDN, etc.)
 * - 'lucide': Lucide icon name (dynamically resolved)
 * - 'svg': Inline SVG string
 *
 * Falls back to custom SVG icons for known integrations (Google products, LinkedIn, etc.),
 * then to a generic Wrench icon if no icon is available.
 */
import { useState } from 'react';
import { Wrench } from 'lucide-react';

import { useIntegrationMetadataStore } from '@/stores/integrationMetadataStore';
import {
  GoogleDriveIcon,
  GoogleSheetsIcon,
  GoogleDocsIcon,
  GmailIcon,
  GoogleCalendarIcon,
  GoogleSlidesIcon,
  LinkedInIcon,
  GitHubIcon,
  DiscordIcon,
  SlackIcon,
} from '@/components/icons/google-products';

import { cn } from '@/lib/utils';

/**
 * Custom SVG icons for integrations where backend metadata may not be available
 */
const CUSTOM_INTEGRATION_ICONS: Record<string, React.ComponentType<React.SVGProps<SVGSVGElement>>> = {
  gmail: GmailIcon,
  google_drive: GoogleDriveIcon,
  google_sheets: GoogleSheetsIcon,
  google_docs: GoogleDocsIcon,
  google_calendar: GoogleCalendarIcon,
  google_slides: GoogleSlidesIcon,
  linkedin: LinkedInIcon,
  github: GitHubIcon,
  discord: DiscordIcon,
  slack: SlackIcon,
};

export interface DynamicIntegrationIconProps {
  /** The integration type (e.g., 'gmail', 'github') */
  integrationType: string | null;
  /** Icon width in pixels (default: 16) */
  width?: number;
  /** Icon height in pixels (default: 16) */
  height?: number;
  /** Additional CSS classes */
  className?: string;
  /** Alt text for image-based icons */
  alt?: string;
}

/**
 * Renders an icon from a URL, with error handling for failed loads.
 */
function UrlIcon({
  url,
  width,
  height,
  className,
  alt,
  onError,
}: {
  url: string;
  width: number;
  height: number;
  className?: string;
  alt: string;
  onError: () => void;
}) {
  return (
    <img
      src={url}
      width={width}
      height={height}
      alt={alt}
      className={cn('object-contain', className)}
      onError={onError}
    />
  );
}

/**
 * Renders an inline SVG string.
 */
function InlineSvgIcon({
  svgString,
  width,
  height,
  className,
}: {
  svgString: string;
  width: number;
  height: number;
  className?: string;
}) {
  // Inject width/height attributes into the SVG string
  const processedSvg = svgString
    .replace(/width="[^"]*"/, `width="${width}"`)
    .replace(/height="[^"]*"/, `height="${height}"`);

  return (
    <span
      className={cn('inline-flex items-center justify-center', className)}
      style={{ width, height }}
      dangerouslySetInnerHTML={{ __html: processedSvg }}
    />
  );
}

/**
 * Generic fallback icon when no icon metadata is available.
 */
function FallbackIcon({
  width,
  height,
  className,
}: {
  width: number;
  height: number;
  className?: string;
}) {
  return <Wrench className={cn('text-muted-foreground', className)} style={{ width, height }} />;
}

/**
 * Main component that renders the appropriate icon based on metadata.
 */
export function DynamicIntegrationIcon({
  integrationType,
  width = 16,
  height = 16,
  className,
  alt,
}: DynamicIntegrationIconProps) {
  const [urlFailed, setUrlFailed] = useState(false);
  const getIcon = useIntegrationMetadataStore((state) => state.getIcon);
  const getDisplayName = useIntegrationMetadataStore((state) => state.getDisplayName);

  // If no integration type, show generic icon
  if (!integrationType) {
    return <FallbackIcon width={width} height={height} className={className} />;
  }

  const icon = getIcon(integrationType);
  const displayName = alt ?? getDisplayName(integrationType);

  // If we have icon metadata from the backend and URL hasn't failed
  if (icon && !urlFailed) {
    switch (icon.type) {
      case 'url':
        return (
          <UrlIcon
            url={icon.value}
            width={width}
            height={height}
            className={className}
            alt={displayName}
            onError={() => setUrlFailed(true)}
          />
        );

      case 'svg':
        return (
          <InlineSvgIcon
            svgString={icon.value}
            width={width}
            height={height}
            className={className}
          />
        );

      case 'lucide':
        // Lucide icons would require dynamic import; fall through to fallback
        break;
    }
  }

  // Try custom SVG icons for known integrations
  const normalizedType = integrationType.toLowerCase().trim();
  const CustomIcon = CUSTOM_INTEGRATION_ICONS[normalizedType];
  if (CustomIcon) {
    return <CustomIcon className={className} style={{ width, height }} />;
  }

  // Final fallback to generic icon
  return <FallbackIcon width={width} height={height} className={className} />;
}
