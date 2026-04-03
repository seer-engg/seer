/**
 * Image Detection Utility
 *
 * Detects if a value is an image URL for rendering inline previews
 * in HITL display items.
 */

// Common image file extensions
const IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg', '.bmp', '.ico'];

// Known image hosting domains (partial matches)
const IMAGE_HOSTS = [
  'oaidalleapiprodscus.blob.core.windows.net', // OpenAI DALL-E
  'i.imgur.com',
  'imgur.com',
  'cloudinary.com',
  'res.cloudinary.com',
  'images.unsplash.com',
  'cdn.midjourney.com',
];

/**
 * workflow_file_ref object structure from workflow execution
 */
export interface WorkflowFileRef {
  _type: 'workflow_file_ref';
  file_id: string;
  filename: string;
  mime_type: string;
  size_bytes?: number;
}

/**
 * Result type for image detection
 */
export type ImageDisplayResult =
  | { type: 'url'; url: string }
  | { type: 'file_ref'; fileRef: WorkflowFileRef }
  | { type: 'none' };

/**
 * Check if a string is a data URL for an image
 */
function isImageDataUrl(value: string): boolean {
  return value.startsWith('data:image/');
}

/**
 * Check if a URL has an image file extension
 */
function hasImageExtension(url: string): boolean {
  try {
    const urlObj = new URL(url);
    const pathname = urlObj.pathname.toLowerCase();
    return IMAGE_EXTENSIONS.some((ext) => pathname.endsWith(ext));
  } catch {
    return false;
  }
}

/**
 * Check if a URL is from a known image hosting service
 */
function isKnownImageHost(url: string): boolean {
  try {
    const urlObj = new URL(url);
    const hostname = urlObj.hostname.toLowerCase();
    return IMAGE_HOSTS.some((host) => hostname.includes(host));
  } catch {
    return false;
  }
}

/**
 * Detect if a string value is an image URL
 *
 * Checks for:
 * - Data URLs (data:image/...)
 * - HTTP URLs with image extensions (.png, .jpg, etc.)
 * - Known image hosting services (DALL-E, imgur, cloudinary, etc.)
 */
export function isImageUrl(value: string): boolean {
  if (!value || typeof value !== 'string') {
    return false;
  }

  // Check data URLs
  if (isImageDataUrl(value)) {
    return true;
  }

  // Check HTTP/HTTPS URLs
  if (value.startsWith('http://') || value.startsWith('https://')) {
    return hasImageExtension(value) || isKnownImageHost(value);
  }

  return false;
}

/**
 * Check if a value is a workflow_file_ref object with an image mime type
 */
export function isImageFileRef(value: unknown): value is WorkflowFileRef {
  if (!value || typeof value !== 'object') {
    return false;
  }

  const obj = value as Record<string, unknown>;

  if (obj._type !== 'workflow_file_ref') {
    return false;
  }

  if (typeof obj.file_id !== 'string' || typeof obj.mime_type !== 'string') {
    return false;
  }

  // Check if mime type is an image
  return obj.mime_type.startsWith('image/');
}

/**
 * Detect the type of image display for a value
 *
 * Returns typed result for routing display logic:
 * - { type: 'url', url: string } - Direct image URL
 * - { type: 'file_ref', fileRef: WorkflowFileRef } - File reference to fetch
 * - { type: 'none' } - Not an image
 *
 * Handles nested structures like image_gen output:
 * { file: { _type: 'workflow_file_ref', ... }, image_url: 'data:image/...' }
 */
export function detectImageDisplay(value: unknown): ImageDisplayResult {
  // Check for string image URLs
  if (typeof value === 'string' && isImageUrl(value)) {
    return { type: 'url', url: value };
  }

  // Check for workflow_file_ref objects directly
  if (isImageFileRef(value)) {
    return { type: 'file_ref', fileRef: value };
  }

  // Check for nested object structures (e.g., image_gen output)
  if (value && typeof value === 'object') {
    const obj = value as Record<string, unknown>;

    // Check for image_url property (common in image_gen output)
    if (typeof obj.image_url === 'string' && isImageUrl(obj.image_url)) {
      return { type: 'url', url: obj.image_url };
    }

    // Check for nested file property containing workflow_file_ref
    if (isImageFileRef(obj.file)) {
      return { type: 'file_ref', fileRef: obj.file };
    }
  }

  return { type: 'none' };
}
