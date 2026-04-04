export interface VariableTreeItem {
  /** Display label at this level (e.g., "columns") */
  label: string;
  /** Full dot-path for insertion (e.g., "postgres_list_schemas-1.columns") */
  fullPath: string;
  /** Whether this item has child properties that can be drilled into */
  hasChildren: boolean;
}

/**
 * Given a flat list of variable paths and a current navigation path,
 * return the items visible at the current drill-down level.
 *
 * At root (currentPath === ""): shows top-level identifiers.
 * At deeper levels: shows immediate children under the current path.
 */
export function getVariableTreeItems(
  flatVariables: string[],
  currentPath: string,
): VariableTreeItem[] {
  const prefix = currentPath ? `${currentPath}.` : '';
  const itemMap = new Map<string, { fullPath: string; hasChildren: boolean }>();

  for (const variable of flatVariables) {
    let segment: string;
    let fullPath: string;

    if (prefix) {
      // Deeper level: only consider variables under the current path
      if (!variable.startsWith(prefix)) continue;
      const remainder = variable.slice(prefix.length);
      if (!remainder) continue;
      segment = getFirstSegment(remainder);
      fullPath = `${currentPath}.${segment}`;
    } else {
      // Root level: extract the first segment
      segment = getFirstSegment(variable);
      fullPath = segment;
    }

    const existing = itemMap.get(segment);
    if (existing) {
      // If the full variable is longer than our segment's fullPath, it has children
      if (variable.length > fullPath.length) {
        existing.hasChildren = true;
      }
    } else {
      itemMap.set(segment, {
        fullPath,
        hasChildren: variable.length > fullPath.length,
      });
    }
  }

  return Array.from(itemMap.entries())
    .map(([label, { fullPath, hasChildren }]) => ({ label, fullPath, hasChildren }))
    .sort((a, b) => a.label.localeCompare(b.label));
}

/**
 * Extract the first path segment from a dot-separated string.
 * Keeps array indices attached: "columns[0].file_id" → "columns[0]"
 */
function getFirstSegment(path: string): string {
  const dotIndex = path.indexOf('.');
  if (dotIndex === -1) return path;
  return path.slice(0, dotIndex);
}

/**
 * Filter tree items by a partial text input (case-insensitive substring match on label).
 */
export function filterTreeItems(
  items: VariableTreeItem[],
  filterText: string,
): VariableTreeItem[] {
  if (!filterText) return items;
  const lower = filterText.toLowerCase();
  return items.filter((item) => item.label.toLowerCase().includes(lower));
}

/**
 * Parse a currentPath string back into breadcrumb segments.
 * Returns array of { label, path } for each level.
 *
 * E.g., "node-1.columns[0]" → [
 *   { label: "node-1", path: "node-1" },
 *   { label: "columns[0]", path: "node-1.columns[0]" }
 * ]
 */
export function parseBreadcrumbs(
  currentPath: string,
): { label: string; path: string }[] {
  if (!currentPath) return [];

  const segments: { label: string; path: string }[] = [];
  let remaining = currentPath;
  let accumulated = '';

  while (remaining) {
    const segment = getFirstSegment(remaining);
    accumulated = accumulated ? `${accumulated}.${segment}` : segment;
    segments.push({ label: segment, path: accumulated });
    remaining = remaining.slice(segment.length);
    if (remaining.startsWith('.')) {
      remaining = remaining.slice(1);
    }
  }

  return segments;
}
