import { useState, useEffect, useCallback, useMemo } from 'react';
import type { IntegrationResource } from '@/lib/api-client';
import type { ResourceItem } from '../types';
import { useResourceFetch } from './useResourceFetch';
import {
  normalizeEndpoint,
  isSupabaseBindingEndpoint,
  isPostgresBindingEndpoint,
  buildBaseEndpoint,
  buildResourceItem,
} from '../resourcePickerHelpers';

interface UseResourcePickerConfig {
  config: {
    endpoint?: string;
    resource_type: string;
    depends_on?: string;
    value_field?: string;
    display_field?: string;
    search_enabled?: boolean;
  };
  provider?: string;
  value: string;
  onChange: (value: string, displayName: string) => void;
  dependsOnValues?: Record<string, unknown>;
}

export function useResourcePicker({
  config,
  provider,
  value,
  onChange,
  dependsOnValues,
}: UseResourcePickerConfig) {
  const [open, setOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [currentPath, setCurrentPath] = useState<Array<{ id: string; name: string }>>([]);
  const [selectedItem, setSelectedItem] = useState<ResourceItem | null>(null);

  const currentParentId = currentPath.length > 0 ? currentPath[currentPath.length - 1].id : undefined;
  const dependsOnParam = dependsOnValues ? JSON.stringify(dependsOnValues) : undefined;
  const hasMissingDependency = config.depends_on && (!dependsOnValues || !dependsOnValues[config.depends_on]);

  const normalizedEndpoint = useMemo(() => normalizeEndpoint(config.endpoint), [config.endpoint]);
  const isSupabaseBindingPicker = useMemo(() => isSupabaseBindingEndpoint(normalizedEndpoint), [normalizedEndpoint]);
  const isPostgresBindingPicker = useMemo(() => isPostgresBindingEndpoint(normalizedEndpoint), [normalizedEndpoint]);
  const baseEndpoint = useMemo(
    () => buildBaseEndpoint(normalizedEndpoint, provider, config.resource_type),
    [normalizedEndpoint, provider, config.resource_type]
  );

  const valueField = config.value_field || 'id';
  const displayField = config.display_field || 'display_name';

  const resourceFetch = useResourceFetch({
    baseEndpoint,
    resourceType: config.resource_type,
    searchQuery,
    currentParentId,
    dependsOnParam,
    valueField,
    displayField,
    normalizedEndpoint,
    searchEnabled: config.search_enabled,
    open,
    hasMissingDependency,
  });

  const applySupabaseBindingSelection = useCallback(
    (resource: IntegrationResource, fallback?: { displayName?: string; projectRef?: string }) => {
      const newItem = buildResourceItem(resource, fallback);
      setSelectedItem(newItem);
      onChange(newItem.id, newItem.display_name);
    },
    [onChange],
  );

  const handleNavigate = useCallback((item: ResourceItem) => {
    if (item.type === 'folder' && item.has_children) {
      setCurrentPath(prev => [...prev, { id: item.id, name: item.name }]);
      setSearchQuery('');
    }
  }, []);

  const handleSelect = useCallback((item: ResourceItem) => {
    if (item.type === 'folder' && resourceFetch.supportsHierarchy) {
      handleNavigate(item);
    } else {
      setSelectedItem(item);
      onChange(item.id, item.display_name);
      setOpen(false);
    }
  }, [resourceFetch.supportsHierarchy, handleNavigate, onChange]);

  const displayValue = selectedItem?.display_name || value || '';

  // Sync selected item with current value from items
  useEffect(() => {
    if (!value) return;
    const match = resourceFetch.items.find(item => item.id === String(value));
    if (match) {
      setSelectedItem(match);
    }
  }, [resourceFetch.items, value]);

  return {
    open,
    setOpen,
    searchQuery,
    setSearchQuery,
    currentPath,
    setCurrentPath,
    displayValue,
    isSupabaseBindingPicker,
    isPostgresBindingPicker,
    hasMissingDependency,
    resourceFetch,
    applySupabaseBindingSelection,
    handleSelect,
  };
}
