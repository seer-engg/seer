import { Search } from 'lucide-react';

import { Input } from '@/components/ui/input';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import type { TemplateCategory } from '@/types/templates';

export interface CategoryFilterProps {
  categories: TemplateCategory[];
  selectedCategory: string | null;
  searchQuery: string;
  onCategoryChange: (category: string | null) => void;
  onSearchChange: (query: string) => void;
  isLoading?: boolean;
}

/**
 * Filter component for the template gallery.
 * Provides search input and category tabs for filtering templates.
 */
export function CategoryFilter({
  categories,
  selectedCategory,
  searchQuery,
  onCategoryChange,
  onSearchChange,
  isLoading,
}: CategoryFilterProps) {
  return (
    <div className="space-y-4">
      {/* Search input */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
        <Input
          placeholder="Search templates..."
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
          className="pl-9"
        />
      </div>

      {/* Category tabs */}
      <Tabs
        value={selectedCategory ?? 'all'}
        onValueChange={(value) => onCategoryChange(value === 'all' ? null : value)}
      >
        <TabsList className="w-full h-auto flex-wrap justify-start gap-1 bg-transparent p-0">
          <TabsTrigger
            value="all"
            disabled={isLoading}
            className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
          >
            All
          </TabsTrigger>
          {(categories ?? []).map((category) => (
            <TabsTrigger
              key={category.slug}
              value={category.slug}
              disabled={isLoading}
              className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
            >
              {category.name}
              {category.template_count > 0 && (
                <span className="ml-1.5 text-xs opacity-70">({category.template_count})</span>
              )}
            </TabsTrigger>
          ))}
        </TabsList>
      </Tabs>
    </div>
  );
}
