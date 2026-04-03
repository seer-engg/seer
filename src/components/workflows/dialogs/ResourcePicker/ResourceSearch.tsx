import { Search } from 'lucide-react';
import { Input } from '@/components/ui/input';

export interface ResourceSearchProps {
  value: string;
  onChange: (value: string) => void;
  supportsSearch: boolean;
  disabled?: boolean;
}

export function ResourceSearch({
  value,
  onChange,
  supportsSearch,
  disabled = false,
}: ResourceSearchProps) {
  if (!supportsSearch) {
    return null;
  }

  return (
    <div className="relative px-1">
      <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
      <Input
        placeholder="Search..."
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="pl-9"
        disabled={disabled}
      />
    </div>
  );
}
