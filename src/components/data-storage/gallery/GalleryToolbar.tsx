import {
  Search,
  ArrowUpDown,
  LayoutGrid,
  List,
  Upload,
} from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

const SORT_OPTIONS = [
  { value: "created_at:desc", label: "Newest first" },
  { value: "created_at:asc", label: "Oldest first" },
  { value: "size_bytes:desc", label: "Largest first" },
  { value: "size_bytes:asc", label: "Smallest first" },
  { value: "filename:asc", label: "Name A-Z" },
  { value: "filename:desc", label: "Name Z-A" },
];

const TYPE_FILTERS = [
  { value: "all", label: "All types" },
  { value: "application/pdf", label: "PDF" },
  { value: "image/*", label: "Images" },
  { value: "text/*", label: "Text" },
  { value: "application/json", label: "JSON" },
];

type ViewMode = "grid" | "list";

export function GalleryToolbar({
  searchQuery,
  typeFilter,
  sortOption,
  viewMode,
  onSearchChange,
  onTypeChange,
  onSortChange,
  onViewModeChange,
  onToggleUpload,
}: {
  searchQuery: string;
  typeFilter: string;
  sortOption: string;
  viewMode: ViewMode;
  onSearchChange: (v: string) => void;
  onTypeChange: (v: string) => void;
  onSortChange: (v: string) => void;
  onViewModeChange: (mode: ViewMode) => void;
  onToggleUpload: () => void;
}) {
  return (
    <div className="flex items-center gap-2 flex-wrap">
      <div className="relative flex-1 min-w-[200px]">
        <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
        <Input placeholder="Search files..." value={searchQuery} onChange={(e) => onSearchChange(e.target.value)} className="pl-8 h-9" />
      </div>
      <Select value={typeFilter} onValueChange={onTypeChange}>
        <SelectTrigger className="w-[130px] h-9"><SelectValue placeholder="Type" /></SelectTrigger>
        <SelectContent>{TYPE_FILTERS.map((opt) => (<SelectItem key={opt.value} value={opt.value}>{opt.label}</SelectItem>))}</SelectContent>
      </Select>
      <Select value={sortOption} onValueChange={onSortChange}>
        <SelectTrigger className="w-[140px] h-9"><ArrowUpDown className="h-3.5 w-3.5 mr-1.5" /><SelectValue placeholder="Sort" /></SelectTrigger>
        <SelectContent>{SORT_OPTIONS.map((opt) => (<SelectItem key={opt.value} value={opt.value}>{opt.label}</SelectItem>))}</SelectContent>
      </Select>
      <div className="flex items-center border rounded-md">
        <Button variant={viewMode === "grid" ? "secondary" : "ghost"} size="sm" className="h-9 w-9 p-0 rounded-r-none" onClick={() => onViewModeChange("grid")} title="Grid view"><LayoutGrid className="h-4 w-4" /></Button>
        <Button variant={viewMode === "list" ? "secondary" : "ghost"} size="sm" className="h-9 w-9 p-0 rounded-l-none" onClick={() => onViewModeChange("list")} title="List view"><List className="h-4 w-4" /></Button>
      </div>
      <Button size="sm" className="h-9" onClick={onToggleUpload}><Upload className="h-4 w-4 mr-1.5" />Upload</Button>
    </div>
  );
}
