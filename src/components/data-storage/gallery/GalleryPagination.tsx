import { ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";

export function GalleryPagination({
  currentPage,
  hasNext,
  hasPrev,
  isFetching,
  onPrev,
  onNext,
}: {
  currentPage: number;
  hasNext: boolean;
  hasPrev: boolean;
  isFetching: boolean;
  onPrev: () => void;
  onNext: () => void;
}) {
  if (!hasNext && !hasPrev) return null;

  return (
    <div className="flex items-center justify-between pt-2 border-t">
      <Button variant="outline" size="sm" onClick={onPrev} disabled={!hasPrev || isFetching} className="h-8">
        <ChevronLeft className="h-4 w-4 mr-1" />
        Previous
      </Button>
      <span className="text-xs text-muted-foreground">Page {currentPage}</span>
      <Button variant="outline" size="sm" onClick={onNext} disabled={!hasNext || isFetching} className="h-8">
        Next
        <ChevronRight className="h-4 w-4 ml-1" />
      </Button>
    </div>
  );
}
