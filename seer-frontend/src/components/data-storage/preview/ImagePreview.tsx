import { useState, useCallback } from "react";
import { ZoomIn, ZoomOut, RotateCcw, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export interface ImagePreviewProps {
  url: string;
  filename: string;
}

const ZOOM_LEVELS = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3];
const DEFAULT_ZOOM_INDEX = 3; // 100%

function ZoomControls({
  zoom,
  zoomIndex,
  onZoomIn,
  onZoomOut,
  onReset,
}: {
  zoom: number;
  zoomIndex: number;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onReset: () => void;
}) {
  return (
    <div className="flex items-center justify-center gap-2 py-2 border-b bg-muted/30">
      <Button
        variant="ghost"
        size="sm"
        onClick={onZoomOut}
        disabled={zoomIndex === 0}
        className="h-8 w-8 p-0"
      >
        <ZoomOut className="h-4 w-4" />
      </Button>
      <span className="text-sm text-muted-foreground w-16 text-center">
        {Math.round(zoom * 100)}%
      </span>
      <Button
        variant="ghost"
        size="sm"
        onClick={onZoomIn}
        disabled={zoomIndex === ZOOM_LEVELS.length - 1}
        className="h-8 w-8 p-0"
      >
        <ZoomIn className="h-4 w-4" />
      </Button>
      <div className="w-px h-4 bg-border mx-1" />
      <Button
        variant="ghost"
        size="sm"
        onClick={onReset}
        disabled={zoomIndex === DEFAULT_ZOOM_INDEX}
        className="h-8 px-2"
      >
        <RotateCcw className="h-3.5 w-3.5 mr-1" />
        Reset
      </Button>
    </div>
  );
}

export function ImagePreview({ url, filename }: ImagePreviewProps) {
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);
  const [zoomIndex, setZoomIndex] = useState(DEFAULT_ZOOM_INDEX);
  const zoom = ZOOM_LEVELS[zoomIndex];

  const handleZoomIn = useCallback(() => {
    setZoomIndex((prev) => Math.min(prev + 1, ZOOM_LEVELS.length - 1));
  }, []);

  const handleZoomOut = useCallback(() => {
    setZoomIndex((prev) => Math.max(prev - 1, 0));
  }, []);

  const handleReset = useCallback(() => setZoomIndex(DEFAULT_ZOOM_INDEX), []);

  const handleWheel = useCallback((e: React.WheelEvent) => {
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      if (e.deltaY < 0) {
        setZoomIndex((prev) => Math.min(prev + 1, ZOOM_LEVELS.length - 1));
      } else {
        setZoomIndex((prev) => Math.max(prev - 1, 0));
      }
    }
  }, []);

  const mightHaveTransparency = /\.(png|gif|webp|svg)$/i.test(filename);

  if (hasError) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-center">
        <p className="text-sm text-destructive mb-2">Failed to load image</p>
        <p className="text-xs text-muted-foreground">
          The image could not be loaded. Try downloading it instead.
        </p>
      </div>
    );
  }

  const transparencyStyle = mightHaveTransparency
    ? {
        backgroundImage:
          "linear-gradient(45deg, hsl(var(--muted)) 25%, transparent 25%), linear-gradient(-45deg, hsl(var(--muted)) 25%, transparent 25%), linear-gradient(45deg, transparent 75%, hsl(var(--muted)) 75%), linear-gradient(-45deg, transparent 75%, hsl(var(--muted)) 75%)",
      }
    : undefined;

  return (
    <div className="flex flex-col h-full">
      <ZoomControls
        zoom={zoom}
        zoomIndex={zoomIndex}
        onZoomIn={handleZoomIn}
        onZoomOut={handleZoomOut}
        onReset={handleReset}
      />
      <div
        className="flex-1 overflow-auto flex items-center justify-center p-4"
        onWheel={handleWheel}
      >
        <div
          className={cn(
            "relative",
            mightHaveTransparency && "bg-[length:20px_20px] bg-[position:0_0,10px_10px]"
          )}
          style={transparencyStyle}
        >
          {isLoading && (
            <div className="absolute inset-0 flex items-center justify-center">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          )}
          <img
            src={url}
            alt={filename}
            className={cn("max-w-none transition-transform duration-200", isLoading && "opacity-0")}
            style={{ transform: `scale(${zoom})`, transformOrigin: "center center" }}
            onLoad={() => setIsLoading(false)}
            onError={() => {
              setIsLoading(false);
              setHasError(true);
            }}
          />
        </div>
      </div>
      <div className="text-center py-2 border-t">
        <p className="text-xs text-muted-foreground">Ctrl/Cmd + scroll to zoom</p>
      </div>
    </div>
  );
}
