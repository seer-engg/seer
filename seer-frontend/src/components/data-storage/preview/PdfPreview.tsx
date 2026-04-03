import { useState } from "react";
import { Loader2, AlertCircle } from "lucide-react";

export interface PdfPreviewProps {
  url: string;
  filename: string;
}

/**
 * PDF Preview using native browser rendering via <object> tag.
 * This avoids CORS issues that occur with fetch-based libraries like react-pdf.
 */
export function PdfPreview({ url, filename }: PdfPreviewProps) {
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);

  const handleLoad = () => {
    setIsLoading(false);
  };

  const handleError = () => {
    setIsLoading(false);
    setHasError(true);
  };

  if (hasError) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-center">
        <AlertCircle className="h-8 w-8 text-destructive mb-2" />
        <p className="text-sm text-destructive mb-1">Failed to load PDF</p>
        <p className="text-xs text-muted-foreground">
          Your browser may not support inline PDF viewing. Try downloading the file instead.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full relative">
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-background/80 z-10">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      )}
      <object
        data={url}
        type="application/pdf"
        className="flex-1 w-full min-h-[500px]"
        title={filename}
        onLoad={handleLoad}
        onError={handleError}
      >
        {/* Fallback for browsers that don't support object tag for PDFs */}
        <iframe
          src={url}
          className="flex-1 w-full min-h-[500px] border-0"
          title={filename}
          onLoad={handleLoad}
          onError={handleError}
        />
      </object>
    </div>
  );
}
