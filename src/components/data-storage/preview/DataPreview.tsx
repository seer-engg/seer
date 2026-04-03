import { useMemo } from "react";
import { Loader2, AlertCircle } from "lucide-react";
import { JsonViewer } from "@textea/json-viewer";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useTextContent } from "./hooks/useTextContent";

export interface DataPreviewProps {
  url: string;
  filename: string;
  mimeType: string;
}

const MAX_SIZE_BYTES = 5 * 1024 * 1024; // 5MB limit for data preview

function parseCSV(text: string): { headers: string[]; rows: string[][] } {
  const lines = text.split(/\r?\n/).filter((line) => line.trim());
  if (lines.length === 0) return { headers: [], rows: [] };

  const parseLine = (line: string): string[] => {
    const result: string[] = [];
    let current = "";
    let inQuotes = false;

    for (let i = 0; i < line.length; i++) {
      const char = line[i];
      if (char === '"') {
        if (inQuotes && line[i + 1] === '"') {
          current += '"';
          i++;
        } else {
          inQuotes = !inQuotes;
        }
      } else if (char === "," && !inQuotes) {
        result.push(current.trim());
        current = "";
      } else {
        current += char;
      }
    }
    result.push(current.trim());
    return result;
  };

  const headers = parseLine(lines[0]);
  const rows = lines.slice(1).map(parseLine);
  return { headers, rows };
}

function CSVTable({ headers, rows }: { headers: string[]; rows: string[][] }) {
  const displayRows = rows.slice(0, 100);
  const hasMoreRows = rows.length > 100;

  if (headers.length === 0) {
    return (
      <div className="flex items-center justify-center py-8 text-muted-foreground">
        No data to display
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <ScrollArea className="flex-1">
        <Table>
          <TableHeader>
            <TableRow>
              {headers.map((header, i) => (
                <TableHead key={i} className="font-medium whitespace-nowrap">
                  {header || `Column ${i + 1}`}
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {displayRows.map((row, rowIndex) => (
              <TableRow key={rowIndex}>
                {headers.map((_, colIndex) => (
                  <TableCell key={colIndex} className="whitespace-nowrap">
                    {row[colIndex] ?? ""}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </ScrollArea>
      {hasMoreRows && (
        <div className="text-center py-2 border-t text-xs text-muted-foreground">
          Showing first 100 of {rows.length} rows
        </div>
      )}
    </div>
  );
}

function JSONTreeView({ data }: { data: unknown }) {
  const isDark = document.documentElement.classList.contains("dark");

  return (
    <ScrollArea className="h-full">
      <div className="p-4">
        <JsonViewer
          value={data}
          theme={isDark ? "dark" : "light"}
          rootName={false}
          defaultInspectDepth={2}
          displayDataTypes={false}
          displaySize={false}
          enableClipboard
          style={{
            backgroundColor: "transparent",
            fontFamily: "ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, monospace",
            fontSize: "13px",
          }}
        />
      </div>
    </ScrollArea>
  );
}

export function DataPreview({ url, filename, mimeType }: DataPreviewProps) {
  const { content, isLoading, error, isTruncated } = useTextContent(url, MAX_SIZE_BYTES);

  const isCSV = mimeType === "text/csv" || filename.toLowerCase().endsWith(".csv");
  const isJSON = mimeType === "application/json" || filename.toLowerCase().endsWith(".json");

  const parsedData = useMemo(() => {
    if (!content) return null;

    if (isJSON) {
      try {
        return { type: "json" as const, data: JSON.parse(content) };
      } catch {
        return { type: "error" as const, message: "Invalid JSON format" };
      }
    }

    if (isCSV) {
      try {
        const { headers, rows } = parseCSV(content);
        return { type: "csv" as const, headers, rows };
      } catch {
        return { type: "error" as const, message: "Invalid CSV format" };
      }
    }

    // For XML and other data formats, try JSON parsing first
    try {
      return { type: "json" as const, data: JSON.parse(content) };
    } catch {
      return { type: "error" as const, message: "Unsupported data format" };
    }
  }, [content, isJSON, isCSV]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-center">
        <AlertCircle className="h-8 w-8 text-destructive mb-2" />
        <p className="text-sm text-destructive mb-1">Failed to load content</p>
        <p className="text-xs text-muted-foreground">{error}</p>
      </div>
    );
  }

  if (!parsedData || parsedData.type === "error") {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-center">
        <AlertCircle className="h-8 w-8 text-amber-500 mb-2" />
        <p className="text-sm text-amber-600 dark:text-amber-400 mb-1">Parse error</p>
        <p className="text-xs text-muted-foreground">
          {parsedData?.message || "Could not parse file"}
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {isTruncated && (
        <div className="flex items-center gap-2 px-4 py-2 bg-amber-500/10 border-b border-amber-500/20 text-amber-600 dark:text-amber-400">
          <AlertCircle className="h-4 w-4 flex-shrink-0" />
          <p className="text-xs">File is larger than 5MB. Preview may be incomplete.</p>
        </div>
      )}
      <div className="flex-1 min-h-0">
        {parsedData.type === "json" && <JSONTreeView data={parsedData.data} />}
        {parsedData.type === "csv" && <CSVTable headers={parsedData.headers} rows={parsedData.rows} />}
      </div>
    </div>
  );
}
