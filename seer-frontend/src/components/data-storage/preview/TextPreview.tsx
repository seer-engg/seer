import { Loader2, AlertCircle } from "lucide-react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark, oneLight } from "react-syntax-highlighter/dist/esm/styles/prism";
import { ScrollArea } from "@/components/ui/scroll-area";
import { getFileExtension } from "@/lib/files-api";
import { useTextContent } from "./hooks/useTextContent";

export interface TextPreviewProps {
  url: string;
  filename: string;
  mimeType: string;
}

const MAX_SIZE_BYTES = 1024 * 1024; // 1MB limit for text preview

// Map file extensions to syntax highlighter languages
function getLanguageFromFilename(filename: string, mimeType: string): string {
  const ext = getFileExtension(filename).toLowerCase();

  const extensionMap: Record<string, string> = {
    js: "javascript",
    jsx: "jsx",
    ts: "typescript",
    tsx: "tsx",
    py: "python",
    rb: "ruby",
    go: "go",
    rs: "rust",
    java: "java",
    c: "c",
    cpp: "cpp",
    h: "c",
    hpp: "cpp",
    cs: "csharp",
    php: "php",
    swift: "swift",
    kt: "kotlin",
    scala: "scala",
    r: "r",
    sql: "sql",
    sh: "bash",
    bash: "bash",
    zsh: "bash",
    ps1: "powershell",
    yaml: "yaml",
    yml: "yaml",
    json: "json",
    xml: "xml",
    html: "html",
    htm: "html",
    css: "css",
    scss: "scss",
    sass: "sass",
    less: "less",
    md: "markdown",
    markdown: "markdown",
    txt: "text",
    csv: "csv",
    dockerfile: "dockerfile",
    makefile: "makefile",
    toml: "toml",
    ini: "ini",
    conf: "ini",
    cfg: "ini",
  };

  if (extensionMap[ext]) return extensionMap[ext];

  // Fallback based on MIME type
  if (mimeType === "text/markdown") return "markdown";
  if (mimeType === "text/csv") return "csv";
  if (mimeType === "application/json") return "json";
  if (mimeType === "application/xml" || mimeType === "text/xml") return "xml";

  return "text";
}

export function TextPreview({ url, filename, mimeType }: TextPreviewProps) {
  const { content, isLoading, error, isTruncated } = useTextContent(url, MAX_SIZE_BYTES);
  const isDark = document.documentElement.classList.contains("dark");
  const language = getLanguageFromFilename(filename, mimeType);

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

  if (content === null || content.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-center">
        <p className="text-sm text-muted-foreground">File is empty</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {isTruncated && (
        <div className="flex items-center gap-2 px-4 py-2 bg-amber-500/10 border-b border-amber-500/20 text-amber-600 dark:text-amber-400">
          <AlertCircle className="h-4 w-4 flex-shrink-0" />
          <p className="text-xs">File is larger than 1MB. Showing first 1MB only.</p>
        </div>
      )}
      <ScrollArea className="flex-1">
        <SyntaxHighlighter
          language={language}
          style={isDark ? oneDark : oneLight}
          showLineNumbers
          wrapLines
          customStyle={{ margin: 0, borderRadius: 0, fontSize: "13px", minHeight: "100%" }}
          lineNumberStyle={{
            minWidth: "3em",
            paddingRight: "1em",
            color: isDark ? "#636d83" : "#9ca3af",
            userSelect: "none",
          }}
        >
          {content}
        </SyntaxHighlighter>
      </ScrollArea>
    </div>
  );
}
