import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { RefreshCw, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useAppUpdateDetector } from "@/hooks/utility/useAppUpdateDetector";

export function AppUpdateBanner() {
  const { updateAvailable } = useAppUpdateDetector();
  const [dismissed, setDismissed] = useState(false);

  const visible = updateAvailable && !dismissed;

  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 16 }}
          transition={{ duration: 0.25, ease: "easeOut" }}
          className="fixed bottom-6 right-6 z-50 flex items-center gap-3 rounded-lg border border-amber-500/20 bg-amber-500/10 px-4 py-3 text-sm text-amber-600 shadow-lg backdrop-blur-sm dark:text-amber-400"
          role="status"
          aria-live="polite"
        >
          <RefreshCw className="h-4 w-4 shrink-0" />
          <span>A new version is available.</span>
          <Button
            variant="outline"
            size="sm"
            className="h-7 border-amber-500/30 bg-transparent px-2 text-xs text-amber-600 hover:bg-amber-500/10 dark:text-amber-400"
            onClick={() => window.location.reload()}
          >
            Reload
          </Button>
          <button
            aria-label="Dismiss update notification"
            className="ml-1 rounded p-0.5 text-amber-500 opacity-60 transition-opacity hover:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-500"
            onClick={() => setDismissed(true)}
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
