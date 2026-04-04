import { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Video } from 'lucide-react';
import { useMeetings } from '@/hooks/useMeetings';
import { MeetingsTable, MeetingDetailDialog } from '@/components/meetings';

export default function Meetings() {
  const { meetings, isLoading } = useMeetings();
  const [selectedBotId, setSelectedBotId] = useState<string | null>(null);

  const selectedMeeting = selectedBotId
    ? meetings.find((m) => m.bot_id === selectedBotId) ?? null
    : null;

  const handleSelect = useCallback((botId: string) => {
    setSelectedBotId(botId);
  }, []);

  const handleDialogChange = useCallback((open: boolean) => {
    if (!open) setSelectedBotId(null);
  }, []);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.3, ease: 'easeInOut' }}
      className="flex flex-col min-h-screen"
    >
      {/* Header */}
      <div className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container py-6">
          <div className="flex items-center gap-3 mb-1">
            <div className="p-2 rounded-lg bg-violet-500/10">
              <Video className="h-5 w-5 text-violet-500" />
            </div>
            <h1 className="text-2xl font-semibold tracking-tight">Meetings</h1>
          </div>
          <p className="text-muted-foreground text-sm">
            View recorded meetings, transcripts, and recordings from your organization
          </p>
        </div>
      </div>

      {/* Content */}
      <div className="container py-6 flex-1">
        <MeetingsTable
          meetings={meetings}
          isLoading={isLoading}
          onSelect={handleSelect}
        />
      </div>

      {/* Detail Dialog */}
      <MeetingDetailDialog
        meeting={selectedMeeting}
        open={!!selectedBotId}
        onOpenChange={handleDialogChange}
      />
    </motion.div>
  );
}
