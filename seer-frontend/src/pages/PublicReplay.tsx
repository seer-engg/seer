import { useParams } from 'react-router-dom';
import { SessionReplay } from '@/components/browser/SessionReplay';

export default function PublicReplay() {
  const { recordingId } = useParams<{ recordingId: string }>();

  if (!recordingId) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background text-foreground">
        <p className="text-muted-foreground">Invalid recording link.</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-background p-4">
      <div className="w-full max-w-5xl">
        <SessionReplay recordingId={recordingId} autoPlay publicMode />
      </div>
    </div>
  );
}
