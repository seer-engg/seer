/**
 * BrowserHITLPanel Component
 *
 * Displays a live browser viewer alongside a response form for browser_hitl interrupts.
 * The user can interact with the browser directly (e.g., solve a CAPTCHA) and then
 * submit a text response to resume the browser agent.
 */
import { useState, useCallback } from 'react';
import { Monitor, Loader2, AlertCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { BrowserViewer } from '@/components/browser/BrowserViewer';
import type { HitlInterruptData, HitlResumePayload } from './types';

interface BrowserHITLPanelProps {
  interruptData: HitlInterruptData;
  onSubmit: (payload: HitlResumePayload) => void;
  isSubmitting?: boolean;
  error?: Error | null;
}

export function BrowserHITLPanel({
  interruptData,
  onSubmit,
  isSubmitting = false,
  error,
}: BrowserHITLPanelProps) {
  const [response, setResponse] = useState('');

  const handleSubmit = useCallback(() => {
    onSubmit({
      responses: { response: response || 'Done' },
    });
  }, [response, onSubmit]);

  return (
    <div className="flex flex-col lg:flex-row gap-4 h-[calc(100vh-8rem)]">
      {/* Browser viewer — main area */}
      <div className="flex-1 min-h-[400px] lg:min-h-0 border rounded-lg overflow-hidden bg-muted/30">
        <BrowserViewer
          sessionId={interruptData.session_id ?? null}
          showToolbar
          className="w-full h-full"
        />
      </div>

      {/* Response panel — sidebar */}
      <div className="w-full lg:w-80 shrink-0">
        <Card className="border-amber-500/50 bg-amber-500/5">
          <CardHeader className="pb-3">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-full bg-amber-500/20 flex items-center justify-center">
                <Monitor className="w-4 h-4 text-amber-500" />
              </div>
              <div>
                <CardTitle className="text-base">{interruptData.title}</CardTitle>
                {interruptData.description && (
                  <CardDescription className="text-sm mt-1">
                    {interruptData.description}
                  </CardDescription>
                )}
              </div>
            </div>
          </CardHeader>

          <CardContent className="space-y-4">
            <p className="text-xs text-muted-foreground">
              Interact with the browser above to resolve the issue (e.g., solve the CAPTCHA),
              then describe what you did and click Submit.
            </p>

            <div className="space-y-2">
              <Label htmlFor="hitl-response" className="text-sm font-medium">
                Your response
              </Label>
              <Textarea
                id="hitl-response"
                value={response}
                onChange={(e) => setResponse(e.target.value)}
                placeholder="e.g., I solved the CAPTCHA"
                rows={3}
              />
            </div>

            {error && (
              <div className="flex items-center gap-2 text-destructive text-sm">
                <AlertCircle className="w-4 h-4 shrink-0" />
                <span>{error.message || 'Failed to submit response'}</span>
              </div>
            )}

            <Button
              onClick={handleSubmit}
              disabled={isSubmitting}
              className="w-full"
            >
              {isSubmitting ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Resuming...
                </>
              ) : (
                'Submit & Resume Agent'
              )}
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
