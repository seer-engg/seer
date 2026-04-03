import { useRef } from 'react';
import { Mail, Monitor, ExternalLink } from 'lucide-react';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { useConnectIntegration } from '@/hooks/useConnectIntegration';
import { useToolingData } from '@/hooks/useToolingData';
import type { HITLDeliveryChannel, TemplateAutocompleteControls } from '../types';
import { VariableAutocompleteDropdown } from '../widgets/VariableAutocompleteDropdown';

interface HITLDeliveryChannelsEditorProps {
  channels: HITLDeliveryChannel[];
  onChange: (channels: HITLDeliveryChannel[]) => void;
  templateAutocomplete: TemplateAutocompleteControls;
}

/** Platform channel toggle row */
const PlatformChannelRow = ({
  enabled,
  onToggle,
  disabled,
}: {
  enabled: boolean;
  onToggle: (enabled: boolean) => void;
  disabled: boolean;
}) => (
  <div className="flex items-center justify-between p-3 bg-muted/30 rounded-lg border">
    <div className="flex items-center gap-3">
      <div className="p-2 bg-primary/10 rounded-md">
        <Monitor className="w-4 h-4 text-primary" />
      </div>
      <div>
        <p className="text-sm font-medium">Platform</p>
        <p className="text-xs text-muted-foreground">Show in Seer dashboard</p>
      </div>
    </div>
    <Switch checked={enabled} onCheckedChange={onToggle} disabled={disabled} />
  </div>
);

/** Gmail channel toggle row */
const GmailChannelRow = ({
  enabled,
  onToggle,
  isConnected,
}: {
  enabled: boolean;
  onToggle: (enabled: boolean) => void;
  isConnected: boolean;
}) => (
  <div className="flex items-center justify-between p-3 bg-muted/30 rounded-lg border">
    <div className="flex items-center gap-3">
      <div className="p-2 bg-amber-500/10 rounded-md">
        <Mail className="w-4 h-4 text-amber-600 dark:text-amber-400" />
      </div>
      <div className="flex-1">
        <div className="flex items-center gap-2">
          <p className="text-sm font-medium">Gmail</p>
          <Badge
            variant="secondary"
            className={`h-4 px-1 text-[9px] ${
              isConnected
                ? 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20'
                : 'bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20'
            }`}
          >
            {isConnected ? 'Connected' : 'Not Connected'}
          </Badge>
        </div>
        <p className="text-xs text-muted-foreground">Send email with form link</p>
      </div>
    </div>
    <Switch checked={enabled} onCheckedChange={onToggle} />
  </div>
);

/** Gmail configuration panel */
const GmailConfigPanel = ({
  recipientEmail,
  onEmailChange,
  inputRef,
  templateAutocomplete,
  isEmailFieldActive,
  isConnected,
  onConnect,
}: {
  recipientEmail: string;
  onEmailChange: (value: string) => void;
  inputRef: React.RefObject<HTMLInputElement | null>;
  templateAutocomplete: TemplateAutocompleteControls;
  isEmailFieldActive: boolean;
  isConnected: boolean;
  onConnect: () => void;
}) => (
  <div className="ml-6 space-y-3 p-3 bg-muted/20 rounded-lg border border-dashed">
    {!isConnected && (
      <div className="flex items-center justify-between p-2 bg-amber-500/5 border border-amber-500/20 rounded-md">
        <p className="text-xs text-amber-600 dark:text-amber-400">Connect Gmail to send emails</p>
        <Button variant="outline" size="sm" onClick={onConnect} className="h-7 text-xs gap-1">
          <ExternalLink className="w-3 h-3" />
          Connect
        </Button>
      </div>
    )}
    <div className="space-y-2">
      <Label htmlFor="gmail-recipient" className="text-xs">
        Recipient Email
      </Label>
      <div className="relative">
        <Input
          id="gmail-recipient"
          ref={inputRef}
          placeholder="e.g., ${trigger.sender_email} or user@example.com"
          value={recipientEmail}
          onChange={(e) => onEmailChange(e.target.value)}
          onKeyDown={(e) => templateAutocomplete.handleKeyDown(e)}
          className="h-8 text-sm font-mono"
        />
        {isEmailFieldActive && templateAutocomplete.showAutocomplete && (
          <VariableAutocompleteDropdown
            visible={true}
            items={templateAutocomplete.currentLevelItems}
            selectedIndex={templateAutocomplete.selectedIndex}
            currentPath={templateAutocomplete.currentPath}
            onSelect={(variable) => {
              templateAutocomplete.insertVariable(variable, {
                inputId: 'gmail-recipient-email',
                ref: inputRef,
                value: recipientEmail,
                onChange: onEmailChange,
              });
            }}
            onDrillInto={templateAutocomplete.drillInto}
            onNavigateTo={templateAutocomplete.navigateTo}
          />
        )}
      </div>
      <p className="text-[10px] text-muted-foreground">
        Use {'${variable}'} syntax to dynamically set the recipient
      </p>
    </div>
  </div>
);

/**
 * Editor for HITL delivery channels.
 * Allows users to configure how HITL notifications are delivered.
 */
export function HITLDeliveryChannelsEditor({
  channels,
  onChange,
  templateAutocomplete,
}: HITLDeliveryChannelsEditorProps) {
  const emailInputRef = useRef<HTMLInputElement>(null);
  const connectIntegration = useConnectIntegration();
  const { isIntegrationConnected } = useToolingData();
  const isGmailConnected = isIntegrationConnected('gmail');

  const isChannelEnabled = (type: HITLDeliveryChannel['type']) =>
    channels.some((ch) => ch.type === type);

  const gmailChannel = channels.find((ch) => ch.type === 'gmail');
  const recipientEmail = gmailChannel?.gmail?.recipient_email || '';

  const toggleChannel = (type: HITLDeliveryChannel['type'], enabled: boolean) => {
    if (enabled) {
      const newChannel: HITLDeliveryChannel =
        type === 'gmail' ? { type, gmail: { recipient_email: '' } } : { type };
      onChange([...channels, newChannel]);
    } else {
      const filtered = channels.filter((ch) => ch.type !== type);
      onChange(filtered.length === 0 ? [{ type: 'platform' }] : filtered);
    }
  };

  const updateGmailRecipient = (email: string) => {
    onChange(
      channels.map((ch) => (ch.type === 'gmail' ? { ...ch, gmail: { recipient_email: email } } : ch))
    );
  };

  const handleEmailChange = (value: string) => {
    const input = emailInputRef.current;
    if (input) {
      const cursorPos = input.selectionStart || 0;
      templateAutocomplete.checkForAutocomplete(value, cursorPos, {
        inputId: 'gmail-recipient-email',
        ref: emailInputRef,
        value,
        onChange: updateGmailRecipient,
      });
    }
    updateGmailRecipient(value);
  };

  const handleConnectGmail = async () => {
    const redirectUrl = await connectIntegration('gmail', { toolNames: ['gmail_send_email'] });
    if (redirectUrl) window.location.href = redirectUrl;
  };

  const platformEnabled = isChannelEnabled('platform');
  const gmailEnabled = isChannelEnabled('gmail');
  const isEmailFieldActive =
    templateAutocomplete.autocompleteContext?.inputId === 'gmail-recipient-email';

  return (
    <div className="space-y-4">
      <div>
        <Label className="text-sm font-medium">Delivery Channels</Label>
        <p className="text-xs text-muted-foreground mt-1">
          Choose how to notify the reviewer when this step requires human input.
        </p>
      </div>

      <PlatformChannelRow
        enabled={platformEnabled}
        onToggle={(checked) => toggleChannel('platform', checked)}
        disabled={platformEnabled && channels.length === 1}
      />

      <div className="space-y-3">
        <GmailChannelRow
          enabled={gmailEnabled}
          onToggle={(checked) => toggleChannel('gmail', checked)}
          isConnected={isGmailConnected}
        />

        {gmailEnabled && (
          <GmailConfigPanel
            recipientEmail={recipientEmail}
            onEmailChange={handleEmailChange}
            inputRef={emailInputRef}
            templateAutocomplete={templateAutocomplete}
            isEmailFieldActive={isEmailFieldActive}
            isConnected={isGmailConnected}
            onConnect={handleConnectGmail}
          />
        )}
      </div>
    </div>
  );
}
