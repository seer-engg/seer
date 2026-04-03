import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { MessageSquare, Unlink } from "lucide-react";
import { useToast } from "@/hooks/utility/use-toast";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  type WhatsAppLinkStatus,
  getWhatsAppLinkStatus,
  linkWhatsAppPhone,
  verifyWhatsAppPhone,
  unlinkWhatsApp,
} from "@/lib/whatsapp-api";

type CardState = "unlinked" | "code_sent" | "verified";

const QUERY_KEY = ["whatsapp", "link-status"];

function CardIcon() {
  return (
    <div className="w-10 h-10 rounded-lg bg-seer/10 flex items-center justify-center">
      <MessageSquare className="h-5 w-5 text-seer" />
    </div>
  );
}

function VerifiedContent({ phone, status, unlinkMutation }: {
  phone: string;
  status?: WhatsAppLinkStatus;
  unlinkMutation: ReturnType<typeof useMutation<void, Error>>;
}) {
  return (
    <div className="flex items-center justify-between">
      <p className="text-sm text-muted-foreground">
        Linked to <span className="font-medium text-foreground">{phone || status?.phone_number}</span>
      </p>
      <Button variant="outline" size="sm" onClick={() => unlinkMutation.mutate()} disabled={unlinkMutation.isPending}>
        <Unlink className="h-4 w-4 mr-2" />
        {unlinkMutation.isPending ? "Unlinking..." : "Unlink"}
      </Button>
    </div>
  );
}

function CodeSentContent({ code, setCode, phone, verifyMutation, linkMutation }: {
  code: string;
  setCode: (v: string) => void;
  phone: string;
  verifyMutation: ReturnType<typeof useMutation<unknown, Error, { phoneNumber: string; verificationCode: string }>>;
  linkMutation: ReturnType<typeof useMutation<unknown, Error, string>>;
}) {
  return (
    <div className="space-y-2">
      <Label htmlFor="whatsapp-code">Verification code</Label>
      <div className="flex gap-2">
        <Input id="whatsapp-code" placeholder="Enter code" value={code} onChange={(e) => setCode(e.target.value)} />
        <Button
          onClick={() => verifyMutation.mutate({ phoneNumber: phone, verificationCode: code })}
          disabled={!code || verifyMutation.isPending}
        >
          {verifyMutation.isPending ? "Verifying..." : "Verify"}
        </Button>
      </div>
      <button type="button" className="text-xs text-seer hover:underline" onClick={() => linkMutation.mutate(phone)} disabled={linkMutation.isPending}>
        Resend code
      </button>
    </div>
  );
}

function PhoneInputContent({ phone, setPhone, linkMutation }: {
  phone: string;
  setPhone: (v: string) => void;
  linkMutation: ReturnType<typeof useMutation<unknown, Error, string>>;
}) {
  return (
    <div className="space-y-2">
      <Label htmlFor="whatsapp-phone">Phone number</Label>
      <div className="flex gap-2">
        <Input id="whatsapp-phone" placeholder="+1 555 123 4567" value={phone} onChange={(e) => setPhone(e.target.value)} />
        <Button onClick={() => linkMutation.mutate(phone)} disabled={!phone || linkMutation.isPending}>
          {linkMutation.isPending ? "Sending..." : "Link"}
        </Button>
      </div>
    </div>
  );
}

export function WhatsAppCard() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [phone, setPhone] = useState("");
  const [code, setCode] = useState("");
  const [cardState, setCardState] = useState<CardState>("unlinked");

  const { data: status, isLoading } = useQuery({ queryKey: QUERY_KEY, queryFn: getWhatsAppLinkStatus });

  useEffect(() => {
    if (status?.linked) {
      setCardState("verified");
      if (status.phone_number) setPhone(status.phone_number);
    }
  }, [status]);

  const effectiveState = status?.linked ? "verified" : cardState;

  const linkMutation = useMutation({
    mutationFn: (phoneNumber: string) => linkWhatsAppPhone(phoneNumber),
    onSuccess: () => { setCardState("code_sent"); toast({ title: "Verification code sent", description: `Code sent to ${phone}` }); },
    onError: (error: Error) => { toast({ title: "Failed to send code", description: error.message, variant: "destructive" }); },
  });

  const verifyMutation = useMutation({
    mutationFn: ({ phoneNumber, verificationCode }: { phoneNumber: string; verificationCode: string }) =>
      verifyWhatsAppPhone(phoneNumber, verificationCode),
    onSuccess: () => { queryClient.invalidateQueries({ queryKey: QUERY_KEY }); setCode(""); toast({ title: "WhatsApp linked", description: "Your phone is now connected" }); },
    onError: (error: Error) => { toast({ title: "Verification failed", description: error.message, variant: "destructive" }); },
  });

  const unlinkMutation = useMutation({
    mutationFn: unlinkWhatsApp,
    onSuccess: () => { queryClient.invalidateQueries({ queryKey: QUERY_KEY }); setPhone(""); setCode(""); setCardState("unlinked"); toast({ title: "WhatsApp unlinked" }); },
    onError: (error: Error) => { toast({ title: "Failed to unlink", description: error.message, variant: "destructive" }); },
  });

  if (isLoading) {
    return <Card><CardHeader><div className="flex items-center gap-3"><CardIcon /><div><CardTitle className="text-base">WhatsApp</CardTitle><CardDescription>Loading...</CardDescription></div></div></CardHeader></Card>;
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-3">
          <CardIcon />
          <div className="flex-1">
            <div className="flex items-center gap-2">
              <CardTitle className="text-base">WhatsApp</CardTitle>
              {effectiveState === "verified" && (
                <span className="inline-flex items-center rounded-full bg-green-100 px-2 py-0.5 text-xs font-medium text-green-700 dark:bg-green-900/30 dark:text-green-400">Connected</span>
              )}
            </div>
            <CardDescription>Chat with Seer via WhatsApp</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {effectiveState === "verified" ? <VerifiedContent phone={phone} status={status} unlinkMutation={unlinkMutation} />
          : effectiveState === "code_sent" ? <CodeSentContent code={code} setCode={setCode} phone={phone} verifyMutation={verifyMutation} linkMutation={linkMutation} />
          : <PhoneInputContent phone={phone} setPhone={setPhone} linkMutation={linkMutation} />}
      </CardContent>
    </Card>
  );
}
