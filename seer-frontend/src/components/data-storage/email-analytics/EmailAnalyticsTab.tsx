import { useState } from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useEmailSummary } from "@/hooks/useEmailAnalytics";
import { EmailSummaryCards } from "./EmailSummaryCards";
import { EmailEventsTable } from "./EmailEventsTable";

const EMAIL_TYPES = [
  { value: "", label: "All types" },
  { value: "invitation", label: "Invitation" },
  { value: "approval", label: "Approval" },
  { value: "member_joined", label: "Member joined" },
  { value: "workflow", label: "Workflow" },
  { value: "hitl", label: "HITL" },
];

const EVENT_TYPES = [
  { value: "", label: "All events" },
  { value: "sent", label: "Sent" },
  { value: "opened", label: "Opened" },
  { value: "clicked", label: "Clicked" },
];

export function EmailAnalyticsTab() {
  const [emailType, setEmailType] = useState("");
  const [eventType, setEventType] = useState("");
  const { data: summary, isLoading: summaryLoading } = useEmailSummary();

  return (
    <div className="space-y-6">
      {/* Summary cards */}
      <EmailSummaryCards
        totalSent={summary?.total_sent}
        totalOpened={summary?.total_opened}
        totalClicked={summary?.total_clicked}
        openRate={summary?.open_rate}
        clickRate={summary?.click_rate}
        isLoading={summaryLoading}
      />

      {/* Filters */}
      <div className="flex items-center gap-3">
        <Select value={emailType} onValueChange={setEmailType}>
          <SelectTrigger className="w-[160px]">
            <SelectValue placeholder="All types" />
          </SelectTrigger>
          <SelectContent>
            {EMAIL_TYPES.map((t) => (
              <SelectItem key={t.value} value={t.value || "_all"}>
                {t.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Select value={eventType} onValueChange={setEventType}>
          <SelectTrigger className="w-[140px]">
            <SelectValue placeholder="All events" />
          </SelectTrigger>
          <SelectContent>
            {EVENT_TYPES.map((t) => (
              <SelectItem key={t.value} value={t.value || "_all"}>
                {t.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Events table */}
      <EmailEventsTable
        emailType={emailType === "_all" ? undefined : emailType || undefined}
        eventType={eventType === "_all" ? undefined : eventType || undefined}
      />
    </div>
  );
}
