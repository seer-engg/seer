import { subDays, format } from "date-fns";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export interface DateRange {
  start?: string;
  end?: string;
}

interface Preset {
  label: string;
  key: string;
  getRange: () => DateRange;
}

const presets: Preset[] = [
  {
    label: "Billing period",
    key: "billing",
    getRange: () => ({}),
  },
  {
    label: "7 days",
    key: "7d",
    getRange: () => ({
      start: format(subDays(new Date(), 7), "yyyy-MM-dd"),
      end: format(new Date(), "yyyy-MM-dd"),
    }),
  },
  {
    label: "30 days",
    key: "30d",
    getRange: () => ({
      start: format(subDays(new Date(), 30), "yyyy-MM-dd"),
      end: format(new Date(), "yyyy-MM-dd"),
    }),
  },
  {
    label: "90 days",
    key: "90d",
    getRange: () => ({
      start: format(subDays(new Date(), 90), "yyyy-MM-dd"),
      end: format(new Date(), "yyyy-MM-dd"),
    }),
  },
];

interface DateRangeFilterProps {
  activeKey: string;
  onChange: (key: string, range: DateRange) => void;
}

export function DateRangeFilter({ activeKey, onChange }: DateRangeFilterProps) {
  return (
    <div className="flex flex-wrap gap-2">
      {presets.map((preset) => (
        <Button
          key={preset.key}
          variant={activeKey === preset.key ? "default" : "outline"}
          size="sm"
          className={cn("h-8 text-xs")}
          onClick={() => onChange(preset.key, preset.getRange())}
        >
          {preset.label}
        </Button>
      ))}
    </div>
  );
}
