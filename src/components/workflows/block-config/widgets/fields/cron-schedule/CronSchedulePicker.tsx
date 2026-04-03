import { useState, useEffect, useCallback } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Settings2 } from 'lucide-react';
import { cn } from '@/lib/utils';

import type { CronScheduleState, ScheduleMode, IntervalUnit } from './types';
import { generateCronExpression, parseCronExpression } from './utils';
import { IntervalTab } from './IntervalTab';
import { DailyTab } from './DailyTab';
import { WeeklyTab } from './WeeklyTab';
import { MonthlyTab } from './MonthlyTab';
import { PresetsDropdown } from './PresetsDropdown';
import { AdvancedInput } from './AdvancedInput';
import { CronPreview } from './CronPreview';

interface CronSchedulePickerProps {
  value: string;
  onChange: (expression: string) => void;
  showError?: boolean;
}

export function CronSchedulePicker({ value, onChange, showError }: CronSchedulePickerProps) {
  const [state, setState] = useState<CronScheduleState>(() => parseCronExpression(value));
  const [isAdvanced, setIsAdvanced] = useState(() => parseCronExpression(value).mode === 'advanced');

  useEffect(() => {
    const parsed = parseCronExpression(value);
    setState(parsed);
    setIsAdvanced(parsed.mode === 'advanced');
  }, [value]);

  const updateState = useCallback((updates: Partial<CronScheduleState>) => {
    setState(prev => {
      const newState = { ...prev, ...updates };
      onChange(generateCronExpression(newState));
      return newState;
    });
  }, [onChange]);

  const handlePresetChange = (preset: string) => {
    if (preset === 'custom') return;
    const parsed = parseCronExpression(preset);
    setState(parsed);
    setIsAdvanced(parsed.mode === 'advanced');
    onChange(preset);
  };

  const handleModeChange = (mode: string) => updateState({ mode: mode as ScheduleMode });

  const toggleAdvanced = () => {
    if (isAdvanced) {
      const parsed = parseCronExpression(state.rawExpression);
      if (parsed.mode !== 'advanced') {
        setState(parsed);
        setIsAdvanced(false);
      }
    } else {
      const expression = generateCronExpression(state);
      setState(prev => ({ ...prev, mode: 'advanced', rawExpression: expression }));
      setIsAdvanced(true);
    }
  };

  const handleRawChange = (raw: string) => {
    setState(prev => ({ ...prev, rawExpression: raw }));
    onChange(raw);
  };

  const currentExpression = value || generateCronExpression(state);

  return (
    <div className={cn('space-y-3 rounded-md border p-3', showError && 'border-destructive')}>
      <PresetsDropdown currentExpression={value} onSelect={handlePresetChange} />

      <div className="flex items-center justify-between">
        <Button variant="ghost" size="sm" onClick={toggleAdvanced} className="h-7 px-2 text-xs text-muted-foreground hover:text-foreground">
          <Settings2 className="h-3.5 w-3.5 mr-1" />
          {isAdvanced ? 'Simple mode' : 'Advanced mode'}
        </Button>
      </div>

      {isAdvanced ? (
        <AdvancedInput value={state.rawExpression} onChange={handleRawChange} />
      ) : (
        <Tabs value={state.mode} onValueChange={handleModeChange}>
          <TabsList className="h-8 w-full">
            <TabsTrigger value="interval" className="flex-1 text-xs h-7">Interval</TabsTrigger>
            <TabsTrigger value="daily" className="flex-1 text-xs h-7">Daily</TabsTrigger>
            <TabsTrigger value="weekly" className="flex-1 text-xs h-7">Weekly</TabsTrigger>
            <TabsTrigger value="monthly" className="flex-1 text-xs h-7">Monthly</TabsTrigger>
          </TabsList>
          <TabsContent value="interval" className="mt-3">
            <IntervalTab value={state.intervalValue} unit={state.intervalUnit} onValueChange={(v) => updateState({ intervalValue: v })} onUnitChange={(u) => updateState({ intervalUnit: u as IntervalUnit })} />
          </TabsContent>
          <TabsContent value="daily" className="mt-3">
            <DailyTab hour={state.hour} minute={state.minute} onHourChange={(h) => updateState({ hour: h })} onMinuteChange={(m) => updateState({ minute: m })} />
          </TabsContent>
          <TabsContent value="weekly" className="mt-3">
            <WeeklyTab days={state.days} hour={state.hour} minute={state.minute} onDaysChange={(d) => updateState({ days: d })} onHourChange={(h) => updateState({ hour: h })} onMinuteChange={(m) => updateState({ minute: m })} />
          </TabsContent>
          <TabsContent value="monthly" className="mt-3">
            <MonthlyTab dayOfMonth={state.dayOfMonth} hour={state.hour} minute={state.minute} onDayOfMonthChange={(d) => updateState({ dayOfMonth: d })} onHourChange={(h) => updateState({ hour: h })} onMinuteChange={(m) => updateState({ minute: m })} />
          </TabsContent>
        </Tabs>
      )}

      <CronPreview expression={currentExpression} />
    </div>
  );
}
