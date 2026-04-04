import { motion } from 'framer-motion';
import { Zap } from 'lucide-react';
import { useTriggerSubscriptions } from '@/hooks/useTriggerSubscriptions';
import { TriggerFilters, TriggerSubscriptionsTable } from '@/components/triggers';

export default function Triggers() {
  const {
    subscriptions,
    isLoading,
    filters,
    updateFilter,
    clearFilters,
    uniqueTriggerKeys,
    uniqueWorkflows,
    toggleEnabled,
    isToggling,
  } = useTriggerSubscriptions();

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
            <div className="p-2 rounded-lg bg-primary/10">
              <Zap className="h-5 w-5 text-primary" />
            </div>
            <h1 className="text-2xl font-semibold tracking-tight">Triggers</h1>
          </div>
          <p className="text-muted-foreground text-sm">
            View and manage trigger subscriptions across all your workflows
          </p>
        </div>
      </div>

      {/* Content */}
      <div className="container py-6 space-y-6 flex-1">
        {/* Filters */}
        <TriggerFilters
          filters={filters}
          onFilterChange={updateFilter}
          onClearFilters={clearFilters}
          triggerKeys={uniqueTriggerKeys}
          workflows={uniqueWorkflows}
        />

        {/* Table */}
        <TriggerSubscriptionsTable
          subscriptions={subscriptions}
          isLoading={isLoading}
          onToggleEnabled={toggleEnabled}
          isToggling={isToggling}
        />
      </div>
    </motion.div>
  );
}
