import { TimelineHeader, WaterfallBar } from './TimelineComponents';
import type { WorkflowNodeTrace, RunHistoryEntry, TriggerInfo } from './types';
import {
  calculateNodeTiming,
  calculateTotalDuration,
  generateTimeMarkers,
} from './timing-utils';

interface WaterfallTimelineProps {
  nodes: WorkflowNodeTrace[];
  startTime?: string | null;
  endTime?: string | null;
  executionGraph?: RunHistoryEntry['execution_graph'];
  /** Controlled: index into entry.nodes of the currently selected node (-1 = synthetic trigger) */
  selectedNodeIndex?: number | null;
  /** Controlled: Callback when a node is selected/deselected */
  onSelectNode?: (index: number | null) => void;
  triggerInfo?: TriggerInfo;
}

export function WaterfallTimeline({
  nodes,
  startTime,
  endTime,
  executionGraph,
  selectedNodeIndex,
  onSelectNode,
  triggerInfo,
}: WaterfallTimelineProps) {
  const nodesWithTiming = calculateNodeTiming(nodes, startTime);
  const totalDuration = calculateTotalDuration(startTime, endTime);
  const timeMarkers = generateTimeMarkers(totalDuration, 6);

  const hasSynthetic = nodesWithTiming[0]?.is_synthetic ?? false;

  if (!startTime || totalDuration === 0 || nodes.length === 0) {
    return (
      <div className="py-8 text-center">
        <p className="text-sm text-muted-foreground">
          Timing data not available for waterfall view
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-3 pr-2">
      <h3 className="text-sm font-medium">
        Execution Timeline ({nodes.length} nodes)
      </h3>

      <TimelineHeader timeMarkers={timeMarkers} totalDuration={totalDuration} />

      <div className="space-y-1.5">
        {nodesWithTiming.map((node, timelineIndex) => {
          const entryNodeIndex = node.is_synthetic ? -1 : (hasSynthetic ? timelineIndex - 1 : timelineIndex);
          return (
            <WaterfallBar
              key={`${node.node_id}-${timelineIndex}`}
              node={node}
              index={timelineIndex}
              totalDuration={totalDuration}
              timeMarkers={timeMarkers}
              isExpanded={selectedNodeIndex === entryNodeIndex}
              onBarClick={() => onSelectNode?.(entryNodeIndex)}
              executionGraph={executionGraph}
              triggerInfo={node.is_synthetic && node.node_type === 'trigger' ? triggerInfo : undefined}
            />
          );
        })}
      </div>
    </div>
  );
}
