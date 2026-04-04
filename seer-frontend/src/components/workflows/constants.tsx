import React from 'react';
import { Bot, Code, GitBranch, Globe, ImageIcon, Monitor, Repeat, Sparkles, UserCheck, Video } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';

import type { BuiltInBlock } from './buildtypes';
import { SHOW_MEETINGS } from '@/lib/feature-flags';

const BLOCK_ICON_COMPONENTS: Record<string, LucideIcon> = {
  code: Code,
  if_else: GitBranch,
  for_loop: Repeat,
  mcp: Globe,
  hitl: UserCheck,
  browser: Monitor,
  image_gen: ImageIcon,
  agent: Bot,
  ea_bot: Video,
};

// Colors match the node icons on the canvas for consistency
const BLOCK_ICON_COLORS: Record<string, string> = {
  code: 'text-slate-500',
  if_else: 'text-orange-500',
  for_loop: 'text-green-500',
  mcp: 'text-cyan-500',
  hitl: 'text-amber-500',
  browser: 'text-blue-500',
  image_gen: 'text-pink-500',
  agent: 'text-indigo-500',
  ea_bot: 'text-violet-500',
};

export function getBlockIconForType(blockType: string): React.ReactNode {
  const Icon = BLOCK_ICON_COMPONENTS[blockType] ?? Sparkles;
  const colorClass = BLOCK_ICON_COLORS[blockType] ?? 'text-violet-500';
  return <Icon className={`w-4 h-4 ${colorClass}`} />;
}

export const BUILT_IN_BLOCKS: BuiltInBlock[] = [
  {
    type: 'if_else',
    label: 'If/Else',
    description: 'Conditional logic',
    icon: getBlockIconForType('if_else'),
  },
  {
    type: 'for_loop',
    label: 'For Loop',
    description: 'Iterate over array',
    icon: getBlockIconForType('for_loop'),
  },
  {
    type: 'mcp',
    label: 'MCP Tool',
    description: 'Invoke external MCP server tool',
    icon: getBlockIconForType('mcp'),
  },
  {
    type: 'hitl',
    label: 'Human Approval',
    description: 'Pause for human review and input',
    icon: getBlockIconForType('hitl'),
  },
  {
    type: 'browser',
    label: 'Browser',
    description: 'Automate browser actions with natural language',
    icon: getBlockIconForType('browser'),
  },
  {
    type: 'image_gen',
    label: 'Image Gen',
    description: 'Generate images from text prompts',
    icon: getBlockIconForType('image_gen'),
  },
  {
    type: 'agent',
    label: 'Agent',
    description: 'AI agent with tool access for multi-step tasks',
    icon: getBlockIconForType('agent'),
  },
  ...(SHOW_MEETINGS ? [{
    type: 'ea_bot' as const,
    label: 'meetingsEA',
    description: 'Join meeting, record, and return transcript when done',
    icon: getBlockIconForType('ea_bot'),
  }] : []),
];
