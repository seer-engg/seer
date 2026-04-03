/**
 * Test fixtures for common data structures
 */

import type { Node, Edge } from '@xyflow/react';

export const mockWorkflow = {
  id: 'workflow-1',
  name: 'Test Workflow',
  status: 'draft' as const,
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
  graph: {
    nodes: [],
    edges: [],
  },
};

export const mockToolNode: Node = {
  id: 'tool-1',
  type: 'tool',
  position: { x: 200, y: 0 },
  data: {
    blockType: 'tool',
    tool_name: 'gmail',
    config: {},
  },
};

export const mockEdge: Edge = {
  id: 'edge-1',
  source: 'tool-1',
  target: 'tool-1',
};

export const mockUser = {
  id: 'test-user-id',
  email: 'test@example.com',
};

export const mockTool = {
  tool_name: 'gmail',
  display_name: 'Gmail',
  integration_type: 'google',
  required_scopes: ['gmail.readonly', 'gmail.send'],
};

export const mockIntegrationStatus = {
  tool_name: 'gmail',
  connected: false,
  has_required_scopes: false,
  connection_id: null,
};

export const mockTrigger = {
  id: 'trigger-1',
  type: 'webhook',
  enabled: false,
  config: {},
};

export const mockRun = {
  run_id: 'run-123',
  workflow_id: 'workflow-1',
  status: 'running' as const,
  created_at: new Date().toISOString(),
  inputs: {},
};
