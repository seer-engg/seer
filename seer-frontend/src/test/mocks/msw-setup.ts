import { setupServer } from 'msw/node';
import { workflowHandlers } from './handlers/workflows';
import { toolHandlers } from './handlers/tools';
import { authHandlers } from './handlers/auth';

export const server = setupServer(
  ...workflowHandlers,
  ...toolHandlers,
  ...authHandlers
);
