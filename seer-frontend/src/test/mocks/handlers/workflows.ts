import { http, HttpResponse } from 'msw';

export const workflowHandlers = [
  // Get workflows list
  http.get('/api/v1/workflows', () => {
    return HttpResponse.json([
      {
        id: 'workflow-1',
        name: 'Test Workflow',
        status: 'draft',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      },
    ]);
  }),

  // Get single workflow
  http.get('/api/v1/workflows/:id', ({ params }) => {
    return HttpResponse.json({
      id: params.id,
      name: 'Test Workflow',
      status: 'draft',
      graph: {
        nodes: [],
        edges: [],
      },
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    });
  }),

  // Save draft
  http.patch('/api/v1/workflows/:id/draft', async ({ request }) => {
    const body = await request.json();
    return HttpResponse.json({
      success: true,
      workflow: {
        id: 'workflow-1',
        ...body,
        updated_at: new Date().toISOString(),
      }
    });
  }),

  // Publish workflow
  http.post('/api/v1/workflows/:id/publish', () => {
    return HttpResponse.json({
      version_id: 'v1',
      workflow_id: 'workflow-1',
      triggers: [],
      published_at: new Date().toISOString(),
    });
  }),

  // Execute workflow
  http.post('/api/v1/workflows/:id/execute', async ({ request }) => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const body = await request.json() as Record<string, any>;
    return HttpResponse.json({
      run_id: 'run-123',
      workflow_id: 'workflow-1',
      status: 'running',
      created_at: new Date().toISOString(),
      inputs: body?.inputs || {},
    });
  }),

  // Get run history
  http.get('/api/v1/runs/:runId/history', () => {
    return HttpResponse.json({
      history: [
        {
          run_id: 'run-123',
          status: 'completed',
          traces: [],
          created_at: new Date().toISOString(),
          completed_at: new Date().toISOString(),
        }
      ]
    });
  }),
];
