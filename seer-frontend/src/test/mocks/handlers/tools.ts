import { http, HttpResponse } from 'msw';

export const toolHandlers = [
  // Get tools metadata
  http.get('*/api/tools', () => {
    return HttpResponse.json({
      tools: [
        {
          name: 'gmail',
          display_name: 'Gmail',
          description: 'Send and read Gmail emails',
          integration_type: 'google',
          required_scopes: ['gmail.readonly', 'gmail.send'],
          parameters: {
            type: 'object',
            properties: {},
            required: [],
          },
        },
        {
          name: 'google_drive',
          display_name: 'Google Drive',
          description: 'Access Google Drive files',
          integration_type: 'google',
          required_scopes: ['drive.readonly'],
          parameters: {
            type: 'object',
            properties: {},
            required: [],
          },
        },
        {
          name: 'github',
          display_name: 'GitHub',
          description: 'Access GitHub repositories',
          integration_type: 'github',
          required_scopes: ['repo'],
          parameters: {
            type: 'object',
            properties: {},
            required: [],
          },
        },
      ],
    });
  }),

  // Get tool connection status
  http.get('*/api/integrations/tools/status', () => {
    return HttpResponse.json({
      tools: [
        {
          tool_name: 'gmail',
          connected: false,
          has_required_scopes: false,
          connection_id: null,
        },
      ],
    });
  }),

  // Connect integration
  http.post('*/api/integrations/:provider/connect', () => {
    return HttpResponse.json({
      success: true,
      connection_id: 'conn-123',
    });
  }),

  // Disconnect integration
  http.delete('*/api/integrations/:provider/disconnect/:connectionId', () => {
    return HttpResponse.json({
      success: true,
    });
  }),
];
