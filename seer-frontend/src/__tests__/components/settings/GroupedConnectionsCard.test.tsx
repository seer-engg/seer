import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter } from 'react-router-dom';

import { GroupedConnectionsCard } from '@/components/settings/GroupedConnectionsCard';

const {
  mockGetConnectionsByProvider,
  mockDeleteConnectedAccount,
  mockInitiateConnection,
  mockToast,
} = vi.hoisted(() => ({
  mockGetConnectionsByProvider: vi.fn(),
  mockDeleteConnectedAccount: vi.fn(),
  mockInitiateConnection: vi.fn(),
  mockToast: vi.fn(),
}));

vi.mock('@/lib/api-client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/lib/api-client')>();

  return {
    ...actual,
    getConnectionsByProvider: mockGetConnectionsByProvider,
    deleteConnectedAccount: mockDeleteConnectedAccount,
    initiateConnection: mockInitiateConnection,
  };
});

vi.mock('@/hooks/utility/use-toast', () => ({
  useToast: () => ({
    toast: mockToast,
  }),
}));

const baseConnectionsResponse = {
  connections: {
    google: [
      {
        id: 1,
        provider: 'google',
        provider_account_id: '113421198217490684849',
        display_name: 'lokesh@getseer.dev',
        status: 'active',
        scopes: 'email profile https://www.googleapis.com/auth/gmail.compose',
      },
    ],
  },
};

function renderComponent() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
      },
      mutations: { retry: false },
    },
  });

  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <GroupedConnectionsCard />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe('GroupedConnectionsCard', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockGetConnectionsByProvider.mockResolvedValue(baseConnectionsResponse);
    mockDeleteConnectedAccount.mockResolvedValue(undefined);
    mockInitiateConnection.mockResolvedValue({ redirectUrl: '', connectionId: '1' });
  });

  it('renders display_name as the primary account label', async () => {
    renderComponent();

    expect(await screen.findByText('lokesh@getseer.dev')).toBeInTheDocument();
    expect(screen.getByText('Account ID: 113421198217490684849')).toBeInTheDocument();
    expect(screen.getByText('Active')).toBeInTheDocument();
  });

  it('falls back to provider_account_id when display_name is missing', async () => {
    mockGetConnectionsByProvider.mockResolvedValue({
      connections: {
        google: [
          {
            ...baseConnectionsResponse.connections.google[0],
            display_name: null,
          },
        ],
      },
    });

    renderComponent();

    expect(await screen.findByText('113421198217490684849')).toBeInTheDocument();
    expect(screen.queryByText('Account ID: 113421198217490684849')).not.toBeInTheDocument();
  });

  it('renders formatted scope chips after expanding the scope section', async () => {
    const user = userEvent.setup();

    renderComponent();

    await screen.findByText('lokesh@getseer.dev');
    expect(screen.queryByText('Email')).not.toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: /show granted scopes/i }));

    expect(screen.getByText('Email')).toBeInTheDocument();
    expect(screen.getByText('Profile')).toBeInTheDocument();
    expect(screen.getByText('gmail.compose')).toBeInTheDocument();
  });

  it('uses the display label in the delete dialog and success toast', async () => {
    const user = userEvent.setup();

    renderComponent();

    await screen.findByText('lokesh@getseer.dev');
    await user.click(screen.getByRole('button', { name: /disconnect lokesh@getseer\.dev/i }));

    expect(screen.getByRole('alertdialog')).toHaveTextContent(
      'Are you sure you want to disconnect lokesh@getseer.dev? Workflows using this account will need to be reconfigured.',
    );

    await user.click(screen.getByRole('button', { name: 'Delete' }));

    await waitFor(() => {
      expect(mockDeleteConnectedAccount).toHaveBeenCalledWith('1');
    });

    await waitFor(() => {
      expect(mockToast).toHaveBeenCalledWith(
        expect.objectContaining({
          title: 'Connection deleted',
          description: expect.stringContaining('lokesh@getseer.dev'),
        }),
      );
    });
  });
});
