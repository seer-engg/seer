import { ReactElement } from 'react';
import { render, RenderOptions } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';

// Reset all Zustand stores
export function resetAllStores() {
  // For now, stores will be reset individually in tests that need them
  // This is a placeholder that can be expanded as stores are integrated into tests

  // Example of how to reset stores in actual tests:
  // import { useCanvasStore } from '@/stores/canvasStore';
  // useCanvasStore.setState({ nodes: [], edges: [], selectedNodeId: null });
}

// Custom render with providers
function createTestQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0, // Disable caching between tests
      },
      mutations: { retry: false },
    },
    logger: {
      log: console.log,
      warn: console.warn,
      error: () => {}, // Suppress error logs in tests
    },
  });
}

export function renderWithProviders(
  ui: ReactElement,
  options?: RenderOptions & {
    initialRoute?: string;
    queryClient?: QueryClient;
  }
) {
  const {
    initialRoute = '/',
    queryClient = createTestQueryClient(),
    ...renderOptions
  } = options || {};

  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[initialRoute]}>
        {ui}
      </MemoryRouter>
    </QueryClientProvider>,
    renderOptions
  );
}

// eslint-disable-next-line react-refresh/only-export-components
export * from '@testing-library/react';
export { renderWithProviders as render, resetAllStores };
