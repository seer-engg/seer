import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { MemoryCard } from "@/components/settings/MemoryCard";

const {
  mockListMemories,
  mockSearchMemories,
  mockGetStats,
  mockListBanks,
  mockCreateBank,
  mockUpdateBank,
  mockDeleteBank,
  mockSetDefaultBank,
  mockCreateMemory,
  mockUpdateMemory,
  mockDeleteMemory,
  mockDeleteAllMemories,
  mockToast,
} = vi.hoisted(() => ({
  mockListMemories: vi.fn(),
  mockSearchMemories: vi.fn(),
  mockGetStats: vi.fn(),
  mockListBanks: vi.fn(),
  mockCreateBank: vi.fn(),
  mockUpdateBank: vi.fn(),
  mockDeleteBank: vi.fn(),
  mockSetDefaultBank: vi.fn(),
  mockCreateMemory: vi.fn(),
  mockUpdateMemory: vi.fn(),
  mockDeleteMemory: vi.fn(),
  mockDeleteAllMemories: vi.fn(),
  mockToast: vi.fn(),
}));

vi.mock("@/lib/memory-api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/memory-api")>();

  return {
    ...actual,
    memoryApi: {
      listBanks: mockListBanks,
      createBank: mockCreateBank,
      updateBank: mockUpdateBank,
      deleteBank: mockDeleteBank,
      setDefaultBank: mockSetDefaultBank,
      listMemories: mockListMemories,
      searchMemories: mockSearchMemories,
      getStats: mockGetStats,
      createMemory: mockCreateMemory,
      updateMemory: mockUpdateMemory,
      deleteMemory: mockDeleteMemory,
      deleteAllMemories: mockDeleteAllMemories,
    },
  };
});

vi.mock("@/hooks/utility/use-toast", () => ({
  useToast: () => ({
    toast: mockToast,
  }),
}));

const defaultStats = {
  total_memories: 0,
  memory_enabled: true,
  extraction_enabled: true,
  injection_enabled: true,
};

const defaultBanksResponse = {
  items: [
    {
      memory_bank_id: "mb_default",
      name: "Default Memory",
      description: "Workspace default",
      is_default: true,
      memory_count: 0,
      created_at: "2026-03-01T00:00:00+00:00",
      updated_at: "2026-03-01T00:00:00+00:00",
    },
  ],
  total: 1,
};

const emptyMemoriesResponse = {
  memories: [],
  total: 0,
  user_id: "user-123",
};

const emptySearchResponse = {
  memories: [],
  total: 0,
  query: "",
};

const createdMemoryResponse = {
  memory: {
    id: "mem_new",
    memory: "Remember this preference",
    score: null,
    created_at: "2026-03-01T00:00:00+00:00",
    metadata: { source: "manual" },
  },
  message: "Memory created successfully",
};

const updatedMemoryResponse = {
  memory: {
    id: "mem_1",
    memory: "Updated preference",
    score: null,
    created_at: "2026-03-01T00:00:00+00:00",
    updated_at: "2026-03-02T00:00:00+00:00",
    metadata: { source: "manual", edited_at: "2026-03-02T00:00:00+00:00" },
  },
  message: "Memory updated successfully",
};

function setupDefaultMemoryApiMocks() {
  mockGetStats.mockResolvedValue(defaultStats);
  mockListBanks.mockResolvedValue(defaultBanksResponse);
  mockCreateBank.mockResolvedValue(defaultBanksResponse.items[0]);
  mockUpdateBank.mockResolvedValue(defaultBanksResponse.items[0]);
  mockDeleteBank.mockResolvedValue({ deleted: true, message: "Deleted" });
  mockSetDefaultBank.mockResolvedValue(defaultBanksResponse.items[0]);
  mockListMemories.mockResolvedValue(emptyMemoriesResponse);
  mockSearchMemories.mockResolvedValue(emptySearchResponse);
  mockCreateMemory.mockResolvedValue(createdMemoryResponse);
  mockUpdateMemory.mockResolvedValue(updatedMemoryResponse);
  mockDeleteMemory.mockResolvedValue(undefined);
  mockDeleteAllMemories.mockResolvedValue(undefined);
}

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
        <MemoryCard />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("MemoryCard", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupDefaultMemoryApiMocks();
  });

  it("creates a memory from the empty state dialog", async () => {
    const user = userEvent.setup();

    renderComponent();

    expect(await screen.findByText("No memories yet")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /add your first memory/i }));

    const dialog = await screen.findByRole("dialog");
    const textarea = within(dialog).getByLabelText("Memory");
    await user.type(textarea, "  Remember this preference  ");
    await user.click(within(dialog).getByRole("button", { name: "Add Memory" }));

    await waitFor(() => {
      expect(mockCreateMemory).toHaveBeenCalled();
      expect(mockCreateMemory.mock.calls[0][0]).toBe("Remember this preference");
      expect(mockCreateMemory.mock.calls[0][1]).toBe("mb_default");
    });

    await waitFor(() => {
      expect(mockToast).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Memory created",
        }),
      );
    });
  });

  it("opens the edit dialog with current memory text and saves updates", async () => {
    const user = userEvent.setup();
    mockGetStats.mockResolvedValue({
      total_memories: 1,
      memory_enabled: true,
      extraction_enabled: true,
      injection_enabled: true,
    });
    mockListMemories.mockResolvedValue({
      memories: [
        {
          id: "mem_1",
          memory: "Original preference",
          score: null,
          created_at: "2026-03-01T00:00:00+00:00",
          metadata: { source: "chat_session" },
        },
      ],
      total: 1,
      user_id: "user-123",
    });

    renderComponent();

    expect(await screen.findByText("Original preference")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /edit memory:/i }));

    const dialog = await screen.findByRole("dialog");
    const textarea = within(dialog).getByLabelText("Memory");
    expect(textarea).toHaveValue("Original preference");

    await user.clear(textarea);
    await user.type(textarea, "  Updated preference  ");
    await user.click(within(dialog).getByRole("button", { name: "Save Changes" }));

    await waitFor(() => {
      expect(mockUpdateMemory).toHaveBeenCalledWith("mem_1", "Updated preference", "mb_default");
    });

    await waitFor(() => {
      expect(mockToast).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Memory updated",
        }),
      );
    });
  });
});
