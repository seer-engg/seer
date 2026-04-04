import { usageApi, type UsageSummary } from "@/lib/usage-api";
import { createStore } from "./createStore";

export interface UsageState {
  usage: UsageSummary | null;
  isLoading: boolean;
  error: string | null;
  lastFetched: number | null;
  fetchUsage: (options?: { force?: boolean }) => Promise<void>;
}

const FETCH_TTL_MS = 30_000;

export const useUsageStore = createStore<UsageState>((set, get) => ({
  usage: null,
  isLoading: false,
  error: null,
  lastFetched: null,

  fetchUsage: async (options) => {
    const force = options?.force ?? false;
    const now = Date.now();
    const { lastFetched } = get();

    if (!force && lastFetched && now - lastFetched < FETCH_TTL_MS) {
      return;
    }

    set({ isLoading: true, error: null });
    try {
      const usage = await usageApi.getUsageSummary();
      set({ usage, isLoading: false, lastFetched: now });
    } catch (error) {
      console.error("Failed to fetch usage summary", error);
      set({
        error: "Unable to load usage data right now.",
        isLoading: false,
      });
    }
  },
}));
