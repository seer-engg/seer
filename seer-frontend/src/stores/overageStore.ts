import {
  overageApi,
  type OverageConfig,
  type OverageSettings,
  type OverageUsageRecord,
} from "@/lib/overage-api";
import { createStore } from "./createStore";

export interface OverageState {
  // Data
  config: OverageConfig | null;
  settings: OverageSettings | null;
  usageRecords: OverageUsageRecord[];
  usageRecordsTotalCount: number;
  usageRecordsPage: number;
  usageRecordsHasMore: boolean;

  // Loading states
  isLoading: boolean;
  isEnabling: boolean;
  isDisabling: boolean;
  isUpdatingCap: boolean;
  isLoadingUsage: boolean;

  // Error state
  error: string | null;

  // Actions
  fetchConfig: () => Promise<void>;
  fetchSettings: () => Promise<void>;
  enableOverage: (spendingCapCents: number) => Promise<boolean>;
  disableOverage: () => Promise<{ success: boolean; message: string }>;
  updateSpendingCap: (spendingCapCents: number) => Promise<boolean>;
  fetchUsageRecords: (
    page?: number,
    pageSize?: number,
    status?: "pending" | "reported" | "failed"
  ) => Promise<void>;
  clearError: () => void;
}

type SetState = (partial: Partial<OverageState>) => void;
type GetState = () => OverageState;

// ─────────────────────────────────────────────────────────────────────────────
// Action Implementations
// ─────────────────────────────────────────────────────────────────────────────

async function enableOverageAction(
  set: SetState,
  spendingCapCents: number
): Promise<boolean> {
  set({ isEnabling: true, error: null });
  try {
    const response = await overageApi.enable(spendingCapCents);
    if (response.success) {
      set({ settings: response.settings, isEnabling: false });
      return true;
    }
    set({ error: "Failed to enable usage-based pricing", isEnabling: false });
    return false;
  } catch (err) {
    const message = err instanceof Error ? err.message : "Failed to enable usage-based pricing";
    set({ error: message, isEnabling: false });
    return false;
  }
}

async function disableOverageAction(
  set: SetState,
  get: GetState
): Promise<{ success: boolean; message: string }> {
  set({ isDisabling: true, error: null });
  try {
    const response = await overageApi.disable();
    if (response.success) {
      await get().fetchSettings();
      set({ isDisabling: false });
      return { success: true, message: response.message };
    }
    set({ error: "Failed to disable usage-based pricing", isDisabling: false });
    return { success: false, message: "Failed to disable usage-based pricing" };
  } catch (err) {
    const message = err instanceof Error ? err.message : "Failed to disable usage-based pricing";
    set({ error: message, isDisabling: false });
    return { success: false, message };
  }
}

async function updateSpendingCapAction(
  set: SetState,
  get: GetState,
  spendingCapCents: number
): Promise<boolean> {
  set({ isUpdatingCap: true, error: null });
  try {
    const response = await overageApi.updateCap(spendingCapCents);
    if (!response.success) {
      set({ error: "Failed to update spending cap", isUpdatingCap: false });
      return false;
    }

    const currentSettings = get().settings;
    if (currentSettings) {
      const remainingCents = response.spending_cap_cents - currentSettings.current_usage_cents;
      set({
        settings: {
          ...currentSettings,
          spending_cap_cents: response.spending_cap_cents,
          spending_cap_dollars: response.spending_cap_dollars,
          remaining_cents: remainingCents,
          remaining_dollars: remainingCents / 100,
          cap_reached: currentSettings.current_usage_cents >= response.spending_cap_cents,
        },
        isUpdatingCap: false,
      });
    } else {
      set({ isUpdatingCap: false });
    }
    return true;
  } catch (err) {
    const message = err instanceof Error ? err.message : "Failed to update spending cap";
    set({ error: message, isUpdatingCap: false });
    return false;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Store
// ─────────────────────────────────────────────────────────────────────────────

export const useOverageStore = createStore<OverageState>((set, get) => ({
  // Initial state
  config: null,
  settings: null,
  usageRecords: [],
  usageRecordsTotalCount: 0,
  usageRecordsPage: 1,
  usageRecordsHasMore: false,

  isLoading: false,
  isEnabling: false,
  isDisabling: false,
  isUpdatingCap: false,
  isLoadingUsage: false,

  error: null,

  fetchConfig: async () => {
    try {
      const config = await overageApi.getConfig();
      set({ config });
    } catch {
      set({ error: "Failed to fetch overage configuration" });
    }
  },

  fetchSettings: async () => {
    set({ isLoading: true, error: null });
    try {
      const settings = await overageApi.getSettings();
      set({ settings, isLoading: false });
    } catch {
      set({ error: "Failed to fetch overage settings", isLoading: false });
    }
  },

  enableOverage: (spendingCapCents) => enableOverageAction(set, spendingCapCents),

  disableOverage: () => disableOverageAction(set, get),

  updateSpendingCap: (spendingCapCents) => updateSpendingCapAction(set, get, spendingCapCents),

  fetchUsageRecords: async (page = 1, pageSize = 20, status) => {
    set({ isLoadingUsage: true, error: null });
    try {
      const response = await overageApi.getUsageRecords(page, pageSize, status);
      set({
        usageRecords: response.items,
        usageRecordsTotalCount: response.total_count,
        usageRecordsPage: response.page,
        usageRecordsHasMore: response.has_more,
        isLoadingUsage: false,
      });
    } catch {
      set({ error: "Failed to fetch usage records", isLoadingUsage: false });
    }
  },

  clearError: () => set({ error: null }),
}));
