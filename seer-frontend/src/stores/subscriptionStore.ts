import { subscriptionApi, type SubscriptionInfo, type PriceTier } from "@/lib/subscription-api";
import { createStore } from "./createStore";

export interface SubscriptionState {
  subscription: SubscriptionInfo | null;
  pricing: PriceTier[] | null;
  isLoading: boolean;
  error: string | null;
  fetchSubscription: () => Promise<void>;
  fetchPricing: () => Promise<void>;
  redirectToCheckout: (tier: string, interval: string) => Promise<void>;
  redirectToPortal: () => Promise<void>;
}

export const useSubscriptionStore = createStore<SubscriptionState>((set) => ({
  subscription: null,
  pricing: null,
  isLoading: false,
  error: null,

  fetchSubscription: async () => {
    set({ isLoading: true, error: null });
    try {
      const subscription = await subscriptionApi.getCurrentSubscription();
      set({ subscription, isLoading: false });
    } catch {
      set({ error: "Failed to fetch subscription", isLoading: false });
    }
  },

  fetchPricing: async () => {
    try {
      const response = await subscriptionApi.getPricing();
      set({ pricing: response.prices });
    } catch {
      set({ error: "Failed to fetch pricing" });
    }
  },

  redirectToCheckout: async (tier: string, interval: string) => {
    set({ isLoading: true, error: null });
    try {
      const { checkout_url } = await subscriptionApi.createCheckout(tier, interval);
      window.location.href = checkout_url;
    } catch {
      set({ error: "Failed to create checkout session", isLoading: false });
    }
  },

  redirectToPortal: async () => {
    set({ isLoading: true, error: null });
    try {
      const { portal_url } = await subscriptionApi.createPortalSession();
      window.location.href = portal_url;
    } catch {
      set({ error: "Failed to open billing portal", isLoading: false });
    }
  },
}));
