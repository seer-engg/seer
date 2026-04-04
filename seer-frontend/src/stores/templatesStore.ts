import type { StateCreator } from 'zustand';

import { backendApiClient } from '@/lib/api-client';
import type {
  TemplateSummary,
  TemplateDetail,
  TemplateCategory,
  TemplateRequirementsResponse,
  TemplateInstantiateRequest,
  TemplateInstantiateResponse,
  TemplateListResponse,
  TemplateCategoriesResponse,
  TemplateAdminResponse,
  TemplateAdminListResponse,
  TemplateCreateRequest,
  TemplateUpdateRequest,
} from '@/types/templates';

import { createStore } from './createStore';

export interface TemplatesStore {
  // Data
  templates: TemplateSummary[];
  myTemplates: TemplateSummary[];
  communityTemplates: TemplateSummary[];
  categories: TemplateCategory[];
  selectedTemplate: TemplateDetail | null;
  requirements: TemplateRequirementsResponse | null;

  // Loading states
  isLoadingTemplates: boolean;
  isLoadingCategories: boolean;
  isLoadingDetail: boolean;
  isLoadingRequirements: boolean;
  isInstantiating: boolean;

  // Filters
  selectedCategory: string | null;
  searchQuery: string;

  // Error state
  error: string | null;

  // Actions
  loadTemplates: (params?: { category?: string; search?: string; featured?: boolean; scope?: 'mine' | 'community' }) => Promise<TemplateSummary[]>;
  loadMyTemplates: (params?: { category?: string; search?: string }) => Promise<TemplateSummary[]>;
  loadCommunityTemplates: (params?: { category?: string; search?: string }) => Promise<TemplateSummary[]>;
  loadCategories: () => Promise<TemplateCategory[]>;
  loadTemplateDetail: (slug: string) => Promise<TemplateDetail>;
  loadRequirements: (slug: string) => Promise<TemplateRequirementsResponse>;
  instantiateTemplate: (slug: string, request: TemplateInstantiateRequest) => Promise<TemplateInstantiateResponse>;

  // Filter actions
  setSelectedCategory: (category: string | null) => void;
  setSearchQuery: (query: string) => void;

  // Reset actions
  clearSelectedTemplate: () => void;
  resetFilters: () => void;

  // Admin data
  adminTemplates: TemplateAdminResponse[];

  // Admin loading states
  isLoadingAdminTemplates: boolean;
  isCreatingTemplate: boolean;
  isUpdatingTemplate: boolean;
  isDeletingTemplate: boolean;

  // Admin actions
  loadAdminTemplates: () => Promise<TemplateAdminResponse[]>;
  createTemplate: (request: TemplateCreateRequest) => Promise<TemplateAdminResponse>;
  updateTemplate: (slug: string, request: TemplateUpdateRequest) => Promise<TemplateAdminResponse>;
  deleteTemplate: (slug: string) => Promise<void>;
}

const loadTemplatesImpl = async (
  set: (partial: Partial<TemplatesStore>) => void,
  params?: { category?: string; search?: string; featured?: boolean },
) => {
  set({ isLoadingTemplates: true, error: null });
  try {
    const searchParams = new URLSearchParams();
    if (params?.category) searchParams.set('category', params.category);
    if (params?.search) searchParams.set('search', params.search);
    if (params?.featured !== undefined) searchParams.set('featured', String(params.featured));
    if (params?.scope) searchParams.set('scope', params.scope);

    const query = searchParams.toString();
    const endpoint = `/api/v1/templates${query ? `?${query}` : ''}`;

    const response = await backendApiClient.request<TemplateListResponse>(endpoint, {
      method: 'GET',
    });

    set({ templates: response.items, isLoadingTemplates: false });
    return response.items;
  } catch (error) {
    set({
      isLoadingTemplates: false,
      error: error instanceof Error ? error.message : 'Failed to load templates',
    });
    throw error;
  }
};

const loadCategoriesImpl = async (set: (partial: Partial<TemplatesStore>) => void) => {
  set({ isLoadingCategories: true, error: null });
  try {
    const response = await backendApiClient.request<TemplateCategoriesResponse>('/api/v1/templates/categories', {
      method: 'GET',
    });

    set({ categories: response.items, isLoadingCategories: false });
    return response.items;
  } catch (error) {
    set({
      isLoadingCategories: false,
      error: error instanceof Error ? error.message : 'Failed to load categories',
    });
    throw error;
  }
};

const loadTemplateDetailImpl = async (
  set: (partial: Partial<TemplatesStore>) => void,
  slug: string,
) => {
  set({ isLoadingDetail: true, error: null });
  try {
    const response = await backendApiClient.request<TemplateDetail>(`/api/v1/templates/${slug}`, {
      method: 'GET',
    });

    set({ selectedTemplate: response, isLoadingDetail: false });
    return response;
  } catch (error) {
    set({
      isLoadingDetail: false,
      error: error instanceof Error ? error.message : 'Failed to load template details',
    });
    throw error;
  }
};

const loadRequirementsImpl = async (
  set: (partial: Partial<TemplatesStore>) => void,
  slug: string,
) => {
  set({ isLoadingRequirements: true, error: null });
  try {
    const response = await backendApiClient.request<TemplateRequirementsResponse>(
      `/api/v1/templates/${slug}/requirements`,
      { method: 'GET' },
    );

    set({ requirements: response, isLoadingRequirements: false });
    return response;
  } catch (error) {
    set({
      isLoadingRequirements: false,
      error: error instanceof Error ? error.message : 'Failed to load requirements',
    });
    throw error;
  }
};

const instantiateTemplateImpl = async (
  set: (partial: Partial<TemplatesStore>) => void,
  slug: string,
  request: TemplateInstantiateRequest,
) => {
  set({ isInstantiating: true, error: null });
  try {
    const response = await backendApiClient.request<TemplateInstantiateResponse>(
      `/api/v1/templates/${slug}/instantiate`,
      {
        method: 'POST',
        body: request as unknown as Record<string, unknown>,
      },
    );

    set({ isInstantiating: false });
    return response;
  } catch (error) {
    set({
      isInstantiating: false,
      error: error instanceof Error ? error.message : 'Failed to create workflow from template',
    });
    throw error;
  }
};

const loadMyTemplatesImpl = async (
  set: (partial: Partial<TemplatesStore>) => void,
  params?: { category?: string; search?: string },
) => {
  set({ isLoadingTemplates: true, error: null });
  try {
    const searchParams = new URLSearchParams();
    searchParams.set('scope', 'mine');
    if (params?.category) searchParams.set('category', params.category);
    if (params?.search) searchParams.set('search', params.search);

    const query = searchParams.toString();
    const endpoint = `/api/v1/templates?${query}`;

    const response = await backendApiClient.request<TemplateListResponse>(endpoint, {
      method: 'GET',
    });

    set({ myTemplates: response.items, isLoadingTemplates: false });
    return response.items;
  } catch (error) {
    set({
      isLoadingTemplates: false,
      error: error instanceof Error ? error.message : 'Failed to load templates',
    });
    throw error;
  }
};

const loadCommunityTemplatesImpl = async (
  set: (partial: Partial<TemplatesStore>) => void,
  params?: { category?: string; search?: string },
) => {
  set({ isLoadingTemplates: true, error: null });
  try {
    const searchParams = new URLSearchParams();
    searchParams.set('scope', 'community');
    if (params?.category) searchParams.set('category', params.category);
    if (params?.search) searchParams.set('search', params.search);

    const query = searchParams.toString();
    const endpoint = `/api/v1/templates?${query}`;

    const response = await backendApiClient.request<TemplateListResponse>(endpoint, {
      method: 'GET',
    });

    set({ communityTemplates: response.items, isLoadingTemplates: false });
    return response.items;
  } catch (error) {
    set({
      isLoadingTemplates: false,
      error: error instanceof Error ? error.message : 'Failed to load templates',
    });
    throw error;
  }
};

// =============================================================================
// Admin Action Implementations
// =============================================================================

const loadAdminTemplatesImpl = async (set: (partial: Partial<TemplatesStore>) => void) => {
  set({ isLoadingAdminTemplates: true, error: null });
  try {
    const response = await backendApiClient.request<TemplateAdminListResponse>('/api/v1/admin/templates', {
      method: 'GET',
    });

    set({ adminTemplates: response.items, isLoadingAdminTemplates: false });
    return response.items;
  } catch (error) {
    set({
      isLoadingAdminTemplates: false,
      error: error instanceof Error ? error.message : 'Failed to load admin templates',
    });
    throw error;
  }
};

const createTemplateImpl = async (
  set: (partial: Partial<TemplatesStore>) => void,
  get: () => TemplatesStore,
  request: TemplateCreateRequest,
) => {
  set({ isCreatingTemplate: true, error: null });
  try {
    const response = await backendApiClient.request<TemplateAdminResponse>('/api/v1/admin/templates', {
      method: 'POST',
      body: request as unknown as Record<string, unknown>,
    });

    const currentTemplates = get().adminTemplates;
    set({
      adminTemplates: [response, ...currentTemplates],
      isCreatingTemplate: false,
    });
    return response;
  } catch (error) {
    set({
      isCreatingTemplate: false,
      error: error instanceof Error ? error.message : 'Failed to create template',
    });
    throw error;
  }
};

const updateTemplateImpl = async (
  set: (partial: Partial<TemplatesStore>) => void,
  get: () => TemplatesStore,
  slug: string,
  request: TemplateUpdateRequest,
) => {
  set({ isUpdatingTemplate: true, error: null });
  try {
    const response = await backendApiClient.request<TemplateAdminResponse>(`/api/v1/admin/templates/${slug}`, {
      method: 'PUT',
      body: request as unknown as Record<string, unknown>,
    });

    const currentTemplates = get().adminTemplates;
    set({
      adminTemplates: currentTemplates.map((t) => (t.slug === slug ? response : t)),
      isUpdatingTemplate: false,
    });
    return response;
  } catch (error) {
    set({
      isUpdatingTemplate: false,
      error: error instanceof Error ? error.message : 'Failed to update template',
    });
    throw error;
  }
};

const deleteTemplateImpl = async (
  set: (partial: Partial<TemplatesStore>) => void,
  get: () => TemplatesStore,
  slug: string,
) => {
  set({ isDeletingTemplate: true, error: null });
  try {
    await backendApiClient.request(`/api/v1/admin/templates/${slug}`, {
      method: 'DELETE',
    });

    const currentTemplates = get().adminTemplates;
    set({
      adminTemplates: currentTemplates.filter((t) => t.slug !== slug),
      isDeletingTemplate: false,
    });
  } catch (error) {
    set({
      isDeletingTemplate: false,
      error: error instanceof Error ? error.message : 'Failed to delete template',
    });
    throw error;
  }
};

const createTemplatesStore: StateCreator<TemplatesStore> = (set, get) => ({
  // Initial state
  templates: [],
  myTemplates: [],
  communityTemplates: [],
  categories: [],
  selectedTemplate: null,
  requirements: null,

  isLoadingTemplates: false,
  isLoadingCategories: false,
  isLoadingDetail: false,
  isLoadingRequirements: false,
  isInstantiating: false,

  selectedCategory: null,
  searchQuery: '',

  error: null,

  // Actions
  async loadTemplates(params) {
    return loadTemplatesImpl(set, params);
  },
  async loadMyTemplates(params) {
    return loadMyTemplatesImpl(set, params);
  },
  async loadCommunityTemplates(params) {
    return loadCommunityTemplatesImpl(set, params);
  },
  async loadCategories() {
    return loadCategoriesImpl(set);
  },
  async loadTemplateDetail(slug) {
    return loadTemplateDetailImpl(set, slug);
  },
  async loadRequirements(slug) {
    return loadRequirementsImpl(set, slug);
  },
  async instantiateTemplate(slug, request) {
    return instantiateTemplateImpl(set, slug, request);
  },

  // Filter actions
  setSelectedCategory(category) {
    set({ selectedCategory: category });
  },
  setSearchQuery(query) {
    set({ searchQuery: query });
  },

  // Reset actions
  clearSelectedTemplate() {
    set({ selectedTemplate: null, requirements: null });
  },
  resetFilters() {
    set({ selectedCategory: null, searchQuery: '' });
  },

  // Admin state
  adminTemplates: [],
  isLoadingAdminTemplates: false,
  isCreatingTemplate: false,
  isUpdatingTemplate: false,
  isDeletingTemplate: false,

  // Admin actions
  async loadAdminTemplates() {
    return loadAdminTemplatesImpl(set);
  },
  async createTemplate(request) {
    return createTemplateImpl(set, get, request);
  },
  async updateTemplate(slug, request) {
    return updateTemplateImpl(set, get, slug, request);
  },
  async deleteTemplate(slug) {
    return deleteTemplateImpl(set, get, slug);
  },
});

export const useTemplatesStore = createStore(createTemplatesStore);
