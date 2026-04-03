/* eslint-disable max-lines-per-function */
import { useEffect, useCallback, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { Copy, Loader2, Pencil, Trash2 } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { toast } from '@/components/ui/sonner';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from '@/components/ui/table';
import { TemplateCard } from '@/components/templates/TemplateCard';
import { CategoryFilter } from '@/components/templates/CategoryFilter';
import { TemplateDetailView } from '@/components/templates/TemplateDetailView';
import { EditTemplateDialog } from '@/components/templates/admin/EditTemplateDialog';
import { DeleteTemplateDialog } from '@/components/templates/admin/DeleteTemplateDialog';
import { useTemplatesStore } from '@/stores';
import type { TemplateSummary, TemplateAdminResponse, TemplateUpdateRequest } from '@/types/templates';

type PageView = 'gallery' | 'detail';

function GalleryLoadingSkeleton() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {Array.from({ length: 6 }).map((_, i) => (
        <Skeleton key={i} className="h-32 w-full" />
      ))}
    </div>
  );
}

function TableLoadingSkeleton() {
  return (
    <div className="space-y-2">
      {Array.from({ length: 4 }).map((_, i) => (
        <Skeleton key={i} className="h-12 w-full" />
      ))}
    </div>
  );
}

interface GalleryContentProps {
  templates: TemplateSummary[];
  searchQuery: string;
  isLoading: boolean;
  error: string | null;
  onTemplateClick: (template: TemplateSummary) => void;
}

function GalleryContent({ templates, searchQuery, isLoading, error, onTemplateClick }: GalleryContentProps) {
  if (error) {
    return <div className="text-center py-8"><p className="text-destructive">{error}</p></div>;
  }
  if (isLoading) return <GalleryLoadingSkeleton />;
  if (templates.length === 0) {
    return (
      <div className="text-center py-12">
        <p className="text-muted-foreground">
          {searchQuery ? 'No templates match your search.' : 'No templates available.'}
        </p>
      </div>
    );
  }
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {templates.map((template) => (
        <TemplateCard key={template.template_id} template={template} onClick={() => onTemplateClick(template)} />
      ))}
    </div>
  );
}

function TemplateIcon({ template }: { template: TemplateSummary }) {
  if (template.icon) return <span className="text-lg">{template.icon}</span>;
  return (
    <span className="flex h-7 w-7 items-center justify-center rounded-md bg-muted text-xs font-medium">
      {template.name.charAt(0).toUpperCase()}
    </span>
  );
}

interface MyTemplatesTableProps {
  templates: TemplateSummary[];
  searchQuery: string;
  isLoading: boolean;
  error: string | null;
  onTemplateClick: (template: TemplateSummary) => void;
  onEdit: (template: TemplateSummary) => void;
  onDelete: (template: TemplateSummary) => void;
  onCopyLink: (template: TemplateSummary) => void;
}

function MyTemplatesTable({ templates, searchQuery, isLoading, error, onTemplateClick, onEdit, onDelete, onCopyLink }: MyTemplatesTableProps) {
  if (error) {
    return <div className="text-center py-8"><p className="text-destructive">{error}</p></div>;
  }
  if (isLoading) return <TableLoadingSkeleton />;
  if (templates.length === 0) {
    return (
      <div className="text-center py-12">
        <p className="text-muted-foreground">
          {searchQuery ? 'No templates match your search.' : 'No templates yet. Use "Share as Template" in the workflow editor to create one.'}
        </p>
      </div>
    );
  }

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Name</TableHead>
          <TableHead>Description</TableHead>
          <TableHead className="w-[120px]">Category</TableHead>
          <TableHead className="w-[180px]">Tags</TableHead>
          <TableHead className="w-[60px] text-right">Uses</TableHead>
          <TableHead className="w-[110px]" />
        </TableRow>
      </TableHeader>
      <TableBody>
        {templates.map((template) => (
          <TableRow key={template.template_id}>
            <TableCell>
              <button className="flex items-center gap-2 hover:underline text-left" onClick={() => onTemplateClick(template)}>
                <TemplateIcon template={template} />
                <span className="font-medium text-sm">{template.name}</span>
              </button>
            </TableCell>
            <TableCell>
              <span className="text-sm text-muted-foreground line-clamp-1">{template.description || '—'}</span>
            </TableCell>
            <TableCell>
              <Badge variant="outline" className="text-xs">{template.category}</Badge>
            </TableCell>
            <TableCell>
              <div className="flex flex-wrap gap-1">
                {(template.tags || []).slice(0, 3).map((tag) => (
                  <Badge key={tag} variant="secondary" className="text-xs">{tag}</Badge>
                ))}
                {(template.tags || []).length > 3 && (
                  <Badge variant="secondary" className="text-xs">+{template.tags.length - 3}</Badge>
                )}
              </div>
            </TableCell>
            <TableCell className="text-right text-sm text-muted-foreground">{template.usage_count}</TableCell>
            <TableCell>
              <div className="flex gap-0.5">
                <Tooltip><TooltipTrigger asChild><Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => onCopyLink(template)}><Copy className="h-4 w-4" /></Button></TooltipTrigger><TooltipContent>Copy link</TooltipContent></Tooltip>
                <Tooltip><TooltipTrigger asChild><Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => onEdit(template)}><Pencil className="h-4 w-4" /></Button></TooltipTrigger><TooltipContent>Edit</TooltipContent></Tooltip>
                <Tooltip><TooltipTrigger asChild><Button variant="ghost" size="icon" className="h-7 w-7 text-destructive" onClick={() => onDelete(template)}><Trash2 className="h-4 w-4" /></Button></TooltipTrigger><TooltipContent>Delete</TooltipContent></Tooltip>
              </div>
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

export default function Templates() {
  const navigate = useNavigate();
  const { slug: urlSlug } = useParams<{ slug?: string }>();
  const [view, setView] = useState<PageView>('gallery');
  const [searchDebounce, setSearchDebounce] = useState('');
  const [activeTab, setActiveTab] = useState<'mine' | 'community'>('mine');
  const [editingTemplate, setEditingTemplate] = useState<TemplateSummary | null>(null);
  const [deletingTemplate, setDeletingTemplate] = useState<TemplateSummary | null>(null);

  // Store selectors
  const categories = useTemplatesStore((state) => state.categories);
  const selectedTemplate = useTemplatesStore((state) => state.selectedTemplate);
  const requirements = useTemplatesStore((state) => state.requirements);
  const selectedCategory = useTemplatesStore((state) => state.selectedCategory);
  const searchQuery = useTemplatesStore((state) => state.searchQuery);
  const isLoadingTemplates = useTemplatesStore((state) => state.isLoadingTemplates);
  const isLoadingDetail = useTemplatesStore((state) => state.isLoadingDetail);
  const isLoadingRequirements = useTemplatesStore((state) => state.isLoadingRequirements);
  const isInstantiating = useTemplatesStore((state) => state.isInstantiating);
  const isUpdatingTemplate = useTemplatesStore((state) => state.isUpdatingTemplate);
  const isDeletingTemplate = useTemplatesStore((state) => state.isDeletingTemplate);
  const error = useTemplatesStore((state) => state.error);

  const myTemplates = useTemplatesStore((state) => state.myTemplates);
  const communityTemplates = useTemplatesStore((state) => state.communityTemplates);

  // Store actions
  const loadMyTemplates = useTemplatesStore((state) => state.loadMyTemplates);
  const loadCommunityTemplates = useTemplatesStore((state) => state.loadCommunityTemplates);
  const loadCategories = useTemplatesStore((state) => state.loadCategories);
  const loadTemplateDetail = useTemplatesStore((state) => state.loadTemplateDetail);
  const loadRequirements = useTemplatesStore((state) => state.loadRequirements);
  const instantiateTemplate = useTemplatesStore((state) => state.instantiateTemplate);
  const updateTemplate = useTemplatesStore((state) => state.updateTemplate);
  const deleteTemplate = useTemplatesStore((state) => state.deleteTemplate);
  const setSelectedCategory = useTemplatesStore((state) => state.setSelectedCategory);
  const setSearchQuery = useTemplatesStore((state) => state.setSearchQuery);
  const clearSelectedTemplate = useTemplatesStore((state) => state.clearSelectedTemplate);

  // Auto-open template from URL slug
  useEffect(() => {
    if (urlSlug) {
      loadTemplateDetail(urlSlug)
        .then(() => {
          loadRequirements(urlSlug).catch(console.error);
          setView('detail');
        })
        .catch(() => toast.error('Template not found'));
    }
  }, [urlSlug, loadTemplateDetail, loadRequirements]);

  useEffect(() => { loadCategories().catch(console.error); }, [loadCategories]);

  // Debounce search
  useEffect(() => {
    const timer = setTimeout(() => {
      if (searchDebounce !== searchQuery) setSearchQuery(searchDebounce);
    }, 300);
    return () => clearTimeout(timer);
  }, [searchDebounce, searchQuery, setSearchQuery]);

  // Reload templates when filters change
  useEffect(() => {
    if (view === 'gallery') {
      const params = { category: selectedCategory ?? undefined, search: searchQuery || undefined };
      if (activeTab === 'mine') loadMyTemplates(params).catch(console.error);
      else loadCommunityTemplates(params).catch(console.error);
    }
  }, [view, selectedCategory, searchQuery, activeTab, loadMyTemplates, loadCommunityTemplates]);

  const handleTemplateClick = useCallback(async (template: TemplateSummary) => {
    try {
      await loadTemplateDetail(template.slug);
      loadRequirements(template.slug).catch(console.error);
      setView('detail');
    } catch {
      toast.error('Failed to load template details');
    }
  }, [loadTemplateDetail, loadRequirements]);

  const handleBack = useCallback(() => { setView('gallery'); clearSelectedTemplate(); }, [clearSelectedTemplate]);

  const handleInstantiate = useCallback(async (name: string, config: Record<string, unknown>) => {
    if (!selectedTemplate) return;
    try {
      const result = await instantiateTemplate(selectedTemplate.slug, { name, config, provider_connections: {} });
      if (result.missing_integrations?.length > 0) {
        toast.warning(`Workflow created! You need to connect ${result.missing_integrations.length} integration(s).`);
      } else {
        toast.success('Workflow created successfully!');
      }
      navigate(`/workflows/${result.workflow_id}`);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to create workflow');
    }
  }, [selectedTemplate, instantiateTemplate, navigate]);

  const handleCopyLink = useCallback((template: TemplateSummary) => {
    const url = `https://getseer.dev/templates/${template.slug}`;
    navigator.clipboard.writeText(url);
    toast.success('Link copied to clipboard');
  }, []);

  const handleEditSubmit = useCallback(async (slug: string, request: TemplateUpdateRequest) => {
    await updateTemplate(slug, request);
    toast.success('Template updated');
    setEditingTemplate(null);
    loadMyTemplates().catch(console.error);
  }, [updateTemplate, loadMyTemplates]);

  const handleDeleteConfirm = useCallback(async () => {
    if (!deletingTemplate) return;
    await deleteTemplate(deletingTemplate.slug);
    toast.success('Template deleted');
    setDeletingTemplate(null);
    loadMyTemplates().catch(console.error);
  }, [deletingTemplate, deleteTemplate, loadMyTemplates]);

  return (
    <div className="h-full overflow-y-auto scrollbar-thin">
      <div className="p-6 max-w-5xl mx-auto">
        {view === 'gallery' && (
          <div className="mb-6">
            <h1 className="text-2xl font-semibold">Workflow Templates</h1>
            <p className="text-muted-foreground text-sm mt-1">Browse and create workflows from pre-built templates</p>
          </div>
        )}

        {view === 'gallery' ? (
          <div className="space-y-6">
            <CategoryFilter
              categories={categories}
              selectedCategory={selectedCategory}
              searchQuery={searchDebounce}
              onCategoryChange={setSelectedCategory}
              onSearchChange={setSearchDebounce}
              isLoading={false}
            />
            <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as 'mine' | 'community')}>
              <TabsList>
                <TabsTrigger value="mine">My Templates</TabsTrigger>
                <TabsTrigger value="community">Community</TabsTrigger>
              </TabsList>
              <TabsContent value="mine" className="mt-4">
                <MyTemplatesTable
                  templates={myTemplates}
                  searchQuery={searchDebounce}
                  isLoading={isLoadingTemplates}
                  error={error}
                  onTemplateClick={handleTemplateClick}
                  onEdit={setEditingTemplate}
                  onDelete={setDeletingTemplate}
                  onCopyLink={handleCopyLink}
                />
              </TabsContent>
              <TabsContent value="community" className="mt-4">
                <GalleryContent
                  templates={communityTemplates}
                  searchQuery={searchDebounce}
                  isLoading={isLoadingTemplates}
                  error={error}
                  onTemplateClick={handleTemplateClick}
                />
              </TabsContent>
            </Tabs>
          </div>
        ) : (
          <>
            {isLoadingDetail && (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="w-8 h-8 animate-spin text-muted-foreground" />
              </div>
            )}
            {!isLoadingDetail && selectedTemplate && (
              <TemplateDetailView
                template={selectedTemplate}
                requirements={requirements}
                isLoadingRequirements={isLoadingRequirements}
                isInstantiating={isInstantiating}
                onBack={handleBack}
                onInstantiate={handleInstantiate}
              />
            )}
          </>
        )}
      </div>

      <EditTemplateDialog
        open={!!editingTemplate}
        onOpenChange={(open) => { if (!open) setEditingTemplate(null); }}
        template={editingTemplate as unknown as TemplateAdminResponse}
        onSubmit={handleEditSubmit}
        isUpdating={isUpdatingTemplate}
      />

      <DeleteTemplateDialog
        open={!!deletingTemplate}
        onOpenChange={(open) => { if (!open) setDeletingTemplate(null); }}
        templateName={deletingTemplate?.name ?? ''}
        onConfirm={handleDeleteConfirm}
        isDeleting={isDeletingTemplate}
      />
    </div>
  );
}
