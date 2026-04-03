import { useState, useEffect, useMemo, useCallback } from 'react';
import { Loader2, Search, Pencil, Trash2 } from 'lucide-react';

import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { toast } from '@/components/ui/sonner';
import { useTemplatesStore } from '@/stores';
import type { TemplateAdminResponse, TemplateUpdateRequest } from '@/types/templates';

import { EditTemplateDialog } from './EditTemplateDialog';
import { DeleteTemplateDialog } from './DeleteTemplateDialog';

export interface TemplateManagerDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

interface TemplateRowProps {
  template: TemplateAdminResponse;
  onEdit: (template: TemplateAdminResponse) => void;
  onDelete: (template: TemplateAdminResponse) => void;
}

function TemplateRow({ template, onEdit, onDelete }: TemplateRowProps) {
  return (
    <div className="flex items-start justify-between p-4 border rounded-lg hover:bg-muted/50 transition-colors">
      <div className="flex-1 min-w-0 mr-4">
        <div className="flex items-center gap-2 mb-1">
          {template.icon && <span className="text-lg">{template.icon}</span>}
          <span className="font-medium truncate">{template.name}</span>
          {template.is_featured && (
            <Badge variant="outline" className="shrink-0 border-yellow-500/50 text-yellow-600">Featured</Badge>
          )}
        </div>
        <p className="text-sm text-muted-foreground line-clamp-2">{template.description}</p>
        <div className="flex items-center gap-2 mt-2 text-xs text-muted-foreground">
          <Badge variant="outline" className="text-xs">{template.category}</Badge>
          <span>·</span>
          <span>{template.usage_count} uses</span>
        </div>
      </div>
      <div className="flex items-center gap-1 shrink-0">
        <Button variant="ghost" size="icon" onClick={() => onEdit(template)} title="Edit"><Pencil className="h-4 w-4" /></Button>
        <Button variant="ghost" size="icon" onClick={() => onDelete(template)} className="text-destructive hover:text-destructive" title="Delete"><Trash2 className="h-4 w-4" /></Button>
      </div>
    </div>
  );
}

function LoadingSkeleton() {
  return (
    <div className="space-y-3">
      {Array.from({ length: 3 }).map((_, i) => (<Skeleton key={i} className="h-24 w-full" />))}
    </div>
  );
}

function EmptyState({ hasSearch }: { hasSearch: boolean }) {
  return (
    <div className="text-center py-12">
      <p className="text-muted-foreground">
        {hasSearch ? 'No templates match your search.' : 'No templates yet. Create your first template!'}
      </p>
    </div>
  );
}

function useTemplateManager(open: boolean) {
  const adminTemplates = useTemplatesStore((state) => state.adminTemplates);
  const isLoadingAdminTemplates = useTemplatesStore((state) => state.isLoadingAdminTemplates);
  const isUpdatingTemplate = useTemplatesStore((state) => state.isUpdatingTemplate);
  const isDeletingTemplate = useTemplatesStore((state) => state.isDeletingTemplate);
  const loadAdminTemplates = useTemplatesStore((state) => state.loadAdminTemplates);
  const updateTemplate = useTemplatesStore((state) => state.updateTemplate);
  const deleteTemplate = useTemplatesStore((state) => state.deleteTemplate);

  useEffect(() => {
    if (open) {
      loadAdminTemplates().catch((err) => {
        console.error('Failed to load admin templates:', err);
        toast.error('Failed to load templates');
      });
    }
  }, [open, loadAdminTemplates]);

  const handleUpdate = useCallback(async (slug: string, request: TemplateUpdateRequest) => {
    await updateTemplate(slug, request);
    toast.success('Template updated successfully!');
  }, [updateTemplate]);

  const handleDelete = useCallback(async (slug: string) => {
    await deleteTemplate(slug);
    toast.success('Template deleted successfully!');
  }, [deleteTemplate]);

  return {
    adminTemplates, isLoadingAdminTemplates, isUpdatingTemplate, isDeletingTemplate,
    handleUpdate, handleDelete,
  };
}

export function TemplateManagerDialog({ open, onOpenChange }: TemplateManagerDialogProps) {
  const {
    adminTemplates, isLoadingAdminTemplates, isUpdatingTemplate, isDeletingTemplate,
    handleUpdate, handleDelete,
  } = useTemplateManager(open);

  const [searchQuery, setSearchQuery] = useState('');
  const [editingTemplate, setEditingTemplate] = useState<TemplateAdminResponse | null>(null);
  const [deletingTemplate, setDeletingTemplate] = useState<TemplateAdminResponse | null>(null);

  useEffect(() => {
    if (!open) {
      setSearchQuery('');
      setEditingTemplate(null);
      setDeletingTemplate(null);
    }
  }, [open]);

  const filteredTemplates = useMemo(() => {
    if (!searchQuery) return adminTemplates;
    const query = searchQuery.toLowerCase();
    return adminTemplates.filter((t) => t.name.toLowerCase().includes(query) || t.description.toLowerCase().includes(query) || t.slug.toLowerCase().includes(query));
  }, [adminTemplates, searchQuery]);

  const onConfirmDelete = async () => {
    if (!deletingTemplate) return;
    await handleDelete(deletingTemplate.slug);
  };

  return (
    <>
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent className="max-w-3xl max-h-[85vh]">
          <DialogHeader>
            <div className="flex items-center justify-between pr-8">
              <DialogTitle>Manage Templates</DialogTitle>
            </div>
          </DialogHeader>
          <div className="space-y-4">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input placeholder="Search templates..." value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} className="pl-9" />
            </div>
            <div className="max-h-[calc(85vh-200px)] overflow-y-auto space-y-3">
              {isLoadingAdminTemplates ? <LoadingSkeleton /> : filteredTemplates.length === 0 ? <EmptyState hasSearch={!!searchQuery} /> : (
                filteredTemplates.map((template) => (
                  <TemplateRow key={template.template_id} template={template} onEdit={setEditingTemplate} onDelete={setDeletingTemplate} />
                ))
              )}
            </div>
            {(isUpdatingTemplate || isDeletingTemplate) && (
              <div className="flex items-center justify-center text-sm text-muted-foreground"><Loader2 className="h-4 w-4 animate-spin mr-2" />Processing...</div>
            )}
          </div>
        </DialogContent>
      </Dialog>

      <EditTemplateDialog open={!!editingTemplate} onOpenChange={(open) => !open && setEditingTemplate(null)} template={editingTemplate} onSubmit={handleUpdate} isUpdating={isUpdatingTemplate} />
      <DeleteTemplateDialog open={!!deletingTemplate} onOpenChange={(open) => !open && setDeletingTemplate(null)} templateName={deletingTemplate?.name || ''} onConfirm={onConfirmDelete} isDeleting={isDeletingTemplate} />
    </>
  );
}
