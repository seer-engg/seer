import { useState, useCallback, useEffect, useMemo } from 'react';
import { Check, Copy, ExternalLink, Sparkles } from 'lucide-react';
import { toast } from '@/components/ui/sonner';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { backendApiClient } from '@/lib/api-client';
import { TEMPLATE_CATEGORIES } from '@/types/templates';
import { useWorkflowStore } from '@/stores/workflowStore';
import { useCanvasStore } from '@/stores/canvasStore';

interface ShareAsTemplateDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  workflowId: string;
  workflowName: string;
}

interface ShareResponse {
  slug: string;
  name: string;
  public_url: string;
  is_update: boolean;
}

interface TemplateMeta {
  slug: string;
  name: string;
  description: string;
  category: string;
  tags: string[];
  icon: string | null;
  public_url: string;
}

interface TemplateFormState {
  name: string;
  description: string;
  category: string;
  tags: string[];
  icon: string;
}

const TAG_SUGGESTIONS = [
  'gmail', 'slack', 'sheets', 'ai', 'automation',
  'email', 'reporting', 'data', 'social-media', 'crm',
] as const;

function SuccessView({
  result,
  isUpdate,
  onClose,
}: {
  result: ShareResponse;
  isUpdate: boolean;
  onClose: () => void;
}) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async () => {
    await navigator.clipboard.writeText(result.public_url);
    setCopied(true);
    toast.success('URL copied to clipboard');
    setTimeout(() => setCopied(false), 2000);
  }, [result.public_url]);

  return (
    <>
      <DialogHeader>
        <div className="mx-auto mb-2 flex h-12 w-12 items-center justify-center rounded-full bg-emerald-100 dark:bg-emerald-900/30">
          <Check className="h-6 w-6 text-emerald-600" />
        </div>
        <DialogTitle className="text-center">
          {isUpdate ? 'Template Updated!' : 'Template Published!'}
        </DialogTitle>
        <DialogDescription className="text-center">
          {isUpdate
            ? 'Your template has been updated with the latest changes.'
            : 'Your workflow is now available as a public template.'}
        </DialogDescription>
      </DialogHeader>
      <div className="flex items-center gap-2">
        <code className="flex-1 rounded bg-muted px-3 py-2 text-xs break-all">
          {result.public_url}
        </code>
        <Button size="icon" variant="outline" onClick={handleCopy}>
          {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
        </Button>
        <Button size="icon" variant="outline" asChild>
          <a href={result.public_url} target="_blank" rel="noopener noreferrer">
            <ExternalLink className="h-4 w-4" />
          </a>
        </Button>
      </div>
      <DialogFooter>
        <Button onClick={onClose} className="w-full">Done</Button>
      </DialogFooter>
    </>
  );
}

function useGenerateTemplateDescription(
  workflowId: string,
  onFormChange: (patch: Partial<TemplateFormState>) => void,
) {
  const [isGenerating, setIsGenerating] = useState(false);

  const generate = useCallback(async () => {
    setIsGenerating(true);
    try {
      const result = await backendApiClient.request<{
        name: string;
        description: string;
        category: string;
        tags: string[];
      }>(`/api/v1/workflows/${workflowId}/generate-template-description`, { method: 'POST' });
      onFormChange({
        name: result.name,
        description: result.description,
        category: result.category,
        tags: result.tags,
      });
      toast.success('Template details generated');
    } catch {
      toast.error('Failed to generate description');
    } finally {
      setIsGenerating(false);
    }
  }, [workflowId, onFormChange]);

  return { isGenerating, generate };
}

function TemplateForm({
  form,
  onFormChange,
  onPublish,
  onCancel,
  isPublishing,
  isUpdate,
  workflowId,
}: {
  form: TemplateFormState;
  onFormChange: (patch: Partial<TemplateFormState>) => void;
  onPublish: () => void;
  onCancel: () => void;
  isPublishing: boolean;
  isUpdate: boolean;
  workflowId: string;
}) {
  const { isGenerating, generate: handleGenerate } = useGenerateTemplateDescription(workflowId, onFormChange);

  const toggleTag = (tag: string) => {
    const next = form.tags.includes(tag)
      ? form.tags.filter((t) => t !== tag)
      : [...form.tags, tag];
    onFormChange({ tags: next });
  };

  return (
    <>
      <DialogHeader>
        <DialogTitle>{isUpdate ? 'Update Template' : 'Share as Template'}</DialogTitle>
        <DialogDescription>
          {isUpdate
            ? 'Update your published template with the latest workflow changes.'
            : 'Share this workflow as a template others can use.'}
        </DialogDescription>
      </DialogHeader>
      <div className="grid gap-4 py-2">
        <div className="flex items-center justify-between">
          <Button
            variant="ghost"
            size="sm"
            className="text-seer hover:text-seer/80 gap-1.5 px-2"
            onClick={handleGenerate}
            disabled={isGenerating}
          >
            <Sparkles className={`h-4 w-4 ${isGenerating ? 'animate-spin' : ''}`} />
            {isGenerating ? 'Generating...' : 'Generate with AI'}
          </Button>
        </div>
        <div className="grid gap-2">
          <Label htmlFor="template-name">Template Name</Label>
          <Input id="template-name" value={form.name} onChange={(e) => onFormChange({ name: e.target.value })} />
        </div>
        <div className="grid gap-2">
          <Label htmlFor="template-description">Description</Label>
          <Textarea id="template-description" rows={3} value={form.description} onChange={(e) => onFormChange({ description: e.target.value })} placeholder={isGenerating ? 'Generating...' : ''} />
        </div>
        <div className="grid gap-2">
          <Label>Category</Label>
          <Select value={form.category} onValueChange={(v) => onFormChange({ category: v })}>
            <SelectTrigger><SelectValue placeholder="Select a category" /></SelectTrigger>
            <SelectContent>
              {TEMPLATE_CATEGORIES.map((cat) => (<SelectItem key={cat.value} value={cat.value}>{cat.label}</SelectItem>))}
            </SelectContent>
          </Select>
        </div>
        <div className="grid gap-2">
          <Label>Tags</Label>
          <div className="flex flex-wrap gap-1.5">
            {TAG_SUGGESTIONS.map((tag) => (
              <Badge key={tag} variant={form.tags.includes(tag) ? 'default' : 'outline'} className="cursor-pointer" onClick={() => toggleTag(tag)}>
                {tag}
              </Badge>
            ))}
          </div>
        </div>
        <div className="grid gap-2">
          <Label htmlFor="template-icon">Icon (emoji)</Label>
          <Input id="template-icon" className="w-16" value={form.icon} onChange={(e) => onFormChange({ icon: e.target.value })} />
        </div>
      </div>
      <DialogFooter className="gap-2 sm:gap-0">
        <Button variant="outline" onClick={onCancel}>Cancel</Button>
        <Button onClick={onPublish} disabled={isPublishing || !form.name.trim()}>
          {isPublishing
            ? (isUpdate ? 'Updating...' : 'Publishing...')
            : (isUpdate ? 'Update Template' : 'Publish Template')}
        </Button>
      </DialogFooter>
    </>
  );
}

export function ShareAsTemplateDialog({ open, onOpenChange, workflowId, workflowName }: ShareAsTemplateDialogProps) {
  const defaultForm: TemplateFormState = useMemo(() => ({ name: workflowName, description: '', category: '', tags: [], icon: '' }), [workflowName]);
  const [form, setForm] = useState<TemplateFormState>(defaultForm);
  const [isPublishing, setIsPublishing] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<ShareResponse | null>(null);
  const [existingTemplate, setExistingTemplate] = useState<TemplateMeta | null>(null);

  // Load existing template data when dialog opens
  useEffect(() => {
    if (!open) return;
    let cancelled = false;

    async function loadExisting() {
      setIsLoading(true);
      try {
        const meta = await backendApiClient.request<TemplateMeta>(
          `/api/v1/workflows/${workflowId}/template`,
          { method: 'GET' },
        );
        if (cancelled) return;
        setExistingTemplate(meta);
        setForm({
          name: meta.name,
          description: meta.description,
          category: meta.category,
          tags: meta.tags,
          icon: meta.icon || '',
        });
      } catch {
        // 404 = no existing template, use defaults
        if (cancelled) return;
        setExistingTemplate(null);
        setForm({ name: workflowName, description: '', category: '', tags: [], icon: '' });
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    }

    loadExisting();
    return () => { cancelled = true; };
  }, [open, workflowId, workflowName]);

  const resetAndClose = useCallback(() => {
    setForm(defaultForm);
    setResult(null);
    setExistingTemplate(null);
    onOpenChange(false);
  }, [defaultForm, onOpenChange]);

  const handleOpenChange = useCallback((next: boolean) => {
    if (!next) resetAndClose();
    else onOpenChange(true);
  }, [resetAndClose, onOpenChange]);

  const saveWorkflowDraft = useWorkflowStore((state) => state.saveWorkflowDraft);

  const handlePublish = useCallback(async () => {
    setIsPublishing(true);
    try {
      // Flush canvas to backend so template gets the latest spec
      const { nodes, edges } = useCanvasStore.getState();
      await saveWorkflowDraft(workflowId, { graph: { nodes, edges } });
      const response = await backendApiClient.request<ShareResponse>(
        `/api/v1/workflows/${workflowId}/share-as-template`,
        { method: 'POST', body: { ...form } },
      );
      setResult(response);
      toast.success(response.is_update ? 'Template updated successfully' : 'Template published successfully');
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Failed to publish template');
    } finally {
      setIsPublishing(false);
    }
  }, [workflowId, form, saveWorkflowDraft]);

  const patchForm = useCallback((patch: Partial<TemplateFormState>) => {
    setForm((prev) => ({ ...prev, ...patch }));
  }, []);

  const isUpdate = existingTemplate !== null;

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-md">
        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <div className="h-6 w-6 animate-spin rounded-full border-2 border-primary border-t-transparent" />
          </div>
        ) : result ? (
          <SuccessView result={result} isUpdate={result.is_update} onClose={resetAndClose} />
        ) : (
          <TemplateForm form={form} onFormChange={patchForm} onPublish={handlePublish} onCancel={resetAndClose} isPublishing={isPublishing} isUpdate={isUpdate} workflowId={workflowId} />
        )}
      </DialogContent>
    </Dialog>
  );
}
