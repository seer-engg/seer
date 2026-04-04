import { useState, useEffect } from 'react';
import { Loader2 } from 'lucide-react';

import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Checkbox } from '@/components/ui/checkbox';
import { Alert, AlertDescription } from '@/components/ui/alert';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { TEMPLATE_CATEGORIES } from '@/types/templates';
import type { TemplateAdminResponse, TemplateUpdateRequest } from '@/types/templates';

export interface EditTemplateDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  template: TemplateAdminResponse | null;
  onSubmit: (slug: string, request: TemplateUpdateRequest) => Promise<void>;
  isUpdating?: boolean;
}

interface EditFormState {
  slug: string;
  name: string;
  description: string;
  category: string;
  tags: string;
  icon: string;
  isFeatured: boolean;
}

function useEditTemplateForm(template: TemplateAdminResponse | null, open: boolean) {
  const [form, setForm] = useState<EditFormState>({
    slug: '',
    name: '',
    description: '',
    category: '',
    tags: '',
    icon: '',
    isFeatured: false,
  });
  const [error, setError] = useState<string | null>(null);
  const [slugChanged, setSlugChanged] = useState(false);

  useEffect(() => {
    if (template) {
      setForm({
        slug: template.slug,
        name: template.name,
        description: template.description,
        category: template.category,
        tags: template.tags?.join(', ') || '',
        icon: template.icon || '',
        isFeatured: template.is_featured,
      });
      setSlugChanged(false);
    }
  }, [template]);

  useEffect(() => {
    if (!open) {
      setError(null);
      setSlugChanged(false);
    }
  }, [open]);

  const updateField = <K extends keyof EditFormState>(key: K, value: EditFormState[K]) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const handleSlugChange = (value: string) => {
    const newSlug = value.toLowerCase().replace(/[^a-z0-9-]/g, '');
    updateField('slug', newSlug);
    setSlugChanged(newSlug !== template?.slug);
  };

  return { form, error, setError, slugChanged, updateField, handleSlugChange };
}

export function EditTemplateDialog({
  open,
  onOpenChange,
  template,
  onSubmit,
  isUpdating = false,
}: EditTemplateDialogProps) {
  const { form, error, setError, slugChanged, updateField, handleSlugChange } =
    useEditTemplateForm(template, open);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    if (!template) return;

    if (!form.slug || !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(form.slug)) {
      return setError('Please enter a valid slug (lowercase letters, numbers, and hyphens)');
    }
    if (!form.name.trim()) return setError('Please enter a template name');
    if (!form.description.trim()) return setError('Please enter a description');
    if (!form.category) return setError('Please select a category');

    const request: TemplateUpdateRequest = {
      slug: slugChanged ? form.slug : undefined,
      name: form.name.trim(),
      description: form.description.trim(),
      category: form.category,
      tags: form.tags ? form.tags.split(',').map((t) => t.trim()).filter(Boolean) : [],
      icon: form.icon.trim() || undefined,
      is_featured: form.isFeatured,
    };

    try {
      await onSubmit(template.slug, request);
      onOpenChange(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update template');
    }
  };

  if (!template) return null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Edit Template</DialogTitle>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="edit-slug">Template Slug *</Label>
            <Input id="edit-slug" value={form.slug} onChange={(e) => handleSlugChange(e.target.value)} />
            {slugChanged && <p className="text-xs text-warning">Changing the slug will update the template URL</p>}
          </div>

          <div className="space-y-2">
            <Label htmlFor="edit-name">Name *</Label>
            <Input id="edit-name" value={form.name} onChange={(e) => updateField('name', e.target.value)} />
          </div>

          <div className="space-y-2">
            <Label htmlFor="edit-description">Description *</Label>
            <Textarea id="edit-description" value={form.description} onChange={(e) => updateField('description', e.target.value)} rows={3} />
          </div>

          <div className="space-y-2">
            <Label htmlFor="edit-category">Category *</Label>
            <Select value={form.category} onValueChange={(v) => updateField('category', v)}>
              <SelectTrigger id="edit-category"><SelectValue placeholder="Select a category..." /></SelectTrigger>
              <SelectContent>
                {TEMPLATE_CATEGORIES.map((cat) => (<SelectItem key={cat.value} value={cat.value}>{cat.label}</SelectItem>))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="edit-tags">Tags (comma-separated)</Label>
            <Input id="edit-tags" value={form.tags} onChange={(e) => updateField('tags', e.target.value)} placeholder="email, notifications, alerts" />
          </div>

          <div className="space-y-2">
            <Label htmlFor="edit-icon">Icon (emoji)</Label>
            <Input id="edit-icon" value={form.icon} onChange={(e) => updateField('icon', e.target.value)} placeholder="📧" className="w-24" />
          </div>

          <div className="flex items-center space-x-2">
            <Checkbox id="edit-featured" checked={form.isFeatured} onCheckedChange={(checked) => updateField('isFeatured', checked === true)} />
            <label htmlFor="edit-featured" className="text-sm font-medium cursor-pointer">Feature this template</label>
          </div>

          {error && <Alert variant="destructive"><AlertDescription>{error}</AlertDescription></Alert>}

          <div className="flex justify-end gap-2 pt-2">
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>Cancel</Button>
            <Button type="submit" disabled={isUpdating}>
              {isUpdating ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" />Saving...</> : 'Save Changes'}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}
