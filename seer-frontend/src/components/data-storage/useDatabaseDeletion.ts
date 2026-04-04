import { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { deleteIntegrationResource, type IntegrationResource } from '@/lib/api-client';
import { postgresKeys } from '@/lib/query-keys';
import { useToast } from '@/hooks/utility/use-toast';

export function useDatabaseDeletion() {
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [resourceToDelete, setResourceToDelete] = useState<IntegrationResource | null>(null);

  const queryClient = useQueryClient();
  const { toast } = useToast();

  const deleteMutation = useMutation({
    mutationFn: deleteIntegrationResource,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: postgresKeys.bindings() });
      toast({
        title: 'Database removed',
        description: `${resourceToDelete?.name || 'Database'} has been disconnected`,
      });
      setDeleteDialogOpen(false);
      setResourceToDelete(null);
    },
    onError: (error) => {
      toast({
        title: 'Failed to remove database',
        description: error instanceof Error ? error.message : 'An error occurred',
        variant: 'destructive',
      });
    },
  });

  const handleDeleteClick = (resource: IntegrationResource) => {
    setResourceToDelete(resource);
    setDeleteDialogOpen(true);
  };

  const handleConfirmDelete = () => {
    if (resourceToDelete) {
      deleteMutation.mutate(resourceToDelete.id);
    }
  };

  return {
    deleteDialogOpen,
    setDeleteDialogOpen,
    resourceToDelete,
    handleDeleteClick,
    handleConfirmDelete,
    isPending: deleteMutation.isPending,
  };
}
