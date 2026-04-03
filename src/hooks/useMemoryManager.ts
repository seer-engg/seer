import { useState, useEffect, useCallback, useMemo } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";

import { useToast } from "@/hooks/utility/use-toast";
import { memoryApi, type Memory, type MemoryBank } from "@/lib/memory-api";
import { memoryKeys } from "@/lib/query-keys";

type MemoryFormMode = "create" | "edit";
type BankFormMode = "create" | "edit";

function getDefaultBankId(banks: MemoryBank[]): string | null {
  return banks.find((bank) => bank.is_default)?.memory_bank_id ?? banks[0]?.memory_bank_id ?? null;
}

export function useMemoryManager() {
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const [selectedBankId, setSelectedBankId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [debouncedQuery, setDebouncedQuery] = useState("");

  const [memoryFormOpen, setMemoryFormOpen] = useState(false);
  const [memoryFormMode, setMemoryFormMode] = useState<MemoryFormMode>("create");
  const [memoryToEdit, setMemoryToEdit] = useState<Memory | null>(null);
  const [memoryToDelete, setMemoryToDelete] = useState<Memory | null>(null);
  const [showDeleteAllDialog, setShowDeleteAllDialog] = useState(false);
  const [deletingMemoryId, setDeletingMemoryId] = useState<string | null>(null);
  const [savingMemoryId, setSavingMemoryId] = useState<string | null>(null);

  const [bankFormOpen, setBankFormOpen] = useState(false);
  const [bankFormMode, setBankFormMode] = useState<BankFormMode>("create");
  const [bankToEdit, setBankToEdit] = useState<MemoryBank | null>(null);
  const [bankToDelete, setBankToDelete] = useState<MemoryBank | null>(null);
  const [settingDefaultBankId, setSettingDefaultBankId] = useState<string | null>(null);

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedQuery(searchQuery.trim());
    }, 300);

    return () => clearTimeout(timer);
  }, [searchQuery]);

  const statsQuery = useQuery({
    queryKey: memoryKeys.stats(),
    queryFn: memoryApi.getStats,
  });

  const featureEnabled = statsQuery.data?.memory_enabled === true;

  const banksQuery = useQuery({
    queryKey: memoryKeys.banks(),
    queryFn: memoryApi.listBanks,
    enabled: featureEnabled,
  });

  const banks = useMemo(() => banksQuery.data?.items ?? [], [banksQuery.data?.items]);

  useEffect(() => {
    if (!featureEnabled || banks.length === 0) {
      setSelectedBankId(null);
      return;
    }

    if (!selectedBankId || !banks.some((bank) => bank.memory_bank_id === selectedBankId)) {
      setSelectedBankId(getDefaultBankId(banks));
    }
  }, [banks, featureEnabled, selectedBankId]);

  const selectedBank = useMemo(
    () => banks.find((bank) => bank.memory_bank_id === selectedBankId) ?? null,
    [banks, selectedBankId]
  );

  const memoriesQuery = useQuery({
    queryKey: memoryKeys.items(selectedBankId, debouncedQuery),
    queryFn: () =>
      debouncedQuery
        ? memoryApi.searchMemories(debouncedQuery, selectedBankId ?? undefined)
        : memoryApi.listMemories(selectedBankId ?? undefined),
    enabled: featureEnabled && Boolean(selectedBankId),
  });

  const invalidateMemoryQueries = useCallback(async () => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: memoryKeys.all }),
      queryClient.invalidateQueries({ queryKey: memoryKeys.banks() }),
    ]);
  }, [queryClient]);

  const deleteMutation = useMutation({
    mutationFn: ({ memoryId, memoryBankId }: { memoryId: string; memoryBankId: string }) =>
      memoryApi.deleteMemory(memoryId, memoryBankId),
    onSuccess: async () => {
      await invalidateMemoryQueries();
      toast({
        title: "Memory deleted",
        description: "The memory has been removed.",
      });
      setDeletingMemoryId(null);
      setMemoryToDelete(null);
    },
    onError: (error: Error) => {
      toast({
        title: "Failed to delete memory",
        description: error.message,
        variant: "destructive",
      });
      setDeletingMemoryId(null);
    },
  });

  const createMutation = useMutation({
    mutationFn: ({ memory, memoryBankId }: { memory: string; memoryBankId: string }) =>
      memoryApi.createMemory(memory, memoryBankId),
    onSuccess: async () => {
      await invalidateMemoryQueries();
      toast({
        title: "Memory created",
        description: "The memory has been added.",
      });
      setSavingMemoryId(null);
      setMemoryFormOpen(false);
    },
    onError: (error: Error) => {
      toast({
        title: "Failed to create memory",
        description: error.message,
        variant: "destructive",
      });
      setSavingMemoryId(null);
    },
  });

  const updateMutation = useMutation({
    mutationFn: ({
      memoryId,
      memory,
      memoryBankId,
    }: {
      memoryId: string;
      memory: string;
      memoryBankId: string;
    }) => memoryApi.updateMemory(memoryId, memory, memoryBankId),
    onSuccess: async () => {
      await invalidateMemoryQueries();
      toast({
        title: "Memory updated",
        description: "The memory has been updated.",
      });
      setSavingMemoryId(null);
      setMemoryFormOpen(false);
      setMemoryToEdit(null);
    },
    onError: (error: Error) => {
      toast({
        title: "Failed to update memory",
        description: error.message,
        variant: "destructive",
      });
      setSavingMemoryId(null);
    },
  });

  const deleteAllMutation = useMutation({
    mutationFn: (memoryBankId: string) => memoryApi.deleteAllMemories(memoryBankId),
    onSuccess: async () => {
      await invalidateMemoryQueries();
      toast({
        title: "All memories deleted",
        description: "This memory bank has been completely cleared.",
      });
      setShowDeleteAllDialog(false);
    },
    onError: (error: Error) => {
      toast({
        title: "Failed to delete memories",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const createBankMutation = useMutation({
    mutationFn: memoryApi.createBank,
    onSuccess: async (bank) => {
      await invalidateMemoryQueries();
      setSelectedBankId(bank.memory_bank_id);
      setBankFormOpen(false);
      setBankToEdit(null);
      toast({
        title: "Memory bank created",
        description: `${bank.name} is ready.`,
      });
    },
    onError: (error: Error) => {
      toast({
        title: "Failed to create memory bank",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const updateBankMutation = useMutation({
    mutationFn: ({
      memoryBankId,
      name,
      description,
    }: {
      memoryBankId: string;
      name: string;
      description?: string | null;
    }) => memoryApi.updateBank(memoryBankId, { name, description }),
    onSuccess: async (bank) => {
      await invalidateMemoryQueries();
      setBankFormOpen(false);
      setBankToEdit(null);
      toast({
        title: "Memory bank updated",
        description: `${bank.name} has been updated.`,
      });
    },
    onError: (error: Error) => {
      toast({
        title: "Failed to update memory bank",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const deleteBankMutation = useMutation({
    mutationFn: (memoryBankId: string) => memoryApi.deleteBank(memoryBankId),
    onSuccess: async () => {
      const deletedId = bankToDelete?.memory_bank_id ?? null;
      await invalidateMemoryQueries();
      setBankToDelete(null);
      if (deletedId && selectedBankId === deletedId) {
        setSelectedBankId(getDefaultBankId(banks.filter((bank) => bank.memory_bank_id !== deletedId)));
      }
      toast({
        title: "Memory bank deleted",
        description: "The memory bank has been removed.",
      });
    },
    onError: (error: Error) => {
      toast({
        title: "Failed to delete memory bank",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const setDefaultBankMutation = useMutation({
    mutationFn: (memoryBankId: string) => memoryApi.setDefaultBank(memoryBankId),
    onSuccess: async (bank) => {
      await invalidateMemoryQueries();
      setSettingDefaultBankId(null);
      setSelectedBankId(bank.memory_bank_id);
      toast({
        title: "Default memory bank updated",
        description: `${bank.name} is now the workspace default.`,
      });
    },
    onError: (error: Error) => {
      setSettingDefaultBankId(null);
      toast({
        title: "Failed to update default bank",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const handleDeleteMemory = useCallback((memory: Memory) => {
    setMemoryToDelete(memory);
  }, []);

  const handleAddMemory = useCallback(() => {
    setMemoryFormMode("create");
    setMemoryToEdit(null);
    setMemoryFormOpen(true);
  }, []);

  const handleEditMemory = useCallback((memory: Memory) => {
    setMemoryFormMode("edit");
    setMemoryToEdit(memory);
    setMemoryFormOpen(true);
  }, []);

  const handleConfirmDelete = useCallback(() => {
    if (memoryToDelete && selectedBankId) {
      setDeletingMemoryId(memoryToDelete.id);
      deleteMutation.mutate({ memoryId: memoryToDelete.id, memoryBankId: selectedBankId });
    }
  }, [deleteMutation, memoryToDelete, selectedBankId]);

  const handleConfirmDeleteAll = useCallback(() => {
    if (selectedBankId) {
      deleteAllMutation.mutate(selectedBankId);
    }
  }, [deleteAllMutation, selectedBankId]);

  const handleCloseDeleteDialog = useCallback((open: boolean) => {
    if (!open) setMemoryToDelete(null);
  }, []);

  const handleMemoryFormOpenChange = useCallback((open: boolean) => {
    setMemoryFormOpen(open);
    if (!open) {
      setMemoryToEdit(null);
      setSavingMemoryId(null);
      setMemoryFormMode("create");
    }
  }, []);

  const handleSubmitMemory = useCallback(
    (memory: string) => {
      const normalizedMemory = memory.trim();
      if (!selectedBankId) {
        return;
      }

      if (memoryFormMode === "edit" && memoryToEdit) {
        setSavingMemoryId(memoryToEdit.id);
        updateMutation.mutate({
          memoryId: memoryToEdit.id,
          memory: normalizedMemory,
          memoryBankId: selectedBankId,
        });
        return;
      }

      setSavingMemoryId("new");
      createMutation.mutate({ memory: normalizedMemory, memoryBankId: selectedBankId });
    },
    [createMutation, memoryFormMode, memoryToEdit, selectedBankId, updateMutation]
  );

  const handleAddBank = useCallback(() => {
    setBankFormMode("create");
    setBankToEdit(null);
    setBankFormOpen(true);
  }, []);

  const handleEditBank = useCallback((bank: MemoryBank) => {
    setBankFormMode("edit");
    setBankToEdit(bank);
    setBankFormOpen(true);
  }, []);

  const handleBankFormOpenChange = useCallback((open: boolean) => {
    setBankFormOpen(open);
    if (!open) {
      setBankFormMode("create");
      setBankToEdit(null);
    }
  }, []);

  const handleSubmitBank = useCallback(
    ({ name, description }: { name: string; description?: string | null }) => {
      if (bankFormMode === "edit" && bankToEdit) {
        updateBankMutation.mutate({
          memoryBankId: bankToEdit.memory_bank_id,
          name,
          description,
        });
        return;
      }

      createBankMutation.mutate({ name, description });
    },
    [bankFormMode, bankToEdit, createBankMutation, updateBankMutation]
  );

  const handleDeleteBank = useCallback((bank: MemoryBank) => {
    setBankToDelete(bank);
  }, []);

  const handleDeleteBankDialogChange = useCallback((open: boolean) => {
    if (!open) {
      setBankToDelete(null);
    }
  }, []);

  const handleConfirmDeleteBank = useCallback(() => {
    if (bankToDelete) {
      deleteBankMutation.mutate(bankToDelete.memory_bank_id);
    }
  }, [bankToDelete, deleteBankMutation]);

  const handleSetDefaultBank = useCallback(
    (memoryBankId: string) => {
      setSettingDefaultBankId(memoryBankId);
      setDefaultBankMutation.mutate(memoryBankId);
    },
    [setDefaultBankMutation]
  );

  return {
    searchQuery,
    setSearchQuery,
    debouncedQuery,

    stats: statsQuery.data,
    statsLoading: statsQuery.isLoading,
    statsError: statsQuery.error,
    featureEnabled,

    banks,
    banksLoading: banksQuery.isLoading,
    banksError: banksQuery.error,
    selectedBankId,
    selectedBank,
    setSelectedBankId,

    memories: memoriesQuery.data?.memories ?? [],
    memoriesLoading: memoriesQuery.isLoading,
    memoriesError: memoriesQuery.error,

    memoryFormOpen,
    memoryFormMode,
    memoryToEdit,
    savingMemoryId,
    isSaving: createMutation.isPending || updateMutation.isPending,
    handleAddMemory,
    handleEditMemory,
    handleSubmitMemory,
    handleMemoryFormOpenChange,

    memoryToDelete,
    deletingMemoryId,
    isDeleting: deleteMutation.isPending,
    handleDeleteMemory,
    handleConfirmDelete,
    handleCloseDeleteDialog,

    showDeleteAllDialog,
    setShowDeleteAllDialog,
    isDeletingAll: deleteAllMutation.isPending,
    handleConfirmDeleteAll,

    bankFormOpen,
    bankFormMode,
    bankToEdit,
    isSavingBank: createBankMutation.isPending || updateBankMutation.isPending,
    handleAddBank,
    handleEditBank,
    handleBankFormOpenChange,
    handleSubmitBank,

    bankToDelete,
    isDeletingBank: deleteBankMutation.isPending,
    handleDeleteBank,
    handleDeleteBankDialogChange,
    handleConfirmDeleteBank,

    settingDefaultBankId,
    handleSetDefaultBank,
  };
}
