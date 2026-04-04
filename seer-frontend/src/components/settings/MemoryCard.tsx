import { Brain, Database, Pencil, Plus, Search, Star, Trash2 } from "lucide-react";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useMemoryManager } from "@/hooks/useMemoryManager";
import { useOrganizationStore } from "@/stores/organizationStore";
import { MemoryRow } from "./MemoryRow";
import { MemoryEmptyState } from "./MemoryEmptyState";
import { MemoryStatsSection } from "./MemoryStatsSection";
import { DeleteMemoryDialog } from "./DeleteMemoryDialog";
import { DeleteAllMemoriesDialog } from "./DeleteAllMemoriesDialog";
import { MemoryFormDialog } from "./MemoryFormDialog";
import { MemoryBankFormDialog } from "./MemoryBankFormDialog";
import { DeleteMemoryBankDialog } from "./DeleteMemoryBankDialog";

type MemoryManager = ReturnType<typeof useMemoryManager>;
type SelectedMemoryBank = NonNullable<MemoryManager["selectedBank"]>;

function CardHeaderContent({
  isLoading,
  featureEnabled,
}: {
  isLoading?: boolean;
  featureEnabled?: boolean;
}) {
  return (
    <CardHeader>
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-lg bg-seer/10 flex items-center justify-center">
          <Brain className="h-5 w-5 text-seer" />
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <CardTitle className="text-base">Memory</CardTitle>
            {featureEnabled === false && (
              <Badge
                variant="secondary"
                className="h-5 px-1.5 text-[10px] bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20"
              >
                Disabled
              </Badge>
            )}
          </div>
          <CardDescription>
            {isLoading
              ? "Loading..."
              : "Manage workspace memory banks and the memories stored inside them"}
          </CardDescription>
        </div>
      </div>
    </CardHeader>
  );
}

function LoadingSkeleton() {
  return (
    <div className="space-y-2">
      {Array.from({ length: 3 }).map((_, i) => (
        <div key={i} className="flex items-start gap-3 py-3 px-3">
          <div className="flex-1 space-y-2">
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-3 w-20" />
          </div>
          <Skeleton className="h-7 w-7 rounded-md" />
        </div>
      ))}
    </div>
  );
}

function MemoryList({
  mgr,
}: {
  mgr: MemoryManager;
}) {
  if (mgr.memoriesLoading) {
    return <LoadingSkeleton />;
  }

  if (mgr.memories.length === 0) {
    return (
      <MemoryEmptyState
        hasSearch={mgr.debouncedQuery.length > 0}
        onAdd={mgr.handleAddMemory}
      />
    );
  }

  return (
    <div className="space-y-1 max-h-[400px] overflow-y-auto scrollbar-thin">
      {mgr.memories.map((memory) => (
        <MemoryRow
          key={memory.id}
          memory={memory}
          onEdit={() => mgr.handleEditMemory(memory)}
          onDelete={() => mgr.handleDeleteMemory(memory)}
          isMutating={mgr.deletingMemoryId === memory.id || mgr.savingMemoryId === memory.id}
        />
      ))}
    </div>
  );
}

function MemoryErrorState({
  featureEnabled,
  message,
}: {
  featureEnabled?: boolean;
  message?: string;
}) {
  return (
    <Card>
      <CardHeaderContent featureEnabled={featureEnabled} />
      <CardContent>
        <div className="py-6 text-center">
          <p className="text-sm text-destructive">Failed to load memory settings</p>
          <p className="text-xs text-muted-foreground mt-1">{message || "Please try again later"}</p>
        </div>
      </CardContent>
    </Card>
  );
}

function SelectedBankActions({
  mgr,
  selectedBank,
}: {
  mgr: MemoryManager;
  selectedBank: SelectedMemoryBank;
}) {
  return (
    <div className="flex flex-wrap gap-2">
      {!selectedBank.is_default && (
        <Button
          variant="outline"
          size="sm"
          onClick={() => mgr.handleSetDefaultBank(selectedBank.memory_bank_id)}
          disabled={mgr.settingDefaultBankId === selectedBank.memory_bank_id}
        >
          <Star className="mr-2 h-4 w-4" />
          Make Default
        </Button>
      )}
      <Button variant="outline" size="sm" onClick={() => mgr.handleEditBank(selectedBank)}>
        <Pencil className="mr-2 h-4 w-4" />
        Edit
      </Button>
      {!selectedBank.is_default && (
        <Button variant="outline" size="sm" onClick={() => mgr.handleDeleteBank(selectedBank)}>
          <Trash2 className="mr-2 h-4 w-4" />
          Delete
        </Button>
      )}
    </div>
  );
}

function SelectedBankSummary({
  mgr,
  canManageBanks,
  selectedBankCount,
}: {
  mgr: MemoryManager;
  canManageBanks: boolean;
  selectedBankCount: number;
}) {
  if (!mgr.selectedBank) {
    return null;
  }

  return (
    <div className="flex flex-wrap items-start justify-between gap-3 rounded-md bg-muted/30 p-3">
      <div className="min-w-0 space-y-1">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-sm font-medium">{mgr.selectedBank.name}</span>
          {mgr.selectedBank.is_default && (
            <Badge variant="secondary" className="h-5 px-1.5 text-[10px]">
              <Star className="mr-1 h-3 w-3" />
              Default
            </Badge>
          )}
          <Badge variant="outline" className="h-5 px-1.5 text-[10px]">
            <Database className="mr-1 h-3 w-3" />
            {selectedBankCount} memories
          </Badge>
        </div>
        {mgr.selectedBank.description && (
          <p className="text-xs text-muted-foreground">{mgr.selectedBank.description}</p>
        )}
      </div>

      {canManageBanks && <SelectedBankActions mgr={mgr} selectedBank={mgr.selectedBank} />}
    </div>
  );
}

function MemoryBankSection({
  mgr,
  canManageBanks,
  selectedBankCount,
}: {
  mgr: MemoryManager;
  canManageBanks: boolean;
  selectedBankCount: number;
}) {
  return (
    <div className="space-y-3 rounded-lg border p-3">
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="text-sm font-medium">Memory bank</p>
          <p className="text-xs text-muted-foreground">Choose which bank to inspect and edit.</p>
        </div>
        {canManageBanks && (
          <Button variant="outline" size="sm" onClick={mgr.handleAddBank}>
            <Plus className="mr-2 h-4 w-4" />
            New Bank
          </Button>
        )}
      </div>

      {mgr.banksLoading ? (
        <Skeleton className="h-10 w-full" />
      ) : mgr.banks.length === 0 ? (
        <div className="rounded-md border border-dashed p-3 text-sm text-muted-foreground">
          No memory banks found.
        </div>
      ) : (
        <>
          <Select
            value={mgr.selectedBankId ?? mgr.banks[0]?.memory_bank_id ?? ""}
            onValueChange={mgr.setSelectedBankId}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select a memory bank" />
            </SelectTrigger>
            <SelectContent>
              {mgr.banks.map((bank) => (
                <SelectItem key={bank.memory_bank_id} value={bank.memory_bank_id}>
                  {bank.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <SelectedBankSummary
            mgr={mgr}
            canManageBanks={canManageBanks}
            selectedBankCount={selectedBankCount}
          />
        </>
      )}
    </div>
  );
}

function MemoryContent({
  mgr,
}: {
  mgr: MemoryManager;
}) {
  if (!mgr.selectedBankId) {
    return (
      <div className="rounded-lg border border-dashed p-4 text-sm text-muted-foreground">
        Create a memory bank to start storing reusable context.
      </div>
    );
  }

  return (
    <>
      <div className="flex justify-end">
        <Button variant="outline" onClick={mgr.handleAddMemory}>
          <Plus className="h-4 w-4 mr-2" />
          Add Memory
        </Button>
      </div>

      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
        <Input
          placeholder="Search memories semantically..."
          value={mgr.searchQuery}
          onChange={(e) => mgr.setSearchQuery(e.target.value)}
          className="pl-9"
        />
      </div>

      <MemoryList mgr={mgr} />

      {mgr.memories.length > 0 && (
        <Button
          variant="destructive"
          className="w-full"
          onClick={() => mgr.setShowDeleteAllDialog(true)}
        >
          <Trash2 className="h-4 w-4 mr-2" />
          Delete All Memories In This Bank
        </Button>
      )}
    </>
  );
}

function MemoryCardDialogs({
  mgr,
  selectedBankCount,
}: {
  mgr: MemoryManager;
  selectedBankCount: number;
}) {
  return (
    <>
      <DeleteMemoryDialog
        open={mgr.memoryToDelete !== null}
        onOpenChange={mgr.handleCloseDeleteDialog}
        memory={mgr.memoryToDelete}
        onConfirm={mgr.handleConfirmDelete}
        isDeleting={mgr.isDeleting}
      />

      <MemoryFormDialog
        open={mgr.memoryFormOpen}
        mode={mgr.memoryFormMode}
        memory={mgr.memoryToEdit}
        onOpenChange={mgr.handleMemoryFormOpenChange}
        onSubmit={mgr.handleSubmitMemory}
        isSaving={mgr.isSaving}
      />

      <DeleteAllMemoriesDialog
        open={mgr.showDeleteAllDialog}
        onOpenChange={mgr.setShowDeleteAllDialog}
        totalCount={selectedBankCount}
        onConfirm={mgr.handleConfirmDeleteAll}
        isDeleting={mgr.isDeletingAll}
      />

      <MemoryBankFormDialog
        open={mgr.bankFormOpen}
        mode={mgr.bankFormMode}
        bank={mgr.bankToEdit}
        onOpenChange={mgr.handleBankFormOpenChange}
        onSubmit={mgr.handleSubmitBank}
        isSaving={mgr.isSavingBank}
      />

      <DeleteMemoryBankDialog
        open={mgr.bankToDelete !== null}
        onOpenChange={mgr.handleDeleteBankDialogChange}
        bank={mgr.bankToDelete}
        onConfirm={mgr.handleConfirmDeleteBank}
        isDeleting={mgr.isDeletingBank}
      />
    </>
  );
}

function MemoryCardBody({
  mgr,
  canManageBanks,
  selectedBankCount,
}: {
  mgr: MemoryManager;
  canManageBanks: boolean;
  selectedBankCount: number;
}) {
  return (
    <Card>
      <CardHeaderContent featureEnabled={mgr.stats?.memory_enabled} />
      <CardContent className="space-y-4">
        {mgr.stats && (
          <MemoryStatsSection
            stats={{
              ...mgr.stats,
              total_memories: mgr.selectedBank ? selectedBankCount : mgr.stats.total_memories,
            }}
          />
        )}

        {!mgr.featureEnabled ? (
          <div className="rounded-lg border border-dashed p-4 text-sm text-muted-foreground">
            Memory is currently disabled for this workspace. Existing banks stay intact, but bank and memory management is unavailable until the feature is enabled again.
          </div>
        ) : (
          <>
            <MemoryBankSection
              mgr={mgr}
              canManageBanks={canManageBanks}
              selectedBankCount={selectedBankCount}
            />
            <MemoryContent mgr={mgr} />
          </>
        )}
      </CardContent>
    </Card>
  );
}

export function MemoryCard() {
  const mgr = useMemoryManager();
  const currentRole = useOrganizationStore((state) => state.getCurrentRole());
  const canManageBanks = currentRole === "owner" || currentRole === "admin";

  const hasError = mgr.statsError || (mgr.featureEnabled && (mgr.banksError || mgr.memoriesError));
  const selectedBankCount = mgr.selectedBank?.memory_count ?? 0;

  if (mgr.statsLoading) {
    return (
      <Card>
        <CardHeaderContent isLoading />
      </Card>
    );
  }

  if (hasError) {
    return (
      <MemoryErrorState
        featureEnabled={mgr.stats?.memory_enabled}
        message={(mgr.statsError || mgr.banksError || mgr.memoriesError)?.message}
      />
    );
  }

  return (
    <>
      <MemoryCardBody
        mgr={mgr}
        canManageBanks={canManageBanks}
        selectedBankCount={selectedBankCount}
      />
      <MemoryCardDialogs mgr={mgr} selectedBankCount={selectedBankCount} />
    </>
  );
}
