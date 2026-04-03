import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { useUser } from "@clerk/clerk-react";
import { useQueryState } from "@/hooks/utility/useQueryState";
import { ProfileCard, BackendUrlCard, LogoutCard } from "@/components/settings/SettingsCards";
import { GroupedConnectionsCard } from "@/components/settings/GroupedConnectionsCard";
import { BillingCard } from "@/components/settings/BillingCard";
import { UsageCard } from "@/components/settings/UsageCard";
import { CostCapCard } from "@/components/settings/CostCapCard";
import { TimezoneCard } from "@/components/settings/TimezoneCard";
import { WhatsAppCard } from "@/components/settings/WhatsAppCard";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  TeamSettingsCard,
  MembersSection,
  WorkflowTransferSection,
  SharedIntegrationsCard,
  ApprovalsSection,
} from "@/components/team";
import { CreateTeamDialog } from "@/components/organization";
import { useOrganizationStore } from "@/stores/organizationStore";
import { MemoryCard } from "@/components/settings/MemoryCard"; // Temporarily hidden - memory tab
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";

import { CreditCard, Shield, BarChart3, Users, Loader2, Plus, Building2, UserPlus, Share2, UserCircle, Variable, Key } from "lucide-react";

import { Brain } from "lucide-react"; // Temporarily hidden - memory tab
import type { LucideIcon } from "lucide-react";
import { UsageAnalyticsTab } from "@/components/settings/analytics/UsageAnalyticsTab";
import { ProfileSettingsCard } from "@/components/settings/ProfileSettingsCard";
import { DefaultModelCard } from "@/components/settings/DefaultModelCard";
import { GlobalVariablesCard } from "@/components/settings/GlobalVariablesCard";
import { ApiKeysSettings } from "@/pages/settings/ApiKeysSettings";
import { McpServersCard } from "@/components/settings/McpServersCard";

interface SettingsTab {
  value: string;
  label: string;
  icon: LucideIcon;
  /** Tab only shows in team organizations */
  teamOnly?: boolean;
  /** Tab only shows in personal workspace */
  personalOnly?: boolean;
}

const ALL_SETTINGS_TABS: SettingsTab[] = [
  { value: "profile", label: "Profile", icon: UserCircle },
  { value: "team", label: "Team", icon: Users }, // Always visible - shows "Create Team" when in personal
  { value: "billing", label: "Billing & Usage", icon: CreditCard },
  { value: "api-keys", label: "API Keys", icon: Key },
  { value: "memory", label: "Memory", icon: Brain }, // Temporarily hidden
  { value: "variables", label: "Variables", icon: Variable },
  { value: "integrations", label: "Integrations", icon: Shield },
  { value: "analytics", label: "Analytics", icon: BarChart3 },
];

/**
 * Prompt shown in Team tab when user is in personal workspace.
 * Provides clear CTA to create a team.
 */
function CreateTeamPrompt() {
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);

  return (
    <>
      <Card className="border-dashed">
        <CardHeader className="text-center pb-2">
          <div className="mx-auto mb-4 h-16 w-16 rounded-full bg-seer/10 flex items-center justify-center">
            <Building2 className="h-8 w-8 text-seer" />
          </div>
          <CardTitle>Create Your First Team</CardTitle>
          <CardDescription className="max-w-md mx-auto">
            Teams let you collaborate with others, share workflows, and manage usage together.
          </CardDescription>
        </CardHeader>
        <CardContent className="text-center space-y-4">
          <div className="flex flex-wrap justify-center gap-4 text-sm text-muted-foreground">
            <div className="flex items-center gap-2">
              <UserPlus className="h-4 w-4 text-seer" />
              <span>Invite team members</span>
            </div>
            <div className="flex items-center gap-2">
              <Share2 className="h-4 w-4 text-seer" />
              <span>Share workflows</span>
            </div>
            <div className="flex items-center gap-2">
              <Users className="h-4 w-4 text-seer" />
              <span>Collaborate together</span>
            </div>
          </div>
          <Button
            variant="brand"
            size="lg"
            onClick={() => setIsCreateDialogOpen(true)}
            className="mt-4"
          >
            <Plus className="h-4 w-4" />
            Create Team
          </Button>
        </CardContent>
      </Card>

      <CreateTeamDialog
        open={isCreateDialogOpen}
        onOpenChange={setIsCreateDialogOpen}
      />
    </>
  );
}

/* eslint-disable max-lines-per-function */
export default function Settings() {
  const { user } = useUser();
  const [activeTab, setActiveTab] = useQueryState("tab", { defaultValue: "profile" });
  const currentOrganization = useOrganizationStore((s) => s.currentOrganization);
  const isSwitchingOrg = useOrganizationStore((s) => s.isSwitching);
  const isTeamOrg = currentOrganization?.type === 'team';

  // Filter tabs based on organization context
  const settingsTabs = useMemo(() => {
    return ALL_SETTINGS_TABS.filter((tab) => {
      if (tab.teamOnly && !isTeamOrg) return false;
      if (tab.personalOnly && isTeamOrg) return false;
      return true;
    });
  }, [isTeamOrg]);

  // Handle case where activeTab doesn't exist in filtered tabs
  // (e.g., user was on Team tab but switched to personal workspace)
  const validTab = settingsTabs.some((t) => t.value === activeTab) ? activeTab : "profile";
  if (validTab !== activeTab) {
    setActiveTab(validTab);
  }

  // Show loading state when org is switching
  if (isSwitchingOrg) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="flex flex-col items-center gap-3">
          <Loader2 className="h-8 w-8 animate-spin text-seer" />
          <p className="text-sm text-muted-foreground">Switching workspace...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto scrollbar-thin">
      <div className="p-6 max-w-5xl mx-auto space-y-6">
        <div>
          <h1 className="text-2xl font-semibold">Settings</h1>
          <p className="text-muted-foreground text-sm mt-1">Manage your account and preferences</p>
        </div>

        <Tabs
          value={activeTab}
          onValueChange={setActiveTab}
          orientation="vertical"
          className="flex flex-col md:flex-row gap-6"
        >
          <TabsList className="h-auto w-full md:w-56 md:shrink-0 flex md:flex-col md:sticky md:top-6 md:self-start items-stretch gap-1 bg-transparent p-0 overflow-x-auto scrollbar-thin">
            {settingsTabs.map((tab) => (
              <TabsTrigger
                key={tab.value}
                value={tab.value}
                className="justify-start gap-3 px-3 py-2.5 rounded-lg text-sm font-medium text-muted-foreground transition-colors whitespace-nowrap hover:bg-accent hover:text-accent-foreground data-[state=active]:bg-accent data-[state=active]:text-foreground data-[state=active]:shadow-none md:border-l-2 md:border-l-transparent md:data-[state=active]:border-l-seer md:rounded-l-none"
              >
                <tab.icon className="h-4 w-4 shrink-0" />
                {tab.label}
              </TabsTrigger>
            ))}
          </TabsList>

          <div className="flex-1 min-w-0">
<TabsContent value="team" className="mt-0">
              <motion.div key="team" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.2 }} className="space-y-6">
                {isTeamOrg && currentOrganization ? (
                  <>
                    <TeamSettingsCard />
                    <MembersSection />
                    <ApprovalsSection />
                    <WorkflowTransferSection targetOrg={currentOrganization} />
                    <SharedIntegrationsCard />
                  </>
                ) : (
                  <CreateTeamPrompt />
                )}
              </motion.div>
            </TabsContent>

            <TabsContent value="billing" className="mt-0">
              <motion.div key="billing" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.2 }} className="space-y-6">
                <BillingCard />
                <UsageCard />
                <CostCapCard />
              </motion.div>
            </TabsContent>

            <TabsContent value="api-keys" className="mt-0">
              <motion.div key="api-keys" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.2 }} className="space-y-6">
                <ApiKeysSettings />
              </motion.div>
            </TabsContent>

            <TabsContent value="memory" className="mt-0">
              <motion.div key="memory" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.2 }} className="space-y-6">
                <MemoryCard />
              </motion.div>
            </TabsContent>
           

            <TabsContent value="variables" className="mt-0">
              <motion.div key="variables" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.2 }} className="space-y-6">
                <GlobalVariablesCard />
              </motion.div>
            </TabsContent>

            <TabsContent value="integrations" className="mt-0">
              <motion.div key="integrations" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.2 }} className="space-y-6">
                <GroupedConnectionsCard />
                <McpServersCard />
                <BackendUrlCard />
              </motion.div>
            </TabsContent>

            <TabsContent value="analytics" className="mt-0">
              <motion.div key="analytics" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.2 }} className="space-y-6">
                <UsageAnalyticsTab />
              </motion.div>
            </TabsContent>

            <TabsContent value="profile" className="mt-0">
              <motion.div key="profile" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.2 }} className="space-y-6">
                <ProfileCard user={user} />
                <ProfileSettingsCard />
                <DefaultModelCard />
                <TimezoneCard />
                <WhatsAppCard />
                <LogoutCard />
              </motion.div>
            </TabsContent>
          </div>
        </Tabs>
      </div>
    </div>
  );
}
