import { useState, useMemo } from 'react';
import { Link, useLocation } from 'react-router-dom';
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarMenuSkeleton,
  SidebarTrigger,
  SidebarFooter,
  SidebarRail,
  // SidebarFooter,
  useSidebar,
} from '@/components/ui/sidebar';
import { LayoutDashboard, LayoutTemplate, Settings, PanelLeftOpen, FileText, Zap, Database, MessageSquare, Globe, Moon, Sun, Monitor, Building2, User, SmilePlus, Video } from 'lucide-react';
import { SHOW_MEETINGS } from '@/lib/feature-flags';
import { useTheme } from 'next-themes';
import { useCurrentUser } from '@/hooks/useAuthProvider';
import { SeerLogo } from '@/components/icons/seer-logo';
import { Avatar, AvatarImage, AvatarFallback } from '@/components/ui/avatar';
import { useOrganizationStore } from '@/stores/organizationStore';
import { OrganizationSwitcher } from '@/components/organization';
import { useWorkflowListQuery } from '@/hooks/useWorkflowQueries';


function RecentWorkflows() {
  const location = useLocation();
  const { data: workflows = [], isLoading, isError, isSuccess } = useWorkflowListQuery();

  const recentWorkflows = useMemo(() => {
    return [...workflows]
      .sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime())
      .slice(0, 3);
  }, [workflows]);

  return (
    <SidebarGroup>
      <SidebarGroupLabel>Recent Workflows</SidebarGroupLabel>
      <SidebarGroupContent>
        <SidebarMenu>
          {isLoading && !isSuccess && (
            <>
              <SidebarMenuSkeleton showIcon />
              <SidebarMenuSkeleton showIcon />
              <SidebarMenuSkeleton showIcon />
            </>
          )}
          {isError && (
            <p className="text-xs text-muted-foreground px-2 py-1 group-data-[collapsible=icon]:hidden">
              Failed to load workflows
            </p>
          )}
          {isSuccess && workflows.length === 0 && (
            <p className="text-xs text-muted-foreground px-2 py-1 group-data-[collapsible=icon]:hidden">
              No workflows yet
            </p>
          )}
          {isSuccess && recentWorkflows.map((workflow) => {
            const isActive = location.pathname.startsWith(`/workflows/${workflow.workflow_id}`);
            return (
              <SidebarMenuItem key={workflow.workflow_id}>
                <SidebarMenuButton asChild isActive={isActive} tooltip={workflow.name}>
                  <Link to={`/workflows/${workflow.workflow_id}`}>
                    <FileText />
                    <span>{workflow.name}</span>
                  </Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
            );
          })}
        </SidebarMenu>
      </SidebarGroupContent>
    </SidebarGroup>
  );
}

type ThemeOption = "dark" | "light" | "system";
const themeOrder: ThemeOption[] = ["dark", "light", "system"];

function ThemeToggleItem() {
  const { theme, setTheme } = useTheme();
  const ThemeIcon = theme === "light" ? Sun : theme === "system" ? Monitor : Moon;

  const cycleTheme = () => {
    const idx = themeOrder.indexOf((theme as ThemeOption) || "dark");
    setTheme(themeOrder[(idx + 1) % themeOrder.length]);
  };

  return (
    <SidebarMenuItem>
      <SidebarMenuButton onClick={cycleTheme} tooltip="Toggle theme">
        <ThemeIcon />
        <span>Theme</span>
      </SidebarMenuButton>
    </SidebarMenuItem>
  );
}

const navItems = [
  { title: 'Home', url: '/', icon: LayoutDashboard },
  { title: 'Chat', url: '/chat', icon: MessageSquare },
  { title: 'Templates', url: '/templates', icon: LayoutTemplate },
  { title: 'Triggers', url: '/triggers', icon: Zap },
  { title: 'Data', url: '/data', icon: Database },
  { title: 'Browser', url: '/browser', icon: Globe },
  ...(SHOW_MEETINGS ? [{ title: 'Meetings', url: '/meetings', icon: Video }] : []),
];

function CollapsedOrgIcon({ onClick, onMouseEnter, onMouseLeave, showExpandIcon }: { onClick: () => void; onMouseEnter: () => void; onMouseLeave: () => void; showExpandIcon: boolean }) {
  const { user } = useCurrentUser();
  const currentOrganization = useOrganizationStore((s) => s.currentOrganization);

  if (showExpandIcon) {
    return <PanelLeftOpen className="h-6 w-6 text-primary flex-shrink-0" />;
  }

  if (currentOrganization?.type === 'personal' && user?.imageUrl) {
    return (
      <Avatar className="h-6 w-6 flex-shrink-0">
        <AvatarImage src={user.imageUrl} alt={user.fullName || 'User'} />
        <AvatarFallback><User className="h-4 w-4" /></AvatarFallback>
      </Avatar>
    );
  }

  if (currentOrganization?.type === 'team') {
    return <Building2 className="h-6 w-6 text-primary flex-shrink-0" />;
  }

  return <User className="h-6 w-6 text-primary flex-shrink-0" />;
}

export function SeerSidebar() {
  const location = useLocation();
  const { state, toggleSidebar } = useSidebar();
  const [showExpandIcon, setShowExpandIcon] = useState(false);

  return (
    <Sidebar collapsible="icon">
      <SidebarHeader className="border-b p-0 overflow-visible">
        {state === "expanded" ? (
          <div className="flex flex-row items-center justify-between px-2 py-1.5">
            <div className="flex-1 min-w-0">
              <OrganizationSwitcher />
            </div>
            <SidebarTrigger />
          </div>
        ) : (
          <div
            className="flex items-center justify-center w-full min-h-[52px] cursor-pointer hover:bg-sidebar-accent/50 transition-colors"
            onClick={toggleSidebar}
            onMouseEnter={() => setShowExpandIcon(true)}
            onMouseLeave={() => setShowExpandIcon(false)}
            title="Expand sidebar"
          >
            <CollapsedOrgIcon onClick={toggleSidebar} onMouseEnter={() => setShowExpandIcon(true)} onMouseLeave={() => setShowExpandIcon(false)} showExpandIcon={showExpandIcon} />
          </div>
        )}
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Navigation</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {navItems.map((item) => {
                const Icon = item.icon;
                const isActive = location.pathname === item.url;
                return (
                  <SidebarMenuItem key={item.url}>
                    <SidebarMenuButton asChild isActive={isActive} tooltip={item.title}>
                      <Link to={item.url}>
                        <Icon />
                        <span>{item.title}</span>
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                );
              })}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
        <RecentWorkflows />
      </SidebarContent>
      <SidebarFooter>
        <SidebarMenu>
          <ThemeToggleItem />
          <SidebarMenuItem>
            <SidebarMenuButton asChild isActive={location.pathname === '/settings'} tooltip="Settings">
              <Link to="/settings"><Settings /><span>Settings</span></Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
          <SidebarMenuItem>
            <SidebarMenuButton asChild tooltip="Help? Call 646-371-6198">
              <a href="tel:+16463716198"><SmilePlus /><span>Help? Call 646-371-6198</span></a>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>
      <SidebarFooter className="border-t p-2">
        <div className="flex items-center gap-2 px-2 py-1">
          <SeerLogo className="h-5 w-5 text-primary flex-shrink-0" />
          <span className="font-semibold text-sm group-data-[collapsible=icon]:hidden">Seer</span>
        </div>
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  );
}
