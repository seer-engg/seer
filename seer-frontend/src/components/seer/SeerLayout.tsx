import { SidebarProvider, SidebarInset } from "@/components/ui/sidebar";
import { SeerSidebar } from "./SeerSidebar";

export function SeerLayout({ children, defaultOpen = true }: { children: React.ReactNode; defaultOpen?: boolean }) {
  return (
    <SidebarProvider defaultOpen={defaultOpen}>
      <div className="min-h-screen bg-background flex w-full">
        <SeerSidebar />
        <SidebarInset className="relative">
          <main className="w-full h-screen overflow-auto">{children}</main>
        </SidebarInset>
      </div>
    </SidebarProvider>
  );
}
