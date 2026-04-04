import { BrowserProfileCard } from "@/components/browser/BrowserProfileCard";

export default function Browser() {
  return (
    <div className="h-full overflow-y-auto scrollbar-thin">
      <div className="p-6 max-w-5xl space-y-6">
        <div>
          <h1 className="text-2xl font-semibold">Browser</h1>
          <p className="text-muted-foreground text-sm mt-1">Manage your browser profiles</p>
        </div>
        <BrowserProfileCard />
      </div>
    </div>
  );
}
