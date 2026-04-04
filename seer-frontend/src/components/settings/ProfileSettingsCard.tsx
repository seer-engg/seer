import { useState, useEffect, useRef, useCallback } from "react";
import { backendApiClient } from "@/lib/api-client";
import { filesApi } from "@/lib/files-api";
import { toast } from "sonner";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { Loader2, Check, X, ExternalLink, Camera } from "lucide-react";

interface UserProfile {
  username: string;
  display_name: string;
  bio: string;
  avatar_url: string;
  avatar_file_id: string;
  twitter_handle: string;
  linkedin_url: string;
  website_url: string;
  interest_tags: string[];
  is_public: boolean;
}

type UsernameStatus = "idle" | "checking" | "available" | "taken";

const SUGGESTED_TAGS = [
  "Productivity", "Marketing", "Sales", "Customer Support",
  "Engineering", "HR", "Legal", "Finance", "No-Code/Automation",
];

function UsernameStatusIcon({ status }: { status: UsernameStatus }) {
  if (status === "checking") return <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />;
  if (status === "available") return <Check className="h-4 w-4 text-green-500" />;
  if (status === "taken") return <X className="h-4 w-4 text-red-500" />;
  return null;
}

function UsernameField({ value, status, onChange }: {
  value: string; status: UsernameStatus; onChange: (v: string) => void;
}) {
  return (
    <div className="space-y-2">
      <Label htmlFor="username">Username</Label>
      <div className="relative">
        <Input id="username" value={value} onChange={(e) => onChange(e.target.value)} placeholder="your-username" className="pr-9" />
        <div className="absolute right-3 top-1/2 -translate-y-1/2">
          <UsernameStatusIcon status={status} />
        </div>
      </div>
      <p className="text-xs text-muted-foreground">getseer.dev/u/{value || "your-username"}</p>
    </div>
  );
}

function SocialLinksFields({ profile, onChange }: {
  profile: UserProfile; onChange: (field: keyof UserProfile, value: string) => void;
}) {
  return (
    <div className="space-y-4">
      <Label>Social Links</Label>
      <div className="grid gap-3">
        <Input value={profile.twitter_handle} onChange={(e) => onChange("twitter_handle", e.target.value)} placeholder="Twitter handle (e.g. @username)" />
        <Input value={profile.linkedin_url} onChange={(e) => onChange("linkedin_url", e.target.value)} placeholder="LinkedIn URL" />
        <Input value={profile.website_url} onChange={(e) => onChange("website_url", e.target.value)} placeholder="Website URL" />
      </div>
    </div>
  );
}

function InterestTagsField({ tags, onToggle, onAdd }: {
  tags: string[]; onToggle: (tag: string) => void; onAdd: (tag: string) => void;
}) {
  const [customTag, setCustomTag] = useState("");
  return (
    <div className="space-y-3">
      <Label>Interests</Label>
      <div className="flex flex-wrap gap-2">
        {SUGGESTED_TAGS.map((tag) => (
          <Badge key={tag} variant={(tags ?? []).includes(tag) ? "default" : "outline"} className="cursor-pointer" onClick={() => onToggle(tag)}>
            {tag}
          </Badge>
        ))}
        {(tags ?? []).filter((t) => !SUGGESTED_TAGS.includes(t)).map((tag) => (
          <Badge key={tag} variant="default" className="cursor-pointer" onClick={() => onToggle(tag)}>
            {tag} <X className="h-3 w-3 ml-1" />
          </Badge>
        ))}
      </div>
      <Input
        value={customTag}
        onChange={(e) => setCustomTag(e.target.value)}
        onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); const t = customTag.trim(); if (t) { onAdd(t); setCustomTag(""); } } }}
        placeholder="Add custom tag and press Enter"
        className="max-w-xs"
      />
    </div>
  );
}

function AvatarUpload({ avatarUrl, displayName, onUploaded }: {
  avatarUrl: string; displayName: string; onUploaded: (fileId: string, previewUrl: string) => void;
}) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);

  const initials = (displayName || "?")
    .split(" ")
    .map((w) => w[0])
    .slice(0, 2)
    .join("")
    .toUpperCase();

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    try {
      const previewUrl = URL.createObjectURL(file);
      const uploaded = await filesApi.uploadFile(file, file.name);
      onUploaded(uploaded.file_id, previewUrl);
    } catch {
      toast.error("Failed to upload avatar");
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  return (
    <div className="flex items-center gap-4">
      <button
        type="button"
        onClick={() => fileInputRef.current?.click()}
        className="relative h-24 w-24 shrink-0 rounded-full border-2 border-dashed border-muted-foreground/30 hover:border-muted-foreground/60 transition-colors overflow-hidden group cursor-pointer"
      >
        {uploading ? (
          <div className="flex items-center justify-center h-full">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        ) : avatarUrl ? (
          <img src={avatarUrl} alt="Avatar" className="h-full w-full object-cover" />
        ) : (
          <div className="flex items-center justify-center h-full bg-muted text-muted-foreground text-xl font-semibold">
            {initials}
          </div>
        )}
        <div className="absolute inset-0 bg-black/50 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
          <Camera className="h-5 w-5 text-white" />
        </div>
      </button>
      <div className="text-sm text-muted-foreground">
        <p className="font-medium text-foreground">Profile photo</p>
        <p>Click to upload a new avatar</p>
      </div>
      <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={handleFileSelect} />
    </div>
  );
}

function useProfile() {
  const [profile, setProfile] = useState<UserProfile>({
    username: "", display_name: "", bio: "", avatar_url: "", avatar_file_id: "",
    twitter_handle: "", linkedin_url: "", website_url: "", interest_tags: [], is_public: false,
  });
  const [saving, setSaving] = useState(false);
  const [loading, setLoading] = useState(true);
  const [usernameStatus, setUsernameStatus] = useState<UsernameStatus>("idle");
  const debounceRef = useRef<ReturnType<typeof setTimeout>>();

  useEffect(() => {
    backendApiClient.request<UserProfile>("/api/users/me/profile")
      .then((data) => {
        const social = data.social_links as Record<string, string | null> | undefined;
        setProfile({
          username: data.username ?? "",
          display_name: data.display_name ?? "",
          bio: data.bio ?? "",
          avatar_url: (data as Record<string, unknown>).avatar_url as string ?? "",
          avatar_file_id: (data as Record<string, unknown>).avatar_file_id as string ?? "",
          twitter_handle: social?.twitter ?? "",
          linkedin_url: social?.linkedin ?? "",
          website_url: social?.website ?? "",
          interest_tags: (data as Record<string, unknown>).tags as string[] ?? [],
          is_public: data.is_public ?? false,
        });
        if (data.username) setUsernameStatus("available");
      })
      .catch(() => toast.error("Failed to load profile"))
      .finally(() => setLoading(false));
  }, []);

  const checkUsername = useCallback((username: string) => {
    if (!username) { setUsernameStatus("idle"); return; }
    setUsernameStatus("checking");
    backendApiClient.request<{ available: boolean }>(`/api/public/username-available/${username}`)
      .then((res) => setUsernameStatus(res.available ? "available" : "taken"))
      .catch(() => setUsernameStatus("idle"));
  }, []);

  const handleUsernameChange = (raw: string) => {
    const sanitized = raw.toLowerCase().replace(/[^a-z0-9-]/g, "");
    setProfile((p) => ({ ...p, username: sanitized }));
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => checkUsername(sanitized), 400);
  };

  const updateField = (field: keyof UserProfile, value: string) => setProfile((p) => ({ ...p, [field]: value }));

  const toggleTag = (tag: string) => {
    setProfile((p) => ({
      ...p, interest_tags: p.interest_tags.includes(tag) ? p.interest_tags.filter((t) => t !== tag) : [...p.interest_tags, tag],
    }));
  };

  const addTag = (tag: string) => {
    if (!profile.interest_tags.includes(tag)) setProfile((p) => ({ ...p, interest_tags: [...p.interest_tags, tag] }));
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      const { twitter_handle, linkedin_url, website_url, interest_tags, avatar_url: _avatarUrl, ...rest } = profile;
      await backendApiClient.request("/api/users/me/profile", {
        method: "PATCH",
        body: { ...rest, tags: interest_tags, social_links: { twitter: twitter_handle, linkedin: linkedin_url, website: website_url } },
      });
      toast.success("Profile saved");
    } catch { toast.error("Failed to save profile"); }
    finally { setSaving(false); }
  };

  return { profile, setProfile, saving, loading, usernameStatus, handleUsernameChange, updateField, toggleTag, addTag, handleSave };
}

export function ProfileSettingsCard() {
  const { profile, setProfile, saving, loading, usernameStatus, handleUsernameChange, updateField, toggleTag, addTag, handleSave } = useProfile();

  if (loading) {
    return <Card><CardContent className="flex items-center justify-center py-12"><Loader2 className="h-5 w-5 animate-spin text-muted-foreground" /></CardContent></Card>;
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Public Profile</CardTitle>
        <CardDescription>Manage your public-facing profile on Seer</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <AvatarUpload
          avatarUrl={profile.avatar_url}
          displayName={profile.display_name}
          onUploaded={(fileId, previewUrl) => setProfile((p) => ({ ...p, avatar_file_id: fileId, avatar_url: previewUrl }))}
        />
        <UsernameField value={profile.username} status={usernameStatus} onChange={handleUsernameChange} />
        <div className="space-y-2">
          <Label htmlFor="display_name">Display Name</Label>
          <Input id="display_name" value={profile.display_name} onChange={(e) => updateField("display_name", e.target.value)} placeholder="Your name" />
        </div>
        <div className="space-y-2">
          <Label htmlFor="bio">Bio</Label>
          <Textarea id="bio" rows={3} value={profile.bio} onChange={(e) => updateField("bio", e.target.value)} placeholder="Tell us about yourself" />
        </div>
        <SocialLinksFields profile={profile} onChange={updateField} />
        <InterestTagsField tags={profile.interest_tags} onToggle={toggleTag} onAdd={addTag} />
        <div className="flex items-center justify-between rounded-lg border p-4">
          <div className="space-y-0.5">
            <Label>Make profile public</Label>
            <p className="text-xs text-muted-foreground">Allow others to discover and view your profile</p>
          </div>
          <Switch checked={profile.is_public} onCheckedChange={(checked) => setProfile((p) => ({ ...p, is_public: checked }))} />
        </div>
        <div className="flex items-center gap-3">
          <Button onClick={handleSave} disabled={saving}>
            {saving && <Loader2 className="h-4 w-4 animate-spin mr-2" />}Save
          </Button>
          {profile.is_public && profile.username && (
            <a href={`https://getseer.dev/u/${profile.username}`} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors">
              View public profile <ExternalLink className="h-3.5 w-3.5" />
            </a>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
