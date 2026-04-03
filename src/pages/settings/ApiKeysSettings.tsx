import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useBYOKStore } from "@/stores/byokStore";
import { Key, Plus, Trash2, CheckCircle2, XCircle, Loader2, Plug } from "lucide-react";
import type { LLMApiKey, LLMProvider, TestConnectionResult } from "@/lib/byok-api";

const PROVIDER_OPTIONS: { value: LLMProvider; label: string; needsBaseUrl: boolean; defaultBaseUrl: string }[] = [
  { value: "openrouter", label: "OpenRouter", needsBaseUrl: false, defaultBaseUrl: "https://openrouter.ai/api/v1" },
  { value: "openai", label: "OpenAI", needsBaseUrl: false, defaultBaseUrl: "https://api.openai.com/v1" },
  { value: "anthropic", label: "Anthropic", needsBaseUrl: false, defaultBaseUrl: "https://api.anthropic.com" },
  { value: "google", label: "Google AI", needsBaseUrl: false, defaultBaseUrl: "https://generativelanguage.googleapis.com/v1beta" },
  { value: "custom", label: "Custom (OpenAI-compatible)", needsBaseUrl: true, defaultBaseUrl: "" },
];

function detectProviderFromUrl(url: string): LLMProvider | null {
  if (url.includes("openrouter.ai")) return "openrouter";
  if (url.includes("api.openai.com")) return "openai";
  if (url.includes("api.anthropic.com")) return "anthropic";
  if (url.includes("generativelanguage.googleapis.com")) return "google";
  return null;
}

interface TestResultBannerProps {
  testResult: TestConnectionResult | null;
}

function TestResultBanner({ testResult }: TestResultBannerProps) {
  if (!testResult) return null;
  return (
    <div className={`text-sm p-3 rounded-md ${testResult.success ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400" : "bg-red-500/10 text-red-600 dark:text-red-400"}`}>
      {testResult.success ? (
        <div>
          <div className="flex items-center gap-1 font-medium">
            <CheckCircle2 className="h-4 w-4" /> Connection successful
          </div>
          {testResult.models.length > 0 && (
            <p className="mt-1 text-xs opacity-80">{testResult.models.length} models available</p>
          )}
        </div>
      ) : (
        <div className="flex items-center gap-1">
          <XCircle className="h-4 w-4" /> {testResult.error}
        </div>
      )}
    </div>
  );
}

interface KeyRowProps {
  apiKey: LLMApiKey;
  onDelete: (id: number) => void;
}

function KeyRow({ apiKey: key, onDelete }: KeyRowProps) {
  return (
    <div className="flex items-center justify-between p-3 border rounded-lg">
      <div className="flex items-center gap-3">
        <Key className="h-4 w-4 text-muted-foreground" />
        <div>
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium">{key.label}</span>
            {key.is_active && (
              <Badge variant="secondary" className="h-4 px-1 text-[9px] bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20">
                Active
              </Badge>
            )}
          </div>
          <div className="text-xs text-muted-foreground">
            {key.key_masked} · {key.provider} · {key.base_url}
          </div>
        </div>
      </div>
      <Button variant="ghost" size="icon" onClick={() => onDelete(key.id)} className="h-8 w-8 text-muted-foreground hover:text-destructive">
        <Trash2 className="h-4 w-4" />
      </Button>
    </div>
  );
}

interface AddKeyFormProps {
  onCancel: () => void;
  onSaved: () => void;
}

function AddKeyForm({ onCancel, onSaved }: AddKeyFormProps) {
  const { testResult, addKey, testConnection } = useBYOKStore();
  const [label, setLabel] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [provider, setProvider] = useState<LLMProvider>("openrouter");
  const [baseUrl, setBaseUrl] = useState("https://openrouter.ai/api/v1");
  const [submitting, setSubmitting] = useState(false);
  const [testing, setTesting] = useState(false);

  const selectedProviderConfig = PROVIDER_OPTIONS.find((p) => p.value === provider);
  const showBaseUrl = provider === "custom";

  const handleProviderChange = (value: string) => {
    const newProvider = value as LLMProvider;
    setProvider(newProvider);
    const config = PROVIDER_OPTIONS.find((p) => p.value === newProvider);
    if (config) {
      setBaseUrl(config.defaultBaseUrl);
      if (!label.trim()) {
        setLabel(config.label);
      }
    }
  };

  const handleBaseUrlChange = (url: string) => {
    setBaseUrl(url);
    const detected = detectProviderFromUrl(url);
    if (detected && detected !== provider) {
      setProvider(detected);
    }
  };

  const handleAdd = async () => {
    if (!label.trim() || !apiKey.trim()) return;
    setSubmitting(true);
    try {
      await addKey(label.trim(), apiKey.trim(), provider, baseUrl.trim() || undefined);
      onSaved();
    } finally {
      setSubmitting(false);
    }
  };

  const handleTest = async () => {
    setTesting(true);
    try {
      await testConnection(apiKey.trim() || undefined, baseUrl.trim() || undefined);
    } finally {
      setTesting(false);
    }
  };

  return (
    <div className="border rounded-lg p-4 space-y-3 bg-muted/30">
      <div className="space-y-2">
        <label className="text-sm font-medium">Provider</label>
        <Select value={provider} onValueChange={handleProviderChange}>
          <SelectTrigger>
            <SelectValue placeholder="Select provider" />
          </SelectTrigger>
          <SelectContent>
            {PROVIDER_OPTIONS.map((opt) => (
              <SelectItem key={opt.value} value={opt.value}>
                {opt.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <div className="space-y-2">
        <label className="text-sm font-medium">Label</label>
        <Input placeholder={`e.g., ${selectedProviderConfig?.label ?? "Provider"} Key`} value={label} onChange={(e) => setLabel(e.target.value)} />
      </div>
      <div className="space-y-2">
        <label className="text-sm font-medium">API Key</label>
        <Input type="password" placeholder="sk-..." value={apiKey} onChange={(e) => setApiKey(e.target.value)} />
      </div>
      {showBaseUrl && (
        <div className="space-y-2">
          <label className="text-sm font-medium">Base URL</label>
          <Input placeholder="https://your-provider.com/v1" value={baseUrl} onChange={(e) => handleBaseUrlChange(e.target.value)} />
        </div>
      )}
      <div className="flex gap-2">
        <Button onClick={handleAdd} disabled={submitting || !label.trim() || !apiKey.trim()} size="sm">
          {submitting ? <Loader2 className="h-4 w-4 animate-spin mr-1" /> : <Plus className="h-4 w-4 mr-1" />}
          Save Key
        </Button>
        <Button variant="outline" size="sm" onClick={handleTest} disabled={testing || !apiKey.trim()}>
          {testing ? <Loader2 className="h-4 w-4 animate-spin mr-1" /> : <Plug className="h-4 w-4 mr-1" />}
          Test Connection
        </Button>
        <Button variant="ghost" size="sm" onClick={onCancel}>Cancel</Button>
      </div>
      <TestResultBanner testResult={testResult} />
    </div>
  );
}

export function ApiKeysSettings() {
  const { keys, loading, fetchKeys, deleteKey } = useBYOKStore();
  const [showForm, setShowForm] = useState(false);

  useEffect(() => {
    fetchKeys();
  }, [fetchKeys]);

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Key className="h-5 w-5" />
                LLM API Keys
              </CardTitle>
              <CardDescription>
                Bring your own API key to use your LLM provider directly. Your key is encrypted at rest.
              </CardDescription>
            </div>
            {!showForm && (
              <Button variant="outline" size="sm" onClick={() => setShowForm(true)}>
                <Plus className="h-4 w-4 mr-1" />
                Add Key
              </Button>
            )}
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {showForm && <AddKeyForm onCancel={() => setShowForm(false)} onSaved={() => setShowForm(false)} />}
          {loading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
            </div>
          ) : keys.length === 0 ? (
            <div className="text-center py-8 text-sm text-muted-foreground">
              No API keys configured. Add one to use your own LLM provider.
            </div>
          ) : (
            <div className="space-y-2">
              {keys.map((key) => (
                <KeyRow key={key.id} apiKey={key} onDelete={deleteKey} />
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
