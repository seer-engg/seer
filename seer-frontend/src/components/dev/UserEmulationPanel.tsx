import { useEffect, useRef, useState } from "react";
import { backendApiClient } from "@/lib/api-client";

const STORAGE_KEY = "dev_emulate_user_id";

interface UserSummary {
  user_id: string;
  email: string | null;
  first_name: string | null;
  last_name: string | null;
}

function displayName(u: UserSummary): string {
  const name = [u.first_name, u.last_name].filter(Boolean).join(" ");
  return name || u.email || u.user_id;
}

interface EmulationBannerProps {
  currentEmulatedUser: string;
}

function EmulationBanner({ currentEmulatedUser }: EmulationBannerProps) {
  if (!currentEmulatedUser) {
    return null;
  }

  return (
    <div
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        zIndex: 9999,
        backgroundColor: "#dc2626",
        color: "white",
        textAlign: "center",
        padding: "4px 8px",
        fontSize: "12px",
        fontWeight: 600,
        fontFamily: "monospace",
      }}
    >
      DEV: Emulating {currentEmulatedUser}
    </div>
  );
}

interface SearchResultsProps {
  currentEmulatedUser: string;
  loading: boolean;
  results: UserSummary[];
  onSelectUser: (user: UserSummary) => void;
}

function SearchResults({
  currentEmulatedUser,
  loading,
  results,
  onSelectUser,
}: SearchResultsProps) {
  return (
    <div
      style={{
        maxHeight: 200,
        overflowY: "auto",
        borderRadius: 4,
        border: "1px solid #334155",
      }}
    >
      {loading && <div style={{ padding: "6px 8px", opacity: 0.5 }}>Searching…</div>}
      {!loading && results.length === 0 && (
        <div style={{ padding: "6px 8px", opacity: 0.5 }}>No users found</div>
      )}
      {!loading &&
        results.map((user) => (
          <button
            key={user.user_id}
            onClick={() => onSelectUser(user)}
            style={{
              display: "block",
              width: "100%",
              textAlign: "left",
              background:
                user.user_id === currentEmulatedUser ? "#1e3a5f" : "transparent",
              border: "none",
              borderBottom: "1px solid #1e293b",
              color: "#f1f5f9",
              padding: "6px 8px",
              cursor: "pointer",
              fontFamily: "monospace",
              fontSize: 12,
            }}
            onMouseEnter={(event) => {
              event.currentTarget.style.background = "#1e3a5f";
            }}
            onMouseLeave={(event) => {
              event.currentTarget.style.background =
                user.user_id === currentEmulatedUser ? "#1e3a5f" : "transparent";
            }}
          >
            <div style={{ fontWeight: 600 }}>{displayName(user)}</div>
            <div style={{ opacity: 0.6, fontSize: 11 }}>{user.email ?? user.user_id}</div>
          </button>
        ))}
    </div>
  );
}

interface ActiveUserRowProps {
  currentEmulatedUser: string;
  onClear: () => void;
}

function ActiveUserRow({ currentEmulatedUser, onClear }: ActiveUserRowProps) {
  if (!currentEmulatedUser) {
    return null;
  }

  return (
    <div
      style={{
        marginTop: 8,
        display: "flex",
        alignItems: "center",
        gap: 6,
      }}
    >
      <span style={{ opacity: 0.6, flexShrink: 0 }}>Active:</span>
      <span
        style={{
          color: "#fbbf24",
          flex: 1,
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
        }}
      >
        {currentEmulatedUser}
      </span>
      <button
        onClick={onClear}
        style={{
          background: "#7f1d1d",
          color: "white",
          border: "none",
          borderRadius: 4,
          padding: "2px 8px",
          fontSize: 12,
          cursor: "pointer",
          fontFamily: "monospace",
          flexShrink: 0,
        }}
      >
        Clear
      </button>
    </div>
  );
}

interface EmulationPopoverProps {
  currentEmulatedUser: string;
  inputRef: React.RefObject<HTMLInputElement | null>;
  loading: boolean;
  onClear: () => void;
  onQueryChange: (value: string) => void;
  onSelectUser: (user: UserSummary) => void;
  query: string;
  results: UserSummary[];
}

function EmulationPopover({
  currentEmulatedUser,
  inputRef,
  loading,
  onClear,
  onQueryChange,
  onSelectUser,
  query,
  results,
}: EmulationPopoverProps) {
  return (
    <div
      style={{
        backgroundColor: "#1e293b",
        color: "#f1f5f9",
        border: "1px solid #475569",
        borderRadius: 8,
        padding: 12,
        fontSize: 12,
        fontFamily: "monospace",
        boxShadow: "0 4px 20px rgba(0,0,0,0.5)",
        width: 300,
      }}
    >
      <div style={{ fontWeight: 700, marginBottom: 8 }}>🔧 Emulate User</div>
      <input
        ref={inputRef}
        type="text"
        value={query}
        onChange={(event) => onQueryChange(event.target.value)}
        placeholder="Search by name or email…"
        style={{
          width: "100%",
          boxSizing: "border-box",
          background: "#0f172a",
          border: "1px solid #334155",
          borderRadius: 4,
          color: "#f1f5f9",
          padding: "5px 8px",
          fontSize: 12,
          fontFamily: "monospace",
          outline: "none",
          marginBottom: 6,
        }}
      />
      <SearchResults
        currentEmulatedUser={currentEmulatedUser}
        loading={loading}
        results={results}
        onSelectUser={onSelectUser}
      />
      <ActiveUserRow currentEmulatedUser={currentEmulatedUser} onClear={onClear} />
    </div>
  );
}

export function UserEmulationPanel() {
  const currentEmulatedUser = localStorage.getItem(STORAGE_KEY) ?? "";
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<UserSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!open) return;
    inputRef.current?.focus();
    fetchUsers("");
  }, [open]);

  useEffect(() => {
    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
    };
  }, []);

  const fetchUsers = async (q: string) => {
    setLoading(true);
    try {
      const data = await backendApiClient.request<UserSummary[]>(
        `/api/dev/users/search?q=${encodeURIComponent(q)}`
      );
      setResults(data);
    } catch {
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  const handleQueryChange = (value: string) => {
    setQuery(value);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => fetchUsers(value), 250);
  };

  const selectUser = (user: UserSummary) => {
    localStorage.setItem(STORAGE_KEY, user.user_id);
    window.location.reload();
  };

  const handleClear = () => {
    localStorage.removeItem(STORAGE_KEY);
    window.location.reload();
  };

  return (
    <>
      <EmulationBanner currentEmulatedUser={currentEmulatedUser} />
      <div
        style={{
          position: "fixed",
          bottom: 16,
          right: 16,
          zIndex: 9998,
          display: "flex",
          flexDirection: "column",
          alignItems: "flex-end",
          gap: 8,
        }}
      >
        {open && (
          <EmulationPopover
            currentEmulatedUser={currentEmulatedUser}
            inputRef={inputRef}
            loading={loading}
            onClear={handleClear}
            onQueryChange={handleQueryChange}
            onSelectUser={selectUser}
            query={query}
            results={results}
          />
        )}

        <button
          onClick={() => setOpen((v) => !v)}
          title="Dev: Emulate User"
          style={{
            background: currentEmulatedUser ? "#dc2626" : "#1e293b",
            color: "white",
            border: "1px solid #475569",
            borderRadius: 6,
            padding: "6px 10px",
            fontSize: 12,
            fontFamily: "monospace",
            cursor: "pointer",
            boxShadow: "0 2px 8px rgba(0,0,0,0.4)",
          }}
        >
          🔧 {currentEmulatedUser ? "Emulating" : "Emulate User"}
        </button>
      </div>
    </>
  );
}
