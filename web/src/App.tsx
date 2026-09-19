import {
  ArrowUpRight,
  BarChart3,
  BookOpenCheck,
  Check,
  CircleAlert,
  Code2,
  Database,
  ExternalLink,
  FileCode2,
  GitBranch,
  Github,
  Library,
  Link2,
  LoaderCircle,
  MessageSquareCode,
  Play,
  Plus,
  RefreshCw,
  Search,
  ShieldCheck,
  Sparkles,
  Trash2,
} from "lucide-react";
import { FormEvent, ReactNode, useEffect, useMemo, useState } from "react";

type View = "investigate" | "repositories" | "evaluation";

type SyncRun = {
  id: string;
  status: "queued" | "running" | "completed" | "failed";
  patch_cards_indexed: number;
  chunks_indexed: number;
  pull_requests_seen?: number;
  error: string | null;
  created_at: string;
  finished_at: string | null;
};

type EvaluationRun = {
  id: string;
  status: "running" | "completed" | "failed";
  benchmark_cases: number;
  metrics: {
    recall_at_1?: number | null;
    recall_at_3?: number | null;
    mrr?: number | null;
    quality?: "verified_issue_to_fix" | "smoke_test";
    case_breakdown?: { verified_issue_to_fix: number; pr_description_smoke_test: number };
    failure_samples?: {
      expected_pull_number?: number;
      issue_url?: string | null;
      query: string;
      failure_type: string;
      source_kind: string;
    }[];
    note?: string;
  };
  error: string | null;
  created_at: string;
};

type Repository = {
  id: string;
  github_url: string;
  full_name: string;
  owner: string;
  name: string;
  default_branch: string;
  description: string | null;
  is_private: boolean;
  latest_sync: SyncRun | null;
  latest_evaluation?: EvaluationRun | null;
};

type PatchMatch = {
  pull_number: number | null;
  pull_url: string;
  title: string;
  score: number;
  merge_commit_sha: string | null;
  linked_issue_number?: number | null;
  linked_issue_url?: string | null;
  changed_files: string[];
  test_files: string[];
  excerpt: string;
  recommended_solution?: RecommendedSolution;
};

type RecommendedSolution = {
  assessment: string;
  implementation_steps: string[];
  code_guidance: string | null;
  verification: string[];
  cautions: string[];
  source_pull_numbers: number[];
  mode: "model_assisted" | "evidence_guided";
};

type Investigation = {
  issue: string;
  summary: string;
  matches: PatchMatch[];
  confidence: "high" | "moderate" | "low" | "no_similar_fix_found";
  recommended_solution?: RecommendedSolution;
};

type User = {
  id: string;
  email: string;
  created_at: string;
};

type Session = {
  token: string;
  expires_at: string;
  user: User;
};

type SavedInvestigation = {
  id: string;
  repository_id: string;
  issue: string;
  result: Investigation;
  created_at: string;
};

type GitHubConnection = {
  configured: boolean;
  connected: boolean;
  github_login: string | null;
  scopes: string[];
  access_token_expires_at?: string | null;
  updated_at?: string | null;
};

const API_URL = (import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
const SESSION_KEY = "patchwork.session";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  const token = sessionToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const response = await fetch(`${API_URL}${path}`, { ...init, headers });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail ?? "The request could not be completed.");
  }
  return response.json() as Promise<T>;
}

function sessionToken() {
  if (typeof window === "undefined") return null;
  try {
    return (JSON.parse(window.localStorage.getItem(SESSION_KEY) ?? "null") as Session | null)?.token ?? null;
  } catch {
    return null;
  }
}

function storeSession(session: Session) {
  window.localStorage.setItem(SESSION_KEY, JSON.stringify(session));
}

function clearSession() {
  window.localStorage.removeItem(SESSION_KEY);
}

function isWorking(status?: string) {
  return status === "queued" || status === "running";
}

function statusText(status?: string) {
  if (!status) return "Not indexed";
  return status[0].toUpperCase() + status.slice(1);
}

function formattedDate(value?: string | null) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.valueOf())
    ? "Recently"
    : new Intl.DateTimeFormat("en", {
        month: "short",
        day: "numeric",
        hour: "numeric",
        minute: "2-digit",
      }).format(date);
}

function percent(value?: number | null) {
  return value === null || value === undefined ? "—" : `${Math.round(value * 100)}%`;
}

export default function App() {
  const [view, setView] = useState<View>("investigate");
  const [repositories, setRepositories] = useState<Repository[]>([]);
  const [selectedRepositoryId, setSelectedRepositoryId] = useState("");
  const [repositoryUrl, setRepositoryUrl] = useState("https://github.com/fastapi/fastapi");
  const [issue, setIssue] = useState("");
  const [investigation, setInvestigation] = useState<Investigation | null>(null);
  const [investigations, setInvestigations] = useState<SavedInvestigation[]>([]);
  const [user, setUser] = useState<User | null>(null);
  const [githubConnection, setGitHubConnection] = useState<GitHubConnection | null>(null);
  const [authReady, setAuthReady] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [isConnecting, setIsConnecting] = useState(false);
  const [isInvestigating, setIsInvestigating] = useState(false);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const selectedRepository = useMemo(
    () => repositories.find((repository) => repository.id === selectedRepositoryId) ?? null,
    [repositories, selectedRepositoryId],
  );
  const hasActiveSync = repositories.some((repository) => isWorking(repository.latest_sync?.status));

  async function refreshInvestigations(repositoryId = selectedRepositoryId) {
    if (!repositoryId) {
      setInvestigations([]);
      return;
    }
    try {
      const response = await request<SavedInvestigation[]>(
        `/patchwork/repositories/${repositoryId}/investigations`,
      );
      setInvestigations(response);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not load saved investigations.");
    }
  }

  async function refreshRepositories() {
    try {
      const response = await request<Repository[]>("/patchwork/repositories");
      setRepositories(response);
      setSelectedRepositoryId((current) => {
        if (response.some((repository) => repository.id === current)) return current;
        return response[0]?.id ?? "";
      });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not load repositories.");
    } finally {
      setIsLoading(false);
    }
  }

  async function refreshGitHubConnection() {
    try {
      setGitHubConnection(await request<GitHubConnection>("/auth/github/status"));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not load GitHub access status.");
    }
  }

  useEffect(() => {
    async function restoreSession() {
      if (!sessionToken()) {
        setAuthReady(true);
        return;
      }
      try {
        setUser(await request<User>("/auth/me"));
      } catch {
        clearSession();
      } finally {
        setAuthReady(true);
      }
    }
    void restoreSession();
  }, []);

  useEffect(() => {
    if (user) {
      void refreshRepositories();
      void refreshGitHubConnection();
    }
  }, [user]);

  useEffect(() => {
    if (!user) return;
    const outcome = new URLSearchParams(window.location.search).get("github");
    if (!outcome) return;
    window.history.replaceState({}, "", window.location.pathname);
    if (outcome === "connected") {
      setNotice("GitHub is linked. You can now add repositories your GitHub account can read.");
      void refreshGitHubConnection();
    } else if (outcome === "cancelled") {
      setNotice("GitHub linking was cancelled. Public repositories still work without it.");
    } else {
      setError("GitHub could not be linked. Confirm the OAuth callback URL and try again.");
    }
  }, [user]);

  useEffect(() => {
    if (user) void refreshInvestigations();
  }, [user, selectedRepositoryId]);

  useEffect(() => {
    if (!hasActiveSync) return;
    const interval = window.setInterval(() => void refreshRepositories(), 3000);
    return () => window.clearInterval(interval);
  }, [hasActiveSync]);

  function selectRepository(repositoryId: string, nextView?: View) {
    setSelectedRepositoryId(repositoryId);
    setInvestigation(null);
    if (nextView) setView(nextView);
  }

  async function connectRepository(event: FormEvent) {
    event.preventDefault();
    if (!repositoryUrl.trim()) return;
    setError(null);
    setNotice(null);
    setIsConnecting(true);
    try {
      const response = await request<{ repository: Repository }>("/patchwork/repositories", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ github_url: repositoryUrl.trim(), pull_limit: 15 }),
      });
      setSelectedRepositoryId(response.repository.id);
      setView("investigate");
      setNotice(`Connected ${response.repository.full_name}. Indexing its historical fixes now.`);
      await refreshRepositories();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not connect this repository.");
    } finally {
      setIsConnecting(false);
    }
  }

  async function syncRepository(repository = selectedRepository) {
    if (!repository) return;
    setError(null);
    setNotice(null);
    try {
      await request(`/patchwork/repositories/${repository.id}/sync`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pull_limit: 15 }),
      });
      setNotice(`Refreshing historical fixes for ${repository.full_name}.`);
      await refreshRepositories();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not start a repository sync.");
    }
  }

  async function investigateIssue(event: FormEvent) {
    event.preventDefault();
    if (!selectedRepository || !issue.trim()) return;
    setError(null);
    setNotice(null);
    setIsInvestigating(true);
    try {
      const response = await request<Investigation>(
        `/patchwork/repositories/${selectedRepository.id}/investigate`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ issue: issue.trim(), limit: 5 }),
        },
      );
      setInvestigation(response);
      await refreshInvestigations();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Patchwork could not investigate this issue.");
    } finally {
      setIsInvestigating(false);
    }
  }

  async function evaluateRepository() {
    if (!selectedRepository) return;
    setError(null);
    setNotice(null);
    setIsEvaluating(true);
    try {
      const evaluation = await request<EvaluationRun>(
        `/patchwork/repositories/${selectedRepository.id}/evaluate`,
        { method: "POST" },
      );
      setNotice(
        evaluation.benchmark_cases
          ? evaluation.metrics.quality === "verified_issue_to_fix"
            ? `Evaluation finished across ${evaluation.benchmark_cases} verified issue-to-fix examples.`
            : `Indexing smoke test finished across ${evaluation.benchmark_cases} historical fixes.`
          : "No historical fix examples are available yet. Finish a repository sync and try again.",
      );
      await refreshRepositories();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not run the evaluation.");
    } finally {
      setIsEvaluating(false);
    }
  }

  async function deleteRepository(repository = selectedRepository) {
    if (!repository) return;
    if (!window.confirm(`Remove ${repository.full_name} and its indexed patch history?`)) return;
    try {
      await request(`/patchwork/repositories/${repository.id}`, { method: "DELETE" });
      if (repository.id === selectedRepositoryId) {
        setSelectedRepositoryId("");
        setInvestigation(null);
      }
      setNotice(`${repository.full_name} was removed from Patchwork.`);
      await refreshRepositories();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not remove this repository.");
    }
  }

  async function logout() {
    try {
      await request<void>("/auth/logout", { method: "POST" });
    } finally {
      clearSession();
      setUser(null);
      setRepositories([]);
      setSelectedRepositoryId("");
      setInvestigations([]);
      setInvestigation(null);
      setGitHubConnection(null);
    }
  }

  async function linkGitHub() {
    setError(null);
    try {
      const response = await request<{ authorization_url: string }>("/auth/github/authorize", {
        method: "POST",
      });
      window.location.assign(response.authorization_url);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not start GitHub linking.");
    }
  }

  async function unlinkGitHub() {
    setError(null);
    try {
      await request("/auth/github/connection", { method: "DELETE" });
      setNotice("GitHub was unlinked. Existing local Patchwork data is unchanged.");
      await refreshGitHubConnection();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not unlink GitHub.");
    }
  }

  function completeAuthentication(session: Session) {
    storeSession(session);
    setUser(session.user);
    setError(null);
  }

  if (!authReady) return <AuthLoading />;
  if (!user) return <AuthScreen onAuthenticated={completeAuthentication} />;

  return (
    <main className="min-h-screen bg-paper text-ink">
      <div className="pointer-events-none fixed inset-0 grid-noise opacity-50" />
      <div className="relative mx-auto flex min-h-screen max-w-[1600px]">
        <aside className="hidden w-[270px] shrink-0 flex-col bg-ink px-5 py-6 text-[#eaf1e9] lg:flex">
          <div className="flex items-center gap-3 px-2">
            <div className="grid size-10 place-items-center rounded-2xl bg-sage text-ink shadow-lg shadow-black/20">
              <GitBranch size={20} strokeWidth={2.6} />
            </div>
            <div>
              <p className="text-lg font-semibold tracking-tight">Patchwork</p>
              <p className="text-xs text-[#aabeb2]">Codebase memory</p>
            </div>
          </div>

          <nav className="mt-12 space-y-1 text-sm" aria-label="Workspace navigation">
            <NavItem icon={<Search size={18} />} label="Investigate" active={view === "investigate"} onClick={() => setView("investigate")} />
            <NavItem icon={<Library size={18} />} label="Repositories" count={repositories.length} active={view === "repositories"} onClick={() => setView("repositories")} />
            <NavItem icon={<BarChart3 size={18} />} label="Evaluation lab" active={view === "evaluation"} onClick={() => setView("evaluation")} />
          </nav>

          <div className="mt-10">
            <div className="mb-3 flex items-center justify-between px-2 text-[11px] font-bold uppercase tracking-[0.15em] text-[#80978a]">
              <span>Connected</span>
              <button onClick={() => void refreshRepositories()} className="text-sage" type="button" aria-label="Refresh connected repositories">
                <RefreshCw size={13} />
              </button>
            </div>
            <div className="space-y-1">
              {repositories.slice(0, 5).map((repository) => (
                <button
                  key={repository.id}
                  type="button"
                  onClick={() => selectRepository(repository.id, "investigate")}
                  className={`flex w-full items-center gap-2 rounded-xl px-3 py-2.5 text-left text-xs transition ${selectedRepository?.id === repository.id ? "bg-white/12 text-white" : "text-[#b9cbbf] hover:bg-white/6"}`}
                >
                  <Github size={15} />
                  <span className="min-w-0 flex-1 truncate font-medium">{repository.full_name}</span>
                  <StatusDot status={repository.latest_sync?.status} />
                </button>
              ))}
              {!isLoading && repositories.length === 0 && <p className="px-2 py-3 text-xs leading-5 text-[#80978a]">Add a repository to build its patch history.</p>}
            </div>
          </div>

          <div className="mt-auto rounded-2xl border border-white/10 bg-white/5 p-4">
            <div className="flex items-center gap-2 text-xs font-medium text-sage"><ShieldCheck size={15} /> Read-only indexing</div>
            <p className="mt-3 text-sm leading-5 text-[#c5d3ca]">Patchwork reads history; it never writes to a connected repository.</p>
          </div>
        </aside>

        <section className="min-w-0 flex-1 px-5 py-5 sm:px-8 lg:px-10 lg:py-7">
          <header className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-3 lg:hidden">
              <div className="grid size-10 place-items-center rounded-2xl bg-ink text-sage"><GitBranch size={20} /></div>
              <span className="text-lg font-semibold">Patchwork</span>
            </div>
            <div className="hidden lg:block">
              <p className="text-sm font-medium text-[#648276]">{view === "investigate" ? "Investigate a regression" : view === "repositories" ? "Repository library" : "Retrieval evaluation"}</p>
              <p className="mt-1 text-xs text-[#92a79c]">Historical fixes, grounded in code</p>
            </div>
            <div className="ml-auto flex items-center gap-2 rounded-full border border-[#dce4da] bg-white/75 px-3 py-2 text-xs font-semibold text-[#547468]">
              {githubConnection?.configured && <GitHubAccess connection={githubConnection} onLink={() => void linkGitHub()} onUnlink={() => void unlinkGitHub()} />}
              <span className="hidden max-w-40 truncate sm:inline">{user.email}</span>
              <button type="button" onClick={() => void logout()} className="border-l border-[#dce4da] pl-2 text-[#547468] hover:text-ink">Sign out</button>
            </div>
          </header>

          <nav className="mt-6 grid grid-cols-3 gap-2 lg:hidden" aria-label="Workspace navigation">
            <MobileNavItem label="Investigate" active={view === "investigate"} onClick={() => setView("investigate")} />
            <MobileNavItem label="Repositories" active={view === "repositories"} onClick={() => setView("repositories")} />
            <MobileNavItem label="Evaluate" active={view === "evaluation"} onClick={() => setView("evaluation")} />
          </nav>

          {error && <Notice type="error">{error}</Notice>}
          {notice && <Notice type="success">{notice}</Notice>}

          {view === "investigate" && (
            <InvestigateView
              repository={selectedRepository}
              repositoryUrl={repositoryUrl}
              onRepositoryUrlChange={setRepositoryUrl}
              onConnect={connectRepository}
              isConnecting={isConnecting}
              issue={issue}
              onIssueChange={setIssue}
              onInvestigate={investigateIssue}
              isInvestigating={isInvestigating}
              investigation={investigation}
              investigations={investigations}
              onReopen={(saved) => { setIssue(saved.issue); setInvestigation(saved.result); }}
              onSync={() => void syncRepository()}
              onOpenEvaluation={() => setView("evaluation")}
              isLoading={isLoading}
            />
          )}

          {view === "repositories" && (
            <RepositoryLibrary
              repositories={repositories}
              selectedRepositoryId={selectedRepositoryId}
              repositoryUrl={repositoryUrl}
              onRepositoryUrlChange={setRepositoryUrl}
              onConnect={connectRepository}
              isConnecting={isConnecting}
              onSelect={(id) => selectRepository(id, "investigate")}
              onEvaluate={(id) => selectRepository(id, "evaluation")}
              onSync={(repository) => void syncRepository(repository)}
              onDelete={(repository) => void deleteRepository(repository)}
            />
          )}

          {view === "evaluation" && (
            <EvaluationLab
              repositories={repositories}
              repository={selectedRepository}
              onSelect={(id) => selectRepository(id)}
              onEvaluate={() => void evaluateRepository()}
              isEvaluating={isEvaluating}
              onOpenRepositories={() => setView("repositories")}
            />
          )}
        </section>
      </div>
    </main>
  );
}

function InvestigateView({
  repository,
  repositoryUrl,
  onRepositoryUrlChange,
  onConnect,
  isConnecting,
  issue,
  onIssueChange,
  onInvestigate,
  isInvestigating,
  investigation,
  investigations,
  onReopen,
  onSync,
  onOpenEvaluation,
  isLoading,
}: {
  repository: Repository | null;
  repositoryUrl: string;
  onRepositoryUrlChange: (value: string) => void;
  onConnect: (event: FormEvent) => void;
  isConnecting: boolean;
  issue: string;
  onIssueChange: (value: string) => void;
  onInvestigate: (event: FormEvent) => void;
  isInvestigating: boolean;
  investigation: Investigation | null;
  investigations: SavedInvestigation[];
  onReopen: (saved: SavedInvestigation) => void;
  onSync: () => void;
  onOpenEvaluation: () => void;
  isLoading: boolean;
}) {
  const syncing = isWorking(repository?.latest_sync?.status);
  return <div className="mt-10 grid gap-8 xl:grid-cols-[minmax(0,1fr)_320px]">
    <div className="min-w-0">
      <section>
        <p className="text-sm font-semibold uppercase tracking-[0.18em] text-moss">Investigate</p>
        <h1 className="mt-2 max-w-3xl text-4xl font-semibold tracking-[-0.045em] text-ink sm:text-5xl">Find the fix your repository already learned.</h1>
        <p className="mt-4 max-w-2xl text-base leading-7 text-[#6d8579]">Describe a new bug in plain language. Patchwork searches this repository’s past fixes and returns the most useful implementation references.</p>
      </section>

      {!repository && <ConnectCard repositoryUrl={repositoryUrl} onRepositoryUrlChange={onRepositoryUrlChange} onConnect={onConnect} isConnecting={isConnecting} />}

      {repository ? <>
        <RepositoryOverview repository={repository} onSync={onSync} />
        <section className="soft-shadow mt-7 rounded-2xl border border-[#dfe7dd] bg-white p-5 sm:p-6">
          <div className="flex items-start gap-3">
            <span className="grid size-9 shrink-0 place-items-center rounded-xl bg-[#eff5ed] text-moss"><MessageSquareCode size={18} /></span>
            <div><h2 className="font-semibold tracking-tight">Describe the issue</h2><p className="mt-1 text-sm text-[#71877c]">Include an error message, affected file, expected behavior, or a short stack trace when available.</p></div>
          </div>
          <form onSubmit={onInvestigate} className="mt-5">
            <textarea value={issue} onChange={(event) => onIssueChange(event.target.value)} className="min-h-32 w-full resize-y rounded-xl border border-[#d9e3d8] bg-[#fbfcfa] p-4 text-sm leading-6 outline-none placeholder:text-[#9daf9f] focus:border-moss focus:ring-4 focus:ring-[#dfeee0]" placeholder="Example: Streaming requests hang after a client disconnects. Find a similar historical fix and the regression test it added." />
            <div className="mt-3 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <p className="text-xs text-[#84968c]">Results are evidence to review—not an automatic code change.</p>
              <button className="inline-flex items-center justify-center gap-2 rounded-xl bg-moss px-4 py-3 text-sm font-bold text-white transition hover:-translate-y-0.5 hover:bg-[#17483b] disabled:cursor-not-allowed disabled:opacity-50" type="submit" disabled={isInvestigating || syncing || !issue.trim()}>{isInvestigating ? <LoaderCircle size={17} className="animate-spin" /> : <Search size={17} />}{isInvestigating ? "Searching history…" : "Find similar fixes"}</button>
            </div>
          </form>
        </section>
        {investigation && <InvestigationPanel investigation={investigation} />}
        <SavedInvestigations investigations={investigations} onReopen={onReopen} />
      </> : <EmptyState loading={isLoading} />}
    </div>

    <aside className="space-y-6">
      {repository && <section className="soft-shadow overflow-hidden rounded-2xl bg-ink text-white"><div className="relative overflow-hidden px-6 pb-6 pt-7"><div className="absolute -right-12 -top-10 size-40 rounded-full bg-sage/15 blur-2xl" /><div className="relative flex items-center justify-between"><span className="grid size-10 place-items-center rounded-xl bg-white/10 text-sage"><BarChart3 size={19} /></span><span className="rounded-full border border-sage/20 bg-sage/10 px-2.5 py-1 text-[10px] font-bold uppercase tracking-[0.15em] text-sage">Measure retrieval</span></div><h2 className="relative mt-5 text-xl font-semibold tracking-tight">Evaluation lab</h2><p className="relative mt-2 text-sm leading-5 text-[#bfd1c5]">Measure whether historical issue reports retrieve their correct fixes.</p></div><div className="border-t border-white/10 bg-black/10 p-4"><button type="button" onClick={onOpenEvaluation} className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-sage px-4 py-3 text-sm font-bold text-ink transition hover:-translate-y-0.5 hover:bg-[#d4e5c4]"><BarChart3 size={16} /> Open evaluation lab</button></div></section>}
      <section className="rounded-2xl border border-[#e5ddd4] bg-[#fbf5ef] p-5"><div className="flex items-center gap-2 text-xs font-bold uppercase tracking-[0.14em] text-[#a2633d]"><BookOpenCheck size={15} /> How results work</div><ol className="mt-4 space-y-3 text-sm leading-5 text-[#765743]"><li className="flex gap-3"><span className="grid size-5 shrink-0 place-items-center rounded-full bg-white text-[10px] font-bold text-[#a2633d]">1</span>Search only the selected repository.</li><li className="flex gap-3"><span className="grid size-5 shrink-0 place-items-center rounded-full bg-white text-[10px] font-bold text-[#a2633d]">2</span>Match your issue to historical PRs and their tests.</li><li className="flex gap-3"><span className="grid size-5 shrink-0 place-items-center rounded-full bg-white text-[10px] font-bold text-[#a2633d]">3</span>Review the linked code before applying a fix.</li></ol></section>
    </aside>
  </div>;
}

function SavedInvestigations({ investigations, onReopen }: { investigations: SavedInvestigation[]; onReopen: (saved: SavedInvestigation) => void; }) {
  if (!investigations.length) return null;
  return <section className="mt-7 rounded-2xl border border-[#dfe7dd] bg-white p-5 sm:p-6"><div className="flex items-center justify-between gap-3"><div><p className="text-xs font-bold uppercase tracking-[0.14em] text-moss">Saved investigations</p><h2 className="mt-1 font-semibold tracking-tight">Revisit your recent research</h2></div><span className="rounded-full bg-[#eef4ec] px-2.5 py-1 text-xs font-bold text-moss">{investigations.length}</span></div><div className="mt-4 divide-y divide-[#edf0ec]">{investigations.slice(0, 5).map((saved) => <button key={saved.id} type="button" onClick={() => onReopen(saved)} className="flex w-full items-center justify-between gap-4 py-3 text-left transition hover:text-moss"><span className="min-w-0"><span className="block truncate text-sm font-medium text-ink">{saved.issue}</span><span className="mt-1 block text-xs text-[#7b9085]">{formattedDate(saved.created_at)} · {saved.result.matches.length} historical match{saved.result.matches.length === 1 ? "" : "es"}</span></span><ArrowUpRight size={16} className="shrink-0 text-[#779084]" /></button>)}</div></section>;
}

function RepositoryLibrary({ repositories, selectedRepositoryId, repositoryUrl, onRepositoryUrlChange, onConnect, isConnecting, onSelect, onEvaluate, onSync, onDelete }: {
  repositories: Repository[]; selectedRepositoryId: string; repositoryUrl: string; onRepositoryUrlChange: (value: string) => void; onConnect: (event: FormEvent) => void; isConnecting: boolean; onSelect: (id: string) => void; onEvaluate: (id: string) => void; onSync: (repository: Repository) => void; onDelete: (repository: Repository) => void;
}) {
  return <div className="mt-10 max-w-5xl"><section><p className="text-sm font-semibold uppercase tracking-[0.18em] text-moss">Repository library</p><h1 className="mt-2 text-4xl font-semibold tracking-[-0.045em] sm:text-5xl">Your connected codebases.</h1><p className="mt-4 max-w-2xl text-base leading-7 text-[#6d8579]">Each repository keeps an isolated history of patches, so investigations stay relevant and private to that codebase.</p></section><ConnectCard repositoryUrl={repositoryUrl} onRepositoryUrlChange={onRepositoryUrlChange} onConnect={onConnect} isConnecting={isConnecting} /><section className="mt-8"><div className="mb-4 flex items-center justify-between"><h2 className="font-semibold">Connected repositories</h2><span className="text-xs font-semibold text-[#71877c]">{repositories.length} total</span></div><div className="grid gap-4">{repositories.map((repository) => <RepositoryCard key={repository.id} repository={repository} selected={repository.id === selectedRepositoryId} onSelect={() => onSelect(repository.id)} onEvaluate={() => onEvaluate(repository.id)} onSync={() => onSync(repository)} onDelete={() => onDelete(repository)} />)}{repositories.length === 0 && <EmptyState loading={false} />}</div></section></div>;
}

function EvaluationLab({ repositories, repository, onSelect, onEvaluate, isEvaluating, onOpenRepositories }: { repositories: Repository[]; repository: Repository | null; onSelect: (id: string) => void; onEvaluate: () => void; isEvaluating: boolean; onOpenRepositories: () => void; }) {
  const evaluation = repository?.latest_evaluation ?? null;
  const syncing = isWorking(repository?.latest_sync?.status);
  return <div className="mt-10 max-w-5xl"><section className="flex flex-col gap-5 sm:flex-row sm:items-end sm:justify-between"><div><p className="text-sm font-semibold uppercase tracking-[0.18em] text-moss">Evaluation lab</p><h1 className="mt-2 text-4xl font-semibold tracking-[-0.045em] sm:text-5xl">Measure retrieval, not vibes.</h1><p className="mt-4 max-w-2xl text-base leading-7 text-[#6d8579]">Patchwork measures whether a real issue report retrieves the pull request that fixed it. When a repository exposes no links, it runs a clearly labeled indexing smoke test instead.</p></div>{repositories.length > 0 && <label className="text-xs font-bold uppercase tracking-[0.12em] text-[#789084]">Repository<select value={repository?.id ?? ""} onChange={(event) => onSelect(event.target.value)} className="mt-2 block w-full rounded-xl border border-[#d9e3d8] bg-white px-3 py-2.5 text-sm font-medium normal-case tracking-normal text-ink outline-none focus:border-moss"><option value="" disabled>Select a repository</option>{repositories.map((item) => <option key={item.id} value={item.id}>{item.full_name}</option>)}</select></label>}</section>{repository ? <><section className="soft-shadow mt-8 grid overflow-hidden rounded-2xl border border-[#dfe7dd] bg-white lg:grid-cols-[1.2fr_.8fr]"><div className="p-6 sm:p-7"><div className="flex items-center gap-3"><span className="grid size-10 place-items-center rounded-xl bg-[#eef4ec] text-moss"><BarChart3 size={20} /></span><div><h2 className="font-semibold tracking-tight">Evaluate {repository.full_name}</h2><p className="mt-1 text-sm text-[#71877c]">Verified issue-to-fix pairs are stored at sync time. Their issue text is excluded from target patch cards to avoid leakage.</p></div></div><div className="mt-6 grid gap-3 sm:grid-cols-3"><Metric label="Recall at 1" value={percent(evaluation?.metrics.recall_at_1)} icon={<Search size={16} />} /><Metric label="Recall at 3" value={percent(evaluation?.metrics.recall_at_3)} icon={<Library size={16} />} /><Metric label="Mean rank" value={evaluation?.metrics.mrr === null || evaluation?.metrics.mrr === undefined ? "—" : String(evaluation.metrics.mrr)} icon={<BarChart3 size={16} />} /></div></div><div className="border-t border-[#e7ece6] bg-[#f7faf6] p-6 lg:border-l lg:border-t-0"><p className="text-sm font-semibold">Run a check</p><p className="mt-2 text-sm leading-6 text-[#71877c]">The report states whether scores are verified issue-to-fix retrieval or a lower-confidence indexing smoke test.</p><button type="button" onClick={onEvaluate} disabled={isEvaluating || syncing} className="mt-5 inline-flex w-full items-center justify-center gap-2 rounded-xl bg-ink px-4 py-3 text-sm font-bold text-white transition hover:bg-[#1b4035] disabled:cursor-not-allowed disabled:opacity-50">{isEvaluating ? <LoaderCircle size={17} className="animate-spin" /> : <Play size={16} />}{isEvaluating ? "Evaluating…" : "Run evaluation"}</button></div></section><EvaluationReportCard evaluation={evaluation} /></> : <section className="soft-shadow mt-8 rounded-2xl border border-dashed border-[#ccdace] bg-white/70 px-6 py-16 text-center"><div className="mx-auto grid size-14 place-items-center rounded-2xl bg-[#eaf2e7] text-moss"><Library size={24} /></div><h2 className="mt-5 text-lg font-semibold">Connect a repository first</h2><p className="mx-auto mt-2 max-w-md text-sm leading-6 text-[#71877c]">An evaluation needs a repository with indexed historical fixes.</p><button type="button" onClick={onOpenRepositories} className="mt-5 inline-flex items-center gap-2 rounded-xl bg-ink px-4 py-3 text-sm font-bold text-white">Open repository library <ArrowUpRight size={16} /></button></section>}</div>;
}

function ConnectCard({ repositoryUrl, onRepositoryUrlChange, onConnect, isConnecting }: { repositoryUrl: string; onRepositoryUrlChange: (value: string) => void; onConnect: (event: FormEvent) => void; isConnecting: boolean; }) {
  return <section className="soft-shadow mt-8 rounded-2xl border border-[#dfe7dd] bg-white p-5 sm:p-6"><div className="flex items-start gap-3"><span className="grid size-9 shrink-0 place-items-center rounded-xl bg-[#eef4ec] text-moss"><Link2 size={18} /></span><div><h2 className="font-semibold tracking-tight">Add a GitHub repository</h2><p className="mt-1 text-sm text-[#71877c]">Public repositories work immediately. Link GitHub from the account menu to index private repositories you can read.</p></div></div><form onSubmit={onConnect} className="mt-5 flex flex-col gap-3 sm:flex-row"><div className="flex min-w-0 flex-1 items-center gap-3 rounded-xl border border-[#d9e3d8] bg-[#fbfcfa] px-3 focus-within:border-moss focus-within:ring-4 focus-within:ring-[#dfeee0]"><Github size={18} className="shrink-0 text-[#748d80]" /><input value={repositoryUrl} onChange={(event) => onRepositoryUrlChange(event.target.value)} className="min-w-0 flex-1 bg-transparent py-3 text-sm outline-none placeholder:text-[#9daf9f]" placeholder="https://github.com/owner/repository" /></div><button className="inline-flex items-center justify-center gap-2 rounded-xl bg-ink px-4 py-3 text-sm font-bold text-white transition hover:-translate-y-0.5 hover:bg-[#1b4035] disabled:cursor-not-allowed disabled:opacity-50" type="submit" disabled={isConnecting}>{isConnecting ? <LoaderCircle size={17} className="animate-spin" /> : <Plus size={17} />}{isConnecting ? "Connecting…" : "Connect & index"}</button></form></section>;
}

function RepositoryOverview({ repository, onSync }: { repository: Repository; onSync: () => void }) {
  const syncing = isWorking(repository.latest_sync?.status);
  return <section className="soft-shadow mt-7 overflow-hidden rounded-2xl border border-[#dfe7dd] bg-white"><div className="flex flex-col gap-5 border-b border-[#e7ece6] px-5 py-5 sm:flex-row sm:items-center sm:justify-between"><div className="flex min-w-0 items-center gap-3"><span className="grid size-10 shrink-0 place-items-center rounded-xl bg-ink text-sage"><Github size={19} /></span><div className="min-w-0"><div className="flex items-center gap-2"><h2 className="truncate font-semibold tracking-tight">{repository.full_name}</h2><a href={repository.github_url} target="_blank" rel="noreferrer" className="text-[#779084] hover:text-moss" aria-label={`Open ${repository.full_name} on GitHub`}><ExternalLink size={15} /></a></div><p className="mt-1 truncate text-xs text-[#809489]">{repository.description || `Default branch: ${repository.default_branch}`}</p></div></div><div className="flex items-center gap-2"><StatusPill status={repository.latest_sync?.status} /><button onClick={onSync} disabled={syncing} type="button" className="inline-flex items-center gap-2 rounded-lg bg-[#eef4ec] px-3 py-2 text-xs font-bold text-moss transition hover:bg-[#dcebd7] disabled:opacity-50">{syncing ? <LoaderCircle size={14} className="animate-spin" /> : <RefreshCw size={14} />}{syncing ? "Indexing" : "Sync history"}</button></div></div><div className="grid divide-y divide-[#edf0ec] sm:grid-cols-3 sm:divide-x sm:divide-y-0"><Metric label="PRs scanned" value={String(repository.latest_sync?.pull_requests_seen ?? 0)} icon={<Github size={16} />} /><Metric label="Historical fixes" value={String(repository.latest_sync?.patch_cards_indexed ?? 0)} icon={<FileCode2 size={16} />} /><Metric label="Search chunks" value={String(repository.latest_sync?.chunks_indexed ?? 0)} icon={<Database size={16} />} /></div>{repository.latest_sync?.error && <div className="border-t border-amber-100 bg-amber-50 px-5 py-3 text-xs leading-5 text-amber-800">{repository.latest_sync.error}</div>}</section>;
}

function RepositoryCard({ repository, selected, onSelect, onEvaluate, onSync, onDelete }: { repository: Repository; selected: boolean; onSelect: () => void; onEvaluate: () => void; onSync: () => void; onDelete: () => void; }) {
  const syncing = isWorking(repository.latest_sync?.status);
  return <article className={`soft-shadow rounded-2xl border bg-white p-5 transition ${selected ? "border-moss ring-4 ring-[#e5f0e1]" : "border-[#dfe7dd]"}`}><div className="flex flex-col gap-5 sm:flex-row sm:items-start sm:justify-between"><div className="min-w-0"><div className="flex items-center gap-3"><span className="grid size-10 shrink-0 place-items-center rounded-xl bg-ink text-sage"><Github size={19} /></span><div className="min-w-0"><div className="flex items-center gap-2"><h2 className="truncate font-semibold">{repository.full_name}</h2><a href={repository.github_url} target="_blank" rel="noreferrer" className="text-[#779084] hover:text-moss"><ExternalLink size={14} /></a></div><p className="mt-1 line-clamp-2 text-sm leading-5 text-[#71877c]">{repository.description || `Default branch: ${repository.default_branch}`}</p></div></div><div className="mt-5 flex flex-wrap gap-2"><CompactMetric label="Fixes" value={String(repository.latest_sync?.patch_cards_indexed ?? 0)} /><CompactMetric label="Chunks" value={String(repository.latest_sync?.chunks_indexed ?? 0)} /><span className="inline-flex items-center gap-1.5 rounded-full bg-[#f1f5ef] px-2.5 py-1 text-[10px] font-bold uppercase tracking-[0.1em] text-[#698075]"><StatusDot status={repository.latest_sync?.status} /> {statusText(repository.latest_sync?.status)}</span></div></div><div className="flex shrink-0 flex-wrap gap-2"><button type="button" onClick={onSelect} className="rounded-lg bg-ink px-3 py-2 text-xs font-bold text-white hover:bg-[#1b4035]">Investigate</button><button type="button" onClick={onSync} disabled={syncing} className="rounded-lg bg-[#eef4ec] px-3 py-2 text-xs font-bold text-moss disabled:opacity-50">{syncing ? "Indexing…" : "Sync"}</button><button type="button" onClick={onEvaluate} className="rounded-lg bg-[#f3f6f2] px-3 py-2 text-xs font-bold text-[#547468]">Evaluate</button><button type="button" onClick={onDelete} className="rounded-lg px-2 py-2 text-[#a66a55] hover:bg-red-50" aria-label={`Remove ${repository.full_name}`}><Trash2 size={16} /></button></div></div></article>;
}

function InvestigationPanel({ investigation }: { investigation: Investigation }) {
  const [bestMatch, ...relatedMatches] = investigation.matches;
  if (!bestMatch) return <section className="soft-shadow mt-7 overflow-hidden rounded-2xl border border-[#dfe7dd] bg-white"><div className="px-6 py-10 text-center"><CircleAlert className="mx-auto text-[#8ba095]" size={24} /><p className="mt-3 font-semibold">No close historical fix found</p><p className="mx-auto mt-2 max-w-md text-sm leading-6 text-[#71877c]">Try including an error message, affected file, or a shorter description of the behavior. Patchwork will only recommend evidence from this repository.</p></div></section>;

  return <section className="soft-shadow mt-7 overflow-hidden rounded-2xl border border-[#dfe7dd] bg-white"><div className="border-b border-[#e7ece6] bg-[#fcfdfb] px-5 py-5 sm:px-6"><div className="flex flex-wrap items-center gap-2 text-xs font-bold uppercase tracking-[0.14em] text-moss"><Sparkles size={15} /> Suggested starting point <EvidencePill confidence={investigation.confidence} /></div><p className="mt-3 max-w-3xl text-sm leading-6 text-[#527064]">{resultGuidance(investigation.confidence)}</p></div><article className="px-5 py-6 sm:px-6"><div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between"><div className="min-w-0"><div className="flex flex-wrap items-center gap-2"><span className="rounded-md bg-[#eaf2e7] px-1.5 py-0.5 text-[10px] font-bold text-moss">BEST MATCH · PR #{bestMatch.pull_number ?? "?"}</span>{bestMatch.linked_issue_number && <span className="rounded-md bg-[#f3f6f2] px-1.5 py-0.5 text-[10px] font-bold text-[#698075]">Fixes issue #{bestMatch.linked_issue_number}</span>}</div><h2 className="mt-3 text-lg font-semibold tracking-tight text-ink">{cleanTitle(bestMatch.title)}</h2><p className="mt-3 max-w-3xl text-sm leading-6 text-[#5f776a]">{bestMatch.excerpt}</p></div><a href={bestMatch.pull_url} target="_blank" rel="noreferrer" className="inline-flex shrink-0 items-center justify-center gap-2 rounded-xl bg-ink px-3.5 py-2.5 text-xs font-bold text-white transition hover:bg-[#1b4035]">Open pull request <ExternalLink size={14} /></a></div><MatchEvidence match={bestMatch} /></article>{relatedMatches.length > 0 && <div className="border-t border-[#edf0ec] bg-[#fbfcfa] px-5 py-5 sm:px-6"><h3 className="text-sm font-semibold text-ink">Other related fixes</h3><p className="mt-1 text-xs text-[#789084]">Use these as secondary references if the closest fix does not fit.</p><div className="mt-4 grid gap-3">{relatedMatches.map((match, index) => <RelatedMatch key={`${match.pull_url}-${index}`} match={match} />)}</div></div>}</section>;
}

function MatchEvidence({ match }: { match: PatchMatch }) {
  const files = match.changed_files.slice(0, 4);
  const tests = match.test_files.slice(0, 3);
  if (!files.length && !tests.length) return null;
  return <><div className="mt-5 grid gap-4 rounded-xl border border-[#e7ece6] bg-[#f8faf7] p-4 sm:grid-cols-2"><EvidenceFiles label="Files to inspect" files={files} icon={<Code2 size={13} />} /><EvidenceFiles label="Regression coverage" files={tests} icon={<Check size={13} />} emptyText="No test file was detected." /><div className="sm:col-span-2"><RelevanceBadge score={match.score} /></div></div><SolutionPlan solution={match.recommended_solution} /></>;
}

function RelevanceBadge({ score }: { score: number }) {
  const percent = Math.round(Math.min(1, Math.max(0, score)) * 100);
  return <p className="text-xs text-[#71877c]"><span className="font-semibold text-[#527064]">Relevance: {percent}%</span><span className="ml-2">Similarity to your report, not a probability of a correct fix.</span></p>;
}

function SolutionPlan({ solution }: { solution?: RecommendedSolution }) {
  if (!solution) return null;
  const modelAssisted = solution.mode === "model_assisted";
  return <section className="mt-5 rounded-xl border border-[#dce9d8] bg-[#f5faf2] p-4"><div className="flex flex-wrap items-center justify-between gap-2"><h3 className="text-sm font-semibold text-ink">Suggested implementation approach</h3><span className={`rounded-full px-2 py-1 text-[10px] font-bold ${modelAssisted ? "bg-[#dbeee0] text-moss" : "bg-[#edf1eb] text-[#627a6d]"}`}>{modelAssisted ? "Model-assisted" : "Evidence-guided"}</span></div><p className="mt-2 text-sm leading-6 text-[#527064]">{solution.assessment}</p><PlanList label="Suggested steps" items={solution.implementation_steps} /><PlanList label="Verify" items={solution.verification} /><PlanList label="Keep in mind" items={solution.cautions} muted />{solution.code_guidance && <div className="mt-4"><p className="text-[10px] font-bold uppercase tracking-[0.12em] text-[#71877c]">Illustrative code guidance</p><pre className="mt-2 overflow-x-auto rounded-lg bg-[#17372f] p-3 text-xs leading-5 text-[#e9f4e5]"><code>{solution.code_guidance}</code></pre></div>}</section>;
}

function PlanList({ label, items, muted = false }: { label: string; items: string[]; muted?: boolean }) {
  if (!items.length) return null;
  return <div className="mt-4"><p className="text-[10px] font-bold uppercase tracking-[0.12em] text-[#71877c]">{label}</p><ul className={`mt-2 space-y-1.5 text-sm leading-5 ${muted ? "text-[#71877c]" : "text-[#527064]"}`}>{items.map((item, index) => <li key={`${label}-${index}`} className="flex gap-2"><span className="mt-2 size-1.5 shrink-0 rounded-full bg-sage" />{item}</li>)}</ul></div>;
}

function EvidenceFiles({ label, files, icon, emptyText }: { label: string; files: string[]; icon: ReactNode; emptyText?: string }) {
  return <div><p className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-[0.12em] text-[#71877c]">{icon}{label}</p>{files.length ? <div className="mt-2 flex flex-wrap gap-1.5">{files.map((file) => <span key={file} className="rounded-md bg-white px-2 py-1 font-mono text-[11px] text-[#527064] shadow-sm ring-1 ring-[#e1e9df]">{file}</span>)}</div> : emptyText ? <p className="mt-2 text-xs text-[#82958b]">{emptyText}</p> : null}</div>;
}

function RelatedMatch({ match }: { match: PatchMatch }) {
  return <article className="rounded-xl border border-[#e2e9e0] bg-white px-4 py-3"><div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between"><div className="min-w-0"><div className="flex items-center gap-2"><span className="rounded-md bg-[#eef4ec] px-1.5 py-0.5 text-[10px] font-bold text-moss">PR #{match.pull_number ?? "?"}</span><a href={match.pull_url} target="_blank" rel="noreferrer" className="truncate text-sm font-semibold text-ink hover:text-moss">{cleanTitle(match.title)}</a></div><p className="mt-2 line-clamp-2 text-sm leading-5 text-[#71877c]">{match.excerpt}</p></div><a href={match.pull_url} target="_blank" rel="noreferrer" className="shrink-0 text-xs font-bold text-moss hover:underline">Review <ArrowUpRight className="inline" size={13} /></a></div></article>;
}

function EvaluationReport({ evaluation }: { evaluation: EvaluationRun | null }) {
  if (!evaluation) return <section className="mt-7 rounded-2xl border border-[#dfe7dd] bg-white p-6"><div className="flex items-center gap-2 text-xs font-bold uppercase tracking-[0.14em] text-moss"><BarChart3 size={15} /> No evaluation yet</div><p className="mt-3 max-w-2xl text-sm leading-6 text-[#71877c]">Run an evaluation after syncing. Patchwork only uses verified links between an issue and the PR that closed it, so the metrics remain meaningful.</p></section>;
  const failures = evaluation.metrics.failure_samples ?? [];
  return <section className="mt-7 rounded-2xl border border-[#dfe7dd] bg-white p-6"><div className="flex flex-wrap items-center justify-between gap-3"><div><div className="flex items-center gap-2 text-xs font-bold uppercase tracking-[0.14em] text-moss"><BarChart3 size={15} /> Latest evaluation</div><p className="mt-2 text-sm text-[#71877c]">Run {formattedDate(evaluation.created_at)} · {evaluation.benchmark_cases} verified {evaluation.benchmark_cases === 1 ? "example" : "examples"}</p></div><StatusPill status={evaluation.status} /></div><p className="mt-5 max-w-3xl rounded-xl bg-[#f5f8f4] p-4 text-sm leading-6 text-[#587164]">{evaluation.metrics.note ?? "Metrics are available after the evaluation completes."}</p>{evaluation.error && <p className="mt-4 text-sm text-red-700">{evaluation.error}</p>}{failures.length > 0 && <div className="mt-5"><h3 className="text-sm font-semibold">Examples that need review</h3><div className="mt-3 space-y-2">{failures.map((failure, index) => <div key={`${failure.expected_pull_number}-${index}`} className="rounded-xl border border-amber-100 bg-amber-50 px-4 py-3 text-sm text-amber-900"><span className="font-semibold">Expected PR #{failure.expected_pull_number ?? "?"}</span><span className="mx-2 text-amber-400">·</span>{failure.query}</div>)}</div></div>}</section>;
}

function EvaluationReportCard({ evaluation }: { evaluation: EvaluationRun | null }) {
  if (!evaluation) {
    return <section className="mt-7 rounded-2xl border border-[#dfe7dd] bg-white p-6"><div className="flex items-center gap-2 text-xs font-bold uppercase tracking-[0.14em] text-moss"><BarChart3 size={15} /> No evaluation yet</div><p className="mt-3 max-w-2xl text-sm leading-6 text-[#71877c]">Run an evaluation after syncing. Patchwork prioritizes GitHub-confirmed issue-to-fix pairs and falls back to an explicitly non-reportable smoke test only when the repository has no linked issues.</p></section>;
  }
  const failures = evaluation.metrics.failure_samples ?? [];
  const verified = evaluation.metrics.quality === "verified_issue_to_fix";
  const breakdown = evaluation.metrics.case_breakdown;
  return <section className="mt-7 rounded-2xl border border-[#dfe7dd] bg-white p-6"><div className="flex flex-wrap items-center justify-between gap-3"><div><div className="flex items-center gap-2 text-xs font-bold uppercase tracking-[0.14em] text-moss"><BarChart3 size={15} /> Latest evaluation</div><p className="mt-2 text-sm text-[#71877c]">Run {formattedDate(evaluation.created_at)} · {evaluation.benchmark_cases} {verified ? "verified issue-to-fix" : "indexing smoke-test"} {evaluation.benchmark_cases === 1 ? "example" : "examples"}</p></div><span className={`rounded-full px-2.5 py-1 text-[10px] font-bold uppercase tracking-[0.1em] ${verified ? "bg-[#eaf4e5] text-moss" : "bg-amber-50 text-amber-700"}`}>{verified ? "Verified data" : "Smoke test"}</span></div><p className="mt-5 max-w-3xl rounded-xl bg-[#f5f8f4] p-4 text-sm leading-6 text-[#587164]">{evaluation.metrics.note ?? "Metrics are available after the evaluation completes."}</p>{breakdown && <div className="mt-4 flex flex-wrap gap-2 text-xs"><span className="rounded-lg bg-[#eef4ec] px-2.5 py-1.5 font-semibold text-moss">{breakdown.verified_issue_to_fix} verified pairs</span><span className="rounded-lg bg-amber-50 px-2.5 py-1.5 font-semibold text-amber-700">{breakdown.pr_description_smoke_test} smoke checks</span></div>}{evaluation.error && <p className="mt-4 text-sm text-red-700">{evaluation.error}</p>}{failures.length > 0 && <div className="mt-5"><h3 className="text-sm font-semibold">Examples that need review</h3><div className="mt-3 space-y-2">{failures.map((failure, index) => <div key={`${failure.expected_pull_number}-${index}`} className="rounded-xl border border-amber-100 bg-amber-50 px-4 py-3 text-sm text-amber-900"><span className="font-semibold">Expected PR #{failure.expected_pull_number ?? "?"}</span><span className="mx-2 text-amber-400">·</span>{failure.query}</div>)}</div></div>}</section>;
}

function AuthLoading() {
  return <main className="grid min-h-screen place-items-center bg-paper text-ink"><div className="flex items-center gap-3 text-sm font-semibold text-[#648276]"><LoaderCircle size={18} className="animate-spin" /> Loading your workspace…</div></main>;
}

function AuthScreen({ onAuthenticated }: { onAuthenticated: (session: Session) => void }) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setIsSubmitting(true);
    try {
      const session = await request<Session>(`/auth/${mode === "login" ? "login" : "register"}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      onAuthenticated(session);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not sign you in.");
    } finally {
      setIsSubmitting(false);
    }
  }

  return <main className="min-h-screen bg-paper px-5 py-8 text-ink"><div className="pointer-events-none fixed inset-0 grid-noise opacity-50" /><div className="relative mx-auto grid min-h-[calc(100vh-4rem)] max-w-5xl items-center gap-10 lg:grid-cols-[1fr_430px]"><section className="hidden lg:block"><div className="flex items-center gap-3"><div className="grid size-11 place-items-center rounded-2xl bg-ink text-sage"><GitBranch size={22} /></div><div><p className="text-xl font-semibold">Patchwork</p><p className="text-sm text-[#71877c]">Codebase memory</p></div></div><p className="mt-16 text-sm font-semibold uppercase tracking-[0.18em] text-moss">Repository intelligence</p><h1 className="mt-3 max-w-xl text-5xl font-semibold tracking-[-0.05em]">Keep your team’s debugging knowledge within reach.</h1><p className="mt-6 max-w-lg text-lg leading-8 text-[#6d8579]">Your repositories, sync history, and investigations are saved to your account and remain isolated from every other workspace.</p><div className="mt-10 grid max-w-lg gap-3"><AuthBenefit icon={<ShieldCheck size={17} />} text="Private, account-scoped repository history" /><AuthBenefit icon={<Search size={17} />} text="Saved investigations you can reopen later" /><AuthBenefit icon={<BookOpenCheck size={17} />} text="Grounded historical fixes and regression tests" /></div></section><section className="soft-shadow rounded-3xl border border-[#dfe7dd] bg-white p-6 sm:p-8"><div className="flex items-center gap-3 lg:hidden"><div className="grid size-10 place-items-center rounded-2xl bg-ink text-sage"><GitBranch size={20} /></div><span className="text-lg font-semibold">Patchwork</span></div><p className="mt-8 text-sm font-semibold uppercase tracking-[0.16em] text-moss">{mode === "login" ? "Welcome back" : "Create your workspace"}</p><h2 className="mt-2 text-3xl font-semibold tracking-tight">{mode === "login" ? "Sign in to Patchwork" : "Start saving your investigations"}</h2><p className="mt-3 text-sm leading-6 text-[#71877c]">{mode === "login" ? "Access your connected repositories and debugging history." : "Use an email and a password of at least 10 characters."}</p><form onSubmit={submit} className="mt-7 space-y-4"><label className="block text-sm font-semibold">Email<input type="email" autoComplete="email" required value={email} onChange={(event) => setEmail(event.target.value)} className="mt-2 w-full rounded-xl border border-[#d9e3d8] bg-[#fbfcfa] px-3 py-3 outline-none focus:border-moss focus:ring-4 focus:ring-[#dfeee0]" placeholder="you@example.com" /></label><label className="block text-sm font-semibold">Password<input type="password" autoComplete={mode === "login" ? "current-password" : "new-password"} required minLength={10} value={password} onChange={(event) => setPassword(event.target.value)} className="mt-2 w-full rounded-xl border border-[#d9e3d8] bg-[#fbfcfa] px-3 py-3 outline-none focus:border-moss focus:ring-4 focus:ring-[#dfeee0]" placeholder="At least 10 characters" /></label>{error && <div className="flex gap-2 rounded-xl border border-red-200 bg-red-50 px-3 py-3 text-sm text-red-700"><CircleAlert size={17} className="shrink-0" />{error}</div>}<button type="submit" disabled={isSubmitting} className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-ink px-4 py-3.5 text-sm font-bold text-white transition hover:bg-[#1b4035] disabled:opacity-50">{isSubmitting ? <LoaderCircle size={17} className="animate-spin" /> : <GitBranch size={17} />}{isSubmitting ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"}</button></form><p className="mt-6 text-center text-sm text-[#71877c]">{mode === "login" ? "New to Patchwork?" : "Already have an account?"} <button type="button" onClick={() => { setMode(mode === "login" ? "register" : "login"); setError(null); }} className="font-bold text-moss hover:underline">{mode === "login" ? "Create an account" : "Sign in"}</button></p></section></div></main>;
}

function AuthBenefit({ icon, text }: { icon: ReactNode; text: string }) { return <div className="flex items-center gap-3 rounded-xl border border-[#dfe7dd] bg-white/70 px-4 py-3 text-sm font-medium text-[#527064]"><span className="text-moss">{icon}</span>{text}</div>; }

function GitHubAccess({ connection, onLink, onUnlink }: { connection: GitHubConnection; onLink: () => void; onUnlink: () => void; }) {
  if (connection.connected) return <button type="button" onClick={onUnlink} className="inline-flex items-center gap-1.5 border-r border-[#dce4da] pr-2 text-moss hover:text-ink" title="Unlink GitHub"><Github size={14} /><span className="hidden max-w-24 truncate sm:inline">{connection.github_login}</span></button>;
  return <button type="button" onClick={onLink} className="inline-flex items-center gap-1.5 border-r border-[#dce4da] pr-2 text-moss hover:text-ink"><Github size={14} /><span className="hidden sm:inline">Link GitHub</span></button>;
}

function EmptyState({ loading }: { loading: boolean }) { return <section className="soft-shadow mt-7 rounded-2xl border border-dashed border-[#ccdace] bg-white/70 px-6 py-16 text-center"><div className="mx-auto grid size-14 place-items-center rounded-2xl bg-[#eaf2e7] text-moss"><GitBranch size={24} /></div><h2 className="mt-5 text-lg font-semibold">{loading ? "Loading your workspace…" : "Connect your first repository"}</h2><p className="mx-auto mt-2 max-w-md text-sm leading-6 text-[#71877c]">Paste a public GitHub repository URL above. Patchwork will turn its historical fixes into a searchable reference library.</p></section>; }
function NavItem({ icon, label, count, active = false, onClick }: { icon: ReactNode; label: string; count?: number; active?: boolean; onClick: () => void; }) { return <button type="button" onClick={onClick} className={`flex w-full items-center gap-3 rounded-xl px-3 py-3 text-left transition ${active ? "bg-white/12 text-white shadow-sm" : "text-[#b9cbbf] hover:bg-white/6 hover:text-white"}`}>{icon}<span className="flex-1 font-medium">{label}</span>{typeof count === "number" && <span className="rounded-md bg-white/10 px-1.5 py-0.5 text-[10px] font-bold">{count}</span>}</button>; }
function MobileNavItem({ label, active, onClick }: { label: string; active: boolean; onClick: () => void; }) { return <button type="button" onClick={onClick} className={`rounded-xl border px-3 py-2.5 text-xs font-bold transition ${active ? "border-moss bg-moss text-white" : "border-[#dce4da] bg-white text-[#648276]"}`}>{label}</button>; }
function Metric({ icon, label, value }: { icon: ReactNode; label: string; value: string }) { return <div className="px-5 py-4"><div className="flex items-center gap-2 text-[#789084]">{icon}<span className="text-[11px] font-bold uppercase tracking-[0.1em]">{label}</span></div><p className="mt-3 text-2xl font-semibold tracking-tight text-ink">{value}</p></div>; }
function CompactMetric({ label, value }: { label: string; value: string }) { return <span className="rounded-lg bg-[#f3f7f1] px-2.5 py-1.5 text-xs font-semibold text-[#587164]"><span className="text-[#82958b]">{label}</span> {value}</span>; }
function StatusDot({ status }: { status?: string }) { return <span className={`size-2 rounded-full ${status === "completed" ? "bg-sage" : status === "failed" ? "bg-red-400" : status ? "bg-amber-400" : "bg-[#6d887a]"}`} />; }
function StatusPill({ status }: { status?: string }) { const tone = status === "completed" ? "bg-[#eaf4e5] text-moss" : status === "failed" ? "bg-red-50 text-red-700" : status ? "bg-amber-50 text-amber-700" : "bg-[#f0f3ef] text-[#809287]"; return <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[10px] font-bold uppercase tracking-[0.1em] ${tone}`}><StatusDot status={status} />{statusText(status)}</span>; }
function EvidencePill({ confidence }: { confidence: Investigation["confidence"] }) { const label = confidence === "high" ? "Strong evidence" : confidence === "moderate" ? "Relevant evidence" : confidence === "low" ? "Limited evidence" : "No match"; const tone = confidence === "high" ? "bg-[#eaf4e5] text-moss" : confidence === "moderate" ? "bg-[#edf4f8] text-[#41718e]" : confidence === "low" ? "bg-amber-50 text-amber-700" : "bg-[#f0f3ef] text-[#809287]"; return <span className={`rounded-full px-2 py-1 text-[10px] font-bold normal-case tracking-normal ${tone}`}>{label}</span>; }
function Notice({ type, children }: { type: "success" | "error"; children: ReactNode }) { const isError = type === "error"; return <div className={`mt-6 flex items-start gap-3 rounded-xl border px-4 py-3 text-sm ${isError ? "border-red-200 bg-red-50 text-red-700" : "border-[#cfe3c4] bg-[#eef6e9] text-moss"}`}>{isError ? <CircleAlert size={17} className="mt-0.5 shrink-0" /> : <Check size={17} className="mt-0.5 shrink-0" />}{children}</div>; }
function cleanTitle(title: string) { return title.replace(/^PR #\d+\s*·\s*/, ""); }
function resultGuidance(confidence: Investigation["confidence"]) { if (confidence === "high") return "This fix is supported by multiple details in your report and includes regression coverage. Start here, then confirm the behavior in your current code."; if (confidence === "moderate") return "This is a useful starting point with meaningful overlap. Compare the affected code before carrying its approach forward."; if (confidence === "low") return "This is a possible lead rather than a recommendation. Verify the behavior and tests before relying on it."; return "No indexed historical fix was similar enough to recommend."; }
