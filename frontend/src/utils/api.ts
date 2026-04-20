const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export interface PredictionResult {
  year: number;
  predicted_winner: string;
  confidence: number;
  reasoning: string;
  top_teams: { team: string; probability: number; reasoning: string; momentum?: string[] }[];
  key_factors: string[];
  historical_context: string;
  retrieved_chunks: number;
  cached: boolean;
}

export interface TeamStat {
  year: number;
  team: string;
  matches: number;
  wins: number;
  losses: number;
  points: number;
  nrr: number;
  position: number;
  qualified_playoffs: boolean;
  won_title: boolean;
  momentum?: string[]; // Added: W-W-L sequence
}

export interface SyncStatus {
  last_sync: string;
  is_syncing: boolean;
  total_syncs: number;
}

export interface Season {
  year: number;
  winner: string;
  teams: number;
  momentum?: string[];
}

export interface TitleCount {
  team: string;
  titles: number;
  years: number[];
}

export const api = {
  predict: (year: number, forceRefresh = false) =>
    apiFetch<PredictionResult>(`/predict/${year}?force_refresh=${forceRefresh}`),

  seasons: () => apiFetch<Season[]>("/stats/seasons"),

  seasonStats: (year: number) =>
    apiFetch<{ year: number; teams: TeamStat[] }>(`/stats/${year}`),

  teamHistory: (team: string) =>
    apiFetch<{ team: string; titles: number; seasons: TeamStat[] }>(`/stats/team/${encodeURIComponent(team)}`),

  h2h: (team1: string, team2: string) =>
    apiFetch<any>(`/stats/h2h/${encodeURIComponent(team1)}/${encodeURIComponent(team2)}`),

  allTitles: () => apiFetch<TitleCount[]>("/stats/titles/all"),

  matches: (year: number) => apiFetch<any[]>(`/stats/matches/${year}`),

  ask: (question: string, year?: number) =>
    apiFetch<{ question: string; answer: string }>("/ask", {
      method: "POST",
      body: JSON.stringify({ question, year }),
    }),

  syncStatus: () => apiFetch<SyncStatus>("/stats/sync-status"),
};
