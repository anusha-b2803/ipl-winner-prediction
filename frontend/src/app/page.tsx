"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, Cell, RadarChart, PolarGrid, PolarAngleAxis, Radar
} from "recharts";
import {
  Trophy, Zap, MessageSquare, RefreshCw,
  ChevronDown, Activity, Award, Shield,
  Target, TrendingUp, History, Send,
  Star, BarChart2, Flame, Clock, Check
} from "lucide-react";
import { api, PredictionResult, TeamStat, Season, TitleCount, SyncStatus } from "@/utils/api";

// ─── Team palette ─────────────────────────────────────────────────────────────
const TEAM_META: Record<string, { color: string; accent: string; abbr: string }> = {
  "Mumbai Indians":           { color: "#004BA0", accent: "#6EB4FF", abbr: "MI" },
  "Chennai Super Kings":      { color: "#FCCE00", accent: "#FFE566", abbr: "CSK" },
  "Kolkata Knight Riders":    { color: "#3A225D", accent: "#B084FF", abbr: "KKR" },
  "Royal Challengers Bangalore": { color: "#CC0000", accent: "#FF6B6B", abbr: "RCB" },
  "Royal Challengers Bengaluru": { color: "#CC0000", accent: "#FF6B6B", abbr: "RCB" },
  "Delhi Capitals":           { color: "#0057A8", accent: "#4C9EEB", abbr: "DC" },
  "Sunrisers Hyderabad":      { color: "#FF6B00", accent: "#FFAA66", abbr: "SRH" },
  "Rajasthan Royals":         { color: "#E91E8C", accent: "#FF88D0", abbr: "RR" },
  "Punjab Kings":             { color: "#C8102E", accent: "#FF6680", abbr: "PBKS" },
  "Gujarat Titans":           { color: "#1C2B4B", accent: "#6FA8D6", abbr: "GT" },
  "Lucknow Super Giants":     { color: "#004FA3", accent: "#5FA3EB", abbr: "LSG" },
  "Deccan Chargers":          { color: "#1A237E", accent: "#7986CB", abbr: "DC2" },
};

const teamMeta = (t: string) => TEAM_META[t] ?? { color: "#475569", accent: "#94A3B8", abbr: t.slice(0, 3).toUpperCase() };

// ─── Animation variants ───────────────────────────────────────────────────────
const fadeUp = {
  initial: { opacity: 0, y: 24 },
  animate: { opacity: 1, y: 0 },
  exit:    { opacity: 0, y: -16 },
  transition: { duration: 0.45, ease: [0.4, 0, 0.2, 1] },
};
const stagger = { animate: { transition: { staggerChildren: 0.08 } } };
const fadeItem = {
  initial: { opacity: 0, y: 16 },
  animate: { opacity: 1, y: 0 },
};

// ─── Confidence Gauge ─────────────────────────────────────────────────────────
function ConfidenceGauge({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const circumference = 2 * Math.PI * 54;
  const offset = circumference - circumference * value;
  const color = value > 0.78 ? "#F5A623" : value > 0.6 ? "#34D399" : "#94A3B8";
  const label = value > 0.78 ? "High" : value > 0.6 ? "Moderate" : "Low";

  return (
    <div className="flex flex-col items-center gap-3">
      <div className="relative w-36 h-36">
        {/* Outer decorative ring */}
        <svg className="absolute inset-0 w-full h-full animate-spin-slow opacity-20" viewBox="0 0 144 144">
          <circle cx="72" cy="72" r="68" fill="none" stroke={color} strokeWidth="1"
            strokeDasharray="8 14" strokeLinecap="round" />
        </svg>
        {/* Main gauge */}
        <svg className="w-full h-full -rotate-90" viewBox="0 0 144 144">
          <circle cx="72" cy="72" r="54" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="10" />
          <motion.circle
            cx="72" cy="72" r="54"
            fill="none" stroke={color} strokeWidth="10"
            strokeDasharray={circumference}
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset: offset }}
            transition={{ duration: 1.8, ease: "easeOut" }}
            strokeLinecap="round"
            style={{ filter: `drop-shadow(0 0 8px ${color}88)` }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <motion.span
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.6 }}
            className="text-4xl font-black text-white stat-num"
          >{pct}%</motion.span>
          <span className="text-[9px] uppercase tracking-[0.25em] font-bold" style={{ color }}>{label}</span>
        </div>
      </div>
      <p className="text-[10px] uppercase tracking-widest font-bold text-white/30">AI Confidence</p>
    </div>
  );
}

// ─── Team Pill / Avatar ───────────────────────────────────────────────────────
function TeamAvatar({ team, size = "md" }: { team: string; size?: "sm" | "md" | "lg" }) {
  const meta = teamMeta(team);
  const sizes = { sm: "w-8 h-8 text-[10px]", md: "w-10 h-10 text-xs", lg: "w-14 h-14 text-sm" };
  return (
    <div
      className={`${sizes[size]} rounded-2xl flex items-center justify-center font-black shrink-0`}
      style={{ background: `linear-gradient(135deg, ${meta.color}cc, ${meta.color}55)`, border: `1px solid ${meta.color}66` }}
    >
      <span style={{ color: meta.accent }}>{meta.abbr}</span>
    </div>
  );
}

// ─── Stat Card ────────────────────────────────────────────────────────────────
function StatCard({ label, value, icon: Icon, color, sub }: {
  label: string; value: string | number; icon: any; color: string; sub?: string;
}) {
  return (
    <motion.div variants={fadeItem} className="card p-6 flex items-center gap-5 group">
      <div className="p-3.5 rounded-2xl shrink-0 transition-all duration-300 group-hover:scale-110"
        style={{ background: `${color}18`, border: `1px solid ${color}30` }}>
        <Icon size={22} style={{ color }} />
      </div>
      <div>
        <p className="text-[10px] font-bold text-white/35 uppercase tracking-widest mb-1">{label}</p>
        <p className="text-2xl font-black text-white stat-num leading-none">{value}</p>
        {sub && <p className="text-[11px] text-white/30 mt-1">{sub}</p>}
      </div>
    </motion.div>
  );
}

// ─── Key Factor Pills ─────────────────────────────────────────────────────────
function KeyFactors({ factors }: { factors: string[] }) {
  return (
    <div className="flex flex-wrap gap-2 mt-4">
      {factors.map((f, i) => (
        <motion.span
          key={i} initial={{ opacity: 0, scale: 0.85 }} animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 + i * 0.08 }}
          className="factor-pill"
        >
          <span className="text-ipl-gold opacity-70">◆</span> {f}
        </motion.span>
      ))}
    </div>
  );
}

// ─── Momentum Pill ────────────────────────────────────────────────────────────
function MomentumPill({ sequence }: { sequence: string[] }) {
  if (!sequence || sequence.length === 0) return null;
  return (
    <div className="flex gap-1 mt-1">
      {sequence.slice(0, 5).map((res, i) => {
        const isWin = res === "W";
        const isNR = res === "N";
        return (
          <span
            key={i}
            className={`w-3.5 h-3.5 rounded-full flex items-center justify-center text-[7px] font-black
              ${isWin ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30" : 
                isNR ? "bg-white/10 text-white/50 border border-white/20" : 
                "bg-rose-500/20 text-rose-400 border border-rose-500/30"}`}
          >
            {res}
          </span>
        );
      })}
    </div>
  );
}

// ─── Chat Message ─────────────────────────────────────────────────────────────
function ChatBubble({ text, isAi }: { text: string; isAi: boolean }) {
  return (
    <motion.div {...fadeUp} className={`flex gap-3 ${isAi ? "" : "flex-row-reverse"}`}>
      <div className={`w-7 h-7 rounded-xl flex items-center justify-center shrink-0 text-[10px] font-black
        ${isAi ? "bg-ipl-gold/10 border border-ipl-gold/20 text-ipl-gold" : "bg-white/5 border border-white/10 text-white/50"}`}>
        {isAi ? "AI" : "U"}
      </div>
      <div className={`max-w-[85%] px-5 py-3.5 rounded-2xl text-sm leading-relaxed
        ${isAi
          ? "bg-white/[0.04] border border-white/[0.07] text-white/80"
          : "bg-ipl-gold/10 border border-ipl-gold/15 text-white/70"
        }`}>
        {cleanText(text)}
      </div>
    </motion.div>
  );
}

// ─── Utilities ────────────────────────────────────────────────────────────────
const cleanText = (txt: string) => {
  if (!txt) return "";
  return txt
    .replace(/#{1,6}\s?/g, "") // Remove headers
    .replace(/\*{2}/g, "")      // Remove bold markers
    .replace(/\*/g, "")        // Remove bullet/italic markers
    .replace(/`{1,3}/g, "")    // Remove code ticks
    .trim();
};

// ─── Custom Tooltip ───────────────────────────────────────────────────────────
function ChartTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-[#0C1525] border border-white/10 rounded-2xl px-4 py-3 text-sm shadow-2xl">
      <p className="font-bold text-white/70">{payload[0]?.payload?.team}</p>
      <p className="text-ipl-gold font-black">{payload[0]?.value} pts</p>
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function Home() {
  const [selectedYear, setSelectedYear] = useState(2026);
  const [tab, setTab] = useState<"predict" | "stats" | "history">("predict");
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [standings, setStandings]   = useState<TeamStat[]>([]);
  const [seasons, setSeasons]       = useState<Season[]>([]);
  const [titles, setTitles]         = useState<TitleCount[]>([]);
  const [loading, setLoading]       = useState(false);
  const [messages, setMessages]     = useState<{ text: string; isAi: boolean }[]>([]);
  const [aiLoading, setAiLoading]   = useState(false);
  const [question, setQuestion]     = useState("");
  const [isYearPickerOpen, setIsYearPickerOpen] = useState(false);
  const [syncStatus, setSyncStatus] = useState<SyncStatus | null>(null);
  
  // Scoped Refs
  const chatRef = useRef<HTMLDivElement>(null);
  const yearPickerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.seasons().then(setSeasons);
    api.allTitles().then(setTitles);
    api.syncStatus().then(setSyncStatus);
  }, []);

  // Handle click outside year picker
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (yearPickerRef.current && !yearPickerRef.current.contains(e.target as Node)) {
        setIsYearPickerOpen(false);
      }
    };
    if (isYearPickerOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [isYearPickerOpen]);

  // Auto-scroll chat
  useEffect(() => {
    chatRef.current?.scrollTo({ top: chatRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const fetchPrediction = useCallback(async (force = false) => {
    setLoading(true);
    setPrediction(null);
    try {
      const res = await api.predict(selectedYear, force);
      setPrediction(res);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }, [selectedYear]);

  const fetchStats = useCallback(async () => {
    try {
      const res = await api.seasonStats(selectedYear);
      setStandings(res.teams);
    } catch { setStandings([]); }
  }, [selectedYear]);

  useEffect(() => {
    if (tab === "predict") fetchPrediction();
    if (tab === "stats")   fetchStats();
  }, [selectedYear, tab, fetchPrediction, fetchStats]);

  const askAi = async () => {
    if (!question.trim() || aiLoading) return;
    const q = question.trim();
    setQuestion("");
    setMessages(m => [...m, { text: q, isAi: false }]);
    setAiLoading(true);
    try {
      const res = await api.ask(q, selectedYear);
      setMessages(m => [...m, { text: res.answer, isAi: true }]);
    } catch {
      setMessages(m => [...m, { text: "Failed to reach AI analyst. Please try again.", isAi: true }]);
    } finally {
      setAiLoading(false);
    }
  };

  const years = seasons.length > 0
    ? [...new Set(seasons.map(s => s.year))].sort((a, b) => b - a)
    : [2026, 2025, 2024];

  const currentYear = new Date().getFullYear();
  const isLiveSeason = selectedYear >= currentYear;

  // ─── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="relative min-h-screen text-slate-200">

      {/* Animated background */}
      <div className="bg-mesh">
        <div className="bg-orb bg-orb-1" />
        <div className="bg-orb bg-orb-2" />
        <div className="bg-orb bg-orb-3" />
        <div className="bg-grid" />
      </div>

      {/* ── Navbar ──────────────────────────────────────────────────────────── */}
      <nav className="sticky top-0 z-50 py-4 px-6 border-b border-white/[0.06] backdrop-blur-3xl bg-black/30">
        <div className="max-w-7xl mx-auto flex items-center justify-between gap-6">
          {/* Logo */}
          <div className="flex items-center gap-4 shrink-0">
            <div className="relative">
              <div className="w-11 h-11 rounded-[14px] bg-gradient-to-br from-amber-400 to-orange-500
                              flex items-center justify-center shadow-lg shadow-amber-500/30 animate-float">
                <Trophy size={20} className="text-black" />
              </div>
            </div>
            <div>
              <h1 className="text-[1.1rem] font-black tracking-tighter text-white uppercase font-display">
                IPL <span className="gradient-text">Prophet</span>
              </h1>
              <p className="text-[9px] uppercase tracking-[0.22em] font-bold text-white/25">
                RAG · Transformer · Gemini
              </p>
            </div>
          </div>

          {/* Centre — Live badge */}
          {isLiveSeason && (
            <div className="hidden md:flex items-center gap-2 badge badge-green px-4 py-1.5">
              <span className="live-dot" />
              <span>Live Season {selectedYear}</span>
            </div>
          )}

          {/* Year picker */}
          <div className="flex items-center gap-3">
            <div className="hidden sm:block h-6 w-px bg-white/10 mx-2" />
                
                {/* Sync Status Badge */}
                {syncStatus && (
                  <div className="hidden lg:flex items-center gap-2 px-3 py-1.5 rounded-xl bg-white/5 border border-white/10">
                    <div className={`w-1.5 h-1.5 rounded-full ${syncStatus.is_syncing ? "bg-amber-400 animate-pulse" : "bg-emerald-400"}`} />
                    <div className="flex flex-col">
                      <span className="text-[9px] uppercase tracking-tighter font-bold text-white/40">Live Sync</span>
                      <span className="text-[10px] font-medium text-white/70">
                        {syncStatus.is_syncing ? "Syncing..." : syncStatus.last_sync === "Never" ? "Pending" : `${syncStatus.last_sync.split(' ')[1]}`}
                      </span>
                    </div>
                  </div>
                )}

                <div className="relative" ref={yearPickerRef}>
                  <button
                    onClick={() => setIsYearPickerOpen(!isYearPickerOpen)}
                    className="flex items-center gap-3 bg-white/[0.05] border border-white/[0.1] text-white text-sm
                               pl-5 pr-4 py-2.5 rounded-2xl focus:outline-none focus:ring-2 focus:ring-amber-500/30
                               cursor-pointer transition-all hover:bg-white/[0.08]"
                  >
                    <span className="font-bold stat-num">{selectedYear}</span>
                    <ChevronDown size={14} className={`text-white/30 transition-transform duration-300 ${isYearPickerOpen ? "rotate-180" : ""}`} />
                  </button>

                  <AnimatePresence>
                    {isYearPickerOpen && (
                      <motion.div
                        initial={{ opacity: 0, y: 10, scale: 0.95 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: 10, scale: 0.95 }}
                        className="absolute right-0 mt-3 w-64 bg-[#0C1525] border border-white/10 rounded-[1.75rem] shadow-2xl p-4 z-[60] backdrop-blur-3xl"
                      >
                        <p className="text-[9px] uppercase tracking-widest font-black text-white/20 mb-4 px-2">Select Season</p>
                        <div className="grid grid-cols-3 gap-2">
                          {years.map(y => (
                            <button
                              key={y}
                              onClick={() => {
                                setSelectedYear(y);
                                setIsYearPickerOpen(false);
                              }}
                              className={`py-2.5 rounded-xl text-xs font-bold transition-all stat-num
                                ${selectedYear === y 
                                  ? "bg-ipl-gold text-black shadow-lg shadow-ipl-gold/20" 
                                  : "bg-white/[0.03] border border-white/5 text-white/50 hover:bg-white/10 hover:text-white"}`}
                            >
                              {y}
                            </button>
                          ))}
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-12">

        {/* ── Hero header ──────────────────────────────────────────────────── */}
        <motion.header
          initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
          className="mb-14 text-center"
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.15 }}
            className="badge badge-gold mx-auto mb-6"
          >
            <Zap size={10} fill="currentColor" /> RAG Intelligence Active
          </motion.div>

          <h2 className="text-5xl sm:text-7xl md:text-8xl font-black text-white mb-5 font-display leading-[0.88] tracking-tight">
            {selectedYear > new Date().getFullYear() ? "Who Will\nConquer" : "The Predicted\nChampion of"}
            <br />
            <span className="gradient-text italic">IPL {selectedYear}</span>
          </h2>
          <p className="text-base text-white/40 max-w-xl mx-auto leading-relaxed">
            18 seasons of IPL data, vector search, and transformer analysis combined to project the most likely champion.
          </p>
        </motion.header>

        {/* ── Tabs ─────────────────────────────────────────────────────────── */}
        <div className="flex gap-1.5 p-1.5 bg-white/[0.04] border border-white/[0.06] rounded-2xl
                        w-fit mx-auto mb-12 backdrop-blur-xl">
          {(["predict", "stats", "history"] as const).map(t => (
            <button key={t} onClick={() => setTab(t)} className={`tab ${tab === t ? "active" : ""}`}>
              {t === "predict" && <Zap size={12} className="inline mr-1.5 -mt-0.5" />}
              {t === "stats"   && <BarChart2 size={12} className="inline mr-1.5 -mt-0.5" />}
              {t === "history" && <History size={12} className="inline mr-1.5 -mt-0.5" />}
              <span className="capitalize">{t}</span>
            </button>
          ))}
        </div>

        {/* ── Tab content ──────────────────────────────────────────────────── */}
        <AnimatePresence mode="wait">

          {/* ━━━ PREDICT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
          {tab === "predict" && (
            <motion.div key="predict" {...fadeUp} className="space-y-8">

              {loading ? (
                <div className="space-y-6">
                  <div className="h-80 rounded-[1.75rem] shimmer" />
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
                    {[1,2,3].map(i => <div key={i} className="h-28 rounded-[1.75rem] shimmer" />)}
                  </div>
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div className="h-72 rounded-[1.75rem] shimmer" />
                    <div className="h-72 rounded-[1.75rem] shimmer" />
                  </div>
                </div>
      ) : prediction ? (
                <>
                  {/* ── Champion Retrospective (for past years) ─────────────── */}
                  {selectedYear < 2026 && seasons.find(s => s.year === selectedYear)?.winner && (
                    <motion.div initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }}
                      className="card-hero mb-8 border-ipl-gold/30 !from-ipl-gold/5 !to-transparent overflow-hidden relative">
                      <div className="absolute -right-20 -top-20 opacity-[0.03] rotate-12">
                        <Trophy size={400} />
                      </div>
                      <div className="relative z-10 flex flex-col md:flex-row items-center gap-8 py-8 px-10">
                        <div className="shrink-0">
                          <div className="w-40 h-40 rounded-full bg-gradient-to-br from-ipl-gold to-orange-400 p-1 animate-pulse-slow">
                            <div className="w-full h-full rounded-full bg-[#0C1525] flex items-center justify-center p-6">
                              <Trophy size={64} className="text-ipl-gold" />
                            </div>
                          </div>
                        </div>
                        <div className="text-center md:text-left">
                          <div className="badge badge-gold mb-4 mx-auto md:mx-0 font-black tracking-[.25em]">CHAMPION DECIDED</div>
                          <h3 className="text-4xl md:text-6xl font-black text-white mb-2 leading-none uppercase tracking-tighter">
                            {seasons.find(s => s.year === selectedYear)?.winner}
                          </h3>
                          <p className="text-lg text-white/50 font-bold max-w-xl">
                            The {selectedYear} season has concluded with {seasons.find(s => s.year === selectedYear)?.winner} emerging as the ultimate champions.
                          </p>
                          <div className="flex gap-1.5 mt-6 justify-center md:justify-start">
                            <span className="factor-pill !bg-ipl-gold/10 !border-ipl-gold/20 !text-ipl-gold font-black">1 CHAMPIONSHIP TITLE</span>
                            <span className="factor-pill">SEASON STATS INCLUDED BELOW</span>
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  )}

                  {/* ── Winner Hero (Prediction Mode) ────────────────────────── */}
                  <div className="card-hero p-8 md:p-12">
                    <div className="flex flex-col md:flex-row items-start md:items-center gap-10">

                      {/* Confidence gauge */}
                      <div className="shrink-0 flex flex-col items-center">
                        <ConfidenceGauge value={prediction.confidence} />
                        <div className="mt-4 flex items-center gap-2">
                          {prediction.cached
                            ? <span className="badge badge-blue"><Clock size={9} />Cached</span>
                            : <span className="badge badge-green"><Check size={9} />Live</span>}
                          <span className="badge badge-gold">{prediction.retrieved_chunks} contexts</span>
                        </div>
                      </div>

                      <div className="divider hidden md:block w-px h-36 !bg-white/[0.07] !h-auto" style={{ width: 1, minHeight: 140, background: 'rgba(255,255,255,0.07)', borderRadius: 2 }} />

                      {/* Winner info */}
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-4">
                          <TeamAvatar team={prediction.predicted_winner} size="lg" />
                          <div>
                            <p className="text-[10px] font-bold uppercase tracking-[0.25em] text-white/35 mb-1">
                              {prediction.year > new Date().getFullYear() ? "Projected Champion" : "Predicted Champion"}
                            </p>
                            <h3 className="text-3xl md:text-4xl font-black text-white leading-tight tracking-tight">
                              {prediction.predicted_winner}
                            </h3>
                          </div>
                        </div>

                        <blockquote className="text-[0.95rem] text-white/65 leading-relaxed border-l-2 pl-5
                                               border-amber-500/50 mb-5">
                          {cleanText(prediction.reasoning)}
                        </blockquote>

                        {/* Historical context */}
                        {prediction.historical_context && (
                          <p className="text-xs text-white/35 italic mb-4">
                            📜 {prediction.historical_context}
                          </p>
                        )}

                        {/* Key factors */}
                        {prediction.key_factors?.length > 0 && (
                          <div>
                            <p className="text-[10px] font-bold uppercase tracking-widest text-white/30 mb-3">Key Factors</p>
                            <KeyFactors factors={prediction.key_factors} />
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Force refresh */}
                    <div className="mt-8 flex justify-end">
                      <button onClick={() => fetchPrediction(true)}
                        className="btn-ghost flex items-center gap-2 text-xs">
                        <RefreshCw size={13} /> Refresh Prediction
                      </button>
                    </div>
                  </div>

                  {/* ── Quick stats ────────────────────────────────────────── */}
                  <motion.div variants={stagger} animate="animate"
                    className="grid grid-cols-1 sm:grid-cols-3 gap-5">
                    <StatCard label="Win Probability"
                      value={`${Math.round((prediction.top_teams[0]?.probability ?? 0) * 100)}%`}
                      icon={TrendingUp} color="#F5A623"
                      sub={prediction.top_teams[0]?.team} />
                    <StatCard label="Contexts Retrieved"
                      value={prediction.retrieved_chunks}
                      icon={Activity} color="#34D399"
                      sub="Vector + SQL chunks" />
                    <StatCard label="Prediction Mode"
                      value={prediction.cached ? "Cached" : "Live"}
                      icon={Flame} color="#60A5FA"
                      sub={prediction.cached ? `Expires in ~6h` : "Just computed"} />
                  </motion.div>

                  {/* ── Contenders + Chat ──────────────────────────────────── */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

                    {/* Contenders */}
                    <div className="card p-8">
                      <h4 className="text-[11px] font-black uppercase tracking-[0.22em] text-white/30 mb-7
                                     flex items-center gap-2">
                        <Target size={13} /> Top Contenders
                      </h4>
                      <div className="space-y-7">
                        {prediction.top_teams.map((team, idx) => {
                          const meta = teamMeta(team.team);
                          return (
                            <div key={team.team}>
                              <div className="flex items-center gap-3 mb-2">
                                <TeamAvatar team={team.team} size="sm" />
                                <div className="flex-1">
                                  <div className="flex items-center justify-between">
                                    <span className="font-bold text-white text-sm">{team.team}</span>
                                    <span className="font-black text-sm stat-num" style={{ color: meta.accent }}>
                                      {Math.round(team.probability * 100)}%
                                    </span>
                                  </div>
                                  {team.momentum && <MomentumPill sequence={team.momentum as any} />}
                                </div>
                              </div>
                              <div className="prob-track ml-10">
                                <motion.div
                                  initial={{ width: 0 }}
                                  animate={{ width: `${team.probability * 100}%` }}
                                  transition={{ duration: 1.1, delay: idx * 0.15, ease: "easeOut" }}
                                  className="h-full rounded-full"
                                  style={{ background: `linear-gradient(90deg, ${meta.color}, ${meta.accent})` }}
                                />
                              </div>
                              {team.reasoning && (
                                <p className="text-[11px] text-white/25 mt-1.5 ml-10 italic leading-snug">
                                  {team.reasoning}
                                </p>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    </div>

                    {/* AI Chat */}
                    <div className="card p-8 flex flex-col">
                      <h4 className="text-[11px] font-black uppercase tracking-[0.22em] text-white/30 mb-6
                                     flex items-center gap-2">
                        <MessageSquare size={13} /> AI Analyst Desk
                      </h4>

                      {/* Messages */}
                      <div ref={chatRef}
                        className="flex-1 space-y-4 mb-5 overflow-y-auto pr-1 min-h-[180px] max-h-[260px]">
                        {messages.length === 0 ? (
                          <div className="h-full flex flex-col items-center justify-center gap-3 text-center py-6">
                            <div className="w-12 h-12 rounded-2xl bg-ipl-gold/10 border border-ipl-gold/20
                                            flex items-center justify-center">
                              <Star size={20} className="text-ipl-gold/60" />
                            </div>
                            <p className="text-xs text-white/25 max-w-[220px]">
                              Ask about squad matchups, historical trends, or venue advantages.
                            </p>
                          </div>
                        ) : (
                          messages.map((msg, i) => (
                            <ChatBubble key={i} text={msg.text} isAi={msg.isAi} />
                          ))
                        )}
                        {aiLoading && (
                          <div className="flex gap-3 items-center">
                            <div className="w-7 h-7 rounded-xl bg-ipl-gold/10 border border-ipl-gold/20
                                            flex items-center justify-center text-[10px] font-black text-ipl-gold">AI</div>
                            <div className="flex gap-1.5 px-4 py-3 bg-white/[0.04] border border-white/[0.07] rounded-2xl">
                              {[0,1,2].map(i => (
                                <motion.div key={i} className="w-1.5 h-1.5 rounded-full bg-white/40"
                                  animate={{ y: [0,-5,0] }}
                                  transition={{ duration: 0.7, repeat: Infinity, delay: i * 0.15 }} />
                              ))}
                            </div>
                          </div>
                        )}
                      </div>

                      {/* Input */}
                      <div className="flex gap-2">
                        <input
                          value={question}
                          onChange={e => setQuestion(e.target.value)}
                          onKeyDown={e => e.key === "Enter" && askAi()}
                          placeholder="Why does MI have an edge?"
                          className="flex-1 bg-white/[0.04] border border-white/[0.09] text-white rounded-2xl
                                     px-5 py-3.5 text-sm focus:outline-none focus:border-amber-500/40 transition-colors
                                     placeholder:text-white/20"
                        />
                        <button onClick={askAi} disabled={aiLoading || !question.trim()}
                          className="btn-primary px-5 py-3.5 flex items-center justify-center gap-2 min-w-[56px]">
                          {aiLoading
                            ? <RefreshCw size={16} className="animate-spin" />
                            : <Send size={16} />}
                        </button>
                      </div>
                    </div>
                  </div>
                </>
              ) : (
                <div className="card py-24 text-center">
                  <p className="text-white/20 italic">No prediction available.</p>
                </div>
              )}
            </motion.div>
          )}

          {/* ━━━ STATS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
          {tab === "stats" && (
            <motion.div key="stats" {...fadeUp} className="space-y-8">
              {standings.length > 0 ? (
                <>
                  {/* Standings table */}
                  <div className="card overflow-hidden">
                    <div className="px-8 py-6 border-b border-white/[0.06] flex items-center justify-between">
                      <h4 className="text-[11px] font-black uppercase tracking-[0.22em] text-white/35">
                        Season Standings
                      </h4>
                      <span className="badge badge-gold">{selectedYear}</span>
                    </div>
                    <div className="overflow-x-auto">
                      <table className="w-full text-left">
                        <thead>
                          <tr className="border-b border-white/[0.06] text-[10px] text-white/25 uppercase tracking-[0.18em] font-black">
                            <th className="px-7 py-5 w-12">#</th>
                            <th className="px-4 py-5">Franchise</th>
                            <th className="px-4 py-5 text-center">M</th>
                            <th className="px-4 py-5 text-center">W</th>
                            <th className="px-4 py-5 text-center">L</th>
                            <th className="px-4 py-5 text-center">Pts</th>
                            <th className="px-7 py-5 text-right">NRR</th>
                            <th className="px-4 py-5 text-center">Status</th>
                          </tr>
                        </thead>
                        <tbody className="text-sm divide-y divide-white/[0.04]">
                          {standings.map((t, idx) => {
                            const meta = teamMeta(t.team);
                            const isChamp = Boolean(t.won_title);
                            const isQual  = Boolean(t.qualified_playoffs);
                            return (
                              <tr key={t.team}
                                className={`group transition-colors hover:bg-white/[0.025]
                                  ${isChamp ? "bg-amber-500/[0.04]" : ""}`}>
                                <td className="px-7 py-4">
                                  <span className="text-base font-black text-white/15 stat-num">{idx + 1}</span>
                                </td>
                                <td className="px-4 py-4">
                                  <div className="flex items-center gap-3">
                                    <TeamAvatar team={t.team} size="sm" />
                                    <span className="font-bold text-white">{t.team}</span>
                                    {isChamp && <Trophy size={13} className="text-ipl-gold animate-float ml-1" />}
                                  </div>
                                </td>
                                <td className="px-4 py-4 text-center text-white/45 stat-num">{t.matches}</td>
                                <td className="px-4 py-4 text-center font-bold text-emerald-400 stat-num">{t.wins}</td>
                                <td className="px-4 py-4 text-center text-rose-400 stat-num">{t.losses}</td>
                                <td className="px-4 py-4 text-center font-black text-white stat-num">{t.points}</td>
                                <td className="px-7 py-4 text-right font-mono text-sm stat-num">
                                  <span className={t.nrr >= 0 ? "text-emerald-400" : "text-rose-400"}>
                                    {t.nrr > 0 ? "+" : ""}{Number(t.nrr).toFixed(3)}
                                  </span>
                                </td>
                                <td className="px-4 py-4 text-center">
                                  {isChamp
                                    ? <span className="badge badge-gold py-0.5">🏆 Champion</span>
                                    : isQual
                                      ? <span className="badge badge-green py-0.5"><Check size={8} />Playoffs</span>
                                      : <span className="text-[10px] text-white/20 font-bold uppercase tracking-wider">—</span>}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* Charts */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Points bar chart */}
                    <div className="card p-8">
                      <h4 className="text-[11px] font-black uppercase tracking-[0.22em] text-white/30 mb-7">
                        Points Distribution
                      </h4>
                      <div className="h-[260px]">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={standings} barSize={28}>
                            <CartesianGrid vertical={false} stroke="rgba(255,255,255,0.04)" />
                            <XAxis dataKey="team" hide />
                            <YAxis hide />
                            <Tooltip content={<ChartTooltip />} cursor={{ fill: "rgba(255,255,255,0.03)" }} />
                            <Bar dataKey="points" radius={[8, 8, 0, 0]}>
                              {standings.map((entry, i) => (
                                <Cell key={i}
                                  fill={teamMeta(entry.team).color}
                                  style={{ filter: `drop-shadow(0 0 6px ${teamMeta(entry.team).color}66)` }} />
                              ))}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                      {/* Legend */}
                      <div className="mt-4 flex flex-wrap gap-x-4 gap-y-2">
                        {standings.slice(0,6).map(t => (
                          <div key={t.team} className="flex items-center gap-1.5">
                            <div className="w-2 h-2 rounded-full" style={{ background: teamMeta(t.team).color }} />
                            <span className="text-[10px] text-white/35">{teamMeta(t.team).abbr}</span>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Win/Loss radar */}
                    <div className="card p-8">
                      <h4 className="text-[11px] font-black uppercase tracking-[0.22em] text-white/30 mb-7">
                        Win Rate (Top 6)
                      </h4>
                      <div className="h-[260px]">
                        <ResponsiveContainer width="100%" height="100%">
                          <RadarChart data={standings.slice(0,6).map(t => ({
                            team: teamMeta(t.team).abbr,
                            winRate: t.matches > 0 ? Math.round((t.wins / t.matches) * 100) : 0,
                          }))}>
                            <PolarGrid stroke="rgba(255,255,255,0.06)" />
                            <PolarAngleAxis dataKey="team"
                              tick={{ fill: "rgba(255,255,255,0.35)", fontSize: 11, fontWeight: 700 }} />
                            <Radar dataKey="winRate" stroke="#F5A623" fill="#F5A623" fillOpacity={0.12}
                              strokeWidth={2} dot={{ fill: "#F5A623", r: 3 }} />
                          </RadarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </div>
                </>
              ) : (
                <div className="card py-24 text-center">
                  <Shield size={40} className="text-white/10 mx-auto mb-4" />
                  <p className="text-white/25 italic">No standings data for {selectedYear}.</p>
                </div>
              )}
            </motion.div>
          )}

          {/* ━━━ HISTORY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
          {tab === "history" && (
            <motion.div key="history" {...fadeUp} className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">

                {/* Championship Hall  */}
                <div className="lg:col-span-3 card p-8">
                  <h4 className="text-[11px] font-black uppercase tracking-[0.22em] text-white/30 mb-7
                                 flex items-center gap-2">
                    <Trophy size={12} /> Championship Hall
                  </h4>
                  <div className="space-y-1 max-h-[480px] overflow-y-auto pr-2">
                    {seasons.map((s, idx) => {
                      const isSelected = s.year === selectedYear;
                      return (
                        <motion.div
                          key={s.year}
                          initial={{ opacity: 0, x: -16 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: idx * 0.04 }}
                          onClick={() => { setSelectedYear(s.year); setTab("predict"); }}
                          className={`flex items-center justify-between p-4 rounded-2xl cursor-pointer transition-all
                            ${isSelected
                              ? "bg-ipl-gold/10 border border-ipl-gold/20"
                              : "hover:bg-white/[0.04] border border-transparent"}`}
                        >
                          <div className="flex items-center gap-4">
                            <span className="text-2xl font-black text-white/10 stat-num w-12">{s.year}</span>
                            <div className="flex items-center gap-3">
                              {s.winner && <TeamAvatar team={s.winner} size="sm" />}
                              <div className="flex flex-col">
                                <span className={`font-bold ${isSelected ? "text-white" : "text-white/80"}`}>
                                  {s.winner || "TBD"}
                                </span>
                                {s.momentum && <MomentumPill sequence={s.momentum as any} />}
                              </div>
                            </div>
                          </div>
                          {isSelected
                            ? <span className="badge badge-gold">Selected</span>
                            : <ChevronDown size={14} className="text-white/15 -rotate-90" />}
                        </motion.div>
                      );
                    })}
                  </div>
                </div>

                {/* Title leaderboard */}
                <div className="lg:col-span-2 card p-8">
                  <h4 className="text-[11px] font-black uppercase tracking-[0.22em] text-white/30 mb-7
                                 flex items-center gap-2">
                    <Award size={12} /> Title Leaderboard
                  </h4>
                  <div className="space-y-5">
                    {titles.sort((a,b) => b.titles - a.titles).map((t, idx) => {
                      const maxTitles = Math.max(...titles.map(x => x.titles), 1);
                      const meta = teamMeta(t.team);
                      return (
                        <div key={t.team} className="flex items-center gap-4 group">
                          <span className="text-xl font-black text-white/10 w-6 stat-num">{idx + 1}</span>
                          <TeamAvatar team={t.team} size="sm" />
                          <div className="flex-1">
                            <div className="flex justify-between items-center mb-2">
                              <span className="text-xs font-bold text-white/70 group-hover:text-white transition-colors">
                                {t.team}
                              </span>
                              <span className="text-ipl-gold font-black stat-num text-sm">{t.titles}</span>
                            </div>
                            <div className="h-1.5 bg-white/[0.05] rounded-full overflow-hidden">
                              <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${(t.titles / maxTitles) * 100}%` }}
                                transition={{ duration: 1, delay: idx * 0.1, ease: "easeOut" }}
                                className="h-full rounded-full"
                                style={{ background: `linear-gradient(90deg, ${meta.color}, ${meta.accent})` }}
                              />
                            </div>
                            {t.years?.length > 0 && (
                              <p className="text-[10px] text-white/20 mt-1 font-mono">
                                {t.years.join(" · ")}
                              </p>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            </motion.div>
          )}

        </AnimatePresence>
      </main>

      <footer className="py-16 text-center border-t border-white/[0.04]">
        <div className="flex items-center justify-center gap-5 mb-5 text-white/15">
          <History size={16} /> <Shield size={16} /> <Target size={16} />
        </div>
        <p className="text-[10px] uppercase font-bold tracking-[0.4em] text-white/15">
          Gemini · Qdrant · Transformer · SQLite — © 2026 IPL Prophet
        </p>
      </footer>
    </div>
  );
}
