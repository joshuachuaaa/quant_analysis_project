import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectTrigger, SelectContent, SelectItem, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Download } from "lucide-react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, BarChart, Bar, Legend,
} from "recharts";

/* =========================
   Types
========================= */
type Meta = {
  n_rows: { dataset1: number; dataset2: number };
  time_span: { dataset1: [string, string] | null; dataset2: [string, string] | null };
  outputs: Record<string, any>;
};

type MetricRow = {
  contract: string;
  n: number;
  mae: number;
  rmse: number;
  mape: number;
  median_spread: number;
  corr: number | null;
  corr_log: number | null;
};

type UptimeRow = {
  contract: string;
  d1_present_close_rate: number;
  d2_present_close_rate: number;
  d1_trade_active_rate: number;
  d2_trade_active_rate: number;
  both_trade_active_rate: number;
  expected_bars: number;
};

type PanelRow = {
  timestamp: string; // ISO
  close_d1_scaled?: number | null;
  close_d2?: number | null;
  spread?: number | null;
  trade_active_d1?: boolean;
  trade_active_d2?: boolean;
};

type RRRow = { contract: string; timestamp: string; rolling_ratio: number };

/* =========================
   Config
========================= */
const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

const COLORS = {
  lines: {
    d1: "#1f77b4",     // blue
    d2: "#ff7f0e",     // orange
    spread: "#2ca02c", // green
    rr: "#17becf",     // cyan
  },
  bars: {
    d1Present: "#1f77b4",
    d2Present: "#ff7f0e",
    d1Active: "#2ca02c",
    d2Active: "#d62728",
    bothActive: "#9467bd",
  },
} as const;

/* =========================
   Utilities / UI bits
========================= */
function num(n?: number | null, digits = 3) {
  if (n === undefined || n === null || isNaN(n as number)) return "—";
  return Number(n).toFixed(digits);
}

const SectionTitle: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="text-xl font-semibold tracking-tight mb-2">{children}</div>
);

const CardStat: React.FC<{ title: string; value: string }> = ({ title, value }) => (
  <Card className="shadow-sm">
    <CardHeader className="pb-2">
      <CardTitle className="text-sm text-muted-foreground">{title}</CardTitle>
    </CardHeader>
    <CardContent className="text-2xl font-semibold">{value}</CardContent>
  </Card>
);

/** Safe fetch hook: bails on empty URL and aborts on unmount */
const useFetch = <T,>(url: string, deps: any[] = []) => {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    if (!url) {
      setLoading(false);
      setErr(null);
      setData(null);
      return;
    }
    const ctrl = new AbortController();
    setLoading(true);
    fetch(url, { signal: ctrl.signal })
      .then(r => {
        if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
        return r.json();
      })
      .then(j => { if (live) { setData(j); setErr(null); } })
      .catch(e => { if (live && (e as any).name !== "AbortError") setErr(String(e)); })
      .finally(() => { if (live) setLoading(false); });
    return () => { live = false; ctrl.abort(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return { data, loading, err } as const;
};

/* =========================
   Dashboard
========================= */
export default function Dashboard() {
  const { data: meta } = useFetch<Meta>(`${API_BASE}/api/meta`, []);
  const { data: contractsData } = useFetch<{ contracts: string[] }>(`${API_BASE}/api/contracts`, []);
  const { data: metrics } = useFetch<MetricRow[]>(`${API_BASE}/api/metrics`, []);
  const { data: uptimePerContract } = useFetch<UptimeRow[]>(`${API_BASE}/api/uptime/contracts`, []);
  const { data: rr } = useFetch<RRRow[]>(`${API_BASE}/api/metrics/rolling_ratio`, []);

  const [selected, setSelected] = useState<string | null>(null);
  const contracts = contractsData?.contracts || [];

  useEffect(() => {
    if (!selected && contracts.length > 0) setSelected(contracts[0]);
  }, [contracts, selected]);

  const selectedMetrics = useMemo(() => {
    if (!metrics || !selected) return null;
    return metrics.find(m => m.contract === selected) || null;
  }, [metrics, selected]);

  const { data: panel } = useFetch<PanelRow[]>(
    selected ? `${API_BASE}/api/panel?contract=${encodeURIComponent(selected)}` : "",
    [selected]
  );

  const closesSeries = useMemo(() => {
    if (!panel) return [] as { t: string; d1: number | null; d2: number | null; spread: number | null }[];
    return panel
      .filter(row => row.timestamp)
      .map(row => ({
        t: new Date(row.timestamp).toISOString(),
        d1: typeof row.close_d1_scaled === "number" ? row.close_d1_scaled : null,
        d2: typeof row.close_d2 === "number" ? row.close_d2 : null,
        spread:
          typeof row.spread === "number"
            ? row.spread
            : (typeof row.close_d1_scaled === "number" && typeof row.close_d2 === "number"
                ? row.close_d1_scaled - row.close_d2
                : null),
      }));
  }, [panel]);

  const rrSeries = useMemo(() => {
    if (!rr || !selected) return [] as { t: string; v: number }[];
    return rr
      .filter(r => r.contract === selected)
      .map(r => ({ t: new Date(r.timestamp).toISOString(), v: r.rolling_ratio }));
  }, [rr, selected]);

  const uptimeData = useMemo(
    () => (uptimePerContract || []).filter(u => u.contract === selected),
    [uptimePerContract, selected]
  );

  // heatmap image paths (hide gracefully if missing)
  const perContractHm1 = selected
    ? `${API_BASE}/artifacts/heatmaps/per_contract/heatmap_dataset1_futures_${selected}.png`
    : "";
  const perContractHm2 = selected
    ? `${API_BASE}/artifacts/heatmaps/per_contract/heatmap_dataset2_futures_${selected}.png`
    : "";
  const globalHm1 = `${API_BASE}/artifacts/heatmaps/global/missing_heatmap_dataset1_futures.png`;
  const globalHm2 = `${API_BASE}/artifacts/heatmaps/global/missing_heatmap_dataset2_futures.png`;

  return (
    <div className="min-h-screen bg-white text-neutral-900">
        <div className="min-h-screen bg-white">
        <div className="mx-auto max-w-6xl p-6 space-y-6">
            <div className="flex items-center justify-between">
            <h1 className="text-2xl font-bold tracking-tight">Candle Data Reconciliation</h1>
            <div className="flex items-center gap-2">
                <Button asChild variant="outline">
                <a href={`${API_BASE}/api/report.xlsx`}>
                    <span className="inline-flex items-center gap-2">
                    <Download className="w-4 h-4" />
                    Download workbook
                    </span>
                </a>
                </Button>

                {/* Fixed-width, placeholder-aware Select (works with stub or real shadcn) */}
                <Select className="w-56" value={selected ?? ""} onValueChange={setSelected}>
                <SelectTrigger>
                    <SelectValue placeholder="Choose contract" />
                </SelectTrigger>
                <SelectContent>
                    {contracts.map(c => (
                    <SelectItem key={c} value={c}>{c}</SelectItem>
                    ))}
                </SelectContent>
                </Select>
            </div>
            </div>

            {/* Summary */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <CardStat title="Rows (Dataset 1)" value={meta ? String(meta.n_rows.dataset1) : "—"} />
            <CardStat title="Rows (Dataset 2)" value={meta ? String(meta.n_rows.dataset2) : "—"} />
            <CardStat
                title="Span (D1)"
                value={meta?.time_span.dataset1 ? `${meta.time_span.dataset1[0].slice(0,10)} → ${meta.time_span.dataset1[1].slice(0,10)}` : "—"}
            />
            <CardStat
                title="Span (D2)"
                value={meta?.time_span.dataset2 ? `${meta.time_span.dataset2[0].slice(0,10)} → ${meta.time_span.dataset2[1].slice(0,10)}` : "—"}
            />
            </div>

            {/* Metrics table */}
            <Card className="shadow-sm">
            <CardHeader>
                <CardTitle>Per-Contract Metrics</CardTitle>
            </CardHeader>
            <CardContent>
                <Table>
                <TableHeader>
                    <TableRow>
                    <TableHead>Contract</TableHead>
                    <TableHead className="text-right">n</TableHead>
                    <TableHead className="text-right">MAE</TableHead>
                    <TableHead className="text-right">RMSE</TableHead>
                    <TableHead className="text-right">MAPE (%)</TableHead>
                    <TableHead className="text-right">Median Spread</TableHead>
                    <TableHead className="text-right">Corr</TableHead>
                    <TableHead className="text-right">Corr (log)</TableHead>
                    </TableRow>
                </TableHeader>
                <TableBody>
                    {(metrics || []).map((m) => (
                    <TableRow key={m.contract} className={m.contract === selected ? "bg-muted/30" : ""}>
                        <TableCell className="font-medium">{m.contract}</TableCell>
                        <TableCell className="text-right">{m.n}</TableCell>
                        <TableCell className="text-right">{num(m.mae, 3)}</TableCell>
                        <TableCell className="text-right">{num(m.rmse, 3)}</TableCell>
                        <TableCell className="text-right">{num(m.mape, 2)}</TableCell>
                        <TableCell className="text-right">{num(m.median_spread, 4)}</TableCell>
                        <TableCell className="text-right">{num(m.corr, 3)}</TableCell>
                        <TableCell className="text-right">{num(m.corr_log, 3)}</TableCell>
                    </TableRow>
                    ))}
                </TableBody>
                </Table>
            </CardContent>
            </Card>

            {/* Time series: closes & spread */}
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
            <Card className="shadow-sm">
                <CardHeader><CardTitle>{selected || "—"} — Close (D1 scaled vs D2)</CardTitle></CardHeader>
                <CardContent className="h-[340px]">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={closesSeries.map(({ t, d1, d2 }) => ({ t, d1, d2 }))}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="t" tickFormatter={(t) => String(t).slice(5,16)} />
                    <YAxis />
                    <Tooltip labelFormatter={(t) => new Date(String(t)).toLocaleString()} />
                    <Legend />
                    <Line type="monotone" dataKey="d1" name="D1 Scaled Close" dot={false} stroke={COLORS.lines.d1} isAnimationActive={false} />
                    <Line type="monotone" dataKey="d2" name="D2 Close"        dot={false} stroke={COLORS.lines.d2} isAnimationActive={false} />
                    </LineChart>
                </ResponsiveContainer>
                </CardContent>
            </Card>

            <Card className="shadow-sm">
                <CardHeader><CardTitle>{selected || "—"} — Spread (D1scaled − D2)</CardTitle></CardHeader>
                <CardContent className="h-[340px]">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={closesSeries.map(({ t, spread }) => ({ t, spread }))}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="t" tickFormatter={(t) => String(t).slice(5,16)} />
                    <YAxis />
                    <Tooltip labelFormatter={(t) => new Date(String(t)).toLocaleString()} />
                    <Legend />
                    <Line type="monotone" dataKey="spread" name="Spread" dot={false} stroke={COLORS.lines.spread} isAnimationActive={false} />
                    </LineChart>
                </ResponsiveContainer>
                </CardContent>
            </Card>
            </div>

            {/* Uptime (per contract) */}
            <Card className="shadow-sm">
            <CardHeader><CardTitle>Uptime — Present Close & Trade Active</CardTitle></CardHeader>
            <CardContent className="h-[360px]">
                <ResponsiveContainer width="100%" height="100%">
                <BarChart data={uptimeData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="contract" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="d1_present_close_rate"  name="D1 Present Close"  fill={COLORS.bars.d1Present} />
                    <Bar dataKey="d2_present_close_rate"  name="D2 Present Close"  fill={COLORS.bars.d2Present} />
                    <Bar dataKey="d1_trade_active_rate"   name="D1 Trade Active"   fill={COLORS.bars.d1Active} />
                    <Bar dataKey="d2_trade_active_rate"   name="D2 Trade Active"   fill={COLORS.bars.d2Active} />
                    <Bar dataKey="both_trade_active_rate" name="Both Trade Active" fill={COLORS.bars.bothActive} />
                </BarChart>
                </ResponsiveContainer>
            </CardContent>
            </Card>

            {/* Rolling ratio */}
            <Card className="shadow-sm">
            <CardHeader><CardTitle>{selected || "—"} — Rolling median (D1_scaled / D2)</CardTitle></CardHeader>
            <CardContent className="h-[320px]">
                <ResponsiveContainer width="100%" height="100%">
                <LineChart data={rrSeries}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="t" tickFormatter={(t) => String(t).slice(5,16)} />
                    <YAxis />
                    <Tooltip labelFormatter={(t) => new Date(String(t)).toLocaleString()} />
                    <Legend />
                    <Line type="monotone" dataKey="v" name="Rolling ratio" dot={false} stroke={COLORS.lines.rr} isAnimationActive={false} />
                </LineChart>
                </ResponsiveContainer>
            </CardContent>
            </Card>

            {/* Heatmaps */}
            <div className="space-y-3">
            <SectionTitle>Heatmaps</SectionTitle>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card className="shadow-sm">
                <CardHeader><CardTitle>Per-Contract: Dataset 1</CardTitle></CardHeader>
                <CardContent>
                    {selected ? (
                    <img
                        src={perContractHm1}
                        onError={(e) => ((e.target as HTMLImageElement).style.display = "none")}
                        className="w-full rounded-md"
                    />
                    ) : <div className="text-sm text-muted-foreground">Select a contract</div>}
                </CardContent>
                </Card>

                <Card className="shadow-sm">
                <CardHeader><CardTitle>Per-Contract: Dataset 2</CardTitle></CardHeader>
                <CardContent>
                    {selected ? (
                    <img
                        src={perContractHm2}
                        onError={(e) => ((e.target as HTMLImageElement).style.display = "none")}
                        className="w-full rounded-md"
                    />
                    ) : <div className="text-sm text-muted-foreground">Select a contract</div>}
                </CardContent>
                </Card>

                <Card className="shadow-sm">
                <CardHeader><CardTitle>Global: Dataset 1</CardTitle></CardHeader>
                <CardContent>
                    <img
                    src={globalHm1}
                    onError={(e) => ((e.target as HTMLImageElement).style.display = "none")}
                    className="w-full rounded-md"
                    />
                </CardContent>
                </Card>

                <Card className="shadow-sm">
                <CardHeader><CardTitle>Global: Dataset 2</CardTitle></CardHeader>
                <CardContent>
                    <img
                    src={globalHm2}
                    onError={(e) => ((e.target as HTMLImageElement).style.display = "none")}
                    className="w-full rounded-md"
                    />
                </CardContent>
                </Card>
            </div>
            </div>
        </div>
        </div>
    </div>
  );
}
