export type Meta = {
  n_rows: { dataset1: number; dataset2: number };
  time_span: { dataset1: [string, string] | null; dataset2: [string, string] | null };
  outputs: Record<string, any>;
};

export type MetricRow = {
  contract: string;
  n: number;
  mae: number;
  rmse: number;
  mape: number;
  median_spread: number;
  corr: number | null;
  corr_log: number | null;
};

export type UptimeRow = {
  contract: string;
  d1_present_close_rate: number;
  d2_present_close_rate: number;
  d1_trade_active_rate: number;
  d2_trade_active_rate: number;
  both_trade_active_rate: number;
  expected_bars: number;
};

export type PanelRow = {
  timestamp: string; // ISO
  close_d1_scaled?: number | null;
  close_d2?: number | null;
  spread?: number | null;
  trade_active_d1?: boolean;
  trade_active_d2?: boolean;
};

export type RRRow = { contract: string; timestamp: string; rolling_ratio: number };