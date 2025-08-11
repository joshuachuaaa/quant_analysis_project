import * as React from "react";
import { ResponsiveContainer, BarChart, CartesianGrid, XAxis, YAxis, Tooltip, Legend, Bar } from "recharts";
import { COLORS } from "@/lib/colors";
import { UptimeRow } from "@/lib/types";

export const UptimeBarChart = React.memo(function UptimeBarChart({ data, height = 360 }: { data: UptimeRow[]; height?: number }) {
  return (
    <div style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data}>
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
    </div>
  );
});