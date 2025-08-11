import * as React from "react";
import { ResponsiveContainer, LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, Legend } from "recharts";
import { COLORS } from "@/lib/colors";

type Point = { t: string; spread: number | null };

export const SpreadChart = React.memo(function SpreadChart({ data, height = 340 }: { data: Point[]; height?: number }) {
  return (
    <div style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="t" tickFormatter={(t) => String(t).slice(5, 16)} />
          <YAxis />
          <Tooltip labelFormatter={(t) => new Date(String(t)).toLocaleString()} />
          <Legend />
          <Line type="monotone" dataKey="spread" name="Spread" dot={false} stroke={COLORS.lines.spread} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
});