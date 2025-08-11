import * as React from "react";
import { ResponsiveContainer, LineChart, CartesianGrid, XAxis, YAxis, Tooltip, Legend, Line } from "recharts";
import { COLORS } from "@/lib/colors";

type Point = { t: string; v: number };

export const RollingRatioChart = React.memo(function RollingRatioChart({ data, height = 320 }: { data: Point[]; height?: number }) {
  return (
    <div style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="t" tickFormatter={(t) => String(t).slice(5, 16)} />
          <YAxis />
          <Tooltip labelFormatter={(t) => new Date(String(t)).toLocaleString()} />
          <Legend />
          <Line type="monotone" dataKey="v" name="Rolling ratio" dot={false} stroke={COLORS.lines.rr} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
});