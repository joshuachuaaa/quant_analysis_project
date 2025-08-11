import * as React from "react";
import { ResponsiveContainer, LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, Legend } from "recharts";
import { COLORS } from "@/lib/colors";

type Point = { t: string; d1: number | null; d2: number | null };

export const CloseChart = React.memo(function CloseChart({ data, height = 340 }: { data: Point[]; height?: number }) {
  return (
    <div style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="t" tickFormatter={(t) => String(t).slice(5, 16)} />
          <YAxis />
          <Tooltip labelFormatter={(t) => new Date(String(t)).toLocaleString()} />
          <Legend />
          <Line type="monotone" dataKey="d1" name="D1 Scaled Close" dot={false} stroke={COLORS.lines.d1} isAnimationActive={false} />
          <Line type="monotone" dataKey="d2" name="D2 Close"        dot={false} stroke={COLORS.lines.d2} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
});