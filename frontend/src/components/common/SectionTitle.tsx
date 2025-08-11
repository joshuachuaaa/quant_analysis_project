import * as React from "react";

export const SectionTitle: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="text-xl font-semibold tracking-tight mb-2">{children}</div>
);