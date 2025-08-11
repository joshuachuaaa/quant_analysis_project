// src/components/ui/select.tsx
import * as React from "react";

// --- Public API (shadcn-compatible surface) ---
export function Select({
  value,
  onValueChange,
  children,
  className,
}: {
  value?: string;
  onValueChange?: (v: string) => void;
  children?: React.ReactNode;
  className?: string;
}) {
  const items = React.useMemo(() => collectItems(children), [children]);
  const placeholder = React.useMemo(() => findPlaceholder(children), [children]);

  return (
    <select
      value={value ?? ""}
      onChange={(e) => onValueChange?.(e.target.value)}
      className={"rounded-md border px-3 py-2 text-sm " + (className ?? "")}
    >
      {/* show a placeholder row when no value is selected */}
      {(!value || value === "") && placeholder ? (
        <option value="" disabled>
          {placeholder}
        </option>
      ) : null}

      {items.map((it) => (
        <option key={it.value} value={it.value}>
          {it.label}
        </option>
      ))}
    </select>
  );
}

// Presentational shims so existing JSX compiles.
export function SelectTrigger(p: React.HTMLAttributes<HTMLDivElement>) { return <>{p.children}</>; }
export function SelectContent(p: React.HTMLAttributes<HTMLDivElement>) { return <>{p.children}</>; }
export function SelectItem({ value, children }: { value: string; children: React.ReactNode }) {
  // We don't render anything here — the parent <Select> consumes these.
  return <option value={value}>{children}</option>;
}
SelectItem.displayName = "SelectItem";

export function SelectValue({ placeholder }: { placeholder?: string }) {
  // Not rendered directly; the parent <Select> reads this prop.
  return <span data-placeholder={placeholder} />;
}
SelectValue.displayName = "SelectValue";

// --- Helpers: walk children to collect items and placeholder ---
function collectItems(node: React.ReactNode): { value: string; label: React.ReactNode }[] {
  const out: { value: string; label: React.ReactNode }[] = [];
  walk(node, (el) => {
    if (React.isValidElement(el) && (el.type as any)?.displayName === "SelectItem") {
      out.push({ value: String((el.props as any).value), label: (el.props as any).children });
    }
  });
  return out;
}

function findPlaceholder(node: React.ReactNode): string | undefined {
  let ph: string | undefined;
  walk(node, (el) => {
    if (React.isValidElement(el) && (el.type as any)?.displayName === "SelectValue") {
      const p = (el.props as any)?.placeholder;
      if (typeof p === "string") ph = p;
    }
  });
  return ph;
}

function walk(node: React.ReactNode, fn: (el: React.ReactElement) => void) {
  React.Children.forEach(node, (child) => {
    if (!child) return;
    if (typeof child === "string" || typeof child === "number") return;
    if (React.isValidElement(child)) {
      fn(child);
      if ((child.props as any)?.children) {
        walk((child.props as any).children, fn);
      }
    }
  });
}
