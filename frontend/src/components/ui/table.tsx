import * as React from "react";
export function Table(p: React.HTMLAttributes<HTMLTableElement>) { return <table {...p} className={"w-full border-collapse " + (p.className ?? "")} />; }
export function TableHeader(p: React.HTMLAttributes<HTMLTableSectionElement>) { return <thead {...p} />; }
export function TableBody(p: React.HTMLAttributes<HTMLTableSectionElement>) { return <tbody {...p} />; }
export function TableRow(p: React.HTMLAttributes<HTMLTableRowElement>) { return <tr {...p} />; }
export function TableHead(p: React.ThHTMLAttributes<HTMLTableCellElement>) { return <th {...p} />; }
export function TableCell(p: React.TdHTMLAttributes<HTMLTableCellElement>) { return <td {...p} />; }