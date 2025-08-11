import { useEffect, useState } from "react";

export function useFetch<T>(url: string, deps: any[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    if (!url) { setData(null); setErr(null); setLoading(false); return; }

    const ctrl = new AbortController();
    setLoading(true);
    fetch(url, { signal: ctrl.signal })
      .then(r => { if (!r.ok) throw new Error(`${r.status} ${r.statusText}`); return r.json(); })
      .then(j => { if (live) { setData(j); setErr(null); } })
      .catch(e => { if (live && e.name !== 'AbortError') setErr(String(e)); })
      .finally(() => { if (live) setLoading(false); });

    return () => { live = false; ctrl.abort(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return { data, loading, err } as const;
}