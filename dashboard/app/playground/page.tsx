"use client";

import { useEffect, useState } from "react";
import { fetchCollections, search, CollectionStats, SearchResultItem } from "@/lib/api";

export default function PlaygroundPage() {
  const [collections, setCollections] = useState<CollectionStats[]>([]);
  const [selectedCol, setSelectedCol] = useState("");
  const [query, setQuery] = useState("");
  const [alpha, setAlpha] = useState(0.5);
  const [topK, setTopK] = useState(10);
  const [results, setResults] = useState<SearchResultItem[]>([]);
  const [latency, setLatency] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchCollections().then(cols => {
      setCollections(cols);
      if (cols.length > 0) setSelectedCol(cols[0].name);
    });
  }, []);

  const handleSearch = async () => {
    if (!selectedCol || !query.trim()) return;
    setLoading(true);
    setError("");
    try {
      const resp = await search(selectedCol, query, topK, alpha);
      setResults(resp.results);
      setLatency(resp.latency_ms);
    } catch (e) {
      setError(String(e));
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") handleSearch();
  };

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-1">Search Playground</h1>
        <p style={{ color: "var(--text-muted)" }}>Test hybrid search interactively</p>
      </div>

      {/* Controls */}
      <div className="rounded-xl p-6 mb-6" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
        <div className="grid grid-cols-3 gap-4 mb-4">
          {/* Collection selector */}
          <div>
            <label className="block text-sm mb-1" style={{ color: "var(--text-muted)" }}>Collection</label>
            <select value={selectedCol} onChange={e => setSelectedCol(e.target.value)}
              className="w-full rounded-lg px-3 py-2 text-sm"
              style={{ background: "var(--background)", border: "1px solid var(--border)", color: "var(--foreground)" }}>
              {collections.length === 0 && <option value="">No collections</option>}
              {collections.map(c => <option key={c.name} value={c.name}>{c.name}</option>)}
            </select>
          </div>

          {/* Top K */}
          <div>
            <label className="block text-sm mb-1" style={{ color: "var(--text-muted)" }}>Top K</label>
            <input type="number" min={1} max={100} value={topK}
              onChange={e => setTopK(Number(e.target.value))}
              className="w-full rounded-lg px-3 py-2 text-sm"
              style={{ background: "var(--background)", border: "1px solid var(--border)", color: "var(--foreground)" }} />
          </div>

          {/* Alpha */}
          <div>
            <label className="block text-sm mb-1" style={{ color: "var(--text-muted)" }}>
              Alpha: {alpha.toFixed(2)} &nbsp;
              <span className="text-xs">(0=BM25 / 1=vector)</span>
            </label>
            <input type="range" min={0} max={1} step={0.05} value={alpha}
              onChange={e => setAlpha(Number(e.target.value))}
              className="w-full" style={{ accentColor: "var(--accent)" }} />
          </div>
        </div>

        {/* Query input */}
        <div className="flex gap-3">
          <input
            type="text"
            placeholder="Type your search query..."
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={handleKey}
            className="flex-1 rounded-lg px-4 py-2"
            style={{ background: "var(--background)", border: "1px solid var(--border)", color: "var(--foreground)" }}
          />
          <button onClick={handleSearch} disabled={loading || !selectedCol}
            className="px-6 py-2 rounded-lg font-semibold transition-opacity disabled:opacity-40"
            style={{ background: "var(--accent)", color: "#fff" }}>
            {loading ? "Searching..." : "Search"}
          </button>
        </div>
      </div>

      {/* Results */}
      {error && <div className="mb-4 p-3 rounded-lg text-red-400" style={{ background: "#1a0a0a" }}>{error}</div>}

      {latency !== null && (
        <div className="mb-4 text-sm" style={{ color: "var(--text-muted)" }}>
          {results.length} results in {latency.toFixed(2)}ms
        </div>
      )}

      <div className="flex flex-col gap-3">
        {results.map((r, i) => (
          <div key={r.id} className="rounded-xl p-4"
            style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-3">
                <span className="text-xs w-5 h-5 rounded-full flex items-center justify-center font-bold"
                  style={{ background: "var(--accent)", color: "#fff" }}>{i + 1}</span>
                <span className="font-mono font-semibold">{r.id}</span>
              </div>
              <div className="flex gap-4 text-xs" style={{ color: "var(--text-muted)" }}>
                <span>score: <strong>{r.score.toFixed(5)}</strong></span>
                {r.bm25_rank && <span>bm25: #{r.bm25_rank}</span>}
                {r.vector_rank && <span>vec: #{r.vector_rank}</span>}
              </div>
            </div>
            <pre className="text-xs overflow-x-auto rounded p-2"
              style={{ background: "var(--background)", color: "#a5b4fc" }}>
              {JSON.stringify(r.document, null, 2)}
            </pre>
          </div>
        ))}
      </div>

      {results.length === 0 && latency !== null && !loading && (
        <div className="text-center py-12" style={{ color: "var(--text-muted)" }}>
          No results found
        </div>
      )}
    </div>
  );
}
