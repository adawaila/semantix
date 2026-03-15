"use client";

import { useEffect, useRef, useState } from "react";
import { fetchCollections, CollectionStats } from "@/lib/api";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export default function IngestPage() {
  const [collections, setCollections] = useState<CollectionStats[]>([]);
  const [selectedCol, setSelectedCol] = useState("");
  const [jobId, setJobId] = useState<string | null>(null);
  const [progress, setProgress] = useState<{ done: number; total: number; status: string } | null>(null);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const [fileName, setFileName] = useState("");
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    fetchCollections().then(cols => {
      setCollections(cols);
      if (cols.length > 0) setSelectedCol(cols[0].name);
    });
  }, []);

  const startWs = (jid: string) => {
    const wsUrl = BASE.replace(/^http/, "ws") + `/jobs/${jid}/stream`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;
    ws.onmessage = evt => {
      const data = JSON.parse(evt.data);
      setProgress({ done: data.done, total: data.total, status: data.status });
      if (data.status === "done" || data.status === "error") ws.close();
    };
    ws.onerror = () => setError("WebSocket error — check server connection");
  };

  const handleFile = async (file: File) => {
    if (!selectedCol) { setError("Select a collection first"); return; }
    setError("");
    setProgress(null);
    setFileName(file.name);

    let docs: unknown[];
    try {
      const text = await file.text();
      const parsed = JSON.parse(text);
      docs = Array.isArray(parsed) ? parsed : [parsed];
    } catch {
      setError("Invalid JSON file");
      return;
    }

    try {
      const res = await fetch(`${BASE}/collections/${selectedCol}/documents/bulk`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ documents: docs }),
      });
      if (!res.ok) {
        const body = await res.json();
        setError(body.detail ?? "Server error");
        return;
      }
      const data = await res.json();
      setJobId(data.job_id);
      setProgress({ done: 0, total: data.total, status: "queued" });
      startWs(data.job_id);
    } catch (e) {
      setError(String(e));
    }
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  const pct = progress && progress.total > 0 ? Math.round((progress.done / progress.total) * 100) : 0;
  const statusColor: Record<string, string> = {
    queued: "#f59e0b", running: "#6366f1", done: "#22c55e", error: "#ef4444",
  };

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-1">Ingest Documents</h1>
        <p style={{ color: "var(--text-muted)" }}>Upload a JSON file to bulk-ingest into a collection</p>
      </div>

      {/* Collection selector */}
      <div className="rounded-xl p-6 mb-6" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
        <label className="block text-sm mb-2" style={{ color: "var(--text-muted)" }}>Target Collection</label>
        <select value={selectedCol} onChange={e => setSelectedCol(e.target.value)}
          className="w-full max-w-xs rounded-lg px-3 py-2 text-sm"
          style={{ background: "var(--background)", border: "1px solid var(--border)", color: "var(--foreground)" }}>
          {collections.length === 0 && <option value="">No collections — create one first</option>}
          {collections.map(c => <option key={c.name} value={c.name}>{c.name} ({c.doc_count} docs)</option>)}
        </select>
      </div>

      {/* Drop zone */}
      <div
        onDragOver={e => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className="rounded-xl border-2 border-dashed p-16 text-center transition-colors cursor-pointer"
        style={{
          borderColor: dragging ? "var(--accent)" : "var(--border)",
          background: dragging ? "rgba(99,102,241,0.05)" : "var(--surface)",
        }}
        onClick={() => {
          const inp = document.createElement("input");
          inp.type = "file";
          inp.accept = ".json";
          inp.onchange = (e) => {
            const f = (e.target as HTMLInputElement).files?.[0];
            if (f) handleFile(f);
          };
          inp.click();
        }}>
        <div className="text-4xl mb-3">⬆</div>
        <p className="text-lg font-semibold mb-1">
          {fileName ? fileName : "Drop JSON file here or click to browse"}
        </p>
        <p className="text-sm" style={{ color: "var(--text-muted)" }}>
          Accepts a JSON array of documents, each with an &quot;id&quot; field
        </p>
      </div>

      {error && <div className="mt-4 p-3 rounded-lg text-red-400" style={{ background: "#1a0a0a" }}>{error}</div>}

      {/* Progress */}
      {progress && (
        <div className="mt-6 rounded-xl p-6" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
          <div className="flex items-center justify-between mb-3">
            <div>
              <span className="font-semibold">Job: </span>
              <span className="font-mono text-sm">{jobId}</span>
            </div>
            <span className="px-2 py-1 rounded text-xs font-semibold"
              style={{ background: statusColor[progress.status] + "22", color: statusColor[progress.status] }}>
              {progress.status}
            </span>
          </div>
          <div className="w-full rounded-full h-3 mb-2" style={{ background: "var(--border)" }}>
            <div className="h-3 rounded-full transition-all"
              style={{ width: `${pct}%`, background: statusColor[progress.status] }} />
          </div>
          <div className="text-sm" style={{ color: "var(--text-muted)" }}>
            {progress.done} / {progress.total} documents indexed ({pct}%)
          </div>
        </div>
      )}

      {/* Format help */}
      <div className="mt-6 rounded-xl p-4" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
        <p className="text-sm font-semibold mb-2">Expected JSON format:</p>
        <pre className="text-xs rounded p-3" style={{ background: "var(--background)", color: "#a5b4fc" }}>
{`[
  {"id": "1", "description": "wireless headphones", "price": 49.99},
  {"id": "2", "description": "bluetooth speaker", "price": 29.99}
]`}
        </pre>
      </div>
    </div>
  );
}
