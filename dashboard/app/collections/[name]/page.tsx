"use client";

import { useEffect, useState, useCallback } from "react";
import { useParams } from "next/navigation";
import { CollectionStats, fetchCollection, fetchDocuments, deleteDocument } from "@/lib/api";

const PAGE_SIZE = 20;

function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-xl p-4" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
      <div className="text-xs uppercase tracking-widest mb-1" style={{ color: "var(--text-muted)" }}>{label}</div>
      <div className="text-2xl font-bold">{value}</div>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex justify-between py-3" style={{ borderBottom: "1px solid var(--border)" }}>
      <span style={{ color: "var(--text-muted)" }}>{label}</span>
      <span className="font-mono font-semibold">{value}</span>
    </div>
  );
}

export default function CollectionDetailPage() {
  const { name } = useParams<{ name: string }>();
  const [col, setCol] = useState<CollectionStats | null>(null);
  const [loading, setLoading] = useState(true);

  // Document browser state
  const [docs, setDocs] = useState<Record<string, unknown>[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [docsLoading, setDocsLoading] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState("");
  const [expandedId, setExpandedId] = useState<string | null>(null);

  useEffect(() => {
    fetchCollection(name).then(data => { setCol(data); setLoading(false); });
    const interval = setInterval(() => fetchCollection(name).then(setCol), 10000);
    return () => clearInterval(interval);
  }, [name]);

  const loadDocs = useCallback(async (off: number) => {
    setDocsLoading(true);
    setDeleteError("");
    try {
      const res = await fetchDocuments(name, PAGE_SIZE, off);
      setDocs(res.documents);
      setTotal(res.total);
      setOffset(off);
    } finally {
      setDocsLoading(false);
    }
  }, [name]);

  useEffect(() => { loadDocs(0); }, [loadDocs]);

  const handleDelete = async (docId: string) => {
    setDeletingId(docId);
    setDeleteError("");
    try {
      await deleteDocument(name, docId);
      await loadDocs(offset);
      fetchCollection(name).then(setCol);
    } catch (e) {
      setDeleteError(String(e));
    } finally {
      setDeletingId(null);
    }
  };

  if (loading) return (
    <div className="text-center py-20" style={{ color: "var(--text-muted)" }}>Loading...</div>
  );
  if (!col) return (
    <div className="text-center py-20 text-red-400">Collection not found: {name}</div>
  );

  const totalPages = Math.ceil(total / PAGE_SIZE);
  const currentPage = Math.floor(offset / PAGE_SIZE) + 1;

  return (
    <div>
      {/* Header */}
      <div className="mb-8">
        <div className="text-sm mb-1" style={{ color: "var(--text-muted)" }}>Collections /</div>
        <h1 className="text-3xl font-bold">{col.name}</h1>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-3 gap-4 mb-8">
        <StatCard label="Documents" value={col.doc_count.toLocaleString()} />
        <StatCard label="Vectors" value={col.vector_count.toLocaleString()} />
        <StatCard label="Queries Run" value={col.query_count.toLocaleString()} />
      </div>

      {/* Configuration */}
      <div className="rounded-xl p-6 mb-8" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
        <h2 className="text-lg font-semibold mb-4">Configuration</h2>
        <Row label="Embedding field" value={col.embedding_field ?? "description"} />
        <Row label="Embedding dim" value={col.embedding_dim} />
        <Row label="Index type" value={col.index_type} />
        <Row label="Provider" value={col.provider} />
        <Row label="Vocab size" value={col.vocab_size.toLocaleString()} />
        <Row label="Created" value={new Date(col.created_at * 1000).toLocaleString()} />
      </div>

      {/* Document browser */}
      <div className="rounded-xl" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
        <div className="flex items-center justify-between px-6 py-4" style={{ borderBottom: "1px solid var(--border)" }}>
          <h2 className="text-lg font-semibold">
            Documents
            <span className="ml-2 text-sm font-normal" style={{ color: "var(--text-muted)" }}>
              {total.toLocaleString()} total
            </span>
          </h2>
          <button
            onClick={() => loadDocs(offset)}
            className="text-xs px-3 py-1 rounded-lg"
            style={{ background: "var(--background)", border: "1px solid var(--border)", color: "var(--text-muted)" }}
          >
            Refresh
          </button>
        </div>

        {deleteError && (
          <div className="mx-6 mt-4 p-3 rounded-lg text-red-400 text-sm" style={{ background: "#1a0a0a" }}>
            {deleteError}
          </div>
        )}

        {docsLoading ? (
          <div className="p-8 text-center" style={{ color: "var(--text-muted)" }}>Loading documents...</div>
        ) : docs.length === 0 ? (
          <div className="p-8 text-center" style={{ color: "var(--text-muted)" }}>No documents yet.</div>
        ) : (
          <div>
            {docs.map((doc) => {
              const docId = String(doc.id ?? "");
              const isExpanded = expandedId === docId;
              const isDeleting = deletingId === docId;

              // Show a preview of a few fields
              const previewFields = Object.entries(doc)
                .filter(([k]) => k !== "id")
                .slice(0, 3);

              return (
                <div
                  key={docId}
                  style={{ borderBottom: "1px solid var(--border)" }}
                >
                  <div
                    className="flex items-start justify-between px-6 py-4 cursor-pointer hover:bg-white/5 transition-colors"
                    onClick={() => setExpandedId(isExpanded ? null : docId)}
                  >
                    <div className="flex-1 min-w-0">
                      <div className="font-mono text-sm font-semibold mb-1" style={{ color: "var(--accent)" }}>
                        {docId}
                      </div>
                      <div className="text-xs truncate" style={{ color: "var(--text-muted)" }}>
                        {previewFields.map(([k, v]) => (
                          <span key={k} className="mr-3">
                            <span style={{ color: "var(--foreground)" }}>{k}:</span>{" "}
                            {typeof v === "string" ? v.slice(0, 60) : JSON.stringify(v).slice(0, 60)}
                          </span>
                        ))}
                      </div>
                    </div>
                    <div className="flex items-center gap-2 ml-4 flex-shrink-0">
                      <span className="text-xs" style={{ color: "var(--text-muted)" }}>
                        {isExpanded ? "▲" : "▼"}
                      </span>
                      <button
                        onClick={(e) => { e.stopPropagation(); handleDelete(docId); }}
                        disabled={isDeleting}
                        className="text-xs px-2 py-1 rounded transition-colors"
                        style={{
                          background: isDeleting ? "rgba(239,68,68,0.1)" : "rgba(239,68,68,0.08)",
                          color: "#ef4444",
                          border: "1px solid rgba(239,68,68,0.3)",
                          opacity: isDeleting ? 0.6 : 1,
                        }}
                      >
                        {isDeleting ? "Deleting…" : "Delete"}
                      </button>
                    </div>
                  </div>

                  {isExpanded && (
                    <div className="px-6 pb-4">
                      <pre
                        className="text-xs rounded p-3 overflow-auto"
                        style={{ background: "var(--background)", color: "#a5b4fc", maxHeight: "300px" }}
                      >
                        {JSON.stringify(doc, null, 2)}
                      </pre>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between px-6 py-4" style={{ borderTop: "1px solid var(--border)" }}>
            <span className="text-sm" style={{ color: "var(--text-muted)" }}>
              Page {currentPage} of {totalPages}
            </span>
            <div className="flex gap-2">
              <button
                onClick={() => loadDocs(Math.max(0, offset - PAGE_SIZE))}
                disabled={offset === 0}
                className="text-xs px-3 py-1 rounded-lg disabled:opacity-40"
                style={{ background: "var(--background)", border: "1px solid var(--border)" }}
              >
                ← Prev
              </button>
              <button
                onClick={() => loadDocs(offset + PAGE_SIZE)}
                disabled={offset + PAGE_SIZE >= total}
                className="text-xs px-3 py-1 rounded-lg disabled:opacity-40"
                style={{ background: "var(--background)", border: "1px solid var(--border)" }}
              >
                Next →
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
