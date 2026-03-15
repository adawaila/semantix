"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { CollectionStats, fetchCollections } from "@/lib/api";

function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-xl p-4" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
      <div className="text-xs uppercase tracking-widest mb-1" style={{ color: "var(--text-muted)" }}>{label}</div>
      <div className="text-2xl font-bold">{value}</div>
    </div>
  );
}

function CollectionRow({ col }: { col: CollectionStats }) {
  return (
    <Link href={`/collections/${col.name}`}>
      <div className="flex items-center justify-between px-4 py-3 rounded-lg cursor-pointer transition-colors hover:opacity-80"
        style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
        <div>
          <span className="font-semibold">{col.name}</span>
          <span className="text-xs ml-3 px-2 py-0.5 rounded"
            style={{ background: "var(--border)", color: "var(--text-muted)" }}>
            {col.index_type}
          </span>
        </div>
        <div className="flex gap-6 text-sm" style={{ color: "var(--text-muted)" }}>
          <span>{col.doc_count.toLocaleString()} docs</span>
          <span>{col.vector_count.toLocaleString()} vectors</span>
          <span>{col.query_count.toLocaleString()} queries</span>
          <span style={{ color: "var(--accent)" }}>→</span>
        </div>
      </div>
    </Link>
  );
}

export default function OverviewPage() {
  const [collections, setCollections] = useState<CollectionStats[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchCollections().then(data => {
      setCollections(data);
      setLoading(false);
    });
    const interval = setInterval(() => {
      fetchCollections().then(setCollections);
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const totalDocs = collections.reduce((s, c) => s + c.doc_count, 0);
  const totalVectors = collections.reduce((s, c) => s + c.vector_count, 0);
  const totalQueries = collections.reduce((s, c) => s + c.query_count, 0);

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-1">Overview</h1>
        <p style={{ color: "var(--text-muted)" }}>Real-time dashboard for your semantix instance</p>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-4 gap-4 mb-8">
        <StatCard label="Collections" value={collections.length} />
        <StatCard label="Total Documents" value={totalDocs.toLocaleString()} />
        <StatCard label="Total Vectors" value={totalVectors.toLocaleString()} />
        <StatCard label="Total Queries" value={totalQueries.toLocaleString()} />
      </div>

      {/* Collections list */}
      <div className="rounded-xl p-6" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold">Collections</h2>
          <span className="text-xs px-2 py-1 rounded" style={{ background: "var(--border)", color: "var(--text-muted)" }}>
            auto-refresh 5s
          </span>
        </div>

        {loading ? (
          <div className="text-center py-12" style={{ color: "var(--text-muted)" }}>Loading...</div>
        ) : collections.length === 0 ? (
          <div className="text-center py-12" style={{ color: "var(--text-muted)" }}>
            <p className="text-lg mb-2">No collections yet</p>
            <p className="text-sm">Create one with the SDK or POST /collections</p>
          </div>
        ) : (
          <div className="flex flex-col gap-2">
            {collections.map(c => <CollectionRow key={c.name} col={c} />)}
          </div>
        )}
      </div>

      {/* Quick start */}
      <div className="mt-8 rounded-xl p-6" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
        <h2 className="text-lg font-semibold mb-3">Quick Start</h2>
        <pre className="text-sm rounded p-4 overflow-x-auto" style={{ background: "var(--background)", color: "#a5b4fc" }}>
{`from sdk import Semantix

client = Semantix("http://localhost:8000")
client.create_collection("products", embedding_field="description")
client.ingest("products", [
    {"id": "1", "description": "wireless headphones with noise cancelling"}
])
results = client.search("products", "wireless headphones", top_k=10)`}
        </pre>
      </div>
    </div>
  );
}
