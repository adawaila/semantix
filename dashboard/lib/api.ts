const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const API_KEY = process.env.NEXT_PUBLIC_API_KEY ?? "";

function authHeaders(): HeadersInit {
  return API_KEY ? { "X-API-Key": API_KEY } : {};
}

export interface CollectionStats {
  name: string;
  doc_count: number;
  vocab_size: number;
  vector_count: number;
  embedding_dim: number;
  embedding_field?: string;
  index_type: string;
  provider: string;
  query_count: number;
  created_at: number;
}

export interface SearchResultItem {
  id: string;
  score: number;
  document: Record<string, unknown>;
  bm25_rank?: number;
  vector_rank?: number;
}

export interface SearchResponse {
  results: SearchResultItem[];
  total_docs: number;
  latency_ms: number;
  query: string;
}

export interface DocumentListResponse {
  documents: Record<string, unknown>[];
  total: number;
  limit: number;
  offset: number;
}

export async function fetchCollections(): Promise<CollectionStats[]> {
  const res = await fetch(`${BASE}/collections`, {
    cache: "no-store",
    headers: authHeaders(),
  });
  if (!res.ok) return [];
  const data = await res.json();
  return data.collections ?? [];
}

export async function fetchCollection(name: string): Promise<CollectionStats | null> {
  const res = await fetch(`${BASE}/collections/${name}`, {
    cache: "no-store",
    headers: authHeaders(),
  });
  if (!res.ok) return null;
  return res.json();
}

export async function fetchDocuments(
  collection: string,
  limit = 50,
  offset = 0,
): Promise<DocumentListResponse> {
  const res = await fetch(
    `${BASE}/collections/${collection}/documents?limit=${limit}&offset=${offset}`,
    { cache: "no-store", headers: authHeaders() },
  );
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function deleteDocument(collection: string, docId: string): Promise<void> {
  const res = await fetch(
    `${BASE}/collections/${collection}/documents/${encodeURIComponent(docId)}`,
    { method: "DELETE", headers: authHeaders() },
  );
  if (!res.ok && res.status !== 204) throw new Error(await res.text());
}

export async function search(
  collection: string,
  query: string,
  topK: number,
  alpha: number,
): Promise<SearchResponse> {
  const res = await fetch(`${BASE}/collections/${collection}/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ query, top_k: topK, alpha }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
