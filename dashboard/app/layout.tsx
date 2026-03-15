import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "semantix — hybrid search dashboard",
  description: "Self-hostable hybrid search engine dashboard",
};

const NAV = [
  { href: "/", label: "Overview" },
  { href: "/playground", label: "Playground" },
  { href: "/ingest", label: "Ingest" },
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased min-h-screen" style={{ background: "var(--background)", color: "var(--foreground)" }}>
        {/* Navbar */}
        <nav style={{ background: "var(--surface)", borderBottom: "1px solid var(--border)" }}
          className="sticky top-0 z-50 px-6 py-3 flex items-center gap-8">
          <Link href="/" className="font-bold text-lg tracking-tight" style={{ color: "var(--accent)" }}>
            semantix
          </Link>
          <div className="flex gap-6 text-sm">
            {NAV.map(n => (
              <Link key={n.href} href={n.href}
                className="transition-colors hover:text-white"
                style={{ color: "var(--text-muted)" }}>
                {n.label}
              </Link>
            ))}
          </div>
          <div className="ml-auto text-xs px-2 py-1 rounded"
            style={{ background: "var(--border)", color: "var(--text-muted)" }}>
            v0.1.0
          </div>
        </nav>

        <main className="mx-auto max-w-6xl px-6 py-8">
          {children}
        </main>
      </body>
    </html>
  );
}
