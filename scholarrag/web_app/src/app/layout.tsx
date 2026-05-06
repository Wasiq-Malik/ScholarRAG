import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "ScholaRAG",
  description: "Scientific search and RAG over recent arXiv papers.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
