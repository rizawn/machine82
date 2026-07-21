import type { Metadata } from "next";
import Sidebar from "@/components/layout/Sidebar";
import Header from "@/components/layout/Header";
import "./globals.css";

export const metadata: Metadata = {
  title: "The Trinity: Institutional AI Trading Architecture",
  description: "Unified AI trading platform combining quantitative muscle, semantic memory brain, and real-time dashboard presentation.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full bg-zinc-950 text-white">
      <body className="h-full flex overflow-hidden">
        <Sidebar />
        <div className="flex-1 flex flex-col min-w-0 h-screen overflow-hidden">
          <Header />
          <main className="flex-1 overflow-y-auto bg-zinc-950 p-8">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
