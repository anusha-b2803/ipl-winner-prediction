import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "IPL Prophet — AI Winner Prediction",
  description: "RAG-powered IPL champion prediction using historical statistics from 2008–2024",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="bg-ipl-blue text-white font-['DM_Sans',sans-serif] antialiased">
        {children}
      </body>
    </html>
  );
}
