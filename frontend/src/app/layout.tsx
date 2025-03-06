// src/app/layout.tsx
import React from "react";
import { ModelProvider } from "@/context/ModelContext";
import ClientNavBar from "@/app/ClientNavBar";
import "@/app/globals.css";

/**
 * Export metadata (this file is a server component)
 */
export const metadata = {
  title: "Ailo Forge",
  description: "Online LLM Modification, Training, and Chatting Platform",
};

/**
 * RootLayout wraps your entire app.
 */
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <ModelProvider>
          <ClientNavBar />
          <main>{children}</main>
        </ModelProvider>
      </body>
    </html>
  );
}

