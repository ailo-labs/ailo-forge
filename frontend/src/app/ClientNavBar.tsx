// src/app/components/ClientNavBar.tsx
"use client";
import React, { useState } from "react";
import Link from "next/link";
import "@/app/styles/ClientNavBar.css"; // Create this CSS file to style the navbar

export default function ClientNavBar() {
  const [menuOpen, setMenuOpen] = useState(false);
  return (
    <header className="navbar">
      <div className="navbar-container">
        <Link href="/" legacyBehavior>
          <a className="navbar-logo">Ailo Forge™</a>
        </Link>
        <nav className={`navbar-links ${menuOpen ? "active" : ""}`}>
          <Link href="/models" legacyBehavior>
            <a>MODELS</a>
          </Link>
          <Link href="/modify" legacyBehavior>
            <a>MODIFY</a>
          </Link>
          <Link href="/chat" legacyBehavior>
            <a>CHAT</a>
          </Link>
          <Link href="/train" legacyBehavior>
            <a>TRAIN</a>
          </Link>
        </nav>
        <button onClick={() => setMenuOpen(!menuOpen)} className="hamburger-btn">
          <span className="hamburger-line"></span>
          <span className="hamburger-line"></span>
          <span className="hamburger-line"></span>
        </button>
      </div>
    </header>
  );
}
