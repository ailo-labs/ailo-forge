// src/app/components/Footer.tsx
"use client";
import React from "react";
import Link from "next/link";
import "@/app/styles/Footer.css";

export default function Footer() {
  return (
    <footer className="footer">
      <div className="footer-nav">
        <Link href="/" legacyBehavior><a>Ailo Forge</a></Link>
        <span>|</span>
        <Link href="/models" legacyBehavior><a>Models</a></Link>
        <span>|</span>
        <Link href="/modify" legacyBehavior><a>Modify</a></Link>
        <span>|</span>
        <Link href="/chat" legacyBehavior><a>Chat</a></Link>
        <span>|</span>
        <Link href="/train" legacyBehavior><a>Train</a></Link>
      </div>
      <div className="footer-description">
        <p>
          <strong>Models:</strong> Select and view available AI models.
          &nbsp;&nbsp;
          <strong>Modify:</strong> Customize model parameters.
          &nbsp;&nbsp;
          <strong>Chat:</strong> Interact with your custom model.
          &nbsp;&nbsp;
          
          <strong>Train:</strong> Fine-tune your model with your own dataset.
        </p>
        <p>
          For more details, visit our{" "}
          <a href="https://github.com/your-repo" target="_blank" rel="noopener noreferrer">
            GitHub documentation
          </a>.
        </p>
      </div>
    </footer>
  );
}
