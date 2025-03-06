// MouseEffect.tsx – Implements a custom square cursor (without trailing effect).
// This client component uses simple mousemove events to update cursor position.
"use client";
import React, { useEffect, useRef } from "react";
import "@/app/styles/MouseEffect.css";

export default function MouseEffect() {
  const cursorRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const cursor = cursorRef.current;
    if (!cursor) return;
    const moveCursor = (e: MouseEvent) => {
      cursor.style.transform = `translate(${e.clientX}px, ${e.clientY}px)`;
    };
    document.addEventListener("mousemove", moveCursor);
    return () => {
      document.removeEventListener("mousemove", moveCursor);
    };
  }, []);

  return <div ref={cursorRef} className="custom-cursor" />;
}
