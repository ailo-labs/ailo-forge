// chat/page.tsx – Chat page that allows the user to converse with their modified model.
// Uses GPT‑4 for current information. The UI is modern and uses framer-motion for simple animations.
"use client";
import React, { useState } from "react";
import axios from "axios";
import { useModel } from "@/context/ModelContext";
import "@/app/styles/ChatPage.css";
import Footer from "@/components/Footer";

export default function ChatPage() {
  const { selectedModel } = useModel();
  const [prompt, setPrompt] = useState("");
  const [chatHistory, setChatHistory] = useState<{ role: string; message: string }[]>([]);
  const [instructMode, setInstructMode] = useState(false);

  // Handle sending a chat message to the backend
  const handleSend = async () => {
    if (!prompt.trim()) return;
    const userMsg = { role: "user", message: prompt };
    setChatHistory((prev) => [...prev, userMsg]);
    const backendUrl = (process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000").replace(/\/+$/, "");
    try {
      const { data } = await axios.post(`${backendUrl}/run`, {
        model_version: selectedModel,
        prompt,
        mode: instructMode ? "instruct" : "conversation"
      });
      const botMsg = { role: "bot", message: data.response.text };
      setChatHistory((prev) => [...prev, botMsg]);
    } catch (error) {
      console.error("Error in chat:", error);
      alert("Error communicating with the model.");
    }
    setPrompt("");
  };

  return (
    <div className="chat-page container">
      <h2>Chat with {selectedModel ? `Custom ${selectedModel}` : "your model"}</h2>
      <div className="chat-mode-toggle">
        <label>
          <input
            type="checkbox"
            checked={instructMode}
            onChange={() => setInstructMode(!instructMode)}
          />
          Instruct Mode
        </label>
      </div>
      <div className="chat-history">
        {chatHistory.map((msg, index) => (
          <div key={index} className={`chat-bubble ${msg.role}`}>
            {msg.message}
          </div>
        ))}
      </div>
      <div className="chat-input">
        <input
          type="text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Type your message..."
        />
        <button onClick={handleSend} className="btn chat-send-btn">Send</button>
      </div>
      <Footer />
    </div>
  );
}
