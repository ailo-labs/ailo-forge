// models/page.tsx – Model selection page where users choose a model to modify.
// The page uses a modern design with a list of models.
"use client";
import React from "react";
import { useRouter } from "next/navigation";
import { useModel } from "@/context/ModelContext";
import "@/app/styles/ModelsPage.css";
import Footer from "@/components/Footer";

export default function ModelsPage() {
  const { setSelectedModel, setIsModified } = useModel();
  const router = useRouter();
  // Extended list of models with extra options for demonstration
  const models = [
    { id: "7B-Base", name: "7B-Base", description: "Base conversational model with standard tuning." },
    { id: "67B-Chat", name: "67B-Chat", description: "High capacity chat model with advanced dialogue capabilities." },
    { id: "Llama-3-8B", name: "Llama-3-8B", description: "Optimized for Llama-based tasks with industry-specific parameters." },
    { id: "Mistral-7B", name: "Mistral-7B", description: "Fast and efficient responses with low latency." },
    { id: "Falcon-40B", name: "Falcon-40B", description: "Large model with high versatility and custom tuning." },
    { id: "GPT-J-6B", name: "GPT-J-6B", description: "Open-source model for general tasks with customizable settings." },
    { id: "DeepSeek-Coder-33B", name: "DeepSeek-Coder-33B", description: "Code specialized model with enhanced debugging features." }
  ];

  const handleSelect = (modelId: string) => {
    setSelectedModel(modelId);
    setIsModified(false);
    router.push("/modify");
  };

  return (
    <div className="models-page container">
      <h2>Select an AI Model to Modify</h2>
      <ul className="models-list">
        {models.map((model) => (
          <li key={model.id} onClick={() => handleSelect(model.id)}>
            <h3>{model.name}</h3>
            <p>{model.description}</p>
          </li>
        ))}
      </ul>
      <Footer />
    </div>
  );
}
