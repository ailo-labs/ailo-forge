// ModelContext.tsx – React Context to share the selected model and modification state across pages.
// This file is used to persist state for multi–user interactions.
"use client";
import React, { createContext, useContext, useState, ReactNode } from "react";

type ModelContextType = {
  selectedModel: string | null;
  setSelectedModel: (model: string | null) => void;
  isModified: boolean;
  setIsModified: (val: boolean) => void;
};

const ModelContext = createContext<ModelContextType>({
  selectedModel: null,
  setSelectedModel: () => {},
  isModified: false,
  setIsModified: () => {},
});

export function ModelProvider({ children }: { children: ReactNode }) {
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [isModified, setIsModified] = useState<boolean>(false);
  return (
    <ModelContext.Provider value={{ selectedModel, setSelectedModel, isModified, setIsModified }}>
      {children}
    </ModelContext.Provider>
  );
}

export function useModel() {
  return useContext(ModelContext);
}

