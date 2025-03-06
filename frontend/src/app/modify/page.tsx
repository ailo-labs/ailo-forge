// modify/page.tsx – Allows users to modify a model’s configuration.
// Includes standard and advanced modes, progress bars, and pause/resume buttons.
"use client";
import React, { useState } from "react";
import { useRouter } from "next/navigation";
import axios from "axios";
import { useModel } from "@/context/ModelContext";
import SliderControl from "@/components/SliderControl";
import "@/app/styles/ModifyModelPage.css";
import Footer from "@/components/Footer";

export default function ModifyModelPage() {
  const { selectedModel, setIsModified } = useModel();
  const router = useRouter();
  const [mode, setMode] = useState<"standard" | "advanced">("standard");

  // Standard mode state
  const [temperature, setTemperature] = useState(0.5);
  const [responseSpeed, setResponseSpeed] = useState(50);
  const [tokenLimit, setTokenLimit] = useState(512);
  const [unrestrictedMode, setUnrestrictedMode] = useState(false);
  const [instructions, setInstructions] = useState("");

  // Advanced mode JSON state
  const [advancedJSON, setAdvancedJSON] = useState(
    `{"specialMode": true, "customDomain": "finance", "compliance": "strict", "additionalSettings": {}}`
  );

  // Progress and pause state
  const [isModifying, setIsModifying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [jobId, setJobId] = useState("");

  // Poll the progress endpoint and update state
  const pollProgress = async (jobId: string) => {
    const backendUrl = (process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000").replace(/\/+$/, "");
    try {
      const { data } = await axios.get(`${backendUrl}/progress/${jobId}`);
      setProgress(data.percent);
      if (data.percent < 100 && data.status !== "canceled") {
        setTimeout(() => pollProgress(jobId), 1000);
      } else {
        setIsModifying(false);
        setProgress(100);
        alert("Modification complete!");
        setIsModified(true);
      }
    } catch (error) {
      console.error("Error polling progress:", error);
      setIsModifying(false);
      alert("Error fetching progress.");
    }
  };

  // Handle starting the modification process
  const handleModify = async () => {
    if (!selectedModel) {
      alert("Please select a model first.");
      return;
    }
    setIsModifying(true);
    setProgress(0);
    let payload;
    if (mode === "standard") {
      payload = {
        model_version: selectedModel,
        modifications: {
          instructions,
          temperature,
          responseSpeed,
          tokenLimit,
          unrestrictedMode,
        },
      };
    } else {
      try {
        payload = {
          model_version: selectedModel,
          advanced: true,
          modifications: JSON.parse(advancedJSON),
        };
      } catch {
        alert("Invalid JSON in advanced mode!");
        setIsModifying(false);
        return;
      }
    }
    const backendUrl = (process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000").replace(/\/+$/, "");
    try {
      const { data } = await axios.post(`${backendUrl}/modify-file`, payload, {
        headers: { "Content-Type": "application/json" },
      });
      if (data.job_id) {
        setJobId(data.job_id);
        pollProgress(data.job_id);
      } else {
        alert("Modification did not start. Check your inputs.");
        setIsModifying(false);
      }
    } catch (error: any) {
      console.error("Error modifying model:", error);
      alert(`Error: ${error.response?.data.detail || "Modification failed."}`);
      setIsModifying(false);
    }
  };

  // Function to toggle pause/resume of the modification process
  const handlePauseToggle = async () => {
    const backendUrl = (process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000").replace(/\/+$/, "");
    try {
      await axios.post(`${backendUrl}/toggle-pause/${jobId}`);
    } catch (error) {
      console.error("Error toggling pause:", error);
      alert("Error toggling pause.");
    }
  };

  const NextActions = () => (
    <div className="next-actions">
      <button className="btn next-btn" onClick={() => router.push("/chat")}>Go to Chat</button>
      <button className="btn next-btn" onClick={() => router.push("/train")}>Go to Train</button>
    </div>
  );

  if (!selectedModel) {
    return (
      <div className="modify-model-page container">
        <h2>No Model Selected</h2>
        <p>Please go to <a href="/models">Model Selection</a> first.</p>
      </div>
    );
  }

  return (
    <div className="modify-model-page container">
      <h2>Modify {selectedModel}</h2>
      <p className="modify-subtitle">
        Fine-tune your model’s behavior. Choose a mode:
        <br />
        <small>Standard: Use sliders & instructions. Advanced: Provide raw JSON modifications.</small>
      </p>
      <div className="mode-toggle">
        <button className={`mode-btn ${mode === "standard" ? "active" : ""}`} onClick={() => setMode("standard")} disabled={isModifying}>
          Standard Mode
        </button>
        <button className={`mode-btn ${mode === "advanced" ? "active" : ""}`} onClick={() => setMode("advanced")} disabled={isModifying}>
          Advanced Mode
        </button>
      </div>
      {mode === "standard" ? (
        <div className="standard-interface">
          <SliderControl
            label="Creativity (Temperature)"
            min={0}
            max={1}
            step={0.01}
            value={temperature}
            onChange={(e) => setTemperature(parseFloat(e.target.value))}
            description="Adjust how creative responses are."
            disabled={isModifying}
          />
          <SliderControl
            label="Response Speed"
            min={10}
            max={100}
            step={1}
            value={responseSpeed}
            onChange={(e) => setResponseSpeed(parseInt(e.target.value))}
            description="Set the speed of responses."
            disabled={isModifying}
          />
          <SliderControl
            label="Token Limit"
            min={128}
            max={2048}
            step={1}
            value={tokenLimit}
            onChange={(e) => setTokenLimit(parseInt(e.target.value))}
            description="Set maximum tokens per response."
            disabled={isModifying}
          />
          <div className="toggle-control">
            <label className="tooltip">
              <input
                type="checkbox"
                checked={unrestrictedMode}
                onChange={() => setUnrestrictedMode(!unrestrictedMode)}
                disabled={isModifying}
              />
              Unrestricted Mode
              <span className="tooltiptext">Allows generation without standard filters.</span>
            </label>
          </div>
          <div className="instructions-control">
            <label>Additional Instructions:</label>
            <textarea
              placeholder="E.g. Focus on finance compliance..."
              value={instructions}
              onChange={(e) => setInstructions(e.target.value)}
              disabled={isModifying}
            />
          </div>
        </div>
      ) : (
        <div className="advanced-json">
          <label>Enter Advanced JSON:</label>
          <textarea
            value={advancedJSON}
            onChange={(e) => setAdvancedJSON(e.target.value)}
            rows={8}
            disabled={isModifying}
          />
          <small className="json-hint">
            Example: {"{\"fineTuneDataset\": \"marketing_corpus\", \"unrestricted\": true}"}
          </small>
        </div>
      )}
      <div className="button-group">
        {isModifying ? (
          <div className="progress-bar-container">
            <div className="progress-bar" style={{ width: `${progress}%` }}></div>
            <p>{progress}% completed</p>
            <button onClick={handlePauseToggle} className="btn pause-btn">
              Pause/Resume
            </button>
          </div>
        ) : (
          <button onClick={handleModify} className="btn modify-btn">
            Modify File
          </button>
        )}
      </div>
      {!isModifying && progress === 100 && <NextActions />}
      <Footer />
    </div>
  );
}
