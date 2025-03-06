// train/page.tsx – Training page that simulates auto-training and includes pause/resume control.
// The layout is designed to take up most of the screen.
"use client";
import React, { useState } from "react";
import axios from "axios";
import "@/app/styles/TrainPage.css";
import Footer from "@/components/Footer";


export default function TrainPage() {
  const [objective, setObjective] = useState("");
  const [selectedFiles, setSelectedFiles] = useState<FileList | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [jobId, setJobId] = useState("");

  // Poll progress from the backend
  const pollProgress = async (jobId: string) => {
    const backendUrl = (process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000").replace(/\/+$/, "");
    try {
      const { data } = await axios.get(`${backendUrl}/progress/${jobId}`);
      setProgress(data.percent);
      if (data.percent < 100 && data.status !== "canceled") {
        setTimeout(() => pollProgress(jobId), 1000);
      } else {
        setIsTraining(false);
        setProgress(100);
        alert("Training complete! You may now proceed to chat or modify.");
      }
    } catch (error) {
      console.error("Error polling training progress:", error);
      setIsTraining(false);
      alert("Error fetching training progress.");
    }
  };

  // Handle training start
  const handleTrain = async () => {
    if (!objective.trim() || !selectedFiles || selectedFiles.length === 0) {
      alert("Please provide a training objective and select dataset files.");
      return;
    }
    const formData = new FormData();
    formData.append("trainingObjective", objective);
    for (let i = 0; i < selectedFiles.length; i++) {
      formData.append("dataset", selectedFiles[i]);
    }
    setIsTraining(true);
    setProgress(0);
    const backendUrl = (process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000").replace(/\/+$/, "");
    try {
      const { data } = await axios.post(`${backendUrl}/train`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      if (data.job_id) {
        setJobId(data.job_id);
        pollProgress(data.job_id);
      } else {
        alert("Training did not start. Check your inputs.");
        setIsTraining(false);
      }
    } catch (error) {
      console.error("Error initiating training:", error);
      alert("Error initiating training. Check console for details.");
      setIsTraining(false);
    }
  };

  // Pause/Resume training
  const handlePauseToggle = async () => {
    const backendUrl = (process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000").replace(/\/+$/, "");
    try {
      await axios.post(`${backendUrl}/toggle-pause/${jobId}`);
    } catch (error) {
      console.error("Error toggling pause in training:", error);
      alert("Error toggling pause.");
    }
  };

  return (
    <div className="train-page container">
      <h2>Train Your Model </h2>
      <div className="train-form">
        <label>Training Objective:</label>
        <textarea
          value={objective}
          onChange={(e) => setObjective(e.target.value)}
          placeholder="Describe what you want the model to learn..."
        />
        <label>Dataset Files (text files):</label>
        <input type="file" multiple onChange={(e) => setSelectedFiles(e.target.files)} />
        <div className="button-group">
          {isTraining ? (
            <>
              <div className="progress-bar-container">
                <div className="progress-bar" style={{ width: `${progress}%` }}></div>
                <p>{progress}% completed</p>
              </div>
              <button onClick={handlePauseToggle} className="btn pause-btn">
                Pause/Resume Training
              </button>
            </>
          ) : (
            <button onClick={handleTrain} className="btn train-btn">
              Start Training
            </button>
          )}
        </div>
      </div>
      <Footer />
    </div>
  );
}
