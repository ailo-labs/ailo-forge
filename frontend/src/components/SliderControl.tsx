// SliderControl.tsx – A reusable slider component.
// It displays a label, a range input, and a description.
// This component is used on the Modify page.
"use client";
import React from "react";
import "@/app/styles/SliderControl.css";

type SliderControlProps = {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  description: string;
  disabled?: boolean;
};

export default function SliderControl(props: SliderControlProps) {
  return (
    <div className="slider-control">
      <label>{props.label}</label>
      <input
        type="range"
        min={props.min}
        max={props.max}
        step={props.step}
        value={props.value}
        onChange={props.onChange}
        disabled={props.disabled}
      />
      <div className="slider-value">{props.value}</div>
      <div className="slider-description">{props.description}</div>
    </div>
  );
}
