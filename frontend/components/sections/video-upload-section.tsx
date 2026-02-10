"use client";

import React from "react"

import { useRef, useState } from "react";
import { Camera, Upload, Play, X } from "lucide-react";

export function VideoUploadSection() {
  const videoInputRef = useRef<HTMLInputElement>(null);
  const cameraVideoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [uploadedVideo, setUploadedVideo] = useState<string | null>(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const [recordedChunks, setRecordedChunks] = useState<Blob[]>([]);

  // Handle file upload
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith("video/")) {
      const url = URL.createObjectURL(file);
      setUploadedVideo(url);
    }
  };

  // Start camera
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
        audio: true,
      });
      
      if (cameraVideoRef.current) {
        cameraVideoRef.current.srcObject = stream;
        streamRef.current = stream;
        setIsCameraActive(true);
      }
    } catch (err) {
      console.error("Error accessing camera:", err);
      alert("Unable to access camera. Please check permissions.");
    }
  };

  // Stop camera
  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    setIsCameraActive(false);
    setIsRecording(false);
  };

  // Start recording
  const startRecording = () => {
    if (!streamRef.current) return;

    const chunks: Blob[] = [];
    const recorder = new MediaRecorder(streamRef.current, {
      mimeType: "video/webm;codecs=vp9",
    });

    recorder.ondataavailable = (e) => {
      chunks.push(e.data);
    };

    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: "video/webm" });
      const url = URL.createObjectURL(blob);
      setUploadedVideo(url);
      setRecordedChunks(chunks);
    };

    recorder.start();
    setMediaRecorder(recorder);
    setIsRecording(true);
  };

  // Stop recording
  const stopRecording = () => {
    if (mediaRecorder && isRecording) {
      mediaRecorder.stop();
      setIsRecording(false);
      stopCamera();
    }
  };

  // Clear uploaded video
  const clearVideo = () => {
    if (uploadedVideo) {
      URL.revokeObjectURL(uploadedVideo);
    }
    setUploadedVideo(null);
    setRecordedChunks([]);
  };

  return (
    <section className="min-h-screen bg-background px-4 py-20 md:py-32">
      <div className="mx-auto max-w-4xl">
        {/* Section Title */}
        <div className="mb-16 text-center">
          <h1 className="mb-4 text-4xl font-bold tracking-tight md:text-5xl">
            Upload Your Content
          </h1>
          <p className="text-lg text-muted-foreground">
            Capture or upload a video to explore our interactive gallery
          </p>
        </div>

        {/* Video Display */}
        {uploadedVideo && (
          <div className="mb-12 overflow-hidden rounded-2xl bg-black/5">
            <div className="relative aspect-video w-full bg-black">
              <video
                src={uploadedVideo}
                controls
                className="h-full w-full object-contain"
              />
              <button
                onClick={clearVideo}
                className="absolute right-4 top-4 rounded-full bg-background/80 p-2 backdrop-blur-sm transition-all hover:bg-background"
                aria-label="Clear video"
              >
                <X size={20} />
              </button>
            </div>
          </div>
        )}

        {/* Camera Preview */}
        {isCameraActive && !uploadedVideo && (
          <div className="mb-12 overflow-hidden rounded-2xl bg-black">
            <div className="relative aspect-video w-full bg-black">
              <video
                ref={cameraVideoRef}
                autoPlay
                playsInline
                className="h-full w-full object-cover"
              />
            </div>
          </div>
        )}

        {/* Upload Options */}
        {!uploadedVideo && !isCameraActive && (
          <div className="mb-12 grid gap-6 md:grid-cols-2">
            {/* File Upload */}
            <div className="group relative overflow-hidden rounded-2xl border-2 border-dashed border-border bg-secondary/50 p-12 text-center transition-all hover:border-foreground hover:bg-secondary">
              <input
                ref={videoInputRef}
                type="file"
                accept="video/*"
                onChange={handleFileUpload}
                className="absolute inset-0 cursor-pointer opacity-0"
              />
              <div className="flex flex-col items-center gap-4">
                <div className="rounded-full bg-background p-4 transition-transform group-hover:scale-110">
                  <Upload size={28} className="text-foreground" />
                </div>
                <div>
                  <h3 className="mb-2 text-lg font-semibold">Upload Video</h3>
                  <p className="text-sm text-muted-foreground">
                    Click to select a video file from your device
                  </p>
                </div>
              </div>
            </div>

            {/* Camera Capture */}
            <button
              onClick={startCamera}
              className="group relative overflow-hidden rounded-2xl border-2 border-dashed border-border bg-secondary/50 p-12 text-center transition-all hover:border-foreground hover:bg-secondary"
            >
              <div className="flex flex-col items-center gap-4">
                <div className="rounded-full bg-background p-4 transition-transform group-hover:scale-110">
                  <Camera size={28} className="text-foreground" />
                </div>
                <div>
                  <h3 className="mb-2 text-lg font-semibold">Capture Video</h3>
                  <p className="text-sm text-muted-foreground">
                    Use your device camera to record
                  </p>
                </div>
              </div>
            </button>
          </div>
        )}

        {/* Camera Controls */}
        {isCameraActive && (
          <div className="mb-12 flex flex-wrap justify-center gap-4">
            {!isRecording ? (
              <button
                onClick={startRecording}
                className="flex items-center gap-2 rounded-full bg-foreground px-6 py-3 font-semibold text-background transition-all hover:opacity-90 active:scale-95"
              >
                <div className="h-3 w-3 animate-pulse rounded-full bg-current" />
                Start Recording
              </button>
            ) : (
              <button
                onClick={stopRecording}
                className="flex items-center gap-2 rounded-full bg-red-600 px-6 py-3 font-semibold text-white transition-all hover:opacity-90 active:scale-95"
              >
                <div className="h-3 w-3 animate-pulse rounded-full bg-current" />
                Stop Recording
              </button>
            )}

            <button
              onClick={stopCamera}
              className="rounded-full border-2 border-border px-6 py-3 font-semibold transition-all hover:border-foreground hover:bg-secondary"
            >
              Close Camera
            </button>
          </div>
        )}

        {/* Info Text */}
        <div className="rounded-2xl bg-secondary/50 p-8 text-center">
          <p className="text-sm text-muted-foreground">
            {uploadedVideo
              ? "Video ready! Scroll down to explore the gallery."
              : isCameraActive
              ? isRecording
                ? "Recording in progress..."
                : "Camera is ready. Press 'Start Recording' to begin."
              : "Upload a video or use your device camera to get started."}
          </p>
        </div>
      </div>

      {/* Hidden canvas for recording */}
      <canvas ref={canvasRef} className="hidden" />
    </section>
  );
}
