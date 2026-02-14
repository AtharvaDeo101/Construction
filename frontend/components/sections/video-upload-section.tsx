"use client";

import React, { useRef, useState } from "react";
import { Camera, Upload, X } from "lucide-react";

type OrientationSample = {
  t: number;                 // timestamp (ms since page load)
  alpha: number | null;      // rotation around z axis (deg)
  beta: number | null;       // rotation around x axis (deg)
  gamma: number | null;      // rotation around y axis (deg)
};

export function VideoUploadSection() {
  const videoInputRef = useRef<HTMLInputElement>(null);
  const cameraVideoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const orientationListenerRef =
    useRef<((e: DeviceOrientationEvent) => void) | null>(null);

  const [uploadedVideoUrl, setUploadedVideoUrl] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const [recordedChunks, setRecordedChunks] = useState<Blob[]>([]);
  const [orientationData, setOrientationData] = useState<OrientationSample[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processError, setProcessError] = useState<string | null>(null);
  const [result, setResult] = useState<{
    outputDir: string;
    transformsPath: string;
    pointCloudPath: string;
    orientationPath?: string;
  } | null>(null);

  // Handle file upload
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith("video/")) {
      const url = URL.createObjectURL(file);
      setUploadedVideoUrl(url);
      setUploadedFile(file);
      setResult(null);
      setProcessError(null);
      setOrientationData([]);
    }
  };

  // Start orientation tracking
  const startOrientationTracking = async () => {
    if (typeof window === "undefined") return;
    if (orientationListenerRef.current) return;

    const handler = (event: DeviceOrientationEvent) => {
      setOrientationData((prev) => [
        ...prev,
        {
          t: performance.now(),
          alpha: event.alpha,
          beta: event.beta,
          gamma: event.gamma,
        },
      ]);
    };

    try {
      // iOS 13+
      // @ts-ignore
      if (
        typeof DeviceOrientationEvent !== "undefined" &&
        // @ts-ignore
        typeof DeviceOrientationEvent.requestPermission === "function"
      ) {
        // @ts-ignore
        const response = await DeviceOrientationEvent.requestPermission();
        if (response === "granted") {
          window.addEventListener("deviceorientation", handler);
          orientationListenerRef.current = handler;
        } else {
          console.warn("DeviceOrientation permission denied");
        }
      } else {
        window.addEventListener("deviceorientation", handler);
        orientationListenerRef.current = handler;
      }
    } catch (e) {
      console.error("Error requesting device orientation permission:", e);
    }
  };

  const stopOrientationTracking = () => {
    if (typeof window === "undefined") return;
    if (orientationListenerRef.current) {
      window.removeEventListener(
        "deviceorientation",
        orientationListenerRef.current
      );
      orientationListenerRef.current = null;
    }
  };

  // Start camera and show live preview
  const startCamera = async () => {
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert("Camera not supported in this browser / context.");
        return;
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
        audio: true,
      });

      streamRef.current = stream;
      setIsCameraActive(true);
      setUploadedVideoUrl(null);
      setUploadedFile(null);
      setRecordedChunks([]);
      setResult(null);
      setProcessError(null);
      setOrientationData([]);

      if (cameraVideoRef.current) {
        cameraVideoRef.current.srcObject = stream;
        cameraVideoRef.current
          .play()
          .catch((err) => console.warn("camera play() failed:", err));
      }

      await startOrientationTracking();
    } catch (err) {
      console.error("Error accessing camera:", err);
      alert(
        "Unable to access camera. Make sure you're on HTTPS/localhost and camera permission is allowed."
      );
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
    stopOrientationTracking();
  };

  // Start recording
  const startRecording = () => {
    if (!streamRef.current) return;

    const chunks: Blob[] = [];
    let recorder: MediaRecorder;

    try {
      recorder = new MediaRecorder(streamRef.current, {
        mimeType: "video/webm;codecs=vp9",
      });
    } catch (e) {
      console.error("MediaRecorder init error:", e);
      alert("MediaRecorder not supported with this mimeType in this browser.");
      return;
    }

    recorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) {
        chunks.push(e.data);
      }
    };

    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: "video/webm" });
      const file = new File([blob], `capture-${Date.now()}.webm`, {
        type: "video/webm",
      });

      const url = URL.createObjectURL(blob);
      setUploadedVideoUrl(url);
      setUploadedFile(file);
      setRecordedChunks(chunks);
      setResult(null);
      setProcessError(null);
      // orientationData already contains samples from this session
    };

    recorder.start();
    setMediaRecorder(recorder);
    setIsRecording(true);

    if (
      cameraVideoRef.current &&
      cameraVideoRef.current.srcObject !== streamRef.current
    ) {
      cameraVideoRef.current.srcObject = streamRef.current;
      cameraVideoRef.current
        .play()
        .catch((err) =>
          console.warn("camera play() during recording failed:", err)
        );
    }
  };

  // Stop recording
  const stopRecording = () => {
    if (mediaRecorder && isRecording) {
      mediaRecorder.stop();
      setIsRecording(false);
    }
  };

  // Clear uploaded/recorded video
  const clearVideo = () => {
    if (uploadedVideoUrl) {
      URL.revokeObjectURL(uploadedVideoUrl);
    }
    setUploadedVideoUrl(null);
    setUploadedFile(null);
    setRecordedChunks([]);
    setResult(null);
    setProcessError(null);
    setOrientationData([]);
  };

  // Call backend to process video → step1 + step2
  const handleProcessVideo = async () => {
    if (!uploadedFile) {
      alert("Please upload or record a video first.");
      return;
    }

    try {
      setIsProcessing(true);
      setProcessError(null);
      setResult(null);

      const formData = new FormData();
      formData.append("video", uploadedFile);
      formData.append("orientation", JSON.stringify(orientationData));

      const res = await fetch("/api/process-video", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errText = await res.text();
        throw new Error(errText || "Processing failed");
      }

      const json = await res.json();
      setResult(json);
    } catch (err: any) {
      console.error(err);
      setProcessError(err.message || "Processing failed");
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <section className="relative min-h-screen bg-background px-4 py-20 md:py-32 overflow-hidden">
      {/* Background video */}
      <div className="pointer-events-none absolute inset-0 z-0">
        <div className="relative h-full w-full">
          <video
            autoPlay
            loop
            muted
            playsInline
            className="absolute inset-0 h-full w-full object-cover"
            src="/images/upload.mp4"
          />
        </div>
        <div className="absolute inset-0 bg-black/40" />
      </div>

      {/* Foreground content */}
      <div className="relative z-10 mx-auto max-w-4xl">
        <div className="mb-8 text-center">
          <h1 className="mb-2 text-4xl font-bold tracking-tight md:text-5xl">
            Upload Your Content
          </h1>
          <p className="text-lg text-muted-foreground">
            Capture or upload a video to generate a 3D model with camera poses
          </p>
        </div>

        {/* Recorded video preview */}
        {uploadedVideoUrl && (
          <div className="mb-8 overflow-hidden rounded-2xl bg-black/5">
            <div className="relative aspect-video w-full bg-black">
              <video
                src={uploadedVideoUrl}
                controls
                className="h-full w-full object-contain"
              />
              <div className="absolute left-4 top-4 rounded-full bg-black/60 px-3 py-1 text-xs font-semibold text-white">
                Recorded video
              </div>
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

        {/* Live camera preview */}
        {isCameraActive && !uploadedVideoUrl && (
          <div className="mb-8 overflow-hidden rounded-2xl bg-black">
            <div className="relative aspect-video w-full bg-black">
              <video
                ref={cameraVideoRef}
                autoPlay
                muted
                playsInline
                className="h-full w-full object-cover"
              />
              {isRecording && (
                <div className="absolute left-4 top-4 flex items-center gap-2 rounded-full bg-black/60 px-3 py-1 text-xs font-semibold text-red-400">
                  <span className="h-2 w-2 rounded-full bg-red-500 animate-pulse" />
                  REC
                </div>
              )}
            </div>
          </div>
        )}

        {/* Upload / capture options */}
        {!uploadedVideoUrl && !isCameraActive && (
          <div className="mb-8 grid gap-6 md:grid-cols-2">
            {/* File upload */}
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

            {/* Camera capture */}
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

        {/* Camera controls */}
        {isCameraActive && (
          <div className="mb-8 flex flex-wrap justify-center gap-4">
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

        {/* Process button + status */}
        <div className="mb-6 flex flex-col items-center gap-4">
          <button
            onClick={handleProcessVideo}
            disabled={!uploadedFile || isProcessing}
            className="rounded-full bg-foreground px-8 py-3 font-semibold text-background transition-all hover:opacity-90 active:scale-95 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {isProcessing ? "Processing 3D model..." : "Generate 3D Model"}
          </button>

          <p className="text-sm text-muted-foreground text-center">
            {uploadedVideoUrl
              ? "Video ready. Click “Generate 3D Model” to run depth + pose + reconstruction."
              : isCameraActive
              ? isRecording
                ? "Recording in progress..."
                : "Camera is ready. Press 'Start Recording' to begin."
              : "Upload a video or use your device camera to get started."}
          </p>

          {processError && (
            <p className="text-sm text-red-500">{processError}</p>
          )}

          {result && (
            <div className="mt-4 w-full max-w-xl rounded-xl border border-border bg-secondary/40 p-4 text-sm text-left">
              <p className="font-semibold mb-2">3D model generated:</p>
              <ul className="space-y-1">
                <li>
                  Output directory:{" "}
                  <code className="break-all">{result.outputDir}</code>
                </li>
                <li>
                  Camera poses (transforms.json):{" "}
                  <code className="break-all">{result.transformsPath}</code>
                </li>
                <li>
                  Point cloud:{" "}
                  <code className="break-all">{result.pointCloudPath}</code>
                </li>
                {result.orientationPath && (
                  <li>
                    Device orientation:{" "}
                    <code className="break-all">{result.orientationPath}</code>
                  </li>
                )}
              </ul>
              <p className="mt-2 text-muted-foreground">
                DepthAnything3 camera poses and device orientation samples are
                saved together for better 3D reconstruction and analysis.
              </p>
            </div>
          )}

          {orientationData.length > 0 && (
            <p className="text-xs text-muted-foreground">
              Collected {orientationData.length} orientation samples while the
              camera was active.
            </p>
          )}
        </div>
      </div>
    </section>
  );
}
