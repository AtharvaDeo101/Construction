"use client";

import React, { useRef, useState, useEffect } from "react";
import { Camera, Upload, X, Loader2 } from "lucide-react";
import { InteractiveModelViewer } from "../interactive-model-viewer";

export function VideoUploadSection() {
  const videoInputRef = useRef<HTMLInputElement>(null);
  const cameraVideoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [uploadedVideoUrl, setUploadedVideoUrl] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const [recordedChunks, setRecordedChunks] = useState<Blob[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processError, setProcessError] = useState<string | null>(null);
  const [result, setResult] = useState<{
    sessionId: string;
    status: string;
    progress?: number;
    detail?: string;
    outputs: Record<string, string>;
    isRetrying?: boolean;
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

      if (cameraVideoRef.current) {
        cameraVideoRef.current.srcObject = stream;
        // Important: explicitly start playback
        cameraVideoRef.current
          .play()
          .catch((err) => console.warn("camera play() failed:", err));
      }
    } catch (err) {
      console.error("Error accessing camera:", err);
      alert(
        "Unable to access camera. Make sure you are on HTTPS/localhost and have allowed camera permission."
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
    };

    recorder.start();
    setMediaRecorder(recorder);
    setIsRecording(true);

    // Ensure preview keeps playing from the same stream while recording
    if (cameraVideoRef.current && cameraVideoRef.current.srcObject !== streamRef.current) {
      cameraVideoRef.current.srcObject = streamRef.current;
      cameraVideoRef.current
        .play()
        .catch((err) => console.warn("camera play() during recording failed:", err));
    }
  };

  // Stop recording
  const stopRecording = () => {
    if (mediaRecorder && isRecording) {
      mediaRecorder.stop();
      setIsRecording(false);
      // optional: keep camera open so they can re-record
      // stopCamera();
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
  };

  // Call backend to process video → step1 + step2 + step3
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

      const uploadRes = await fetch("/api/upload-video", {
        method: "POST",
        body: formData,
      });

      if (!uploadRes.ok) {
        const errText = await uploadRes.text();
        let errMsg = errText || "Upload failed";
        try {
          const errJson = JSON.parse(errText);
          const d = errJson.detail;
          errMsg = Array.isArray(d) ? d.map((x: unknown) => String(x)).join("; ") : (d != null ? String(d) : errMsg);
        } catch {
          if (errText) errMsg = errText;
        }
        throw new Error(errMsg);
      }

      const data = await uploadRes.json();
      const session_id = data?.session_id;
      if (!session_id) throw new Error("No session_id returned");

      // Set initial state - polling useEffect will take over from here
      setResult({
        sessionId: session_id,
        status: "processing",
        outputs: {},
        progress: 0.1,
        detail: "Starting pipeline"
      } as any);

    } catch (err: any) {
      console.error(err);
      setProcessError(err.message || "Processing failed");
      setResult(null);
    } finally {
      setIsProcessing(false);
    }
  };

  // Poll for status every 4-5 seconds
  useEffect(() => {
    let interval: NodeJS.Timeout;

    if (result && result.status !== "done" && result.status !== "error") {
      interval = setInterval(async () => {
        try {
          const res = await fetch(`/api/sessions/${result.sessionId}/status`);

          if (!res.ok) {
            console.warn(`Polling status non-OK: ${res.status}`);
            setResult(prev => prev ? { ...prev, isRetrying: true } : null);
            return;
          }

          const data = await res.json();

          setResult(prev => {
            if (!prev || prev.sessionId !== result.sessionId) return prev;

            // Only update if meaningful data changed
            if (prev.status === data.status &&
              (prev as any).progress === data.progress &&
              (prev as any).detail === data.detail &&
              JSON.stringify(prev.outputs) === JSON.stringify(data.outputs) &&
              !prev.isRetrying) {
              return prev;
            }

            return {
              ...prev,
              status: data.status,
              progress: data.progress,
              detail: data.detail,
              outputs: data.outputs || {},
              isRetrying: false
            } as any;
          });
        } catch (err) {
          console.error("Polling error:", err);
          setResult(prev => prev ? { ...prev, isRetrying: true } : null);
        }
      }, 4500); // 4.5 seconds polling
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [result]);

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

        {/* Live camera preview – this is exactly what is being recorded */}
        {isCameraActive && !uploadedVideoUrl && (
          <div className="mb-8 overflow-hidden rounded-2xl bg-black">
            <div className="relative aspect-video w-full bg-black">
              <video
                ref={cameraVideoRef}
                autoPlay
                muted // important for some browsers
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
            <div className="mt-8 w-full max-w-2xl rounded-2xl border border-border bg-secondary/20 p-6 backdrop-blur-md">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h3 className="text-xl font-semibold">
                    {result.status === "done" ? "3D Model Ready" : "Processing Your Space"}
                  </h3>
                  <p className="text-sm text-muted-foreground flex items-center gap-2 mt-1">
                    Session: <code className="bg-muted px-1 rounded">{result.sessionId.slice(0, 8)}...</code>
                    {result.status !== "done" && result.status !== "error" && (
                      <span className="flex items-center gap-1 text-primary animate-pulse">
                        <Loader2 size={14} className="animate-spin" />
                        {result.isRetrying ? "Re-connecting..." : `${result.status}...`}
                      </span>
                    )}
                  </p>
                  {result.detail && (
                    <p className="text-xs text-muted-foreground mt-2 italic flex items-center gap-2">
                      <span className="h-1 w-1 rounded-full bg-primary animate-ping" />
                      {result.detail}
                    </p>
                  )}
                </div>
                {result.status === "done" && (
                  <div className="rounded-full bg-green-500/10 px-3 py-1 text-xs font-semibold text-green-500">
                    Complete
                  </div>
                )}
              </div>

              {/* Progress Bar */}
              {result.status !== "done" && result.status !== "error" && (
                <div className="mb-6">
                  <div className="h-2 w-full overflow-hidden rounded-full bg-secondary/50">
                    <div
                      className="h-full bg-primary transition-all duration-500 ease-out"
                      style={{ width: `${((result as any).progress || 0) * 100}%` }}
                    />
                  </div>
                  <div className="mt-2 flex justify-between text-[10px] text-muted-foreground uppercase tracking-wider font-medium">
                    <span>Processing</span>
                    <span>{Math.round(((result as any).progress || 0) * 100)}%</span>
                  </div>
                </div>
              )}

              {/* Show 3D Viewer if done and GLB exists */}
              {result.status === "done" && result.outputs?.mesh ? (
                <InteractiveModelViewer url={`/api/sessions/${result.sessionId}/outputs/${result.outputs.mesh}`} />
              ) : result.status !== "error" ? (
                <div className="aspect-video w-full rounded-xl bg-black/5 flex flex-col items-center justify-center border border-dashed border-border p-8 text-center">
                  <div className="relative mb-4">
                    <div className="h-16 w-16 rounded-full border-4 border-primary/20" />
                    <div className="absolute inset-0 h-16 w-16 rounded-full border-4 border-primary border-t-transparent animate-spin" />
                  </div>
                  <h4 className="text-lg font-medium mb-1">Building Your 3D Model</h4>
                  <p className="text-sm text-muted-foreground max-w-sm">
                    We're converting your video into a detailed 3D mesh with Draco compression. This usually takes a few minutes.
                  </p>
                </div>
              ) : null}

              {result.status === "error" && (
                <div className="rounded-xl bg-red-500/10 p-4 text-red-500 border border-red-500/20 mt-4">
                  <p className="font-semibold">Processing Failed</p>
                  <p className="text-sm opacity-90">{processError || "An unexpected error occurred."}</p>
                </div>
              )}

              {Object.keys(result.outputs || {}).length > 0 && (
                <div className="mt-6 pt-6 border-t border-border">
                  <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-3">Downloadable Assets</p>
                  <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                    {Object.entries(result.outputs).map(([key, path]) => (
                      <a
                        key={key}
                        href={`/api/sessions/${result.sessionId}/outputs/${path}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex flex-col items-center gap-1 rounded-lg border border-border bg-background p-3 text-center transition-all hover:border-primary hover:shadow-md"
                      >
                        <span className="text-xs font-medium capitalize">{key.replace("_", " ")}</span>
                        <span className="text-[10px] text-muted-foreground truncate w-full">{String(path).split('/').pop()}</span>
                      </a>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
