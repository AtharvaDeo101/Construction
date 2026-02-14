"use client";

import React, { Suspense, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import {
  OrbitControls,
  Stage,
  useGLTF,
  Environment,
  ContactShadows,
  PerspectiveCamera,
  Center
} from "@react-three/drei";
import { Loader2 } from "lucide-react";

interface ModelProps {
  url: string;
}

function Model({ url }: ModelProps) {
  const { scene } = useGLTF(url);

  // Clone the scene to ensure we can manipulate it if needed
  // and to avoid issues with multiple instances
  return <primitive object={scene} />;
}

// Fallback for when the model fails to load
function ModelError() {
  return (
    <div className="flex flex-col items-center justify-center h-full w-full bg-muted/20 rounded-xl border-2 border-dashed border-muted">
      <p className="text-destructive font-medium">Failed to load 3D model</p>
      <p className="text-xs text-muted-foreground mt-1">Please check the console or retry processing.</p>
    </div>
  );
}

// Loading state while model is being fetched
function ModelLoader() {
  return (
    <div className="flex flex-col items-center justify-center h-full w-full">
      <Loader2 className="h-10 w-10 text-primary animate-spin" />
      <p className="mt-4 text-sm font-medium text-muted-foreground">Loading 3D workspace...</p>
    </div>
  );
}

export function InteractiveModelViewer({ url }: { url: string }) {
  return (
    <div className="relative w-full aspect-video md:aspect-[21/9] bg-black/5 rounded-2xl border border-white/10 overflow-hidden shadow-2xl group">
      <Suspense fallback={<ModelLoader />}>
        <Canvas shadows dpr={[1, 2]}>
          <PerspectiveCamera makeDefault position={[5, 3, 5]} fov={45} />

          <Stage environment="city" intensity={0.5}>
            <Center top>
              <Model url={url} />
            </Center>
          </Stage>

          <Environment preset="city" />

          <ContactShadows
            opacity={0.4}
            scale={10}
            blur={2}
            far={4.5}
            resolution={256}
            color="#000000"
          />

          <OrbitControls
            makeDefault
            autoRotate
            autoRotateSpeed={0.5}
            enableDamping
            dampingFactor={0.05}
            minDistance={2}
            maxDistance={20}
          />
        </Canvas>
      </Suspense>

      {/* Subtle overlay hint */}
      <div className="absolute bottom-4 left-4 pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-500">
        <div className="bg-black/60 backdrop-blur-md px-3 py-1.5 rounded-full text-[10px] text-white/80 uppercase tracking-widest font-bold border border-white/10">
          Left click: Rotate • Scroll: Zoom • Right click: Pan
        </div>
      </div>
    </div>
  );
}
