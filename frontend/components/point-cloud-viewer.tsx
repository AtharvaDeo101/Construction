"use client";

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

export function PointCloudViewer({ scanId }: { scanId: string }) {
  const mountRef = useRef<HTMLDivElement>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [error, setError] = useState<string | null>(null);
  const [count, setCount] = useState(0);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0a);

    const camera = new THREE.PerspectiveCamera(
      60,
      mount.clientWidth / mount.clientHeight,
      0.001,
      1000
    );

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    let points: THREE.Points | null = null;
    let raf = 0;
    let disposed = false;

    new PLYLoader().load(
      `/api/scan/${scanId}/cloud`,
      (geometry) => {
        if (disposed) return;

        geometry.computeBoundingSphere();
        const sphere = geometry.boundingSphere;
        const radius = sphere?.radius ?? 1;

        // Depth is relative, so scan extents vary wildly between runs. Size the points
        // and the camera from the cloud's own radius instead of fixed distances.
        const material = new THREE.PointsMaterial({
          size: radius * 0.004,
          vertexColors: geometry.hasAttribute("color"),
          color: geometry.hasAttribute("color") ? 0xffffff : 0x88ccff,
          sizeAttenuation: true,
        });

        points = new THREE.Points(geometry, material);
        scene.add(points);

        const center = sphere?.center ?? new THREE.Vector3();
        controls.target.copy(center);
        camera.position.set(
          center.x + radius * 1.6,
          center.y + radius * 0.7,
          center.z + radius * 1.6
        );
        camera.near = radius / 500;
        camera.far = radius * 100;
        camera.updateProjectionMatrix();
        controls.update();

        setCount(geometry.getAttribute("position").count);
        setStatus("ready");
      },
      undefined,
      (err) => {
        if (disposed) return;
        setError(err instanceof Error ? err.message : "Failed to load point cloud");
        setStatus("error");
      }
    );

    const animate = () => {
      raf = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    const onResize = () => {
      if (!mount.clientWidth || !mount.clientHeight) return;
      camera.aspect = mount.clientWidth / mount.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mount.clientWidth, mount.clientHeight);
    };
    const observer = new ResizeObserver(onResize);
    observer.observe(mount);

    return () => {
      disposed = true;
      cancelAnimationFrame(raf);
      observer.disconnect();
      controls.dispose();
      // Point clouds are large; drop the GPU buffers explicitly rather than waiting
      // for GC, or repeated scans leak a few hundred MB of VRAM each.
      if (points) {
        points.geometry.dispose();
        (points.material as THREE.Material).dispose();
        scene.remove(points);
      }
      renderer.dispose();
      if (renderer.domElement.parentNode === mount) {
        mount.removeChild(renderer.domElement);
      }
    };
  }, [scanId]);

  return (
    <div className="relative w-full overflow-hidden rounded-xl border border-border bg-black">
      <div ref={mountRef} className="h-[480px] w-full" />

      {status === "loading" && (
        <div className="absolute inset-0 grid place-items-center text-sm text-white/70">
          Loading point cloud...
        </div>
      )}
      {status === "error" && (
        <div className="absolute inset-0 grid place-items-center px-4 text-center text-sm text-red-400">
          {error}
        </div>
      )}
      {status === "ready" && (
        <div className="pointer-events-none absolute bottom-2 left-3 text-xs text-white/50">
          {count.toLocaleString()} points &middot; drag to orbit, scroll to zoom
        </div>
      )}
    </div>
  );
}
