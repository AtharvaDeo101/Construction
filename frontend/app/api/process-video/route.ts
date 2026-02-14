import { NextRequest, NextResponse } from "next/server";
import { spawn } from "child_process";
import path from "path";
import fs from "fs/promises";

const PROJECT_ROOT = process.cwd();
const DATA_ROOT = path.join(PROJECT_ROOT, "..", "data");

async function ensureDir(dir: string) {
  await fs.mkdir(dir, { recursive: true });
}

async function saveUploadedVideo(file: File): Promise<string> {
  const buffer = Buffer.from(await file.arrayBuffer());
  const uploadsDir = path.join(DATA_ROOT, "uploads");
  await ensureDir(uploadsDir);

  const ext = (file.name.split(".").pop() || "mp4").toLowerCase();
  const filename = `upload_${Date.now()}.${ext}`;
  const fullPath = path.join(uploadsDir, filename);
  await fs.writeFile(fullPath, buffer);
  return fullPath;
}

async function saveOrientationJson(
  orientationJson: string | null,
  outputDir: string
): Promise<string | null> {
  if (!orientationJson) return null;
  try {
    JSON.parse(orientationJson); // validate
  } catch {
    console.warn("Invalid orientation JSON, skipping save");
    return null;
  }

  await ensureDir(outputDir);
  const orientationPath = path.join(outputDir, "device_orientation.json");
  await fs.writeFile(orientationPath, orientationJson, "utf-8");
  return orientationPath;
}

function runPython(
  scriptPath: string,
  args: string[],
  extraEnv: Record<string, string> = {}
): Promise<void> {
  return new Promise((resolve, reject) => {
    // Adjust this to your venv / Python path
    const pythonExe = path.join(PROJECT_ROOT, "..", ".venv", "Scripts", "python.exe");
    const env = { ...process.env, ...extraEnv };

    const proc = spawn(pythonExe, [scriptPath, ...args], {
      env,
      cwd: path.join(PROJECT_ROOT, ".."),
    });

    proc.stdout.on("data", (data) => {
      console.log(`[python stdout] ${data.toString()}`);
    });
    proc.stderr.on("data", (data) => {
      console.error(`[python stderr] ${data.toString()}`);
    });

    proc.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(`${path.basename(scriptPath)} exited with code ${code}`));
      } else {
        resolve();
      }
    });
  });
}

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const file = formData.get("video");
    const orientation = formData.get("orientation");

    if (!file || !(file instanceof File)) {
      return new NextResponse("Missing video file", { status: 400 });
    }

    const videoPath = await saveUploadedVideo(file);

    const scanId = `scan_${Date.now()}`;
    const outputDir = path.join(DATA_ROOT, scanId);
    await ensureDir(outputDir);

    const orientationJson =
      typeof orientation === "string" ? orientation : null;
    const orientationPath = await saveOrientationJson(
      orientationJson,
      outputDir
    );

    const step1Path = path.join(
      PROJECT_ROOT,
      "..",
      "src",
      "step1_extract_and_process.py"
    );
    const step2Path = path.join(
      PROJECT_ROOT,
      "..",
      "src",
      "step2_reconstruction.py"
    );

    // STEP 1: video → depth + DA3 poses + raw_cloud.ply
    await runPython(step1Path, [videoPath, outputDir]);

    // STEP 2: reconstruction, controlled via env output dir
    await runPython(step2Path, [], {
      CONSTRUCT_OUTPUT_DIR: outputDir,
    });

    const transformsPath = path.join(outputDir, "transforms.json");
    const pointCloudPath = path.join(
      outputDir,
      "pointcloud",
      "raw_cloud.ply"
    );

    return NextResponse.json({
      success: true,
      outputDir,
      transformsPath,
      pointCloudPath,
      orientationPath,
    });
  } catch (err: any) {
    console.error(err);
    return new NextResponse(err.message || "Internal server error", {
      status: 500,
    });
  }
}
