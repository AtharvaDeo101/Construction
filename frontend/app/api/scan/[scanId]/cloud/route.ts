import { NextRequest, NextResponse } from "next/server";
import path from "path";
import fs from "fs/promises";

const DATA_ROOT = path.join(process.cwd(), "..", "data");

// scanId comes from the client, so it must never be trusted as a path segment.
// Scan dirs are created as `scan_<timestamp>` in process-video/route.ts.
const SCAN_ID = /^[A-Za-z0-9_-]+$/;

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ scanId: string }> }
) {
  const { scanId } = await params;

  if (!SCAN_ID.test(scanId)) {
    return new NextResponse("Invalid scan id", { status: 400 });
  }

  const file = path.join(DATA_ROOT, scanId, "pointcloud", "preview_cloud.ply");

  // Belt and braces: even with the regex, confirm we never escaped DATA_ROOT.
  if (!path.resolve(file).startsWith(path.resolve(DATA_ROOT) + path.sep)) {
    return new NextResponse("Invalid scan id", { status: 400 });
  }

  try {
    const buf = await fs.readFile(file);
    return new NextResponse(new Uint8Array(buf), {
      headers: {
        "Content-Type": "application/octet-stream",
        // Scans are immutable once written.
        "Cache-Control": "public, max-age=31536000, immutable",
      },
    });
  } catch {
    return new NextResponse("Preview cloud not found", { status: 404 });
  }
}
