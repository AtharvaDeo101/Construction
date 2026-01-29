"""Run the Indoor Navigation API server. Execute from project root."""
import sys

import uvicorn

if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8000
    print("Starting Indoor Navigation API...")
    print(f"  API:  http://{host}:{port}")
    print(f"  Docs: http://{host}:{port}/docs")
    print(f"  Health: http://{host}:{port}/health")
    print("  (Ctrl+C to stop)")
    sys.stdout.flush()
    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=False,
    )
