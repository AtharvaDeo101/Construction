"""Run the Indoor Navigation web app (FastAPI + frontend). Execute from project root."""
import sys

import uvicorn

if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8000
    print("Starting Indoor Navigation server...")
    print(f"  Open: http://{host}:{port}")
    print("  Health: http://{}:{}/health".format(host, port))
    print("  (Ctrl+C to stop)")
    sys.stdout.flush()
    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=False,
    )
