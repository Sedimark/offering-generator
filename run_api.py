#!/usr/bin/env python3

import uvicorn
from config import CONFIG

if __name__ == "__main__":
    print("Starting SEDIMARK API Server...")
    print(f"Host: {CONFIG['server']['host']}")
    print(f"Port: {CONFIG['server']['port']}")
    print(f"Workers: {CONFIG['server']['workers']}")

    uvicorn.run(
        "api:app",
        host=CONFIG["server"]["host"],
        port=CONFIG["server"]["port"],
        workers=CONFIG["server"]["workers"],
        reload=False,
        log_level="info"
    )
