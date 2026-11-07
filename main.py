#!/usr/bin/env python3
"""Entry point for Blob Life: Neuroevolution & Economy 2D Simulator."""

import argparse
import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Blob Life — Neuroevolution & Economy Simulator")
    parser.add_argument("--host", default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    args = parser.parse_args()

    print(f"Starting Blob Life GUI on http://localhost:{args.port}")
    uvicorn.run("blob_sim.server:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
