"""
Demo runner script for RAG system.
Starts both the FastAPI backend and Streamlit frontend.
"""

import subprocess
import time
import sys
import os
import signal
import threading
from pathlib import Path

def run_fastapi():
    """Run the FastAPI backend."""
    print("🚀 Starting FastAPI backend...")
    os.chdir(Path(__file__).parent / "src" / "api")
    subprocess.run([sys.executable, "app.py"])

def run_streamlit():
    """Run the Streamlit frontend."""
    print("🎨 Starting Streamlit frontend...")
    time.sleep(5)  # Wait for FastAPI to start
    os.chdir(Path(__file__).parent / "demo")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.port", "8501"])

def main():
    """Main function to run both services."""
    print("🤖 Starting RAG Question-Answering System Demo")
    print("=" * 50)
    
    try:
        # Start FastAPI in a separate thread
        fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
        fastapi_thread.start()
        
        # Start Streamlit in main thread
        run_streamlit()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
        sys.exit(0)

if __name__ == "__main__":
    main()

