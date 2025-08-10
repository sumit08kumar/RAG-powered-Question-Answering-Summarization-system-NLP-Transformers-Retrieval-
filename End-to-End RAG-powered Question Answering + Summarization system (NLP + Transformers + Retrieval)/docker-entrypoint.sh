#!/bin/bash

# Start FastAPI backend in background
cd /app/src/api
python app.py &

# Wait for backend to start
sleep 10

# Start Streamlit frontend
cd /app/demo
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0

