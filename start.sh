#!/bin/bash
# QuantPyTrader Quick Start Script

echo "🚀 Starting QuantPyTrader Development Environment..."

# Activate virtual environment
source .venv/bin/activate

# Copy environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "📋 Creating .env file from template..."
    cp .env.example .env
    echo "✅ Created .env file - please update with your API keys"
fi

# Check if user wants to start FastAPI server
echo ""
echo "Choose what to start:"
echo "1) FastAPI Backend (port 8000)"
echo "2) Streamlit Dashboard (port 8501)"  
echo "3) Both (recommended)"
echo "4) Docker Compose (all services)"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo "🔧 Starting FastAPI Backend..."
        python main.py
        ;;
    2)
        echo "📊 Starting Streamlit Dashboard..."
        streamlit run run_dashboard.py
        ;;
    3)
        echo "🔧📊 Starting both FastAPI and Streamlit..."
        echo "FastAPI will run on http://localhost:8000"
        echo "Streamlit will run on http://localhost:8501"
        python main.py &
        streamlit run run_dashboard.py
        ;;
    4)
        echo "🐳 Starting Docker Compose..."
        docker-compose up --build
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac