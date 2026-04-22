#!/bin/bash

echo "===================================="
echo "   ECG Analyzer with Denoising"
echo "===================================="
echo ""

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing Node.js dependencies..."
    npm install
    echo ""
fi

# Check if Python is available
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo "WARNING: Python not found. Denoising features will not work."
    echo "Please install Python 3.7+ and run: pip install -r requirements.txt"
    echo ""
fi

echo "Starting services..."
echo "- React frontend: http://localhost:3002"
echo "- Denoising API: http://localhost:3001"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Start both servers
npm run dev:all
