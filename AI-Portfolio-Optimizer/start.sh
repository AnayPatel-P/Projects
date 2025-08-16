#!/bin/bash

# AI Portfolio Optimizer Enhanced Version - Startup Script
# This script starts both the backend and frontend services

echo "🚀 Starting AI Portfolio Optimizer Enhanced Version"
echo "=================================================="

# Function to cleanup on exit
cleanup() {
    echo "🛑 Shutting down services..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

# Set cleanup trap
trap cleanup INT TERM

# Start backend
echo "📡 Starting Backend Server..."
cd backend
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8001 &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Check if backend is running
if curl -s http://localhost:8001/health > /dev/null; then
    echo "✅ Backend running at http://localhost:8001"
    echo "   📚 API Docs: http://localhost:8001/docs"
else
    echo "❌ Backend failed to start"
    exit 1
fi

# Start frontend
echo "🎨 Starting Frontend Server..."
cd ../frontend
npm run dev &
FRONTEND_PID=$!

# Wait a moment for frontend to start
sleep 5

# Check if frontend is running
if curl -s http://localhost:5173 > /dev/null; then
    echo "✅ Frontend running at http://localhost:5173"
else
    echo "❌ Frontend failed to start"
    kill $BACKEND_PID
    exit 1
fi

echo ""
echo "🎉 AI Portfolio Optimizer is ready!"
echo "=================================================="
echo "🌐 Frontend: http://localhost:5173"
echo "📡 Backend:  http://localhost:8001"
echo "📚 API Docs: http://localhost:8001/docs"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for user to stop
wait