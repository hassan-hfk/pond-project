#!/bin/bash
# Start all services for the detection system

echo "================================================"
echo "🚀 Starting Detection System Services"
echo "================================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
    echo -e "${RED}❌ Please don't run as root${NC}"
    exit 1
fi

# Function to check if port is available
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        echo -e "${YELLOW}⚠️  Port $1 already in use${NC}"
        return 1
    else
        echo -e "${GREEN}✅ Port $1 available${NC}"
        return 0
    fi
}

# Check Python
echo ""
echo "Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✅ $PYTHON_VERSION${NC}"
else
    echo -e "${RED}❌ Python3 not found${NC}"
    exit 1
fi

# Check required packages
echo ""
echo "Checking required packages..."
python3 -c "import fastapi" 2>/dev/null || {
    echo -e "${RED}❌ FastAPI not installed${NC}"
    echo "Install with: pip install fastapi uvicorn"
    exit 1
}
echo -e "${GREEN}✅ FastAPI installed${NC}"

python3 -c "import flask" 2>/dev/null || {
    echo -e "${RED}❌ Flask not installed${NC}"
    echo "Install with: pip install flask"
    exit 1
}
echo -e "${GREEN}✅ Flask installed${NC}"

# Check ports
echo ""
echo "Checking ports..."
check_port 5000 || {
    echo "Kill process using port 5000 or change Flask port"
}
check_port 8000 || {
    echo "Kill process using port 8000 or change FastAPI port"
}

# Start FastAPI (Feedback Server)
echo ""
echo "================================================"
echo "Starting FastAPI Feedback Server (Port 8000)..."
echo "================================================"
cd "$(dirname "$0")"
python3 src/feedback_api.py > logs/feedback_api.log 2>&1 &
FASTAPI_PID=$!
echo -e "${GREEN}✅ FastAPI started (PID: $FASTAPI_PID)${NC}"
echo "Logs: tail -f logs/feedback_api.log"

# Wait for FastAPI to start
sleep 2

# Check if FastAPI is running
if curl -s http://localhost:8000/health > /dev/null; then
    echo -e "${GREEN}✅ FastAPI is responding${NC}"
else
    echo -e "${RED}❌ FastAPI failed to start${NC}"
    exit 1
fi

# Start Flask (Dashboard)
echo ""
echo "================================================"
echo "Starting Flask Dashboard (Port 5000)..."
echo "================================================"
python3 web/app.py > logs/flask_app.log 2>&1 &
FLASK_PID=$!
echo -e "${GREEN}✅ Flask started (PID: $FLASK_PID)${NC}"
echo "Logs: tail -f logs/flask_app.log"

# Wait for Flask to start
sleep 3

# Check if Flask is running
if curl -s http://localhost:5000 > /dev/null; then
    echo -e "${GREEN}✅ Flask is responding${NC}"
else
    echo -e "${YELLOW}⚠️  Flask may still be starting...${NC}"
fi

# Save PIDs
echo $FASTAPI_PID > logs/fastapi.pid
echo $FLASK_PID > logs/flask.pid

# Display status
echo ""
echo "================================================"
echo "✅ All Services Started"
echo "================================================"
echo ""
echo "📊 Service Status:"
echo "  FastAPI (Feedback):  http://localhost:8000 (PID: $FASTAPI_PID)"
echo "  Flask (Dashboard):   http://localhost:5000 (PID: $FLASK_PID)"
echo ""
echo "📝 Logs:"
echo "  FastAPI: tail -f logs/feedback_api.log"
echo "  Flask:   tail -f logs/flask_app.log"
echo ""
echo "🛑 Stop Services:"
echo "  Run: ./stop_all_services.sh"
echo "  Or:  kill $FASTAPI_PID $FLASK_PID"
echo ""
echo "🧪 Test:"
echo "  Health Check:  curl http://localhost:8000/health"
echo "  Test Email:    curl http://localhost:5000/api/email/test"
echo "  View Stats:    curl http://localhost:8000/stats"
echo ""
echo "================================================"