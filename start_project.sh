#!/bin/bash

# Notebook Agent - Full Stack Startup Script
echo "🚀 Starting Notebook Agent - Full Stack Application"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required tools are installed
check_requirements() {
    print_status "Checking requirements..."
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        print_error "Node.js is not installed. Please install Node.js 18+"
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.8+"
        exit 1
    fi
    
    # Check Modal CLI
    if ! command -v modal &> /dev/null; then
        print_warning "Modal CLI not found. Please install it: pip install modal"
        print_status "You can continue without Modal, but kernel execution won't work"
    fi
    
    print_success "Requirements check completed"
}

# Setup backend
setup_backend() {
    print_status "Setting up backend..."
    
    cd Backend
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_status "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install requirements
    print_status "Installing Python dependencies..."
    pip install -r requirements.txt
    
    print_success "Backend setup completed"
    cd ..
}

# Setup frontend
setup_frontend() {
    print_status "Setting up frontend..."
    
    cd FrontEnd/my-app
    
    # Install dependencies
    print_status "Installing Node.js dependencies..."
    npm install
    
    print_success "Frontend setup completed"
    cd ../..
}

# Start backend in background
start_backend() {
    print_status "Starting backend server..."
    
    cd Backend
    source venv/bin/activate
    
    # Start backend in background
    python main.py &
    BACKEND_PID=$!
    
    # Wait a moment for backend to start
    sleep 3
    
    # Check if backend is running
    if ps -p $BACKEND_PID > /dev/null; then
        print_success "Backend server started (PID: $BACKEND_PID)"
        print_status "Backend API available at: http://localhost:8000"
        print_status "API documentation at: http://localhost:8000/docs"
    else
        print_error "Failed to start backend server"
        exit 1
    fi
    
    cd ..
}

# Start frontend
start_frontend() {
    print_status "Starting frontend server..."
    
    cd FrontEnd/my-app
    
    # Start frontend
    npm run dev &
    FRONTEND_PID=$!
    
    # Wait a moment for frontend to start
    sleep 5
    
    # Check if frontend is running
    if ps -p $FRONTEND_PID > /dev/null; then
        print_success "Frontend server started (PID: $FRONTEND_PID)"
        print_status "Frontend available at: http://localhost:3000"
    else
        print_error "Failed to start frontend server"
        exit 1
    fi
    
    cd ../..
}

# Cleanup function
cleanup() {
    print_status "Shutting down servers..."
    
    # Kill backend if running
    if [ ! -z "$BACKEND_PID" ] && ps -p $BACKEND_PID > /dev/null; then
        kill $BACKEND_PID
        print_success "Backend server stopped"
    fi
    
    # Kill frontend if running
    if [ ! -z "$FRONTEND_PID" ] && ps -p $FRONTEND_PID > /dev/null; then
        kill $FRONTEND_PID
        print_success "Frontend server stopped"
    fi
    
    print_success "Cleanup completed"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Main execution
main() {
    echo "=================================================="
    echo "🔬 Notebook Agent - Full Stack Application"
    echo "=================================================="
    echo ""
    
    # Check requirements
    check_requirements
    
    # Setup both backend and frontend
    setup_backend
    setup_frontend
    
    # Start servers
    start_backend
    start_frontend
    
    echo ""
    echo "=================================================="
    print_success "🚀 Notebook Agent is now running!"
    echo "=================================================="
    echo ""
    print_status "Frontend: http://localhost:3000"
    print_status "Backend API: http://localhost:8000"
    print_status "API Docs: http://localhost:8000/docs"
    echo ""
    print_status "Press Ctrl+C to stop all servers"
    echo ""
    
    # Wait for user to stop
    wait
}

# Run main function
main
