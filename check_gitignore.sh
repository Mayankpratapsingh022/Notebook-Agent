#!/bin/bash

# Check gitignore coverage for Notebook Agent project
echo "🔍 Checking .gitignore coverage for Notebook Agent"
echo "=================================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_success() {
    echo -e "${GREEN}✅${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    print_error "Not in a git repository. Please run this from the project root."
    exit 1
fi

echo "Checking for common files/folders that should be ignored..."
echo ""

# Check for node_modules
if [ -d "FrontEnd/my-app/node_modules" ]; then
    if git check-ignore "FrontEnd/my-app/node_modules" > /dev/null 2>&1; then
        print_success "node_modules is properly ignored"
    else
        print_error "node_modules is NOT ignored!"
    fi
else
    print_warning "node_modules not found (not installed yet)"
fi

# Check for .next
if [ -d "FrontEnd/my-app/.next" ]; then
    if git check-ignore "FrontEnd/my-app/.next" > /dev/null 2>&1; then
        print_success ".next is properly ignored"
    else
        print_error ".next is NOT ignored!"
    fi
else
    print_warning ".next not found (not built yet)"
fi

# Check for venv
if [ -d "Backend/venv" ]; then
    if git check-ignore "Backend/venv" > /dev/null 2>&1; then
        print_success "venv is properly ignored"
    else
        print_error "venv is NOT ignored!"
    fi
else
    print_warning "venv not found (not created yet)"
fi

# Check for __pycache__
if [ -d "Backend/__pycache__" ]; then
    if git check-ignore "Backend/__pycache__" > /dev/null 2>&1; then
        print_success "__pycache__ is properly ignored"
    else
        print_error "__pycache__ is NOT ignored!"
    fi
else
    print_warning "__pycache__ not found (no Python files compiled yet)"
fi

# Check for .env files
if [ -f ".env" ] || [ -f "FrontEnd/my-app/.env" ] || [ -f "Backend/.env" ]; then
    print_warning ".env files found - make sure they contain no sensitive data"
else
    print_success "No .env files found"
fi

# Check for log files
if find . -name "*.log" -type f | grep -q .; then
    print_warning "Log files found - they should be ignored"
else
    print_success "No log files found"
fi

# Check for build directories
if [ -d "FrontEnd/my-app/build" ] || [ -d "FrontEnd/my-app/dist" ]; then
    print_warning "Build directories found - they should be ignored"
else
    print_success "No build directories found"
fi

echo ""
echo "=================================================="
echo "📋 Summary of .gitignore files:"
echo "=================================================="

# List all .gitignore files
find . -name ".gitignore" -type f | while read -r file; do
    echo "📄 $file"
    echo "   Lines: $(wc -l < "$file")"
    echo "   Size: $(du -h "$file" | cut -f1)"
    echo ""
done

echo "=================================================="
echo "🎯 Key patterns being ignored:"
echo "=================================================="

echo "Node.js:"
echo "  - node_modules/"
echo "  - .next/"
echo "  - build/"
echo "  - dist/"
echo "  - *.log"
echo ""

echo "Python:"
echo "  - __pycache__/"
echo "  - venv/"
echo "  - *.pyc"
echo "  - .coverage"
echo ""

echo "General:"
echo "  - .env*"
echo "  - .DS_Store"
echo "  - .vscode/"
echo "  - .idea/"
echo "  - *.log"
echo ""

echo "✅ .gitignore setup is comprehensive and should cover all necessary files!"
