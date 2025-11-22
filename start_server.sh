#!/bin/bash
# Plant Disease Detector - Startup Script

echo "=========================================="
echo "🌿 Plant Disease Detector"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.9+"
    exit 1
fi

echo "✅ Python found: $(python --version)"
echo ""

# Check if Flask is installed
python -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing Flask dependencies..."
    pip install -r web_requirements.txt
    echo ""
fi

# Check if model exists
if [ ! -f "PlantSeg/work_dirs/segnext_mscan-l_test/iter_1000.pth" ]; then
    echo "⚠️  Model checkpoint not found!"
    echo "   Location: PlantSeg/work_dirs/segnext_mscan-l_test/iter_1000.pth"
    echo ""
    echo "📚 To train the model, run:"
    echo "   cd PlantSeg"
    echo "   bash run.sh"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "🚀 Starting Plant Disease Detector..."
echo ""

# Start the Flask server
python web_interface.py

# If server exits, show message
echo ""
echo "⛔ Server stopped"
echo "To restart, run this script again"
