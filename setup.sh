#!/bin/bash
# Setup script for Seer

echo "🔮 Setting up Seer - Event-Driven Multi-Agent System"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Check if in venv
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo ""
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
    echo ""
    echo "Activate it with:"
    echo "  source venv/bin/activate"
    echo ""
    read -p "Press enter to continue..."
    source venv/bin/activate
fi

echo ""
echo "📥 Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo ""
echo "📁 Creating directories..."
mkdir -p logs

echo ""
echo "📝 Setting up .env file..."
if [ ! -f .env ]; then
    echo "OPENAI_API_KEY=your_key_here" > .env
    echo "✅ Created .env file"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env and add your OPENAI_API_KEY"
else
    echo "✅ .env file already exists"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "=" * 60
echo "Next steps:"
echo ""
echo "1. Edit .env and add your OPENAI_API_KEY:"
echo "   nano .env"
echo ""
echo "2. Start your target agent (the one you want to evaluate):"
echo "   cd /path/to/your/agent && langgraph dev"
echo ""
echo "3. Launch Seer (Hybrid Architecture):"
echo "   python run_hybrid.py"
echo ""
echo "4. Open UI in browser:"
echo "   http://localhost:8501"
echo ""
echo "=" * 60
echo ""
echo "🔮 Ready to evaluate agents!"

