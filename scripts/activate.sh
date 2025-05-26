#!/bin/bash
# ============================================================================
# YOLO Classifier Environment Activation Script (Linux/macOS)
# ============================================================================
# This script activates the Python virtual environment and provides
# helpful commands for using the YOLO classifier applications.
#
# Usage: ./activate.sh
# ============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo
echo "========================================"
echo "   YOLO Classifier Environment"
echo "========================================"
echo

# Check if virtual environment exists
if [ ! -f "yolo_env/bin/activate" ]; then
    echo -e "${RED}❌ Virtual environment not found!${NC}"
    echo
    echo "Please run setup first:"
    echo "  python setup.py"
    echo
    exit 1
fi

# Activate the virtual environment
echo -e "${BLUE}🔄 Activating YOLO Classifier environment...${NC}"
source yolo_env/bin/activate

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to activate virtual environment!${NC}"
    echo
    echo "Try running setup again:"
    echo "  python setup.py"
    echo
    exit 1
fi

echo -e "${GREEN}✅ Environment activated successfully!${NC}"
echo

# Display available commands
echo -e "${CYAN}🎯 Available Applications:${NC}"
echo
echo -e "  🐕🐱 Simple Classifier:"
echo "    python run.py --app dog_cat"
echo "    python dog_cat_yolo_gui.py"
echo
echo -e "  🎯 Advanced Classifier:"
echo "    python run.py --app generic"
echo "    python generic_yolo_classifier.py"
echo
echo -e "  🚀 Training Wizard:"
echo "    python run.py --app train"
echo "    python train_model.py --interactive"
echo
echo -e "  ⚙️  Configuration Creator:"
echo "    python run.py --app config"
echo "    python create_config.py --interactive"
echo
echo -e "  📋 Interactive Menu:"
echo "    python run.py"
echo

# Display helpful information
echo -e "${YELLOW}💡 Helpful Commands:${NC}"
echo
echo "  Check installation:     python run.py --check"
echo "  List applications:      python run.py --list"
echo "  Setup help:            python setup.py --help"
echo

# Display current environment info
echo -e "${CYAN}📊 Environment Info:${NC}"
echo "  Virtual Environment: $VIRTUAL_ENV"
echo -n "  Python Version: "
python --version 2>/dev/null || echo "Python not found in PATH"
echo "  Current Directory: $(pwd)"
echo

echo -e "${GREEN}🎉 Ready to use YOLO Classifier!${NC}"
echo "Type 'deactivate' to exit the environment."
echo

# Start a new shell with the activated environment
exec "$SHELL"
