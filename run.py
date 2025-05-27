#!/usr/bin/env python3
"""
Cross-Platform Launcher for YOLO Classifier Project
===================================================

This script provides a unified way to launch any of the YOLO classifier
applications on Windows, Linux, and macOS. It automatically handles
virtual environment activation and provides an interactive menu.

Features:
    - Cross-platform compatibility
    - Automatic environment detection and activation
    - Interactive application menu
    - Direct command-line launching
    - Environment validation

Usage:
    python run.py                      # Interactive menu
    python run.py --app dog_cat        # Launch dog/cat classifier
    python run.py --app generic        # Launch generic classifier
    python run.py --app train          # Launch training wizard
    python run.py --app config         # Launch config creator

Author: YOLO Classifier Project
License: Proprietary - Oceaneering International Inc. Use Only
"""

import sys
import subprocess
import platform
import argparse
from pathlib import Path
from typing import Optional, List

# Configuration
VENV_NAME = "yolo_env"

# Application definitions
APPLICATIONS = {
    "dog_cat": {
        "name": "Dog vs Cat Classifier",
        "script": "dog_cat_yolo_gui.py",
        "description": "Simple dog/cat classification with GUI",
        "emoji": "🐕🐱"
    },
    "generic": {
        "name": "Generic YOLO Classifier", 
        "script": "generic_yolo_classifier.py",
        "description": "Configurable classifier for any objects",
        "emoji": "🎯"
    },
    "train": {
        "name": "Model Training Wizard",
        "script": "train_model.py",
        "description": "Interactive training for custom models",
        "emoji": "🚀",
        "args": ["--interactive"]
    },
    "config": {
        "name": "Configuration Creator",
        "script": "create_config.py", 
        "description": "Create configuration files",
        "emoji": "⚙️",
        "args": ["--interactive"]
    }
}

class Colors:
    """ANSI color codes for cross-platform colored output."""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'
    
    @classmethod
    def disable_on_windows(cls):
        """Disable colors on Windows if not supported."""
        if platform.system() == "Windows":
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            except:
                for attr in dir(cls):
                    if not attr.startswith('_') and attr != 'disable_on_windows':
                        setattr(cls, attr, '')

# Initialize colors
Colors.disable_on_windows()

def print_colored(message: str, color: str = Colors.WHITE) -> None:
    """Print colored message with cross-platform support."""
    print(f"{color}{message}{Colors.END}")

def print_header(title: str) -> None:
    """Print a formatted header."""
    print_colored("\n" + "=" * 60, Colors.CYAN)
    print_colored(f"  {title}", Colors.BOLD + Colors.CYAN)
    print_colored("=" * 60, Colors.CYAN)

def get_venv_python() -> Optional[str]:
    """Get path to Python executable in virtual environment."""
    system = platform.system()
    venv_path = Path(VENV_NAME)
    
    if not venv_path.exists():
        return None
    
    if system == "Windows":
        python_path = venv_path / "Scripts" / "python.exe"
    else:
        python_path = venv_path / "bin" / "python"
    
    return str(python_path) if python_path.exists() else None

def check_environment() -> bool:
    """Check if the environment is properly set up."""
    # Check if virtual environment exists
    venv_python = get_venv_python()
    if not venv_python:
        print_colored("❌ Virtual environment not found!", Colors.RED)
        print_colored("Please run: python setup.py", Colors.YELLOW)
        return False
    
    # Check if required scripts exist
    missing_scripts = []
    for app_id, app_info in APPLICATIONS.items():
        script_path = Path(app_info["script"])
        if not script_path.exists():
            missing_scripts.append(app_info["script"])
    
    if missing_scripts:
        print_colored(f"❌ Missing scripts: {', '.join(missing_scripts)}", Colors.RED)
        return False
    
    print_colored("✅ Environment looks good!", Colors.GREEN)
    return True

def run_application(app_id: str, extra_args: List[str] = None) -> bool:
    """Run a specific application."""
    if app_id not in APPLICATIONS:
        print_colored(f"❌ Unknown application: {app_id}", Colors.RED)
        return False
    
    app_info = APPLICATIONS[app_id]
    venv_python = get_venv_python()
    
    if not venv_python:
        print_colored("❌ Virtual environment not found!", Colors.RED)
        return False
    
    # Build command
    cmd = [venv_python, app_info["script"]]
    
    # Add default args if specified
    if "args" in app_info:
        cmd.extend(app_info["args"])
    
    # Add extra args
    if extra_args:
        cmd.extend(extra_args)
    
    print_colored(f"🚀 Launching {app_info['name']}...", Colors.BLUE)
    print_colored(f"Command: {' '.join(cmd)}", Colors.WHITE)
    
    try:
        # Run the application
        result = subprocess.run(cmd)
        return result.returncode == 0
    except KeyboardInterrupt:
        print_colored("\n⚠️ Application interrupted by user", Colors.YELLOW)
        return True
    except Exception as e:
        print_colored(f"❌ Failed to run application: {e}", Colors.RED)
        return False

def show_interactive_menu() -> None:
    """Show interactive application menu."""
    print_header("YOLO Classifier Launcher")
    
    print_colored("Welcome to the YOLO Classifier Project! 🎯", Colors.CYAN)
    print_colored("Choose an application to launch:\n", Colors.WHITE)
    
    # Display applications
    for i, (app_id, app_info) in enumerate(APPLICATIONS.items(), 1):
        emoji = app_info.get("emoji", "📱")
        name = app_info["name"]
        description = app_info["description"]
        print_colored(f"{i}. {emoji} {name}", Colors.BOLD + Colors.WHITE)
        print_colored(f"   {description}\n", Colors.WHITE)
    
    print_colored("0. Exit", Colors.YELLOW)
    
    while True:
        try:
            choice = input(f"\n{Colors.CYAN}Enter your choice (0-{len(APPLICATIONS)}): {Colors.END}").strip()
            
            if choice == "0":
                print_colored("👋 Goodbye!", Colors.CYAN)
                break
            
            try:
                choice_num = int(choice)
                if 1 <= choice_num <= len(APPLICATIONS):
                    app_id = list(APPLICATIONS.keys())[choice_num - 1]
                    
                    print_colored(f"\n🎯 Selected: {APPLICATIONS[app_id]['name']}", Colors.GREEN)
                    
                    # Ask for additional arguments
                    extra_args_input = input(f"{Colors.WHITE}Additional arguments (optional): {Colors.END}").strip()
                    extra_args = extra_args_input.split() if extra_args_input else []
                    
                    success = run_application(app_id, extra_args)
                    
                    if success:
                        print_colored(f"\n✅ {APPLICATIONS[app_id]['name']} completed", Colors.GREEN)
                    else:
                        print_colored(f"\n❌ {APPLICATIONS[app_id]['name']} failed", Colors.RED)
                    
                    # Ask if user wants to continue
                    continue_choice = input(f"\n{Colors.CYAN}Launch another application? (y/N): {Colors.END}").strip().lower()
                    if continue_choice != 'y':
                        print_colored("👋 Goodbye!", Colors.CYAN)
                        break
                else:
                    print_colored("❌ Invalid choice. Please try again.", Colors.RED)
            except ValueError:
                print_colored("❌ Please enter a valid number.", Colors.RED)
                
        except KeyboardInterrupt:
            print_colored("\n\n👋 Goodbye!", Colors.CYAN)
            break

def list_applications() -> None:
    """List all available applications."""
    print_header("Available Applications")
    
    for app_id, app_info in APPLICATIONS.items():
        emoji = app_info.get("emoji", "📱")
        name = app_info["name"]
        description = app_info["description"]
        script = app_info["script"]
        
        print_colored(f"{emoji} {name} ({app_id})", Colors.BOLD + Colors.WHITE)
        print_colored(f"   Script: {script}", Colors.WHITE)
        print_colored(f"   Description: {description}\n", Colors.WHITE)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="YOLO Classifier Launcher")
    parser.add_argument("--app", choices=list(APPLICATIONS.keys()), 
                       help="Application to launch directly")
    parser.add_argument("--list", action="store_true", 
                       help="List available applications")
    parser.add_argument("--check", action="store_true",
                       help="Check environment setup")
    parser.add_argument("args", nargs="*", 
                       help="Additional arguments to pass to the application")
    
    args = parser.parse_args()
    
    try:
        # Handle list command
        if args.list:
            list_applications()
            return
        
        # Handle check command
        if args.check:
            print_header("Environment Check")
            if check_environment():
                print_colored("✅ Environment is ready!", Colors.GREEN)
            else:
                print_colored("❌ Environment needs setup", Colors.RED)
                print_colored("Run: python setup.py", Colors.YELLOW)
            return
        
        # Check environment before proceeding
        if not check_environment():
            sys.exit(1)
        
        # Handle direct app launch
        if args.app:
            success = run_application(args.app, args.args)
            sys.exit(0 if success else 1)
        
        # Show interactive menu
        show_interactive_menu()
        
    except KeyboardInterrupt:
        print_colored("\n\n👋 Goodbye!", Colors.CYAN)
        sys.exit(0)
    except Exception as e:
        print_colored(f"❌ Unexpected error: {e}", Colors.RED)
        sys.exit(1)

if __name__ == "__main__":
    main() 