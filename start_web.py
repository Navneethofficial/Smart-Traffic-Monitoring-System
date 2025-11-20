#!/usr/bin/env python3
"""
Roadside Car Counter - Web Interface Launcher
Quick start script for the web application
"""

import os
import sys
import webbrowser
import time
from threading import Timer

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'flask',
        'opencv-python',
        'ultralytics',
        'numpy',
        'pillow'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                __import__('cv2')
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n💡 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_files():
    """Check if required files exist"""
    required_files = {
        'sort.py': 'SORT tracker implementation',
        'yolov8n.pt': 'YOLO model weights'
    }
    
    missing_files = []
    
    for file, description in required_files.items():
        if not os.path.exists(file):
            missing_files.append(f"{file} ({description})")
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n💡 Make sure you have:")
        print("   - sort.py in the current directory")
        print("   - yolov8n.pt model file")
        print("   - If yolov8n.pt is missing, it will be auto-downloaded on first run")
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = ['uploads', 'static/results', 'templates']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created/verified directory: {directory}")

def open_browser():
    """Open browser after a short delay"""
    time.sleep(2)
    webbrowser.open('http://localhost:5000')

def main():
    """Main entry point"""
    print("=" * 60)
    print("🚗 Roadside Car Counter - Web Interface")
    print("=" * 60)
    print()
    
    # Check dependencies
    print("📦 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    print("✓ All dependencies installed\n")
    
    # Check files
    print("📁 Checking required files...")
    if not check_files():
        print("\n⚠️  Warning: Some files are missing")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    print("✓ Required files present\n")
    
    # Create directories
    print("📂 Setting up directories...")
    create_directories()
    print()
    
    # Start web interface
    print("🚀 Starting web server...")
    print("=" * 60)
    print("📍 Web Interface: http://localhost:5000")
    print("=" * 60)
    print()
    print("Features:")
    print("  • Upload and process videos")
    print("  • Live webcam detection")
    print("  • Real-time vehicle counting")
    print("  • Adjustable confidence threshold")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    print()
    
    # Open browser automatically
    Timer(1.5, open_browser).start()
    
    # Import and run Flask app
    try:
        from web_interface import app
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down server...")
        print("Thanks for using Roadside Car Counter!")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()