#!/usr/bin/env python3
"""
Setup script for Yoga PDF Extractor Backend
This script installs all required dependencies and sets up the environment
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install required Python packages"""
    print("\n📦 Installing Python dependencies...")
    
    # Upgrade pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing requirements"):
        return False
    
    return True

def check_tesseract():
    """Check if Tesseract OCR is installed"""
    print("\n🔍 Checking Tesseract OCR installation...")
    
    try:
        result = subprocess.run("tesseract --version", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Tesseract OCR is installed")
            return True
        else:
            print("❌ Tesseract OCR is not installed")
            print("Please install Tesseract OCR:")
            print("- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
            print("- macOS: brew install tesseract")
            print("- Ubuntu: sudo apt-get install tesseract-ocr")
            return False
    except FileNotFoundError:
        print("❌ Tesseract OCR is not installed")
        print("Please install Tesseract OCR:")
        print("- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        print("- macOS: brew install tesseract")
        print("- Ubuntu: sudo apt-get install tesseract-ocr")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = ['uploads', 'temp', 'logs']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"✅ Directory already exists: {directory}")
    
    return True

def main():
    """Main setup function"""
    print("🚀 Yoga PDF Extractor Backend Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed during dependency installation")
        sys.exit(1)
    
    # Check Tesseract
    tesseract_ok = check_tesseract()
    if not tesseract_ok:
        print("\n⚠️  Warning: Tesseract OCR is not installed")
        print("OCR functionality will not work without Tesseract")
        print("You can still use PyMuPDF and PDFMiner for text extraction")
    
    # Create directories
    if not create_directories():
        print("\n❌ Setup failed during directory creation")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run the server: python run.py")
    print("2. Open your browser to: http://localhost:8000")
    print("3. View API docs: http://localhost:8000/docs")
    print("\n🐍 Python Backend is ready for PDF extraction!")

if __name__ == "__main__":
    main()
