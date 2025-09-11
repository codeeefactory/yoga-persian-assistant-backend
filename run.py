#!/usr/bin/env python3
"""
Yoga PDF Extractor Backend Server
Run this script to start the FastAPI server
"""

import uvicorn
import sys
import os

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("🚀 Starting Yoga PDF Extractor Backend...")
    print("📄 PDF Processing with PyMuPDF, PDFMiner, and OCR")
    print("🌐 Server will be available at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("=" * 50)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )
