#!/usr/bin/env python3
"""
Android-optimized configuration for the PDF extraction backend
"""

import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from main import app

# Android-optimized CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8081",  # Expo web
        "http://localhost:19006", # Expo web alternative
        "http://10.0.2.2:8081",  # Android emulator
        "http://10.0.2.2:19006", # Android emulator alternative
        "exp://192.168.1.100:8081", # Expo development server
        "exp://192.168.1.100:19006", # Expo development server alternative
        "*"  # Allow all origins for development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Android-optimized server configuration
if __name__ == "__main__":
    print("🚀 Starting Android-optimized PDF extraction server...")
    print("📱 Server will be accessible from Android emulator at: http://10.0.2.2:8000")
    print("🌐 Server will be accessible from web at: http://localhost:8000")
    print("📊 Ultra-enhanced extraction with machine learning enabled")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",  # Allow connections from any IP
        port=8000,
        reload=True,
        access_log=True,
        log_level="info"
    )
