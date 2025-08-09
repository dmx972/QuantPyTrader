"""
QuantPyTrader - Main FastAPI Application
Entry point for the quantitative trading platform
"""

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from config.settings import settings
from config.database import init_db
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="Open-Source Quantitative Trading Platform with BE-EMA-MMCUKF Strategy",
    version="1.0.0",
    debug=settings.debug
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    # Initialize database
    init_db()
    print(f" {settings.app_name} started successfully!")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": "1.0.0"
    }


@app.get("/info")
async def app_info():
    """Application information"""
    return {
        "name": settings.app_name,
        "version": "1.0.0",
        "description": "Open-Source Quantitative Trading Platform",
        "features": [
            "BE-EMA-MMCUKF Strategy",
            "Multi-source data pipeline",
            "Regime-aware backtesting",
            "Real-time dashboard",
            "Risk management"
        ]
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )