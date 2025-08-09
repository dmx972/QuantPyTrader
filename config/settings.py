"""
QuantPyTrader Application Settings
Configuration management using Pydantic Settings
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    app_name: str = "QuantPyTrader"
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    secret_key: str = Field(default="dev-secret-key", env="SECRET_KEY")
    
    # Database
    database_url: str = Field(default="sqlite:///./quantpytrader.db", env="DATABASE_URL")
    
    # Redis
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # Market Data APIs
    alpha_vantage_api_key: Optional[str] = Field(default=None, env="ALPHA_VANTAGE_API_KEY")
    polygon_api_key: Optional[str] = Field(default=None, env="POLYGON_API_KEY")
    fred_api_key: Optional[str] = Field(default=None, env="FRED_API_KEY")
    
    # Broker APIs
    alpaca_api_key: Optional[str] = Field(default=None, env="ALPACA_API_KEY")
    alpaca_secret_key: Optional[str] = Field(default=None, env="ALPACA_SECRET_KEY")
    
    # AI/ML APIs
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    google_api_key: Optional[str] = Field(default=None, env="GOOGLE_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    
    # News APIs
    news_api_key: Optional[str] = Field(default=None, env="NEWS_API_KEY")
    reddit_client_id: Optional[str] = Field(default=None, env="REDDIT_CLIENT_ID")
    reddit_client_secret: Optional[str] = Field(default=None, env="REDDIT_CLIENT_SECRET")
    
    # Trading Configuration
    paper_trading: bool = Field(default=True, env="PAPER_TRADING")
    default_risk_tolerance: float = Field(default=0.02, env="DEFAULT_RISK_TOLERANCE")
    max_position_size: float = Field(default=0.1, env="MAX_POSITION_SIZE")
    
    # BE-EMA-MMCUKF Parameters
    kalman_alpha: float = Field(default=0.001, env="KALMAN_ALPHA")
    kalman_beta: float = Field(default=2.0, env="KALMAN_BETA")
    kalman_kappa: float = Field(default=0.0, env="KALMAN_KAPPA")
    regime_count: int = Field(default=6, env="REGIME_COUNT")
    missing_data_threshold: float = Field(default=0.20, env="MISSING_DATA_THRESHOLD")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()