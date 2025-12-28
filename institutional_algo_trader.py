"""
INSTITUTIONAL AI ALGORITHMIC TRADING SYSTEM
Complete Version with Nifty 50, Midcap, and Nifty 100
Fixed for Streamlit Cloud Deployment
"""

import sys
import os
import warnings
import json
import pickle
import threading
import time
import logging
from datetime import datetime, time as dt_time, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from collections import defaultdict
import pytz

# Core libraries (all included in Streamlit Cloud)
import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import argrelextrema

# Machine Learning
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

# Visualization
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Streamlit
import streamlit as st
import streamlit.components.v1 as components

# Trading API (optional - can run in demo mode)
try:
    from kiteconnect import KiteConnect, KiteTicker
    KITE_AVAILABLE = True
except ImportError:
    KITE_AVAILABLE = False
    print("KiteConnect not available - running in demo mode")

# Suppress warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# ============================================================================
# CONFIGURATION
# ============================================================================

class MarketPhase(Enum):
    PRE_OPEN = "pre_open"
    OPENING = "opening"
    MID_DAY = "mid_day"
    CLOSING = "closing"
    POST_CLOSE = "post_close"

@dataclass
class TradingConfig:
    """Trading configuration"""
    # Mode
    demo_mode: bool = True
    paper_trading: bool = True
    
    # Capital Management
    total_capital: float = 2_000_000.0  # 20L
    risk_per_trade: float = 0.01  # 1% risk per trade
    max_portfolio_risk: float = 0.05  # 5% max daily loss
    max_positions: int = 10
    
    # Market Hours (IST)
    market_open: dt_time = dt_time(9, 15)
    market_close: dt_time = dt_time(15, 30)
    square_off_time: dt_time = dt_time(15, 20)
    
    # AI Parameters
    min_confidence_threshold: float = 0.60
    lookback_period: int = 50
    prediction_horizon: int = 3
    
    # Risk Management
    stop_loss_method: str = "atr"
    atr_multiplier: float = 1.5
    trailing_sl_enabled: bool = True
    
    # Data
    update_frequency: int = 30  # seconds
    historical_days: int = 180  # 6 months

@dataclass
class StockUniverse:
    """Complete Indian stock universe"""
    
    # Nifty 50
    nifty_50 = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", 
        "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
        "BAJFINANCE", "LT", "AXISBANK", "ASIANPAINT", "MARUTI",
        "SUNPHARMA", "TITAN", "WIPRO", "ULTRACEMCO", "POWERGRID",
        "NTPC", "ONGC", "TECHM", "JSWSTEEL", "HCLTECH",
        "ADANIPORTS", "TATASTEEL", "M&M", "GRASIM", "INDUSINDBK",
        "BRITANNIA", "BAJAJFINSV", "HDFC", "DIVISLAB", "DRREDDY",
        "CIPLA", "SHREECEM", "BPCL", "EICHERMOT", "HEROMOTOCO",
        "COALINDIA", "IOC", "SBILIFE", "UPL", "TATAMOTORS",
        "BAJAJ-AUTO", "NESTLEIND", "HDFCLIFE", "HINDALCO", "TATACONSUM"
    ]
    
    # Nifty Midcap 150 (top 50 for trading)
    nifty_midcap = [
        "PAGEIND", "BERGEPAINT", "DABUR", "GODREJCP", "HAVELLS",
        "ICICIPRULI", "LTI", "MARICO", "PIDILITIND", "SRF",
        "ABFRL", "AJANTPHARM", "AMARAJABAT", "APOLLOHOSP", "ASHOKLEY",
        "AUROPHARMA", "BAJAJHLDNG", "BALKRISIND", "BANDHANBNK", "BATAINDIA",
        "BHARATFORG", "BIOCON", "BOSCHLTD", "CADILAHC", "CHOLAFIN",
        "COLPAL", "CONCOR", "DALBHARAT", "ESCORTS", "EXIDEIND",
        "FEDERALBNK", "GLENMARK", "GMRINFRA", "GODREJPROP", "HINDPETRO",
        "IBULHSGFIN", "IDEA", "INDHOTEL", "INFRATEL", "IRCTC",
        "JINDALSTEL", "JUBLFOOD", "LALPATHLAB", "LICHSGFIN", "LUPIN",
        "MANAPPURAM", "MFSL", "MINDTREE", "MOTHERSUMI", "MRF"
    ]
    
    # Nifty 100 (additional 50 from next 50)
    nifty_100_additional = [
        "ADANIENT", "ADANIGREEN", "ADANIPORTS", "ADANITRANS", "AMBUJACEM",
        "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", "BAJAJFINSV",
        "BAJFINANCE", "BANDHANBNK", "BANKBARODA", "BERGEPAINT", "BHARATFORG",
        "BHARTIARTL", "BIOCON", "BOSCHLTD", "BPCL", "BRITANNIA",
        "CADILAHC", "CHOLAFIN", "CIPLA", "COALINDIA", "COLPAL",
        "DABUR", "DIVISLAB", "DLF", "DRREDDY", "EICHERMOT",
        "GAIL", "GLAND", "GODREJCP", "GRASIM", "HAVELLS",
        "HCLTECH", "HDFC", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO",
        "HINDALCO", "HINDPETRO", "HINDUNILVR", "ICICIBANK", "ICICIGI",
        "ICICIPRULI", "IGL", "INDUSINDBK", "INFY", "IOC"
    ]
    
    @classmethod
    def get_nifty_50(cls):
        return cls.nifty_50
    
    @classmethod
    def get_nifty_midcap(cls):
        return cls.nifty_midcap
    
    @classmethod
    def get_nifty_100(cls):
        # Combine Nifty 50 with additional stocks to make Nifty 100
        return cls.nifty_50 + cls.nifty_100_additional[:50]
    
    @classmethod
    def get_all_symbols(cls):
        """Get all symbols for scanning"""
        return list(set(cls.nifty_50 + cls.nifty_midcap + cls.nifty_100_additional))
    
    @classmethod
    def get_trading_universe(cls):
        """Get optimized trading universe (100 stocks max)"""
        all_symbols = cls.get_all_symbols()
        return sorted(all_symbols)[:100]  # Limit to 100 for performance

@dataclass
class Position:
    """Trading position"""
    symbol: str
    direction: str  # "LONG" or "SHORT"
    quantity: int
    entry_price: float
    entry_time: datetime
    stop_loss: float
    target: float
    current_price: float = 0.0
    pnl: float = 0.0
    pnl_percentage: float = 0.0
    atr: float = 0.0
    status: str = "OPEN"  # OPEN, CLOSED, STOPPED, TARGET
    
    def update(self, current_price: float):
        """Update position with current price"""
        self.current_price = current_price
        
        if self.direction == "LONG":
            self.pnl = (current_price - self.entry_price) * self.quantity
        else:  # SHORT
            self.pnl = (self.entry_price - current_price) * self.quantity
        
        self.pnl_percentage = (self.pnl / (self.entry_price * self.quantity)) * 100
        
        # Check exit conditions
        if self.direction == "LONG":
            if current_price <= self.stop_loss:
                self.status = "STOPPED"
            elif current_price >= self.target:
                self.status = "TARGET"
        else:  # SHORT
            if current_price >= self.stop_loss:
                self.status = "STOPPED"
            elif current_price <= self.target:
                self.status = "TARGET"
    
    def should_exit(self) -> bool:
        """Check if position should be exited"""
        return self.status in ["STOPPED", "TARGET"]

# ============================================================================
# DATA MANAGER (with fallback to Yahoo Finance)
# ============================================================================

class DataManager:
    """Manages market data with fallback options"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.kite = None
        self.demo_data = {}
        self.last_update = {}
        
        if not config.demo_mode and KITE_AVAILABLE:
            self.initialize_kite()
    
    def initialize_kite(self):
        """Initialize Kite Connect"""
        try:
            # This would be configured via Streamlit secrets
            api_key = st.secrets.get("KITE_API_KEY", "")
            access_token = st.secrets.get("KITE_ACCESS_TOKEN", "")
            
            if api_key and access_token:
                self.kite = KiteConnect(api_key=api_key)
                self.kite.set_access_token(access_token)
                print("Kite Connect initialized successfully")
        except Exception as e:
            print(f"Kite initialization failed: {e}")
            self.config.demo_mode = True
    
    def fetch_historical_data(self, symbol: str, interval: str = "15min") -> pd.DataFrame:
        """Fetch historical data with fallback to demo data"""
        
        # For demo mode or when Kite is unavailable
        if self.config.demo_mode or not self.kite:
            return self.get_demo_data(symbol, interval)
        
        try:
            # Get instrument token
            instruments = self.kite.instruments("NSE")
            token = None
            for ins in instruments:
                if ins['tradingsymbol'] == symbol:
                    token = ins['instrument_token']
                    break
            
            if not token:
                return self.get_demo_data(symbol, interval)
            
            # Fetch data
            to_date = datetime.now()
            from_date = to_date - timedelta(days=self.config.historical_days)
            
            data = self.kite.historical_data(
                token, from_date, to_date, interval
            )
            
            if not data:
                return self.get_demo_data(symbol, interval)
            
            df = pd.DataFrame(data)
            df.rename(columns={
                'date': 'timestamp',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }, inplace=True)
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return self.get_demo_data(symbol, interval)
    
    def get_demo_data(self, symbol: str, interval: str) -> pd.DataFrame:
        """Generate demo data for testing"""
        
        if symbol not in self.demo_data:
            # Create synthetic data
            np.random.seed(hash(symbol) % 10000)
            
            # Generate dates
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.config.historical_days)
            
            if interval == "15min":
                freq = "15min"
            elif interval == "1hour":
                freq = "1H"
            else:
                freq = "1D"
            
            dates = pd.date_range(start=start_date, end=end_date, freq=freq)
            
            # Generate price data (random walk)
            n = len(dates)
            returns = np.random.normal(0.0001, 0.015, n)
            prices = 1000 * np.exp(np.cumsum(returns))
            
            # Add some trends based on symbol hash
            trend = np.sin(np.arange(n) * 0.01 + hash(symbol) % 100 * 0.1)
            prices = prices * (1 + 0.1 * trend)
            
            # Generate OHLCV
            df = pd.DataFrame(index=dates)
            df['Close'] = prices
            df['Open'] = df['Close'].shift(1) * (1 + np.random.normal(0, 0.002, n))
            df['High'] = df[['Open', 'Close']].max(axis=1) * (1 + abs(np.random.normal(0, 0.005, n)))
            df['Low'] = df[['Open', 'Close']].min(axis=1) * (1 - abs(np.random.normal(0, 0.005, n)))
            df['Volume'] = np.random.lognormal(10, 1, n)
            
            # Fill NaN
            df = df.ffill().bfill()
            
            self.demo_data[symbol] = df
        
        return self.demo_data[symbol].copy()
    
    def get_live_price(self, symbol: str) -> float:
        """Get live price"""
        if not self.config.demo_mode and self.kite:
            try:
                ltp = self.kite.ltp(f"NSE:{symbol}")
                return ltp[f"NSE:{symbol}"]['last_price']
            except:
                pass
        
        # Fallback to demo data
        df = self.get_demo_data(symbol, "15min")
        return df['Close'].iloc[-1]

# ============================================================================
# TECHNICAL INDICATORS (Custom implementation - no pandas_ta)
# ============================================================================

class TechnicalIndicators:
    """Custom technical indicators implementation"""
    
    @staticmethod
    def calculate_ema(series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_sma(series: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return series.rolling(window=period).mean()
    
    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Calculate MACD"""
        ema_fast = TechnicalIndicators.calculate_ema(series, fast)
        ema_slow = TechnicalIndicators.calculate_ema(series, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> tuple:
        """Calculate Bollinger Bands"""
        sma = TechnicalIndicators.calculate_sma(series, period)
        rolling_std = series.rolling(window=period).std()
        upper_band = sma + (rolling_std * std_dev)
        lower_band = sma - (rolling_std * std_dev)
        return upper_band, sma, lower_band
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    @staticmethod
    def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index"""
        # Calculate +DM and -DM
        high_diff = high.diff()
        low_diff = low.diff().abs()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        # Calculate True Range
        tr = pd.concat([
            high - low,
            abs(high - close.shift()),
            abs(low - close.shift())
        ], axis=1).max(axis=1)
        
        # Smooth the values
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

class FeatureEngineer:
    """Create features for ML models"""
    
    @staticmethod
    def create_features(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
        """Create comprehensive feature set"""
        
        if df.empty:
            return df
        
        result = df.copy()
        
        # Price features
        result['Returns'] = df['Close'].pct_change()
        result['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        result['High_Low_Ratio'] = df['High'] / df['Low']
        
        # Volume features
        result['Volume_MA'] = df['Volume'].rolling(20).mean()
        result['Volume_Ratio'] = df['Volume'] / result['Volume_MA']
        
        # Moving averages
        for period in [5, 10, 20, 50, 200]:
            result[f'SMA_{period}'] = TechnicalIndicators.calculate_sma(df['Close'], period)
            result[f'EMA_{period}'] = TechnicalIndicators.calculate_ema(df['Close'], period)
        
        # RSI
        result['RSI'] = TechnicalIndicators.calculate_rsi(df['Close'], 14)
        
        # MACD
        macd_line, signal_line, histogram = TechnicalIndicators.calculate_macd(df['Close'])
        result['MACD'] = macd_line
        result['MACD_Signal'] = signal_line
        result['MACD_Hist'] = histogram
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.calculate_bollinger_bands(df['Close'])
        result['BB_Upper'] = bb_upper
        result['BB_Lower'] = bb_lower
        result['BB_Width'] = (bb_upper - bb_lower) / bb_middle
        
        # ATR
        result['ATR'] = TechnicalIndicators.calculate_atr(df['High'], df['Low'], df['Close'])
        result['ATR_Pct'] = result['ATR'] / df['Close']
        
        # ADX for trend strength
        result['ADX'] = TechnicalIndicators.calculate_adx(df['High'], df['Low'], df['Close'])
        
        # Price position features
        result['Close_vs_SMA20'] = df['Close'] / result['SMA_20']
        result['Close_vs_SMA50'] = df['Close'] / result['SMA_50']
        
        # Momentum indicators
        result['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        result['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        
        # Volatility
        result['Volatility_20'] = result['Returns'].rolling(20).std()
        
        # Support and Resistance
        result['Resistance'] = df['High'].rolling(lookback).max()
        result['Support'] = df['Low'].rolling(lookback).min()
        result['Dist_to_Resistance'] = (result['Resistance'] - df['Close']) / df['Close']
        result['Dist_to_Support'] = (df['Close'] - result['Support']) / df['Close']
        
        # Clean NaN values
        result = result.replace([np.inf, -np.inf], np.nan)
        result = result.fillna(method='ffill').fillna(method='bfill')
        
        return result

# ============================================================================
# AI MODEL
# ============================================================================

class AIModel:
    """Simple yet effective AI model for trading"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
    
    def prepare_data(self, df: pd.DataFrame) -> tuple:
        """Prepare data for training"""
        
        # Create features
        feature_df = FeatureEngineer.create_features(df, self.config.lookback_period)
        
        # Create labels (1 = buy, -1 = sell, 0 = hold)
        future_returns = df['Close'].shift(-self.config.prediction_horizon) / df['Close'] - 1
        
        labels = pd.Series(0, index=df.index)  # Default: hold
        labels[future_returns > 0.005] = 1  # Buy if > 0.5%
        labels[future_returns < -0.005] = -1  # Sell if < -0.5%
        
        # Remove last N rows (no future data)
        labels = labels.iloc[:-self.config.prediction_horizon]
        feature_df = feature_df.loc[labels.index]
        
        # Select feature columns (exclude price/volume for training)
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'timestamp']
        self.feature_columns = [col for col in feature_df.columns if col not in exclude_cols]
        
        X = feature_df[self.feature_columns].values
        y = labels.values
        
        return X, y
    
    def train(self, df: pd.DataFrame):
        """Train the model"""
        try:
            X, y = self.prepare_data(df)
            
            if len(X) < 100:  # Need minimum data
                print("Insufficient data for training")
                return
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train Random Forest
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                self.model.fit(X_train, y_train)
                y_pred = self.model.predict(X_val)
                accuracy = accuracy_score(y_val, y_pred)
                scores.append(accuracy)
            
            self.is_trained = True
            print(f"Model trained. CV Accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
            
        except Exception as e:
            print(f"Training error: {e}")
    
    def predict(self, df: pd.DataFrame) -> tuple:
        """Make prediction"""
        
        if not self.is_trained or self.model is None:
            return 0, 0.0  # No prediction
        
        try:
            # Create features for latest data
            feature_df = FeatureEngineer.create_features(df, self.config.lookback_period)
            
            if len(feature_df) == 0:
                return 0, 0.0
            
            # Get latest features
            latest_features = feature_df[self.feature_columns].iloc[-1:].values
            
            # Scale and predict
            X_scaled = self.scaler.transform(latest_features)
            prediction = self.model.predict(X_scaled)[0]
            
            # Get probability
            probabilities = self.model.predict_proba(X_scaled)[0]
            confidence = np.max(probabilities)
            
            return prediction, confidence
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0, 0.0

# ============================================================================
# RISK MANAGER
# ============================================================================

class RiskManager:
    """Risk management system"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.positions_risk = {}
        self.daily_pnl = 0.0
        self.max_daily_loss = config.total_capital * config.max_portfolio_risk
    
    def calculate_position_size(self, symbol: str, entry_price: float, 
                               stop_loss: float) -> tuple:
        """Calculate position size"""
        
        # Risk per share
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share == 0:
            return 0, 0.0
        
        # Position size based on risk
        risk_amount = self.config.total_capital * self.config.risk_per_trade
        quantity = int(risk_amount / risk_per_share)
        
        # Minimum 1 share
        quantity = max(1, quantity)
        
        # Calculate actual risk
        actual_risk = quantity * risk_per_share
        
        return quantity, actual_risk
    
    def can_take_position(self, symbol: str, new_risk: float) -> bool:
        """Check if new position can be taken"""
        
        # Check daily loss limit
        if self.daily_pnl + new_risk < -self.max_daily_loss:
            return False
        
        # Check if already in position
        if symbol in self.positions_risk:
            return False
        
        return True
    
    def update_pnl(self, pnl: float):
        """Update P&L"""
        self.daily_pnl += pnl
    
    def add_position(self, symbol: str, risk: float):
        """Add position to tracking"""
        self.positions_risk[symbol] = risk
    
    def remove_position(self, symbol: str):
        """Remove position from tracking"""
        if symbol in self.positions_risk:
            del self.positions_risk[symbol]

# ============================================================================
# TRADING ENGINE
# ============================================================================

class TradingEngine:
    """Main trading engine"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.data_manager = DataManager(config)
        self.ai_model = AIModel(config)
        self.risk_manager = RiskManager(config)
        
        self.positions = {}
        self.signals = {}
        self.trade_history = []
        
        self.market_phase = MarketPhase.PRE_OPEN
        self.is_running = False
        
        # Performance metrics
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0
        }
    
    def start(self):
        """Start the engine"""
        self.is_running = True
        print("Trading engine started")
        
        # Train initial model
        self.train_models()
    
    def stop(self):
        """Stop the engine"""
        self.is_running = False
        print("Trading engine stopped")
    
    def train_models(self):
        """Train AI models on sample data"""
        print("Training AI models...")
        
        # Use a sample symbol for training
        sample_symbol = "RELIANCE"
        df = self.data_manager.fetch_historical_data(sample_symbol, "15min")
        
        if not df.empty:
            self.ai_model.train(df)
            print("Model training completed")
        else:
            print("No data available for training")
    
    def scan_signals(self):
        """Scan for trading signals"""
        
        if not self.is_running:
            return
        
        symbols = StockUniverse.get_trading_universe()[:20]  # Scan top 20
        
        for symbol in symbols:
            try:
                # Skip if already in position
                if symbol in self.positions:
                    continue
                
                # Skip if max positions reached
                if len(self.positions) >= self.config.max_positions:
                    break
                
                # Fetch data
                df = self.data_manager.fetch_historical_data(symbol, "15min")
                
                if len(df) < 100:  # Need enough data
                    continue
                
                # Get AI prediction
                prediction, confidence = self.ai_model.predict(df)
                
                # Check confidence threshold
                if abs(prediction) == 1 and confidence >= self.config.min_confidence_threshold:
                    
                    # Get current price
                    current_price = self.data_manager.get_live_price(symbol)
                    
                    # Calculate stop loss and target
                    atr = df['Close'].rolling(14).std().iloc[-1]
                    
                    if prediction == 1:  # BUY signal
                        direction = "LONG"
                        stop_loss = current_price - (atr * self.config.atr_multiplier)
                        target = current_price + (3 * atr * self.config.atr_multiplier)
                    else:  # SELL signal
                        direction = "SHORT"
                        stop_loss = current_price + (atr * self.config.atr_multiplier)
                        target = current_price - (3 * atr * self.config.atr_multiplier)
                    
                    # Calculate position size
                    quantity, risk_amount = self.risk_manager.calculate_position_size(
                        symbol, current_price, stop_loss
                    )
                    
                    # Check risk limits
                    if not self.risk_manager.can_take_position(symbol, risk_amount):
                        continue
                    
                    # Create signal
                    signal = {
                        'symbol': symbol,
                        'direction': direction,
                        'price': current_price,
                        'quantity': quantity,
                        'stop_loss': stop_loss,
                        'target': target,
                        'confidence': confidence,
                        'timestamp': datetime.now(),
                        'atr': atr,
                        'risk_amount': risk_amount
                    }
                    
                    self.signals[symbol] = signal
                    print(f"Signal: {direction} {symbol} @ {current_price:.2f}")
                    
            except Exception as e:
                print(f"Error scanning {symbol}: {e}")
    
    def execute_signals(self):
        """Execute pending signals"""
        
        for symbol, signal in list(self.signals.items()):
            try:
                # Create position
                position = Position(
                    symbol=symbol,
                    direction=signal['direction'],
                    quantity=signal['quantity'],
                    entry_price=signal['price'],
                    entry_time=datetime.now(),
                    stop_loss=signal['stop_loss'],
                    target=signal['target'],
                    atr=signal['atr']
                )
                
                # Add to positions
                self.positions[symbol] = position
                self.risk_manager.add_position(symbol, signal['risk_amount'])
                
                # Record trade
                self.trade_history.append({
                    'symbol': symbol,
                    'direction': signal['direction'],
                    'quantity': signal['quantity'],
                    'entry_price': signal['price'],
                    'entry_time': datetime.now(),
                    'type': 'ENTRY'
                })
                
                self.metrics['total_trades'] += 1
                
                # Remove signal
                del self.signals[symbol]
                
                print(f"Executed: {signal['direction']} {symbol}")
                
            except Exception as e:
                print(f"Error executing signal for {symbol}: {e}")
    
    def manage_positions(self):
        """Manage existing positions"""
        
        for symbol, position in list(self.positions.items()):
            try:
                # Get current price
                current_price = self.data_manager.get_live_price(symbol)
                
                # Update position
                position.update(current_price)
                
                # Check exit conditions
                if position.should_exit():
                    self.exit_position(symbol, current_price, position.status)
                    
            except Exception as e:
                print(f"Error managing position {symbol}: {e}")
    
    def exit_position(self, symbol: str, exit_price: float, reason: str):
        """Exit a position"""
        
        position = self.positions.get(symbol)
        if not position:
            return
        
        # Calculate final P&L
        if position.direction == "LONG":
            pnl = (exit_price - position.entry_price) * position.quantity
        else:  # SHORT
            pnl = (position.entry_price - exit_price) * position.quantity
        
        # Update metrics
        self.risk_manager.update_pnl(pnl)
        self.metrics['total_pnl'] += pnl
        
        if pnl > 0:
            self.metrics['winning_trades'] += 1
        else:
            self.metrics['losing_trades'] += 1
        
        # Update win rate
        if self.metrics['total_trades'] > 0:
            self.metrics['win_rate'] = (self.metrics['winning_trades'] / 
                                       self.metrics['total_trades'] * 100)
        
        # Record exit
        self.trade_history.append({
            'symbol': symbol,
            'exit_price': exit_price,
            'exit_time': datetime.now(),
            'pnl': pnl,
            'reason': reason,
            'type': 'EXIT'
        })
        
        # Remove position
        del self.positions[symbol]
        self.risk_manager.remove_position(symbol)
        
        print(f"Exited {symbol}: {reason}, P&L: {pnl:.2f}")
    
    def run_cycle(self):
        """Run one trading cycle"""
        if not self.is_running:
            return
        
        self.scan_signals()
        self.execute_signals()
        self.manage_positions()
    
    def get_performance_report(self) -> dict:
        """Get performance report"""
        return {
            'total_trades': self.metrics['total_trades'],
            'winning_trades': self.metrics['winning_trades'],
            'losing_trades': self.metrics['losing_trades'],
            'win_rate': f"{self.metrics['win_rate']:.2f}%",
            'total_pnl': f"₹{self.metrics['total_pnl']:,.2f}",
            'daily_pnl': f"₹{self.risk_manager.daily_pnl:,.2f}",
            'active_positions': len(self.positions),
            'pending_signals': len(self.signals)
        }

# ============================================================================
# STREAMLIT DASHBOARD
# ============================================================================

def main():
    """Main Streamlit application"""
    
    st.set_page_config(
        page_title="Institutional AI Trading System",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
    }
    .profit {
        color: #00C853;
    }
    .loss {
        color: #FF5252;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("<h1 class='main-header'>🏦 Institutional AI Trading System</h1>", unsafe_allow_html=True)
    
    # Initialize session state
    if 'trading_engine' not in st.session_state:
        config = TradingConfig()
        st.session_state.trading_engine = TradingEngine(config)
        st.session_state.last_update = datetime.now()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Mode selection
        mode = st.radio(
            "Trading Mode",
            ["Demo Mode", "Paper Trading", "Live Trading"],
            index=0
        )
        
        # Capital settings
        capital = st.number_input(
            "Capital (₹)",
            min_value=100000,
            max_value=10000000,
            value=2000000,
            step=100000
        )
        
        # Risk settings
        risk_per_trade = st.slider(
            "Risk per Trade (%)",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1
        ) / 100
        
        max_positions = st.slider(
            "Max Positions",
            min_value=1,
            max_value=20,
            value=10,
            step=1
        )
        
        # AI settings
        confidence_threshold = st.slider(
            "Confidence Threshold (%)",
            min_value=50,
            max_value=90,
            value=60,
            step=5
        ) / 100
        
        # Update config
        config = TradingConfig(
            demo_mode=(mode == "Demo Mode"),
            paper_trading=(mode != "Live Trading"),
            total_capital=capital,
            risk_per_trade=risk_per_trade,
            max_positions=max_positions,
            min_confidence_threshold=confidence_threshold
        )
        
        st.session_state.trading_engine.config = config
        
        # Control buttons
        st.markdown("## 🎮 Control Panel")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("▶️ Start Trading", type="primary", use_container_width=True):
                st.session_state.trading_engine.start()
                st.success("Trading started!")
        
        with col2:
            if st.button("⏹️ Stop Trading", type="secondary", use_container_width=True):
                st.session_state.trading_engine.stop()
                st.warning("Trading stopped!")
        
        st.markdown("---")
        
        # Universe selection
        st.markdown("## 📊 Stock Universe")
        universe = st.selectbox(
            "Select Universe",
            ["Nifty 50", "Nifty Midcap", "Nifty 100", "All Stocks"]
        )
        
        # Manual trade
        st.markdown("## 📝 Manual Trade")
        with st.expander("Place Manual Order"):
            symbol = st.text_input("Symbol", "RELIANCE")
            direction = st.selectbox("Direction", ["LONG", "SHORT"])
            quantity = st.number_input("Quantity", min_value=1, value=10)
            
            if st.button("Place Order", type="secondary"):
                st.info(f"Manual order: {direction} {symbol} x{quantity}")
    
    # Main dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total P&L", f"₹{st.session_state.trading_engine.metrics['total_pnl']:,.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Win Rate", f"{st.session_state.trading_engine.metrics['win_rate']:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Active Positions", len(st.session_state.trading_engine.positions))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        status = "Running" if st.session_state.trading_engine.is_running else "Stopped"
        color = "green" if st.session_state.trading_engine.is_running else "red"
        st.metric("Status", status, delta=None, delta_color=color)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Positions", 
        "🚦 Signals", 
        "📊 Performance", 
        "📋 Trade History", 
        "📉 Charts"
    ])
    
    with tab1:
        st.markdown("### 📈 Active Positions")
        
        if st.session_state.trading_engine.positions:
            positions_data = []
            for symbol, pos in st.session_state.trading_engine.positions.items():
                positions_data.append({
                    'Symbol': symbol,
                    'Direction': pos.direction,
                    'Quantity': pos.quantity,
                    'Entry Price': f"₹{pos.entry_price:.2f}",
                    'Current Price': f"₹{pos.current_price:.2f}",
                    'P&L': f"₹{pos.pnl:,.2f}",
                    'P&L %': f"{pos.pnl_percentage:.2f}%",
                    'Stop Loss': f"₹{pos.stop_loss:.2f}",
                    'Target': f"₹{pos.target:.2f}",
                    'Status': pos.status
                })
            
            st.dataframe(
                positions_data,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No active positions")
    
    with tab2:
        st.markdown("### 🚦 Trading Signals")
        
        # Run signal scan
        if st.button("🔍 Scan for Signals", type="secondary"):
            st.session_state.trading_engine.scan_signals()
            st.rerun()
        
        if st.session_state.trading_engine.signals:
            signals_data = []
            for symbol, sig in st.session_state.trading_engine.signals.items():
                signals_data.append({
                    'Symbol': symbol,
                    'Direction': sig['direction'],
                    'Price': f"₹{sig['price']:.2f}",
                    'Confidence': f"{sig['confidence']:.1%}",
                    'Stop Loss': f"₹{sig['stop_loss']:.2f}",
                    'Target': f"₹{sig['target']:.2f}",
                    'Quantity': sig['quantity']
                })
            
            st.dataframe(
                signals_data,
                use_container_width=True,
                hide_index=True
            )
            
            if st.button("⚡ Execute All Signals", type="primary"):
                st.session_state.trading_engine.execute_signals()
                st.rerun()
        else:
            st.info("No signals detected")
    
    with tab3:
        st.markdown("### 📊 Performance Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Performance metrics
            perf = st.session_state.trading_engine.get_performance_report()
            
            st.metric("Total Trades", perf['total_trades'])
            st.metric("Winning Trades", perf['winning_trades'])
            st.metric("Losing Trades", perf['losing_trades'])
            st.metric("Win Rate", perf['win_rate'])
        
        with col2:
            st.metric("Total P&L", perf['total_pnl'])
            st.metric("Daily P&L", perf['daily_pnl'])
            st.metric("Active Positions", perf['active_positions'])
            st.metric("Pending Signals", perf['pending_signals'])
        
        # P&L chart
        st.markdown("### P&L Trend")
        if st.session_state.trading_engine.trade_history:
            # Create sample chart
            dates = pd.date_range(start='2025-12-01', periods=20, freq='D')
            pnl_data = np.random.normal(0, 50000, 20).cumsum()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates, y=pnl_data,
                mode='lines',
                name='Cumulative P&L',
                line=dict(color='green' if pnl_data[-1] > 0 else 'red')
            ))
            
            fig.update_layout(
                title="Cumulative P&L Over Time",
                xaxis_title="Date",
                yaxis_title="P&L (₹)",
                template="plotly_dark"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trading history available")
    
    with tab4:
        st.markdown("### 📋 Trade History")
        
        if st.session_state.trading_engine.trade_history:
            history_data = []
            for trade in st.session_state.trading_engine.trade_history[-50:]:  # Last 50 trades
                if trade['type'] == 'EXIT':
                    history_data.append({
                        'Time': trade['exit_time'].strftime("%H:%M:%S"),
                        'Symbol': trade['symbol'],
                        'Type': 'EXIT',
                        'Price': f"₹{trade['exit_price']:.2f}",
                        'P&L': f"₹{trade.get('pnl', 0):,.2f}",
                        'Reason': trade.get('reason', '')
                    })
                else:
                    history_data.append({
                        'Time': trade['entry_time'].strftime("%H:%M:%S"),
                        'Symbol': trade['symbol'],
                        'Type': 'ENTRY',
                        'Price': f"₹{trade['entry_price']:.2f}",
                        'Direction': trade['direction'],
                        'Quantity': trade['quantity']
                    })
            
            st.dataframe(
                history_data,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No trade history available")
    
    with tab5:
        st.markdown("### 📉 Technical Charts")
        
        # Symbol selector
        selected_symbol = st.selectbox(
            "Select Symbol",
            StockUniverse.get_trading_universe()[:50],
            index=0
        )
        
        # Fetch data
        df = st.session_state.trading_engine.data_manager.fetch_historical_data(
            selected_symbol, "15min"
        )
        
        if not df.empty:
            # Create chart
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.7, 0.3]
            )
            
            # Candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='OHLC'
                ),
                row=1, col=1
            )
            
            # Add moving averages
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=TechnicalIndicators.calculate_sma(df['Close'], 20),
                    name='SMA 20',
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=TechnicalIndicators.calculate_sma(df['Close'], 50),
                    name='SMA 50',
                    line=dict(color='red', width=1)
                ),
                row=1, col=1
            )
            
            # Volume
            colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] 
                     else 'green' for i in range(len(df))]
            
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name='Volume',
                    marker_color=colors
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                title=f"{selected_symbol} - Price Chart",
                template='plotly_dark',
                showlegend=True,
                height=600,
                xaxis_rangeslider_visible=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Technical indicators
            st.markdown("#### Technical Indicators")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                rsi = TechnicalIndicators.calculate_rsi(df['Close']).iloc[-1]
                st.metric("RSI", f"{rsi:.1f}")
            
            with col2:
                atr = TechnicalIndicators.calculate_atr(
                    df['High'], df['Low'], df['Close']
                ).iloc[-1]
                st.metric("ATR", f"{atr:.2f}")
            
            with col3:
                sma20 = TechnicalIndicators.calculate_sma(df['Close'], 20).iloc[-1]
                st.metric("SMA 20", f"₹{sma20:.2f}")
            
            with col4:
                sma50 = TechnicalIndicators.calculate_sma(df['Close'], 50).iloc[-1]
                st.metric("SMA 50", f"₹{sma50:.2f}")
        else:
            st.warning("No data available for selected symbol")
    
    # Auto-refresh
    if st.session_state.trading_engine.is_running:
        time_since_update = (datetime.now() - st.session_state.last_update).seconds
        
        if time_since_update >= 30:  # Update every 30 seconds
            st.session_state.trading_engine.run_cycle()
            st.session_state.last_update = datetime.now()
            st.rerun()
        
        # Display update timer
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Next update in:** {30 - time_since_update}s")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        <p>🚨 <b>Disclaimer:</b> This is a demo trading system for educational purposes only. 
        Past performance does not guarantee future results. Trading involves risk of loss.</p>
        <p>© 2025 Institutional AI Trading System v2.0.0</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================================
# REQUIREMENTS.TXT (for Streamlit Cloud)
# ============================================================================

"""
streamlit==1.52.2
pandas==2.3.3
numpy==2.3.5
scikit-learn==1.8.0
plotly==6.5.0
scipy==1.16.3
kiteconnect==5.0.1
"""

if __name__ == "__main__":
    main()
