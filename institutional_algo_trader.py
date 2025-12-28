"""
INSTITUTIONAL AI ALGORITHMIC TRADING BOT
Production Ready - Zero Dependency Issues
Complete KiteConnect Integration
"""

import sys
import os
import warnings
import json
import threading
import time
import logging
from datetime import datetime, time as dt_time, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from collections import deque
import hashlib

# Core libraries - These are standard and always available
import pandas as pd
import numpy as np

# Scipy - handle gracefully
try:
    from scipy import stats
    from scipy.signal import argrelextrema
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("ℹ️ scipy not available - some advanced features disabled")

# Machine Learning - Optional
ML_AVAILABLE = False
try:
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.preprocessing import RobustScaler
    from sklearn.linear_model import LogisticRegression
    ML_AVAILABLE = True
except ImportError:
    print("ℹ️ scikit-learn not available - using basic models")

# Visualization
try:
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("ℹ️ plotly not available - charts disabled")

# Streamlit
import streamlit as st

# Database
import sqlite3

# KiteConnect - Optional
KITE_AVAILABLE = False
try:
    from kiteconnect import KiteConnect, KiteTicker
    KITE_AVAILABLE = True
    print("✅ KiteConnect available")
except ImportError:
    print("ℹ️ KiteConnect not installed. Run: pip install kiteconnect")

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STOCK UNIVERSE
# ============================================================================

class StockUniverse:
    """Stock Universe for Trading"""
    
    @staticmethod
    def get_trading_universe():
        return [
            'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR', 
            'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK', 'BAJFINANCE', 'WIPRO',
            'AXISBANK', 'LT', 'HCLTECH', 'ASIANPAINT', 'MARUTI', 'SUNPHARMA',
            'TITAN', 'ULTRACEMCO', 'TATAMOTORS', 'NTPC', 'ONGC', 'POWERGRID',
            'NESTLEIND', 'TATASTEEL', 'JSWSTEEL', 'ADANIPORTS', 'TECHM',
            'BAJAJFINSV', 'BRITANNIA', 'GRASIM', 'DIVISLAB', 'DRREDDY',
            'SHREECEM', 'HDFCLIFE', 'SBILIFE', 'BPCL', 'IOC', 'COALINDIA'
        ]
    
    @staticmethod
    def get_nifty_50():
        return [
            'ADANIPORTS', 'APOLLOHOSP', 'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO',
            'BAJAJFINSV', 'BAJFINANCE', 'BHARTIARTL', 'BPCL', 'BRITANNIA',
            'CIPLA', 'COALINDIA', 'DIVISLAB', 'DRREDDY', 'EICHERMOT',
            'GRASIM', 'HCLTECH', 'HDFCBANK', 'HDFCLIFE', 'HEROMOTOCO',
            'HINDUNILVR', 'ICICIBANK', 'INDUSINDBK', 'INFY',
            'ITC', 'JSWSTEEL', 'KOTAKBANK', 'LT', 'M&M', 'MARUTI',
            'NESTLEIND', 'NTPC', 'ONGC', 'POWERGRID', 'RELIANCE', 'SBILIFE',
            'SBIN', 'SHREECEM', 'SUNPHARMA', 'TATACONSUM', 'TATAMOTORS',
            'TATASTEEL', 'TCS', 'TECHM', 'TITAN', 'ULTRACEMCO', 'WIPRO'
        ]
    
    @staticmethod
    def get_bank_nifty():
        return ['HDFCBANK', 'ICICIBANK', 'AXISBANK', 'KOTAKBANK', 'SBIN', 'INDUSINDBK']

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
    """Trading Configuration"""
    demo_mode: bool = True
    paper_trading: bool = True
    live_trading: bool = False
    
    total_capital: float = 2_000_000.0
    risk_per_trade: float = 0.01
    max_portfolio_risk: float = 0.05
    max_positions: int = 15
    max_daily_trades: int = 50
    
    market_open: dt_time = dt_time(9, 15)
    market_close: dt_time = dt_time(15, 30)
    
    min_confidence: float = 0.65
    lookback_period: int = 100
    prediction_horizon: int = 5
    
    atr_multiplier: float = 1.5
    take_profit_ratio: float = 2.0
    trailing_stop_enabled: bool = True
    
    use_smc: bool = True
    detect_market_structure: bool = True
    use_order_blocks: bool = True
    
    update_frequency: int = 10
    data_resolution: str = "15min"

# ============================================================================
# DATABASE MANAGER
# ============================================================================

class DatabaseManager:
    """Thread-safe Database Manager"""
    
    def __init__(self, db_path: str = "trading_bot.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.init_database()
    
    def init_database(self):
        """Initialize database"""
        with self.lock:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT UNIQUE,
                    symbol TEXT,
                    direction TEXT,
                    entry_time DATETIME,
                    entry_price REAL,
                    exit_time DATETIME,
                    exit_price REAL,
                    quantity INTEGER,
                    stop_loss REAL,
                    take_profit REAL,
                    pnl REAL,
                    pnl_percentage REAL,
                    status TEXT,
                    confidence REAL,
                    strategy TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    position_id TEXT UNIQUE,
                    symbol TEXT,
                    direction TEXT,
                    entry_time DATETIME,
                    entry_price REAL,
                    quantity INTEGER,
                    stop_loss REAL,
                    take_profit REAL,
                    status TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
    
    def save_trade(self, trade_data: dict):
        """Save trade"""
        with self.lock:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO trades 
                (trade_id, symbol, direction, entry_time, entry_price, exit_time, exit_price,
                 quantity, stop_loss, take_profit, pnl, pnl_percentage, status, confidence, strategy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_data.get('trade_id'),
                trade_data.get('symbol'),
                trade_data.get('direction'),
                trade_data.get('entry_time'),
                trade_data.get('entry_price'),
                trade_data.get('exit_time'),
                trade_data.get('exit_price'),
                trade_data.get('quantity'),
                trade_data.get('stop_loss'),
                trade_data.get('take_profit'),
                trade_data.get('pnl'),
                trade_data.get('pnl_percentage'),
                trade_data.get('status'),
                trade_data.get('confidence'),
                trade_data.get('strategy')
            ))
            
            conn.commit()
            conn.close()
    
    def get_trade_history(self, limit: int = 100):
        """Get trade history"""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                df = pd.read_sql_query(f'SELECT * FROM trades ORDER BY entry_time DESC LIMIT {limit}', conn)
                conn.close()
                return df
            except:
                return pd.DataFrame()

# ============================================================================
# KITE CONNECT BROKER
# ============================================================================

class KiteBroker:
    """KiteConnect Broker Integration"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.kite = None
        self.ticker = None
        self.connected = False
        self.instruments_cache = {}
        self.price_cache = {}
        self.access_token = None
        self.api_key = None
        
        if KITE_AVAILABLE and not config.demo_mode:
            self.initialize()
    
    def initialize(self):
        """Initialize KiteConnect"""
        try:
            # Get API credentials
            self.api_key = self.get_credential("KITE_API_KEY")
            self.access_token = self.get_credential("KITE_ACCESS_TOKEN")
            
            if not self.api_key:
                print("⚠️ KITE_API_KEY not found. Set in Streamlit secrets or environment variable.")
                print("   Streamlit secrets: .streamlit/secrets.toml")
                print("   [secrets]")
                print("   KITE_API_KEY = 'your_api_key'")
                print("   KITE_ACCESS_TOKEN = 'your_access_token'")
                return False
            
            if not self.access_token:
                print("⚠️ KITE_ACCESS_TOKEN not found.")
                print(f"   Generate access token at: https://kite.zerodha.com/connect/login?api_key={self.api_key}")
                return False
            
            # Initialize KiteConnect
            self.kite = KiteConnect(api_key=self.api_key)
            self.kite.set_access_token(self.access_token)
            
            # Test connection
            profile = self.kite.profile()
            print(f"✅ Connected to Zerodha Kite - User: {profile.get('user_name', 'Unknown')}")
            
            # Load instruments
            self.load_instruments()
            
            # Setup WebSocket
            self.setup_websocket()
            
            self.connected = True
            return True
            
        except Exception as e:
            print(f"❌ KiteConnect initialization failed: {e}")
            print(f"   Error details: {str(e)}")
            self.connected = False
            return False
    
    def get_credential(self, key: str) -> str:
        """Get credential from secrets or environment"""
        value = ""
        
        # Try Streamlit secrets
        try:
            if hasattr(st, 'secrets') and key in st.secrets:
                value = st.secrets[key]
        except:
            pass
        
        # Try environment variable
        if not value:
            value = os.environ.get(key, "")
        
        return value
    
    def load_instruments(self):
        """Load instrument tokens"""
        try:
            instruments = self.kite.instruments("NSE")
            for inst in instruments:
                symbol = inst['tradingsymbol']
                self.instruments_cache[symbol] = inst['instrument_token']
            print(f"✅ Loaded {len(self.instruments_cache)} instruments")
        except Exception as e:
            print(f"⚠️ Failed to load instruments: {e}")
    
    def setup_websocket(self):
        """Setup WebSocket for live data"""
        try:
            self.ticker = KiteTicker(self.api_key, self.access_token)
            
            def on_ticks(ws, ticks):
                for tick in ticks:
                    token = tick.get('instrument_token')
                    ltp = tick.get('last_price')
                    if token and ltp:
                        self.price_cache[token] = ltp
            
            def on_connect(ws, response):
                print("✅ WebSocket connected")
                # Subscribe to top stocks
                tokens = [self.instruments_cache.get(s) for s in ['RELIANCE', 'TCS', 'HDFCBANK'] 
                         if self.instruments_cache.get(s)]
                if tokens:
                    ws.subscribe(tokens)
                    ws.set_mode(ws.MODE_LTP, tokens)
            
            def on_error(ws, code, reason):
                print(f"⚠️ WebSocket error: {code} - {reason}")
            
            def on_close(ws, code, reason):
                print(f"ℹ️ WebSocket closed: {code} - {reason}")
            
            self.ticker.on_ticks = on_ticks
            self.ticker.on_connect = on_connect
            self.ticker.on_error = on_error
            self.ticker.on_close = on_close
            
            # Start WebSocket in background
            threading.Thread(target=self.ticker.connect, daemon=True).start()
            
        except Exception as e:
            print(f"⚠️ WebSocket setup failed: {e}")
    
    def get_ltp(self, symbol: str) -> float:
        """Get Last Traded Price"""
        if self.connected and self.kite:
            try:
                # Try cache first
                token = self.instruments_cache.get(symbol)
                if token and token in self.price_cache:
                    return self.price_cache[token]
                
                # Fetch from API
                ltp_data = self.kite.ltp([f"NSE:{symbol}"])
                return ltp_data[f"NSE:{symbol}"]["last_price"]
            except Exception as e:
                print(f"⚠️ Error fetching LTP for {symbol}: {e}")
        
        # Fallback to synthetic price
        return self.get_synthetic_price(symbol)
    
    def get_historical_data(self, symbol: str, interval: str, 
                           from_date: datetime, to_date: datetime) -> pd.DataFrame:
        """Get historical OHLCV data"""
        
        if self.connected and self.kite:
            try:
                # Map interval
                kite_interval_map = {
                    "5min": "5minute",
                    "15min": "15minute",
                    "1hour": "60minute",
                    "1day": "day"
                }
                kite_interval = kite_interval_map.get(interval, "15minute")
                
                # Get instrument token
                token = self.instruments_cache.get(symbol)
                
                if not token:
                    print(f"⚠️ Instrument token not found for {symbol}")
                    return self.generate_synthetic_data(symbol, interval, from_date, to_date)
                
                # Fetch data
                data = self.kite.historical_data(
                    instrument_token=token,
                    from_date=from_date,
                    to_date=to_date,
                    interval=kite_interval
                )
                
                if data:
                    df = pd.DataFrame(data)
                    df['timestamp'] = pd.to_datetime(df['date'])
                    df = df.rename(columns={
                        'open': 'Open',
                        'high': 'High',
                        'low': 'Low',
                        'close': 'Close',
                        'volume': 'Volume'
                    })
                    df = df.set_index('timestamp')
                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                    return df
                
            except Exception as e:
                print(f"⚠️ Error fetching historical data for {symbol}: {e}")
        
        # Fallback to synthetic data
        return self.generate_synthetic_data(symbol, interval, from_date, to_date)
    
    def place_order(self, symbol: str, direction: str, quantity: int,
                   order_type: str = "MARKET", price: float = None) -> dict:
        """Place order via KiteConnect"""
        
        # Paper trading simulation
        if self.config.paper_trading or self.config.demo_mode:
            return self.simulate_order(symbol, direction, quantity, order_type, price)
        
        # Live trading
        if not self.connected:
            return {"status": "error", "message": "Not connected to Kite"}
        
        try:
            transaction_type = self.kite.TRANSACTION_TYPE_BUY if direction == "LONG" else self.kite.TRANSACTION_TYPE_SELL
            
            order_params = {
                "tradingsymbol": symbol,
                "exchange": self.kite.EXCHANGE_NSE,
                "transaction_type": transaction_type,
                "quantity": quantity,
                "order_type": self.kite.ORDER_TYPE_MARKET if order_type == "MARKET" else self.kite.ORDER_TYPE_LIMIT,
                "product": self.kite.PRODUCT_MIS,
                "validity": self.kite.VALIDITY_DAY
            }
            
            if price and order_type == "LIMIT":
                order_params["price"] = price
            
            order_id = self.kite.place_order(variety=self.kite.VARIETY_REGULAR, **order_params)
            
            return {
                "status": "success",
                "order_id": order_id,
                "message": f"Order placed: {direction} {quantity} {symbol}",
                "executed_price": price or self.get_ltp(symbol)
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Order failed: {str(e)}"}
    
    def simulate_order(self, symbol: str, direction: str, quantity: int,
                      order_type: str, price: float) -> dict:
        """Simulate paper trade"""
        import random
        order_id = f"PAPER_{int(time.time())}_{random.randint(1000, 9999)}"
        exec_price = price or self.get_ltp(symbol)
        
        return {
            "status": "success",
            "order_id": order_id,
            "message": f"📝 Paper Trade: {direction} {quantity} {symbol} @ ₹{exec_price:.2f}",
            "executed_price": exec_price,
            "paper_trade": True
        }
    
    def get_synthetic_price(self, symbol: str) -> float:
        """Generate deterministic synthetic price"""
        seed = abs(hash(symbol)) % 10000
        return 1000 + (seed / 10)
    
    def generate_synthetic_data(self, symbol: str, interval: str,
                               from_date: datetime, to_date: datetime) -> pd.DataFrame:
        """Generate synthetic OHLCV data"""
        
        freq_map = {"5min": "5min", "15min": "15min", "1hour": "1H", "1day": "1D"}
        freq = freq_map.get(interval, "15min")
        
        dates = pd.date_range(start=from_date, end=to_date, freq=freq)
        if len(dates) == 0:
            dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                                 end=datetime.now(), freq=freq)
        
        n = len(dates)
        seed = abs(hash(symbol)) % 10000
        np.random.seed(seed)
        
        base_price = 1000 + (seed % 5000)
        returns = np.random.normal(0.0001, 0.015, n)
        prices = base_price * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame(index=dates)
        df['Close'] = prices
        df['Open'] = df['Close'].shift(1).fillna(df['Close'][0]) * (1 + np.random.normal(0, 0.002, n))
        df['High'] = df[['Open', 'Close']].max(axis=1) * (1 + abs(np.random.normal(0, 0.005, n)))
        df['Low'] = df[['Open', 'Close']].min(axis=1) * (1 - abs(np.random.normal(0, 0.005, n)))
        df['Volume'] = np.random.lognormal(10, 1, n).astype(int)
        
        return df

# ============================================================================
# SMC PRO ANALYZER
# ============================================================================

class SMCProAnalyzer:
    """Smart Money Concepts Analyzer"""
    
    def detect_order_blocks(self, df: pd.DataFrame, lookback: int = 20) -> list:
        """Detect order blocks"""
        if len(df) < lookback:
            return []
        
        order_blocks = []
        
        for i in range(lookback, len(df)):
            # Bullish Order Block
            if (df['Close'].iloc[i] > df['Open'].iloc[i] and
                df['Close'].iloc[i] > df['High'].iloc[i-lookback:i].max()):
                
                order_blocks.append({
                    'type': 'BULLISH',
                    'timestamp': df.index[i],
                    'price': df['Close'].iloc[i],
                    'high': df['High'].iloc[i],
                    'low': df['Low'].iloc[i]
                })
            
            # Bearish Order Block
            elif (df['Close'].iloc[i] < df['Open'].iloc[i] and
                  df['Close'].iloc[i] < df['Low'].iloc[i-lookback:i].min()):
                
                order_blocks.append({
                    'type': 'BEARISH',
                    'timestamp': df.index[i],
                    'price': df['Close'].iloc[i],
                    'high': df['High'].iloc[i],
                    'low': df['Low'].iloc[i]
                })
        
        return order_blocks[-10:]  # Return last 10
    
    def detect_fair_value_gaps(self, df: pd.DataFrame) -> list:
        """Detect Fair Value Gaps"""
        fvgs = []
        
        for i in range(1, len(df)-1):
            curr = df.iloc[i]
            prev = df.iloc[i-1]
            next_candle = df.iloc[i+1]
            
            # Bullish FVG
            if curr['Low'] > prev['High']:
                fvgs.append({
                    'type': 'BULLISH',
                    'timestamp': df.index[i],
                    'top': curr['Low'],
                    'bottom': prev['High']
                })
            
            # Bearish FVG
            elif curr['High'] < prev['Low']:
                fvgs.append({
                    'type': 'BEARISH',
                    'timestamp': df.index[i],
                    'top': prev['Low'],
                    'bottom': curr['High']
                })
        
        return fvgs[-10:]  # Return last 10
    
    def determine_market_phase(self, df: pd.DataFrame) -> str:
        """Determine market phase"""
        if len(df) < 50:
            return "UNKNOWN"
        
        recent = df.iloc[-20:]
        older = df.iloc[-50:-20]
        
        recent_high = recent['High'].max()
        recent_low = recent['Low'].min()
        older_high = older['High'].max()
        older_low = older['Low'].min()
        
        if recent_high > older_high and recent_low > older_low:
            return "UPTREND"
        elif recent_high < older_high and recent_low < older_low:
            return "DOWNTREND"
        else:
            return "CONSOLIDATION"
    
    def analyze(self, df: pd.DataFrame) -> dict:
        """Full SMC analysis"""
        return {
            'order_blocks': self.detect_order_blocks(df),
            'fair_value_gaps': self.detect_fair_value_gaps(df),
            'market_phase': self.determine_market_phase(df)
        }

# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================

class TechnicalIndicators:
    """Technical Indicators"""
    
    @staticmethod
    def calculate_atr(high, low, close, period=14):
        """ATR"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    @staticmethod
    def calculate_rsi(series, period=14):
        """RSI"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_macd(series, fast=12, slow=26, signal=9):
        """MACD"""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line, macd - signal_line

# ============================================================================
# AI MODELS
# ============================================================================

class AITradingModels:
    """AI Trading Models"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.ml_available = ML_AVAILABLE
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features"""
        feat = df.copy()
        
        # Price features
        feat['returns'] = df['Close'].pct_change()
        feat['high_low_ratio'] = df['High'] / df['Low']
        
        # Technical
        feat['rsi'] = TechnicalIndicators.calculate_rsi(df['Close'])
        feat['atr'] = TechnicalIndicators.calculate_atr(df['High'], df['Low'], df['Close'])
        
        # Moving averages
        for p in [5, 10, 20]:
            feat[f'sma_{p}'] = df['Close'].rolling(p).mean()
        
        # Clean
        feat = feat.replace([np.inf, -np.inf], np.nan)
        feat = feat.fillna(method='ffill').fillna(method='bfill') if hasattr(feat, 'fillna') else feat.ffill().bfill()
        
        return feat
    
    def train_model(self, df: pd.DataFrame, symbol: str):
        """Train model"""
        if not self.ml_available:
            return None
        
        try:
            feat = self.create_features(df)
            
            # Target
            future_ret = df['Close'].shift(-5) / df['Close'] - 1
            y = pd.cut(future_ret, bins=[-np.inf, -0.01, 0.01, np.inf], labels=[-1, 0, 1])
            
            # Valid data
            valid = ~feat.isna().any(axis=1) & ~y.isna()
            X = feat.loc[valid, ['returns', 'high_low_ratio', 'rsi', 'atr', 'sma_5', 'sma_10', 'sma_20']]
            y = y.loc[valid]
            
            if len(X) < 50:
                return None
            
            # Train
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_scaled, y)
            
            self.models[symbol] = model
            self.scalers[symbol] = scaler
            
            print(f"✅ Model trained for {symbol}")
            return model
            
        except Exception as e:
            print(f"⚠️ Training failed for {symbol}: {e}")
            return None
    
    def predict(self, df: pd.DataFrame, symbol: str) -> Tuple[int, float]:
        """Predict"""
        if symbol not in self.models:
            return 0, 0.0
        
        try:
            feat = self.create_features(df)
            latest = feat[['returns', 'high_low_ratio', 'rsi', 'atr', 'sma_5', 'sma_10', 'sma_20']].iloc[-1:].values
            
            scaled = self.scalers[symbol].transform(latest)
            pred = self.models[symbol].predict(scaled)[0]
            proba = self.models[symbol].predict_proba(scaled)[0]
            conf = np.max(proba)
            
            return pred, conf
        except:
            return 0, 0.0

# ============================================================================
# RISK MANAGER
# ============================================================================

class RiskManager:
    """Risk Management"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.positions = {}
        self.daily_stats = {
            'trades_today': 0,
            'pnl_today': 0.0,
            'winning_trades':
