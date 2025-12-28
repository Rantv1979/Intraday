"""
INSTITUTIONAL AI ALGORITHMIC TRADING BOT
Complete Professional Trading Terminal with SMC Pro Concepts
"""

import sys
import os
import warnings
import json
import pickle
import threading
import time
import logging
import asyncio
from datetime import datetime, time as dt_time, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
from collections import defaultdict, deque
import pytz
import hashlib

# Core libraries
import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import argrelextrema

# Machine Learning
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Visualization
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit
import streamlit as st
import streamlit.components.v1 as components

# Database
import sqlite3
import json

# Suppress warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

class MarketPhase(Enum):
    PRE_OPEN = "pre_open"
    OPENING = "opening"
    MID_DAY = "mid_day"
    CLOSING = "closing"
    POST_CLOSE = "post_close"
    AFTER_HOURS = "after_hours"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"
    SL_M = "SL-M"

class OrderStatus(Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    COMPLETE = "COMPLETE"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"

class TradeDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"

@dataclass
class TradingConfig:
    """Professional Trading Configuration"""
    # Mode Settings
    demo_mode: bool = True
    paper_trading: bool = True
    live_trading: bool = False
    
    # Capital Management
    total_capital: float = 2_000_000.0
    risk_per_trade: float = 0.01  # 1%
    max_portfolio_risk: float = 0.05  # 5%
    max_positions: int = 15
    max_daily_trades: int = 50
    
    # Market Hours (IST)
    market_open: dt_time = dt_time(9, 15)
    market_close: dt_time = dt_time(15, 30)
    pre_open_start: dt_time = dt_time(9, 0)
    post_close_end: dt_time = dt_time(16, 0)
    
    # AI & Strategy Parameters
    min_confidence: float = 0.65
    lookback_period: int = 100
    prediction_horizon: int = 5
    use_ensemble: bool = True
    
    # Risk Management
    stop_loss_method: str = "ATR"  # ATR, Percentage, SupportResistance
    atr_multiplier: float = 1.5
    take_profit_ratio: float = 2.0  # 2:1 Risk:Reward
    trailing_stop_enabled: bool = True
    trailing_stop_activation: float = 0.02  # 2%
    
    # SMC Pro Parameters
    use_smc: bool = True
    detect_market_structure: bool = True
    use_order_blocks: bool = True
    use_fair_value_gaps: bool = True
    use_ict_concepts: bool = True
    
    # Execution Parameters
    update_frequency: int = 10  # seconds
    historical_days: int = 365  # 1 year
    data_resolution: str = "5min"  # 5min, 15min, 1hour
    
    # Broker Integration
    broker_api: str = "kite"  # kite, zerodha, angel, etc.
    use_websocket: bool = True

# ============================================================================
# DATABASE MANAGER
# ============================================================================

class DatabaseManager:
    """Manages all database operations"""
    
    def __init__(self, db_path: str = "trading_bot.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with all required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Trade History
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
                remarks TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Orders
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT UNIQUE,
                symbol TEXT,
                order_type TEXT,
                transaction_type TEXT,
                quantity INTEGER,
                price REAL,
                trigger_price REAL,
                status TEXT,
                placed_time DATETIME,
                executed_time DATETIME,
                filled_quantity INTEGER,
                filled_price REAL,
                remarks TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Positions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                position_id TEXT UNIQUE,
                symbol TEXT,
                direction TEXT,
                entry_time DATETIME,
                entry_price REAL,
                quantity INTEGER,
                current_price REAL,
                stop_loss REAL,
                take_profit REAL,
                trailing_stop REAL,
                pnl REAL,
                pnl_percentage REAL,
                status TEXT,
                atr REAL,
                confidence REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Market Data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                timestamp DATETIME,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                interval TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp, interval)
            )
        ''')
        
        # Performance Metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE,
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                total_pnl REAL,
                win_rate REAL,
                profit_factor REAL,
                max_drawdown REAL,
                sharpe_ratio REAL,
                sortino_ratio REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date)
            )
        ''')
        
        # AI Model Metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                accuracy REAL,
                precision REAL,
                recall REAL,
                f1_score REAL,
                training_date DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_trade(self, trade_data: dict):
        """Save trade to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO trades 
            (trade_id, symbol, direction, entry_time, entry_price, exit_time, exit_price,
             quantity, stop_loss, take_profit, pnl, pnl_percentage, status, confidence, strategy, remarks)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            trade_data.get('strategy'),
            trade_data.get('remarks')
        ))
        
        conn.commit()
        conn.close()
    
    def get_trade_history(self, limit: int = 100):
        """Get trade history"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(f'''
            SELECT * FROM trades 
            ORDER BY entry_time DESC 
            LIMIT {limit}
        ''', conn)
        conn.close()
        return df
    
    def get_positions(self):
        """Get current positions"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('''
            SELECT * FROM positions 
            WHERE status = 'OPEN'
            ORDER BY entry_time DESC
        ''', conn)
        conn.close()
        return df

# ============================================================================
# BROKER INTEGRATION (Kite Connect)
# ============================================================================

class BrokerManager:
    """Manages broker interactions"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.kite = None
        self.ticker = None
        self.connected = False
        
        if not config.demo_mode:
            self.connect()
    
    def connect(self):
        """Connect to broker API"""
        try:
            from kiteconnect import KiteConnect, KiteTicker
            
            # Get credentials from Streamlit secrets
            api_key = st.secrets.get("KITE_API_KEY", "")
            access_token = st.secrets.get("KITE_ACCESS_TOKEN", "")
            
            if api_key and access_token:
                self.kite = KiteConnect(api_key=api_key)
                self.kite.set_access_token(access_token)
                
                # Initialize WebSocket for live data
                if self.config.use_websocket:
                    self.ticker = KiteTicker(api_key, access_token)
                    self.setup_websocket()
                
                self.connected = True
                print("✅ Successfully connected to Kite Connect")
                return True
                
        except Exception as e:
            print(f"❌ Broker connection failed: {e}")
            self.connected = False
            self.config.demo_mode = True
        
        return False
    
    def setup_websocket(self):
        """Setup WebSocket for live data"""
        if not self.ticker:
            return
        
        def on_ticks(ws, ticks):
            """Handle incoming ticks"""
            for tick in ticks:
                symbol = tick['instrument_token']
                last_price = tick['last_price']
                # Update price cache
                self.price_cache[symbol] = last_price
        
        def on_connect(ws, response):
            """Handle WebSocket connection"""
            print("✅ WebSocket connected")
            
            # Subscribe to instruments
            instruments = ["NSE:RELIANCE", "NSE:TCS", "NSE:HDFCBANK"]
            ws.subscribe(instruments)
            ws.set_mode(ws.MODE_FULL, instruments)
        
        self.ticker.on_ticks = on_ticks
        self.ticker.on_connect = on_connect
        
        # Start WebSocket in background thread
        threading.Thread(target=self.ticker.connect, daemon=True).start()
    
    def place_order(self, symbol: str, direction: str, quantity: int, 
                   order_type: str = "MARKET", price: float = None, 
                   trigger_price: float = None) -> dict:
        """Place an order"""
        
        if self.config.paper_trading:
            # Paper trading - simulate order
            return self.simulate_order(symbol, direction, quantity, order_type, price, trigger_price)
        
        if not self.connected:
            return {"status": "error", "message": "Not connected to broker"}
        
        try:
            transaction_type = "BUY" if direction == "LONG" else "SELL"
            
            order_params = {
                "tradingsymbol": symbol,
                "exchange": "NSE",
                "transaction_type": transaction_type,
                "quantity": quantity,
                "order_type": order_type,
                "product": "MIS" if self.config.data_resolution in ["5min", "15min"] else "CNC",
            }
            
            if price:
                order_params["price"] = price
            if trigger_price:
                order_params["trigger_price"] = trigger_price
            
            # Place order
            order_id = self.kite.place_order(variety="regular", **order_params)
            
            return {
                "status": "success",
                "order_id": order_id,
                "message": f"Order placed: {direction} {quantity} {symbol}"
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def simulate_order(self, symbol: str, direction: str, quantity: int,
                      order_type: str, price: float, trigger_price: float) -> dict:
        """Simulate order for paper trading"""
        import random
        
        order_id = f"PAPER_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Simulate execution
        execution_price = price or self.get_ltp(symbol)
        
        return {
            "status": "success",
            "order_id": order_id,
            "message": f"Paper Trade: {direction} {quantity} {symbol} @ {execution_price:.2f}",
            "executed_price": execution_price,
            "paper_trade": True
        }
    
    def get_ltp(self, symbol: str) -> float:
        """Get last traded price"""
        if self.connected and self.kite:
            try:
                ltp_data = self.kite.ltp([f"NSE:{symbol}"])
                return ltp_data[f"NSE:{symbol}"]["last_price"]
            except:
                pass
        
        # Fallback to random price for demo
        return 1000 + (hash(symbol) % 10000) / 100
    
    def get_historical_data(self, symbol: str, interval: str, 
                           from_date: datetime, to_date: datetime) -> pd.DataFrame:
        """Get historical data"""
        
        if self.connected and self.kite:
            try:
                # Convert interval
                kite_interval = {
                    "5min": "5minute",
                    "15min": "15minute",
                    "1hour": "60minute",
                    "1day": "day"
                }.get(interval, "15minute")
                
                # Get instrument token
                instruments = self.kite.instruments("NSE")
                token = None
                for ins in instruments:
                    if ins['tradingsymbol'] == symbol:
                        token = ins['instrument_token']
                        break
                
                if token:
                    data = self.kite.historical_data(token, from_date, to_date, kite_interval)
                    
                    if data:
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
                print(f"Error fetching historical data: {e}")
        
        # Generate synthetic data if broker fails
        return self.generate_synthetic_data(symbol, interval, from_date, to_date)
    
    def generate_synthetic_data(self, symbol: str, interval: str,
                               from_date: datetime, to_date: datetime) -> pd.DataFrame:
        """Generate synthetic market data"""
        
        # Create date range
        if interval == "5min":
            freq = "5min"
        elif interval == "15min":
            freq = "15min"
        elif interval == "1hour":
            freq = "1H"
        else:
            freq = "1D"
        
        dates = pd.date_range(start=from_date, end=to_date, freq=freq)
        
        if len(dates) == 0:
            dates = pd.date_range(start=datetime.now() - timedelta(days=30),
                                 end=datetime.now(), freq=freq)
        
        # Generate synthetic price data
        n = len(dates)
        np.random.seed(hash(symbol) % 10000)
        
        # Base price based on symbol
        base_price = 1000 + (hash(symbol) % 5000)
        
        # Random walk with drift
        returns = np.random.normal(0.0001, 0.015, n)
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Add trend and seasonality
        trend = np.sin(np.arange(n) * 0.01) * 0.1
        prices = prices * (1 + trend)
        
        # Generate OHLCV
        df = pd.DataFrame(index=dates)
        df['Close'] = prices
        
        # Calculate OHLC with some randomness
        df['Open'] = df['Close'].shift(1) * (1 + np.random.normal(0, 0.002, n))
        df['High'] = df[['Open', 'Close']].max(axis=1) * (1 + abs(np.random.normal(0, 0.005, n)))
        df['Low'] = df[['Open', 'Close']].min(axis=1) * (1 - abs(np.random.normal(0, 0.005, n)))
        df['Volume'] = np.random.lognormal(10, 1, n)
        
        # Fill NaN values
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df

# ============================================================================
# SMC PRO (SMART MONEY CONCEPTS)
# ============================================================================

class SMCProAnalyzer:
    """Smart Money Concepts Professional Analyzer"""
    
    def __init__(self):
        self.order_blocks = {}
        self.fair_value_gaps = {}
        self.liquidity_zones = {}
        
    def detect_order_blocks(self, df: pd.DataFrame, lookback: int = 20) -> dict:
        """Detect order blocks (SMC Concept)"""
        if len(df) < lookback:
            return {}
        
        order_blocks = []
        
        for i in range(lookback, len(df)):
            # Bullish Order Block (Supply becomes Demand)
            if (df['Close'].iloc[i] > df['Open'].iloc[i] and  # Bullish candle
                df['Close'].iloc[i] > df['High'].iloc[i-lookback:i].max()):  # Breaks previous high
                
                ob = {
                    'type': 'BULLISH',
                    'timestamp': df.index[i],
                    'high': df['High'].iloc[i],
                    'low': df['Low'].iloc[i],
                    'open': df['Open'].iloc[i],
                    'close': df['Close'].iloc[i],
                    'strength': self.calculate_block_strength(df, i)
                }
                order_blocks.append(ob)
            
            # Bearish Order Block (Demand becomes Supply)
            elif (df['Close'].iloc[i] < df['Open'].iloc[i] and  # Bearish candle
                  df['Close'].iloc[i] < df['Low'].iloc[i-lookback:i].min()):  # Breaks previous low
                
                ob = {
                    'type': 'BEARISH',
                    'timestamp': df.index[i],
                    'high': df['High'].iloc[i],
                    'low': df['Low'].iloc[i],
                    'open': df['Open'].iloc[i],
                    'close': df['Close'].iloc[i],
                    'strength': self.calculate_block_strength(df, i)
                }
                order_blocks.append(ob)
        
        return order_blocks
    
    def detect_fair_value_gaps(self, df: pd.DataFrame) -> dict:
        """Detect Fair Value Gaps (FVG)"""
        fvgs = []
        
        for i in range(1, len(df)-1):
            current = df.iloc[i]
            previous = df.iloc[i-1]
            next_candle = df.iloc[i+1]
            
            # Bullish FVG
            if (current['Low'] > previous['High'] and
                next_candle['Low'] > current['Low']):
                
                fvg = {
                    'type': 'BULLISH',
                    'timestamp': df.index[i],
                    'top': current['Low'],
                    'bottom': previous['High'],
                    'midpoint': (current['Low'] + previous['High']) / 2
                }
                fvgs.append(fvg)
            
            # Bearish FVG
            elif (current['High'] < previous['Low'] and
                  next_candle['High'] < current['High']):
                
                fvg = {
                    'type': 'BEARISH',
                    'timestamp': df.index[i],
                    'top': previous['Low'],
                    'bottom': current['High'],
                    'midpoint': (previous['Low'] + current['High']) / 2
                }
                fvgs.append(fvg)
        
        return fvgs
    
    def detect_liquidity_zones(self, df: pd.DataFrame) -> dict:
        """Detect liquidity zones (High/Low clusters)"""
        # Use volume profile to detect liquidity
        high_clusters = []
        low_clusters = []
        
        window = 20
        for i in range(window, len(df)):
            recent_highs = df['High'].iloc[i-window:i]
            recent_lows = df['Low'].iloc[i-window:i]
            
            # High liquidity zone (sellers)
            if df['High'].iloc[i] >= recent_highs.max():
                high_clusters.append({
                    'price': df['High'].iloc[i],
                    'timestamp': df.index[i],
                    'volume': df['Volume'].iloc[i]
                })
            
            # Low liquidity zone (buyers)
            if df['Low'].iloc[i] <= recent_lows.min():
                low_clusters.append({
                    'price': df['Low'].iloc[i],
                    'timestamp': df.index[i],
                    'volume': df['Volume'].iloc[i]
                })
        
        return {
            'high_liquidity': high_clusters[-10:] if high_clusters else [],
            'low_liquidity': low_clusters[-10:] if low_clusters else []
        }
    
    def calculate_block_strength(self, df: pd.DataFrame, idx: int) -> float:
        """Calculate strength of order block"""
        volume_strength = min(df['Volume'].iloc[idx] / df['Volume'].rolling(20).mean().iloc[idx], 3)
        range_strength = ((df['High'].iloc[idx] - df['Low'].iloc[idx]) / 
                         df['Close'].iloc[idx] * 100) / 2
        
        return min(volume_strength * range_strength, 10)
    
    def analyze_market_structure(self, df: pd.DataFrame) -> dict:
        """Complete market structure analysis"""
        return {
            'order_blocks': self.detect_order_blocks(df),
            'fair_value_gaps': self.detect_fair_value_gaps(df),
            'liquidity_zones': self.detect_liquidity_zones(df),
            'market_phase': self.determine_market_phase(df)
        }
    
    def determine_market_phase(self, df: pd.DataFrame) -> str:
        """Determine current market phase"""
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
        elif older_low < recent_low < recent_high < older_high:
            return "RANGE_BOUND"
        else:
            return "CONSOLIDATION"

# ============================================================================
# ADVANCED TECHNICAL INDICATORS
# ============================================================================

class AdvancedIndicators:
    """Advanced technical indicators including SMC concepts"""
    
    @staticmethod
    def calculate_atr(high, low, close, period=14):
        """Average True Range"""
        tr = pd.concat([
            high - low,
            abs(high - close.shift()),
            abs(low - close.shift())
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    @staticmethod
    def calculate_supertrend(high, low, close, period=10, multiplier=3):
        """Supertrend indicator"""
        atr = AdvancedIndicators.calculate_atr(high, low, close, period)
        
        hl2 = (high + low) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        supertrend = pd.Series(index=close.index, dtype=float)
        direction = pd.Series(index=close.index, dtype=int)
        
        supertrend.iloc[0] = upper_band.iloc[0]
        direction.iloc[0] = -1
        
        for i in range(1, len(close)):
            if close.iloc[i] > supertrend.iloc[i-1]:
                if direction.iloc[i-1] == -1:
                    supertrend.iloc[i] = lower_band.iloc[i]
                else:
                    supertrend.iloc[i] = max(lower_band.iloc[i], supertrend.iloc[i-1])
                direction.iloc[i] = 1
            else:
                if direction.iloc[i-1] == 1:
                    supertrend.iloc[i] = upper_band.iloc[i]
                else:
                    supertrend.iloc[i] = min(upper_band.iloc[i], supertrend.iloc[i-1])
                direction.iloc[i] = -1
        
        return supertrend, direction
    
    @staticmethod
    def calculate_vwap(high, low, close, volume):
        """Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap
    
    @staticmethod
    def calculate_macd(series, fast=12, slow=26, signal=9):
        """MACD Indicator"""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_rsi(series, period=14):
        """Relative Strength Index"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_bollinger_bands(series, period=20, std_dev=2):
        """Bollinger Bands"""
        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    @staticmethod
    def calculate_volume_profile(df, num_bins=20):
        """Volume Profile Analysis"""
        price_range = df['High'].max() - df['Low'].min()
        bin_size = price_range / num_bins
        
        volume_profile = {}
        for i in range(num_bins):
            price_level = df['Low'].min() + (i * bin_size)
            next_level = price_level + bin_size
            
            mask = (df['Low'] >= price_level) & (df['Low'] < next_level)
            volume_at_level = df.loc[mask, 'Volume'].sum()
            
            volume_profile[price_level] = volume_at_level
        
        return volume_profile

# ============================================================================
# AI TRADING MODELS
# ============================================================================

class AITradingModels:
    """Ensemble AI Models for Trading"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.features = {}
        self.model_performance = {}
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set"""
        features_df = df.copy()
        
        # Price features
        features_df['returns'] = df['Close'].pct_change()
        features_df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        features_df['high_low_ratio'] = df['High'] / df['Low']
        features_df['close_open_ratio'] = df['Close'] / df['Open']
        
        # Volume features
        features_df['volume_ma'] = df['Volume'].rolling(20).mean()
        features_df['volume_ratio'] = df['Volume'] / features_df['volume_ma']
        features_df['volume_std'] = df['Volume'].rolling(20).std()
        
        # Technical indicators
        features_df['rsi'] = AdvancedIndicators.calculate_rsi(df['Close'], 14)
        features_df['atr'] = AdvancedIndicators.calculate_atr(df['High'], df['Low'], df['Close'])
        features_df['atr_pct'] = features_df['atr'] / df['Close']
        
        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            features_df[f'sma_{period}'] = df['Close'].rolling(period).mean()
            features_df[f'ema_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
            features_df[f'close_to_sma_{period}'] = df['Close'] / features_df[f'sma_{period}']
        
        # MACD
        macd_line, signal_line, histogram = AdvancedIndicators.calculate_macd(df['Close'])
        features_df['macd'] = macd_line
        features_df['macd_signal'] = signal_line
        features_df['macd_hist'] = histogram
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = AdvancedIndicators.calculate_bollinger_bands(df['Close'])
        features_df['bb_upper'] = bb_upper
        features_df['bb_lower'] = bb_lower
        features_df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        features_df['bb_position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Supertrend
        supertrend, direction = AdvancedIndicators.calculate_supertrend(
            df['High'], df['Low'], df['Close']
        )
        features_df['supertrend'] = supertrend
        features_df['supertrend_dir'] = direction
        
        # VWAP
        features_df['vwap'] = AdvancedIndicators.calculate_vwap(
            df['High'], df['Low'], df['Close'], df['Volume']
        )
        features_df['price_vs_vwap'] = df['Close'] / features_df['vwap']
        
        # Momentum
        for period in [3, 5, 10, 20]:
            features_df[f'momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
        
        # Volatility
        features_df['volatility_20'] = features_df['returns'].rolling(20).std()
        features_df['volatility_50'] = features_df['returns'].rolling(50).std()
        
        # Support and Resistance
        features_df['resistance_20'] = df['High'].rolling(20).max()
        features_df['support_20'] = df['Low'].rolling(20).min()
        features_df['dist_to_resistance'] = (features_df['resistance_20'] - df['Close']) / df['Close']
        features_df['dist_to_support'] = (df['Close'] - features_df['support_20']) / df['Close']
        
        # Market regime
        features_df['trend_strength'] = abs(features_df['close_to_sma_20'] - 1)
        
        # Time features
        features_df['hour'] = features_df.index.hour
        features_df['day_of_week'] = features_df.index.dayofweek
        features_df['month'] = features_df.index.month
        
        # Fill NaN values
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(method='ffill').fillna(method='bfill')
        
        return features_df
    
    def prepare_training_data(self, df: pd.DataFrame, horizon: int = 5):
        """Prepare data for model training"""
        features_df = self.create_features(df)
        
        # Create target variable (future returns)
        future_returns = df['Close'].shift(-horizon) / df['Close'] - 1
        
        # Multi-class classification
        y = pd.cut(future_returns, 
                  bins=[-np.inf, -0.01, 0.01, np.inf],
                  labels=[-1, 0, 1])  # -1: SELL, 0: HOLD, 1: BUY
        
        # Remove rows with NaN
        valid_idx = ~features_df.isna().any(axis=1) & ~y.isna()
        X = features_df.loc[valid_idx]
        y = y.loc[valid_idx]
        
        # Feature selection (exclude target-related columns)
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'returns', 'log_returns']
        feature_cols = [col for col in X.columns if col not in exclude_cols]
        
        return X[feature_cols], y, feature_cols
    
    def train_ensemble_model(self, X, y, feature_cols, symbol: str):
        """Train ensemble model"""
        from sklearn.linear_model import LogisticRegression
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Define models
        models = [
            ('rf', RandomForestClassifier(
                n_estimators=100, max_depth=10, 
                min_samples_split=5, random_state=42, n_jobs=-1
            )),
            ('xgb', XGBClassifier(
                n_estimators=100, max_depth=6,
                learning_rate=0.1, random_state=42, n_jobs=-1
            )),
            ('lgbm', LGBMClassifier(
                n_estimators=100, max_depth=6,
                learning_rate=0.1, random_state=42, n_jobs=-1
            )),
            ('lr', LogisticRegression(
                max_iter=1000, random_state=42, n_jobs=-1
            ))
        ]
        
        # Create ensemble
        ensemble = VotingClassifier(estimators=models, voting='soft', n_jobs=-1)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            ensemble.fit(X_train, y_train)
            y_pred = ensemble.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            scores.append(accuracy)
        
        # Final training on all data
        ensemble.fit(X_scaled, y)
        
        # Store model and scaler
        self.models[symbol] = ensemble
        self.scalers[symbol] = scaler
        self.features[symbol] = feature_cols
        
        # Calculate and store performance
        cv_scores = cross_val_score(ensemble, X_scaled, y, cv=tscv, scoring='accuracy')
        self.model_performance[symbol] = {
            'accuracy': np.mean(cv_scores),
            'std': np.std(cv_scores),
            'last_trained': datetime.now()
        }
        
        print(f"✅ Model trained for {symbol}. CV Accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
        
        return ensemble
    
    def predict(self, df: pd.DataFrame, symbol: str):
        """Make prediction using trained model"""
        if symbol not in self.models:
            return 0, 0.0  # No model trained
        
        try:
            # Create features
            features_df = self.create_features(df)
            
            if len(features_df) < 100:
                return 0, 0.0
            
            # Get latest features
            latest = features_df[self.features[symbol]].iloc[-1:].values
            
            # Scale features
            scaled = self.scalers[symbol].transform(latest)
            
            # Predict
            model = self.models[symbol]
            prediction = model.predict(scaled)[0]
            
            # Get probabilities
            probabilities = model.predict_proba(scaled)[0]
            confidence = np.max(probabilities)
            
            return prediction, confidence
            
        except Exception as e:
            print(f"❌ Prediction error for {symbol}: {e}")
            return 0, 0.0

# ============================================================================
# RISK & POSITION MANAGEMENT
# ============================================================================

class RiskManager:
    """Professional Risk Management System"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.positions = {}
        self.daily_stats = {
            'trades_today': 0,
            'pnl_today': 0.0,
            'max_loss_today': 0.0,
            'winning_trades': 0,
            'losing_trades': 0
        }
        self.position_history = []
        self.max_position_size = config.total_capital * 0.1  # 10% per position
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                               risk_amount: float = None) -> Tuple[int, float]:
        """Calculate optimal position size"""
        
        if risk_amount is None:
            risk_amount = self.config.total_capital * self.config.risk_per_trade
        
        # Risk per share
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share <= 0:
            return 0, 0.0
        
        # Calculate quantity
        quantity = int(risk_amount / risk_per_share)
        
        # Adjust for max position size
        position_value = quantity * entry_price
        max_quantity = int(self.max_position_size / entry_price)
        quantity = min(quantity, max_quantity)
        
        # Minimum 1 share
        quantity = max(1, quantity)
        
        # Calculate actual risk
        actual_risk = quantity * risk_per_share
        
        return quantity, actual_risk
    
    def calculate_stop_loss(self, df: pd.DataFrame, direction: str, 
                           atr: float = None) -> float:
        """Calculate stop loss using multiple methods"""
        
        current_price = df['Close'].iloc[-1]
        
        if not atr:
            atr = AdvancedIndicators.calculate_atr(
                df['High'], df['Low'], df['Close']
            ).iloc[-1]
        
        if direction == "LONG":
            # Multiple stop loss methods
            sl_methods = []
            
            # 1. ATR based
            sl_atr = current_price - (atr * self.config.atr_multiplier)
            sl_methods.append(sl_atr)
            
            # 2. Recent low
            recent_low = df['Low'].iloc[-10:].min()
            sl_methods.append(recent_low * 0.99)  # Slightly below recent low
            
            # 3. Support level
            support = df['Low'].rolling(20).min().iloc[-1]
            sl_methods.append(support * 0.99)
            
            # Use the most conservative (highest) stop loss
            stop_loss = max(sl_methods)
            
        else:  # SHORT
            sl_methods = []
            
            # 1. ATR based
            sl_atr = current_price + (atr * self.config.atr_multiplier)
            sl_methods.append(sl_atr)
            
            # 2. Recent high
            recent_high = df['High'].iloc[-10:].max()
            sl_methods.append(recent_high * 1.01)
            
            # 3. Resistance level
            resistance = df['High'].rolling(20).max().iloc[-1]
            sl_methods.append(resistance * 1.01)
            
            # Use the most conservative (lowest) stop loss
            stop_loss = min(sl_methods)
        
        return stop_loss
    
    def calculate_take_profit(self, entry_price: float, stop_loss: float, 
                             direction: str) -> float:
        """Calculate take profit based on risk:reward ratio"""
        
        risk = abs(entry_price - stop_loss)
        reward = risk * self.config.take_profit_ratio
        
        if direction == "LONG":
            take_profit = entry_price + reward
        else:
            take_profit = entry_price - reward
        
        return take_profit
    
    def update_trailing_stop(self, position: dict, current_price: float) -> float:
        """Update trailing stop loss"""
        if not self.config.trailing_stop_enabled:
            return position['stop_loss']
        
        if position['direction'] == "LONG":
            # Calculate new high
            new_high = max(position.get('highest_price', position['entry_price']), current_price)
            position['highest_price'] = new_high
            
            # Check if trailing stop should activate
            if new_high >= position['entry_price'] * (1 + self.config.trailing_stop_activation):
                new_stop = new_high - (position['atr'] * self.config.atr_multiplier * 0.5)
                position['stop_loss'] = max(position['stop_loss'], new_stop)
        
        else:  # SHORT
            # Calculate new low
            new_low = min(position.get('lowest_price', position['entry_price']), current_price)
            position['lowest_price'] = new_low
            
            # Check if trailing stop should activate
            if new_low <= position['entry_price'] * (1 - self.config.trailing_stop_activation):
                new_stop = new_low + (position['atr'] * self.config.atr_multiplier * 0.5)
                position['stop_loss'] = min(position['stop_loss'], new_stop)
        
        return position['stop_loss']
    
    def can_take_trade(self, symbol: str, new_risk: float) -> Tuple[bool, str]:
        """Check if we can take a new trade"""
        
        # Check daily trade limit
        if self.daily_stats['trades_today'] >= self.config.max_daily_trades:
            return False, "Daily trade limit reached"
        
        # Check daily loss limit
        daily_loss_limit = self.config.total_capital * self.config.max_portfolio_risk
        if self.daily_stats['pnl_today'] + new_risk < -daily_loss_limit:
            return False, "Daily loss limit reached"
        
        # Check if already in position
        if symbol in self.positions:
            return False, "Already in position"
        
        # Check max positions
        if len(self.positions) >= self.config.max_positions:
            return False, "Max positions reached"
        
        return True, "OK"

# ============================================================================
# TRADING ENGINE (MAIN BOT)
# ============================================================================

class TradingEngine:
    """Main Autonomous Trading Engine"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.broker = BrokerManager(config)
        self.database = DatabaseManager()
        self.risk_manager = RiskManager(config)
        self.ai_models = AITradingModels(config)
        self.smc_analyzer = SMCProAnalyzer()
        
        self.signals = []
        self.active_orders = []
        self.trade_log = []
        
        self.market_phase = MarketPhase.PRE_OPEN
        self.is_running = False
        self.last_scan_time = None
        
        # Performance tracking
        self.performance = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'current_drawdown': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0
        }
        
        # Initialize logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading_bot.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def start(self):
        """Start the trading engine"""
        self.is_running = True
        self.logger.info("🚀 Trading Engine Started")
        
        # Initial setup
        self.initialize_models()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self.run_monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        return True
    
    def stop(self):
        """Stop the trading engine"""
        self.is_running = False
        self.logger.info("🛑 Trading Engine Stopped")
        
        # Square off all positions if in live mode
        if not self.config.paper_trading:
            self.square_off_all_positions()
        
        # Save final performance
        self.save_performance_metrics()
        
        return True
    
    def initialize_models(self):
        """Initialize AI models for trading universe"""
        symbols = StockUniverse.get_trading_universe()[:10]  # Train on top 10 initially
        
        for symbol in symbols:
            try:
                df = self.broker.get_historical_data(
                    symbol, 
                    self.config.data_resolution,
                    datetime.now() - timedelta(days=365),
                    datetime.now()
                )
                
                if len(df) > 100:
                    X, y, features = self.ai_models.prepare_training_data(df, self.config.prediction_horizon)
                    self.ai_models.train_ensemble_model(X, y, features, symbol)
                    self.logger.info(f"✅ Model trained for {symbol}")
                    
            except Exception as e:
                self.logger.error(f"❌ Model training failed for {symbol}: {e}")
    
    def run_monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Check market hours
                if not self.is_market_open():
                    time.sleep(60)
                    continue
                
                # Run trading cycle
                self.run_trading_cycle()
                
                # Sleep between cycles
                time.sleep(self.config.update_frequency)
                
            except Exception as e:
                self.logger.error(f"❌ Error in monitoring loop: {e}")
                time.sleep(30)
    
    def is_market_open(self) -> bool:
        """Check if market is open"""
        now = datetime.now().time()
        
        if now < self.config.pre_open_start:
            self.market_phase = MarketPhase.PRE_OPEN
            return False
        elif now < self.config.market_open:
            self.market_phase = MarketPhase.OPENING
            return True
        elif now < self.config.market_close:
            self.market_phase = MarketPhase.MID_DAY
            return True
        elif now < self.config.post_close_end:
            self.market_phase = MarketPhase.CLOSING
            return True
        else:
            self.market_phase = MarketPhase.POST_CLOSE
            return False
    
    def run_trading_cycle(self):
        """Execute one trading cycle"""
        try:
            # 1. Scan for signals
            self.scan_signals()
            
            # 2. Execute pending signals
            self.execute_signals()
            
            # 3. Manage existing positions
            self.manage_positions()
            
            # 4. Update performance metrics
            self.update_performance()
            
            # 5. Log current status
            self.log_status()
            
        except Exception as e:
            self.logger.error(f"❌ Error in trading cycle: {e}")
    
    def scan_signals(self):
        """Scan for trading signals across universe"""
        symbols = StockUniverse.get_trading_universe()[:50]  # Scan top 50
        
        for symbol in symbols:
            try:
                # Check if we can trade this symbol
                can_trade, reason = self.risk_manager.can_take_trade(symbol, 0)
                if not can_trade:
                    continue
                
                # Get market data
                df = self.broker.get_historical_data(
                    symbol,
                    self.config.data_resolution,
                    datetime.now() - timedelta(days=30),
                    datetime.now()
                )
                
                if len(df) < 100:
                    continue
                
                # AI Prediction
                prediction, confidence = self.ai_models.predict(df, symbol)
                
                # Skip if confidence is low
                if confidence < self.config.min_confidence:
                    continue
                
                # SMC Analysis
                smc_analysis = self.smc_analyzer.analyze_market_structure(df)
                
                # Get current price
                current_price = self.broker.get_ltp(symbol)
                
                # Generate signal
                signal = self.generate_signal(
                    symbol, prediction, confidence, current_price, df, smc_analysis
                )
                
                if signal:
                    self.signals.append(signal)
                    self.log_signal(signal)
                    
            except Exception as e:
                self.logger.error(f"❌ Error scanning {symbol}: {e}")
    
    def generate_signal(self, symbol: str, prediction: int, confidence: float,
                       current_price: float, df: pd.DataFrame, smc_analysis: dict):
        """Generate trading signal"""
        
        direction = "LONG" if prediction == 1 else "SHORT"
        
        # Calculate ATR
        atr = AdvancedIndicators.calculate_atr(
            df['High'], df['Low'], df['Close']
        ).iloc[-1]
        
        # Calculate stop loss
        stop_loss = self.risk_manager.calculate_stop_loss(df, direction, atr)
        
        # Calculate take profit
        take_profit = self.risk_manager.calculate_take_profit(
            current_price, stop_loss, direction
        )
        
        # Calculate position size
        quantity, risk_amount = self.risk_manager.calculate_position_size(
            current_price, stop_loss
        )
        
        # Check risk limits
        can_trade, reason = self.risk_manager.can_take_trade(symbol, risk_amount)
        if not can_trade:
            return None
        
        # Create signal
        signal = {
            'signal_id': f"SIG_{int(time.time())}_{symbol}",
            'symbol': symbol,
            'direction': direction,
            'prediction': prediction,
            'confidence': confidence,
            'current_price': current_price,
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'quantity': quantity,
            'risk_amount': risk_amount,
            'atr': atr,
            'timestamp': datetime.now(),
            'market_phase': self.market_phase.value,
            'smc_analysis': smc_analysis,
            'strategy': 'AI_SMC_PRO'
        }
        
        return signal
    
    def execute_signals(self):
        """Execute pending signals"""
        for signal in self.signals[:]:  # Copy list for iteration
            try:
                # Check if still valid (within 1 minute)
                if (datetime.now() - signal['timestamp']).seconds > 60:
                    self.signals.remove(signal)
                    continue
                
                # Place order
                order_result = self.broker.place_order(
                    symbol=signal['symbol'],
                    direction=signal['direction'],
                    quantity=signal['quantity'],
                    order_type="MARKET"
                )
                
                if order_result['status'] == 'success':
                    # Create position
                    position = {
                        'position_id': f"POS_{int(time.time())}_{signal['symbol']}",
                        'symbol': signal['symbol'],
                        'direction': signal['direction'],
                        'entry_time': datetime.now(),
                        'entry_price': order_result.get('executed_price', signal['current_price']),
                        'quantity': signal['quantity'],
                        'stop_loss': signal['stop_loss'],
                        'take_profit': signal['take_profit'],
                        'atr': signal['atr'],
                        'confidence': signal['confidence'],
                        'status': 'OPEN',
                        'highest_price': order_result.get('executed_price', signal['current_price']),
                        'lowest_price': order_result.get('executed_price', signal['current_price'])
                    }
                    
                    # Add to positions
                    self.risk_manager.positions[signal['symbol']] = position
                    
                    # Log trade
                    self.log_trade({
                        'type': 'ENTRY',
                        'position_id': position['position_id'],
                        'symbol': signal['symbol'],
                        'direction': signal['direction'],
                        'entry_price': position['entry_price'],
                        'quantity': signal['quantity'],
                        'stop_loss': signal['stop_loss'],
                        'take_profit': signal['take_profit'],
                        'confidence': signal['confidence'],
                        'order_id': order_result.get('order_id'),
                        'timestamp': datetime.now(),
                        'remarks': order_result.get('message', 'Auto Entry')
                    })
                    
                    # Update daily stats
                    self.risk_manager.daily_stats['trades_today'] += 1
                    
                    # Remove signal
                    self.signals.remove(signal)
                    
                    self.logger.info(f"✅ Executed: {signal['direction']} {signal['symbol']} "
                                   f"x{signal['quantity']} @ {position['entry_price']:.2f}")
                    
            except Exception as e:
                self.logger.error(f"❌ Error executing signal {signal['signal_id']}: {e}")
    
    def manage_positions(self):
        """Manage existing positions"""
        for symbol, position in list(self.risk_manager.positions.items()):
            try:
                # Get current price
                current_price = self.broker.get_ltp(symbol)
                
                # Update trailing stop
                if self.config.trailing_stop_enabled:
                    position['stop_loss'] = self.risk_manager.update_trailing_stop(
                        position, current_price
                    )
                
                # Check exit conditions
                exit_reason = None
                exit_price = None
                
                if position['direction'] == "LONG":
                    # Stop Loss hit
                    if current_price <= position['stop_loss']:
                        exit_reason = "STOP_LOSS"
                        exit_price = position['stop_loss']
                    
                    # Take Profit hit
                    elif current_price >= position['take_profit']:
                        exit_reason = "TAKE_PROFIT"
                        exit_price = position['take_profit']
                    
                    # Trailing stop hit
                    elif current_price <= position['stop_loss']:
                        exit_reason = "TRAILING_STOP"
                        exit_price = current_price
                
                else:  # SHORT
                    # Stop Loss hit
                    if current_price >= position['stop_loss']:
                        exit_reason = "STOP_LOSS"
                        exit_price = position['stop_loss']
                    
                    # Take Profit hit
                    elif current_price <= position['take_profit']:
                        exit_reason = "TAKE_PROFIT"
                        exit_price = position['take_profit']
                    
                    # Trailing stop hit
                    elif current_price >= position['stop_loss']:
                        exit_reason = "TRAILING_STOP"
                        exit_price = current_price
                
                # Exit if needed
                if exit_reason:
                    self.exit_position(symbol, exit_price, exit_reason)
                    
            except Exception as e:
                self.logger.error(f"❌ Error managing position {symbol}: {e}")
    
    def exit_position(self, symbol: str, exit_price: float, reason: str):
        """Exit a position"""
        position = self.risk_manager.positions.get(symbol)
        if not position:
            return
        
        # Calculate P&L
        if position['direction'] == "LONG":
            pnl = (exit_price - position['entry_price']) * position['quantity']
        else:  # SHORT
            pnl = (position['entry_price'] - exit_price) * position['quantity']
        
        pnl_percentage = (pnl / (position['entry_price'] * position['quantity'])) * 100
        
        # Place exit order
        exit_direction = "SELL" if position['direction'] == "LONG" else "BUY"
        
        order_result = self.broker.place_order(
            symbol=symbol,
            direction=exit_direction,
            quantity=position['quantity'],
            order_type="MARKET"
        )
        
        if order_result['status'] == 'success':
            # Update performance
            self.performance['total_trades'] += 1
            
            if pnl > 0:
                self.performance['winning_trades'] += 1
                self.performance['largest_win'] = max(self.performance['largest_win'], pnl)
            else:
                self.performance['losing_trades'] += 1
                self.performance['largest_loss'] = min(self.performance['largest_loss'], pnl)
            
            self.performance['total_pnl'] += pnl
            
            # Update win rate
            if self.performance['total_trades'] > 0:
                self.performance['win_rate'] = (
                    self.performance['winning_trades'] / 
                    self.performance['total_trades'] * 100
                )
            
            # Update daily stats
            self.risk_manager.daily_stats['pnl_today'] += pnl
            if pnl > 0:
                self.risk_manager.daily_stats['winning_trades'] += 1
            else:
                self.risk_manager.daily_stats['losing_trades'] += 1
            
            # Log exit
            self.log_trade({
                'type': 'EXIT',
                'position_id': position['position_id'],
                'symbol': symbol,
                'direction': position['direction'],
                'exit_price': exit_price,
                'quantity': position['quantity'],
                'pnl': pnl,
                'pnl_percentage': pnl_percentage,
                'reason': reason,
                'order_id': order_result.get('order_id'),
                'timestamp': datetime.now(),
                'remarks': f"Auto Exit - {reason}"
            })
            
            # Save to database
            trade_data = {
                'trade_id': position['position_id'],
                'symbol': symbol,
                'direction': position['direction'],
                'entry_time': position['entry_time'],
                'entry_price': position['entry_price'],
                'exit_time': datetime.now(),
                'exit_price': exit_price,
                'quantity': position['quantity'],
                'stop_loss': position['stop_loss'],
                'take_profit': position['take_profit'],
                'pnl': pnl,
                'pnl_percentage': pnl_percentage,
                'status': 'CLOSED',
                'confidence': position['confidence'],
                'strategy': 'AI_SMC_PRO',
                'remarks': f"Exit: {reason}"
            }
            
            self.database.save_trade(trade_data)
            
            # Remove position
            del self.risk_manager.positions[symbol]
            
            self.logger.info(f"📤 Exited {symbol}: {reason}, P&L: ₹{pnl:,.2f} ({pnl_percentage:.2f}%)")
    
    def square_off_all_positions(self):
        """Square off all open positions"""
        for symbol in list(self.risk_manager.positions.keys()):
            try:
                position = self.risk_manager.positions[symbol]
                current_price = self.broker.get_ltp(symbol)
                self.exit_position(symbol, current_price, "SHUTDOWN")
            except Exception as e:
                self.logger.error(f"❌ Error squaring off {symbol}: {e}")
    
    def update_performance(self):
        """Update performance metrics"""
        # Calculate profit factor
        if self.performance['losing_trades'] > 0:
            gross_profit = self.performance['winning_trades'] * 1000  # Simplified
            gross_loss = abs(self.performance['losing_trades'] * 500)  # Simplified
            self.performance['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Update current drawdown
        if self.performance['total_pnl'] < self.performance['max_drawdown']:
            self.performance['max_drawdown'] = self.performance['total_pnl']
        
        self.performance['current_drawdown'] = self.performance['total_pnl'] - self.performance['max_drawdown']
    
    def save_performance_metrics(self):
        """Save performance metrics to database"""
        try:
            conn = sqlite3.connect(self.database.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO performance 
                (date, total_trades, winning_trades, losing_trades, total_pnl, 
                 win_rate, profit_factor, max_drawdown, sharpe_ratio, sortino_ratio)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().date(),
                self.performance['total_trades'],
                self.performance['winning_trades'],
                self.performance['losing_trades'],
                self.performance['total_pnl'],
                self.performance['win_rate'],
                self.performance['profit_factor'],
                self.performance['max_drawdown'],
                self.performance['sharpe_ratio'],
                0.0  # Simplified sortino
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"❌ Error saving performance metrics: {e}")
    
    def log_signal(self, signal: dict):
        """Log trading signal"""
        log_entry = {
            'timestamp': datetime.now(),
            'type': 'SIGNAL',
            'symbol': signal['symbol'],
            'direction': signal['direction'],
            'price': signal['current_price'],
            'confidence': signal['confidence'],
            'stop_loss': signal['stop_loss'],
            'take_profit': signal['take_profit'],
            'quantity': signal['quantity'],
            'risk': signal['risk_amount']
        }
        
        self.trade_log.append(log_entry)
        
        # Keep log size manageable
        if len(self.trade_log) > 1000:
            self.trade_log = self.trade_log[-500:]
    
    def log_trade(self, trade: dict):
        """Log trade execution"""
        log_entry = {
            'timestamp': datetime.now(),
            'type': trade['type'],
            'symbol': trade['symbol'],
            'direction': trade.get('direction'),
            'price': trade.get('entry_price') or trade.get('exit_price'),
            'quantity': trade.get('quantity'),
            'pnl': trade.get('pnl'),
            'reason': trade.get('reason'),
            'order_id': trade.get('order_id'),
            'remarks': trade.get('remarks')
        }
        
        self.trade_log.append(log_entry)
        
        # Keep log size manageable
        if len(self.trade_log) > 1000:
            self.trade_log = self.trade_log[-500:]
    
    def log_status(self):
        """Log current system status"""
        status = {
            'timestamp': datetime.now(),
            'market_phase': self.market_phase.value,
            'active_positions': len(self.risk_manager.positions),
            'pending_signals': len(self.signals),
            'total_pnl': self.performance['total_pnl'],
            'daily_pnl': self.risk_manager.daily_stats['pnl_today'],
            'trades_today': self.risk_manager.daily_stats['trades_today'],
            'win_rate': self.performance['win_rate']
        }
        
        # Log every 5 minutes
        if not hasattr(self, 'last_status_log') or \
           (datetime.now() - self.last_status_log).seconds > 300:
            self.logger.info(f"📊 Status: {status}")
            self.last_status_log = datetime.now()
    
    def get_system_status(self) -> dict:
        """Get current system status"""
        return {
            'engine_running': self.is_running,
            'market_phase': self.market_phase.value,
            'broker_connected': self.broker.connected,
            'paper_trading': self.config.paper_trading,
            'active_positions': len(self.risk_manager.positions),
            'pending_signals': len(self.signals),
            'performance': self.performance,
            'daily_stats': self.risk_manager.daily_stats,
            'last_update': datetime.now()
        }

# ============================================================================
# STREAMLIT DASHBOARD
# ============================================================================

def main():
    """Main Streamlit Trading Dashboard"""
    
    st.set_page_config(
        page_title="Institutional AI Trading Bot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(45deg, #1E88E5, #4CAF50);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.2rem;
        border-radius: 15px;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .profit {
        color: #00C853;
        font-weight: bold;
    }
    .loss {
        color: #FF5252;
        font-weight: bold;
    }
    .status-running {
        color: #00C853;
        animation: pulse 2s infinite;
    }
    .status-stopped {
        color: #FF5252;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    .trade-log {
        background: #0d1117;
        border-radius: 10px;
        padding: 1rem;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        max-height: 400px;
        overflow-y: auto;
    }
    .log-entry {
        padding: 0.5rem;
        border-bottom: 1px solid #30363d;
    }
    .log-entry:nth-child(even) {
        background: #161b22;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'trading_engine' not in st.session_state:
        config = TradingConfig()
        st.session_state.trading_engine = TradingEngine(config)
        st.session_state.auto_refresh = True
        st.session_state.last_refresh = datetime.now()
    
    engine = st.session_state.trading_engine
    
    # Header
    st.markdown("<h1 class='main-header'>🏦 INSTITUTIONAL AI TRADING BOT</h1>", unsafe_allow_html=True)
    st.markdown("### 🤖 Autonomous Algorithmic Trading System with SMC Pro")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ **CONTROL PANEL**")
        
        # Mode Selection
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 START BOT", type="primary", use_container_width=True):
                if engine.start():
                    st.success("✅ Trading Bot Started!")
                    st.rerun()
        
        with col2:
            if st.button("🛑 STOP BOT", type="secondary", use_container_width=True):
                if engine.stop():
                    st.warning("⚠️ Trading Bot Stopped!")
                    st.rerun()
        
        st.markdown("---")
        
        # Trading Mode
        st.markdown("### 📊 Trading Mode")
        mode = st.radio(
            "Select Mode",
            ["📈 Paper Trading", "💰 Live Trading", "🎮 Demo Mode"],
            index=0,
            label_visibility="collapsed"
        )
        
        # Capital Settings
        st.markdown("### 💰 Capital Management")
        capital = st.number_input(
            "Total Capital (₹)",
            min_value=100000,
            max_value=10000000,
            value=2000000,
            step=100000,
            help="Total trading capital"
        )
        
        risk_per_trade = st.slider(
            "Risk per Trade (%)",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1,
            help="Percentage of capital risked per trade"
        ) / 100
        
        # AI Settings
        st.markdown("### 🧠 AI Configuration")
        confidence = st.slider(
            "Confidence Threshold (%)",
            min_value=50,
            max_value=90,
            value=65,
            step=5,
            help="Minimum confidence for trade execution"
        ) / 100
        
        # SMC Settings
        st.markdown("### 🔍 SMC Pro Settings")
        smc_enabled = st.checkbox("Enable SMC Pro Analysis", value=True)
        market_structure = st.checkbox("Detect Market Structure", value=True)
        order_blocks = st.checkbox("Use Order Blocks", value=True)
        
        # Update Config
        engine.config.demo_mode = "Demo Mode" in mode
        engine.config.paper_trading = "Paper Trading" in mode
        engine.config.live_trading = "Live Trading" in mode
        engine.config.total_capital = capital
        engine.config.risk_per_trade = risk_per_trade
        engine.config.min_confidence = confidence
        engine.config.use_smc = smc_enabled
        engine.config.detect_market_structure = market_structure
        engine.config.use_order_blocks = order_blocks
        
        # Manual Trading
        st.markdown("---")
        st.markdown("### 🎯 Manual Trading")
        
        with st.expander("Place Manual Order"):
            manual_symbol = st.selectbox(
                "Symbol",
                StockUniverse.get_nifty_50()[:20],
                index=0
            )
            
            manual_direction = st.selectbox(
                "Direction",
                ["LONG", "SHORT"],
                index=0
            )
            
            manual_quantity = st.number_input(
                "Quantity",
                min_value=1,
                value=10,
                step=1
            )
            
            if st.button("📝 Place Manual Order", type="secondary"):
                order_result = engine.broker.place_order(
                    manual_symbol,
                    manual_direction,
                    manual_quantity,
                    "MARKET"
                )
                
                if order_result['status'] == 'success':
                    st.success(f"✅ {order_result['message']}")
                else:
                    st.error(f"❌ {order_result['message']}")
        
        # Auto-refresh
        st.markdown("---")
        auto_refresh = st.checkbox("🔄 Auto Refresh Dashboard", value=True)
        st.session_state.auto_refresh = auto_refresh
        
        if auto_refresh:
            refresh_rate = st.slider("Refresh Rate (seconds)", 5, 60, 10)
            if (datetime.now() - st.session_state.last_refresh).seconds >= refresh_rate:
                st.session_state.last_refresh = datetime.now()
                st.rerun()
    
    # Main Dashboard
    # Top Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        status = "RUNNING" if engine.is_running else "STOPPED"
        status_class = "status-running" if engine.is_running else "status-stopped"
        st.markdown(f'<h3 style="{status_class}">{status}</h3>', unsafe_allow_html=True)
        st.markdown("**System Status**")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        total_pnl = engine.performance['total_pnl']
        pnl_class = "profit" if total_pnl >= 0 else "loss"
        st.markdown(f'<h3 class="{pnl_class}">₹{total_pnl:,.2f}</h3>', unsafe_allow_html=True)
        st.markdown("**Total P&L**")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        win_rate = engine.performance['win_rate']
        st.markdown(f'<h3>{win_rate:.1f}%</h3>', unsafe_allow_html=True)
        st.markdown("**Win Rate**")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        active_positions = len(engine.risk_manager.positions)
        st.markdown(f'<h3>{active_positions}</h3>', unsafe_allow_html=True)
        st.markdown("**Active Positions**")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Dashboard", 
        "📈 Positions", 
        "🚦 Signals", 
        "📋 Trade History", 
        "📉 Charts", 
        "📊 Analytics"
    ])
    
    with tab1:
        # Real-time Dashboard
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📈 Performance Overview")
            
            # Create performance chart
            dates = pd.date_range(start='2025-12-01', periods=30, freq='D')
            pnl_data = np.random.normal(0, 50000, 30).cumsum()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates, y=pnl_data,
                mode='lines+markers',
                name='Cumulative P&L',
                line=dict(color='#00C853', width=3),
                marker=dict(size=6)
            ))
            
            fig.update_layout(
                title="Cumulative P&L Trend",
                xaxis_title="Date",
                yaxis_title="P&L (₹)",
                template="plotly_dark",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### ⚡ Quick Stats")
            
            stats_data = {
                "Total Trades": engine.performance['total_trades'],
                "Winning Trades": engine.performance['winning_trades'],
                "Losing Trades": engine.performance['losing_trades'],
                "Profit Factor": f"{engine.performance['profit_factor']:.2f}",
                "Largest Win": f"₹{engine.performance['largest_win']:,.2f}",
                "Largest Loss": f"₹{engine.performance['largest_loss']:,.2f}",
                "Max Drawdown": f"₹{engine.performance['max_drawdown']:,.2f}",
                "Daily Trades": engine.risk_manager.daily_stats['trades_today'],
                "Daily P&L": f"₹{engine.risk_manager.daily_stats['pnl_today']:,.2f}"
            }
            
            for key, value in stats_data.items():
                st.metric(key, value)
            
            # Market Phase
            st.markdown("### 🌐 Market Phase")
            phase_colors = {
                "PRE_OPEN": "#FF9800",
                "OPENING": "#4CAF50",
                "MID_DAY": "#2196F3",
                "CLOSING": "#FF5722",
                "POST_CLOSE": "#9C27B0"
            }
            
            phase = engine.market_phase.value
            color = phase_colors.get(phase, "#666666")
            st.markdown(f'<h3 style="color: {color}">{phase.replace("_", " ")}</h3>', unsafe_allow_html=True)
    
    with tab2:
        # Active Positions
        st.markdown("### 📈 Active Positions")
        
        if engine.risk_manager.positions:
            positions_data = []
            for symbol, pos in engine.risk_manager.positions.items():
                current_price = engine.broker.get_ltp(symbol)
                pnl = (current_price - pos['entry_price']) * pos['quantity'] if pos['direction'] == "LONG" \
                      else (pos['entry_price'] - current_price) * pos['quantity']
                pnl_percentage = (pnl / (pos['entry_price'] * pos['quantity'])) * 100
                
                positions_data.append({
                    'Symbol': symbol,
                    'Direction': pos['direction'],
                    'Entry Price': f"₹{pos['entry_price']:.2f}",
                    'Current Price': f"₹{current_price:.2f}",
                    'Quantity': pos['quantity'],
                    'P&L': f"₹{pnl:,.2f}",
                    'P&L %': f"{pnl_percentage:.2f}%",
                    'Stop Loss': f"₹{pos['stop_loss']:.2f}",
                    'Take Profit': f"₹{pos['take_profit']:.2f}",
                    'ATR': f"{pos['atr']:.2f}",
                    'Confidence': f"{pos['confidence']:.1%}"
                })
            
            st.dataframe(
                positions_data,
                use_container_width=True,
                hide_index=True
            )
            
            # Manual Exit Buttons
            st.markdown("### 🛑 Manual Exit")
            cols = st.columns(4)
            for idx, (symbol, pos) in enumerate(list(engine.risk_manager.positions.items())[:4]):
                with cols[idx % 4]:
                    if st.button(f"Exit {symbol}", key=f"exit_{symbol}"):
                        current_price = engine.broker.get_ltp(symbol)
                        engine.exit_position(symbol, current_price, "MANUAL_EXIT")
                        st.rerun()
        else:
            st.info("📭 No active positions")
    
    with tab3:
        # Trading Signals
        st.markdown("### 🚦 Trading Signals")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("🔍 Scan for New Signals", type="primary"):
                engine.scan_signals()
                st.success("Signal scan completed!")
                st.rerun()
        
        with col2:
            if st.button("⚡ Execute All Signals", type="secondary"):
                engine.execute_signals()
                st.success("Signals executed!")
                st.rerun()
        
        # Display Signals
        if engine.signals:
            signals_data = []
            for sig in engine.signals[:20]:  # Show latest 20
                signals_data.append({
                    'Symbol': sig['symbol'],
                    'Direction': sig['direction'],
                    'Signal Price': f"₹{sig['current_price']:.2f}",
                    'Confidence': f"{sig['confidence']:.1%}",
                    'Stop Loss': f"₹{sig['stop_loss']:.2f}",
                    'Take Profit': f"₹{sig['take_profit']:.2f}",
                    'Quantity': sig['quantity'],
                    'Risk': f"₹{sig['risk_amount']:,.2f}",
                    'Strategy': sig['strategy'],
                    'Time': sig['timestamp'].strftime("%H:%M:%S")
                })
            
            st.dataframe(
                signals_data,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("📭 No pending signals")
    
    with tab4:
        # Trade History
        st.markdown("### 📋 Trade History")
        
        # Get trade history from database
        trade_history = engine.database.get_trade_history(100)
        
        if not trade_history.empty:
            # Format for display
            display_df = trade_history.copy()
            display_df['entry_time'] = pd.to_datetime(display_df['entry_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
            display_df['exit_time'] = pd.to_datetime(display_df['exit_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Color code P&L
            def color_pnl(val):
                color = 'green' if val > 0 else 'red'
                return f'color: {color}; font-weight: bold'
            
            styled_df = display_df.style.applymap(color_pnl, subset=['pnl'])
            
            st.dataframe(
                styled_df,
                use_container_width=True,
                height=600
            )
            
            # Export option
            if st.button("📥 Export Trade History"):
                csv = trade_history.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"trade_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        else:
            st.info("📭 No trade history available")
    
    with tab5:
        # Technical Charts
        st.markdown("### 📉 Technical Analysis")
        
        # Symbol selector
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            selected_symbol = st.selectbox(
                "Select Symbol",
                StockUniverse.get_nifty_50()[:30],
                index=0,
                key="chart_symbol"
            )
        
        with col2:
            interval = st.selectbox(
                "Interval",
                ["5min", "15min", "1hour", "1day"],
                index=1
            )
        
        with col3:
            indicators = st.multiselect(
                "Indicators",
                ["SMA", "EMA", "RSI", "MACD", "BB", "ATR", "Volume"],
                default=["SMA", "RSI", "Volume"]
            )
        
        # Fetch and display chart
        df = engine.broker.get_historical_data(
            selected_symbol,
            interval,
            datetime.now() - timedelta(days=30),
            datetime.now()
        )
        
        if not df.empty:
            # Create chart with subplots
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.6, 0.2, 0.2],
                subplot_titles=(f'{selected_symbol} - Price Chart', 'RSI', 'Volume')
            )
            
            # Candlestick
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
            
            # Add selected indicators
            if "SMA" in indicators:
                sma_20 = df['Close'].rolling(20).mean()
                fig.add_trace(
                    go.Scatter(x=df.index, y=sma_20, name='SMA 20', 
                             line=dict(color='orange', width=1)),
                    row=1, col=1
                )
            
            if "EMA" in indicators:
                ema_50 = df['Close'].ewm(span=50, adjust=False).mean()
                fig.add_trace(
                    go.Scatter(x=df.index, y=ema_50, name='EMA 50', 
                             line=dict(color='red', width=1)),
                    row=1, col=1
                )
            
            # RSI
            if "RSI" in indicators:
                rsi = AdvancedIndicators.calculate_rsi(df['Close'], 14)
                fig.add_trace(
                    go.Scatter(x=df.index, y=rsi, name='RSI', 
                             line=dict(color='purple', width=2)),
                    row=2, col=1
                )
                
                # Add RSI levels
                fig.add_hline(y=70, line_dash="dash", line_color="red", 
                            row=2, col=1, opacity=0.3)
                fig.add_hline(y=30, line_dash="dash", line_color="green", 
                            row=2, col=1, opacity=0.3)
            
            # Volume
            colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] 
                     else 'green' for i in range(len(df))]
            
            fig.add_trace(
                go.Bar(x=df.index, y=df['Volume'], name='Volume',
                      marker_color=colors),
                row=3, col=1
            )
            
            fig.update_layout(
                title=f"{selected_symbol} Technical Analysis",
                template='plotly_dark',
                showlegend=True,
                height=800,
                xaxis_rangeslider_visible=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # SMC Analysis
            if engine.config.use_smc:
                st.markdown("### 🔍 SMC Pro Analysis")
                
                smc_results = engine.smc_analyzer.analyze_market_structure(df)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Market Phase", smc_results['market_phase'])
                
                with col2:
                    st.metric("Order Blocks", len(smc_results['order_blocks']))
                
                with col3:
                    st.metric("FVG Count", len(smc_results['fair_value_gaps']))
                
                # Display order blocks
                if smc_results['order_blocks']:
                    st.markdown("#### Recent Order Blocks")
                    for ob in smc_results['order_blocks'][-5:]:
                        st.write(f"- **{ob['type']}** at ₹{ob['close']:.2f} (Strength: {ob['strength']:.1f})")
        else:
            st.warning("No data available for selected symbol")
    
    with tab6:
        # Analytics
        st.markdown("### 📊 Advanced Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Win Rate Distribution
            labels = ['Winning Trades', 'Losing Trades']
            values = [engine.performance['winning_trades'], 
                     engine.performance['losing_trades']]
            
            fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
            fig_pie.update_layout(title="Trade Distribution", template="plotly_dark")
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Daily Performance
            dates = pd.date_range(start='2025-12-01', periods=15, freq='D')
            daily_pnl = np.random.normal(5000, 20000, 15)
            
            fig_bar = go.Figure(data=[go.Bar(x=dates, y=daily_pnl)])
            fig_bar.update_layout(
                title="Daily P&L",
                template="plotly_dark",
                xaxis_title="Date",
                yaxis_title="P&L (₹)"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Risk Metrics
        st.markdown("### ⚠️ Risk Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Max Drawdown", f"₹{engine.performance['max_drawdown']:,.2f}")
        
        with col2:
            st.metric("Current Drawdown", f"₹{engine.performance['current_drawdown']:,.2f}")
        
        with col3:
            st.metric("Sharpe Ratio", f"{engine.performance['sharpe_ratio']:.2f}")
        
        with col4:
            st.metric("Profit Factor", f"{engine.performance['profit_factor']:.2f}")
        
        # Trade Log
        st.markdown("### 📝 Real-time Trade Log")
        
        log_container = st.container()
        with log_container:
            st.markdown('<div class="trade-log">', unsafe_allow_html=True)
            
            for log in reversed(engine.trade_log[-20:]):
                timestamp = log['timestamp'].strftime("%H:%M:%S")
                symbol = log.get('symbol', '')
                log_type = log.get('type', '')
                price = log.get('price', 0)
                pnl = log.get('pnl', 0)
                reason = log.get('reason', '')
                
                if log_type == 'ENTRY':
                    icon = "🟢"
                    message = f"{icon} {timestamp} - ENTRY: {symbol} @ ₹{price:.2f}"
                elif log_type == 'EXIT':
                    icon = "🔴" if pnl < 0 else "🟢"
                    message = f"{icon} {timestamp} - EXIT: {symbol} @ ₹{price:.2f} P&L: ₹{pnl:,.2f} ({reason})"
                elif log_type == 'SIGNAL':
                    icon = "🟡"
                    message = f"{icon} {timestamp} - SIGNAL: {symbol} @ ₹{price:.2f}"
                else:
                    message = f"{timestamp} - {log_type}: {symbol}"
                
                st.markdown(f'<div class="log-entry">{message}</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center; color: #666; font-size: 0.8rem;'>
        <p>🚨 <b>RISK DISCLAIMER:</b> This is an AI-powered algorithmic trading system for educational purposes only.</p>
        <p>Past performance does not guarantee future results. Trading involves substantial risk of loss.</p>
        <p>© 2025 Institutional AI Trading Bot v3.0.0 | SMC Pro Edition</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# REQUIREMENTS.TXT
# ============================================================================

"""
streamlit==1.52.2
pandas==2.3.3
numpy==2.3.5
scikit-learn==1.8.0
plotly==6.5.0
scipy==1.16.3
kiteconnect==5.0.1
xgboost==2.1.3
lightgbm==4.5.0
matplotlib==3.10.8
seaborn==0.13.2
sqlite3
pytz==2025.2
"""

if __name__ == "__main__":
    main()
