"""
INSTITUTIONAL AI ALGO TRADING SYSTEM
For Nifty 50 & Midcap Stocks
Complete with ML Models, Risk Management, and Dashboard
"""

import sys
import os
import warnings
import json
import pickle
import threading
import schedule
import hashlib
import logging
import time
from datetime import datetime, time as dt_time, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from collections import deque, defaultdict
import pytz

# Data & ML Libraries
import pandas as pd
import numpy as np
import pandas_ta as ta
from scipy import stats
from scipy.signal import argrelextrema

# Machine Learning
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import joblib
from imblearn.over_sampling import SMOTE

# Dashboard & Visualization
import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc

# Trading & Finance
try:
    from kiteconnect import KiteConnect, KiteTicker
except ImportError:
    print("Install kiteconnect: pip install kiteconnect")
    sys.exit(1)

# Suppress warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

class MarketPhase(Enum):
    PRE_OPEN = "pre_open"
    OPENING = "opening"
    MID_DAY = "mid_day"
    CLOSING = "closing"
    POST_CLOSE = "post_close"

@dataclass
class TradingConfig:
    """Complete trading configuration"""
    # API Configuration
    api_key: str = "YOUR_API_KEY"
    api_secret: str = "YOUR_API_SECRET"
    access_token: str = ""
    
    # Capital Management
    total_capital: float = 5_000_000.0  # 50L capital
    risk_per_trade: float = 0.005  # 0.5% risk per trade
    max_portfolio_risk: float = 0.02  # 2% max daily loss
    max_positions: int = 10
    position_sizing_method: str = "kelly"  # kelly, fixed, volatility
    
    # Market Hours (IST)
    market_open: dt_time = dt_time(9, 15)
    market_close: dt_time = dt_time(15, 30)
    pre_open_start: dt_time = dt_time(9, 0)
    square_off_time: dt_time = dt_time(15, 20)
    
    # AI Model Parameters
    model_retrain_hours: int = 24
    min_confidence_threshold: float = 0.65
    ensemble_weight_rf: float = 0.4
    ensemble_weight_xgb: float = 0.4
    ensemble_weight_gb: float = 0.2
    
    # Risk Management
    stop_loss_method: str = "atr"  # atr, volatility, support_resistance
    atr_multiplier: float = 2.0
    trailing_sl_enabled: bool = True
    trailing_sl_activation: float = 1.5  # RR ratio to activate
    max_drawdown_limit: float = 0.08  # 8% max drawdown
    
    # Strategy Parameters
    lookback_period: int = 100  # Candles for feature engineering
    prediction_horizon: int = 5  # Predict 5 candles ahead
    
    # Data Parameters
    historical_days: int = 365  # 1 year data
    update_frequency: int = 60  # Seconds
    
    # Dashboard
    dashboard_port: int = 8050
    update_interval: int = 5000  # ms

@dataclass
class StockUniverse:
    """Nifty 50 & Midcap stocks with metadata"""
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
    
    @classmethod
    def get_all_symbols(cls):
        return cls.nifty_50 + cls.nifty_midcap

@dataclass
class Position:
    """Complete position tracking"""
    symbol: str
    direction: str  # "LONG" or "SHORT"
    quantity: int
    entry_price: float
    entry_time: datetime
    stop_loss: float
    target: float
    current_price: float = 0.0
    trailing_sl: float = 0.0
    high_water: float = 0.0
    low_water: float = 0.0
    is_trailing: bool = False
    pnl: float = 0.0
    pnl_percentage: float = 0.0
    risk_amount: float = 0.0
    atr: float = 0.0
    
    def update(self, current_price: float):
        """Update position with current price"""
        self.current_price = current_price
        
        if self.direction == "LONG":
            self.pnl = (current_price - self.entry_price) * self.quantity
            self.high_water = max(self.high_water, current_price)
        else:  # SHORT
            self.pnl = (self.entry_price - current_price) * self.quantity
            self.low_water = min(self.low_water, current_price)
        
        self.pnl_percentage = (self.pnl / (self.entry_price * self.quantity)) * 100
        
        # Update trailing SL
        if self.is_trailing:
            if self.direction == "LONG":
                self.trailing_sl = max(self.trailing_sl, 
                                      self.high_water * (1 - 0.01))  # 1% trail
            else:
                self.trailing_sl = min(self.trailing_sl,
                                      self.low_water * (1 + 0.01))
    
    def should_exit(self) -> Tuple[bool, str]:
        """Check exit conditions"""
        if self.direction == "LONG":
            if self.current_price <= self.stop_loss:
                return True, "Stop Loss Hit"
            elif self.current_price <= self.trailing_sl and self.is_trailing:
                return True, "Trailing SL Hit"
            elif self.current_price >= self.target:
                return True, "Target Achieved"
        else:  # SHORT
            if self.current_price >= self.stop_loss:
                return True, "Stop Loss Hit"
            elif self.current_price >= self.trailing_sl and self.is_trailing:
                return True, "Trailing SL Hit"
            elif self.current_price <= self.target:
                return True, "Target Achieved"
        
        return False, ""

# ============================================================================
# LOGGING SETUP
# ============================================================================

class TradingLogger:
    """Enhanced logging with file rotation and different log levels"""
    
    def __init__(self, name: str = "AI_Trading_System"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler with rotation
        file_handler = logging.FileHandler(f'trading_system_{datetime.now().date()}.log')
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, msg: str):
        self.logger.info(msg)
    
    def warning(self, msg: str):
        self.logger.warning(msg)
    
    def error(self, msg: str):
        self.logger.error(msg)
    
    def critical(self, msg: str):
        self.logger.critical(msg)

# ============================================================================
# DATA MANAGER
# ============================================================================

class DataManager:
    """Handles all data operations including live streaming"""
    
    def __init__(self, config: TradingConfig, logger: TradingLogger):
        self.config = config
        self.logger = logger
        self.kite = None
        self.ticker = None
        self.instrument_tokens = {}
        self.historical_data = {}
        self.live_data = {}
        self.initialize_kite()
    
    def initialize_kite(self):
        """Initialize Kite Connect API"""
        try:
            self.kite = KiteConnect(api_key=self.config.api_key)
            
            if not self.config.access_token:
                # Generate access token if not provided
                print("\n" + "="*50)
                print("Visit this URL to get request token:")
                print(self.kite.login_url())
                print("="*50 + "\n")
                
                request_token = input("Enter request token: ")
                data = self.kite.generate_session(request_token, 
                                                 api_secret=self.config.api_secret)
                self.config.access_token = data['access_token']
                self.logger.info("Access token generated successfully")
            
            self.kite.set_access_token(self.config.access_token)
            self.logger.info("Kite Connect initialized successfully")
            
            # Initialize ticker for live data
            self.ticker = KiteTicker(
                self.config.api_key,
                self.config.access_token
            )
            
            # Load instrument tokens
            self.load_instrument_tokens()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Kite Connect: {str(e)}")
            raise
    
    def load_instrument_tokens(self):
        """Load instrument tokens for NSE stocks"""
        instruments = self.kite.instruments("NSE")
        for ins in instruments:
            if ins['tradingsymbol'] in StockUniverse.get_all_symbols():
                self.instrument_tokens[ins['tradingsymbol']] = ins['instrument_token']
        
        self.logger.info(f"Loaded {len(self.instrument_tokens)} instrument tokens")
    
    def fetch_historical_data(self, symbol: str, interval: str = "15minute") -> pd.DataFrame:
        """Fetch historical OHLCV data"""
        try:
            token = self.instrument_tokens.get(symbol)
            if not token:
                self.logger.warning(f"No token found for {symbol}")
                return pd.DataFrame()
            
            to_date = datetime.now()
            from_date = to_date - timedelta(days=self.config.historical_days)
            
            data = self.kite.historical_data(
                token, from_date, to_date, interval
            )
            
            if not data:
                return pd.DataFrame()
            
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
            
            # Store in cache
            self.historical_data[symbol] = df
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def start_live_stream(self, symbols: List[str], callback):
        """Start live data streaming"""
        tokens = [self.instrument_tokens[s] for s in symbols 
                 if s in self.instrument_tokens]
        
        def on_ticks(ws, ticks):
            for tick in ticks:
                symbol = self.get_symbol_from_token(tick['instrument_token'])
                if symbol:
                    self.live_data[symbol] = {
                        'ltp': tick['last_price'],
                        'volume': tick['volume_traded'],
                        'oi': tick['oi'],
                        'timestamp': datetime.now()
                    }
                    callback(symbol, tick)
        
        def on_connect(ws, response):
            self.logger.info("Live data stream connected")
            ws.subscribe(tokens)
            ws.set_mode(ws.MODE_FULL, tokens)
        
        self.ticker.on_ticks = on_ticks
        self.ticker.on_connect = on_connect
        self.ticker.connect(threaded=True)
    
    def get_symbol_from_token(self, token: int) -> Optional[str]:
        """Get symbol from instrument token"""
        for symbol, tok in self.instrument_tokens.items():
            if tok == token:
                return symbol
        return None
    
    def get_latest_data(self, symbol: str) -> Optional[Dict]:
        """Get latest live data for symbol"""
        return self.live_data.get(symbol)

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

class FeatureEngineer:
    """Creates comprehensive features for ML models"""
    
    @staticmethod
    def create_features(df: pd.DataFrame, lookback: int = 100) -> pd.DataFrame:
        """Create comprehensive feature set"""
        
        # Basic OHLC features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Open_Close_Ratio'] = df['Open'] / df['Close']
        
        # Volume features
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        df['Volume_ROC'] = df['Volume'].pct_change()
        df['Price_Volume_Corr'] = df['Close'].rolling(20).corr(df['Volume'])
        
        # Price action features
        df['Body_Size'] = abs(df['Close'] - df['Open']) / (df['High'] - df['Low'] + 1e-10)
        df['Upper_Shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / (df['High'] - df['Low'] + 1e-10)
        df['Lower_Shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / (df['High'] - df['Low'] + 1e-10)
        
        # Moving averages
        periods = [5, 8, 13, 21, 34, 50, 89, 144, 200]
        for period in periods:
            df[f'MA_{period}'] = df['Close'].rolling(period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        
        # Moving average crossovers
        df['MA_Cross_5_21'] = (df['MA_5'] > df['MA_21']).astype(int)
        df['MA_Cross_8_34'] = (df['MA_8'] > df['MA_34']).astype(int)
        df['MA_Cross_13_55'] = (df['MA_13'] > df['MA_55']).astype(int)
        
        # Bollinger Bands
        bb = ta.bbands(df['Close'], length=20, std=2)
        df['BB_Upper'] = bb['BBU_20_2.0']
        df['BB_Lower'] = bb['BBL_20_2.0']
        df['BB_Middle'] = bb['BBM_20_2.0']
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # RSI
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['RSI_SMA'] = df['RSI'].rolling(14).mean()
        
        # MACD
        macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_Signal'] = macd['MACD_12_26_9']
        df['MACD_Hist'] = macd['MACDh_12_26_9']
        
        # ATR and volatility
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['ATR_Pct'] = df['ATR'] / df['Close']
        df['Volatility'] = df['Returns'].rolling(20).std()
        
        # ADX for trend strength
        adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
        df['ADX'] = adx['ADX_14']
        df['DMI_Plus'] = adx['DMP_14']
        df['DMI_Minus'] = adx['DMN_14']
        
        # Stochastic
        stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3)
        df['Stoch_K'] = stoch['STOCHk_14_3_3']
        df['Stoch_D'] = stoch['STOCHd_14_3_3']
        
        # Volume indicators
        df['OBV'] = ta.obv(df['Close'], df['Volume'])
        df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)
        
        # Support and Resistance levels
        df = FeatureEngineer.add_support_resistance(df, lookback=50)
        
        # Market regime
        df['Market_Regime'] = FeatureEngineer.calculate_market_regime(df)
        
        # Statistical features
        df['Skewness'] = df['Returns'].rolling(20).skew()
        df['Kurtosis'] = df['Returns'].rolling(20).kurt()
        df['Z_Score'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()
        
        # Fourier transform for cycle detection
        df = FeatureEngineer.add_fourier_features(df)
        
        # Time-based features
        df['Hour'] = df.index.hour
        df['DayOfWeek'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Is_Month_End'] = df.index.is_month_end.astype(int)
        df['Is_Month_Start'] = df.index.is_month_start.astype(int)
        
        # Clean NaN values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    @staticmethod
    def add_support_resistance(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
        """Identify support and resistance levels"""
        
        df['Resistance'] = df['High'].rolling(lookback, center=True).max()
        df['Support'] = df['Low'].rolling(lookback, center=True).min()
        
        # Distance to support/resistance
        df['Dist_to_Resistance'] = (df['Resistance'] - df['Close']) / df['Close']
        df['Dist_to_Support'] = (df['Close'] - df['Support']) / df['Close']
        
        # Pivot points
        df['Pivot'] = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
        df['R1'] = 2 * df['Pivot'] - df['Low'].shift(1)
        df['S1'] = 2 * df['Pivot'] - df['High'].shift(1)
        
        return df
    
    @staticmethod
    def calculate_market_regime(df: pd.DataFrame) -> pd.Series:
        """Calculate market regime (0: ranging, 1: uptrend, 2: downtrend)"""
        
        # Use ADX for trend strength
        adx_threshold = 25
        
        # Use MA cross for direction
        ma_fast = df['Close'].rolling(20).mean()
        ma_slow = df['Close'].rolling(50).mean()
        
        regime = pd.Series(0, index=df.index)  # Default: ranging
        
        # Uptrend: ADX > threshold and fast MA > slow MA
        uptrend_condition = (df['ADX'] > adx_threshold) & (ma_fast > ma_slow)
        regime[uptrend_condition] = 1
        
        # Downtrend: ADX > threshold and fast MA < slow MA
        downtrend_condition = (df['ADX'] > adx_threshold) & (ma_fast < ma_slow)
        regime[downtrend_condition] = 2
        
        return regime
    
    @staticmethod
    def add_fourier_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add Fourier transform features for cycle detection"""
        
        prices = df['Close'].values
        
        # Perform FFT
        fft = np.fft.fft(prices)
        frequencies = np.fft.fftfreq(len(prices))
        
        # Get dominant frequencies
        magnitude = np.abs(fft)
        dominant_idx = np.argsort(magnitude)[-5:]  # Top 5 frequencies
        
        for i, idx in enumerate(dominant_idx[:3]):
            df[f'Fourier_Amp_{i+1}'] = magnitude[idx]
            df[f'Fourier_Freq_{i+1}'] = abs(frequencies[idx])
        
        return df

# ============================================================================
# AI MODEL MANAGER
# ============================================================================

class AIModelManager:
    """Manages multiple ML models for trading signals"""
    
    def __init__(self, config: TradingConfig, logger: TradingLogger):
        self.config = config
        self.logger = logger
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.load_or_train_models()
    
    def create_labels(self, df: pd.DataFrame, horizon: int = 5) -> pd.Series:
        """Create labels for supervised learning"""
        
        # Future returns
        future_returns = df['Close'].shift(-horizon) / df['Close'] - 1
        
        # Create ternary labels: 1 (buy), -1 (sell), 0 (hold)
        threshold = 0.005  # 0.5%
        
        labels = pd.Series(0, index=df.index)  # Default: hold
        
        # Buy if future return > threshold
        labels[future_returns > threshold] = 1
        
        # Sell if future return < -threshold
        labels[future_returns < -threshold] = -1
        
        # Remove last 'horizon' rows (no future data)
        labels = labels.iloc[:-horizon]
        
        return labels
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and labels for training"""
        
        # Create features
        feature_df = FeatureEngineer.create_features(df, self.config.lookback_period)
        
        # Create labels
        labels = self.create_labels(feature_df, self.config.prediction_horizon)
        
        # Align features and labels
        feature_df = feature_df.loc[labels.index]
        
        # Select features (exclude price and volume columns for training)
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'timestamp']
        feature_cols = [col for col in feature_df.columns if col not in exclude_cols]
        
        X = feature_df[feature_cols].values
        y = labels.values
        
        # Handle class imbalance with SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        return X_resampled, y_resampled, feature_cols
    
    def train_ensemble_model(self, symbol: str, df: pd.DataFrame):
        """Train ensemble model for a symbol"""
        
        self.logger.info(f"Training ensemble model for {symbol}")
        
        try:
            # Prepare data
            X, y, feature_cols = self.prepare_training_data(df)
            
            # Time series split for validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Define individual models
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
            
            gb_model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
            
            # Create ensemble
            ensemble = VotingClassifier(
                estimators=[
                    ('rf', rf_model),
                    ('xgb', xgb_model),
                    ('gb', gb_model)
                ],
                voting='soft',
                weights=[
                    self.config.ensemble_weight_rf,
                    self.config.ensemble_weight_xgb,
                    self.config.ensemble_weight_gb
                ]
            )
            
            # Train with cross-validation
            cv_scores = []
            feature_importances = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Scale features
                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                # Train ensemble
                ensemble.fit(X_train_scaled, y_train)
                
                # Predict and score
                y_pred = ensemble.predict(X_val_scaled)
                accuracy = accuracy_score(y_val, y_pred)
                cv_scores.append(accuracy)
                
                # Collect feature importance from Random Forest
                rf_importance = ensemble.named_estimators['rf'].feature_importances_
                feature_importances.append(rf_importance)
            
            # Train final model on all data
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            ensemble.fit(X_scaled, y)
            
            # Store model and scaler
            self.models[symbol] = ensemble
            self.scalers[symbol] = scaler
            
            # Calculate average feature importance
            avg_importance = np.mean(feature_importances, axis=0)
            self.feature_importance[symbol] = dict(zip(feature_cols, avg_importance))
            
            # Store performance metrics
            self.model_performance[symbol] = {
                'cv_mean_accuracy': np.mean(cv_scores),
                'cv_std_accuracy': np.std(cv_scores),
                'last_trained': datetime.now(),
                'training_samples': len(X)
            }
            
            # Save model
            self.save_model(symbol)
            
            self.logger.info(f"Model for {symbol} trained. CV Accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
            
        except Exception as e:
            self.logger.error(f"Error training model for {symbol}: {str(e)}")
    
    def predict(self, symbol: str, features: pd.DataFrame) -> Tuple[int, float]:
        """Make prediction for a symbol"""
        
        if symbol not in self.models:
            return 0, 0.0  # No model, return hold with 0 confidence
        
        try:
            # Get model and scaler
            model = self.models[symbol]
            scaler = self.scalers[symbol]
            
            # Select features
            feature_cols = list(self.feature_importance[symbol].keys())
            X = features[feature_cols].values.reshape(1, -1)
            
            # Scale features
            X_scaled = scaler.transform(X)
            
            # Predict
            prediction = model.predict(X_scaled)[0]
            probabilities = model.predict_proba(X_scaled)[0]
            
            # Calculate confidence
            confidence = np.max(probabilities)
            
            return prediction, confidence
            
        except Exception as e:
            self.logger.error(f"Error predicting for {symbol}: {str(e)}")
            return 0, 0.0
    
    def save_model(self, symbol: str):
        """Save model to disk"""
        
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, f"{symbol}_model.pkl")
        scaler_path = os.path.join(model_dir, f"{symbol}_scaler.pkl")
        
        joblib.dump(self.models[symbol], model_path)
        joblib.dump(self.scalers[symbol], scaler_path)
    
    def load_model(self, symbol: str):
        """Load model from disk"""
        
        model_path = f"models/{symbol}_model.pkl"
        scaler_path = f"models/{symbol}_scaler.pkl"
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.models[symbol] = joblib.load(model_path)
            self.scalers[symbol] = joblib.load(scaler_path)
            return True
        return False
    
    def load_or_train_models(self):
        """Load existing models or train new ones"""
        
        symbols = StockUniverse.get_all_symbols()
        
        for symbol in symbols:
            if self.load_model(symbol):
                self.logger.info(f"Loaded existing model for {symbol}")
            else:
                self.logger.info(f"No model found for {symbol}, will train when data is available")

# ============================================================================
# RISK MANAGER
# ============================================================================

class RiskManager:
    """Comprehensive risk management system"""
    
    def __init__(self, config: TradingConfig, logger: TradingLogger):
        self.config = config
        self.logger = logger
        self.daily_pnl = 0.0
        self.max_daily_loss = config.total_capital * config.max_portfolio_risk
        self.positions_risk = defaultdict(float)
        self.correlation_matrix = {}
        self.initialize_correlation_matrix()
    
    def initialize_correlation_matrix(self):
        """Initialize correlation matrix for portfolio risk"""
        # This would ideally be calculated from historical returns
        # For now, using sector-based correlations
        self.correlation_matrix = {
            'BANKING': {'BANKING': 1.0, 'IT': 0.3, 'AUTO': 0.2, 'FMCG': 0.1},
            'IT': {'BANKING': 0.3, 'IT': 1.0, 'AUTO': 0.1, 'FMCG': 0.2},
            'AUTO': {'BANKING': 0.2, 'IT': 0.1, 'AUTO': 1.0, 'FMCG': 0.3},
            'FMCG': {'BANKING': 0.1, 'IT': 0.2, 'AUTO': 0.3, 'FMCG': 1.0}
        }
    
    def calculate_position_size(self, symbol: str, entry_price: float, 
                               stop_loss: float, atr: float) -> Tuple[int, float]:
        """Calculate position size using Kelly Criterion"""
        
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss)
        
        # Estimate win rate and payoff ratio (would come from backtesting)
        win_rate = 0.55  # Conservative estimate
        avg_win = risk_per_share * 2.5  # 2.5:1 reward:risk
        avg_loss = risk_per_share
        
        # Kelly fraction
        if avg_win > 0:
            payoff_ratio = avg_win / avg_loss
            kelly_f = win_rate - ((1 - win_rate) / payoff_ratio)
            kelly_f = max(0.01, min(kelly_f, 0.05))  # Cap at 5% per trade
        else:
            kelly_f = 0.01  # Minimum 1%
        
        # Calculate position value
        position_value = self.config.total_capital * kelly_f
        
        # Calculate quantity
        quantity = int(position_value / entry_price)
        
        # Ensure minimum quantity
        min_quantity = max(1, int((self.config.total_capital * 0.001) / entry_price))
        quantity = max(min_quantity, quantity)
        
        # Calculate actual risk amount
        risk_amount = quantity * risk_per_share
        
        return quantity, risk_amount
    
    def check_portfolio_risk(self, new_risk: float) -> bool:
        """Check if new position exceeds portfolio risk limits"""
        
        total_risk = sum(self.positions_risk.values()) + new_risk
        
        # Check daily loss limit
        if self.daily_pnl + total_risk < -self.max_daily_loss:
            self.logger.warning("Daily loss limit would be exceeded")
            return False
        
        # Check concentration risk
        if new_risk > self.config.total_capital * self.config.risk_per_trade:
            self.logger.warning("Position exceeds single trade risk limit")
            return False
        
        return True
    
    def update_daily_pnl(self, pnl: float):
        """Update daily P&L"""
        self.daily_pnl += pnl
    
    def reset_daily(self):
        """Reset daily metrics"""
        self.daily_pnl = 0.0
        self.positions_risk.clear()
    
    def calculate_stop_loss(self, symbol: str, entry_price: float, 
                           direction: str, atr: float, df: pd.DataFrame) -> float:
        """Calculate dynamic stop loss"""
        
        if self.config.stop_loss_method == "atr":
            # ATR-based stop loss
            if direction == "LONG":
                stop_loss = entry_price - (atr * self.config.atr_multiplier)
            else:  # SHORT
                stop_loss = entry_price + (atr * self.config.atr_multiplier)
        
        elif self.config.stop_loss_method == "volatility":
            # Volatility-based stop loss
            volatility = df['Returns'].std() * np.sqrt(252)
            stop_distance = entry_price * volatility * 2
            
            if direction == "LONG":
                stop_loss = entry_price - stop_distance
            else:
                stop_loss = entry_price + stop_distance
        
        else:  # support_resistance
            # Support/resistance based
            if direction == "LONG":
                # Use recent swing low
                recent_low = df['Low'].rolling(20).min().iloc[-1]
                stop_loss = recent_low * 0.995  # 0.5% below support
            else:
                # Use recent swing high
                recent_high = df['High'].rolling(20).max().iloc[-1]
                stop_loss = recent_high * 1.005  # 0.5% above resistance
        
        return stop_loss
    
    def calculate_target(self, entry_price: float, stop_loss: float, 
                        direction: str) -> float:
        """Calculate target price based on risk-reward"""
        
        risk = abs(entry_price - stop_loss)
        reward = risk * 2.5  # 2.5:1 reward-risk ratio
        
        if direction == "LONG":
            target = entry_price + reward
        else:  # SHORT
            target = entry_price - reward
        
        return target

# ============================================================================
# TRADING ENGINE
# ============================================================================

class TradingEngine:
    """Main trading engine that coordinates everything"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = TradingLogger("TradingEngine")
        self.data_manager = DataManager(config, self.logger)
        self.ai_manager = AIModelManager(config, self.logger)
        self.risk_manager = RiskManager(config, self.logger)
        
        self.positions = {}
        self.pending_orders = []
        self.trade_history = []
        self.signals = {}
        
        self.market_phase = MarketPhase.PRE_OPEN
        self.is_running = False
        
        # Performance metrics
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0
        }
    
    def start(self):
        """Start the trading engine"""
        self.logger.info("Starting AI Trading Engine...")
        self.is_running = True
        
        # Start data collection
        symbols = StockUniverse.get_all_symbols()
        self.data_manager.start_live_stream(symbols[:50], self.on_tick)
        
        # Start main loop in separate thread
        self.trading_thread = threading.Thread(target=self.trading_loop)
        self.trading_thread.daemon = True
        self.trading_thread.start()
        
        # Schedule daily tasks
        self.schedule_tasks()
        
        self.logger.info("Trading Engine started successfully")
    
    def trading_loop(self):
        """Main trading loop"""
        
        while self.is_running:
            try:
                now = datetime.now(pytz.timezone("Asia/Kolkata"))
                current_time = now.time()
                
                # Update market phase
                self.update_market_phase(current_time)
                
                # Market open hours
                if self.config.market_open <= current_time <= self.config.market_close:
                    
                    # Scan for new signals
                    if now.second % 30 == 0:  # Every 30 seconds
                        self.scan_signals()
                    
                    # Manage existing positions
                    self.manage_positions()
                    
                    # Execute pending orders
                    self.execute_pending_orders()
                
                # Square off before market close
                elif current_time >= self.config.square_off_time:
                    self.square_off_all_positions("Market closing soon")
                
                # Wait for next iteration
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {str(e)}")
                time.sleep(5)
    
    def update_market_phase(self, current_time: dt_time):
        """Update current market phase"""
        
        if current_time < self.config.pre_open_start:
            self.market_phase = MarketPhase.PRE_OPEN
        elif current_time < self.config.market_open:
            self.market_phase = MarketPhase.OPENING
        elif current_time < dt_time(12, 0):
            self.market_phase = MarketPhase.MID_DAY
        elif current_time < self.config.square_off_time:
            self.market_phase = MarketPhase.CLOSING
        else:
            self.market_phase = MarketPhase.POST_CLOSE
    
    def scan_signals(self):
        """Scan for trading signals"""
        
        self.logger.info("Scanning for trading signals...")
        
        # Get active symbols (not already in positions)
        all_symbols = StockUniverse.get_all_symbols()
        active_symbols = [s for s in all_symbols if s not in self.positions]
        
        # Limit scanning to 20 symbols at a time for performance
        for symbol in active_symbols[:20]:
            try:
                # Fetch recent data
                df = self.data_manager.fetch_historical_data(symbol, "15minute")
                if df.empty or len(df) < 100:
                    continue
                
                # Create features
                feature_df = FeatureEngineer.create_features(df, self.config.lookback_period)
                latest_features = feature_df.iloc[-1]
                
                # Get AI prediction
                prediction, confidence = self.ai_manager.predict(symbol, feature_df)
                
                # Only act on high confidence signals
                if abs(prediction) == 1 and confidence >= self.config.min_confidence_threshold:
                    
                    # Get current price
                    live_data = self.data_manager.get_latest_data(symbol)
                    if not live_data:
                        continue
                    
                    current_price = live_data['ltp']
                    
                    # Calculate stop loss and target
                    atr = latest_features['ATR']
                    direction = "LONG" if prediction == 1 else "SHORT"
                    
                    stop_loss = self.risk_manager.calculate_stop_loss(
                        symbol, current_price, direction, atr, df
                    )
                    
                    target = self.risk_manager.calculate_target(
                        current_price, stop_loss, direction
                    )
                    
                    # Calculate position size
                    quantity, risk_amount = self.risk_manager.calculate_position_size(
                        symbol, current_price, stop_loss, atr
                    )
                    
                    # Check risk limits
                    if not self.risk_manager.check_portfolio_risk(risk_amount):
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
                    
                    # Store signal
                    self.signals[symbol] = signal
                    self.logger.info(
                        f"Signal generated: {direction} {symbol} @ {current_price:.2f} "
                        f"(Confidence: {confidence:.2%})"
                    )
                    
            except Exception as e:
                self.logger.error(f"Error scanning {symbol}: {str(e)}")
    
    def execute_signal(self, signal: Dict):
        """Execute a trading signal"""
        
        try:
            symbol = signal['symbol']
            
            # Check if already in position
            if symbol in self.positions:
                return
            
            # Place order (simulated for now - replace with actual Kite order)
            order_id = f"ORD_{int(time.time())}_{symbol}"
            
            # Create position
            position = Position(
                symbol=symbol,
                direction=signal['direction'],
                quantity=signal['quantity'],
                entry_price=signal['price'],
                entry_time=datetime.now(),
                stop_loss=signal['stop_loss'],
                target=signal['target'],
                atr=signal['atr'],
                risk_amount=signal['risk_amount']
            )
            
            # Add to positions
            self.positions[symbol] = position
            self.risk_manager.positions_risk[symbol] = signal['risk_amount']
            
            # Record trade
            trade_record = {
                'order_id': order_id,
                'symbol': symbol,
                'direction': signal['direction'],
                'quantity': signal['quantity'],
                'entry_price': signal['price'],
                'entry_time': datetime.now(),
                'stop_loss': signal['stop_loss'],
                'target': signal['target'],
                'confidence': signal['confidence']
            }
            
            self.trade_history.append(trade_record)
            self.metrics['total_trades'] += 1
            
            self.logger.info(
                f"Order executed: {signal['direction']} {symbol} "
                f"{signal['quantity']} shares @ {signal['price']:.2f}"
            )
            
            # Remove signal
            if symbol in self.signals:
                del self.signals[symbol]
                
        except Exception as e:
            self.logger.error(f"Error executing signal: {str(e)}")
    
    def manage_positions(self):
        """Manage existing positions"""
        
        for symbol, position in list(self.positions.items()):
            try:
                # Get current price
                live_data = self.data_manager.get_latest_data(symbol)
                if not live_data:
                    continue
                
                current_price = live_data['ltp']
                
                # Update position
                position.update(current_price)
                
                # Check exit conditions
                should_exit, reason = position.should_exit()
                
                if should_exit:
                    self.exit_position(symbol, current_price, reason)
                else:
                    # Check for trailing SL activation
                    if not position.is_trailing and self.config.trailing_sl_enabled:
                        if position.direction == "LONG":
                            rr_ratio = (current_price - position.entry_price) / abs(position.entry_price - position.stop_loss)
                            if rr_ratio >= self.config.trailing_sl_activation:
                                position.is_trailing = True
                                position.trailing_sl = position.entry_price
                                self.logger.info(f"Trailing SL activated for {symbol}")
                        else:  # SHORT
                            rr_ratio = (position.entry_price - current_price) / abs(position.entry_price - position.stop_loss)
                            if rr_ratio >= self.config.trailing_sl_activation:
                                position.is_trailing = True
                                position.trailing_sl = position.entry_price
                                self.logger.info(f"Trailing SL activated for {symbol}")
                
            except Exception as e:
                self.logger.error(f"Error managing position {symbol}: {str(e)}")
    
    def exit_position(self, symbol: str, exit_price: float, reason: str):
        """Exit a position"""
        
        try:
            position = self.positions.get(symbol)
            if not position:
                return
            
            # Calculate final P&L
            if position.direction == "LONG":
                pnl = (exit_price - position.entry_price) * position.quantity
            else:  # SHORT
                pnl = (position.entry_price - exit_price) * position.quantity
            
            # Update metrics
            self.risk_manager.update_daily_pnl(pnl)
            self.metrics['total_pnl'] += pnl
            
            if pnl > 0:
                self.metrics['winning_trades'] += 1
            else:
                self.metrics['losing_trades'] += 1
            
            # Update win rate
            self.metrics['win_rate'] = (
                self.metrics['winning_trades'] / self.metrics['total_trades']
                if self.metrics['total_trades'] > 0 else 0
            )
            
            # Remove position
            del self.positions[symbol]
            if symbol in self.risk_manager.positions_risk:
                del self.risk_manager.positions_risk[symbol]
            
            # Log exit
            self.logger.info(
                f"Position closed: {symbol} @ {exit_price:.2f} "
                f"(P&L: {pnl:.2f}, Reason: {reason})"
            )
            
            # Record exit in trade history
            for trade in reversed(self.trade_history):
                if trade['symbol'] == symbol and 'exit_price' not in trade:
                    trade['exit_price'] = exit_price
                    trade['exit_time'] = datetime.now()
                    trade['pnl'] = pnl
                    trade['exit_reason'] = reason
                    break
            
        except Exception as e:
            self.logger.error(f"Error exiting position {symbol}: {str(e)}")
    
    def square_off_all_positions(self, reason: str):
        """Square off all positions"""
        
        if not self.positions:
            return
        
        self.logger.warning(f"Squaring off all positions: {reason}")
        
        for symbol in list(self.positions.keys()):
            try:
                live_data = self.data_manager.get_latest_data(symbol)
                if live_data:
                    self.exit_position(symbol, live_data['ltp'], reason)
            except Exception as e:
                self.logger.error(f"Error squaring off {symbol}: {str(e)}")
    
    def execute_pending_orders(self):
        """Execute pending orders"""
        
        # For now, execute all signals immediately
        # In production, you might want to add additional filters
        
        for symbol, signal in list(self.signals.items()):
            if len(self.positions) < self.config.max_positions:
                self.execute_signal(signal)
    
    def on_tick(self, symbol: str, tick: Dict):
        """Handle incoming tick data"""
        
        # Update live data
        self.data_manager.live_data[symbol] = {
            'ltp': tick['last_price'],
            'volume': tick['volume_traded'],
            'oi': tick.get('oi', 0),
            'timestamp': datetime.now()
        }
    
    def schedule_tasks(self):
        """Schedule recurring tasks"""
        
        # Retrain models daily
        schedule.every().day.at("18:00").do(self.retrain_models)
        
        # Reset daily metrics
        schedule.every().day.at("09:00").do(self.risk_manager.reset_daily)
        
        # Start scheduler in separate thread
        def run_scheduler():
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)
        
        scheduler_thread = threading.Thread(target=run_scheduler)
        scheduler_thread.daemon = True
        scheduler_thread.start()
    
    def retrain_models(self):
        """Retrain AI models"""
        
        self.logger.info("Starting model retraining...")
        
        symbols = StockUniverse.get_all_symbols()[:20]  # Train on first 20 symbols
        
        for symbol in symbols:
            try:
                df = self.data_manager.fetch_historical_data(symbol, "15minute")
                if not df.empty and len(df) > 200:
                    self.ai_manager.train_ensemble_model(symbol, df)
            except Exception as e:
                self.logger.error(f"Error retraining model for {symbol}: {str(e)}")
        
        self.logger.info("Model retraining completed")
    
    def stop(self):
        """Stop the trading engine"""
        
        self.logger.info("Stopping Trading Engine...")
        self.is_running = False
        
        # Square off all positions
        self.square_off_all_positions("System shutdown")
        
        # Disconnect from Kite
        if self.data_manager.ticker:
            self.data_manager.ticker.close()
        
        self.logger.info("Trading Engine stopped")

# ============================================================================
# DASHBOARD
# ============================================================================

class TradingDashboard:
    """Interactive dashboard for monitoring and control"""
    
    def __init__(self, trading_engine: TradingEngine):
        self.engine = trading_engine
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.DARKLY],
            meta_tags=[{'name': 'viewport', 
                       'content': 'width=device-width, initial-scale=1.0'}]
        )
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Setup dashboard layout"""
        
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("AI Trading System Dashboard", 
                           className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            # System Status Row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("System Status"),
                        dbc.CardBody([
                            html.H4("🟢 RUNNING", id="system-status", 
                                   className="text-success"),
                            html.P("Market Phase: ", id="market-phase"),
                            html.P("Positions: ", id="active-positions"),
                            html.P("Signals: ", id="active-signals"),
                            dbc.Button("Emergency Stop", id="stop-btn", 
                                      color="danger", n_clicks=0)
                        ])
                    ])
                ], width=3),
                
                # Performance Metrics
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Performance Metrics"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.H6("Total P&L"),
                                    html.H3("₹0.00", id="total-pnl", 
                                           className="text-success")
                                ]),
                                dbc.Col([
                                    html.H6("Win Rate"),
                                    html.H3("0%", id="win-rate")
                                ]),
                                dbc.Col([
                                    html.H6("Total Trades"),
                                    html.H3("0", id="total-trades")
                                ]),
                                dbc.Col([
                                    html.H6("Daily P&L"),
                                    html.H3("₹0.00", id="daily-pnl")
                                ])
                            ])
                        ])
                    ])
                ], width=9)
            ], className="mb-4"),
            
            # Tabs
            dbc.Tabs([
                # Positions Tab
                dbc.Tab([
                    dbc.Card([
                        dbc.CardBody([
                            dash_table.DataTable(
                                id='positions-table',
                                columns=[
                                    {'name': 'Symbol', 'id': 'symbol'},
                                    {'name': 'Direction', 'id': 'direction'},
                                    {'name': 'Qty', 'id': 'quantity'},
                                    {'name': 'Entry', 'id': 'entry_price'},
                                    {'name': 'Current', 'id': 'current_price'},
                                    {'name': 'P&L', 'id': 'pnl'},
                                    {'name': 'P&L %', 'id': 'pnl_percentage'},
                                    {'name': 'SL', 'id': 'stop_loss'},
                                    {'name': 'Target', 'id': 'target'}
                                ],
                                style_table={'overflowX': 'auto'},
                                style_cell={'textAlign': 'center'},
                                style_header={'fontWeight': 'bold'}
                            )
                        ])
                    ])
                ], label='Positions', tab_id='tab-positions'),
                
                # Signals Tab
                dbc.Tab([
                    dbc.Card([
                        dbc.CardBody([
                            dash_table.DataTable(
                                id='signals-table',
                                columns=[
                                    {'name': 'Symbol', 'id': 'symbol'},
                                    {'name': 'Direction', 'id': 'direction'},
                                    {'name': 'Price', 'id': 'price'},
                                    {'name': 'Confidence', 'id': 'confidence'},
                                    {'name': 'Stop Loss', 'id': 'stop_loss'},
                                    {'name': 'Target', 'id': 'target'},
                                    {'name': 'Time', 'id': 'timestamp'}
                                ],
                                style_table={'overflowX': 'auto'},
                                style_cell={'textAlign': 'center'}
                            )
                        ])
                    ])
                ], label='Signals', tab_id='tab-signals'),
                
                # Trade History Tab
                dbc.Tab([
                    dbc.Card([
                        dbc.CardBody([
                            dash_table.DataTable(
                                id='history-table',
                                columns=[
                                    {'name': 'Time', 'id': 'entry_time'},
                                    {'name': 'Symbol', 'id': 'symbol'},
                                    {'name': 'Direction', 'id': 'direction'},
                                    {'name': 'Entry', 'id': 'entry_price'},
                                    {'name': 'Exit', 'id': 'exit_price'},
                                    {'name': 'P&L', 'id': 'pnl'},
                                    {'name': 'Reason', 'id': 'exit_reason'}
                                ],
                                style_table={'overflowX': 'auto'},
                                style_cell={'textAlign': 'center'},
                                page_size=10
                            )
                        ])
                    ])
                ], label='Trade History', tab_id='tab-history'),
                
                # Charts Tab
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Select Symbol"),
                                dbc.CardBody([
                                    dcc.Dropdown(
                                        id='symbol-selector',
                                        options=[{'label': s, 'value': s} 
                                                for s in StockUniverse.get_all_symbols()[:20]],
                                        value='RELIANCE'
                                    )
                                ])
                            ])
                        ], width=3),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Price Chart"),
                                dbc.CardBody([
                                    dcc.Graph(id='price-chart')
                                ])
                            ])
                        ], width=9)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Model Performance"),
                                dbc.CardBody([
                                    dcc.Graph(id='model-performance')
                                ])
                            ])
                        ])
                    ], className="mt-4")
                ], label='Charts', tab_id='tab-charts'),
                
                # Risk Management Tab
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Portfolio Risk"),
                                dbc.CardBody([
                                    dcc.Graph(id='risk-metrics')
                                ])
                            ])
                        ], width=6),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Risk Limits"),
                                dbc.CardBody([
                                    html.Ul([
                                        html.Li(f"Max Daily Loss: ₹{self.engine.config.total_capital * self.engine.config.max_portfolio_risk:,.0f}"),
                                        html.Li(f"Risk per Trade: {self.engine.config.risk_per_trade*100}%"),
                                        html.Li(f"Max Positions: {self.engine.config.max_positions}"),
                                        html.Li(f"Max Drawdown: {self.engine.config.max_drawdown_limit*100}%")
                                    ])
                                ])
                            ])
                        ], width=6)
                    ])
                ], label='Risk Management', tab_id='tab-risk')
            ]),
            
            # Update Interval
            dcc.Interval(
                id='interval-component',
                interval=5000,  # 5 seconds
                n_intervals=0
            )
        ], fluid=True)
    
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            [Output('system-status', 'children'),
             Output('market-phase', 'children'),
             Output('active-positions', 'children'),
             Output('active-signals', 'children'),
             Output('total-pnl', 'children'),
             Output('win-rate', 'children'),
             Output('total-trades', 'children'),
             Output('daily-pnl', 'children'),
             Output('positions-table', 'data'),
             Output('signals-table', 'data'),
             Output('history-table', 'data')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            """Update dashboard data"""
            
            # System status
            status = "🟢 RUNNING" if self.engine.is_running else "🔴 STOPPED"
            market_phase = f"Market Phase: {self.engine.market_phase.value.upper()}"
            positions_count = f"Positions: {len(self.engine.positions)}"
            signals_count = f"Signals: {len(self.engine.signals)}"
            
            # Performance metrics
            total_pnl = f"₹{self.engine.metrics['total_pnl']:,.2f}"
            win_rate = f"{self.engine.metrics['win_rate']*100:.1f}%"
            total_trades = str(self.engine.metrics['total_trades'])
            daily_pnl = f"₹{self.engine.risk_manager.daily_pnl:,.2f}"
            
            # Positions table data
            positions_data = []
            for symbol, pos in self.engine.positions.items():
                positions_data.append({
                    'symbol': symbol,
                    'direction': pos.direction,
                    'quantity': pos.quantity,
                    'entry_price': f"₹{pos.entry_price:.2f}",
                    'current_price': f"₹{pos.current_price:.2f}",
                    'pnl': f"₹{pos.pnl:,.2f}",
                    'pnl_percentage': f"{pos.pnl_percentage:.2f}%",
                    'stop_loss': f"₹{pos.stop_loss:.2f}",
                    'target': f"₹{pos.target:.2f}"
                })
            
            # Signals table data
            signals_data = []
            for symbol, signal in self.engine.signals.items():
                signals_data.append({
                    'symbol': symbol,
                    'direction': signal['direction'],
                    'price': f"₹{signal['price']:.2f}",
                    'confidence': f"{signal['confidence']:.1%}",
                    'stop_loss': f"₹{signal['stop_loss']:.2f}",
                    'target': f"₹{signal['target']:.2f}",
                    'timestamp': signal['timestamp'].strftime("%H:%M:%S")
                })
            
            # History table data
            history_data = []
            for trade in self.engine.trade_history[-20:]:  # Last 20 trades
                if 'exit_price' in trade:
                    history_data.append({
                        'entry_time': trade['entry_time'].strftime("%H:%M"),
                        'symbol': trade['symbol'],
                        'direction': trade['direction'],
                        'entry_price': f"₹{trade['entry_price']:.2f}",
                        'exit_price': f"₹{trade.get('exit_price', 0):.2f}",
                        'pnl': f"₹{trade.get('pnl', 0):,.2f}",
                        'exit_reason': trade.get('exit_reason', '')
                    })
            
            return (status, market_phase, positions_count, signals_count,
                   total_pnl, win_rate, total_trades, daily_pnl,
                   positions_data, signals_data, history_data)
        
        @self.app.callback(
            Output('price-chart', 'figure'),
            [Input('symbol-selector', 'value'),
             Input('interval-component', 'n_intervals')]
        )
        def update_chart(symbol, n):
            """Update price chart"""
            
            try:
                # Fetch data
                df = self.engine.data_manager.fetch_historical_data(symbol, "15minute")
                if df.empty:
                    return go.Figure()
                
                # Create candlestick chart
                fig = make_subplots(rows=2, cols=1, 
                                   shared_xaxes=True,
                                   vertical_spacing=0.05,
                                   row_heights=[0.7, 0.3])
                
                # Candlesticks
                fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='OHLC'
                ), row=1, col=1)
                
                # Add moving averages
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['Close'].rolling(50).mean(),
                    name='MA 50',
                    line=dict(color='orange', width=1)
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['Close'].rolling(200).mean(),
                    name='MA 200',
                    line=dict(color='red', width=1)
                ), row=1, col=1)
                
                # Volume
                colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] 
                         else 'green' for i in range(len(df))]
                
                fig.add_trace(go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name='Volume',
                    marker_color=colors
                ), row=2, col=1)
                
                # Update layout
                fig.update_layout(
                    title=f"{symbol} - Price Chart",
                    template='plotly_dark',
                    showlegend=True,
                    height=600,
                    xaxis_rangeslider_visible=False
                )
                
                fig.update_xaxes(title_text="Time", row=2, col=1)
                fig.update_yaxes(title_text="Price", row=1, col=1)
                fig.update_yaxes(title_text="Volume", row=2, col=1)
                
                return fig
                
            except Exception as e:
                print(f"Error updating chart: {e}")
                return go.Figure()
        
        @self.app.callback(
            Output('model-performance', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_model_performance(n):
            """Update model performance chart"""
            
            try:
                # Get model performances
                symbols = list(self.engine.ai_manager.model_performance.keys())[:10]
                accuracies = [self.engine.ai_manager.model_performance[s]['cv_mean_accuracy'] 
                            for s in symbols]
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=symbols,
                        y=accuracies,
                        text=[f"{a:.1%}" for a in accuracies],
                        textposition='auto',
                        marker_color='lightblue'
                    )
                ])
                
                fig.update_layout(
                    title="Model Performance (Cross-Validation Accuracy)",
                    template='plotly_dark',
                    yaxis_title="Accuracy",
                    yaxis_tickformat='.0%'
                )
                
                return fig
                
            except Exception:
                return go.Figure()
        
        @self.app.callback(
            Output('risk-metrics', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_risk_metrics(n):
            """Update risk metrics chart"""
            
            try:
                # Create pie chart of position risks
                symbols = list(self.engine.positions.keys())
                risks = [pos.risk_amount for pos in self.engine.positions.values()]
                
                if not risks:
                    risks = [0]
                    symbols = ['No Positions']
                
                fig = go.Figure(data=[
                    go.Pie(
                        labels=symbols,
                        values=risks,
                        hole=0.3,
                        textinfo='label+percent'
                    )
                ])
                
                fig.update_layout(
                    title="Portfolio Risk Distribution",
                    template='plotly_dark'
                )
                
                return fig
                
            except Exception:
                return go.Figure()
        
        @self.app.callback(
            Output('stop-btn', 'children'),
            [Input('stop-btn', 'n_clicks')]
        )
        def emergency_stop(n_clicks):
            """Emergency stop button"""
            
            if n_clicks > 0:
                self.engine.stop()
                return "🛑 SYSTEM STOPPED"
            
            return "Emergency Stop"
    
    def run(self):
        """Run the dashboard"""
        
        self.app.run_server(
            port=self.engine.config.dashboard_port,
            debug=False,
            host='0.0.0.0'
        )

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("\n" + "="*60)
    print("INSTITUTIONAL AI TRADING SYSTEM - NIFTY 50 & MIDCAP")
    print("="*60 + "\n")
    
    # Configuration
    config = TradingConfig()
    
    # Initialize trading engine
    engine = TradingEngine(config)
    
    # Initialize dashboard
    dashboard = TradingDashboard(engine)
    
    # Start in separate threads
    engine_thread = threading.Thread(target=engine.start)
    dashboard_thread = threading.Thread(target=dashboard.run)
    
    engine_thread.daemon = True
    dashboard_thread.daemon = True
    
    engine_thread.start()
    
    print("\n✅ Trading Engine started successfully!")
    print(f"📊 Dashboard available at: http://localhost:{config.dashboard_port}")
    print("\nPress Ctrl+C to stop the system...\n")
    
    try:
        dashboard_thread.start()
        dashboard_thread.join()
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down system...")
        engine.stop()
        print("✅ System shutdown complete.")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        engine.stop()

if __name__ == "__main__":
    main()
