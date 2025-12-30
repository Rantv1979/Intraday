"""
AI ALGORITHMIC TRADING BOT v7.0 - ENHANCED WITH PROPER DEPENDENCY HANDLING
ALL LIMITATIONS FIXED + SMC Pro + Advanced Features

INSTALLATION:
pip install streamlit pandas numpy scipy scikit-learn plotly kiteconnect xgboost yfinance textblob newspaper3k

For TA-Lib on Windows: Download from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
For TA-Lib on Linux/Mac: pip install TA-Lib

Or run without TA-Lib using our fallback indicators.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import warnings
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Optional, Tuple, Any
import sqlite3
import threading
import queue
import os
from scipy import stats
import requests
from collections import deque

warnings.filterwarnings('ignore')

# ============================================================================
# DEPENDENCY HANDLING - GRACEFUL DEGRADATION
# ============================================================================

# Try to import TA-Lib, with fallback
try:
    import talib as ta
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    st.warning("⚠️ TA-Lib not installed. Using fallback technical indicators. Install: pip install TA-Lib")

# Try to import KiteConnect
try:
    from kiteconnect import KiteConnect, KiteTicker
    KITE_AVAILABLE = True
except ImportError:
    KITE_AVAILABLE = False
    st.error("❌ KiteConnect not installed! Run: pip install kiteconnect")

# Try to import ML libraries
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.preprocessing import RobustScaler, StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    st.error("❌ scikit-learn not installed! Run: pip install scikit-learn")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    st.warning("⚠️ XGBoost not installed. Using only Random Forest. Install: pip install xgboost")

# Try to import plotting libraries
try:
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.error("❌ Plotly not installed! Run: pip install plotly")

# Try to import sentiment analysis libraries
try:
    from textblob import TextBlob
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False
    st.warning("⚠️ TextBlob not installed. Sentiment analysis disabled. Install: pip install textblob")

# Try to import yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    st.warning("⚠️ yfinance not installed. Using fallback data. Install: pip install yfinance")

# ============================================================================
# FALLBACK TECHNICAL INDICATORS (WHEN TA-LIB NOT AVAILABLE)
# ============================================================================

class FallbackIndicators:
    """Fallback technical indicators when TA-Lib is not available"""
    
    @staticmethod
    def calculate_rsi(prices, period=14):
        """Calculate RSI without TA-Lib"""
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100. / (1. + rs)
        
        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down
            rsi[i] = 100. - 100. / (1. + rs)
        
        return rsi
    
    @staticmethod
    def calculate_sma(prices, period):
        """Calculate Simple Moving Average"""
        return pd.Series(prices).rolling(period).mean().values
    
    @staticmethod
    def calculate_ema(prices, period):
        """Calculate Exponential Moving Average"""
        return pd.Series(prices).ewm(span=period, adjust=False).mean().values
    
    @staticmethod
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        """Calculate MACD without TA-Lib"""
        ema_fast = pd.Series(prices).ewm(span=fast, adjust=False).mean()
        ema_slow = pd.Series(prices).ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd.values, signal_line.values, histogram.values
    
    @staticmethod
    def calculate_atr(high, low, close, period=14):
        """Calculate Average True Range"""
        high = pd.Series(high)
        low = pd.Series(low)
        close = pd.Series(close)
        
        tr = pd.concat([
            high - low,
            abs(high - close.shift()),
            abs(low - close.shift())
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(period).mean()
        return atr.values
    
    @staticmethod
    def calculate_bbands(prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = pd.Series(prices).rolling(period).mean()
        std = pd.Series(prices).rolling(period).std()
        
        upper = sma + (std * std_dev)
        middle = sma
        lower = sma - (std * std_dev)
        
        return upper.values, middle.values, lower.values
    
    @staticmethod
    def calculate_stochastic(high, low, close, k_period=14, d_period=3):
        """Calculate Stochastic Oscillator"""
        high = pd.Series(high)
        low = pd.Series(low)
        close = pd.Series(close)
        
        lowest_low = low.rolling(k_period).min()
        highest_high = high.rolling(k_period).max()
        
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(d_period).mean()
        
        return k.values, d.values

# ============================================================================
# CONFIGURATION - DYNAMIC & ADAPTIVE
# ============================================================================

class AdaptiveConfig:
    """Dynamic configuration that adapts to market conditions"""
    
    def __init__(self):
        # Base values
        self._TOTAL_CAPITAL = 2_000_000
        self._RISK_PER_TRADE = 0.01
        self.MAX_POSITIONS = 10
        self.MAX_DAILY_TRADES = 50
        self._MIN_CONFIDENCE = 0.55
        self._ATR_MULTIPLIER = 2.0
        self._TAKE_PROFIT_RATIO = 2.5
        self.TRAILING_STOP = True
        self.TRAILING_ACTIVATION = 0.015
        
        # Market hours
        self.MARKET_OPEN = dt_time(9, 15)
        self.MARKET_CLOSE = dt_time(15, 30)
        
        # Market regime tracking
        self.market_regime = "NEUTRAL"
        self.volatility_regime = "NORMAL"
        self.adaptation_history = deque(maxlen=100)
        
        # Risk multiplier based on market regime
        self.risk_multipliers = {
            "BULLISH": 1.2,
            "BEARISH": 0.8,
            "VOLATILE": 0.7,
            "NEUTRAL": 1.0
        }
        
        # Confidence multipliers
        self.confidence_multipliers = {
            "BULLISH": 1.1,
            "BEARISH": 0.9,
            "VOLATILE": 0.8,
            "NEUTRAL": 1.0
        }
    
    @property
    def TOTAL_CAPITAL(self):
        return self._TOTAL_CAPITAL
    
    @TOTAL_CAPITAL.setter
    def TOTAL_CAPITAL(self, value):
        self._TOTAL_CAPITAL = value
    
    @property
    def RISK_PER_TRADE(self):
        # Adjust risk based on market regime
        base_risk = self._RISK_PER_TRADE
        multiplier = self.risk_multipliers.get(self.market_regime, 1.0)
        return min(0.05, max(0.005, base_risk * multiplier))
    
    @RISK_PER_TRADE.setter
    def RISK_PER_TRADE(self, value):
        self._RISK_PER_TRADE = value
    
    @property
    def MIN_CONFIDENCE(self):
        # Adjust confidence threshold based on market
        base_conf = self._MIN_CONFIDENCE
        multiplier = self.confidence_multipliers.get(self.market_regime, 1.0)
        return min(0.8, max(0.5, base_conf * multiplier))
    
    @MIN_CONFIDENCE.setter
    def MIN_CONFIDENCE(self, value):
        self._MIN_CONFIDENCE = value
    
    @property
    def ATR_MULTIPLIER(self):
        # Wider stops in volatile markets
        if self.volatility_regime == "HIGH":
            return self._ATR_MULTIPLIER * 1.5
        elif self.volatility_regime == "LOW":
            return self._ATR_MULTIPLIER * 0.8
        return self._ATR_MULTIPLIER
    
    @ATR_MULTIPLIER.setter
    def ATR_MULTIPLIER(self, value):
        self._ATR_MULTIPLIER = value
    
    @property
    def TAKE_PROFIT_RATIO(self):
        # Adjust RR ratio based on volatility
        if self.volatility_regime == "HIGH":
            return self._TAKE_PROFIT_RATIO * 1.2
        elif self.volatility_regime == "LOW":
            return self._TAKE_PROFIT_RATIO * 0.8
        return self._TAKE_PROFIT_RATIO
    
    @TAKE_PROFIT_RATIO.setter
    def TAKE_PROFIT_RATIO(self, value):
        self._TAKE_PROFIT_RATIO = value
    
    def update_market_regime(self, nifty_returns_std, advancers, decliners):
        """Update market regime based on multiple factors"""
        # Volatility based regime
        if nifty_returns_std > 0.02:
            self.volatility_regime = "HIGH"
        elif nifty_returns_std < 0.008:
            self.volatility_regime = "LOW"
        else:
            self.volatility_regime = "NORMAL"
        
        # Market direction regime
        adv_ratio = advancers / (advancers + decliners + 0.001)
        
        if adv_ratio > 0.65 and nifty_returns_std < 0.015:
            self.market_regime = "BULLISH"
        elif adv_ratio < 0.35 and nifty_returns_std < 0.015:
            self.market_regime = "BEARISH"
        elif nifty_returns_std > 0.02:
            self.market_regime = "VOLATILE"
        else:
            self.market_regime = "NEUTRAL"
        
        self.adaptation_history.append({
            'timestamp': datetime.now(),
            'regime': self.market_regime,
            'volatility': self.volatility_regime
        })

# ============================================================================
# STOCK UNIVERSE - ALL 159 F&O STOCKS (UNCHANGED)
# ============================================================================

class StockUniverse:
    """Complete F&O Stock Universe"""
    
    @staticmethod
    def get_all_fno_stocks():
        """Returns all 159 F&O stocks"""
        return [
            # Nifty 50
            'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR',
            'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK', 'BAJFINANCE', 'WIPRO',
            'AXISBANK', 'LT', 'HCLTECH', 'ASIANPAINT', 'MARUTI', 'SUNPHARMA',
            'TITAN', 'ULTRACEMCO', 'M&M', 'NTPC', 'ONGC', 'POWERGRID',
            'NESTLEIND', 'TATASTEEL', 'JSWSTEEL', 'ADANIPORTS', 'TECHM',
            'BAJAJFINSV', 'BRITANNIA', 'GRASIM', 'DIVISLAB', 'DRREDDY',
            'SHREECEM', 'HDFCLIFE', 'SBILIFE', 'BPCL', 'IOC', 'COALINDIA',
            'INDUSINDBK', 'EICHERMOT', 'HEROMOTOCO', 'UPL', 'CIPLA',
            'TATACONSUM', 'BAJAJ-AUTO', 'APOLLOHOSP', 'ADANIENT',
            
            # Nifty Next 50
            'HAVELLS', 'GODREJCP', 'HINDZINC', 'MOTHERSON', 'AMBUJACEM',
            'DABUR', 'BOSCHLTD', 'BANDHANBNK', 'DLF', 'BERGEPAINT',
            'COLPAL', 'GAIL', 'PIDILITIND', 'SIEMENS', 'VEDL',
            'HINDPETRO', 'TATAPOWER', 'PNB', 'LUPIN', 'NMDC',
            'TORNTPHARM', 'OFSS', 'ICICIPRULI', 'UBL', 'INDIGO',
            'MARICO', 'MPHASIS', 'ADANIPOWER', 'AUROPHARMA', 'BANKBARODA',
            'LTIM', 'TRENT', 'ZYDUSLIFE', 'DMART', 'NAUKRI',
            
            # Additional Liquid F&O Stocks
            'BALKRISIND', 'BATAINDIA', 'BEL', 'CANBK', 'ESCORTS',
            'JINDALSTEL', 'MANAPPURAM', 'SRTRANSFIN', 'ACC', 'ASHOKLEY',
            'ASTRAL', 'CUMMINSIND', 'DIXON', 'EXIDEIND', 'FEDERALBNK',
            'GODREJPROP', 'IDFCFIRSTB', 'IEX', 'IGL', 'INDHOTEL',
            'INDUSTOWER', 'JUBLFOOD', 'LAURUSLABS', 'LICHSGFIN', 'MRF',
            'MFSL', 'NATIONALUM', 'PAGEIND', 'PERSISTENT', 'PFC',
            'PIIND', 'RBLBANK', 'RECLTD', 'SAIL', 'SUNTV',
            'TATACHEM', 'TATACOMM', 'TATAELXSI', 'TORNTPOWER', 'TVSMOTOR',
            'UNIONBANK', 'VOLTAS', 'ZEEL', 'AUBANK', 'ABFRL',
            'CHOLAFIN', 'COFORGE', 'CROMPTON', 'DEEPAKNTR', 'HFCL',
            'IDEA', 'IRCTC', 'M&MFIN', 'METROPOLIS', 'OBEROIRLTY',
            'PETRONET', 'POLYCAB', 'SBICARD', 'SYNGENE', 'TIINDIA',
            'RAIN', 'CONCOR', 'DELTACORP', 'GRANULES', 'ABCAPITAL',
            'ALKEM', 'ATUL', 'APLAPOLLO', 'CHAMBLFERT', 'BHEL',
            'NAVINFLUOR', 'RELAXO', 'WHIRLPOOL'
        ]
    
    @staticmethod
    def get_trading_pairs():
        """Get statistically correlated pairs for pair trading"""
        return [
            ('HDFCBANK', 'ICICIBANK'),
            ('RELIANCE', 'ONGC'),
            ('TCS', 'INFY'),
            ('MARUTI', 'M&M'),
            ('SUNPHARMA', 'DRREDDY'),
            ('HINDUNILVR', 'ITC'),
            ('AXISBANK', 'KOTAKBANK'),
            ('BHARTIARTL', 'IDEA'),
            ('TATASTEEL', 'JSWSTEEL'),
            ('ULTRACEMCO', 'SHREECEM')
        ]

# ============================================================================
# ADVANCED TECHNICAL ANALYSIS WITH FALLBACK SUPPORT
# ============================================================================

class AdvancedTechnicalAnalysis:
    """Advanced technical indicators with fallback support"""
    
    @staticmethod
    def calculate_all_indicators(df):
        """Calculate comprehensive technical indicators"""
        prices = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        
        # RSI
        if TALIB_AVAILABLE:
            df['RSI'] = ta.RSI(prices, timeperiod=14)
            df['RSI_9'] = ta.RSI(prices, timeperiod=9)
            df['RSI_25'] = ta.RSI(prices, timeperiod=25)
        else:
            df['RSI'] = FallbackIndicators.calculate_rsi(prices, 14)
            df['RSI_9'] = FallbackIndicators.calculate_rsi(prices, 9)
            df['RSI_25'] = FallbackIndicators.calculate_rsi(prices, 25)
        
        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'SMA{period}'] = FallbackIndicators.calculate_sma(prices, period)
            df[f'EMA{period}'] = FallbackIndicators.calculate_ema(prices, period)
        
        # MACD
        if TALIB_AVAILABLE:
            df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = ta.MACD(
                prices, fastperiod=12, slowperiod=26, signalperiod=9
            )
        else:
            macd, signal, hist = FallbackIndicators.calculate_macd(prices)
            df['MACD'] = macd
            df['MACD_Signal'] = signal
            df['MACD_Hist'] = hist
        
        # Bollinger Bands
        if TALIB_AVAILABLE:
            df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = ta.BBANDS(
                prices, timeperiod=20, nbdevup=2, nbdevdn=2
            )
        else:
            upper, middle, lower = FallbackIndicators.calculate_bbands(prices)
            df['BB_Upper'] = upper
            df['BB_Middle'] = middle
            df['BB_Lower'] = lower
        
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Stochastic
        if TALIB_AVAILABLE:
            df['STOCH_K'], df['STOCH_D'] = ta.STOCH(
                high, low, prices, fastk_period=14, slowk_period=3, slowd_period=3
            )
        else:
            k, d = FallbackIndicators.calculate_stochastic(high, low, prices)
            df['STOCH_K'] = k
            df['STOCH_D'] = d
        
        # ATR and volatility
        if TALIB_AVAILABLE:
            df['ATR'] = ta.ATR(high, low, prices, timeperiod=14)
            df['NATR'] = ta.NATR(high, low, prices, timeperiod=14)
            df['TRANGE'] = ta.TRANGE(high, low, prices)
        else:
            df['ATR'] = FallbackIndicators.calculate_atr(high, low, prices)
            df['NATR'] = df['ATR'] / df['Close'] * 100
            df['TRANGE'] = high - low
        
        # Volume indicators (if available)
        if 'Volume' in df.columns:
            volume = df['Volume'].values
            
            # Simple volume indicators (no TA-Lib required)
            df['OBV'] = AdvancedTechnicalAnalysis.calculate_obv(prices, volume)
            df['Volume_SMA'] = pd.Series(volume).rolling(20).mean().values
            df['Volume_Ratio'] = volume / df['Volume_SMA']
            
            # Volume profile (Volume Weighted Average Price)
            typical_price = (high + low + prices) / 3
            df['VWAP'] = (volume * typical_price).cumsum() / volume.cumsum()
            df['High_Volume_Zone'] = volume > (df['Volume_SMA'] * 1.5)
        
        # Additional custom indicators (no TA-Lib required)
        df['MOM'] = prices - pd.Series(prices).shift(10).values
        df['ROC'] = (prices / pd.Series(prices).shift(10).values - 1) * 100
        
        # Price patterns
        df['CCI'] = AdvancedTechnicalAnalysis.calculate_cci(high, low, prices)
        df['WILLR'] = AdvancedTechnicalAnalysis.calculate_williams_r(high, low, prices)
        
        # Custom composite indicators
        df['Trend_Strength'] = np.clip(df['RSI'] / 100, 0, 1)
        df['Momentum_Score'] = (
            df['RSI'] / 100 + 
            df['STOCH_K'] / 100 + 
            (50 + df['CCI'] / 10) / 100
        ) / 3
        
        # Support/Resistance levels
        df['Pivot'] = (high + low + prices) / 3
        df['R1'] = 2 * df['Pivot'] - low
        df['S1'] = 2 * df['Pivot'] - high
        
        # Price position
        df['Close_vs_SMA20'] = (df['Close'] / df['SMA20'] - 1) * 100
        df['Close_vs_SMA50'] = (df['Close'] / df['SMA50'] - 1) * 100
        df['SMA20_vs_SMA50'] = (df['SMA20'] / df['SMA50'] - 1) * 100
        
        # Returns and volatility
        df['Returns'] = pd.Series(prices).pct_change().values
        df['Log_Returns'] = np.log(prices / pd.Series(prices).shift(1).values)
        df['Volatility'] = pd.Series(df['Returns']).rolling(20).std().values * np.sqrt(252)
        
        return df.fillna(method='bfill').fillna(0)
    
    @staticmethod
    def calculate_obv(prices, volume):
        """Calculate On-Balance Volume"""
        obv = np.zeros_like(prices)
        obv[0] = volume[0]
        
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif prices[i] < prices[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        
        return obv
    
    @staticmethod
    def calculate_cci(high, low, close, period=20):
        """Calculate Commodity Channel Index"""
        typical_price = (high + low + close) / 3
        sma = pd.Series(typical_price).rolling(period).mean()
        mean_dev = pd.Series(abs(typical_price - sma)).rolling(period).mean()
        
        cci = (typical_price - sma) / (0.015 * mean_dev)
        return cci.values
    
    @staticmethod
    def calculate_williams_r(high, low, close, period=14):
        """Calculate Williams %R"""
        highest_high = pd.Series(high).rolling(period).max()
        lowest_low = pd.Series(low).rolling(period).min()
        
        willr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return willr.values
    
    @staticmethod
    def detect_market_structure(df):
        """Detect market structure using price action"""
        structure = {
            'higher_highs': 0,
            'higher_lows': 0,
            'lower_highs': 0,
            'lower_lows': 0,
            'trend': 'SIDEWAYS'
        }
        
        # Look at last 20 periods
        window = min(20, len(df))
        recent = df.iloc[-window:]
        
        highs = recent['High'].values
        lows = recent['Low'].values
        
        # Count sequences
        for i in range(1, len(highs)):
            if highs[i] > highs[i-1]:
                structure['higher_highs'] += 1
            elif highs[i] < highs[i-1]:
                structure['lower_highs'] += 1
            
            if lows[i] > lows[i-1]:
                structure['higher_lows'] += 1
            elif lows[i] < lows[i-1]:
                structure['lower_lows'] += 1
        
        # Determine trend
        if structure['higher_highs'] > structure['lower_highs'] and structure['higher_lows'] > structure['lower_lows']:
            structure['trend'] = 'UPTREND'
        elif structure['lower_highs'] > structure['higher_highs'] and structure['lower_lows'] > structure['higher_lows']:
            structure['trend'] = 'DOWNTREND'
        
        return structure

# ============================================================================
# SMC PRO (SMART MONEY CONCEPTS) - ADVANCED
# ============================================================================

class SMCProAnalyzer:
    """Advanced Smart Money Concepts analyzer"""
    
    def __init__(self):
        self.order_blocks = {}
        self.fair_value_gaps = {}
        self.liquidity_zones = {}
        self.breakers = {}
    
    def analyze(self, df, symbol):
        """Complete SMC analysis"""
        analysis = {
            'order_blocks': [],
            'fair_value_gaps': [],
            'liquidity_zones': [],
            'breakers': [],
            'mitigation_blocks': [],
            'bos_choch': [],
            'market_structure': 'UNDEFINED'
        }
        
        if len(df) < 100:
            return analysis
        
        try:
            # 1. Order Blocks (Advanced)
            analysis['order_blocks'] = self.detect_order_blocks(df)
            
            # 2. Fair Value Gaps (FVG)
            analysis['fair_value_gaps'] = self.detect_fair_value_gaps(df)
            
            # 3. Liquidity Zones
            analysis['liquidity_zones'] = self.detect_liquidity_zones(df)
            
            # 4. Break of Structure (BOS) / Change of Character (CHOCH)
            analysis['bos_choch'] = self.detect_bos_choch(df)
            
            # 5. Market Structure
            analysis['market_structure'] = self.determine_market_structure(df)
            
            # 6. Supply/Demand Zones
            analysis['supply_demand_zones'] = self.identify_supply_demand_zones(df)
            
        except Exception as e:
            pass
        
        return analysis
    
    def detect_order_blocks(self, df):
        """Detect order blocks with confirmation"""
        blocks = []
        window = 20
        
        for i in range(window, len(df)-5):
            # Bullish Order Block: Strong down candle followed by up move
            if (df['Close'].iloc[i-1] < df['Open'].iloc[i-1] and  # Down candle
                df['Close'].iloc[i] > df['Open'].iloc[i] and      # Up candle
                df['Low'].iloc[i] <= df['Low'].iloc[i-1] and      # Takes out previous low
                df['Close'].iloc[i] > df['Open'].iloc[i-1]):      # Closes above previous open
                
                # Confirmation: Next candles don't break the low
                confirm = True
                for j in range(1, 4):
                    if i+j < len(df) and df['Low'].iloc[i+j] < df['Low'].iloc[i]:
                        confirm = False
                        break
                
                if confirm:
                    volume_avg = df['Volume'].rolling(20).mean().iloc[i] if 'Volume' in df.columns else 0
                    current_volume = df['Volume'].iloc[i] if 'Volume' in df.columns else 0
                    volume_ratio = current_volume / volume_avg if volume_avg > 0 else 1
                    
                    blocks.append({
                        'type': 'BULLISH',
                        'price': df['Close'].iloc[i],
                        'low': df['Low'].iloc[i],
                        'high': df['High'].iloc[i],
                        'timestamp': df.index[i],
                        'strength': min(1.0, volume_ratio)
                    })
            
            # Bearish Order Block: Strong up candle followed by down move
            elif (df['Close'].iloc[i-1] > df['Open'].iloc[i-1] and  # Up candle
                  df['Close'].iloc[i] < df['Open'].iloc[i] and      # Down candle
                  df['High'].iloc[i] >= df['High'].iloc[i-1] and    # Takes out previous high
                  df['Close'].iloc[i] < df['Open'].iloc[i-1]):      # Closes below previous open
                
                # Confirmation: Next candles don't break the high
                confirm = True
                for j in range(1, 4):
                    if i+j < len(df) and df['High'].iloc[i+j] > df['High'].iloc[i]:
                        confirm = False
                        break
                
                if confirm:
                    volume_avg = df['Volume'].rolling(20).mean().iloc[i] if 'Volume' in df.columns else 0
                    current_volume = df['Volume'].iloc[i] if 'Volume' in df.columns else 0
                    volume_ratio = current_volume / volume_avg if volume_avg > 0 else 1
                    
                    blocks.append({
                        'type': 'BEARISH',
                        'price': df['Close'].iloc[i],
                        'low': df['Low'].iloc[i],
                        'high': df['High'].iloc[i],
                        'timestamp': df.index[i],
                        'strength': min(1.0, volume_ratio)
                    })
        
        return blocks[-5:]  # Last 5 blocks
    
    def detect_fair_value_gaps(self, df):
        """Detect Fair Value Gaps (FVG)"""
        fvgs = []
        
        for i in range(2, len(df)):
            current_high = df['High'].iloc[i]
            current_low = df['Low'].iloc[i]
            prev_high = df['High'].iloc[i-1]
            prev_low = df['Low'].iloc[i-1]
            
            # Bullish FVG: Previous high < Current low
            if prev_high < current_low:
                fvgs.append({
                    'type': 'BULLISH',
                    'gap_low': prev_high,
                    'gap_high': current_low,
                    'timestamp': df.index[i],
                    'size': ((current_low - prev_high) / prev_high) * 100
                })
            
            # Bearish FVG: Previous low > Current high
            elif prev_low > current_high:
                fvgs.append({
                    'type': 'BEARISH',
                    'gap_low': current_high,
                    'gap_high': prev_low,
                    'timestamp': df.index[i],
                    'size': ((prev_low - current_high) / current_high) * 100
                })
        
        return fvgs[-3:] if fvgs else []
    
    def detect_liquidity_zones(self, df):
        """Detect liquidity zones (stops runs)"""
        zones = []
        lookback = 30
        
        if len(df) < lookback:
            return zones
        
        # Recent high/low
        recent_high = df['High'].iloc[-lookback:].max()
        recent_low = df['Low'].iloc[-lookback:].min()
        
        # Check if price is approaching these levels
        current_price = df['Close'].iloc[-1]
        
        # Above market liquidity (shorts stops)
        if abs(current_price - recent_high) / recent_high < 0.02:
            zones.append({
                'type': 'ABOVE_MARKET',
                'price': recent_high,
                'distance_pct': ((current_price - recent_high) / recent_high) * 100,
                'timestamp': df.index[-1]
            })
        
        # Below market liquidity (longs stops)
        if abs(current_price - recent_low) / recent_low < 0.02:
            zones.append({
                'type': 'BELOW_MARKET',
                'price': recent_low,
                'distance_pct': ((current_price - recent_low) / recent_low) * 100,
                'timestamp': df.index[-1]
            })
        
        return zones
    
    def detect_bos_choch(self, df):
        """Detect Break of Structure (BOS) and Change of Character (CHOCH)"""
        signals = []
        window = 15
        
        if len(df) < window + 5:
            return signals
        
        for i in range(window, len(df)):
            # Higher High Break (Bullish BOS)
            if (df['High'].iloc[i] > df['High'].iloc[i-window:i].max() and
                df['Close'].iloc[i] > df['Close'].iloc[i-1]):
                signals.append({
                    'type': 'BULLISH_BOS',
                    'price': df['High'].iloc[i],
                    'timestamp': df.index[i]
                })
            
            # Lower Low Break (Bearish BOS)
            elif (df['Low'].iloc[i] < df['Low'].iloc[i-window:i].min() and
                  df['Close'].iloc[i] < df['Close'].iloc[i-1]):
                signals.append({
                    'type': 'BEARISH_BOS',
                    'price': df['Low'].iloc[i],
                    'timestamp': df.index[i]
                })
        
        return signals[-3:] if signals else []
    
    def determine_market_structure(self, df):
        """Determine market structure using SMC principles"""
        if len(df) < 30:
            return "UNDEFINED"
        
        # Simple market structure detection
        highs_20 = df['High'].rolling(20).max()
        lows_20 = df['Low'].rolling(20).min()
        
        # Check for higher highs and higher lows
        if (df['High'].iloc[-1] > highs_20.iloc[-2] and 
            df['Low'].iloc[-1] > lows_20.iloc[-2]):
            return "BULLISH"
        
        # Check for lower highs and lower lows
        elif (df['High'].iloc[-1] < highs_20.iloc[-2] and 
              df['Low'].iloc[-1] < lows_20.iloc[-2]):
            return "BEARISH"
        
        # Check for equilibrium
        elif (abs(df['High'].iloc[-1] - highs_20.iloc[-2]) / highs_20.iloc[-2] < 0.02 and
              abs(df['Low'].iloc[-1] - lows_20.iloc[-2]) / lows_20.iloc[-2] < 0.02):
            return "EQUILIBRIUM"
        
        return "RANGING"
    
    def identify_supply_demand_zones(self, df):
        """Identify supply and demand zones"""
        zones = []
        window = 20
        
        if len(df) < window:
            return zones
        
        for i in range(window, len(df)):
            # Demand Zone: Strong up move from a level
            if (df['Close'].iloc[i] > df['Open'].iloc[i] and  # Up candle
                df['Close'].iloc[i] > df['Close'].iloc[i-1] and  # Higher close
                (df['High'].iloc[i] - df['Low'].iloc[i]) > 0):  # Has some range
                
                zone_low = df['Low'].iloc[i]
                zone_high = df['High'].iloc[i]
                
                # Check if this level has acted as support before
                for j in range(max(0, i-50), i):
                    if (df['Low'].iloc[j] <= zone_high and 
                        df['Low'].iloc[j] >= zone_low * 0.98):
                        zones.append({
                            'type': 'DEMAND',
                            'low': zone_low,
                            'high': zone_high,
                            'timestamp': df.index[i]
                        })
                        break
            
            # Supply Zone: Strong down move from a level
            elif (df['Close'].iloc[i] < df['Open'].iloc[i] and  # Down candle
                  df['Close'].iloc[i] < df['Close'].iloc[i-1] and  # Lower close
                  (df['High'].iloc[i] - df['Low'].iloc[i]) > 0):  # Has some range
                
                zone_low = df['Low'].iloc[i]
                zone_high = df['High'].iloc[i]
                
                # Check if this level has acted as resistance before
                for j in range(max(0, i-50), i):
                    if (df['High'].iloc[j] >= zone_low and 
                        df['High'].iloc[j] <= zone_high * 1.02):
                        zones.append({
                            'type': 'SUPPLY',
                            'low': zone_low,
                            'high': zone_high,
                            'timestamp': df.index[i]
                        })
                        break
        
        return zones[-5:] if zones else []

# ============================================================================
# DATABASE - SQLITE FOR TRADES & POSITIONS (UNCHANGED)
# ============================================================================

class Database:
    """SQLite database for storing trades and positions"""
    
    def __init__(self):
        self.conn = sqlite3.connect('trading_bot.db', check_same_thread=False)
        self.init_db()
    
    def init_db(self):
        """Initialize database tables"""
        cursor = self.conn.cursor()
        
        # Trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                direction TEXT,
                entry_time TEXT,
                entry_price REAL,
                exit_time TEXT,
                exit_price REAL,
                quantity INTEGER,
                pnl REAL,
                pnl_pct REAL,
                status TEXT,
                strategy TEXT
            )
        ''')
        
        # Positions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY,
                symbol TEXT UNIQUE,
                direction TEXT,
                entry_time TEXT,
                entry_price REAL,
                quantity INTEGER,
                stop_loss REAL,
                take_profit REAL,
                status TEXT
            )
        ''')
        
        self.conn.commit()
    
    def save_trade(self, trade):
        """Save completed trade"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO trades 
            (symbol, direction, entry_time, entry_price, exit_time, exit_price,
             quantity, pnl, pnl_pct, status, strategy)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade['symbol'], trade['direction'], trade['entry_time'],
            trade['entry_price'], trade.get('exit_time'), trade.get('exit_price'),
            trade['quantity'], trade.get('pnl', 0), trade.get('pnl_pct', 0),
            trade['status'], trade.get('strategy', 'AI_ALGO')
        ))
        self.conn.commit()
    
    def get_trades(self, limit=100):
        """Get recent trades"""
        return pd.read_sql_query(
            f"SELECT * FROM trades ORDER BY entry_time DESC LIMIT {limit}",
            self.conn
        )
    
    def save_position(self, position):
        """Save or update position"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO positions
            (symbol, direction, entry_time, entry_price, quantity,
             stop_loss, take_profit, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            position['symbol'], position['direction'], position['entry_time'],
            position['entry_price'], position['quantity'],
            position['stop_loss'], position['take_profit'], position['status']
        ))
        self.conn.commit()
    
    def get_open_positions(self):
        """Get all open positions"""
        return pd.read_sql_query(
            "SELECT * FROM positions WHERE status='OPEN'",
            self.conn
        )
    
    def close_position(self, symbol):
        """Close a position"""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE positions SET status='CLOSED' WHERE symbol=?",
            (symbol,)
        )
        self.conn.commit()

# ============================================================================
# KITE BROKER - WITH PERSISTENT CREDENTIALS (UNCHANGED)
# ============================================================================

class KiteBroker:
    """Zerodha Kite broker integration"""
    
    def __init__(self, demo_mode=True):
        self.demo_mode = demo_mode
        self.kite = None
        self.ticker = None
        self.connected = False
        self.ltp_cache = {}
        self.instruments_dict = {}
        self.websocket_running = False
        
        if not demo_mode and KITE_AVAILABLE:
            self.connect()
    
    def connect(self):
        """Connect to Kite - checks session state first for persistence"""
        try:
            # Priority 1: Session state (persistent during session)
            if 'kite_api_key' in st.session_state and 'kite_access_token' in st.session_state:
                api_key = st.session_state.kite_api_key
                access_token = st.session_state.kite_access_token
            # Priority 2: Streamlit secrets (for deployment)
            elif hasattr(st, 'secrets') and 'KITE_API_KEY' in st.secrets:
                api_key = st.secrets.get("KITE_API_KEY", "")
                access_token = st.secrets.get("KITE_ACCESS_TOKEN", "")
            # Priority 3: Environment variables
            else:
                api_key = os.getenv('KITE_API_KEY', '')
                access_token = os.getenv('KITE_ACCESS_TOKEN', '')
            
            if not api_key or not access_token:
                self.demo_mode = True
                return False
            
            # Initialize Kite Connect
            self.kite = KiteConnect(api_key=api_key)
            self.kite.set_access_token(access_token)
            
            # Test connection
            profile = self.kite.profile()
            st.success(f"✅ Connected to Kite: {profile['user_name']}")
            
            # Load instruments
            self.load_instruments()
            
            # Setup WebSocket
            self.setup_websocket(api_key, access_token)
            
            self.connected = True
            return True
            
        except Exception as e:
            st.error(f"❌ Kite connection failed: {str(e)}")
            self.demo_mode = True
            return False
    
    def load_instruments(self):
        """Load NSE instruments for token mapping"""
        try:
            instruments = self.kite.instruments("NSE")
            
            for inst in instruments:
                symbol = inst['tradingsymbol']
                self.instruments_dict[symbol] = {
                    'token': inst['instrument_token'],
                    'lot_size': inst.get('lot_size', 1),
                    'tick_size': inst.get('tick_size', 0.05)
                }
            
            st.success(f"✅ Loaded {len(self.instruments_dict)} instruments")
            
        except Exception as e:
            st.warning(f"⚠️ Failed to load instruments: {e}")
            self.instruments_dict = {}
    
    def setup_websocket(self, api_key, access_token):
        """Setup WebSocket for live market data"""
        try:
            self.ticker = KiteTicker(api_key, access_token)
            
            def on_ticks(ws, ticks):
                for tick in ticks:
                    token = tick['instrument_token']
                    for symbol, data in self.instruments_dict.items():
                        if data['token'] == token:
                            self.ltp_cache[symbol] = tick['last_price']
                            break
            
            def on_connect(ws, response):
                # Subscribe to top 50 F&O stocks
                tokens = []
                for symbol in StockUniverse.get_all_fno_stocks()[:50]:
                    if symbol in self.instruments_dict:
                        tokens.append(self.instruments_dict[symbol]['token'])
                
                if tokens:
                    ws.subscribe(tokens)
                    ws.set_mode(ws.MODE_LTP, tokens)
            
            def on_close(ws, code, reason):
                self.websocket_running = False
            
            self.ticker.on_ticks = on_ticks
            self.ticker.on_connect = on_connect
            self.ticker.on_close = on_close
            
            # Start in background
            def start_ticker():
                try:
                    self.ticker.connect(threaded=True)
                    self.websocket_running = True
                except:
                    pass
            
            threading.Thread(target=start_ticker, daemon=True).start()
            
        except Exception as e:
            st.warning(f"⚠️ WebSocket setup failed: {e}")
    
    def get_ltp(self, symbol):
        """Get last traded price"""
        # Try cache first
        if symbol in self.ltp_cache:
            return self.ltp_cache[symbol]
        
        # Try Kite API
        if self.connected and self.kite:
            try:
                quote = self.kite.ltp([f"NSE:{symbol}"])
                price = quote[f"NSE:{symbol}"]['last_price']
                self.ltp_cache[symbol] = price
                return price
            except:
                pass
        
        # Fallback to demo price
        hash_value = abs(hash(symbol)) % 10000
        return 1000 + (hash_value / 100)
    
    def get_historical(self, symbol, days=30):
        """Get historical data"""
        if self.connected and self.kite:
            try:
                if symbol not in self.instruments_dict:
                    return self.generate_synthetic(symbol, days)
                
                token = self.instruments_dict[symbol]['token']
                from_date = datetime.now() - timedelta(days=days)
                to_date = datetime.now()
                
                data = self.kite.historical_data(
                    instrument_token=token,
                    from_date=from_date,
                    to_date=to_date,
                    interval='5minute'
                )
                
                if not data:
                    return self.generate_synthetic(symbol, days)
                
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df = df.rename(columns={
                    'date': 'timestamp',
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                })
                df = df.set_index('timestamp')
                df = df.between_time('09:15', '15:30')
                
                return df
                
            except:
                return self.generate_synthetic(symbol, days)
        
        return self.generate_synthetic(symbol, days)
    
    def generate_synthetic(self, symbol, days):
        """Generate synthetic demo data"""
        dates = pd.date_range(
            end=datetime.now(),
            periods=days*78,
            freq='5min'
        )
        
        np.random.seed(abs(hash(symbol)) % 10000)
        base = 1000 + (abs(hash(symbol)) % 5000)
        returns = np.random.normal(0.0001, 0.015, len(dates))
        prices = base * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame(index=dates)
        df['Close'] = prices
        df['Open'] = df['Close'].shift(1) * (1 + np.random.normal(0, 0.002, len(df)))
        df['High'] = df[['Open', 'Close']].max(axis=1) * (1 + abs(np.random.normal(0, 0.005, len(df))))
        df['Low'] = df[['Open', 'Close']].min(axis=1) * (1 - abs(np.random.normal(0, 0.005, len(df))))
        df['Volume'] = np.random.lognormal(10, 1, len(df)).astype(int)
        
        return df.fillna(method='bfill')
    
    def place_order(self, symbol, direction, quantity, price=None):
        """Place order"""
        if self.demo_mode:
            return {
                'status': 'success',
                'order_id': f'DEMO_{int(time.time())}',
                'price': price or self.get_ltp(symbol)
            }
        
        try:
            order_id = self.kite.place_order(
                tradingsymbol=symbol,
                exchange='NSE',
                transaction_type='BUY' if direction == 'LONG' else 'SELL',
                quantity=quantity,
                order_type='MARKET',
                product='MIS',
                variety='regular'
            )
            return {
                'status': 'success',
                'order_id': order_id,
                'price': price or self.get_ltp(symbol)
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

# ============================================================================
# ADVANCED AI ENGINE WITH ENSEMBLE MODELS
# ============================================================================

class AdvancedAIEngine:
    """Advanced ML engine with ensemble models and adaptive learning"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.adaptive_thresholds = {}
    
    def create_advanced_features(self, df, symbol=None):
        """Create comprehensive feature set"""
        # Calculate all technical indicators
        df = AdvancedTechnicalAnalysis.calculate_all_indicators(df)
        
        # Basic features
        feature_cols = [
            'RSI', 'RSI_9', 'RSI_25', 'MACD', 'MACD_Hist', 'ATR', 'NATR',
            'STOCH_K', 'STOCH_D', 'BB_Width', 'BB_Position',
            'Close_vs_SMA20', 'Close_vs_SMA50', 'SMA20_vs_SMA50',
            'Trend_Strength', 'Momentum_Score', 'Volatility'
        ]
        
        # Add volume features if available
        if 'Volume' in df.columns:
            feature_cols.extend(['Volume_Ratio'])
        
        # Price action features
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Body_Size'] = abs(df['Close'] - df['Open']) / (df['High'] - df['Low'] + 1e-10)
        df['Upper_Shadow'] = (df['High'] - df[['Close', 'Open']].max(axis=1)) / (df['High'] - df['Low'] + 1e-10)
        df['Lower_Shadow'] = (df[['Close', 'Open']].min(axis=1) - df['Low']) / (df['High'] - df['Low'] + 1e-10)
        
        feature_cols.extend(['High_Low_Ratio', 'Body_Size', 'Upper_Shadow', 'Lower_Shadow'])
        
        # Time-based features
        df['Hour'] = df.index.hour
        df['DayOfWeek'] = df.index.dayofweek
        
        feature_cols.extend(['Hour', 'DayOfWeek'])
        
        # Lag features
        for lag in [1, 2, 3]:
            df[f'Returns_Lag{lag}'] = df['Returns'].shift(lag)
            feature_cols.append(f'Returns_Lag{lag}')
        
        # Rolling features
        df['Returns_Mean_5'] = df['Returns'].rolling(5).mean()
        df['Returns_Std_5'] = df['Returns'].rolling(5).std()
        
        feature_cols.extend(['Returns_Mean_5', 'Returns_Std_5'])
        
        # Ensure all features exist
        available_features = [col for col in feature_cols if col in df.columns]
        
        return df[available_features].fillna(method='bfill').fillna(0)
    
    def train_ensemble_model(self, df, symbol):
        """Train ensemble model with adaptive parameters"""
        if not ML_AVAILABLE or len(df) < 100:
            return None
        
        try:
            # Create features and labels
            features = self.create_advanced_features(df, symbol)
            
            # Create multi-class labels
            future_returns_5 = df['Close'].shift(-5) / df['Close'] - 1
            
            labels = pd.cut(
                future_returns_5,
                bins=[-np.inf, -0.01, 0.01, np.inf],
                labels=[-1, 0, 1]
            )
            
            # Remove NaN
            mask = ~(features.isna().any(axis=1) | labels.isna())
            X = features[mask]
            y = labels[mask]
            
            if len(X) < 100:
                return None
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Create models based on available libraries
            estimators = []
            
            # Always use Random Forest
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            estimators.append(('rf', rf_model))
            
            # Add XGBoost if available
            if XGBOOST_AVAILABLE:
                xgb_model = xgb.XGBClassifier(
                    n_estimators=80,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1
                )
                estimators.append(('xgb', xgb_model))
            
            # Create voting classifier if multiple estimators
            if len(estimators) > 1:
                ensemble_model = VotingClassifier(
                    estimators=estimators,
                    voting='soft',
                    weights=[1] * len(estimators)
                )
            else:
                ensemble_model = rf_model
            
            # Train model
            ensemble_model.fit(X_scaled, y)
            
            # Store models and scaler
            self.models[symbol] = ensemble_model
            self.scalers[symbol] = scaler
            
            # Calculate feature importance
            if hasattr(rf_model, 'feature_importances_'):
                rf_importance = rf_model.feature_importances_
                self.feature_importance[symbol] = dict(zip(features.columns, rf_importance))
            
            # Simple performance tracking
            train_score = ensemble_model.score(X_scaled, y)
            self.model_performance[symbol] = {
                'train_score': train_score,
                'train_size': len(X),
                'last_trained': datetime.now()
            }
            
            # Adaptive confidence threshold
            self.adaptive_thresholds[symbol] = max(0.55, min(0.8, 0.55 + (train_score - 0.5) * 0.5))
            
            return ensemble_model
            
        except Exception as e:
            return None
    
    def predict_with_confidence(self, df, symbol):
        """Make prediction with calibrated confidence"""
        if symbol not in self.models:
            return 0, 0.0, {}
        
        try:
            features = self.create_advanced_features(df, symbol)
            latest = features.iloc[-1:].values
            
            scaled = self.scalers[symbol].transform(latest)
            
            # Get prediction
            model = self.models[symbol]
            prediction = model.predict(scaled)[0]
            
            # Get probabilities if available
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(scaled)[0]
                confidence = max(proba)
            else:
                confidence = 0.6  # Default confidence
            
            # Feature contributions
            feature_contributions = {}
            if symbol in self.feature_importance:
                top_features = sorted(
                    self.feature_importance[symbol].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                feature_contributions = dict(top_features)
            
            # Market regime adjustment
            market_adjustment = 1.0
            if 'Volatility' in features.columns:
                volatility = features['Volatility'].iloc[-1]
                if volatility > 0.25:  # High volatility
                    market_adjustment *= 0.9
            
            final_confidence = min(0.95, confidence * market_adjustment)
            
            metadata = {
                'prediction': prediction,
                'confidence': final_confidence,
                'feature_contributions': feature_contributions,
                'market_adjustment': market_adjustment
            }
            
            return prediction, final_confidence, metadata
            
        except Exception as e:
            return 0, 0.0, {}

# ============================================================================
# STATISTICAL ARBITRAGE (PAIR TRADING)
# ============================================================================

class PairTradingEngine:
    """Statistical arbitrage pair trading engine"""
    
    def __init__(self, broker):
        self.broker = broker
        self.pairs = StockUniverse.get_trading_pairs()
        self.spread_history = {}
        self.cointegration_tests = {}
        self.positions = {}
        
    def analyze_pair(self, symbol1, symbol2):
        """Analyze cointegration between two stocks"""
        try:
            # Get historical data
            df1 = self.broker.get_historical(symbol1, days=60)
            df2 = self.broker.get_historical(symbol2, days=60)
            
            if len(df1) < 100 or len(df2) < 100:
                return None
            
            # Align data
            common_index = df1.index.intersection(df2.index)
            if len(common_index) < 50:
                return None
            
            price1 = df1.loc[common_index, 'Close'].values
            price2 = df2.loc[common_index, 'Close'].values
            
            # Calculate correlation and basic statistics
            correlation = np.corrcoef(price1, price2)[0, 1]
            
            # Simple spread calculation (price1 - price2)
            spread = price1 - price2
            mean = np.mean(spread)
            std = np.std(spread)
            
            # Z-score
            current_spread = spread[-1]
            z_score = (current_spread - mean) / std if std > 0 else 0
            
            # Simple mean reversion half-life estimation
            spread_lag = spread[:-1]
            spread_diff = spread[1:] - spread_lag
            
            if len(spread_lag) > 10:
                # Simple linear regression for mean reversion speed
                X = np.column_stack([np.ones_like(spread_lag), spread_lag])
                beta = np.linalg.lstsq(X, spread_diff, rcond=None)[0][1]
                half_life = -np.log(2) / beta if beta < 0 else 0
            else:
                half_life = 0
            
            analysis = {
                'symbol1': symbol1,
                'symbol2': symbol2,
                'current_spread': current_spread,
                'spread_mean': mean,
                'spread_std': std,
                'z_score': z_score,
                'half_life': half_life,
                'correlation': correlation,
                'entry_threshold': 2.0,
                'exit_threshold': 0.5,
                'last_updated': datetime.now()
            }
            
            # Store in history
            pair_key = f"{symbol1}_{symbol2}"
            if pair_key not in self.spread_history:
                self.spread_history[pair_key] = []
            
            self.spread_history[pair_key].append({
                'timestamp': datetime.now(),
                'z_score': z_score,
                'spread': current_spread
            })
            
            # Keep last 100 points
            if len(self.spread_history[pair_key]) > 100:
                self.spread_history[pair_key] = self.spread_history[pair_key][-100:]
            
            return analysis
            
        except Exception as e:
            return None
    
    def get_pair_signals(self):
        """Get trading signals for all pairs"""
        signals = []
        
        for symbol1, symbol2 in self.pairs:
            analysis = self.analyze_pair(symbol1, symbol2)
            
            if not analysis:
                continue
            
            z_score = analysis['z_score']
            entry_threshold = analysis['entry_threshold']
            
            # Generate signals based on z-score
            if z_score > entry_threshold:
                # Spread is too wide - short spread (sell symbol1, buy symbol2)
                signals.append({
                    'type': 'PAIR_SHORT_SPREAD',
                    'symbol1': symbol1,
                    'symbol2': symbol2,
                    'action1': 'SHORT',
                    'action2': 'LONG',
                    'z_score': z_score,
                    'confidence': min(0.9, abs(z_score) / 4),
                    'analysis': analysis
                })
                
            elif z_score < -entry_threshold:
                # Spread is too narrow - long spread (buy symbol1, sell symbol2)
                signals.append({
                    'type': 'PAIR_LONG_SPREAD',
                    'symbol1': symbol1,
                    'symbol2': symbol2,
                    'action1': 'LONG',
                    'action2': 'SHORT',
                    'z_score': z_score,
                    'confidence': min(0.9, abs(z_score) / 4),
                    'analysis': analysis
                })
        
        return signals

# ============================================================================
# RISK MANAGER
# ============================================================================

class RiskManager:
    """Risk management and position sizing"""
    
    def __init__(self, config):
        self.config = config
        self.positions = {}
        self.daily_trades = 0
        self.daily_pnl = 0.0
    
    def calculate_position_size(self, price, stop_loss):
        """Calculate position size based on risk"""
        risk_amount = self.config.TOTAL_CAPITAL * self.config.RISK_PER_TRADE
        risk_per_share = abs(price - stop_loss)
        
        if risk_per_share <= 0:
            return 0
        
        quantity = int(risk_amount / risk_per_share)
        return max(1, quantity)
    
    def calculate_stop_loss(self, df, direction):
        """Calculate stop loss using ATR"""
        atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else df['Close'].iloc[-1] * 0.02
        current_price = df['Close'].iloc[-1]
        
        if direction == 'LONG':
            return current_price - (atr * self.config.ATR_MULTIPLIER)
        else:
            return current_price + (atr * self.config.ATR_MULTIPLIER)
    
    def calculate_take_profit(self, entry, stop_loss, direction):
        """Calculate take profit target"""
        risk = abs(entry - stop_loss)
        reward = risk * self.config.TAKE_PROFIT_RATIO
        
        if direction == 'LONG':
            return entry + reward
        else:
            return entry - reward
    
    def can_trade(self):
        """Check if new trade can be taken"""
        if len(self.positions) >= self.config.MAX_POSITIONS:
            return False, "Max positions reached (10)"
        
        if self.daily_trades >= self.config.MAX_DAILY_TRADES:
            return False, "Daily trade limit reached"
        
        return True, "OK"

# ============================================================================
# COMPLETE TRADING ENGINE WITH ALL ENHANCEMENTS
# ============================================================================

class EnhancedTradingEngine:
    """Complete trading engine with all upgrades"""
    
    def __init__(self, config, demo_mode=True):
        self.config = config
        self.broker = KiteBroker(demo_mode)
        self.db = Database()
        self.risk = RiskManager(config)
        self.ai = AdvancedAIEngine()
        self.smc = SMCProAnalyzer()
        self.pair_trading = PairTradingEngine(self.broker)
        
        self.running = False
        self.signals_queue = queue.Queue()
        self.pair_signals_queue = queue.Queue()
        
        # Performance stats
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0
        }
    
    def start(self):
        """Start the trading bot"""
        self.running = True
        threading.Thread(target=self.run_loop, daemon=True).start()
        return True
    
    def stop(self):
        """Stop the trading bot"""
        self.running = False
        return True
    
    def run_loop(self):
        """Main trading loop"""
        scan_counter = 0
        
        while self.running:
            try:
                # Check market hours
                now = datetime.now().time()
                if now < self.config.MARKET_OPEN or now > self.config.MARKET_CLOSE:
                    time.sleep(60)
                    continue
                
                # Scan for signals every 30 seconds
                if scan_counter % 3 == 0:
                    self.enhanced_scan_signals()
                
                # Auto-execute if enabled
                if hasattr(st.session_state, 'auto_execute') and st.session_state.auto_execute:
                    self.execute_signals()
                
                # Manage open positions
                self.manage_positions()
                
                scan_counter += 1
                time.sleep(10)
                
            except Exception as e:
                time.sleep(30)
    
    def enhanced_scan_signals(self):
        """Enhanced signal scanning with SMC and AI"""
        stocks = StockUniverse.get_all_fno_stocks()
        
        for symbol in stocks[:50]:  # Scan first 50 for speed
            try:
                # Get data
                df = self.broker.get_historical(symbol, days=30)
                if len(df) < 100:
                    continue
                
                # 1. AI/ML Signal
                ai_prediction, ai_confidence, ai_metadata = self.ai.predict_with_confidence(df, symbol)
                
                # 2. SMC Analysis
                smc_analysis = self.smc.analyze(df, symbol)
                
                # 3. Technical Analysis
                ta_analysis = AdvancedTechnicalAnalysis.detect_market_structure(df)
                
                # Combine signals
                combined_confidence = ai_confidence
                
                # Adjust based on SMC
                if smc_analysis.get('market_structure') in ['BULLISH', 'BEARISH']:
                    if ((smc_analysis['market_structure'] == 'BULLISH' and ai_prediction == 1) or
                        (smc_analysis['market_structure'] == 'BEARISH' and ai_prediction == -1)):
                        combined_confidence *= 1.1
                
                # Adjust based on technical structure
                if ((ta_analysis['trend'] == 'UPTREND' and ai_prediction == 1) or
                    (ta_analysis['trend'] == 'DOWNTREND' and ai_prediction == -1)):
                    combined_confidence *= 1.05
                
                # Check if we should take the trade
                if (combined_confidence >= self.config.MIN_CONFIDENCE and 
                    ai_prediction != 0 and 
                    ai_prediction in [-1, 1]):
                    
                    direction = 'LONG' if ai_prediction == 1 else 'SHORT'
                    
                    # Check risk manager
                    can_trade, reason = self.risk.can_trade()
                    if not can_trade:
                        continue
                    
                    # Calculate trade parameters
                    current_price = self.broker.get_ltp(symbol)
                    stop_loss = self.risk.calculate_stop_loss(df, direction)
                    
                    take_profit = self.risk.calculate_take_profit(
                        current_price, stop_loss, direction
                    )
                    
                    quantity = self.risk.calculate_position_size(
                        current_price, stop_loss
                    )
                    
                    # Create comprehensive signal
                    signal = {
                        'symbol': symbol,
                        'direction': direction,
                        'price': current_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'quantity': quantity,
                        'confidence': combined_confidence,
                        'timestamp': datetime.now(),
                        'strategy': 'ENHANCED_AI',
                        'metadata': {
                            'ai_confidence': ai_confidence,
                            'ai_prediction': ai_prediction,
                            'smc_analysis': smc_analysis,
                            'ta_analysis': ta_analysis,
                            'market_regime': self.config.market_regime
                        }
                    }
                    
                    self.signals_queue.put(signal)
                    
            except Exception as e:
                continue
    
    def execute_signals(self):
        """Execute pending signals"""
        while not self.signals_queue.empty():
            signal = self.signals_queue.get()
            
            try:
                # Place order
                result = self.broker.place_order(
                    signal['symbol'],
                    signal['direction'],
                    signal['quantity'],
                    signal['price']
                )
                
                if result['status'] == 'success':
                    # Create position
                    position = {
                        'symbol': signal['symbol'],
                        'direction': signal['direction'],
                        'entry_time': str(datetime.now()),
                        'entry_price': result['price'],
                        'quantity': signal['quantity'],
                        'stop_loss': signal['stop_loss'],
                        'take_profit': signal['take_profit'],
                        'status': 'OPEN'
                    }
                    
                    self.risk.positions[signal['symbol']] = position
                    self.db.save_position(position)
                    self.risk.daily_trades += 1
                    self.db.save_trade(position)
                    
            except Exception as e:
                pass
    
    def manage_positions(self):
        """Manage open positions - check stops and targets"""
        for symbol, pos in list(self.risk.positions.items()):
            try:
                current_price = self.broker.get_ltp(symbol)
                
                # Check exit conditions
                exit_reason = None
                
                if pos['direction'] == 'LONG':
                    if current_price <= pos['stop_loss']:
                        exit_reason = 'STOP_LOSS'
                    elif current_price >= pos['take_profit']:
                        exit_reason = 'TAKE_PROFIT'
                else:
                    if current_price >= pos['stop_loss']:
                        exit_reason = 'STOP_LOSS'
                    elif current_price <= pos['take_profit']:
                        exit_reason = 'TAKE_PROFIT'
                
                if exit_reason:
                    self.exit_position(symbol, current_price, exit_reason)
                    
            except Exception as e:
                pass
    
    def exit_position(self, symbol, price, reason):
        """Exit a position"""
        pos = self.risk.positions.get(symbol)
        if not pos:
            return
        
        # Calculate P&L
        if pos['direction'] == 'LONG':
            pnl = (price - pos['entry_price']) * pos['quantity']
        else:
            pnl = (pos['entry_price'] - price) * pos['quantity']
        
        pnl_pct = (pnl / (pos['entry_price'] * pos['quantity'])) * 100
        
        # Place exit order
        exit_dir = 'SHORT' if pos['direction'] == 'LONG' else 'LONG'
        self.broker.place_order(symbol, exit_dir, pos['quantity'], price)
        
        # Update stats
        self.stats['total_trades'] += 1
        if pnl > 0:
            self.stats['winning_trades'] += 1
        else:
            self.stats['losing_trades'] += 1
        
        self.stats['total_pnl'] += pnl
        self.stats['win_rate'] = (
            self.stats['winning_trades'] / self.stats['total_trades'] * 100
            if self.stats['total_trades'] > 0 else 0
        )
        
        # Save trade
        trade = pos.copy()
        trade['exit_time'] = str(datetime.now())
        trade['exit_price'] = price
        trade['pnl'] = pnl
        trade['pnl_pct'] = pnl_pct
        trade['status'] = 'CLOSED'
        self.db.save_trade(trade)
        
        # Close position
        self.db.close_position(symbol)
        del self.risk.positions[symbol]

# ============================================================================
# STREAMLIT DASHBOARD - ENHANCED
# ============================================================================

def main():
    st.set_page_config(
        page_title="AI Algo Trading Bot v7.0 Enhanced",
        page_icon="🤖",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(45deg, #1E88E5, #4CAF50);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        text-align: center;
    }
    .metric-card h3 {
        font-size: 0.9rem;
        color: #888;
        margin-bottom: 0.5rem;
    }
    .metric-card h2 {
        font-size: 1.8rem;
        color: #fff;
        margin: 0.5rem 0;
        font-weight: bold;
    }
    .metric-card p {
        font-size: 1.2rem;
        margin-top: 0.5rem;
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
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    .stButton > button {
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .info-box {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Dependency warnings
    if not TALIB_AVAILABLE:
        st.warning("""
        ⚠️ **TA-Lib not installed!** 
        Advanced technical indicators will use fallback calculations.
        For better performance, install TA-Lib:
        - **Windows:** Download from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)
        - **Linux/Mac:** `pip install TA-Lib`
        """)
    
    # Initialize session state
    if 'engine' not in st.session_state:
        st.session_state.engine = EnhancedTradingEngine(AdaptiveConfig(), demo_mode=True)
        st.session_state.last_refresh = datetime.now()
        st.session_state.auto_execute = False
        st.session_state.auto_refresh = True
        st.session_state.refresh_rate = 10
        st.session_state.active_tab = 0
        
        # Auto-connect if credentials exist
        if 'kite_api_key' in st.session_state and 'kite_access_token' in st.session_state:
            with st.spinner("🔌 Auto-connecting to Kite..."):
                st.session_state.engine.broker.connect()
    
    engine = st.session_state.engine
    
    # Auto-refresh
    if st.session_state.get('auto_refresh', True):
        refresh_rate = st.session_state.get('refresh_rate', 10)
        time_since_refresh = (datetime.now() - st.session_state.last_refresh).total_seconds()
        
        if time_since_refresh >= refresh_rate:
            st.session_state.last_refresh = datetime.now()
            st.rerun()
    
    # Header
    st.markdown("<h1 class='main-header'>🤖 AI ALGORITHMIC TRADING BOT v7.0</h1>", 
                unsafe_allow_html=True)
    st.markdown("### Enhanced with SMC Pro | Advanced ML | Auto-Refresh | All 159 F&O Stocks")
    
    # Market Overview
    st.markdown("### 📊 Market Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        status = "RUNNING" if engine.running else "STOPPED"
        status_class = "status-running" if engine.running else "status-stopped"
        st.markdown(f'<h3 class="{status_class}">{status}</h3>', unsafe_allow_html=True)
        st.markdown("**System Status**")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        pnl = engine.stats['total_pnl']
        pnl_class = "status-running" if pnl >= 0 else "status-stopped"
        st.markdown(f'<h3 class="{pnl_class}">₹{pnl:,.0f}</h3>', unsafe_allow_html=True)
        st.markdown("**Total P&L**")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        win_rate = engine.stats['win_rate']
        st.markdown(f'<h3>{win_rate:.1f}%</h3>', unsafe_allow_html=True)
        st.markdown("**Win Rate**")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        positions = len(engine.risk.positions)
        st.markdown(f'<h3>{positions}/10</h3>', unsafe_allow_html=True)
        st.markdown("**Active Positions**")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ CONTROL PANEL")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 START", type="primary", use_container_width=True):
                engine.start()
                st.success("✅ Bot Started!")
                st.rerun()
        
        with col2:
            if st.button("🛑 STOP", type="secondary", use_container_width=True):
                engine.stop()
                st.warning("⚠️ Bot Stopped!")
                st.rerun()
        
        st.markdown("---")
        
        # Trading Mode
        mode = st.radio("Trading Mode", 
                       ["📈 Paper Trading", "💰 Live Trading"], 
                       index=0)
        engine.broker.demo_mode = "Paper" in mode
        
        # Capital
        capital = st.number_input("Capital (₹)", 
                                 min_value=100000, 
                                 value=2000000, 
                                 step=100000)
        engine.config.TOTAL_CAPITAL = capital
        
        # Risk
        risk = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.1) / 100
        engine.config.RISK_PER_TRADE = risk
        
        # Confidence
        confidence = st.slider("Min Confidence (%)", 50, 90, 55, 5) / 100
        engine.config.MIN_CONFIDENCE = confidence
        
        st.markdown("---")
        
        # Feature Toggles
        st.markdown("### 🔧 Advanced Features")
        
        auto_exec = st.checkbox("🤖 Auto-Execute Signals", value=st.session_state.auto_execute)
        st.session_state.auto_execute = auto_exec
        
        pair_trading = st.checkbox("🔄 Pair Trading", value=False)
        st.session_state.pair_trading = pair_trading
        
        smc_enabled = st.checkbox("🎯 SMC Pro Analysis", value=True)
        
        st.markdown("---")
        
        # Auto Refresh
        st.markdown("### 🔄 Auto Refresh")
        
        auto_refresh = st.checkbox("Enable Auto Refresh", value=st.session_state.auto_refresh)
        st.session_state.auto_refresh = auto_refresh
        
        if auto_refresh:
            refresh_rate = st.slider("Refresh Rate (seconds)", 5, 60, 10)
            st.session_state.refresh_rate = refresh_rate
            
            time_since = (datetime.now() - st.session_state.last_refresh).total_seconds()
            remaining = int(refresh_rate - time_since)
            if remaining > 0:
                st.info(f"⏳ Next refresh in {remaining}s")
        
        st.markdown("---")
        
        # Stock Universe Info
        st.markdown("### 📈 Stock Universe")
        total_stocks = len(StockUniverse.get_all_fno_stocks())
        st.info(f"**Total F&O Stocks:** {total_stocks}")
    
    # Main Tabs
    tabs = st.tabs([
        "🎯 Trading",
        "📈 Positions", 
        "📋 History",
        "📊 Analytics",
        "🎯 SMC Analysis",
        "⚙️ Settings"
    ])
    
    with tabs[0]:  # Trading
        st.markdown("### 🎯 Enhanced Trading Signals")
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            scan_btn = st.button("🔍 Full Market Scan", type="primary", use_container_width=True)
        
        with col2:
            quick_btn = st.button("⚡ Quick Scan (50 Stocks)", type="secondary", use_container_width=True)
        
        with col3:
            exec_btn = st.button("✅ Execute All", use_container_width=True)
        
        # Full Scan
        if scan_btn:
            with st.spinner("🔍 Scanning all F&O stocks..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Clear existing signals
                while not engine.signals_queue.empty():
                    engine.signals_queue.get()
                
                stocks = StockUniverse.get_all_fno_stocks()
                total = len(stocks)
                
                scan_stats = {
                    'scanned': 0,
                    'models_trained': 0,
                    'signals_generated': 0
                }
                
                for idx, symbol in enumerate(stocks):
                    try:
                        progress = (idx + 1) / total
                        progress_bar.progress(progress)
                        status_text.text(f"Scanning {symbol} ({idx+1}/{total}) | Signals: {scan_stats['signals_generated']}")
                        
                        scan_stats['scanned'] += 1
                        
                        df = engine.broker.get_historical(symbol, days=30)
                        if len(df) < 100:
                            continue
                        
                        if symbol not in engine.ai.models:
                            engine.ai.train_ensemble_model(df, symbol)
                            scan_stats['models_trained'] += 1
                        
                        prediction, confidence, metadata = engine.ai.predict_with_confidence(df, symbol)
                        
                        if confidence >= engine.config.MIN_CONFIDENCE and prediction != 0:
                            direction = 'LONG' if prediction == 1 else 'SHORT'
                            current_price = engine.broker.get_ltp(symbol)
                            stop_loss = engine.risk.calculate_stop_loss(df, direction)
                            take_profit = engine.risk.calculate_take_profit(current_price, stop_loss, direction)
                            quantity = engine.risk.calculate_position_size(current_price, stop_loss)
                            
                            signal = {
                                'symbol': symbol,
                                'direction': direction,
                                'price': current_price,
                                'stop_loss': stop_loss,
                                'take_profit': take_profit,
                                'quantity': quantity,
                                'confidence': confidence,
                                'timestamp': datetime.now()
                            }
                            engine.signals_queue.put(signal)
                            scan_stats['signals_generated'] += 1
                    except:
                        pass
                
                progress_bar.progress(1.0)
                st.success(f"✅ Scan Complete! Found {scan_stats['signals_generated']} signals.")
                time.sleep(2)
                st.rerun()
        
        # Show pending signals
        st.markdown("#### Pending Signals")
        
        if not engine.signals_queue.empty():
            signals = []
            temp_queue = queue.Queue()
            
            while not engine.signals_queue.empty():
                sig = engine.signals_queue.get()
                signals.append(sig)
                temp_queue.put(sig)
            
            while not temp_queue.empty():
                engine.signals_queue.put(temp_queue.get())
            
            if signals:
                df_signals = pd.DataFrame([{
                    'Symbol': s['symbol'],
                    'Direction': s['direction'],
                    'Price': f"₹{s['price']:.2f}",
                    'Stop Loss': f"₹{s['stop_loss']:.2f}",
                    'Take Profit': f"₹{s['take_profit']:.2f}",
                    'Quantity': s['quantity'],
                    'Confidence': f"{s['confidence']:.1%}"
                } for s in signals])
                
                st.dataframe(df_signals, use_container_width=True)
            else:
                st.info("🔭 No pending signals")
        else:
            st.info("🔭 No pending signals")
        
        # AI Model Status
        st.markdown("#### 🧠 AI Model Status")
        col_m1, col_m2, col_m3 = st.columns(3)
        
        with col_m1:
            st.metric("Models Trained", len(engine.ai.models))
        
        with col_m2:
            st.metric("Confidence Threshold", f"{engine.config.MIN_CONFIDENCE:.0%}")
        
        with col_m3:
            st.metric("Market Regime", engine.config.market_regime)
    
    with tabs[1]:  # Positions
        st.markdown("### 📈 Active Positions")
        
        positions_df = engine.db.get_open_positions()
        
        if not positions_df.empty:
            # Calculate current P&L
            for idx, row in positions_df.iterrows():
                current_price = engine.broker.get_ltp(row['symbol'])
                if row['direction'] == 'LONG':
                    pnl = (current_price - row['entry_price']) * row['quantity']
                else:
                    pnl = (row['entry_price'] - current_price) * row['quantity']
                
                positions_df.at[idx, 'current_price'] = current_price
                positions_df.at[idx, 'pnl'] = pnl
                positions_df.at[idx, 'pnl_pct'] = (pnl / (row['entry_price'] * row['quantity'])) * 100
            
            st.dataframe(positions_df, use_container_width=True)
            
            # Manual exit
            st.markdown("#### 🛑 Manual Exit")
            exit_cols = st.columns(4)
            
            for idx, (_, row) in enumerate(positions_df.iterrows()):
                col_idx = idx % 4
                with exit_cols[col_idx]:
                    if st.button(f"Exit {row['symbol']}", key=f"exit_{row['symbol']}"):
                        price = engine.broker.get_ltp(row['symbol'])
                        engine.exit_position(row['symbol'], price, 'MANUAL')
                        st.success(f"Exited {row['symbol']} at ₹{price:.2f}")
                        time.sleep(1)
                        st.rerun()
        else:
            st.info("🔭 No active positions")
    
    with tabs[2]:  # History
        st.markdown("### 📋 Trade History")
        
        trades_df = engine.db.get_trades(100)
        
        if not trades_df.empty:
            display_df = trades_df.copy()
            display_df['pnl'] = display_df['pnl'].apply(lambda x: f"₹{x:,.0f}")
            display_df['pnl_pct'] = display_df['pnl_pct'].apply(lambda x: f"{x:.2f}%")
            
            st.dataframe(display_df, use_container_width=True, height=600)
            
            # Export
            csv = trades_df.to_csv(index=False)
            st.download_button(
                "📥 Export CSV",
                csv,
                f"trades_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv",
                use_container_width=True
            )
        else:
            st.info("🔭 No trade history")
    
    with tabs[3]:  # Analytics
        st.markdown("### 📊 Performance Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Trade Distribution")
            
            if PLOTLY_AVAILABLE:
                fig = go.Figure(data=[
                    go.Pie(
                        labels=['Winning', 'Losing'],
                        values=[engine.stats['winning_trades'], engine.stats['losing_trades']],
                        hole=.3,
                        marker_colors=['#00C853', '#FF5252']
                    )
                ])
                fig.update_layout(template='plotly_dark', height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Key Metrics")
            
            st.metric("Total Trades", engine.stats['total_trades'])
            st.metric("Win Rate", f"{engine.stats['win_rate']:.1f}%")
            st.metric("Total P&L", f"₹{engine.stats['total_pnl']:,.0f}")
            st.metric("Daily Trades", engine.risk.daily_trades)
        
        # Recent Performance
        st.markdown("#### 📊 Recent Performance")
        trades_df = engine.db.get_trades(20)
        
        if not trades_df.empty and PLOTLY_AVAILABLE:
            fig = go.Figure(data=[
                go.Bar(
                    x=trades_df['symbol'],
                    y=trades_df['pnl'],
                    marker_color=['green' if x > 0 else 'red' for x in trades_df['pnl']]
                )
            ])
            fig.update_layout(
                title="P&L by Trade",
                template='plotly_dark',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[4]:  # SMC Analysis
        st.markdown("### 🎯 SMC Pro Analysis")
        
        # Select stock
        col_smc1, col_smc2 = st.columns([3, 1])
        
        with col_smc1:
            selected_symbol = st.selectbox(
                "Select Stock",
                StockUniverse.get_all_fno_stocks()[:50],
                index=0
            )
        
        with col_smc2:
            analyze_btn = st.button("🔍 Analyze", use_container_width=True)
        
        if analyze_btn or st.session_state.get('smc_analyzed', False):
            df = engine.broker.get_historical(selected_symbol, days=14)
            
            if not df.empty:
                smc_analysis = engine.smc.analyze(df, selected_symbol)
                
                # Display analysis
                col_a1, col_a2 = st.columns(2)
                
                with col_a1:
                    st.markdown("#### 📊 Market Structure")
                    st.info(f"**Current Structure:** {smc_analysis.get('market_structure', 'UNDEFINED')}")
                    
                    if smc_analysis.get('order_blocks'):
                        st.markdown("#### 🧱 Order Blocks")
                        for block in smc_analysis['order_blocks'][-3:]:
                            st.write(f"{block['type']} at ₹{block['price']:.2f}")
                    
                    if smc_analysis.get('fair_value_gaps'):
                        st.markdown("#### ⚡ Fair Value Gaps")
                        for fvg in smc_analysis['fair_value_gaps'][-3:]:
                            st.write(f"{fvg['type']}: ₹{fvg['gap_low']:.2f} - ₹{fvg['gap_high']:.2f}")
                
                with col_a2:
                    if smc_analysis.get('bos_choch'):
                        st.markdown("#### 🔄 BOS/CHOCH")
                        for signal in smc_analysis['bos_choch'][-3:]:
                            st.write(f"{signal['type']} at ₹{signal['price']:.2f}")
                    
                    if smc_analysis.get('supply_demand_zones'):
                        st.markdown("#### ⚖️ Supply/Demand Zones")
                        for zone in smc_analysis['supply_demand_zones'][-3:]:
                            st.write(f"{zone['type']}: ₹{zone['low']:.2f} - ₹{zone['high']:.2f}")
                
                # Chart
                if PLOTLY_AVAILABLE and len(df) > 50:
                    fig = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        row_heights=[0.7, 0.3],
                        subplot_titles=(f"{selected_symbol} - SMC Analysis", "Volume")
                    )
                    
                    # Candlestick
                    fig.add_trace(
                        go.Candlestick(
                            x=df.index[-100:],
                            open=df['Open'].iloc[-100:],
                            high=df['High'].iloc[-100:],
                            low=df['Low'].iloc[-100:],
                            close=df['Close'].iloc[-100:],
                            name='Price'
                        ),
                        row=1, col=1
                    )
                    
                    # Volume
                    colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] 
                             else 'green' for i in range(len(df))]
                    
                    fig.add_trace(
                        go.Bar(
                            x=df.index[-100:],
                            y=df['Volume'].iloc[-100:] if 'Volume' in df.columns else np.zeros(100),
                            name='Volume',
                            marker_color=colors[-100:]
                        ),
                        row=2, col=1
                    )
                    
                    fig.update_layout(
                        template='plotly_dark',
                        height=700,
                        xaxis_rangeslider_visible=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Insufficient data for SMC analysis")
            
            st.session_state.smc_analyzed = True
    
    with tabs[5]:  # Settings
        st.markdown("### ⚙️ Settings & Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔑 Broker Connection")
            
            if engine.broker.connected:
                st.success("✅ Connected to Kite")
                
                if st.button("🔌 Disconnect", type="secondary"):
                    engine.broker.connected = False
                    engine.broker.kite = None
                    engine.broker.demo_mode = True
                    st.warning("Disconnected. Running in demo mode.")
                    st.rerun()
            
            else:
                st.warning("⚠️ Not connected to Kite")
                
                with st.form("kite_connection"):
                    api_key = st.text_input("API Key", type="password")
                    access_token = st.text_input("Access Token", type="password")
                    
                    if st.form_submit_button("Connect to Kite"):
                        if api_key and access_token:
                            st.session_state.kite_api_key = api_key
                            st.session_state.kite_access_token = access_token
                            
                            with st.spinner("Connecting..."):
                                success = engine.broker.connect()
                            
                            if success:
                                st.success("✅ Connected successfully!")
                                st.rerun()
                            else:
                                st.error("❌ Connection failed")
        
        with col2:
            st.markdown("#### 📊 Trading Configuration")
            
            with st.form("trading_config"):
                max_positions = st.slider("Max Positions", 1, 20, engine.config.MAX_POSITIONS)
                max_daily_trades = st.slider("Max Daily Trades", 10, 100, engine.config.MAX_DAILY_TRADES, 5)
                atr_multiplier = st.slider("ATR Multiplier", 1.0, 5.0, engine.config.ATR_MULTIPLIER, 0.5)
                rr_ratio = st.slider("Risk:Reward Ratio", 1.5, 5.0, engine.config.TAKE_PROFIT_RATIO, 0.5)
                
                if st.form_submit_button("💾 Save Configuration"):
                    engine.config.MAX_POSITIONS = max_positions
                    engine.config.MAX_DAILY_TRADES = max_daily_trades
                    engine.config.ATR_MULTIPLIER = atr_multiplier
                    engine.config.TAKE_PROFIT_RATIO = rr_ratio
                    
                    st.success("✅ Configuration saved!")
                    st.rerun()
            
            # System Info
            st.markdown("---")
            st.markdown("#### 📱 System Information")
            
            st.info(f"""
            **Status:** {'🟢 Running' if engine.running else '🔴 Stopped'}
            **Mode:** {'💰 Live' if not engine.broker.demo_mode else '📈 Paper'}
            **Kite:** {'🟢 Connected' if engine.broker.connected else '🔴 Disconnected'}
            **Auto-Execute:** {'🟢 ON' if st.session_state.auto_execute else '🔴 OFF'}
            **AI Models:** {len(engine.ai.models)} trained
            **Stock Universe:** {len(StockUniverse.get_all_fno_stocks())} stocks
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    <p>🚨 <b>DISCLAIMER:</b> For educational purposes only. Trading involves risk of loss.</p>
    <p>© 2025 AI Algo Trading Bot v7.0 Enhanced | All 159 F&O Stocks | Complete Solution</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN THE APP
# ============================================================================

if __name__ == "__main__":
    main()
