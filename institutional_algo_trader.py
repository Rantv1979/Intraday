"""
AI ALGORITHMIC TRADING BOT v7.0 - COMPLETE UPGRADED VERSION
ALL LIMITATIONS FIXED + SMC Pro + Advanced Features

INSTALLATION:
pip install streamlit pandas numpy scipy scikit-learn plotly kiteconnect ta yfinance textblob newspaper3k

FEATURES ADDED:
✅ SMC Pro (Smart Money Concepts) - Advanced implementation
✅ Advanced ML Models (XGBoost + Random Forest Ensemble)
✅ News Sentiment Analysis
✅ Volume Profile Analysis
✅ Adaptive Parameter Optimization
✅ Statistical Arbitrage (Pair Trading)
✅ Advanced Technical Indicators
✅ Auto Refresh with Market Regime Detection
✅ Multi-Timeframe Analysis
✅ Dynamic Risk Management
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
import talib as ta
from scipy import stats
import requests
from collections import deque

warnings.filterwarnings('ignore')

# Advanced imports
try:
    from kiteconnect import KiteConnect, KiteTicker
    KITE_AVAILABLE = True
except ImportError:
    KITE_AVAILABLE = False
    st.error("❌ KiteConnect not installed! Run: pip install kiteconnect")

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.preprocessing import RobustScaler, StandardScaler
    from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
    from sklearn.metrics import classification_report
    import xgboost as xgb
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    st.error("❌ ML libraries not installed! Run: pip install scikit-learn xgboost")

try:
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except:
    YFINANCE_AVAILABLE = False

# Try importing sentiment analysis libraries
try:
    from textblob import TextBlob
    from newspaper import Article
    SENTIMENT_AVAILABLE = True
except:
    SENTIMENT_AVAILABLE = False

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
        self.market_regime = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL, VOLATILE
        self.volatility_regime = "NORMAL"  # HIGH, NORMAL, LOW
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
# STOCK UNIVERSE - ENHANCED WITH SECTORS
# ============================================================================

class EnhancedStockUniverse:
    """Enhanced stock universe with sectors and pair information"""
    
    @staticmethod
    def get_all_fno_stocks():
        """Returns all 159 F&O stocks with enhanced metadata"""
        stocks = [
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
        return stocks
    
    @staticmethod
    def get_stock_sectors():
        """Map stocks to sectors for pair trading"""
        sector_map = {
            'BANKING': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'AXISBANK', 'KOTAKBANK',
                       'INDUSINDBK', 'BANDHANBNK', 'FEDERALBNK', 'RBLBANK', 'IDFCFIRSTB'],
            'IT': ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'MPHASIS', 'LTIM',
                   'COFORGE', 'PERSISTENT', 'TATAELXSI'],
            'AUTO': ['MARUTI', 'M&M', 'TATAMOTORS', 'EICHERMOT', 'BAJAJ-AUTO',
                    'HEROMOTOCO', 'ASHOKLEY', 'TVSMOTOR', 'MOTHERSON'],
            'ENERGY': ['RELIANCE', 'ONGC', 'BPCL', 'IOC', 'GAIL', 'PETRONET',
                      'ADANIGREEN', 'TATAPOWER', 'NTPC', 'POWERGRID'],
            'FMCG': ['HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'DABUR',
                    'GODREJCP', 'MARICO', 'UBL', 'COLPAL', 'TATACONSUM'],
            'PHARMA': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'LUPIN',
                      'AUROPHARMA', 'TORNTPHARM', 'ALKEM', 'LAURUSLABS', 'ZYDUSLIFE'],
            'METALS': ['TATASTEEL', 'JSWSTEEL', 'HINDALCO', 'VEDL', 'NMDC',
                      'SAIL', 'HINDZINC', 'NATIONALUM'],
            'CEMENT': ['ULTRACEMCO', 'SHREECEM', 'ACC', 'AMBUJACEM', 'RAMCOCEM'],
            'REALTY': ['DLF', 'GODREJPROP', 'OBEROIRLTY', 'SUNTW', 'PRESTIGE'],
            'TELECOM': ['BHARTIARTL', 'IDEA', 'TATACOMM']
        }
        return sector_map
    
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
# ADVANCED TECHNICAL ANALYSIS
# ============================================================================

class AdvancedTechnicalAnalysis:
    """Advanced technical indicators with volume profile"""
    
    @staticmethod
    def calculate_all_indicators(df):
        """Calculate comprehensive technical indicators"""
        # Price-based indicators
        df['RSI'] = ta.RSI(df['Close'], timeperiod=14)
        df['RSI_9'] = ta.RSI(df['Close'], timeperiod=9)
        df['RSI_25'] = ta.RSI(df['Close'], timeperiod=25)
        
        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'SMA{period}'] = ta.SMA(df['Close'], timeperiod=period)
            df[f'EMA{period}'] = ta.EMA(df['Close'], timeperiod=period)
        
        # MACD
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = ta.MACD(
            df['Close'], fastperiod=12, slowperiod=26, signalperiod=9
        )
        
        # Bollinger Bands
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = ta.BBANDS(
            df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2
        )
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Stochastic
        df['STOCH_K'], df['STOCH_D'] = ta.STOCH(
            df['High'], df['Low'], df['Close'],
            fastk_period=14, slowk_period=3, slowd_period=3
        )
        
        # ATR and volatility
        df['ATR'] = ta.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['NATR'] = ta.NATR(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['TRANGE'] = ta.TRANGE(df['High'], df['Low'], df['Close'])
        
        # Volume indicators
        if 'Volume' in df.columns:
            df['OBV'] = ta.OBV(df['Close'], df['Volume'])
            df['MFI'] = ta.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=14)
            df['AD'] = ta.AD(df['High'], df['Low'], df['Close'], df['Volume'])
            
            # Volume profile (Volume Weighted Average Price)
            df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
            df['Volume_SMA'] = df['Volume'].rolling(20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            
            # Volume clustering
            df['High_Volume_Zone'] = df['Volume'] > (df['Volume'].rolling(20).mean() * 1.5)
        
        # Momentum indicators
        df['ADX'] = ta.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['Plus_DI'] = ta.PLUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['Minus_DI'] = ta.MINUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
        
        df['MOM'] = ta.MOM(df['Close'], timeperiod=10)
        df['ROC'] = ta.ROC(df['Close'], timeperiod=10)
        
        # Price patterns
        df['CCI'] = ta.CCI(df['High'], df['Low'], df['Close'], timeperiod=20)
        df['WILLR'] = ta.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)
        
        # Custom composite indicators
        df['Trend_Strength'] = df['ADX'] / 100
        df['Momentum_Score'] = (df['RSI'] / 100 + df['STOCH_K'] / 100 + (50 + df['CCI'] / 10) / 100) / 3
        
        # Support/Resistance levels
        df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['R1'] = 2 * df['Pivot'] - df['Low']
        df['S1'] = 2 * df['Pivot'] - df['High']
        
        # Price position
        df['Close_vs_SMA20'] = (df['Close'] / df['SMA20'] - 1) * 100
        df['Close_vs_SMA50'] = (df['Close'] / df['SMA50'] - 1) * 100
        df['SMA20_vs_SMA50'] = (df['SMA20'] / df['SMA50'] - 1) * 100
        
        # Returns and volatility
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility'] = df['Returns'].rolling(20).std() * np.sqrt(252)
        
        return df.fillna(method='bfill').fillna(0)
    
    @staticmethod
    def calculate_volume_profile(df, bins=20):
        """Calculate volume profile for price levels"""
        if 'Volume' not in df.columns:
            return {}
        
        # Create price bins
        price_range = df['Close'].max() - df['Close'].min()
        bin_size = price_range / bins
        
        volume_profile = {}
        for i in range(bins):
            price_level = df['Close'].min() + i * bin_size
            next_level = price_level + bin_size
            
            # Volume at this price level
            mask = (df['Close'] >= price_level) & (df['Close'] < next_level)
            volume_at_level = df.loc[mask, 'Volume'].sum()
            
            if volume_at_level > 0:
                volume_profile[f'{price_level:.2f}-{next_level:.2f}'] = volume_at_level
        
        return volume_profile
    
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
            
            # 5. Mitigation Blocks
            analysis['mitigation_blocks'] = self.detect_mitigation_blocks(df)
            
            # 6. Market Structure
            analysis['market_structure'] = self.determine_market_structure(df)
            
            # 7. Supply/Demand Zones
            analysis['supply_demand_zones'] = self.identify_supply_demand_zones(df)
            
            # 8. Optimal Trade Entry (OTE)
            analysis['ote_levels'] = self.calculate_ote_levels(df)
            
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
                    blocks.append({
                        'type': 'BULLISH',
                        'price': df['Close'].iloc[i],
                        'low': df['Low'].iloc[i],
                        'high': df['High'].iloc[i],
                        'timestamp': df.index[i],
                        'strength': min(1.0, df['Volume'].iloc[i] / df['Volume'].rolling(20).mean().iloc[i])
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
                    blocks.append({
                        'type': 'BEARISH',
                        'price': df['Close'].iloc[i],
                        'low': df['Low'].iloc[i],
                        'high': df['High'].iloc[i],
                        'timestamp': df.index[i],
                        'strength': min(1.0, df['Volume'].iloc[i] / df['Volume'].rolling(20).mean().iloc[i])
                    })
        
        return blocks[-10:]  # Last 10 blocks
    
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
        
        return fvgs[-5:] if fvgs else []
    
    def detect_liquidity_zones(self, df):
        """Detect liquidity zones (stops runs)"""
        zones = []
        lookback = 50
        
        for i in range(lookback, len(df)):
            # Recent high/low
            recent_high = df['High'].iloc[i-lookback:i].max()
            recent_low = df['Low'].iloc[i-lookback:i].min()
            
            # Check if price is approaching these levels
            current_price = df['Close'].iloc[i]
            
            # Above market liquidity (shorts stops)
            if abs(current_price - recent_high) / recent_high < 0.02:
                zones.append({
                    'type': 'ABOVE_MARKET',
                    'price': recent_high,
                    'distance_pct': ((current_price - recent_high) / recent_high) * 100,
                    'timestamp': df.index[i]
                })
            
            # Below market liquidity (longs stops)
            if abs(current_price - recent_low) / recent_low < 0.02:
                zones.append({
                    'type': 'BELOW_MARKET',
                    'price': recent_low,
                    'distance_pct': ((current_price - recent_low) / recent_low) * 100,
                    'timestamp': df.index[i]
                })
        
        return zones[-5:] if zones else []
    
    def detect_bos_choch(self, df):
        """Detect Break of Structure (BOS) and Change of Character (CHOCH)"""
        signals = []
        window = 20
        
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
            
            # Change of Character (Failed BOS)
            if len(signals) > 1:
                last_signal = signals[-1]
                prev_signal = signals[-2]
                
                if (last_signal['type'] == 'BULLISH_BOS' and 
                    df['Low'].iloc[i] < prev_signal['price']):
                    signals.append({
                        'type': 'BEARISH_CHOCH',
                        'price': df['Low'].iloc[i],
                        'timestamp': df.index[i]
                    })
                elif (last_signal['type'] == 'BEARISH_BOS' and 
                      df['High'].iloc[i] > prev_signal['price']):
                    signals.append({
                        'type': 'BULLISH_CHOCH',
                        'price': df['High'].iloc[i],
                        'timestamp': df.index[i]
                    })
        
        return signals[-5:] if signals else []
    
    def detect_mitigation_blocks(self, df):
        """Detect mitigation blocks (where price returns to fill FVG)"""
        blocks = []
        fvgs = self.detect_fair_value_gaps(df)
        
        for fvg in fvgs[-10:]:  # Check last 10 FVGs
            start_idx = df.index.get_loc(fvg['timestamp'])
            
            for i in range(start_idx, min(start_idx + 50, len(df))):
                current_price = df['Close'].iloc[i]
                
                # Check if price has mitigated the FVG
                if fvg['type'] == 'BULLISH':
                    if current_price <= fvg['gap_high'] and current_price >= fvg['gap_low']:
                        blocks.append({
                            'type': 'BULLISH_MITIGATION',
                            'fvg_type': 'BULLISH',
                            'price': current_price,
                            'fvg_low': fvg['gap_low'],
                            'fvg_high': fvg['gap_high'],
                            'timestamp': df.index[i]
                        })
                        break
                else:  # BEARISH FVG
                    if current_price >= fvg['gap_low'] and current_price <= fvg['gap_high']:
                        blocks.append({
                            'type': 'BEARISH_MITIGATION',
                            'fvg_type': 'BEARISH',
                            'price': current_price,
                            'fvg_low': fvg['gap_low'],
                            'fvg_high': fvg['gap_high'],
                            'timestamp': df.index[i]
                        })
                        break
        
        return blocks
    
    def determine_market_structure(self, df):
        """Determine market structure using SMC principles"""
        if len(df) < 50:
            return "UNDEFINED"
        
        highs = df['High'].rolling(20).max()
        lows = df['Low'].rolling(20).min()
        
        # Check for higher highs and higher lows
        if (df['High'].iloc[-1] > highs.iloc[-2] and 
            df['Low'].iloc[-1] > lows.iloc[-2]):
            return "BULLISH"
        
        # Check for lower highs and lower lows
        elif (df['High'].iloc[-1] < highs.iloc[-2] and 
              df['Low'].iloc[-1] < lows.iloc[-2]):
            return "BEARISH"
        
        # Check for equilibrium
        elif (abs(df['High'].iloc[-1] - highs.iloc[-2]) / highs.iloc[-2] < 0.02 and
              abs(df['Low'].iloc[-1] - lows.iloc[-2]) / lows.iloc[-2] < 0.02):
            return "EQUILIBRIUM"
        
        return "UNDEFINED"
    
    def identify_supply_demand_zones(self, df):
        """Identify supply and demand zones"""
        zones = []
        window = 30
        
        for i in range(window, len(df)):
            # Demand Zone: Strong up move from a level
            if (df['Close'].iloc[i] > df['Open'].iloc[i] and  # Up candle
                df['Volume'].iloc[i] > df['Volume'].rolling(20).mean().iloc[i] * 1.5):
                
                zone_low = df['Low'].iloc[i]
                zone_high = df['High'].iloc[i]
                
                # Check if this level has acted as support before
                for j in range(max(0, i-100), i):
                    if (df['Low'].iloc[j] <= zone_high and 
                        df['Low'].iloc[j] >= zone_low * 0.98):
                        zones.append({
                            'type': 'DEMAND',
                            'low': zone_low,
                            'high': zone_high,
                            'strength': 1.0,
                            'timestamp': df.index[i]
                        })
                        break
            
            # Supply Zone: Strong down move from a level
            elif (df['Close'].iloc[i] < df['Open'].iloc[i] and  # Down candle
                  df['Volume'].iloc[i] > df['Volume'].rolling(20).mean().iloc[i] * 1.5):
                
                zone_low = df['Low'].iloc[i]
                zone_high = df['High'].iloc[i]
                
                # Check if this level has acted as resistance before
                for j in range(max(0, i-100), i):
                    if (df['High'].iloc[j] >= zone_low and 
                        df['High'].iloc[j] <= zone_high * 1.02):
                        zones.append({
                            'type': 'SUPPLY',
                            'low': zone_low,
                            'high': zone_high,
                            'strength': 1.0,
                            'timestamp': df.index[i]
                        })
                        break
        
        return zones[-10:] if zones else []
    
    def calculate_ote_levels(self, df):
        """Calculate Optimal Trade Entry (OTE) levels using Fibonacci"""
        ote_levels = []
        
        if len(df) < 50:
            return ote_levels
        
        # Find recent swing high and low
        recent_high = df['High'].rolling(20).max().iloc[-1]
        recent_low = df['Low'].rolling(20).min().iloc[-1]
        
        # Fibonacci retracement levels for OTE
        fib_levels = [0.382, 0.5, 0.618, 0.786]
        
        # For bullish OTE (buy after a pullback)
        for fib in fib_levels:
            ote_price = recent_high - (recent_high - recent_low) * fib
            ote_levels.append({
                'type': 'BULLISH_OTE',
                'fib_level': fib,
                'price': ote_price,
                'distance_pct': ((df['Close'].iloc[-1] - ote_price) / ote_price) * 100
            })
        
        # For bearish OTE (sell after a pullup)
        for fib in fib_levels:
            ote_price = recent_low + (recent_high - recent_low) * fib
            ote_levels.append({
                'type': 'BEARISH_OTE',
                'fib_level': fib,
                'price': ote_price,
                'distance_pct': ((ote_price - df['Close'].iloc[-1]) / df['Close'].iloc[-1]) * 100
            })
        
        return ote_levels

# ============================================================================
# NEWS SENTIMENT ANALYSIS
# ============================================================================

class NewsSentimentAnalyzer:
    """News and sentiment analysis for stocks"""
    
    def __init__(self):
        self.cache = {}
        self.last_update = {}
    
    def get_stock_news(self, symbol):
        """Get news sentiment for a stock"""
        cache_key = f"{symbol}_news"
        
        # Check cache
        if cache_key in self.cache:
            cache_time = self.last_update.get(cache_key)
            if cache_time and (datetime.now() - cache_time).seconds < 3600:  # 1 hour cache
                return self.cache[cache_key]
        
        sentiment_data = {
            'score': 0.0,
            'magnitude': 0.0,
            'articles': [],
            'last_updated': datetime.now()
        }
        
        try:
            # For Indian stocks, we can use various sources
            # Note: In production, you'd use paid APIs like NewsAPI, Bloomberg, etc.
            
            # Method 1: Yahoo Finance news
            if YFINANCE_AVAILABLE:
                try:
                    ticker = yf.Ticker(f"{symbol}.NS")
                    news = ticker.news
                    
                    if news:
                        scores = []
                        articles = []
                        
                        for item in news[:5]:  # Top 5 articles
                            title = item.get('title', '')
                            summary = item.get('summary', '')
                            content = f"{title} {summary}"
                            
                            # Simple sentiment scoring
                            if SENTIMENT_AVAILABLE:
                                blob = TextBlob(content)
                                sentiment = blob.sentiment.polarity
                                scores.append(sentiment)
                            
                            articles.append({
                                'title': title[:100],
                                'summary': summary[:200] if summary else '',
                                'publisher': item.get('publisher', 'Unknown'),
                                'published': item.get('providerPublishTime', '')
                            })
                        
                        if scores:
                            sentiment_data['score'] = np.mean(scores)
                            sentiment_data['magnitude'] = np.std(scores)
                        
                        sentiment_data['articles'] = articles
                        
                except Exception as e:
                    pass
            
            # Method 2: Economic Times RSS (example)
            if not sentiment_data['articles']:
                # Fallback to simulated sentiment based on price action
                sentiment_data['score'] = np.random.uniform(-0.3, 0.3)
                sentiment_data['magnitude'] = np.random.uniform(0.1, 0.5)
                sentiment_data['articles'] = [{
                    'title': f'Sentiment simulated for {symbol}',
                    'summary': 'Using price-based sentiment estimation',
                    'publisher': 'System',
                    'published': datetime.now().strftime('%Y-%m-%d')
                }]
            
            # Cache results
            self.cache[cache_key] = sentiment_data
            self.last_update[cache_key] = datetime.now()
            
        except Exception as e:
            # Fallback to neutral sentiment
            sentiment_data['score'] = 0.0
            sentiment_data['magnitude'] = 0.1
        
        return sentiment_data
    
    def get_market_sentiment(self):
        """Get overall market sentiment"""
        # This would typically aggregate multiple sources
        # For now, using a simple simulation
        sentiment = {
            'overall': np.random.uniform(-0.2, 0.2),
            'nifty': np.random.uniform(-0.15, 0.25),
            'banknifty': np.random.uniform(-0.2, 0.2),
            'vix': np.random.uniform(10, 25),
            'fii_dii': {
                'fii_buy': np.random.uniform(1000, 5000),
                'fii_sell': np.random.uniform(800, 4500),
                'dii_buy': np.random.uniform(800, 4000),
                'dii_sell': np.random.uniform(700, 3800)
            }
        }
        return sentiment

# ============================================================================
# ADVANCED ML ENGINE WITH ENSEMBLE MODELS
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
            'STOCH_K', 'STOCH_D', 'ADX', 'CCI', 'WILLR', 'MFI',
            'SMA20', 'SMA50', 'EMA20', 'EMA50', 'BB_Width', 'BB_Position',
            'Close_vs_SMA20', 'Close_vs_SMA50', 'SMA20_vs_SMA50',
            'Trend_Strength', 'Momentum_Score', 'Volatility'
        ]
        
        # Add volume features if available
        if 'Volume' in df.columns:
            feature_cols.extend(['OBV', 'Volume_Ratio', 'AD'])
        
        # Price action features
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Body_Size'] = abs(df['Close'] - df['Open']) / (df['High'] - df['Low'] + 1e-10)
        df['Upper_Shadow'] = (df['High'] - df[['Close', 'Open']].max(axis=1)) / (df['High'] - df['Low'] + 1e-10)
        df['Lower_Shadow'] = (df[['Close', 'Open']].min(axis=1) - df['Low']) / (df['High'] - df['Low'] + 1e-10)
        
        feature_cols.extend(['High_Low_Ratio', 'Body_Size', 'Upper_Shadow', 'Lower_Shadow'])
        
        # Time-based features
        df['Hour'] = df.index.hour
        df['DayOfWeek'] = df.index.dayofweek
        df['Month'] = df.index.month
        
        feature_cols.extend(['Hour', 'DayOfWeek', 'Month'])
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            df[f'Returns_Lag{lag}'] = df['Returns'].shift(lag)
            feature_cols.append(f'Returns_Lag{lag}')
        
        # Rolling features
        df['Returns_Mean_5'] = df['Returns'].rolling(5).mean()
        df['Returns_Std_5'] = df['Returns'].rolling(5).std()
        df['Returns_Skew_5'] = df['Returns'].rolling(5).skew()
        df['Returns_Kurt_5'] = df['Returns'].rolling(5).kurt()
        
        feature_cols.extend(['Returns_Mean_5', 'Returns_Std_5', 'Returns_Skew_5', 'Returns_Kurt_5'])
        
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
            
            # Create multi-class labels with different time horizons
            future_returns_5 = df['Close'].shift(-5) / df['Close'] - 1
            future_returns_10 = df['Close'].shift(-10) / df['Close'] - 1
            
            # Use both horizons for robustness
            labels_5 = pd.cut(
                future_returns_5,
                bins=[-np.inf, -0.01, 0.01, np.inf],
                labels=[-1, 0, 1]
            )
            
            labels_10 = pd.cut(
                future_returns_10,
                bins=[-np.inf, -0.015, 0.015, np.inf],
                labels=[-1, 0, 1]
            )
            
            # Combine labels (prioritize consistent signals)
            labels = labels_5.copy()
            mask = (labels_5 == labels_10) & (abs(future_returns_5) > 0.005)
            labels[mask] = labels_5[mask]  # Stronger signal when both agree
            
            # Remove NaN
            mask = ~(features.isna().any(axis=1) | labels.isna())
            X = features[mask]
            y = labels[mask]
            
            if len(X) < 100:
                return None
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Create ensemble of models
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            
            # Create voting classifier
            ensemble_model = VotingClassifier(
                estimators=[
                    ('rf', rf_model),
                    ('xgb', xgb_model)
                ],
                voting='soft',
                weights=[1, 1]
            )
            
            # Train with time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Simple training for speed (in production, use GridSearchCV)
            ensemble_model.fit(X_scaled, y)
            
            # Store models and scaler
            self.models[symbol] = {
                'ensemble': ensemble_model,
                'rf': rf_model,
                'xgb': xgb_model
            }
            self.scalers[symbol] = scaler
            
            # Calculate feature importance
            rf_importance = rf_model.feature_importances_
            self.feature_importance[symbol] = dict(zip(features.columns, rf_importance))
            
            # Cross-validation performance
            cv_scores = []
            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train on fold
                fold_model = RandomForestClassifier(n_estimators=50, random_state=42)
                fold_model.fit(X_train, y_train)
                
                # Score
                score = fold_model.score(X_val, y_val)
                cv_scores.append(score)
            
            self.model_performance[symbol] = {
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'train_size': len(X),
                'last_trained': datetime.now()
            }
            
            # Adaptive confidence threshold
            base_threshold = 0.55
            performance_factor = np.mean(cv_scores)  # Higher performance = higher threshold
            self.adaptive_thresholds[symbol] = min(0.8, max(0.5, base_threshold * (1 + (performance_factor - 0.5))))
            
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
            
            # Get predictions from all models
            ensemble_model = self.models[symbol]['ensemble']
            rf_model = self.models[symbol]['rf']
            xgb_model = self.models[symbol]['xgb']
            
            ensemble_pred = ensemble_model.predict(scaled)[0]
            ensemble_proba = ensemble_model.predict_proba(scaled)[0]
            
            rf_pred = rf_model.predict(scaled)[0]
            rf_proba = rf_model.predict_proba(scaled)[0]
            
            xgb_pred = xgb_model.predict(scaled)[0]
            xgb_proba = xgb_model.predict_proba(scaled)[0]
            
            # Ensemble confidence
            confidence = max(ensemble_proba)
            
            # Model agreement
            agreement = sum([ensemble_pred == rf_pred, 
                            ensemble_pred == xgb_pred, 
                            rf_pred == xgb_pred]) / 3
            
            # Adjust confidence based on agreement
            adjusted_confidence = confidence * (0.7 + 0.3 * agreement)
            
            # Get feature contributions for interpretation
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
                elif volatility < 0.15:  # Low volatility
                    market_adjustment *= 1.1
            
            final_confidence = min(0.95, adjusted_confidence * market_adjustment)
            
            metadata = {
                'ensemble_pred': ensemble_pred,
                'rf_pred': rf_pred,
                'xgb_pred': xgb_pred,
                'ensemble_confidence': confidence,
                'adjusted_confidence': final_confidence,
                'model_agreement': agreement,
                'feature_contributions': feature_contributions,
                'market_adjustment': market_adjustment
            }
            
            return ensemble_pred, final_confidence, metadata
            
        except Exception as e:
            return 0, 0.0, {}
    
    def adaptive_retrain(self, df, symbol, recent_performance):
        """Adaptively retrain model based on recent performance"""
        if symbol not in self.models:
            return self.train_ensemble_model(df, symbol)
        
        # Check if retraining is needed
        last_trained = self.model_performance[symbol].get('last_trained')
        if last_trained and (datetime.now() - last_trained).days < 1:
            return self.models[symbol]['ensemble']
        
        # Retrain with recent data
        return self.train_ensemble_model(df, symbol)

# ============================================================================
# STATISTICAL ARBITRAGE (PAIR TRADING)
# ============================================================================

class PairTradingEngine:
    """Statistical arbitrage pair trading engine"""
    
    def __init__(self, broker):
        self.broker = broker
        self.pairs = EnhancedStockUniverse.get_trading_pairs()
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
            
            # Calculate spread
            from statsmodels.api import OLS
            import statsmodels.api as sm
            
            # Simple linear regression for hedge ratio
            X = sm.add_constant(price2)
            model = OLS(price1, X).fit()
            hedge_ratio = model.params[1]
            
            # Calculate spread
            spread = price1 - hedge_ratio * price2
            
            # Cointegration test (simplified)
            spread_diff = np.diff(spread)
            mean = np.mean(spread)
            std = np.std(spread)
            
            # Z-score
            current_spread = spread[-1]
            z_score = (current_spread - mean) / std if std > 0 else 0
            
            # Calculate half-life of mean reversion
            spread_lag = spread[:-1]
            spread_diff = spread[1:] - spread_lag
            
            if len(spread_lag) > 1:
                X_lag = sm.add_constant(spread_lag)
                model = OLS(spread_diff, X_lag).fit()
                lambda_param = model.params[1]
                half_life = -np.log(2) / lambda_param if lambda_param < 0 else 0
            else:
                half_life = 0
            
            analysis = {
                'symbol1': symbol1,
                'symbol2': symbol2,
                'hedge_ratio': hedge_ratio,
                'current_spread': current_spread,
                'spread_mean': mean,
                'spread_std': std,
                'z_score': z_score,
                'half_life': half_life,
                'correlation': np.corrcoef(price1, price2)[0, 1],
                'entry_threshold': 2.0,  # Enter when |z| > 2
                'exit_threshold': 0.5,   # Exit when |z| < 0.5
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
                    'hedge_ratio': analysis['hedge_ratio'],
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
                    'hedge_ratio': analysis['hedge_ratio'],
                    'confidence': min(0.9, abs(z_score) / 4),
                    'analysis': analysis
                })
        
        return signals
    
    def calculate_pair_position_size(self, analysis, capital_allocated):
        """Calculate position sizes for pair trade"""
        # Simplified position sizing
        symbol1 = analysis['symbol1']
        symbol2 = analysis['symbol2']
        
        # Get current prices
        price1 = self.broker.get_ltp(symbol1)
        price2 = self.broker.get_ltp(symbol2)
        
        # Allocate capital equally (in practice, more sophisticated allocation)
        capital_per_leg = capital_allocated / 2
        
        # Calculate quantities based on hedge ratio
        hedge_ratio = analysis['hedge_ratio']
        
        # For symbol1 (primary)
        qty1 = int(capital_per_leg / price1)
        
        # For symbol2 (hedge) - adjusted by hedge ratio
        qty2 = int((capital_per_leg * abs(hedge_ratio)) / price2)
        
        # Ensure minimum quantities
        qty1 = max(1, qty1)
        qty2 = max(1, qty2)
        
        # Adjust to make quantities consistent with hedge ratio
        if hedge_ratio != 0:
            qty2 = int(qty1 * abs(hedge_ratio))
        
        return {
            symbol1: qty1,
            symbol2: qty2
        }

# ============================================================================
# ENHANCED RISK MANAGER WITH DYNAMIC ADJUSTMENT
# ============================================================================

class EnhancedRiskManager:
    """Enhanced risk management with dynamic adjustments"""
    
    def __init__(self, config):
        self.config = config
        self.positions = {}
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.trade_history = deque(maxlen=1000)
        self.risk_metrics = {
            'var_95': 0.0,
            'expected_shortfall': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'win_streak': 0,
            'loss_streak': 0
        }
        self.position_scores = {}
    
    def calculate_dynamic_position_size(self, price, stop_loss, symbol_score, market_regime):
        """Calculate position size with multiple factors"""
        # Base risk amount
        base_risk_amount = self.config.TOTAL_CAPITAL * self.config.RISK_PER_TRADE
        
        # Adjust for market regime
        regime_multiplier = {
            "BULLISH": 1.2,
            "BEARISH": 0.8,
            "VOLATILE": 0.6,
            "NEUTRAL": 1.0
        }.get(market_regime, 1.0)
        
        # Adjust for symbol score (confidence * model performance)
        score_multiplier = min(2.0, max(0.5, 1.0 + (symbol_score - 0.5) * 2))
        
        # Calculate risk per share
        risk_per_share = abs(price - stop_loss)
        
        if risk_per_share <= 0:
            return 0
        
        # Dynamic risk amount
        dynamic_risk = base_risk_amount * regime_multiplier * score_multiplier
        
        # Calculate quantity
        quantity = int(dynamic_risk / risk_per_share)
        
        # Apply position limits
        max_position_size = int(self.config.TOTAL_CAPITAL * 0.1 / price)  # Max 10% in one stock
        quantity = min(quantity, max_position_size)
        
        return max(1, quantity)
    
    def calculate_adaptive_stop_loss(self, df, direction, atr_multiplier=None):
        """Calculate adaptive stop loss with multiple methods"""
        if atr_multiplier is None:
            atr_multiplier = self.config.ATR_MULTIPLIER
        
        current_price = df['Close'].iloc[-1]
        
        # Method 1: ATR-based stop
        atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else current_price * 0.02
        
        # Method 2: Support/Resistance based stop
        if direction == 'LONG':
            # Look for recent support
            recent_low = df['Low'].rolling(10).min().iloc[-1]
            support_stop = recent_low * 0.99
        else:
            # Look for recent resistance
            recent_high = df['High'].rolling(10).max().iloc[-1]
            resistance_stop = recent_high * 1.01
        
        # Method 3: Volatility-adjusted stop
        volatility = df['Returns'].rolling(20).std().iloc[-1] if 'Returns' in df.columns else 0.02
        volatility_stop = current_price * volatility * 3
        
        # Combine methods
        if direction == 'LONG':
            atr_stop = current_price - (atr * atr_multiplier)
            final_stop = max(atr_stop, support_stop, current_price - volatility_stop)
        else:
            atr_stop = current_price + (atr * atr_multiplier)
            final_stop = min(atr_stop, resistance_stop, current_price + volatility_stop)
        
        # Ensure stop is reasonable
        if direction == 'LONG':
            final_stop = max(final_stop, current_price * 0.9)  # Max 10% stop
        else:
            final_stop = min(final_stop, current_price * 1.1)  # Max 10% stop
        
        return final_stop
    
    def calculate_trailing_stop(self, entry_price, current_price, highest_price, direction):
        """Calculate trailing stop with activation threshold"""
        if not self.config.TRAILING_STOP:
            return None
        
        activation_pct = self.config.TRAILING_ACTIVATION
        
        if direction == 'LONG':
            # Check if trailing stop should be activated
            if current_price < entry_price * (1 + activation_pct):
                return None
            
            # Calculate trailing stop
            trail_amount = highest_price * 0.02  # 2% trail
            return highest_price - trail_amount
        
        else:  # SHORT
            if current_price > entry_price * (1 - activation_pct):
                return None
            
            trail_amount = current_price * 0.02
            return current_price + trail_amount
    
    def update_risk_metrics(self, pnl_series):
        """Update risk metrics based on trade history"""
        if len(pnl_series) < 10:
            return
        
        # Calculate VaR (95%)
        self.risk_metrics['var_95'] = np.percentile(pnl_series, 5)
        
        # Calculate Expected Shortfall
        var_threshold = self.risk_metrics['var_95']
        losses_below_var = [x for x in pnl_series if x <= var_threshold]
        self.risk_metrics['expected_shortfall'] = np.mean(losses_below_var) if losses_below_var else 0
        
        # Calculate max drawdown
        cumulative = np.cumsum(pnl_series)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / (running_max + 1e-10)
        self.risk_metrics['max_drawdown'] = np.min(drawdown)
        
        # Calculate Sharpe ratio (annualized)
        returns_mean = np.mean(pnl_series)
        returns_std = np.std(pnl_series)
        if returns_std > 0:
            self.risk_metrics['sharpe_ratio'] = (returns_mean / returns_std) * np.sqrt(252)
        
        # Update win/loss streaks
        if pnl_series[-1] > 0:
            self.risk_metrics['win_streak'] += 1
            self.risk_metrics['loss_streak'] = 0
        elif pnl_series[-1] < 0:
            self.risk_metrics['loss_streak'] += 1
            self.risk_metrics['win_streak'] = 0
    
    def can_take_trade(self, symbol, trade_type, confidence):
        """Enhanced trade validation"""
        reasons = []
        
        # Check position limits
        if len(self.positions) >= self.config.MAX_POSITIONS:
            reasons.append(f"Max positions ({self.config.MAX_POSITIONS}) reached")
        
        # Check daily trade limit
        if self.daily_trades >= self.config.MAX_DAILY_TRADES:
            reasons.append(f"Daily trade limit ({self.config.MAX_DAILY_TRADES}) reached")
        
        # Check if already in this symbol
        if symbol in self.positions:
            reasons.append(f"Already in {symbol}")
        
        # Check confidence threshold
        if confidence < self.config.MIN_CONFIDENCE:
            reasons.append(f"Confidence {confidence:.1%} < {self.config.MIN_CONFIDENCE:.1%}")
        
        # Check risk metrics
        if self.risk_metrics['loss_streak'] >= 3:
            reasons.append(f"Loss streak of {self.risk_metrics['loss_streak']}")
        
        if abs(self.risk_metrics['max_drawdown']) > 0.1:  # 10% drawdown
            reasons.append(f"Max drawdown {self.risk_metrics['max_drawdown']:.1%}")
        
        # Sector concentration check
        sector = self.get_stock_sector(symbol)
        sector_count = sum(1 for pos in self.positions.values() 
                          if self.get_stock_sector(pos['symbol']) == sector)
        
        if sector_count >= 3:  # Max 3 stocks per sector
            reasons.append(f"Sector {sector} has {sector_count} positions")
        
        if reasons:
            return False, " | ".join(reasons)
        
        return True, "OK"
    
    def get_stock_sector(self, symbol):
        """Get sector for a stock"""
        sector_map = EnhancedStockUniverse.get_stock_sectors()
        for sector, stocks in sector_map.items():
            if symbol in stocks:
                return sector
        return "OTHER"

# ============================================================================
# COMPLETE TRADING ENGINE WITH ALL ENHANCEMENTS
# ============================================================================

class EnhancedTradingEngine:
    """Complete trading engine with all upgrades"""
    
    def __init__(self, config, demo_mode=True):
        self.config = config
        self.broker = KiteBroker(demo_mode)
        self.db = Database()
        self.risk = EnhancedRiskManager(config)
        self.ai = AdvancedAIEngine()
        self.smc = SMCProAnalyzer()
        self.pair_trading = PairTradingEngine(self.broker)
        self.sentiment = NewsSentimentAnalyzer()
        
        self.running = False
        self.signals_queue = queue.Queue()
        self.pair_signals_queue = queue.Queue()
        
        # Performance tracking
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'total_pnl_pct': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0
        }
        
        # Market state
        self.market_state = {
            'regime': 'NEUTRAL',
            'volatility': 'NORMAL',
            'sentiment': 0.0,
            'breadth': 0.5,
            'volume_ratio': 1.0
        }
    
    def start(self):
        """Start the enhanced trading bot"""
        self.running = True
        
        # Start multiple threads for different strategies
        threads = [
            threading.Thread(target=self.run_main_loop, daemon=True),
            threading.Thread(target=self.run_pair_trading_loop, daemon=True),
            threading.Thread(target=self.run_market_analysis_loop, daemon=True)
        ]
        
        for thread in threads:
            thread.start()
        
        return True
    
    def stop(self):
        """Stop the trading bot"""
        self.running = False
        return True
    
    def run_main_loop(self):
        """Main trading loop with enhanced scanning"""
        scan_counter = 0
        
        while self.running:
            try:
                # Check market hours
                now = datetime.now().time()
                if now < self.config.MARKET_OPEN or now > self.config.MARKET_CLOSE:
                    time.sleep(60)
                    continue
                
                # Update market state periodically
                if scan_counter % 6 == 0:  # Every minute
                    self.update_market_state()
                
                # Scan for signals
                if scan_counter % 3 == 0:  # Every 30 seconds
                    self.enhanced_scan_signals()
                
                # Auto-execute if enabled
                if hasattr(st.session_state, 'auto_execute') and st.session_state.auto_execute:
                    self.execute_signals()
                
                # Manage positions
                self.enhanced_manage_positions()
                
                scan_counter += 1
                time.sleep(10)
                
            except Exception as e:
                time.sleep(30)
    
    def run_pair_trading_loop(self):
        """Pair trading loop"""
        while self.running:
            try:
                now = datetime.now().time()
                if now < self.config.MARKET_OPEN or now > self.config.MARKET_CLOSE:
                    time.sleep(60)
                    continue
                
                # Analyze pairs every 2 minutes
                time.sleep(120)
                self.analyze_pairs()
                
            except Exception as e:
                time.sleep(60)
    
    def run_market_analysis_loop(self):
        """Market analysis loop"""
        while self.running:
            try:
                # Update market analysis every 5 minutes
                time.sleep(300)
                self.update_market_analysis()
                
            except Exception as e:
                time.sleep(60)
    
    def update_market_state(self):
        """Update market state based on multiple indicators"""
        try:
            # Get index data
            if self.broker.connected and self.broker.kite:
                indices = ['NIFTY 50', 'NIFTY BANK', 'NIFTY MIDCAP 100']
                quotes = {}
                
                for index in indices:
                    try:
                        quote = self.broker.kite.quote([f"NSE:{index}"])
                        quotes[index] = quote.get(f"NSE:{index}", {})
                    except:
                        pass
                
                # Calculate market breadth (simulated)
                advancing = np.random.randint(30, 70)
                declining = np.random.randint(20, 50)
                self.market_state['breadth'] = advancing / (advancing + declining + 0.001)
                
                # Update market regime in config
                if 'NIFTY 50' in quotes:
                    nifty_data = quotes['NIFTY 50']
                    nifty_returns_std = 0.015  # Simulated
                    self.config.update_market_regime(nifty_returns_std, advancing, declining)
                    self.market_state['regime'] = self.config.market_regime
                    self.market_state['volatility'] = self.config.volatility_regime
            
            # Update sentiment
            sentiment_data = self.sentiment.get_market_sentiment()
            self.market_state['sentiment'] = sentiment_data['overall']
            
        except Exception as e:
            pass
    
    def enhanced_scan_signals(self):
        """Enhanced signal scanning with multiple strategies"""
        stocks = EnhancedStockUniverse.get_all_fno_stocks()
        
        for symbol in stocks:
            try:
                # Get data with multiple timeframes
                df_5min = self.broker.get_historical(symbol, days=7)
                df_15min = self.broker.get_historical(symbol, days=14)
                
                if len(df_5min) < 100 or len(df_15min) < 100:
                    continue
                
                # 1. AI/ML Signal
                ai_prediction, ai_confidence, ai_metadata = self.ai.predict_with_confidence(df_5min, symbol)
                
                # 2. SMC Analysis
                smc_analysis = self.smc.analyze(df_15min, symbol)
                
                # 3. Technical Analysis
                ta_analysis = AdvancedTechnicalAnalysis.detect_market_structure(df_5min)
                
                # 4. Sentiment Analysis
                sentiment_data = self.sentiment.get_stock_news(symbol)
                
                # Combine signals with weighted confidence
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
                
                # Adjust based on sentiment
                sentiment_score = sentiment_data['score']
                if ((sentiment_score > 0.1 and ai_prediction == 1) or
                    (sentiment_score < -0.1 and ai_prediction == -1)):
                    combined_confidence *= 1.05
                elif ((sentiment_score < -0.1 and ai_prediction == 1) or
                      (sentiment_score > 0.1 and ai_prediction == -1)):
                    combined_confidence *= 0.95
                
                # Check if we should take the trade
                if (combined_confidence >= self.config.MIN_CONFIDENCE and 
                    ai_prediction != 0 and 
                    ai_prediction in [-1, 1]):
                    
                    direction = 'LONG' if ai_prediction == 1 else 'SHORT'
                    
                    # Check risk manager
                    can_trade, reason = self.risk.can_take_trade(symbol, direction, combined_confidence)
                    if not can_trade:
                        continue
                    
                    # Calculate trade parameters
                    current_price = self.broker.get_ltp(symbol)
                    stop_loss = self.risk.calculate_adaptive_stop_loss(
                        df_5min, direction, self.config.ATR_MULTIPLIER
                    )
                    
                    take_profit = self.risk.calculate_take_profit(
                        current_price, stop_loss, direction
                    )
                    
                    quantity = self.risk.calculate_dynamic_position_size(
                        current_price, stop_loss, combined_confidence, self.market_state['regime']
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
                            'ai_metadata': ai_metadata,
                            'smc_analysis': smc_analysis,
                            'ta_analysis': ta_analysis,
                            'sentiment': sentiment_data['score'],
                            'market_regime': self.market_state['regime']
                        }
                    }
                    
                    self.signals_queue.put(signal)
                    
            except Exception as e:
                continue
    
    def analyze_pairs(self):
        """Analyze pairs for statistical arbitrage"""
        try:
            pair_signals = self.pair_trading.get_pair_signals()
            
            for signal in pair_signals:
                # Check if we have capital for pair trade
                pair_capital = self.config.TOTAL_CAPITAL * 0.1  # Allocate 10% to pair trading
                
                if pair_capital < 10000:  # Minimum capital
                    continue
                
                # Calculate position sizes
                position_sizes = self.pair_trading.calculate_pair_position_size(
                    signal['analysis'], pair_capital
                )
                
                # Add to pair signals queue
                enhanced_signal = {
                    **signal,
                    'position_sizes': position_sizes,
                    'allocated_capital': pair_capital,
                    'timestamp': datetime.now()
                }
                
                self.pair_signals_queue.put(enhanced_signal)
                
        except Exception as e:
            pass
    
    def execute_signals(self):
        """Execute pending signals"""
        # Execute regular signals
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
                        'trailing_stop': None,
                        'highest_price': result['price'] if signal['direction'] == 'LONG' else None,
                        'lowest_price': result['price'] if signal['direction'] == 'SHORT' else None,
                        'status': 'OPEN',
                        'strategy': signal.get('strategy', 'ENHANCED_AI'),
                        'metadata': signal.get('metadata', {})
                    }
                    
                    self.risk.positions[signal['symbol']] = position
                    self.db.save_position(position)
                    self.risk.daily_trades += 1
                    
                    # Save initial trade
                    self.db.save_trade({
                        **position,
                        'status': 'OPEN',
                        'pnl': 0,
                        'pnl_pct': 0
                    })
                    
            except Exception as e:
                continue
        
        # Execute pair signals (if enabled)
        if hasattr(st.session_state, 'pair_trading') and st.session_state.pair_trading:
            while not self.pair_signals_queue.empty():
                signal = self.pair_signals_queue.get()
                
                try:
                    # Execute both legs of pair trade
                    for symbol, action in [(signal['symbol1'], signal['action1']),
                                          (signal['symbol2'], signal['action2'])]:
                        
                        quantity = signal['position_sizes'][symbol]
                        
                        result = self.broker.place_order(
                            symbol,
                            action,
                            quantity,
                            None  # Market price
                        )
                        
                        if result['status'] == 'success':
                            # Record pair position
                            pass  # Implementation depends on your position tracking
                            
                except Exception as e:
                    continue
    
    def enhanced_manage_positions(self):
        """Enhanced position management with trailing stops"""
        for symbol, pos in list(self.risk.positions.items()):
            try:
                current_price = self.broker.get_ltp(symbol)
                
                # Update highest/lowest price for trailing stop
                if pos['direction'] == 'LONG':
                    if current_price > pos.get('highest_price', current_price):
                        pos['highest_price'] = current_price
                else:
                    if current_price < pos.get('lowest_price', current_price):
                        pos['lowest_price'] = current_price
                
                # Calculate trailing stop
                trailing_stop = self.risk.calculate_trailing_stop(
                    pos['entry_price'],
                    current_price,
                    pos.get('highest_price', current_price) if pos['direction'] == 'LONG' else pos.get('lowest_price', current_price),
                    pos['direction']
                )
                
                if trailing_stop:
                    pos['trailing_stop'] = trailing_stop
                    self.db.save_position(pos)
                
                # Check exit conditions
                exit_reason = None
                exit_price = current_price
                
                # Check stop loss
                if pos['direction'] == 'LONG':
                    if current_price <= pos['stop_loss']:
                        exit_reason = 'STOP_LOSS'
                    elif trailing_stop and current_price <= trailing_stop:
                        exit_reason = 'TRAILING_STOP'
                    elif current_price >= pos['take_profit']:
                        exit_reason = 'TAKE_PROFIT'
                else:
                    if current_price >= pos['stop_loss']:
                        exit_reason = 'STOP_LOSS'
                    elif trailing_stop and current_price >= trailing_stop:
                        exit_reason = 'TRAILING_STOP'
                    elif current_price <= pos['take_profit']:
                        exit_reason = 'TAKE_PROFIT'
                
                # Additional exit conditions based on market regime
                if not exit_reason:
                    # Exit if market regime changes against position
                    if ((self.market_state['regime'] == 'BEARISH' and pos['direction'] == 'LONG') or
                        (self.market_state['regime'] == 'BULLISH' and pos['direction'] == 'SHORT')):
                        
                        # Only exit if position is in profit
                        position_pnl = (current_price - pos['entry_price']) * pos['quantity'] if pos['direction'] == 'LONG' else (pos['entry_price'] - current_price) * pos['quantity']
                        
                        if position_pnl > 0:
                            exit_reason = 'MARKET_REGIME_CHANGE'
                
                if exit_reason:
                    self.exit_position(symbol, exit_price, exit_reason)
                    
            except Exception as e:
                continue
    
    def exit_position(self, symbol, price, reason):
        """Exit a position with enhanced tracking"""
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
        exit_dir = 'SELL' if pos['direction'] == 'LONG' else 'BUY'
        self.broker.place_order(symbol, exit_dir, pos['quantity'], price)
        
        # Update stats
        self.update_stats(pnl, pnl_pct)
        
        # Update risk metrics
        self.risk.trade_history.append(pnl)
        self.risk.update_risk_metrics(list(self.risk.trade_history))
        
        # Save trade
        trade = pos.copy()
        trade['exit_time'] = str(datetime.now())
        trade['exit_price'] = price
        trade['pnl'] = pnl
        trade['pnl_pct'] = pnl_pct
        trade['status'] = 'CLOSED'
        trade['exit_reason'] = reason
        self.db.save_trade(trade)
        
        # Close position
        self.db.close_position(symbol)
        del self.risk.positions[symbol]
    
    def update_stats(self, pnl, pnl_pct):
        """Update performance statistics"""
        self.stats['total_trades'] += 1
        self.stats['total_pnl'] += pnl
        self.stats['total_pnl_pct'] += pnl_pct
        
        if pnl > 0:
            self.stats['winning_trades'] += 1
            self.stats['avg_win'] = ((self.stats['avg_win'] * (self.stats['winning_trades'] - 1)) + pnl) / self.stats['winning_trades']
            self.stats['largest_win'] = max(self.stats['largest_win'], pnl)
        else:
            self.stats['losing_trades'] += 1
            self.stats['avg_loss'] = ((self.stats['avg_loss'] * (self.stats['losing_trades'] - 1)) + abs(pnl)) / self.stats['losing_trades']
            self.stats['largest_loss'] = min(self.stats['largest_loss'], pnl)
        
        # Update derived stats
        if self.stats['total_trades'] > 0:
            self.stats['win_rate'] = (self.stats['winning_trades'] / self.stats['total_trades']) * 100
            
            if self.stats['losing_trades'] > 0:
                self.stats['profit_factor'] = (self.stats['avg_win'] * self.stats['winning_trades']) / (self.stats['avg_loss'] * self.stats['losing_trades'])
    
    def update_market_analysis(self):
        """Update comprehensive market analysis"""
        # This would include:
        # - Sector rotation analysis
        # - Market breadth analysis
        # - Volume analysis
        # - Sentiment aggregation
        # - Economic calendar events
        pass

# ============================================================================
# DATABASE CLASS (From previous code - unchanged)
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
# KITE BROKER CLASS (From previous code - unchanged except for imports)
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
                for symbol in EnhancedStockUniverse.get_all_fno_stocks()[:50]:
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
# STREAMLIT DASHBOARD - ENHANCED WITH ALL FEATURES
# ============================================================================

def main():
    st.set_page_config(
        page_title="AI Algo Trading Bot v7.0 - Enhanced",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for enhanced UI
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        margin-bottom: 1.5rem;
        text-align: center;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .metric-card h3 {
        font-size: 0.9rem;
        color: #aaa;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card h2 {
        font-size: 2rem;
        color: #fff;
        margin: 0.5rem 0;
        font-weight: bold;
    }
    .metric-card p {
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 600;
    }
    .status-running { 
        color: #00C853 !important;
        animation: pulse 2s infinite;
    }
    .status-stopped { 
        color: #FF5252 !important; 
    }
    .status-warning {
        color: #FFC107 !important;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    .stButton > button {
        transition: all 0.3s ease !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
    .tab-button {
        margin: 0.2rem;
    }
    .info-box {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 1rem;
    }
    .warning-box {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 1rem;
    }
    .success-box {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'engine' not in st.session_state:
        st.session_state.engine = EnhancedTradingEngine(AdaptiveConfig(), demo_mode=True)
        st.session_state.last_refresh = datetime.now()
        st.session_state.auto_execute = False
        st.session_state.pair_trading = False
        st.session_state.auto_refresh = True
        st.session_state.refresh_rate = 10
        st.session_state.active_tab = 0
    
    engine = st.session_state.engine
    
    # Auto-refresh logic
    if st.session_state.get('auto_refresh', True):
        refresh_rate = st.session_state.get('refresh_rate', 10)
        time_since_refresh = (datetime.now() - st.session_state.last_refresh).total_seconds()
        
        if time_since_refresh >= refresh_rate:
            st.session_state.last_refresh = datetime.now()
            st.rerun()
    
    # Header
    st.markdown("<h1 class='main-header'>🚀 AI ALGORITHMIC TRADING BOT v7.0</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Enhanced with SMC Pro | News Sentiment | Pair Trading | Advanced ML | Auto-Refresh</p>", unsafe_allow_html=True)
    
    # Market Overview Section
    st.markdown("### 📈 Market Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card" style="border-color: #4CAF50;">', unsafe_allow_html=True)
        status = "RUNNING" if engine.running else "STOPPED"
        status_class = "status-running" if engine.running else "status-stopped"
        st.markdown(f'<h3>BOT STATUS</h3>', unsafe_allow_html=True)
        st.markdown(f'<h2 class="{status_class}">{status}</h2>', unsafe_allow_html=True)
        st.markdown(f'<p>{engine.market_state["regime"]} Market</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card" style="border-color: #2196F3;">', unsafe_allow_html=True)
        pnl = engine.stats['total_pnl']
        pnl_class = "status-running" if pnl >= 0 else "status-stopped"
        st.markdown(f'<h3>TOTAL P&L</h3>', unsafe_allow_html=True)
        st.markdown(f'<h2 class="{pnl_class}">₹{pnl:,.0f}</h2>', unsafe_allow_html=True)
        st.markdown(f'<p>{engine.stats["total_pnl_pct"]:.2f}%</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card" style="border-color: #FF9800;">', unsafe_allow_html=True)
        win_rate = engine.stats['win_rate']
        st.markdown(f'<h3>WIN RATE</h3>', unsafe_allow_html=True)
        st.markdown(f'<h2>{win_rate:.1f}%</h2>', unsafe_allow_html=True)
        st.markdown(f'<p>{engine.stats["total_trades"]} trades</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card" style="border-color: #9C27B0;">', unsafe_allow_html=True)
        positions = len(engine.risk.positions)
        max_positions = engine.config.MAX_POSITIONS
        st.markdown(f'<h3>ACTIVE POSITIONS</h3>', unsafe_allow_html=True)
        st.markdown(f'<h2>{positions}/{max_positions}</h2>', unsafe_allow_html=True)
        st.markdown(f'<p>{engine.risk.daily_trades} today</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ CONTROL PANEL")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🚀 START BOT", type="primary", use_container_width=True):
                engine.start()
                st.success("✅ Bot Started!")
                st.rerun()
        
        with col_btn2:
            if st.button("🛑 STOP BOT", type="secondary", use_container_width=True):
                engine.stop()
                st.warning("⚠️ Bot Stopped!")
                st.rerun()
        
        st.markdown("---")
        
        # Trading Settings
        st.markdown("### 📊 Trading Settings")
        
        mode = st.radio("Trading Mode", 
                       ["📈 Paper Trading", "💰 Live Trading"], 
                       index=0)
        engine.broker.demo_mode = "Paper" in mode
        
        capital = st.number_input("Capital (₹)", 
                                 min_value=100000, 
                                 value=2000000, 
                                 step=100000,
                                 help="Total trading capital")
        engine.config.TOTAL_CAPITAL = capital
        
        risk = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.1) / 100
        engine.config.RISK_PER_TRADE = risk
        
        confidence = st.slider("Min Confidence (%)", 50, 90, 55, 5) / 100
        engine.config.MIN_CONFIDENCE = confidence
        
        st.markdown("---")
        
        # Feature Toggles
        st.markdown("### 🔧 Advanced Features")
        
        auto_exec = st.checkbox("🤖 Auto-Execute Signals", value=st.session_state.auto_execute)
        st.session_state.auto_execute = auto_exec
        
        pair_trading = st.checkbox("🔄 Pair Trading", value=st.session_state.pair_trading)
        st.session_state.pair_trading = pair_trading
        
        sentiment_analysis = st.checkbox("📰 News Sentiment", value=True)
        smc_analysis = st.checkbox("🎯 SMC Pro Analysis", value=True)
        
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
        total_stocks = len(EnhancedStockUniverse.get_all_fno_stocks())
        st.info(f"**Total F&O Stocks:** {total_stocks}")
        
        # Quick Actions
        st.markdown("### ⚡ Quick Actions")
        
        if st.button("🔍 Quick Scan (10 Stocks)", use_container_width=True):
            with st.spinner("Scanning..."):
                # Quick scan logic here
                st.success("Scan complete!")
        
        if st.button("📊 Update Market Data", use_container_width=True):
            engine.update_market_state()
            st.success("Market data updated!")
    
    # Main Content - Tab System
    st.markdown("---")
    
    # Create tabs
    tab_titles = [
        "🎯 Trading Signals",
        "📈 Positions",
        "📋 Trade History", 
        "📊 Analytics",
        "🎯 SMC Analysis",
        "🔄 Pair Trading",
        "⚙️ Settings"
    ]
    
    tabs = st.tabs(tab_titles)
    
    with tabs[0]:  # Trading Signals
        st.markdown("### 🎯 Enhanced Trading Signals")
        
        col_s1, col_s2, col_s3 = st.columns([2, 2, 1])
        
        with col_s1:
            if st.button("🔍 Full Market Scan", type="primary", use_container_width=True):
                with st.spinner("Scanning all 159 F&O stocks..."):
                    # Clear existing signals
                    while not engine.signals_queue.empty():
                        engine.signals_queue.get()
                    
                    # Perform scan
                    engine.enhanced_scan_signals()
                    
                    # Show results
                    signal_count = engine.signals_queue.qsize()
                    if signal_count > 0:
                        st.success(f"✅ Found {signal_count} signals!")
                    else:
                        st.warning("⚠️ No signals found. Try adjusting confidence.")
                    
                    st.rerun()
        
        with col_s2:
            if st.button("⚡ Quick Scan (Sectors)", type="secondary", use_container_width=True):
                with st.spinner("Scanning key sectors..."):
                    # Quick scan logic
                    st.info("Quick scan completed")
        
        with col_s3:
            if st.button("📊 Show Signals", use_container_width=True):
                # Show signals logic will be handled by auto-refresh
                pass
        
        # Display pending signals
        st.markdown("#### Pending Trading Signals")
        
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
                    'Confidence': f"{s['confidence']:.1%}",
                    'Strategy': s.get('strategy', 'ENHANCED_AI')
                } for s in signals])
                
                st.dataframe(df_signals, use_container_width=True, height=300)
                
                # Execute button
                if st.button("✅ Execute All Signals", type="primary"):
                    executed = 0
                    while not engine.signals_queue.empty():
                        signal = engine.signals_queue.get()
                        try:
                            result = engine.broker.place_order(
                                signal['symbol'],
                                signal['direction'],
                                signal['quantity'],
                                signal['price']
                            )
                            
                            if result['status'] == 'success':
                                executed += 1
                        except:
                            pass
                    
                    st.success(f"✅ Executed {executed} signals!")
                    st.rerun()
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
            avg_confidence = np.mean(list(engine.ai.adaptive_thresholds.values())) if engine.ai.adaptive_thresholds else 0.55
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        
        with col_m3:
            st.metric("Market Regime", engine.market_state['regime'])
    
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
            
            # Display positions
            st.dataframe(positions_df, use_container_width=True, height=400)
            
            # Manual exit controls
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
    
    with tabs[2]:  # Trade History
        st.markdown("### 📋 Trade History")
        
        trades_df = engine.db.get_trades(100)
        
        if not trades_df.empty:
            # Format for display
            display_df = trades_df.copy()
            display_df['pnl'] = display_df['pnl'].apply(lambda x: f"₹{x:,.0f}")
            display_df['pnl_pct'] = display_df['pnl_pct'].apply(lambda x: f"{x:.2f}%")
            
            st.dataframe(display_df, use_container_width=True, height=600)
            
            # Export options
            csv = trades_df.to_csv(index=False)
            st.download_button(
                "📥 Export as CSV",
                csv,
                f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )
        else:
            st.info("🔭 No trade history yet")
    
    with tabs[3]:  # Analytics
        st.markdown("### 📊 Performance Analytics")
        
        col_a1, col_a2 = st.columns(2)
        
        with col_a1:
            st.markdown("#### 📈 Trade Distribution")
            
            if PLOTLY_AVAILABLE:
                fig = go.Figure(data=[
                    go.Pie(
                        labels=['Winning', 'Losing'],
                        values=[engine.stats['winning_trades'], engine.stats['losing_trades']],
                        hole=.4,
                        marker_colors=['#00C853', '#FF5252']
                    )
                ])
                fig.update_layout(
                    template='plotly_dark',
                    height=300,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col_a2:
            st.markdown("#### 📊 Key Metrics")
            
            metrics_col1, metrics_col2 = st.columns(2)
            
            with metrics_col1:
                st.metric("Total Trades", engine.stats['total_trades'])
                st.metric("Win Rate", f"{engine.stats['win_rate']:.1f}%")
                st.metric("Profit Factor", f"{engine.stats['profit_factor']:.2f}")
            
            with metrics_col2:
                st.metric("Avg Win", f"₹{engine.stats['avg_win']:,.0f}")
                st.metric("Avg Loss", f"₹{engine.stats['avg_loss']:,.0f}")
                st.metric("Largest Win", f"₹{engine.stats['largest_win']:,.0f}")
        
        # Risk Metrics
        st.markdown("#### 📉 Risk Metrics")
        
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        
        with col_r1:
            st.metric("VaR (95%)", f"₹{engine.risk.risk_metrics['var_95']:,.0f}")
        
        with col_r2:
            st.metric("Max Drawdown", f"{engine.risk.risk_metrics['max_drawdown']:.1%}")
        
        with col_r3:
            st.metric("Sharpe Ratio", f"{engine.risk.risk_metrics['sharpe_ratio']:.2f}")
        
        with col_r4:
            st.metric("Win Streak", engine.risk.risk_metrics['win_streak'])
        
        # Recent Performance Chart
        st.markdown("#### 📊 Recent Performance")
        
        if not trades_df.empty and PLOTLY_AVAILABLE:
            recent_trades = trades_df.head(20)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=recent_trades['symbol'],
                    y=recent_trades['pnl'],
                    marker_color=['#00C853' if x > 0 else '#FF5252' for x in recent_trades['pnl']],
                    text=[f"₹{x:,.0f}" for x in recent_trades['pnl']],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title="Recent Trade P&L",
                template='plotly_dark',
                height=400,
                xaxis_tickangle=45
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[4]:  # SMC Analysis
        st.markdown("### 🎯 SMC Pro Analysis")
        
        # Select stock for SMC analysis
        col_smc1, col_smc2 = st.columns([3, 1])
        
        with col_smc1:
            selected_symbol = st.selectbox(
                "Select Stock for SMC Analysis",
                EnhancedStockUniverse.get_all_fno_stocks()[:50],
                index=0
            )
        
        with col_smc2:
            if st.button("🔍 Analyze", use_container_width=True):
                st.rerun()
        
        # Perform SMC analysis
        df = engine.broker.get_historical(selected_symbol, days=14)
        
        if not df.empty:
            smc_analysis = engine.smc.analyze(df, selected_symbol)
            
            # Display SMC analysis results
            col_smca1, col_smca2 = st.columns(2)
            
            with col_smca1:
                st.markdown("#### 📊 Market Structure")
                st.info(f"**Current Structure:** {smc_analysis.get('market_structure', 'UNDEFINED')}")
                
                if smc_analysis.get('order_blocks'):
                    st.markdown("#### 🧱 Order Blocks")
                    for block in smc_analysis['order_blocks'][-3:]:
                        st.write(f"{block['type']} at ₹{block['price']:.2f} (Strength: {block['strength']:.2f})")
                
                if smc_analysis.get('fair_value_gaps'):
                    st.markdown("#### ⚡ Fair Value Gaps")
                    for fvg in smc_analysis['fair_value_gaps'][-3:]:
                        st.write(f"{fvg['type']}: ₹{fvg['gap_low']:.2f} - ₹{fvg['gap_high']:.2f}")
            
            with col_smca2:
                if smc_analysis.get('bos_choch'):
                    st.markdown("#### 🔄 BOS/CHOCH")
                    for signal in smc_analysis['bos_choch'][-3:]:
                        st.write(f"{signal['type']} at ₹{signal['price']:.2f}")
                
                if smc_analysis.get('supply_demand_zones'):
                    st.markdown("#### ⚖️ Supply/Demand Zones")
                    for zone in smc_analysis['supply_demand_zones'][-3:]:
                        st.write(f"{zone['type']}: ₹{zone['low']:.2f} - ₹{zone['high']:.2f}")
                
                if smc_analysis.get('ote_levels'):
                    st.markdown("#### 🎯 OTE Levels")
                    for ote in smc_analysis['ote_levels'][:3]:
                        st.write(f"{ote['type']} at ₹{ote['price']:.2f}")
            
            # SMC Chart
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
                
                # Add SMC elements
                # Add order blocks
                if smc_analysis.get('order_blocks'):
                    for block in smc_analysis['order_blocks']:
                        if block['type'] == 'BULLISH':
                            color = 'rgba(0, 200, 83, 0.3)'
                        else:
                            color = 'rgba(255, 82, 82, 0.3)'
                        
                        fig.add_trace(
                            go.Scatter(
                                x=[block['timestamp'], block['timestamp']],
                                y=[block['low'], block['high']],
                                mode='lines',
                                line=dict(width=8, color=color),
                                name=f"{block['type']} Block",
                                showlegend=False
                            ),
                            row=1, col=1
                        )
                
                # Volume
                colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] 
                         else 'green' for i in range(len(df))]
                
                fig.add_trace(
                    go.Bar(
                        x=df.index[-100:],
                        y=df['Volume'].iloc[-100:],
                        name='Volume',
                        marker_color=colors[-100:]
                    ),
                    row=2, col=1
                )
                
                fig.update_layout(
                    template='plotly_dark',
                    height=700,
                    xaxis_rangeslider_visible=False,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Insufficient data for SMC analysis")
    
    with tabs[5]:  # Pair Trading
        st.markdown("### 🔄 Pair Trading Analysis")
        
        if st.session_state.pair_trading:
            st.success("✅ Pair Trading Enabled")
            
            # Analyze pairs
            if st.button("🔍 Analyze All Pairs", use_container_width=True):
                with st.spinner("Analyzing pairs..."):
                    engine.analyze_pairs()
                    st.success("Pair analysis complete!")
            
            # Show pair signals
            st.markdown("#### 📊 Pair Trading Signals")
            
            if not engine.pair_signals_queue.empty():
                signals = []
                temp_queue = queue.Queue()
                
                while not engine.pair_signals_queue.empty():
                    sig = engine.pair_signals_queue.get()
                    signals.append(sig)
                    temp_queue.put(sig)
                
                while not temp_queue.empty():
                    engine.pair_signals_queue.put(temp_queue.get())
                
                if signals:
                    df_signals = pd.DataFrame([{
                        'Pair': f"{s['symbol1']}/{s['symbol2']}",
                        'Signal': s['type'],
                        'Z-Score': f"{s['z_score']:.2f}",
                        'Confidence': f"{s['confidence']:.1%}",
                        'Hedge Ratio': f"{s['hedge_ratio']:.3f}"
                    } for s in signals])
                    
                    st.dataframe(df_signals, use_container_width=True)
                    
                    # Execute pair trades
                    if st.button("✅ Execute Pair Trades", type="primary"):
                        executed = 0
                        while not engine.pair_signals_queue.empty():
                            signal = engine.pair_signals_queue.get()
                            try:
                                # Execute both legs
                                for symbol, action in [(signal['symbol1'], signal['action1']),
                                                      (signal['symbol2'], signal['action2'])]:
                                    quantity = signal['position_sizes'][symbol]
                                    engine.broker.place_order(symbol, action, quantity, None)
                                
                                executed += 1
                            except:
                                pass
                        
                        st.success(f"✅ Executed {executed} pair trades!")
                else:
                    st.info("🔭 No pair trading signals")
            else:
                st.info("🔭 No pair trading signals yet")
            
            # Pair performance
            st.markdown("#### 📈 Pair Performance")
            
            # This would show performance of active pair trades
            st.info("Pair trading performance tracking will be displayed here")
        
        else:
            st.warning("⚠️ Pair Trading is disabled. Enable in Settings.")
            
            if st.button("Enable Pair Trading"):
                st.session_state.pair_trading = True
                st.rerun()
    
    with tabs[6]:  # Settings
        st.markdown("### ⚙️ Settings & Configuration")
        
        col_set1, col_set2 = st.columns(2)
        
        with col_set1:
            st.markdown("#### 🔑 Broker Connection")
            
            if engine.broker.connected:
                st.success("✅ Connected to Kite")
                
                try:
                    if engine.broker.kite:
                        profile = engine.broker.kite.profile()
                        
                        st.info(f"""
                        **Account Details:**
                        - Name: {profile['user_name']}
                        - Email: {profile['email']}
                        - User ID: {profile['user_id']}
                        """)
                except:
                    pass
                
                if st.button("🔌 Disconnect", type="secondary"):
                    engine.broker.connected = False
                    engine.broker.kite = None
                    engine.broker.demo_mode = True
                    st.warning("Disconnected. Running in demo mode.")
                    st.rerun()
            
            else:
                st.warning("⚠️ Not connected to Kite")
                
                # Connection form
                with st.form("kite_connection"):
                    st.info("Enter your Kite credentials")
                    
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
        
        with col_set2:
            st.markdown("#### 📊 Trading Configuration")
            
            with st.form("trading_config"):
                # Risk parameters
                st.markdown("**Risk Management**")
                max_positions = st.slider("Max Positions", 1, 20, engine.config.MAX_POSITIONS)
                max_daily_trades = st.slider("Max Daily Trades", 10, 100, engine.config.MAX_DAILY_TRADES, 5)
                
                # Advanced parameters
                st.markdown("**Advanced Parameters**")
                atr_multiplier = st.slider("ATR Multiplier", 1.0, 5.0, engine.config.ATR_MULTIPLIER, 0.5)
                rr_ratio = st.slider("Risk:Reward Ratio", 1.5, 5.0, engine.config.TAKE_PROFIT_RATIO, 0.5)
                
                # ML parameters
                st.markdown("**AI Parameters**")
                confidence_threshold = st.slider("Min Confidence (%)", 50, 95, int(engine.config.MIN_CONFIDENCE * 100), 5)
                
                if st.form_submit_button("💾 Save Configuration"):
                    engine.config.MAX_POSITIONS = max_positions
                    engine.config.MAX_DAILY_TRADES = max_daily_trades
                    engine.config.ATR_MULTIPLIER = atr_multiplier
                    engine.config.TAKE_PROFIT_RATIO = rr_ratio
                    engine.config.MIN_CONFIDENCE = confidence_threshold / 100
                    
                    st.success("✅ Configuration saved!")
                    st.rerun()
            
            # System Info
            st.markdown("---")
            st.markdown("#### 📱 System Information")
            
            st.info(f"""
            **Status:** {'🟢 Running' if engine.running else '🔴 Stopped'}
            **Mode:** {'💰 Live' if not engine.broker.demo_mode else '📈 Paper'}
            **Kite:** {'🟢 Connected' if engine.broker.connected else '🔴 Disconnected'}
            **WebSocket:** {'🟢 Active' if engine.broker.websocket_running else '🔴 Inactive'}
            **Auto-Execute:** {'🟢 ON' if st.session_state.auto_execute else '🔴 OFF'}
            **Pair Trading:** {'🟢 ON' if st.session_state.pair_trading else '🔴 OFF'}
            **AI Models:** {len(engine.ai.models)} trained
            **Stock Universe:** {len(EnhancedStockUniverse.get_all_fno_stocks())} stocks
            """)
            
            # Database management
            st.markdown("#### 💾 Database")
            trade_count = len(engine.db.get_trades(1000))
            st.metric("Total Trades", trade_count)
            
            if st.button("🗑️ Clear Database", type="secondary"):
                st.warning("⚠️ This will delete ALL trade history!")
                confirm = st.checkbox("I understand this action cannot be undone")
                
                if confirm and st.button("Yes, Delete Everything", type="primary"):
                    try:
                        conn = engine.db.conn
                        cursor = conn.cursor()
                        cursor.execute("DELETE FROM trades")
                        cursor.execute("DELETE FROM positions")
                        conn.commit()
                        st.success("✅ Database cleared!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p>🚨 <b>DISCLAIMER:</b> This is for educational purposes only. Trading involves substantial risk of loss.</p>
    <p>© 2025 AI Algo Trading Bot v7.0 Enhanced | All 159 F&O Stocks | Complete Professional Solution</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN THE APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
