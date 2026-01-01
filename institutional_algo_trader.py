"""
AI ALGORITHMIC TRADING BOT v8.1 - FIXED SLIDER ERROR & AUTO-REFRESH
WITH MARKET MOOD GAUGE, LIVE PRICES, AND COLORFUL BUTTON TABS

INSTALLATION:
pip install streamlit pandas numpy scipy scikit-learn plotly kiteconnect pytz

RUN:
streamlit run institutional_algo_trader_fixed.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import warnings
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Optional, Tuple
import sqlite3
import threading
import queue
import os
import pytz

warnings.filterwarnings('ignore')

# Set Indian timezone
IST = pytz.timezone('Asia/Kolkata')

# Import handling
try:
    from kiteconnect import KiteConnect, KiteTicker
    KITE_AVAILABLE = True
except ImportError:
    KITE_AVAILABLE = False
    st.error("❌ KiteConnect not installed! Run: pip install kiteconnect")

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import RobustScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    st.error("❌ scikit-learn not installed! Run: pip install scikit-learn")

try:
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ============================================================================
# CONFIGURATION - FIXED FOR MORE SIGNALS
# ============================================================================

class Config:
    """Trading bot configuration"""
    TOTAL_CAPITAL = 2_000_000
    RISK_PER_TRADE = 0.01  # 1% risk per trade
    
    # Position Limits
    MAX_POSITIONS = 10
    MAX_DAILY_TRADES = 50
    
    # AI Parameters - LOWERED FOR MORE SIGNALS
    MIN_CONFIDENCE = 0.55  # 55% (was 60%)
    
    # Risk Management
    ATR_MULTIPLIER = 2.0
    TAKE_PROFIT_RATIO = 2.5
    TRAILING_STOP = True
    TRAILING_ACTIVATION = 0.015
    
    # Market Hours (IST)
    MARKET_OPEN = dt_time(9, 15)
    MARKET_CLOSE = dt_time(15, 30)
    
    # Auto-refresh settings
    REFRESH_INTERVAL = 5  # seconds

# ============================================================================
# STOCK UNIVERSE - ALL 159 F&O STOCKS
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

# ============================================================================
# MARKET INDICES DATA UPDATER - WITH AUTO REFRESH
# ============================================================================

class MarketIndicesUpdater:
    """Updates market indices from Kite with auto-refresh"""
    
    def __init__(self, broker):
        self.broker = broker
        self.indices_data = {
            'NIFTY 50': {'symbol': 'NIFTY 50', 'ltp': 24350.50, 'change': 0.0, 'change_pct': 0.45, 'timestamp': datetime.now(IST)},
            'NIFTY BANK': {'symbol': 'NIFTY BANK', 'ltp': 52180.75, 'change': 0.0, 'change_pct': 0.28, 'timestamp': datetime.now(IST)},
            'SENSEX': {'symbol': 'SENSEX', 'ltp': 80456.25, 'change': 0.0, 'change_pct': 0.38, 'timestamp': datetime.now(IST)}
        }
        self.last_update = datetime.now(IST)
        self.websocket_ticker = None
        self.websocket_running = False
    
    def update_from_kite(self, force=False):
        """Update indices data from Kite Connect"""
        if not self.broker.connected or not self.broker.kite:
            self.update_demo_data()
            return False
        
        try:
            # Fetch live data from Kite
            indices = self.broker.kite.quote([
                "NSE:NIFTY 50",
                "NSE:NIFTY BANK", 
                "BSE:SENSEX"
            ])
            
            updated = False
            
            # Update NIFTY 50
            nifty_data = indices.get("NSE:NIFTY 50", {})
            if nifty_data:
                old_price = self.indices_data['NIFTY 50']['ltp']
                new_price = nifty_data.get('last_price', old_price)
                self.indices_data['NIFTY 50']['ltp'] = new_price
                self.indices_data['NIFTY 50']['change'] = new_price - old_price
                self.indices_data['NIFTY 50']['change_pct'] = ((new_price - old_price) / old_price * 100) if old_price > 0 else 0
                self.indices_data['NIFTY 50']['timestamp'] = datetime.now(IST)
                updated = True
            
            # Update NIFTY BANK
            banknifty_data = indices.get("NSE:NIFTY BANK", {})
            if banknifty_data:
                old_price = self.indices_data['NIFTY BANK']['ltp']
                new_price = banknifty_data.get('last_price', old_price)
                self.indices_data['NIFTY BANK']['ltp'] = new_price
                self.indices_data['NIFTY BANK']['change'] = new_price - old_price
                self.indices_data['NIFTY BANK']['change_pct'] = ((new_price - old_price) / old_price * 100) if old_price > 0 else 0
                self.indices_data['NIFTY BANK']['timestamp'] = datetime.now(IST)
                updated = True
            
            # Update SENSEX
            sensex_data = indices.get("BSE:SENSEX", {})
            if sensex_data:
                old_price = self.indices_data['SENSEX']['ltp']
                new_price = sensex_data.get('last_price', old_price)
                self.indices_data['SENSEX']['ltp'] = new_price
                self.indices_data['SENSEX']['change'] = new_price - old_price
                self.indices_data['SENSEX']['change_pct'] = ((new_price - old_price) / old_price * 100) if old_price > 0 else 0
                self.indices_data['SENSEX']['timestamp'] = datetime.now(IST)
                updated = True
            
            self.last_update = datetime.now(IST)
            return updated
                
        except Exception as e:
            # Fallback to demo data
            self.update_demo_data()
            return False
    
    def update_demo_data(self):
        """Update with realistic demo data"""
        # Simulate small random changes
        for index in self.indices_data.values():
            change_pct = np.random.uniform(-0.2, 0.2)
            change = index['ltp'] * change_pct / 100
            index['ltp'] += change
            index['change'] = change
            index['change_pct'] = change_pct
            index['timestamp'] = datetime.now(IST)
        
        self.last_update = datetime.now(IST)
    
    def get_market_mood(self):
        """Calculate market mood from average change"""
        avg_change = sum(index['change_pct'] for index in self.indices_data.values()) / 3
        
        if avg_change > 0.5:
            return "🟢 BULLISH", "#00C853", avg_change
        elif avg_change > 0.2:
            return "🟢 SLIGHTLY BULLISH", "#4CAF50", avg_change
        elif avg_change < -0.5:
            return "🔴 BEARISH", "#FF5252", avg_change
        elif avg_change < -0.2:
            return "🔴 SLIGHTLY BEARISH", "#FF9800", avg_change
        else:
            return "🟡 NEUTRAL", "#FFC107", avg_change
    
    def should_refresh(self, interval_seconds=5):
        """Check if data should be refreshed"""
        return (datetime.now(IST) - self.last_update).total_seconds() >= interval_seconds

# ============================================================================
# DATABASE - SQLITE FOR TRADES & POSITIONS
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
                id INTEGER PRIMARY KEY AUTOINCREMENT,
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
                strategy TEXT,
                confidence REAL
            )
        ''')
        
        # Positions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT UNIQUE,
                direction TEXT,
                entry_time TEXT,
                entry_price REAL,
                quantity INTEGER,
                stop_loss REAL,
                take_profit REAL,
                status TEXT,
                confidence REAL
            )
        ''')
        
        # Signals table for tracking AI signals
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                direction TEXT,
                confidence REAL,
                price REAL,
                stop_loss REAL,
                take_profit REAL,
                quantity INTEGER,
                generated_time TEXT,
                executed INTEGER DEFAULT 0
            )
        ''')
        
        self.conn.commit()
    
    def save_trade(self, trade):
        """Save completed trade"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO trades 
            (symbol, direction, entry_time, entry_price, exit_time, exit_price,
             quantity, pnl, pnl_pct, status, strategy, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade['symbol'], trade['direction'], trade['entry_time'],
            trade['entry_price'], trade.get('exit_time'), trade.get('exit_price'),
            trade['quantity'], trade.get('pnl', 0), trade.get('pnl_pct', 0),
            trade['status'], trade.get('strategy', 'AI_ALGO'), trade.get('confidence', 0.0)
        ))
        self.conn.commit()
        return cursor.lastrowid
    
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
             stop_loss, take_profit, status, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            position['symbol'], position['direction'], position['entry_time'],
            position['entry_price'], position['quantity'],
            position['stop_loss'], position['take_profit'], 
            position['status'], position.get('confidence', 0.0)
        ))
        self.conn.commit()
        return cursor.lastrowid
    
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
            "UPDATE positions SET status='CLOSED' WHERE symbol=? AND status='OPEN'",
            (symbol,)
        )
        self.conn.commit()
        return cursor.rowcount > 0
    
    def save_signal(self, signal):
        """Save AI signal"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO signals 
            (symbol, direction, confidence, price, stop_loss, take_profit,
             quantity, generated_time, executed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal['symbol'], signal['direction'], signal['confidence'],
            signal['price'], signal['stop_loss'], signal['take_profit'],
            signal['quantity'], signal['timestamp'], 0
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_recent_signals(self, limit=20):
        """Get recent AI signals"""
        return pd.read_sql_query(
            f"SELECT * FROM signals ORDER BY generated_time DESC LIMIT {limit}",
            self.conn
        )
    
    def mark_signal_executed(self, signal_id):
        """Mark signal as executed"""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE signals SET executed=1 WHERE id=?",
            (signal_id,)
        )
        self.conn.commit()

# ============================================================================
# KITE BROKER - WITH PERSISTENT CREDENTIALS
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
        self.last_ws_update = datetime.now(IST)
        
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
            
            # Also load indices
            indices_instruments = self.kite.instruments("INDICES")
            for inst in indices_instruments:
                symbol = inst['tradingsymbol']
                if symbol in ['NIFTY 50', 'NIFTY BANK', 'SENSEX']:
                    self.instruments_dict[symbol] = {
                        'token': inst['instrument_token'],
                        'lot_size': 1,
                        'tick_size': 0.05
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
                self.last_ws_update = datetime.now(IST)
                for tick in ticks:
                    token = tick['instrument_token']
                    # Update LTP cache
                    for symbol, data in self.instruments_dict.items():
                        if data['token'] == token:
                            self.ltp_cache[symbol] = tick['last_price']
                            break
            
            def on_connect(ws, response):
                # Subscribe to top 50 F&O stocks and indices
                tokens = []
                for symbol in StockUniverse.get_all_fno_stocks()[:50]:
                    if symbol in self.instruments_dict:
                        tokens.append(self.instruments_dict[symbol]['token'])
                
                # Subscribe to indices
                for index in ['NIFTY 50', 'NIFTY BANK', 'SENSEX']:
                    if index in self.instruments_dict:
                        tokens.append(self.instruments_dict[index]['token'])
                
                if tokens:
                    ws.subscribe(tokens)
                    ws.set_mode(ws.MODE_LTP, tokens)
                
                self.websocket_running = True
            
            def on_close(ws, code, reason):
                self.websocket_running = False
            
            self.ticker.on_ticks = on_ticks
            self.ticker.on_connect = on_connect
            self.ticker.on_close = on_close
            
            # Start in background
            def start_ticker():
                try:
                    self.ticker.connect(threaded=True)
                except Exception as e:
                    self.websocket_running = False
            
            threading.Thread(target=start_ticker, daemon=True).start()
            
        except Exception as e:
            st.warning(f"⚠️ WebSocket setup failed: {e}")
            self.websocket_running = False
    
    def get_ltp(self, symbol):
        """Get last traded price"""
        # Try WebSocket cache first
        if symbol in self.ltp_cache:
            return self.ltp_cache[symbol]
        
        # Try Kite API
        if self.connected and self.kite:
            try:
                exchange = 'BSE' if symbol == 'SENSEX' else 'NSE'
                quote = self.kite.ltp([f"{exchange}:{symbol}"])
                price = quote[f"{exchange}:{symbol}"]['last_price']
                self.ltp_cache[symbol] = price
                return price
            except:
                pass
        
        # Fallback to demo price
        hash_value = abs(hash(symbol)) % 10000
        base_price = 1000 + (hash_value / 100)
        # Add small random walk for demo
        if symbol in self.ltp_cache:
            change = np.random.uniform(-0.01, 0.01)
            self.ltp_cache[symbol] = self.ltp_cache[symbol] * (1 + change)
        else:
            self.ltp_cache[symbol] = base_price
        return self.ltp_cache[symbol]
    
    def get_historical(self, symbol, days=30):
        """Get historical data"""
        if self.connected and self.kite and symbol in self.instruments_dict:
            try:
                token = self.instruments_dict[symbol]['token']
                from_date = datetime.now(IST) - timedelta(days=days)
                to_date = datetime.now(IST)
                
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
                # Convert to IST
                df.index = df.index.tz_convert(IST)
                # Filter market hours
                df = df.between_time('09:15', '15:30')
                
                return df
                
            except Exception as e:
                return self.generate_synthetic(symbol, days)
        
        return self.generate_synthetic(symbol, days)
    
    def generate_synthetic(self, symbol, days):
        """Generate synthetic demo data"""
        base_date = datetime.now(IST)
        dates = pd.date_range(
            end=base_date,
            periods=days*78,
            freq='5min',
            tz=IST
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
            order_price = price or self.get_ltp(symbol)
            time.sleep(0.1)  # Simulate order delay
            return {
                'status': 'success',
                'order_id': f'DEMO_{int(time.time())}',
                'price': order_price
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

class TechnicalAnalysis:
    """Technical indicators calculator"""
    
    @staticmethod
    def calculate_indicators(df):
        """Calculate all technical indicators"""
        if df.empty:
            return df
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['ATR'] = ranges.max(axis=1).rolling(14).mean()
        
        # Moving Averages
        for period in [5, 10, 20, 50]:
            df[f'SMA{period}'] = df['Close'].rolling(period).mean()
            df[f'EMA{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        
        # MACD
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        return df.fillna(method='bfill')

# ============================================================================
# AI ENGINE - FIXED FOR MORE SIGNALS
# ============================================================================

class AIEngine:
    """Machine Learning engine for signal generation"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.last_training_time = {}
    
    def create_features(self, df):
        """Create ML features from price data"""
        if df.empty:
            return pd.DataFrame()
        
        df = TechnicalAnalysis.calculate_indicators(df)
        
        feature_cols = [
            'RSI', 'ATR', 'SMA5', 'SMA10', 'SMA20', 'SMA50',
            'EMA5', 'EMA10', 'EMA20', 'MACD', 'MACD_Signal',
            'BB_Upper', 'BB_Middle', 'BB_Lower'
        ]
        
        # Add price features
        df['Returns'] = df['Close'].pct_change()
        df['Returns_5'] = df['Close'].pct_change(5)
        df['Returns_10'] = df['Close'].pct_change(10)
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        df['Price_MA_Ratio'] = df['Close'] / df['SMA20']
        
        feature_cols.extend(['Returns', 'Returns_5', 'Returns_10', 
                           'Volume_Ratio', 'Price_MA_Ratio'])
        
        result = df[feature_cols].fillna(method='bfill').fillna(0)
        return result
    
    def train_model(self, df, symbol):
        """Train ML model - FIXED with lower thresholds"""
        if not ML_AVAILABLE or df.empty or len(df) < 50:
            return None
        
        try:
            # Check if model needs retraining
            if symbol in self.last_training_time:
                hours_since = (datetime.now(IST) - self.last_training_time[symbol]).total_seconds() / 3600
                if hours_since < 24:  # Retrain once per day
                    return self.models.get(symbol)
            
            features = self.create_features(df)
            
            if features.empty:
                return None
            
            # Create labels - More sensitive thresholds
            future_returns = df['Close'].shift(-5) / df['Close'] - 1
            labels = pd.cut(
                future_returns,
                bins=[-np.inf, -0.003, 0.003, np.inf],  # 0.3% thresholds
                labels=[-1, 0, 1]
            )
            
            # Remove NaN
            mask = ~(features.isna().any(axis=1) | labels.isna())
            X = features[mask]
            y = labels[mask]
            
            if len(X) < 50:
                return None
            
            # Scale features
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train model with optimized parameters
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_scaled, y)
            
            self.models[symbol] = model
            self.scalers[symbol] = scaler
            self.last_training_time[symbol] = datetime.now(IST)
            
            return model
            
        except Exception as e:
            return None
    
    def predict(self, df, symbol):
        """Make prediction for a symbol"""
        if symbol not in self.models or df.empty:
            return 0, 0.0
        
        try:
            features = self.create_features(df)
            if features.empty:
                return 0, 0.0
            
            latest = features.iloc[-1:].values
            
            scaled = self.scalers[symbol].transform(latest)
            prediction = self.models[symbol].predict(scaled)[0]
            proba = self.models[symbol].predict_proba(scaled)[0]
            confidence = max(proba)
            
            return prediction, confidence
            
        except Exception as e:
            return 0, 0.0
    
    def get_model_stats(self):
        """Get statistics about trained models"""
        return {
            'total_models': len(self.models),
            'recently_trained': sum(
                1 for t in self.last_training_time.values()
                if (datetime.now(IST) - t).total_seconds() < 3600
            ),
            'models': list(self.models.keys())[:10]  # First 10
        }

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
        self.last_reset_date = datetime.now(IST).date()
    
    def reset_daily_counts(self):
        """Reset daily counts if it's a new day"""
        today = datetime.now(IST).date()
        if today != self.last_reset_date:
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.last_reset_date = today
    
    def calculate_position_size(self, price, stop_loss):
        """Calculate position size based on risk"""
        self.reset_daily_counts()
        
        risk_amount = self.config.TOTAL_CAPITAL * self.config.RISK_PER_TRADE
        risk_per_share = abs(price - stop_loss)
        
        if risk_per_share <= 0:
            return 0
        
        quantity = int(risk_amount / risk_per_share)
        return max(1, quantity)
    
    def calculate_stop_loss(self, df, direction):
        """Calculate stop loss using ATR"""
        if df.empty:
            return 0
        
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
        self.reset_daily_counts()
        
        if len(self.positions) >= self.config.MAX_POSITIONS:
            return False, "Max positions reached (10)"
        
        if self.daily_trades >= self.config.MAX_DAILY_TRADES:
            return False, "Daily trade limit reached"
        
        return True, "OK"

# ============================================================================
# TRADING ENGINE - WITH AUTO-EXECUTE
# ============================================================================

class TradingEngine:
    """Main trading engine"""
    
    def __init__(self, config, demo_mode=True):
        self.config = config
        self.broker = KiteBroker(demo_mode)
        self.db = Database()
        self.risk = RiskManager(config)
        self.ai = AIEngine()
        
        self.running = False
        self.signals_queue = queue.Queue()
        self.last_scan_time = datetime.now(IST)
        self.scan_interval = 30  # seconds
        
        # Performance stats
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'daily_pnl': 0.0,
            'signals_generated': 0,
            'signals_executed': 0
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
                now = datetime.now(IST).time()
                if now < self.config.MARKET_OPEN or now > self.config.MARKET_CLOSE:
                    time.sleep(60)
                    continue
                
                # Scan for signals every interval
                time_since_scan = (datetime.now(IST) - self.last_scan_time).total_seconds()
                if time_since_scan >= self.scan_interval:
                    self.scan_signals()
                    self.last_scan_time = datetime.now(IST)
                
                # Auto-execute if enabled
                if hasattr(st.session_state, 'auto_execute') and st.session_state.auto_execute:
                    self.execute_signals()
                
                # Manage open positions
                self.manage_positions()
                
                scan_counter += 1
                time.sleep(5)  # Reduced sleep for faster response
                
            except Exception as e:
                time.sleep(10)
    
    def scan_signals(self):
        """Scan all F&O stocks for trading signals"""
        stocks = StockUniverse.get_all_fno_stocks()
        
        for symbol in stocks[:20]:  # Scan top 20 for speed
            try:
                can_trade, reason = self.risk.can_trade()
                if not can_trade:
                    break
                
                # Get historical data
                df = self.broker.get_historical(symbol, days=20)  # Reduced for speed
                if len(df) < 50:
                    continue
                
                # Train model if needed
                if symbol not in self.ai.models:
                    self.ai.train_model(df, symbol)
                
                # Get prediction
                prediction, confidence = self.ai.predict(df, symbol)
                
                # Check confidence threshold
                if confidence < self.config.MIN_CONFIDENCE:
                    continue
                
                # Skip HOLD signals
                if prediction == 0:
                    continue
                
                # Generate signal
                direction = 'LONG' if prediction == 1 else 'SHORT'
                current_price = self.broker.get_ltp(symbol)
                stop_loss = self.risk.calculate_stop_loss(df, direction)
                take_profit = self.risk.calculate_take_profit(
                    current_price, stop_loss, direction
                )
                quantity = self.risk.calculate_position_size(
                    current_price, stop_loss
                )
                
                signal = {
                    'symbol': symbol,
                    'direction': direction,
                    'price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'quantity': quantity,
                    'confidence': confidence,
                    'timestamp': datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Save to database
                signal_id = self.db.save_signal(signal)
                signal['id'] = signal_id
                
                self.signals_queue.put(signal)
                self.stats['signals_generated'] += 1
                
            except Exception as e:
                continue  # Silent fail for individual stocks
    
    def execute_signals(self):
        """Execute pending signals"""
        executed_count = 0
        
        while not self.signals_queue.empty() and executed_count < 3:  # Max 3 at once
            signal = self.signals_queue.get()
            
            try:
                can_trade, reason = self.risk.can_trade()
                if not can_trade:
                    self.signals_queue.put(signal)  # Put back in queue
                    break
                
                # Check if already in position
                if signal['symbol'] in self.risk.positions:
                    continue
                
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
                        'entry_time': datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S'),
                        'entry_price': result['price'],
                        'quantity': signal['quantity'],
                        'stop_loss': signal['stop_loss'],
                        'take_profit': signal['take_profit'],
                        'status': 'OPEN',
                        'confidence': signal['confidence']
                    }
                    
                    self.risk.positions[signal['symbol']] = position
                    self.db.save_position(position)
                    self.risk.daily_trades += 1
                    
                    # Mark signal as executed
                    if 'id' in signal:
                        self.db.mark_signal_executed(signal['id'])
                    
                    self.stats['signals_executed'] += 1
                    executed_count += 1
                    
            except Exception as e:
                continue
    
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
                continue
    
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
        exit_dir = 'SELL' if pos['direction'] == 'LONG' else 'BUY'
        self.broker.place_order(symbol, exit_dir, pos['quantity'], price)
        
        # Update stats
        self.stats['total_trades'] += 1
        if pnl > 0:
            self.stats['winning_trades'] += 1
        else:
            self.stats['losing_trades'] += 1
        
        self.stats['total_pnl'] += pnl
        self.stats['daily_pnl'] += pnl
        self.stats['win_rate'] = (
            self.stats['winning_trades'] / self.stats['total_trades'] * 100
            if self.stats['total_trades'] > 0 else 0
        )
        
        # Save trade
        trade = pos.copy()
        trade['exit_time'] = datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')
        trade['exit_price'] = price
        trade['pnl'] = pnl
        trade['pnl_pct'] = pnl_pct
        trade['status'] = 'CLOSED'
        self.db.save_trade(trade)
        
        # Close position
        self.db.close_position(symbol)
        del self.risk.positions[symbol]
    
    def get_pending_signals_count(self):
        """Get count of pending signals"""
        return self.signals_queue.qsize()
    
    def clear_signals(self):
        """Clear all pending signals"""
        while not self.signals_queue.empty():
            self.signals_queue.get()
        self.stats['signals_generated'] = 0

# ============================================================================
# COLORFUL BUTTON TABS - FIXED VERSION
# ============================================================================

def render_colorful_tabs():
    """Render colorful button-style tabs"""
    
    tab_config = [
        {"name": "🎯 Algo Trading", "color": "#1E88E5", "key": "tab_algo"},
        {"name": "📈 Positions", "color": "#4CAF50", "key": "tab_positions"},
        {"name": "📋 Trade History", "color": "#FF9800", "key": "tab_history"},
        {"name": "📊 Live Charts", "color": "#9C27B0", "key": "tab_charts"},
        {"name": "📉 Analytics", "color": "#F44336", "key": "tab_analytics"},
        {"name": "⚙️ Settings", "color": "#607D8B", "key": "tab_settings"}
    ]
    
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0
    
    cols = st.columns(len(tab_config))
    
    for idx, (col, tab) in enumerate(zip(cols, tab_config)):
        with col:
            is_active = st.session_state.active_tab == idx
            
            if st.button(
                tab['name'],
                key=tab['key'],
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                st.session_state.active_tab = idx
                st.rerun()
    
    return st.session_state.active_tab

# ============================================================================
# AUTO-REFRESH COMPONENT - FIXED VERSION
# ============================================================================

def auto_refresh_component(refresh_interval=5):
    """Auto-refresh component for market data"""
    
    # Initialize refresh tracking
    if 'last_auto_refresh' not in st.session_state:
        st.session_state.last_auto_refresh = datetime.now(IST)
        st.session_state.auto_refresh_counter = 0
    
    # Calculate time since last refresh
    current_time = datetime.now(IST)
    time_since_refresh = (current_time - st.session_state.last_auto_refresh).total_seconds()
    
    # Check if refresh is needed
    if time_since_refresh >= refresh_interval:
        # Update timestamp
        st.session_state.last_auto_refresh = current_time
        st.session_state.auto_refresh_counter += 1
        
        # Set a flag to indicate refresh is needed
        if 'auto_refresh_needed' not in st.session_state:
            st.session_state.auto_refresh_needed = True
        else:
            st.session_state.auto_refresh_needed = True
        
        return True
    
    # Show countdown in sidebar
    with st.sidebar:
        remaining = max(0, refresh_interval - time_since_refresh)
        progress = time_since_refresh / refresh_interval
        st.progress(min(progress, 1.0), text=f"🔄 Next refresh in {int(remaining)}s")
    
    return False

# ============================================================================
# MAIN STREAMLIT APP - COMPLETE FIXED VERSION
# ============================================================================

def main():
    st.set_page_config(
        page_title="AI Algo Trading Bot v8.1",
        page_icon="🤖",
        layout="wide"
    )
    
    # Custom CSS for enhanced UI
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
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.2);
    }
    .metric-card h3 {
        font-size: 0.9rem;
        color: #888;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
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
        border-radius: 10px !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .tab-button-active {
        background: linear-gradient(135deg, #1E88E5, #4CAF50) !important;
        color: white !important;
        border: none !important;
        font-weight: bold !important;
    }
    .tab-button-inactive {
        background: #2d3748 !important;
        color: #a0aec0 !important;
        border: 1px solid #4a5568 !important;
    }
    .market-mood-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        height: 100%;
    }
    .signal-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 2px;
    }
    .signal-long {
        background: linear-gradient(135deg, #00C853, #4CAF50);
        color: white;
    }
    .signal-short {
        background: linear-gradient(135deg, #FF5252, #F44336);
        color: white;
    }
    .live-badge {
        display: inline-block;
        padding: 4px 8px;
        background: #FF5252;
        color: white;
        border-radius: 10px;
        font-size: 0.7rem;
        animation: blink 1s infinite;
    }
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    .clock-widget {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
    }
    .clock-time {
        font-size: 1.5rem;
        font-weight: bold;
        color: #4CAF50;
        font-family: 'Courier New', monospace;
    }
    .clock-date {
        font-size: 0.9rem;
        color: #888;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'engine' not in st.session_state:
        st.session_state.engine = TradingEngine(Config(), demo_mode=True)
        st.session_state.market_indices = MarketIndicesUpdater(st.session_state.engine.broker)
        st.session_state.last_refresh = datetime.now(IST)
        st.session_state.auto_execute = False
        st.session_state.auto_refresh = True
        st.session_state.refresh_rate = 5  # 5 seconds for market data
        st.session_state.active_tab = 0
        st.session_state.auto_refresh_counter = 0
        st.session_state.last_auto_refresh = datetime.now(IST)
        
        # Auto-connect if credentials exist
        if 'kite_api_key' in st.session_state and 'kite_access_token' in st.session_state:
            with st.spinner("🔌 Auto-connecting to Kite..."):
                st.session_state.engine.broker.connect()
    
    engine = st.session_state.engine
    market_indices = st.session_state.market_indices
    
    # AUTO REFRESH LOGIC - FIXED
    if st.session_state.get('auto_refresh', True):
        refresh_needed = auto_refresh_component(st.session_state.refresh_rate)
        
        if refresh_needed:
            # Update market indices
            market_indices.update_from_kite()
            
            # Update LTP cache for all positions
            for symbol in list(engine.risk.positions.keys()):
                try:
                    engine.broker.get_ltp(symbol)  # Updates cache
                except:
                    pass
            
            # Force rerun for UI update
            st.rerun()
    
    # Header
    st.markdown("<h1 class='main-header'>🤖 AI ALGORITHMIC TRADING BOT v8.1</h1>", 
                unsafe_allow_html=True)
    st.markdown("### Professional Trading System | All 159 F&O Stocks | Live Market Data | Indian Timezone")
    
    # Clock Widget
    current_time = datetime.now(IST)
    col_clock1, col_clock2, col_clock3 = st.columns(3)
    
    with col_clock1:
        st.markdown(f"""
        <div class="clock-widget">
            <div class="clock-time">{current_time.strftime('%H:%M:%S')}</div>
            <div class="clock-date">{current_time.strftime('%d %b %Y')}</div>
            <div style="font-size: 0.8rem; color: #4CAF50;">⏰ IST Live</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_clock2:
        market_status = "🔴 Closed"
        market_color = "#FF5252"
        now_time = current_time.time()
        if Config.MARKET_OPEN <= now_time <= Config.MARKET_CLOSE:
            market_status = "🟢 Open"
            market_color = "#00C853"
        
        st.markdown(f"""
        <div class="clock-widget">
            <div style="font-size: 1.2rem; color: {market_color}; font-weight: bold;">{market_status}</div>
            <div class="clock-date">Market Hours: 09:15 - 15:30</div>
            <div style="font-size: 0.8rem; color: #888;">NSE | BSE</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_clock3:
        time_since_refresh = (datetime.now(IST) - st.session_state.last_auto_refresh).total_seconds()
        next_refresh = max(0, st.session_state.refresh_rate - time_since_refresh)
        st.markdown(f"""
        <div class="clock-widget">
            <div class="clock-time">{int(next_refresh)}s</div>
            <div class="clock-date">Next Refresh</div>
            <div style="font-size: 0.8rem; color: #1E88E5;">🔄 Auto: {'ON' if st.session_state.auto_refresh else 'OFF'}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ============================================================================
    # MARKET MOOD GAUGE - FIXED WITH LIVE SYMBOLS
    # ============================================================================
    st.markdown("---")
    st.markdown("### 📊 Live Market Dashboard")
    
    # Get current market data
    nifty_data = market_indices.indices_data['NIFTY 50']
    banknifty_data = market_indices.indices_data['NIFTY BANK']
    sensex_data = market_indices.indices_data['SENSEX']
    mood, mood_color, avg_change = market_indices.get_market_mood()
    
    # Show live badge if WebSocket is active
    live_status = ""
    if engine.broker.websocket_running:
        ws_age = (datetime.now(IST) - engine.broker.last_ws_update).total_seconds()
        if ws_age < 10:
            live_status = '<span class="live-badge">LIVE</span>'
    
    col_idx1, col_idx2, col_idx3, col_idx4 = st.columns(4)
    
    with col_idx1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        change_color = "status-running" if nifty_data['change'] >= 0 else "status-stopped"
        st.markdown(f'<h3>📈 NIFTY 50 {live_status}</h3>', unsafe_allow_html=True)
        st.markdown(f'<h2>{nifty_data["ltp"]:,.2f}</h2>', unsafe_allow_html=True)
        st.markdown(f'<p class="{change_color}">{"+" if nifty_data["change"] >= 0 else ""}{nifty_data["change"]:+.2f} ({nifty_data["change_pct"]:+.2f}%)</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size: 0.8rem; color: #888;">{nifty_data["timestamp"].strftime("%H:%M:%S")} IST</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_idx2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        change_color = "status-running" if banknifty_data['change'] >= 0 else "status-stopped"
        st.markdown(f'<h3>🏦 NIFTY BANK {live_status}</h3>', unsafe_allow_html=True)
        st.markdown(f'<h2>{banknifty_data["ltp"]:,.2f}</h2>', unsafe_allow_html=True)
        st.markdown(f'<p class="{change_color}">{"+" if banknifty_data["change"] >= 0 else ""}{banknifty_data["change"]:+.2f} ({banknifty_data["change_pct"]:+.2f}%)</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size: 0.8rem; color: #888;">{banknifty_data["timestamp"].strftime("%H:%M:%S")} IST</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_idx3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        change_color = "status-running" if sensex_data['change'] >= 0 else "status-stopped"
        st.markdown(f'<h3>📊 SENSEX {live_status}</h3>', unsafe_allow_html=True)
        st.markdown(f'<h2>{sensex_data["ltp"]:,.2f}</h2>', unsafe_allow_html=True)
        st.markdown(f'<p class="{change_color}">{"+" if sensex_data["change"] >= 0 else ""}{sensex_data["change"]:+.2f} ({sensex_data["change_pct"]:+.2f}%)</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size: 0.8rem; color: #888;">{sensex_data["timestamp"].strftime("%H:%M:%S")} IST</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_idx4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<h3>🎯 MARKET MOOD</h3>', unsafe_allow_html=True)
        st.markdown(f'<h2 style="color: {mood_color}">{mood}</h2>', unsafe_allow_html=True)
        st.markdown(f'<p>Avg Change: {avg_change:+.2f}%</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size: 0.8rem; color: #888;">Auto-refresh: {st.session_state.refresh_rate}s</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar Controls
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
        
        # Quick Stats
        st.markdown("### 📊 Quick Stats")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric("Positions", len(engine.risk.positions))
            st.metric("Daily Trades", engine.risk.daily_trades)
        with col_s2:
            st.metric("Signals", engine.get_pending_signals_count())
            st.metric("AI Models", len(engine.ai.models))
        
        st.markdown("---")
        
        # Trading Mode
        mode = st.radio("Trading Mode", 
                       ["📈 Paper Trading", "💰 Live Trading"], 
                       index=0,
                       help="Paper trading uses demo data, Live connects to Kite")
        engine.broker.demo_mode = "Paper" in mode
        
        # Capital - FIXED: Removed the slider error
        capital = st.number_input("Capital (₹)", 
                                 min_value=100000, 
                                 value=Config.TOTAL_CAPITAL, 
                                 step=100000,
                                 help="Total trading capital")
        Config.TOTAL_CAPITAL = capital
        
        # Risk - FIXED: Type consistency
        risk = st.slider("Risk per Trade (%)", 0.5, 5.0, 
                        float(Config.RISK_PER_TRADE * 100), 0.1) / 100
        Config.RISK_PER_TRADE = risk
        
        # Confidence - FIXED: Type consistency (this was the error line)
        confidence = st.slider("Min Confidence (%)", 50, 90, 
                              int(Config.MIN_CONFIDENCE * 100), 5) / 100
        Config.MIN_CONFIDENCE = confidence
        
        st.markdown("---")
        
        # Feature Toggles
        st.markdown("### 🔧 Trading Features")
        
        auto_exec = st.checkbox("🤖 Auto-Execute Signals", 
                               value=st.session_state.auto_execute,
                               help="Automatically execute AI signals")
        st.session_state.auto_execute = auto_exec
        
        st.markdown("---")
        
        # Auto Refresh Settings
        st.markdown("### 🔄 Auto Refresh")
        
        auto_refresh = st.checkbox("Enable Auto Refresh", 
                                  value=st.session_state.auto_refresh,
                                  help="Auto-refresh market data")
        st.session_state.auto_refresh = auto_refresh
        
        if auto_refresh:
            refresh_rate = st.slider("Refresh Rate (seconds)", 1, 30, 
                                    int(st.session_state.refresh_rate))
            st.session_state.refresh_rate = refresh_rate
        
        # Manual Refresh Button
        if st.button("🔄 Refresh Now", use_container_width=True, type="secondary"):
            market_indices.update_from_kite()
            st.success("✅ Market data refreshed!")
            st.rerun()
        
        st.markdown("---")
        
        # Market Data Status
        st.markdown("### 📈 Market Data")
        if engine.broker.connected:
            if engine.broker.websocket_running:
                st.success("✅ Live WebSocket")
            else:
                st.success("✅ API Connected")
        else:
            st.warning("⚠️ Demo Mode")
        
        # Stock Universe Info
        st.markdown("### 📊 Stock Universe")
        total_stocks = len(StockUniverse.get_all_fno_stocks())
        scanned = min(20, total_stocks)  # Currently scanning top 20
        st.info(f"**Scanning:** {scanned}/{total_stocks} stocks")
        
        # Signal Controls
        st.markdown("---")
        st.markdown("### 📡 Signal Controls")
        if st.button("🧹 Clear Signals", use_container_width=True):
            engine.clear_signals()
            st.success("✅ Signals cleared!")
            st.rerun()
        
        if st.button("⚡ Force Scan", use_container_width=True):
            engine.scan_signals()
            st.success("✅ Scan completed!")
            st.rerun()
        
        # Debug Info
        st.markdown("---")
        st.markdown("### 🐛 Debug Info")
        st.info(f"Auto-refresh counter: {st.session_state.get('auto_refresh_counter', 0)}")
        if st.button("🔄 Force Rerun"):
            st.rerun()
    
    # Top Metrics
    st.markdown("### 📊 Bot Performance Dashboard")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        status = "RUNNING" if engine.running else "STOPPED"
        status_class = "status-running" if engine.running else "status-stopped"
        st.markdown(f'<h3 class="{status_class}">● {status}</h3>', unsafe_allow_html=True)
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
        st.markdown(f'<h3>{positions}/{Config.MAX_POSITIONS}</h3>', unsafe_allow_html=True)
        st.markdown("**Active Positions**")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col5:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        signals = engine.get_pending_signals_count()
        st.markdown(f'<h3>{signals}</h3>', unsafe_allow_html=True)
        st.markdown("**Pending Signals**")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ============================================================================
    # COLORFUL BUTTON TABS - FIXED VERSION
    # ============================================================================
    st.markdown("---")
    active_tab = render_colorful_tabs()
    st.markdown("---")
    
    # Tab Content
    if active_tab == 0:  # Algo Trading
        st.markdown("### 🎯 AI Algorithm Trading Signals")
        
        # Signal Generation Controls
        col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
        with col1:
            if st.button("🔍 Scan Top 20 Stocks", type="primary", use_container_width=True):
                with st.spinner("🔍 Scanning for signals..."):
                    engine.scan_signals()
                    st.success(f"✅ Scan complete! Found {engine.get_pending_signals_count()} signals")
                    st.rerun()
        
        with col2:
            if st.button("⚡ Quick Execute", type="secondary", use_container_width=True):
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
                            position = {
                                'symbol': signal['symbol'],
                                'direction': signal['direction'],
                                'entry_time': datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S'),
                                'entry_price': result['price'],
                                'quantity': signal['quantity'],
                                'stop_loss': signal['stop_loss'],
                                'take_profit': signal['take_profit'],
                                'status': 'OPEN',
                                'confidence': signal['confidence']
                            }
                            engine.risk.positions[signal['symbol']] = position
                            engine.db.save_position(position)
                            engine.risk.daily_trades += 1
                            
                            if 'id' in signal:
                                engine.db.mark_signal_executed(signal['id'])
                            
                            executed += 1
                    except:
                        pass
                st.success(f"✅ Executed {executed} signals!")
                st.rerun()
        
        with col3:
            if st.button("🧹 Clear", use_container_width=True):
                engine.clear_signals()
                st.success("✅ Signals cleared!")
                st.rerun()
        
        with col4:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        
        # Show pending signals
        st.markdown("#### 📋 Pending Signals")
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
                df_signals = pd.DataFrame(signals)
                df_signals['confidence_pct'] = df_signals['confidence'].apply(lambda x: f"{x:.1%}")
                df_signals['direction_badge'] = df_signals['direction'].apply(
                    lambda x: f'<span class="signal-badge signal-{x.lower()}">{x}</span>'
                )
                
                # Display with columns
                st.markdown(f"**Found {len(signals)} signals**")
                
                for _, signal in df_signals.iterrows():
                    col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns([2, 1, 2, 2, 1])
                    with col_s1:
                        st.markdown(f"**{signal['symbol']}**")
                    with col_s2:
                        st.markdown(f'<div class="signal-badge signal-{signal["direction"].lower()}">{signal["direction"]}</div>', 
                                  unsafe_allow_html=True)
                    with col_s3:
                        st.markdown(f"₹{signal['price']:.2f}")
                    with col_s4:
                        st.markdown(f"Confidence: **{signal['confidence_pct']}**")
                    with col_s5:
                        if st.button("📈 View", key=f"view_{signal['symbol']}"):
                            st.session_state.chart_symbol = signal['symbol']
                            st.session_state.active_tab = 3
                            st.rerun()
                st.markdown("---")
        else:
            st.info("🔭 No pending signals. Click 'Scan Top 20 Stocks' to generate signals.")
        
        # AI Model Status
        st.markdown("#### 🧠 AI Model Status")
        model_stats = engine.ai.get_model_stats()
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Models Trained", model_stats['total_models'])
        with col_m2:
            st.metric("Recently Trained", model_stats['recently_trained'])
        with col_m3:
            st.metric("Confidence Threshold", f"{Config.MIN_CONFIDENCE:.0%}")
        
        # Recent Signals from Database
        st.markdown("#### 📝 Recent Signal History")
        recent_signals = engine.db.get_recent_signals(10)
        if not recent_signals.empty:
            st.dataframe(recent_signals[['symbol', 'direction', 'confidence', 'generated_time', 'executed']], 
                        use_container_width=True)
        else:
            st.info("No recent signals in history")
    
    elif active_tab == 1:  # Positions
        st.markdown("### 📈 Active Positions")
        
        positions_df = engine.db.get_open_positions()
        
        if not positions_df.empty:
            # Add live prices and P&L
            for idx, row in positions_df.iterrows():
                current_price = engine.broker.get_ltp(row['symbol'])
                if row['direction'] == 'LONG':
                    pnl = (current_price - row['entry_price']) * row['quantity']
                else:
                    pnl = (row['entry_price'] - current_price) * row['quantity']
                
                positions_df.at[idx, 'current_price'] = current_price
                positions_df.at[idx, 'pnl'] = pnl
                positions_df.at[idx, 'pnl_pct'] = (pnl / (row['entry_price'] * row['quantity'])) * 100
            
            # Display with formatting
            display_df = positions_df.copy()
            display_df['pnl_display'] = display_df['pnl'].apply(lambda x: f"₹{x:+,.0f}")
            display_df['pnl_pct_display'] = display_df['pnl_pct'].apply(lambda x: f"{x:+.2f}%")
            display_df['entry_price'] = display_df['entry_price'].apply(lambda x: f"₹{x:.2f}")
            display_df['current_price'] = display_df['current_price'].apply(lambda x: f"₹{x:.2f}")
            
            st.dataframe(display_df[['symbol', 'direction', 'entry_price', 'current_price', 
                                    'quantity', 'pnl_display', 'pnl_pct_display', 'confidence']], 
                        use_container_width=True)
            
            # Manual exit buttons
            st.markdown("#### 🛑 Manual Exit")
            cols = st.columns(min(4, len(positions_df)))
            for idx, (_, row) in enumerate(positions_df.iterrows()):
                if idx < 4:  # Show max 4 buttons
                    with cols[idx]:
                        if st.button(f"Exit {row['symbol']}", key=f"exit_{row['symbol']}"):
                            price = engine.broker.get_ltp(row['symbol'])
                            engine.exit_position(row['symbol'], price, 'MANUAL')
                            st.success(f"✅ Position closed: {row['symbol']}")
                            st.rerun()
        else:
            st.info("🔭 No active positions")
    
    elif active_tab == 2:  # History
        st.markdown("### 📋 Trade History")
        
        # Date filter
        col1, col2 = st.columns(2)
        with col1:
            days_back = st.slider("Show trades from last N days", 1, 30, 7)
        
        # Get trades
        all_trades = engine.db.get_trades(1000)  # Get more than needed
        if not all_trades.empty:
            all_trades['entry_time'] = pd.to_datetime(all_trades['entry_time'])
            cutoff_date = datetime.now(IST) - timedelta(days=days_back)
            recent_trades = all_trades[all_trades['entry_time'] >= cutoff_date]
            
            if not recent_trades.empty:
                # Format for display
                display_trades = recent_trades.copy()
                display_trades['pnl_display'] = display_trades['pnl'].apply(lambda x: f"₹{x:+,.0f}")
                display_trades['pnl_pct_display'] = display_trades['pnl_pct'].apply(lambda x: f"{x:+.2f}%")
                display_trades['entry_price'] = display_trades['entry_price'].apply(lambda x: f"₹{x:.2f}")
                display_trades['exit_price'] = display_trades['exit_price'].apply(lambda x: f"₹{x:.2f}" if pd.notna(x) else "")
                
                st.dataframe(display_trades[['symbol', 'direction', 'entry_time', 'entry_price', 
                                           'exit_price', 'quantity', 'pnl_display', 'pnl_pct_display', 
                                           'confidence']], 
                           use_container_width=True,
                           height=400)
                
                # Summary stats
                st.markdown("#### 📊 Trade Summary")
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                with col_s1:
                    st.metric("Total Trades", len(recent_trades))
                with col_s2:
                    winning = len(recent_trades[recent_trades['pnl'] > 0])
                    st.metric("Winning Trades", winning)
                with col_s3:
                    total_pnl = recent_trades['pnl'].sum()
                    st.metric("Total P&L", f"₹{total_pnl:+,.0f}")
                with col_s4:
                    if len(recent_trades) > 0:
                        win_rate = (winning / len(recent_trades)) * 100
                        st.metric("Win Rate", f"{win_rate:.1f}%")
                
                # Export
                csv = recent_trades.to_csv(index=False)
                st.download_button(
                    "📥 Export CSV",
                    csv,
                    f"trades_{datetime.now(IST).strftime('%Y%m%d_%H%M')}.csv",
                    "text/csv",
                    key="export_csv"
                )
            else:
                st.info("🔭 No trades in selected period")
        else:
            st.info("🔭 No trade history")
    
    elif active_tab == 3:  # Charts
        st.markdown("### 📊 Live Charts & Analysis")
        
        # Chart selection
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            # Get symbol from session state or default
            default_symbol = st.session_state.get('chart_symbol', 'RELIANCE')
            symbol = st.selectbox(
                "Select Stock",
                StockUniverse.get_all_fno_stocks()[:50],
                index=StockUniverse.get_all_fno_stocks()[:50].index(default_symbol) if default_symbol in StockUniverse.get_all_fno_stocks()[:50] else 0
            )
        
        with col2:
            timeframe = st.selectbox(
                "Timeframe",
                ["1D", "5D", "1M", "3M"],
                index=1
            )
        
        with col3:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        
        # Get data based on timeframe
        days_map = {"1D": 1, "5D": 5, "1M": 30, "3M": 90}
        df = engine.broker.get_historical(symbol, days=days_map[timeframe])
        
        if PLOTLY_AVAILABLE and not df.empty:
            # Create chart
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.6, 0.2, 0.2],
                subplot_titles=(f"{symbol} - Price Chart", "Volume", "RSI")
            )
            
            # Candlestick
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='OHLC',
                    increasing_line_color='#26a69a',
                    decreasing_line_color='#ef5350'
                ),
                row=1, col=1
            )
            
            # Calculate indicators
            df = TechnicalAnalysis.calculate_indicators(df)
            
            # Moving averages
            if 'SMA20' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['SMA20'], 
                             name='SMA20', line=dict(color='orange', width=1)),
                    row=1, col=1
                )
            
            if 'EMA50' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['EMA50'], 
                             name='EMA50', line=dict(color='red', width=1)),
                    row=1, col=1
                )
            
            # Volume
            colors = ['#ef5350' if df['Close'].iloc[i] < df['Open'].iloc[i] 
                     else '#26a69a' for i in range(len(df))]
            
            fig.add_trace(
                go.Bar(x=df.index, y=df['Volume'], name='Volume',
                      marker_color=colors),
                row=2, col=1
            )
            
            # RSI
            if 'RSI' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                             line=dict(color='purple', width=2)),
                    row=3, col=1
                )
                # Add RSI lines
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
                fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)
            
            fig.update_layout(
                title=f"{symbol} - {timeframe} Chart | Live Analysis",
                template='plotly_dark',
                height=800,
                showlegend=True,
                xaxis_rangeslider_visible=False
            )
            
            fig.update_xaxes(title_text="Time", row=3, col=1)
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            fig.update_yaxes(title_text="RSI", row=3, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Technical indicators panel
            st.markdown("#### 📊 Technical Indicators")
            col_t1, col_t2, col_t3, col_t4, col_t5 = st.columns(5)
            
            with col_t1:
                current = engine.broker.get_ltp(symbol)
                st.metric("LTP", f"₹{current:.2f}")
            with col_t2:
                st.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}" if 'RSI' in df.columns else "N/A")
            with col_t3:
                st.metric("ATR", f"₹{df['ATR'].iloc[-1]:.2f}" if 'ATR' in df.columns else "N/A")
            with col_t4:
                st.metric("MACD", f"{df['MACD'].iloc[-1]:.2f}" if 'MACD' in df.columns else "N/A")
            with col_t5:
                change = ((current - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100) if len(df) > 1 else 0
                st.metric("Change", f"{change:+.2f}%")
            
            # AI Analysis
            st.markdown("#### 🤖 AI Analysis")
            prediction, confidence = engine.ai.predict(df, symbol)
            
            if prediction != 0 and confidence > 0:
                direction = "LONG" if prediction == 1 else "SHORT"
                col_a1, col_a2, col_a3 = st.columns(3)
                with col_a1:
                    st.metric("Prediction", direction)
                with col_a2:
                    st.metric("Confidence", f"{confidence:.1%}")
                with col_a3:
                    if st.button("Generate Signal", key="gen_signal"):
                        stop_loss = engine.risk.calculate_stop_loss(df, direction)
                        take_profit = engine.risk.calculate_take_profit(current, stop_loss, direction)
                        quantity = engine.risk.calculate_position_size(current, stop_loss)
                        
                        signal = {
                            'symbol': symbol,
                            'direction': direction,
                            'price': current,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'quantity': quantity,
                            'confidence': confidence,
                            'timestamp': datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')
                        }
                        engine.db.save_signal(signal)
                        st.success("✅ Signal generated!")
            else:
                st.info("AI model not trained or insufficient confidence")
        
        else:
            st.error("Chart data unavailable")
    
    elif active_tab == 4:  # Analytics
        st.markdown("### 📉 Performance Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Trade Distribution")
            
            if PLOTLY_AVAILABLE:
                labels = ['Winning', 'Losing', 'Breakeven']
                winning = engine.stats['winning_trades']
                losing = engine.stats['losing_trades']
                total = engine.stats['total_trades']
                breakeven = total - winning - losing
                
                fig = go.Figure(data=[
                    go.Pie(
                        labels=labels,
                        values=[winning, losing, breakeven],
                        hole=.4,
                        marker_colors=['#00C853', '#FF5252', '#FFC107']
                    )
                ])
                fig.update_layout(
                    template='plotly_dark',
                    title="Trade Distribution",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📈 Key Metrics")
            
            st.metric("Total Trades", engine.stats['total_trades'])
            st.metric("Win Rate", f"{engine.stats['win_rate']:.1f}%")
            st.metric("Total P&L", f"₹{engine.stats['total_pnl']:+,.0f}")
            st.metric("Daily P&L", f"₹{engine.stats['daily_pnl']:+,.0f}")
            st.metric("Daily Trades", engine.risk.daily_trades)
            st.metric("Signals Generated", engine.stats['signals_generated'])
            st.metric("Signals Executed", engine.stats['signals_executed'])
        
        # P&L Over Time Chart
        st.markdown("#### 📅 P&L Over Time")
        trades_df = engine.db.get_trades(100)
        
        if not trades_df.empty and PLOTLY_AVAILABLE:
            trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
            trades_df = trades_df.sort_values('entry_time')
            trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trades_df['entry_time'],
                y=trades_df['cumulative_pnl'],
                mode='lines+markers',
                name='Cumulative P&L',
                line=dict(color='#4CAF50', width=3),
                fill='tozeroy',
                fillcolor='rgba(76, 175, 80, 0.2)'
            ))
            
            fig.update_layout(
                title="Cumulative P&L Over Time",
                template='plotly_dark',
                height=400,
                xaxis_title="Date",
                yaxis_title="P&L (₹)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk Metrics
        st.markdown("#### ⚠️ Risk Metrics")
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        with col_r1:
            risk_per_trade = Config.RISK_PER_TRADE * 100
            st.metric("Risk per Trade", f"{risk_per_trade:.1f}%")
        with col_r2:
            max_drawdown = 0  # Calculate if you have the data
            st.metric("Max Drawdown", f"{max_drawdown:.1f}%")
        with col_r3:
            sharpe_ratio = 0  # Calculate if you have the data
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        with col_r4:
            avg_win_loss = 0  # Calculate if you have the data
            st.metric("Avg Win/Loss", f"{avg_win_loss:.2f}")
    
    elif active_tab == 5:  # Settings
        st.markdown("### ⚙️ Settings & Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔑 Kite Connect Setup")
            
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
                        - Broker: {profile['broker']}
                        """)
                except:
                    pass
                
                if st.button("🔌 Disconnect", type="secondary"):
                    engine.broker.connected = False
                    engine.broker.kite = None
                    engine.broker.demo_mode = True
                    if 'kite_api_key' in st.session_state:
                        del st.session_state.kite_api_key
                    if 'kite_access_token' in st.session_state:
                        del st.session_state.kite_access_token
                    st.warning("Disconnected. Running in demo mode.")
                    st.rerun()
            
            else:
                # Kite setup form (kept for brevity)
                st.warning("⚠️ Not connected to Kite")
                st.markdown("Connect to Kite in the settings section to enable live trading.")
        
        with col2:
            st.markdown("#### 📊 Bot Configuration")
            
            with st.form("config_form"):
                st.markdown("**Capital Management**")
                new_capital = st.number_input(
                    "Total Capital (₹)",
                    min_value=100000,
                    value=Config.TOTAL_CAPITAL,
                    step=100000
                )
                
                # FIXED: All sliders now have consistent float types
                new_risk = st.slider(
                    "Risk per Trade (%)",
                    0.5, 5.0, float(Config.RISK_PER_TRADE * 100), 0.1
                )
                
                st.markdown("**Position Limits**")
                new_max_positions = st.slider(
                    "Max Concurrent Positions",
                    1, 20, Config.MAX_POSITIONS
                )
                
                new_max_trades = st.slider(
                    "Max Daily Trades",
                    10, 100, Config.MAX_DAILY_TRADES, 5
                )
                
                st.markdown("**AI Parameters**")
                # FIXED: Convert to int for slider, then back to float
                new_confidence = st.slider(
                    "Min Confidence (%)",
                    50, 95, int(Config.MIN_CONFIDENCE * 100), 5
                )
                
                st.markdown("**Risk Management**")
                new_atr_mult = st.slider(
                    "ATR Multiplier",
                    1.0, 5.0, float(Config.ATR_MULTIPLIER), 0.5
                )
                
                new_rr_ratio = st.slider(
                    "Risk:Reward Ratio",
                    1.5, 5.0, float(Config.TAKE_PROFIT_RATIO), 0.5
                )
                
                trailing_stop = st.checkbox(
                    "Enable Trailing Stop",
                    value=Config.TRAILING_STOP
                )
                
                if st.form_submit_button("💾 Save Configuration", type="primary"):
                    Config.TOTAL_CAPITAL = new_capital
                    Config.RISK_PER_TRADE = new_risk / 100  # Convert percentage to decimal
                    Config.MAX_POSITIONS = new_max_positions
                    Config.MAX_DAILY_TRADES = new_max_trades
                    Config.MIN_CONFIDENCE = new_confidence / 100  # Convert percentage to decimal
                    Config.ATR_MULTIPLIER = new_atr_mult
                    Config.TAKE_PROFIT_RATIO = new_rr_ratio
                    Config.TRAILING_STOP = trailing_stop
                    
                    st.success("✅ Configuration saved!")
                    st.rerun()
            
            st.markdown("---")
            st.markdown("#### ⚡ Auto-Execution")
            
            auto_exec = st.checkbox(
                "🤖 Auto-Execute Signals",
                value=st.session_state.get('auto_execute', False),
                help="Auto-execute signals when confidence > threshold"
            )
            st.session_state.auto_execute = auto_exec
            
            if auto_exec:
                st.success("✅ Auto-execution ENABLED")
            else:
                st.warning("⚠️ Auto-execution DISABLED")
            
            st.markdown("---")
            st.markdown("#### 📱 System Info")
            
            # Get system stats
            trade_count = len(engine.db.get_trades(1000))
            
            st.info(f"""
            **Status:** {'🟢 Running' if engine.running else '🔴 Stopped'}
            **Mode:** {'💰 Live' if not engine.broker.demo_mode else '📈 Paper'}
            **Kite:** {'🟢 Connected' if engine.broker.connected else '🔴 Disconnected'}
            **WebSocket:** {'🟢 Active' if engine.broker.websocket_running else '🔴 Inactive'}
            **Auto-Execute:** {'🟢 ON' if st.session_state.get('auto_execute', False) else '🔴 OFF'}
            **Models Trained:** {len(engine.ai.models)}
            **Stock Universe:** {len(StockUniverse.get_all_fno_stocks())} stocks
            **Total Trades:** {trade_count}
            **Auto-refresh count:** {st.session_state.get('auto_refresh_counter', 0)}
            """)
            
            # Database management
            st.markdown("#### 💾 Database Management")
            
            if st.button("🗑️ Clear All Data", type="secondary"):
                st.warning("⚠️ This will delete ALL trading data!")
                confirm = st.checkbox("I understand this cannot be undone")
                if confirm:
                    if st.button("Yes, Delete Everything", type="primary"):
                        try:
                            conn = engine.db.conn
                            cursor = conn.cursor()
                            cursor.execute("DELETE FROM trades")
                            cursor.execute("DELETE FROM positions")
                            cursor.execute("DELETE FROM signals")
                            cursor.execute("VACUUM")
                            conn.commit()
                            
                            # Reset engine stats
                            engine.stats = {
                                'total_trades': 0,
                                'winning_trades': 0,
                                'losing_trades': 0,
                                'total_pnl': 0.0,
                                'win_rate': 0.0,
                                'daily_pnl': 0.0,
                                'signals_generated': 0,
                                'signals_executed': 0
                            }
                            engine.risk.positions = {}
                            engine.risk.daily_trades = 0
                            
                            st.success("✅ Database cleared and reset!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🚨 <b>DISCLAIMER:</b> This is for educational purposes only. Trading involves substantial risk of loss.</p>
    <p>© 2024 AI Algo Trading Bot v8.1 | Complete F&O Solution | Indian Timezone (IST)</p>
    <p style='font-size: 0.8rem;'>Version 8.1 - Fixed Slider Error & Auto-Refresh</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN THE APP
# ============================================================================

if __name__ == "__main__":
    main()
