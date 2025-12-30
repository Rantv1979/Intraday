"""
AI ALGORITHMIC TRADING BOT v6.0 - COMPLETE FIXED VERSION
PART 1 OF 3: Imports, Config, Database, Broker

INSTALLATION:
pip install streamlit pandas numpy scipy scikit-learn plotly kiteconnect

TO COMBINE ALL PARTS:
1. Copy Part 1 (this file)
2. Append Part 2 at the end
3. Append Part 3 at the end
4. Save as: algo_trader_v6_final.py
5. Run: streamlit run algo_trader_v6_final.py
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

warnings.filterwarnings('ignore')

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
    st.error("❌ Plotly not installed! Run: pip install plotly")

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
    
    # Market Hours
    MARKET_OPEN = dt_time(9, 15)
    MARKET_CLOSE = dt_time(15, 30)

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
    """Get last traded price - INTRADAY: Always try API first"""
    
    # PRIORITY 1: Try Kite API for fresh price
    if self.connected and self.kite:
        try:
            quote = self.kite.ltp([f"NSE:{symbol}"])
            price = quote[f"NSE:{symbol}"]['last_price']
            self.ltp_cache[symbol] = price
            return price
        except:
            pass
    
    # PRIORITY 2: WebSocket cache
    if symbol in self.ltp_cache:
        return self.ltp_cache[symbol]
    
    # PRIORITY 3: Demo fallback
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

class TechnicalAnalysis:
    """Technical indicators calculator"""
    
    @staticmethod
    def calculate_indicators(df):
        """Calculate all technical indicators"""
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
        
        return df

# ============================================================================
# AI ENGINE - FIXED FOR MORE SIGNALS
# ============================================================================

class AIEngine:
    """Machine Learning engine for signal generation"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
    
    def create_features(self, df):
        """Create ML features from price data"""
        df = TechnicalAnalysis.calculate_indicators(df)
        
        feature_cols = [
            'RSI', 'ATR', 'SMA5', 'SMA10', 'SMA20', 'SMA50',
            'EMA5', 'EMA10', 'EMA20', 'MACD', 'MACD_Signal'
        ]
        
        # Add price features
        df['Returns'] = df['Close'].pct_change()
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        feature_cols.extend(['Returns', 'Volume_Ratio'])
        
        return df[feature_cols].fillna(method='bfill').fillna(0)
    
    def train_model(self, df, symbol):
        """Train ML model - FIXED with lower thresholds"""
        if not ML_AVAILABLE:
            return None
        
        try:
            features = self.create_features(df)
            
            # Create labels - FIXED: Lower threshold from 1% to 0.5%
            future_returns = df['Close'].shift(-5) / df['Close'] - 1
            labels = pd.cut(
                future_returns,
                bins=[-np.inf, -0.005, 0.005, np.inf],  # 0.5% instead of 1%
                labels=[-1, 0, 1]
            )
            
            # Remove NaN
            mask = ~(features.isna().any(axis=1) | labels.isna())
            X = features[mask]
            y = labels[mask]
            
            # FIXED: Lower minimum data requirement from 100 to 50
            if len(X) < 50:
                return None
            
            # Scale features
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train model - Faster with 50 trees instead of 100
            model = RandomForestClassifier(
                n_estimators=50,
                max_depth=8,
                random_state=42
            )
            model.fit(X_scaled, y)
            
            self.models[symbol] = model
            self.scalers[symbol] = scaler
            
            return model
            
        except Exception as e:
            return None
    
    def predict(self, df, symbol):
        """Make prediction for a symbol"""
        if symbol not in self.models:
            return 0, 0.0
        
        try:
            features = self.create_features(df)
            latest = features.iloc[-1:].values
            
            scaled = self.scalers[symbol].transform(latest)
            prediction = self.models[symbol].predict(scaled)[0]
            proba = self.models[symbol].predict_proba(scaled)[0]
            confidence = max(proba)
            
            return prediction, confidence
            
        except Exception as e:
            return 0, 0.0

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
# INDICES UPDATER - FOR AUTO-REFRESH
# ============================================================================

class IndicesUpdater:
    """Updates market indices with auto-refresh"""
    
    def __init__(self, broker):
        self.broker = broker
        self.indices_data = {
            'nifty': {'ltp': 24350.50, 'change_pct': 0.45},
            'banknifty': {'ltp': 52180.75, 'change_pct': 0.28},
            'sensex': {'ltp': 80456.25, 'change_pct': 0.38}
        }
        self.last_update = datetime.now()
    
    def update(self):
        """Update indices data from Kite"""
        try:
            if self.broker.connected and self.broker.kite:
                quotes = self.broker.kite.quote([
                    "NSE:NIFTY 50",
                    "NSE:NIFTY BANK",
                    "NSE:SENSEX"
                ])
                
                for key, exchange_symbol in [
                    ('nifty', 'NSE:NIFTY 50'),
                    ('banknifty', 'NSE:NIFTY BANK'),
                    ('sensex', 'NSE:SENSEX')
                ]:
                    data = quotes.get(exchange_symbol, {})
                    ltp = data.get('last_price', self.indices_data[key]['ltp'])
                    change = data.get('change', 0)
                    change_pct = (change / (ltp - change) * 100) if ltp and change else 0
                    
                    self.indices_data[key] = {
                        'ltp': ltp,
                        'change_pct': change_pct
                    }
            
            self.last_update = datetime.now()
            
        except Exception as e:
            pass  # Silent fail, keep previous data
    
    def get_mood(self):
        """Calculate market mood from average change"""
        avg_change = sum(d['change_pct'] for d in self.indices_data.values()) / 3
        
        if avg_change > 0.5:
            return "🟢 BULLISH", "#00C853"
        elif avg_change < -0.5:
            return "🔴 BEARISH", "#FF5252"
        else:
            return "🟡 NEUTRAL", "#FFC107"

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
                # INTRADAY: Auto-exit at 3:20 PM
            if now >= dt_time(15, 20):
                if len(self.risk.positions) > 0:
                    print("⏰ 3:20 PM - Auto-exiting all intraday positions")
                    for symbol in list(self.risk.positions.keys()):
                        try:
                            current_price = self.broker.get_ltp(symbol)
                            self.exit_position(symbol, current_price, 'INTRADAY_AUTO_EXIT')
                        except Exception as e:
                            print(f"Failed to exit {symbol}: {e}")
                time.sleep(600)  # Wait 10 minutes after exit
                continue
                # Scan for signals every 30 seconds
                if scan_counter % 3 == 0:
                    self.scan_signals()
                
                # Auto-execute if enabled
                if hasattr(st.session_state, 'auto_execute') and st.session_state.auto_execute:
                    self.execute_signals()
                
                # Manage open positions
                self.manage_positions()
                
                scan_counter += 1
                time.sleep(10)
                
            except Exception as e:
                time.sleep(30)
    
    def scan_signals(self):
        """Scan all F&O stocks for trading signals"""
        stocks = StockUniverse.get_all_fno_stocks()
        
        for symbol in stocks:
            try:
                can_trade, reason = self.risk.can_trade()
                if not can_trade:
                    break
                
                # Get historical data
                df = self.broker.get_historical(symbol, days=30)
                if len(df) < 50:  # FIXED: Lower from 100
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
                    'timestamp': datetime.now()
                }
                
                self.signals_queue.put(signal)
                
            except Exception as e:
                pass  # Silent fail for individual stocks
    
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


def render_colorful_tabs():
    """Render colorful button-style tabs"""
    
    tab_config = [
        {"name": "🎯 Algo Trading", "color": "#1E88E5", "key": "tab_algo"},
        {"name": "📈 Positions", "color": "#4CAF50", "key": "tab_positions"},
        {"name": "📋 History", "color": "#FF9800", "key": "tab_history"},
        {"name": "📊 Charts", "color": "#9C27B0", "key": "tab_charts"},
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
# MAIN STREAMLIT APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="AI Algo Trading Bot v6.0",
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
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize - WITH PERSISTENT CREDENTIALS
    if 'engine' not in st.session_state:
        st.session_state.engine = TradingEngine(Config(), demo_mode=True)
        st.session_state.indices_updater = IndicesUpdater(st.session_state.engine.broker)
        st.session_state.last_refresh = datetime.now()
        st.session_state.auto_execute = False
        st.session_state.refresh_enabled = True
        st.session_state.refresh_rate = 10
        
        # Auto-connect if credentials exist in session
        if 'kite_api_key' in st.session_state and 'kite_access_token' in st.session_state:
            with st.spinner("🔌 Auto-connecting to Kite..."):
                st.session_state.engine.broker.connect()
    
    engine = st.session_state.engine
    indices = st.session_state.indices_updater
    
    # AUTO REFRESH - INTRADAY VERSION
if st.session_state.get('refresh_enabled', True):
    refresh_rate = st.session_state.get('refresh_rate', 5)  # 5 seconds default
    time_since_refresh = (datetime.now() - st.session_state.last_refresh).total_seconds()
    
    if time_since_refresh >= refresh_rate:
        # Update indices
        indices.update()
        
        # CRITICAL: Clear LTP cache for fresh prices
        engine.broker.ltp_cache.clear()
        
        # Update timestamp
        st.session_state.last_refresh = datetime.now()
        
        # Auto-rerun on positions/analytics tabs
        if st.session_state.get('active_tab', 0) in [1, 4]:
            st.rerun()
    
    # Header
    st.markdown("<h1 class='main-header'>🤖 AI ALGORITHMIC TRADING BOT v6.0 FIXED</h1>", 
                unsafe_allow_html=True)
    st.markdown("### All 159 F&O Stocks | Auto-Refresh | Fixed Signals | Persistent Login")
    
    # Market Mood Gauge - WITH AUTO-REFRESH
    st.markdown("---")
    st.markdown("### 📊 Market Mood Gauge")
    
    col_idx1, col_idx2, col_idx3, col_idx4 = st.columns(4)
    
    nifty = indices.indices_data['nifty']
    banknifty = indices.indices_data['banknifty']
    sensex = indices.indices_data['sensex']
    mood, mood_color = indices.get_mood()
    
    with col_idx1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        change_color = "status-running" if nifty['change_pct'] >= 0 else "status-stopped"
        st.markdown(f'<h3>NIFTY 50</h3>', unsafe_allow_html=True)
        st.markdown(f'<h2>{nifty["ltp"]:,.2f}</h2>', unsafe_allow_html=True)
        st.markdown(f'<p class="{change_color}">{nifty["change_pct"]:+.2f}%</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_idx2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        change_color = "status-running" if banknifty['change_pct'] >= 0 else "status-stopped"
        st.markdown(f'<h3>BANK NIFTY</h3>', unsafe_allow_html=True)
        st.markdown(f'<h2>{banknifty["ltp"]:,.2f}</h2>', unsafe_allow_html=True)
        st.markdown(f'<p class="{change_color}">{banknifty["change_pct"]:+.2f}%</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_idx3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        change_color = "status-running" if sensex['change_pct'] >= 0 else "status-stopped"
        st.markdown(f'<h3>SENSEX</h3>', unsafe_allow_html=True)
        st.markdown(f'<h2>{sensex["ltp"]:,.2f}</h2>', unsafe_allow_html=True)
        st.markdown(f'<p class="{change_color}">{sensex["change_pct"]:+.2f}%</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_idx4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<h3>MARKET MOOD</h3>', unsafe_allow_html=True)
        st.markdown(f'<h2 style="color: {mood_color}">{mood}</h2>', unsafe_allow_html=True)
        st.markdown(f'<p>Updated: {indices.last_update.strftime("%H:%M:%S")}</p>', unsafe_allow_html=True)
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
        Config.TOTAL_CAPITAL = capital
        
        # Risk
        risk = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.1) / 100
        Config.RISK_PER_TRADE = risk
        
        # Confidence - LOWERED DEFAULT
        confidence = st.slider("Min Confidence (%)", 50, 90, 55, 5) / 100
        Config.MIN_CONFIDENCE = confidence
        
        st.markdown("---")
        st.markdown("### 📊 Stock Universe")
        st.info(f"**Total F&O Stocks:** {len(StockUniverse.get_all_fno_stocks())}")
        
       # Auto refresh settings - INTRADAY
st.markdown("---")
st.markdown("### 🔄 Auto Refresh (Intraday)")

auto_refresh = st.checkbox("🔄 Enable Auto Refresh", value=True)
st.session_state.refresh_enabled = auto_refresh

if auto_refresh:
    # Faster refresh for intraday: 3-30 seconds
    refresh_rate = st.slider("Refresh Rate (sec)", 3, 30, 5, 1)
    st.session_state.refresh_rate = refresh_rate
    
    # Show countdown
    time_since = (datetime.now() - st.session_state.last_refresh).total_seconds()
    remaining = int(refresh_rate - time_since)
    if remaining > 0:
        st.info(f"⏳ Next refresh in {remaining}s")
    
    st.caption("💡 Lower = More updates | Higher = Less server load")
    st.caption(f"🕐 Last refresh: {st.session_state.last_refresh.strftime('%H:%M:%S')}")
    
    # Top Metrics
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
    
    # Colorful Tabs
    st.markdown("---")
    active_tab = render_colorful_tabs()
    st.markdown("---")
    
    # Tab Content
    if active_tab == 0:  # Algo Trading
        st.markdown("### 🎯 AI Algorithm Trading Signals")
        
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            scan_btn = st.button("🔍 Scan All 159 Stocks", type="primary", use_container_width=True)
        
        with col2:
            quick_test = st.button("⚡ Quick Test (10 Stocks)", type="secondary", use_container_width=True)
        
        with col3:
            exec_btn = st.button("✅ Execute All", use_container_width=True)
        
        # Quick Test
        if quick_test:
            with st.spinner("⚡ Testing 10 stocks..."):
                while not engine.signals_queue.empty():
                    engine.signals_queue.get()
                
                signals_found = 0
                for symbol in StockUniverse.get_all_fno_stocks()[:10]:
                    try:
                        df = engine.broker.get_historical(symbol, days=30)
                        if len(df) < 50:
                            continue
                        
                        if symbol not in engine.ai.models:
                            engine.ai.train_model(df, symbol)
                        
                        prediction, confidence = engine.ai.predict(df, symbol)
                        
                        if confidence >= Config.MIN_CONFIDENCE and prediction != 0:
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
                            signals_found += 1
                    except:
                        pass
                
                if signals_found > 0:
                    st.success(f"✅ Found {signals_found} signals!")
                else:
                    st.warning("⚠️ No signals. Try lowering confidence threshold.")
                
                st.rerun()
        
        # Full Scan
        if scan_btn:
            with st.spinner("🔍 Scanning all 159 F&O stocks..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                while not engine.signals_queue.empty():
                    engine.signals_queue.get()
                
                stocks = StockUniverse.get_all_fno_stocks()
                total = len(stocks)
                
                scan_stats = {
                    'scanned': 0,
                    'models_trained': 0,
                    'predictions_made': 0,
                    'signals_generated': 0,
                    'skipped': 0,
                    'low_confidence': 0
                }
                
                for idx, symbol in enumerate(stocks):
                    try:
                        progress = (idx + 1) / total
                        progress_bar.progress(progress)
                        status_text.text(f"Scanning {symbol} ({idx+1}/{total}) | Signals: {scan_stats['signals_generated']}")
                        
                        scan_stats['scanned'] += 1
                        
                        df = engine.broker.get_historical(symbol, days=30)
                        if len(df) < 50:
                            scan_stats['skipped'] += 1
                            continue
                        
                        if symbol not in engine.ai.models:
                            engine.ai.train_model(df, symbol)
                            scan_stats['models_trained'] += 1
                        
                        prediction, confidence = engine.ai.predict(df, symbol)
                        scan_stats['predictions_made'] += 1
                        
                        if confidence < Config.MIN_CONFIDENCE:
                            scan_stats['low_confidence'] += 1
                            continue
                        
                        if prediction != 0:
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
                st.success(f"✅ Scan Complete!")
                
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    st.metric("Stocks Scanned", scan_stats['scanned'])
                    st.metric("Models Trained", scan_stats['models_trained'])
                with col_s2:
                    st.metric("Predictions Made", scan_stats['predictions_made'])
                    st.metric("Low Confidence", scan_stats['low_confidence'])
                with col_s3:
                    st.metric("🎯 Signals Generated", scan_stats['signals_generated'])
                    st.metric("Skipped", scan_stats['skipped'])
                
                time.sleep(2)
                st.rerun()
        
        # Execute All
        if exec_btn:
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
                            'entry_time': str(datetime.now()),
                            'entry_price': result['price'],
                            'quantity': signal['quantity'],
                            'stop_loss': signal['stop_loss'],
                            'take_profit': signal['take_profit'],
                            'status': 'OPEN'
                        }
                        engine.risk.positions[signal['symbol']] = position
                        engine.db.save_position(position)
                        engine.risk.daily_trades += 1
                        engine.db.save_trade(position)
                        executed += 1
                except:
                    pass
            
            st.success(f"✅ Executed {executed} signals!")
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
            
            df_signals = pd.DataFrame(signals)
            df_signals['confidence'] = df_signals['confidence'].apply(lambda x: f"{x:.1%}")
            st.dataframe(df_signals, use_container_width=True)
        else:
            st.info("🔭 No pending signals")
        
        # AI Status
        st.markdown("#### 🧠 AI Model Status")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Models Trained", len(engine.ai.models))
        with col2:
            st.metric("Confidence Threshold", f"{Config.MIN_CONFIDENCE:.0%}")
        with col3:
            st.metric("Max Positions", Config.MAX_POSITIONS)
    
    elif active_tab == 1:  # Positions - LIVE UPDATES
    st.markdown("### 📈 Active Positions (Live Updates)")
    
    # Force refresh button
    col_refresh1, col_refresh2 = st.columns([3, 1])
    with col_refresh2:
        if st.button("🔄 Force Refresh Now", key="force_refresh_pos"):
            # Clear price cache
            engine.broker.ltp_cache.clear()
            st.rerun()
    
    positions_df = engine.db.get_open_positions()
    
    if not positions_df.empty:
        # IMPORTANT: Create fresh copy
        live_positions = positions_df.copy()
        
        # Update each position with LIVE prices
        for idx, row in live_positions.iterrows():
            try:
                # Clear cache and get fresh price
                if row['symbol'] in engine.broker.ltp_cache:
                    del engine.broker.ltp_cache[row['symbol']]
                
                current_price = engine.broker.get_ltp(row['symbol'])
                
                # Calculate live P&L
                if row['direction'] == 'LONG':
                    pnl = (current_price - row['entry_price']) * row['quantity']
                else:
                    pnl = (row['entry_price'] - current_price) * row['quantity']
                
                pnl_pct = (pnl / (row['entry_price'] * row['quantity'])) * 100
                
                # Update row
                live_positions.at[idx, 'current_price'] = current_price
                live_positions.at[idx, 'pnl'] = pnl
                live_positions.at[idx, 'pnl_pct'] = pnl_pct
                
            except Exception as e:
                live_positions.at[idx, 'current_price'] = row['entry_price']
                live_positions.at[idx, 'pnl'] = 0
                live_positions.at[idx, 'pnl_pct'] = 0
        
        # Show live timestamp
        st.caption(f"🕐 Last updated: {datetime.now().strftime('%H:%M:%S')}")
        
        # Display table
        st.dataframe(live_positions, use_container_width=True, height=400)
        
        # Summary
        st.markdown("#### 📊 Position Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        total_pnl = live_positions['pnl'].sum()
        winning = len(live_positions[live_positions['pnl'] > 0])
        
        with col1:
            st.metric("Total P&L", f"₹{total_pnl:,.0f}")
        with col2:
            st.metric("Winning", winning)
        with col3:
            st.metric("Losing", len(live_positions) - winning)
        with col4:
            st.metric("Total Positions", len(live_positions))
        
        # Manual exit
        st.markdown("#### 🛑 Manual Exit")
        cols = st.columns(min(4, len(live_positions)))
        
        for idx, (_, row) in enumerate(list(live_positions.iterrows())[:4]):
            with cols[idx]:
                pnl_color = "🟢" if row['pnl'] > 0 else "🔴"
                if st.button(
                    f"{pnl_color} {row['symbol']}\n₹{row['pnl']:,.0f}",
                    key=f"exit_{row['symbol']}_{idx}",
                    use_container_width=True
                ):
                    current_price = engine.broker.get_ltp(row['symbol'])
                    engine.exit_position(row['symbol'], current_price, 'MANUAL')
                    st.success(f"✅ Exited {row['symbol']}")
                    time.sleep(0.5)
                    st.rerun()
        
        # Exit all
        st.markdown("---")
        if st.button("🚨 EXIT ALL POSITIONS", type="secondary", use_container_width=True):
            for _, row in live_positions.iterrows():
                try:
                    price = engine.broker.get_ltp(row['symbol'])
                    engine.exit_position(row['symbol'], price, 'MANUAL_EXIT_ALL')
                except:
                    pass
            st.success("✅ All positions exited!")
            time.sleep(1)
            st.rerun()
    else:
        st.info("🔭 No active positions")
    
    elif active_tab == 2:  # History
        st.markdown("### 📋 Trade History")
        
        trades_df = engine.db.get_trades(100)
        
        if not trades_df.empty:
            trades_df['pnl'] = trades_df['pnl'].apply(lambda x: f"₹{x:,.0f}")
            trades_df['pnl_pct'] = trades_df['pnl_pct'].apply(lambda x: f"{x:.2f}%")
            
            st.dataframe(trades_df, use_container_width=True, height=600)
            
            csv = trades_df.to_csv(index=False)
            st.download_button(
                "📥 Export CSV",
                csv,
                f"trades_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )
        else:
            st.info("🔭 No trade history")
    
    elif active_tab == 3:  # Charts
        st.markdown("### 📊 Live Charts")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            symbol = st.selectbox(
                "Select Stock",
                StockUniverse.get_all_fno_stocks()[:50],
                index=0
            )
        
        with col2:
            if st.button("🔄 Refresh Chart"):
                st.rerun()
        
        df = engine.broker.get_historical(symbol, days=7)
        
        if PLOTLY_AVAILABLE and not df.empty:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.7, 0.3]
            )
            
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
            
            df = TechnicalAnalysis.calculate_indicators(df)
            
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
            
            colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] 
                     else 'green' for i in range(len(df))]
            
            fig.add_trace(
                go.Bar(x=df.index, y=df['Volume'], name='Volume',
                      marker_color=colors),
                row=2, col=1
            )
            
            fig.update_layout(
                title=f"{symbol} - Live Chart",
                template='plotly_dark',
                height=700,
                xaxis_rangeslider_visible=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### 📊 Technical Indicators")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}" if 'RSI' in df.columns else "N/A")
            with col2:
                st.metric("ATR", f"₹{df['ATR'].iloc[-1]:.2f}" if 'ATR' in df.columns else "N/A")
            with col3:
                st.metric("MACD", f"{df['MACD'].iloc[-1]:.2f}" if 'MACD' in df.columns else "N/A")
            with col4:
                current = engine.broker.get_ltp(symbol)
                st.metric("LTP", f"₹{current:.2f}")
        else:
            st.error("Chart unavailable")
    
    elif active_tab == 4:  # Analytics
        st.markdown("### 📉 Performance Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Trade Distribution")
            
            if PLOTLY_AVAILABLE:
                fig = go.Figure(data=[
                    go.Pie(
                        labels=['Winning', 'Losing'],
                        values=[engine.stats['winning_trades'], 
                               engine.stats['losing_trades']],
                        hole=.3
                    )
                ])
                fig.update_layout(template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📈 Key Metrics")
            
            st.metric("Total Trades", engine.stats['total_trades'])
            st.metric("Win Rate", f"{engine.stats['win_rate']:.1f}%")
            st.metric("Total P&L", f"₹{engine.stats['total_pnl']:,.0f}")
            st.metric("Daily Trades", engine.risk.daily_trades)
        
        st.markdown("#### 📊 Recent Performance")
        trades_df = engine.db.get_trades(20)
        
        if not trades_df.empty and PLOTLY_AVAILABLE:
            fig = go.Figure(data=[
                go.Bar(x=trades_df['symbol'], y=trades_df['pnl'],
                      marker_color=['green' if x > 0 else 'red' 
                                   for x in trades_df['pnl']])
            ])
            fig.update_layout(
                title="P&L by Trade",
                template='plotly_dark',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif active_tab == 5:  # Settings - COMPLETE VERSION
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
                st.warning("⚠️ Not connected to Kite")
                
                st.markdown("---")
                st.markdown("### 🎫 Generate Access Token")
                
                with st.form("token_generator"):
                    st.info("Generate fresh access token (valid until 6 AM IST)")
                    
                    api_key = st.text_input(
                        "🔑 API Key",
                        type="password",
                        help="From https://developers.kite.trade/"
                    )
                    
                    api_secret = st.text_input(
                        "🔐 API Secret",
                        type="password",
                        help="Keep this secure!"
                    )
                    
                    generate_btn = st.form_submit_button("🔗 Generate Login URL", type="primary")
                    
                    if generate_btn and api_key and api_secret:
                        try:
                            temp_kite = KiteConnect(api_key=api_key)
                            login_url = temp_kite.login_url()
                            
                            st.session_state.temp_api_key = api_key
                            st.session_state.temp_api_secret = api_secret
                            st.session_state.login_url = login_url
                            
                            st.success("✅ Login URL generated!")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
                
                if 'login_url' in st.session_state:
                    st.markdown("---")
                    st.markdown("**Step 2: Login to Zerodha**")
                    st.markdown(f"[🔗 Click here to login]({st.session_state.login_url})")
                    
                    st.warning("""
                    After logging in, you'll see a URL like:
                    `http://127.0.0.1/?request_token=XXXXXX&action=login`
                    
                    Copy the **request_token** from that URL.
                    """)
                    
                    with st.form("access_token_form"):
                        request_token = st.text_input(
                            "📋 Request Token",
                            help="Paste request_token from URL"
                        )
                        
                        generate_token_btn = st.form_submit_button(
                            "🎫 Generate Access Token",
                            type="primary"
                        )
                        
                        if generate_token_btn and request_token:
                            try:
                                temp_kite = KiteConnect(api_key=st.session_state.temp_api_key)
                                data = temp_kite.generate_session(
                                    request_token,
                                    api_secret=st.session_state.temp_api_secret
                                )
                                
                                access_token = data["access_token"]
                                
                                st.session_state.new_access_token = access_token
                                st.success("✅ Access Token Generated!")
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"❌ Failed: {e}")
                                st.info("""
                                **Common issues:**
                                - Wrong API secret
                                - Token already used
                                - Token expired (5 min validity)
                                
                                Click "Start Over" and try again.
                                """)
                
                if 'new_access_token' in st.session_state:
                    st.markdown("---")
                    st.success("### 🎉 Token Generated!")
                    
                    st.code(st.session_state.new_access_token, language="text")
                    
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        if st.button("💾 Save & Connect", type="primary", use_container_width=True):
                            # Store in session state (persistent until browser closes)
                            st.session_state.kite_api_key = st.session_state.temp_api_key
                            st.session_state.kite_access_token = st.session_state.new_access_token
                            
                            st.success("✅ Credentials saved in session!")
                            st.info("""
                            **Session Storage Active:**
                            - Valid until you close browser
                            - Auto-connects on page refresh
                            
                            **For Permanent Storage:**
                            Add to Streamlit Secrets:
                            ```
                            KITE_API_KEY = "your_key"
                            KITE_ACCESS_TOKEN = "your_token"
                            ```
                            """)
                            
                            with st.spinner("🔌 Connecting..."):
                                success = engine.broker.connect()
                            
                            if success:
                                st.success("✅ Connected to Kite!")
                                
                                # Clear temp data
                                for key in ['temp_api_key', 'temp_api_secret', 'login_url', 'new_access_token']:
                                    if key in st.session_state:
                                        del st.session_state[key]
                                
                                st.balloons()
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("❌ Connection failed")
                    
                    with col_b:
                        if st.button("🔄 Start Over", use_container_width=True):
                            for key in ['temp_api_key', 'temp_api_secret', 'login_url', 'new_access_token']:
                                if key in st.session_state:
                                    del st.session_state[key]
                            st.rerun()
        
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
                
                new_risk = st.slider(
                    "Risk per Trade (%)",
                    0.5, 5.0, Config.RISK_PER_TRADE * 100, 0.1
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
                new_confidence = st.slider(
                    "Min Confidence (%)",
                    50, 95, 55, 5
                )
                
                st.markdown("**Risk Management**")
                new_atr_mult = st.slider(
                    "ATR Multiplier",
                    1.0, 5.0, Config.ATR_MULTIPLIER, 0.5
                )
                
                new_rr_ratio = st.slider(
                    "Risk:Reward Ratio",
                    1.5, 5.0, Config.TAKE_PROFIT_RATIO, 0.5
                )
                
                trailing_stop = st.checkbox(
                    "Enable Trailing Stop",
                    value=Config.TRAILING_STOP
                )
                
                if st.form_submit_button("💾 Save Configuration", type="primary"):
                    Config.TOTAL_CAPITAL = new_capital
                    Config.RISK_PER_TRADE = new_risk / 100
                    Config.MAX_POSITIONS = new_max_positions
                    Config.MAX_DAILY_TRADES = new_max_trades
                    Config.MIN_CONFIDENCE = new_confidence / 100
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
            
            st.info(f"""
            **Status:** {'🟢 Running' if engine.running else '🔴 Stopped'}
            **Mode:** {'💰 Live' if not engine.broker.demo_mode else '📈 Paper'}
            **Kite:** {'🟢 Connected' if engine.broker.connected else '🔴 Disconnected'}
            **WebSocket:** {'🟢 Active' if engine.broker.websocket_running else '🔴 Inactive'}
            **Auto-Execute:** {'🟢 ON' if st.session_state.get('auto_execute', False) else '🔴 OFF'}
            **Models Trained:** {len(engine.ai.models)}
            **Stock Universe:** {len(StockUniverse.get_all_fno_stocks())} stocks
            """)
            
            st.markdown("#### 💾 Database")
            trade_count = len(engine.db.get_trades(1000))
            st.metric("Total Trades", trade_count)
            
            if st.button("🗑️ Clear Database", type="secondary"):
                st.warning("⚠️ This will delete ALL data!")
                confirm = st.checkbox("I understand and want to proceed")
                if confirm and st.button("Yes, Delete Everything"):
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
    <div style='text-align: center; color: #666;'>
    <p>🚨 <b>DISCLAIMER:</b> For educational purposes only. Trading involves risk of loss.</p>
    <p>© 2025 AI Algo Trading Bot v6.0 FIXED | All 159 F&O Stocks | Complete Solution</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN THE APP
# ============================================================================

if __name__ == "__main__":
    main()
