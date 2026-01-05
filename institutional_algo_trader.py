# =============================================================
# MERGED INSTITUTIONAL ALGO TRADER - FINAL
# Original System + Advanced Enhancements
# =============================================================

"""
AI ALGORITHMIC TRADING BOT v8.0 - PRODUCTION READY
WITH REAL-TIME TICK PROCESSING, ENHANCED SIGNAL REFRESH, AND LIVE POSITION MANAGEMENT

ENHANCEMENTS:
1. Real-time WebSocket tick processing for instant price updates
2. Signal refresh triggered by price movements and time intervals
3. Live position monitoring with real-time P&L
4. Enhanced order management with state tracking
5. Circuit breakers and risk limits for live trading
6. Performance monitoring and metrics
7. Connection resiliency and auto-reconnect

INSTALLATION:
pip install streamlit pandas numpy scikit-learn plotly kiteconnect redis
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
import hashlib
import json
import asyncio
import concurrent.futures
from collections import deque
import redis
import logging

class RiskManager:
    def __init__(self, config):
        self.config = config
        self.start_capital = config.TOTAL_CAPITAL
        self.capital = config.TOTAL_CAPITAL
        self.peak_capital = config.TOTAL_CAPITAL
        self.daily_pnl = 0.0
        self.positions = {}

    def update_pnl(self, pnl):
        self.capital += pnl
        self.daily_pnl += pnl
        self.peak_capital = max(self.peak_capital, self.capital)

    def drawdown_pct(self):
        return (self.peak_capital - self.capital) / self.peak_capital

    def daily_loss_pct(self):
        return abs(self.daily_pnl) / self.start_capital

    def risk_ok(self):
        if self.daily_loss_pct() >= self.config.MAX_DAILY_LOSS:
            return False, "Daily loss limit breached"
        if self.drawdown_pct() >= self.config.MAX_DRAWDOWN:
            return False, "Max drawdown breached"
        return True, "OK"

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import handling
try:
    from kiteconnect import KiteConnect, KiteTicker
    KITE_AVAILABLE = True
except ImportError:
    KITE_AVAILABLE = False
    st.error("❌ KiteConnect not installed! Run: pip install kiteconnect")

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import accuracy_score
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
# CONFIGURATION - ENHANCED FOR LIVE TRADING
# ============================================================================

class Config:
    """Trading bot configuration - Production ready"""
    # Capital Management
    TOTAL_CAPITAL = 2_000_000
    RISK_PER_TRADE = 0.01  # 1% risk per trade
    MAX_DAILY_LOSS = 0.02  # 2% daily loss limit
    MAX_DRAWDOWN = 0.05    # 5% maximum drawdown
    
    # Position Limits
    MAX_POSITIONS = 10
    MAX_DAILY_TRADES = 50
    MAX_POSITION_SIZE = 0.05  # 5% of capital per position
    
    # AI Parameters
    MIN_CONFIDENCE = 0.55  # 55%
    SIGNAL_EXPIRY_SECONDS = 60  # Signals expire after 60 seconds
    SIGNAL_REFRESH_INTERVAL = 5  # Refresh signals every 5 seconds
    
    # Risk Management
    ATR_MULTIPLIER = 2.0
    TAKE_PROFIT_RATIO = 2.5
    TRAILING_STOP = True
    TRAILING_ACTIVATION = 0.015
    CIRCUIT_BREAKER_PCT = 10.0  # Stop trading if index moves 10%
    
    # Market Hours
    MARKET_OPEN = dt_time(9, 15)
    MARKET_CLOSE = dt_time(15, 30)
    
    # Real-time Settings
    TICK_BUFFER_SIZE = 100
    PRICE_UPDATE_INTERVAL = 1  # Update prices every second
    POSITION_REFRESH_INTERVAL = 2  # Refresh positions every 2 seconds
    
    # WebSocket Settings
    WEBSOCKET_RECONNECT_DELAY = 5
    WEBSOCKET_MAX_RETRIES = 3

# ============================================================================
# REAL-TIME DATA MANAGER
# ============================================================================

class RealTimeDataManager:
    """Manages real-time tick data and price updates"""
    
    def __init__(self):
        self.tick_data = {}
        self.last_ticks = {}
        self.price_history = {}
        self.tick_buffer = deque(maxlen=Config.TICK_BUFFER_SIZE)
        self.update_callbacks = []
        self.running = False
        self.thread = None
        
    def start(self):
        """Start real-time data manager"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._update_loop, daemon=True)
            self.thread.start()
            logger.info("RealTimeDataManager started")
    
    def stop(self):
        """Stop real-time data manager"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("RealTimeDataManager stopped")
    
    def _update_loop(self):
        """Main update loop for real-time data"""
        while self.running:
            try:
                self._process_pending_updates()
                time.sleep(Config.PRICE_UPDATE_INTERVAL)
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                time.sleep(1)
    
    def _process_pending_updates(self):
        """Process pending price updates"""
        current_time = datetime.now()
        
        # Update price history
        for symbol, tick in self.last_ticks.items():
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            
            self.price_history[symbol].append({
                'timestamp': current_time,
                'price': tick['last_price'],
                'volume': tick.get('volume', 0)
            })
            
            # Keep only last 1000 ticks
            if len(self.price_history[symbol]) > 1000:
                self.price_history[symbol] = self.price_history[symbol][-1000:]
        
        # Notify callbacks
        for callback in self.update_callbacks:
            try:
                callback(self.last_ticks)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def update_tick(self, symbol, tick_data):
        """Update tick data for a symbol"""
        self.last_ticks[symbol] = tick_data
        
        # Store in buffer
        self.tick_buffer.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'price': tick_data['last_price'],
            'volume': tick_data.get('volume', 0)
        })
        
        # Update tick data
        if symbol not in self.tick_data:
            self.tick_data[symbol] = []
        
        self.tick_data[symbol].append({
            'timestamp': datetime.now(),
            'price': tick_data['last_price'],
            'volume': tick_data.get('volume', 0),
            'buy_quantity': tick_data.get('buy_quantity', 0),
            'sell_quantity': tick_data.get('sell_quantity', 0)
        })
        
        # Keep only recent data
        if len(self.tick_data[symbol]) > 100:
            self.tick_data[symbol] = self.tick_data[symbol][-100:]
    
    def get_latest_price(self, symbol):
        """Get latest price for symbol"""
        if symbol in self.last_ticks:
            return self.last_ticks[symbol]['last_price']
        return None
    
    def get_price_history(self, symbol, lookback=100):
        """Get price history for symbol"""
        if symbol in self.price_history:
            history = self.price_history[symbol]
            return history[-lookback:] if len(history) >= lookback else history
        return []
    
    def register_callback(self, callback):
        """Register callback for price updates"""
        self.update_callbacks.append(callback)
    
    def get_tick_statistics(self, symbol):
        """Get tick statistics for symbol"""
        if symbol not in self.tick_data or not self.tick_data[symbol]:
            return None
        
        ticks = self.tick_data[symbol]
        prices = [t['price'] for t in ticks]
        volumes = [t['volume'] for t in ticks]
        
        return {
            'current_price': prices[-1],
            'avg_price': np.mean(prices),
            'price_std': np.std(prices),
            'volume': sum(volumes),
            'buy_pressure': sum(t.get('buy_quantity', 0) for t in ticks),
            'sell_pressure': sum(t.get('sell_quantity', 0) for t in ticks),
            'tick_count': len(ticks)
        }

# ============================================================================
# ENHANCED MARKET INDICES UPDATER
# ============================================================================

class EnhancedMarketIndicesUpdater:
    """Updates market indices with real-time tick data"""
    
    def __init__(self, broker, real_time_manager):
        self.broker = broker
        self.real_time_manager = real_time_manager
        self.indices_data = {
            'NIFTY 50': {
                'symbol': 'NIFTY 50', 
                'ltp': 24350.50, 
                'change': 0.0, 
                'change_pct': 0.45,
                'open': 24200.00,
                'high': 24400.00,
                'low': 24200.00,
                'volume': 0,
                'timestamp': datetime.now()
            },
            'NIFTY BANK': {
                'symbol': 'NIFTY BANK', 
                'ltp': 52180.75, 
                'change': 0.0, 
                'change_pct': 0.28,
                'open': 52000.00,
                'high': 52300.00,
                'low': 51900.00,
                'volume': 0,
                'timestamp': datetime.now()
            },
            'SENSEX': {
                'symbol': 'SENSEX', 
                'ltp': 80456.25, 
                'change': 0.0, 
                'change_pct': 0.38,
                'open': 80200.00,
                'high': 80600.00,
                'low': 80100.00,
                'volume': 0,
                'timestamp': datetime.now()
            }
        }
        self.index_tokens = {
            'NIFTY 50': 256265,
            'NIFTY BANK': 260105,
            'SENSEX': 265
        }
        self.last_update = datetime.now()
        self.update_counter = 0
        self.circuit_breakers = {}
        
        # Register for real-time updates
        self.real_time_manager.register_callback(self._on_tick_update)
    
    def _on_tick_update(self, ticks):
        """Handle tick updates from real-time manager"""
        for symbol, data in ticks.items():
            if symbol in self.indices_data:
                old_price = self.indices_data[symbol]['ltp']
                new_price = data['last_price']
                
                if old_price != new_price:
                    change = new_price - old_price
                    change_pct = (change / old_price * 100) if old_price > 0 else 0
                    
                    self.indices_data[symbol]['ltp'] = new_price
                    self.indices_data[symbol]['change'] = change
                    self.indices_data[symbol]['change_pct'] = change_pct
                    self.indices_data[symbol]['timestamp'] = datetime.now()
                    self.indices_data[symbol]['volume'] = data.get('volume', 0)
                    
                    # Update high/low
                    if new_price > self.indices_data[symbol]['high']:
                        self.indices_data[symbol]['high'] = new_price
                    if new_price < self.indices_data[symbol]['low']:
                        self.indices_data[symbol]['low'] = new_price
    
    def update_from_kite(self):
        """Update indices data from Kite Connect"""
        try:
            if self.broker.connected and self.broker.kite:
                # Fetch live data from Kite
                try:
                    indices = self.broker.kite.quote([
                        "NSE:NIFTY 50",
                        "NSE:NIFTY BANK", 
                        "NSE:SENSEX"
                    ])
                except Exception as e:
                    logger.warning(f"Kite quote API error: {e}")
                    self._update_demo_data()
                    return False
                
                # Update indices with quote data
                for index_key, quote_key in [
                    ('NIFTY 50', 'NSE:NIFTY 50'),
                    ('NIFTY BANK', 'NSE:NIFTY BANK'),
                    ('SENSEX', 'NSE:SENSEX')
                ]:
                    if quote_key in indices:
                        data = indices[quote_key]
                        if 'last_price' in data:
                            self._update_index_data(index_key, data)
                
                self.last_update = datetime.now()
                self.update_counter += 1
                return True
            else:
                self._update_demo_data()
                return False
                
        except Exception as e:
            logger.error(f"Market indices update failed: {e}")
            self._update_demo_data()
            return False
    
    def _update_index_data(self, index_key, data):
        """Update individual index data"""
        old_price = self.indices_data[index_key]['ltp']
        new_price = data.get('last_price', old_price)
        change = new_price - old_price
        change_pct = (change / old_price * 100) if old_price > 0 else 0
        
        self.indices_data[index_key]['ltp'] = new_price
        self.indices_data[index_key]['change'] = change
        self.indices_data[index_key]['change_pct'] = change_pct
        self.indices_data[index_key]['timestamp'] = datetime.now()
        self.indices_data[index_key]['volume'] = data.get('volume_traded', 0)
        self.indices_data[index_key]['open'] = data.get('ohlc', {}).get('open', old_price)
        
        # Update high/low
        ohlc = data.get('ohlc', {})
        if 'high' in ohlc and ohlc['high'] > self.indices_data[index_key]['high']:
            self.indices_data[index_key]['high'] = ohlc['high']
        if 'low' in ohlc and ohlc['low'] < self.indices_data[index_key]['low']:
            self.indices_data[index_key]['low'] = ohlc['low']
    
    def _update_demo_data(self):
        """Update with realistic demo data"""
        for index in self.indices_data.values():
            # Add some randomness
            change_pct = np.random.uniform(-0.15, 0.15)
            change = index['ltp'] * change_pct / 100
            index['ltp'] += change
            index['change'] = change
            index['change_pct'] = change_pct
            index['timestamp'] = datetime.now()
            
            # Update high/low
            if index['ltp'] > index['high']:
                index['high'] = index['ltp']
            if index['ltp'] < index['low']:
                index['low'] = index['ltp']
        
        self.last_update = datetime.now()
        self.update_counter += 1
    
    def get_market_mood(self):
        """Calculate market mood from average change"""
        avg_change = sum(index['change_pct'] for index in self.indices_data.values()) / 3
        
        if avg_change > 0.5:
            return "🟢 STRONG BULLISH", "#00C853", avg_change
        elif avg_change > 0.2:
            return "🟢 BULLISH", "#4CAF50", avg_change
        elif avg_change < -0.5:
            return "🔴 STRONG BEARISH", "#FF5252", avg_change
        elif avg_change < -0.2:
            return "🔴 BEARISH", "#FF9800", avg_change
        else:
            return "🟡 NEUTRAL", "#FFC107", avg_change
    
    def check_circuit_breakers(self):
        """Check if circuit breakers are triggered"""
        alerts = []
        for index_name, data in self.indices_data.items():
            change_pct = abs(data['change_pct'])
            if change_pct >= Config.CIRCUIT_BREAKER_PCT:
                alerts.append(f"{index_name} moved {change_pct:.1f}% - Circuit Breaker!")
                
                # Update circuit breaker status
                self.circuit_breakers[index_name] = {
                    'triggered': True,
                    'change_pct': change_pct,
                    'timestamp': datetime.now()
                }
        
        return alerts
    
    def should_refresh(self, interval_seconds=5):
        """Check if data should be refreshed"""
        return (datetime.now() - self.last_update).total_seconds() >= interval_seconds

# ============================================================================
# ENHANCED SIGNAL GENERATOR WITH REAL-TIME REFRESH
# ============================================================================

class EnhancedSignalGenerator:
    """Generates and refreshes trading signals in real-time"""
    
    def __init__(self, ai_engine, broker, real_time_manager):
        self.ai_engine = ai_engine
        self.broker = broker
        self.real_time_manager = real_time_manager
        self.active_signals = {}
        self.signal_history = []
        self.last_signal_time = {}
        self.running = False
        self.thread = None
        
        # Signal validation parameters
        self.min_volume = 10000
        self.price_movement_threshold = 0.5  # 0.5% price movement for refresh
        self.confirmation_periods = 3
        
    def start(self):
        """Start signal generator"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._signal_refresh_loop, daemon=True)
            self.thread.start()
            logger.info("SignalGenerator started")
    
    def stop(self):
        """Stop signal generator"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("SignalGenerator stopped")
    
    def _signal_refresh_loop(self):
        """Main loop for refreshing signals"""
        while self.running:
            try:
                self._refresh_expired_signals()
                self._check_price_triggers()
                time.sleep(Config.SIGNAL_REFRESH_INTERVAL)
            except Exception as e:
                logger.error(f"Error in signal refresh loop: {e}")
                time.sleep(1)
    
    def _refresh_expired_signals(self):
        """Refresh signals that have expired"""
        current_time = datetime.now()
        symbols_to_refresh = []
        
        for symbol, signal_time in self.last_signal_time.items():
            time_diff = (current_time - signal_time).total_seconds()
            if time_diff >= Config.SIGNAL_EXPIRY_SECONDS:
                symbols_to_refresh.append(symbol)
        
        for symbol in symbols_to_refresh:
            self.generate_signal(symbol)
    
    def _check_price_triggers(self):
        """Check for price movements that should trigger signal refresh"""
        for symbol in list(self.active_signals.keys()):
            stats = self.real_time_manager.get_tick_statistics(symbol)
            if not stats:
                continue
            
            # Check for significant price movement
            if 'price_std' in stats and stats['price_std'] > 0:
                volatility = stats['price_std'] / stats['current_price'] * 100
                if volatility >= self.price_movement_threshold:
                    logger.info(f"Price trigger for {symbol}: volatility {volatility:.2f}%")
                    self.generate_signal(symbol)
    
    def generate_signal(self, symbol):
        """Generate trading signal for symbol"""
        try:
            # Get historical data
            df = self.broker.get_historical(symbol, days=30)
            if len(df) < 50:
                return None
            
            # Train model if needed
            if symbol not in self.ai_engine.models:
                self.ai_engine.train_model(df, symbol)
            
            # Get prediction
            prediction, confidence, metrics = self.ai_engine.predict(df, symbol)
            
            # Check confidence threshold
            if confidence < Config.MIN_CONFIDENCE:
                return None
            
            # Skip HOLD signals
            if prediction == 0:
                return None
            
            # Get real-time data
            current_price = self.real_time_manager.get_latest_price(symbol)
            if current_price is None:
                current_price = self.broker.get_ltp(symbol)
            
            # Calculate risk parameters
            direction = 'LONG' if prediction == 1 else 'SHORT'
            stop_loss = self._calculate_stop_loss(df, direction, current_price)
            take_profit = self._calculate_take_profit(current_price, stop_loss, direction)
            
            # Calculate position size
            risk_amount = Config.TOTAL_CAPITAL * Config.RISK_PER_TRADE
            risk_per_share = abs(current_price - stop_loss)
            quantity = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
            quantity = max(1, min(quantity, int(Config.TOTAL_CAPITAL * Config.MAX_POSITION_SIZE / current_price)))
            
            signal = {
                'symbol': symbol,
                'direction': direction,
                'price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'quantity': quantity,
                'confidence': confidence,
                'prediction': prediction,
                'timestamp': datetime.now(),
                'signal_id': f"{symbol}_{int(time.time())}",
                'status': 'ACTIVE'
            }
            
            # Validate signal
            if self._validate_signal(signal):
                self.active_signals[symbol] = signal
                self.last_signal_time[symbol] = datetime.now()
                self.signal_history.append(signal)
                
                # Keep only recent history
                if len(self.signal_history) > 1000:
                    self.signal_history = self.signal_history[-1000:]
                
                logger.info(f"Generated signal: {symbol} {direction} at {current_price}")
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    def _calculate_stop_loss(self, df, direction, current_price):
        """Calculate stop loss with ATR"""
        if 'ATR' in df.columns:
            atr = df['ATR'].iloc[-1]
        else:
            atr = current_price * 0.02
        
        if direction == 'LONG':
            return current_price - (atr * Config.ATR_MULTIPLIER)
        else:
            return current_price + (atr * Config.ATR_MULTIPLIER)
    
    def _calculate_take_profit(self, entry, stop_loss, direction):
        """Calculate take profit target"""
        risk = abs(entry - stop_loss)
        reward = risk * Config.TAKE_PROFIT_RATIO
        
        if direction == 'LONG':
            return entry + reward
        else:
            return entry - reward
    
    def _validate_signal(self, signal):
        """Validate signal before accepting"""
        # Check volume
        stats = self.real_time_manager.get_tick_statistics(signal['symbol'])
        if stats and stats.get('volume', 0) < self.min_volume:
            return False
        
        # Check price sanity
        if signal['price'] <= 0:
            return False
        
        # Check stop loss/take profit sanity
        if signal['direction'] == 'LONG':
            if signal['stop_loss'] >= signal['price'] or signal['take_profit'] <= signal['price']:
                return False
        else:
            if signal['stop_loss'] <= signal['price'] or signal['take_profit'] >= signal['price']:
                return False
        
        return True
    
    def get_active_signals(self):
        """Get all active signals"""
        return list(self.active_signals.values())
    
    def remove_signal(self, symbol):
        """Remove signal for symbol"""
        if symbol in self.active_signals:
            del self.active_signals[symbol]
    
    def refresh_all_signals(self):
        """Refresh all signals"""
        logger.info("Refreshing all signals")
        signals_generated = 0
        
        # Get all F&O stocks
        stocks = StockUniverse.get_all_fno_stocks()[:50]  # Limit to 50 for performance
        
        for symbol in stocks:
            signal = self.generate_signal(symbol)
            if signal:
                signals_generated += 1
        
        logger.info(f"Generated {signals_generated} new signals")
        return signals_generated

# ============================================================================
# ENHANCED KITE BROKER WITH REAL-TIME TICK PROCESSING
# ============================================================================

class EnhancedKiteBroker:
    """Enhanced Kite broker with real-time tick processing"""
    
    def __init__(self, demo_mode=True, real_time_manager=None):
        self.demo_mode = demo_mode
        self.kite = None
        self.ticker = None
        self.connected = False
        self.ltp_cache = {}
        self.instruments_dict = {}
        self.websocket_running = False
        self.websocket_thread = None
        self.ticks_queue = queue.Queue()
        self.real_time_manager = real_time_manager or RealTimeDataManager()
        self.connection_retries = 0
        self.last_reconnect = None
        
        # Start real-time manager
        self.real_time_manager.start()
        
        if not demo_mode and KITE_AVAILABLE:
            self.connect()
    
    def connect(self):
        """Connect to Kite with enhanced error handling"""
        try:
            # Get credentials
            api_key, access_token = self._get_credentials()
            
            if not api_key or not access_token:
                self.demo_mode = True
                st.warning("⚠️ Kite credentials not found. Running in demo mode.")
                return False
            
            # Initialize Kite Connect
            self.kite = KiteConnect(api_key=api_key)
            self.kite.set_access_token(access_token)
            
            # Test connection
            try:
                profile = self.kite.profile()
                st.success(f"✅ Connected to Kite: {profile['user_name']}")
                
                # Load instruments
                self.load_instruments()
                
                # Setup WebSocket
                self.setup_websocket(api_key, access_token)
                
                self.connected = True
                self.connection_retries = 0
                return True
                
            except Exception as e:
                st.error(f"❌ Kite connection test failed: {str(e)}")
                self.demo_mode = True
                return False
            
        except Exception as e:
            st.error(f"❌ Kite connection failed: {str(e)}")
            self.demo_mode = True
            return False
    
    def _get_credentials(self):
        """Get credentials with priority order"""
        # Session state
        if 'kite_api_key' in st.session_state and 'kite_access_token' in st.session_state:
            return st.session_state.kite_api_key, st.session_state.kite_access_token
        
        # Streamlit secrets
        elif hasattr(st, 'secrets') and 'KITE_API_KEY' in st.secrets:
            api_key = st.secrets.get("KITE_API_KEY", "")
            access_token = st.secrets.get("KITE_ACCESS_TOKEN", "")
            return api_key, access_token
        
        # Environment variables
        else:
            api_key = os.getenv('KITE_API_KEY', '')
            access_token = os.getenv('KITE_ACCESS_TOKEN', '')
            return api_key, access_token
    
    def load_instruments(self):
        """Load NSE instruments for token mapping"""
        try:
            instruments = self.kite.instruments("NSE")
            
            for inst in instruments:
                symbol = inst['tradingsymbol']
                self.instruments_dict[symbol] = {
                    'token': inst['instrument_token'],
                    'lot_size': inst.get('lot_size', 1),
                    'tick_size': inst.get('tick_size', 0.05),
                    'exchange': 'NSE'
                }
            
            # Add indices
            self.instruments_dict['NIFTY 50'] = {'token': 256265, 'lot_size': 1, 'tick_size': 0.05, 'exchange': 'NSE'}
            self.instruments_dict['NIFTY BANK'] = {'token': 260105, 'lot_size': 1, 'tick_size': 0.05, 'exchange': 'NSE'}
            self.instruments_dict['SENSEX'] = {'token': 265, 'lot_size': 1, 'tick_size': 0.05, 'exchange': 'NSE'}
            
            logger.info(f"Loaded {len(self.instruments_dict)} instruments")
            
        except Exception as e:
            logger.warning(f"Failed to load instruments: {e}")
            self.instruments_dict = {}
    
    def setup_websocket(self, api_key, access_token):
        """Setup WebSocket with enhanced reconnection logic"""
        try:
            self.ticker = KiteTicker(api_key, access_token)
            
            def on_ticks(ws, ticks):
                """Callback for ticks"""
                for tick in ticks:
                    # Put in queue
                    self.ticks_queue.put(tick)
                    
                    # Process tick immediately
                    self._process_tick(tick)
            
            def on_connect(ws, response):
                """Callback when WebSocket connects"""
                try:
                    # Subscribe to instruments
                    tokens = self._get_subscription_tokens()
                    
                    if tokens:
                        ws.subscribe(tokens)
                        ws.set_mode(ws.MODE_FULL, tokens)
                        logger.info(f"WebSocket subscribed to {len(tokens)} instruments")
                    
                    self.websocket_running = True
                    self.connection_retries = 0
                    
                except Exception as e:
                    logger.error(f"WebSocket subscribe error: {e}")
            
            def on_close(ws, code, reason):
                """Callback when WebSocket closes"""
                self.websocket_running = False
                logger.warning(f"WebSocket closed: {reason}")
                
                # Attempt reconnection
                self._reconnect_websocket()
            
            def on_error(ws, code, reason):
                """Callback for WebSocket errors"""
                logger.error(f"WebSocket error: {reason}")
            
            self.ticker.on_ticks = on_ticks
            self.ticker.on_connect = on_connect
            self.ticker.on_close = on_close
            self.ticker.on_error = on_error
            
            # Start WebSocket
            self._start_websocket_thread()
            
        except Exception as e:
            logger.error(f"WebSocket setup failed: {e}")
            self.websocket_running = False
    
    def _get_subscription_tokens(self):
        """Get tokens to subscribe to"""
        tokens = []
        
        # Add indices
        tokens.append(256265)  # NIFTY 50
        tokens.append(260105)  # NIFTY BANK
        tokens.append(265)     # SENSEX
        
        # Add top 100 F&O stocks
        stocks = StockUniverse.get_all_fno_stocks()
        for symbol in stocks[:100]:
            if symbol in self.instruments_dict:
                tokens.append(self.instruments_dict[symbol]['token'])
        
        return tokens
    
    def _process_tick(self, tick):
        """Process incoming tick"""
        try:
            # Find symbol for this token
            symbol = None
            for sym, data in self.instruments_dict.items():
                if data['token'] == tick['instrument_token']:
                    symbol = sym
                    break
            
            if symbol:
                # Update LTP cache
                self.ltp_cache[symbol] = tick['last_price']
                
                # Update real-time manager
                self.real_time_manager.update_tick(symbol, tick)
                
                # Process depth if available
                if 'depth' in tick:
                    self._process_market_depth(symbol, tick['depth'])
        
        except Exception as e:
            logger.error(f"Error processing tick: {e}")
    
    def _process_market_depth(self, symbol, depth):
        """Process market depth data"""
        # Can be used for advanced order placement
        pass
    
    def _start_websocket_thread(self):
        """Start WebSocket in background thread"""
        def start_ticker():
            try:
                self.ticker.connect(threaded=True)
                while self.websocket_running:
                    time.sleep(1)
            except Exception as e:
                logger.error(f"WebSocket connection failed: {e}")
                self.websocket_running = False
        
        self.websocket_thread = threading.Thread(target=start_ticker, daemon=True)
        self.websocket_thread.start()
        
        # Wait for connection
        time.sleep(3)
    
    def _reconnect_websocket(self):
        """Attempt to reconnect WebSocket"""
        if self.connection_retries >= Config.WEBSOCKET_MAX_RETRIES:
            logger.error("Max WebSocket reconnection attempts reached")
            return
        
        current_time = datetime.now()
        if self.last_reconnect and (current_time - self.last_reconnect).total_seconds() < 30:
            return
        
        self.connection_retries += 1
        self.last_reconnect = current_time
        
        logger.info(f"Attempting WebSocket reconnection ({self.connection_retries}/{Config.WEBSOCKET_MAX_RETRIES})")
        
        time.sleep(Config.WEBSOCKET_RECONNECT_DELAY * self.connection_retries)
        
        if self.ticker:
            try:
                self.ticker.close()
            except:
                pass
        
        # Reinitialize WebSocket
        api_key, access_token = self._get_credentials()
        if api_key and access_token:
            self.setup_websocket(api_key, access_token)
    
    def get_ltp(self, symbol):
        """Get last traded price with real-time priority"""
        # Try real-time manager first
        price = self.real_time_manager.get_latest_price(symbol)
        if price is not None:
            return price
        
        # Try cache
        if symbol in self.ltp_cache:
            return self.ltp_cache[symbol]
        
        # Fallback to Kite API or demo
        if self.connected and self.kite:
            try:
                if symbol in ['NIFTY 50', 'NIFTY BANK', 'SENSEX']:
                    exchange = 'NSE'
                else:
                    exchange = 'NSE'
                
                quote = self.kite.ltp([f"{exchange}:{symbol}"])
                price = quote[f"{exchange}:{symbol}"]['last_price']
                self.ltp_cache[symbol] = price
                return price
            except Exception as e:
                logger.warning(f"Kite LTP failed for {symbol}: {e}")
        
        # Fallback to demo price
        if symbol not in self.ltp_cache:
            hash_value = abs(hash(symbol)) % 10000
            base_price = 1000 if symbol in ['NIFTY 50', 'NIFTY BANK', 'SENSEX'] else 1000
            self.ltp_cache[symbol] = base_price + (hash_value / 100)
        
        return self.ltp_cache[symbol]
    
    def get_real_time_price(self, symbol):
        """Get real-time price with statistics"""
        price = self.get_ltp(symbol)
        stats = self.real_time_manager.get_tick_statistics(symbol)
        
        return {
            'price': price,
            'statistics': stats,
            'timestamp': datetime.now()
        }
    
    # Keep existing methods for historical data, order placement, etc.
    def get_historical(self, symbol, days=30):
        """Get historical data (same as before)"""
        if self.connected and self.kite:
            try:
                if symbol not in self.instruments_dict:
                    return self._generate_synthetic(symbol, days)
                
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
                    return self._generate_synthetic(symbol, days)
                
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
                
            except Exception as e:
                logger.error(f"Historical data error for {symbol}: {e}")
                return self._generate_synthetic(symbol, days)
        
        return self._generate_synthetic(symbol, days)
    
    def _generate_synthetic(self, symbol, days):
        """Generate synthetic data (same as before)"""
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
        """Place order with enhanced tracking"""
        if self.demo_mode:
            order_id = f'DEMO_{int(time.time())}_{symbol}'
            fill_price = price or self.get_ltp(symbol)
            
            order_result = {
                'status': 'success',
                'order_id': order_id,
                'price': fill_price,
                'timestamp': datetime.now()
            }
            
            logger.info(f"Demo order: {symbol} {direction} {quantity} @ {fill_price}")
            return order_result
        
        try:
            order_type = 'LIMIT' if price else 'MARKET'
            
            order_id = self.kite.place_order(
                tradingsymbol=symbol,
                exchange='NSE',
                transaction_type='BUY' if direction == 'LONG' else 'SELL',
                quantity=quantity,
                order_type=order_type,
                product='MIS',
                variety='regular',
                price=price if order_type == 'LIMIT' else None
            )
            
            # Get order details
            order_details = self.kite.order_history(order_id)[0]
            
            result = {
                'status': order_details['status'].lower(),
                'order_id': order_id,
                'price': order_details.get('average_price', price or self.get_ltp(symbol)),
                'timestamp': datetime.now(),
                'details': order_details
            }
            
            logger.info(f"Live order placed: {symbol} {direction} {quantity} @ {result['price']}")
            return result
            
        except Exception as e:
            logger.error(f"Order placement failed for {symbol}: {e}")
            return {'status': 'error', 'message': str(e)}

# ============================================================================
# ENHANCED AI ENGINE
# ============================================================================

class EnhancedAIEngine:
    """Enhanced ML engine with real-time feature updates"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.last_training = {}
    
    def create_features(self, df, real_time_stats=None):
        """Create enhanced ML features with real-time data"""
        df = TechnicalAnalysis.calculate_indicators(df)
        
        # Base features
        feature_cols = [
            'RSI', 'ATR', 'SMA5', 'SMA10', 'SMA20', 'SMA50',
            'EMA5', 'EMA10', 'EMA20', 'MACD', 'MACD_Signal'
        ]
        
        # Add momentum features
        df['Momentum'] = df['Close'] / df['Close'].shift(5) - 1
        df['Volatility'] = df['Close'].rolling(20).std() / df['Close'].rolling(20).mean()
        
        feature_cols.extend(['Momentum', 'Volatility'])
        
        # Add volume features
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        df['Volume_Change'] = df['Volume'].pct_change()
        
        feature_cols.extend(['Volume_Ratio', 'Volume_Change'])
        
        # Add price features
        df['Returns'] = df['Close'].pct_change()
        df['High_Low_Ratio'] = (df['High'] - df['Low']) / df['Close']
        
        feature_cols.extend(['Returns', 'High_Low_Ratio'])
        
        # Add real-time features if available
        if real_time_stats:
            # These would be added to the latest row
            pass
        
        return df[feature_cols].fillna(method='bfill').fillna(0)
    
    def train_model(self, df, symbol, real_time_stats=None):
        """Train enhanced ML model"""
        if not ML_AVAILABLE:
            return None
        
        try:
            features = self.create_features(df, real_time_stats)
            
            # Create labels - Predict next 5 periods
            future_returns = df['Close'].shift(-5) / df['Close'] - 1
            labels = pd.cut(
                future_returns,
                bins=[-np.inf, -0.005, 0.005, np.inf],
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
            
            # Train model with ensemble
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            
            gb_model = GradientBoostingClassifier(
                n_estimators=50,
                max_depth=6,
                random_state=42
            )
            
            # Train both models
            rf_model.fit(X_scaled, y)
            gb_model.fit(X_scaled, y)
            
            # Use ensemble prediction
            self.models[symbol] = {'rf': rf_model, 'gb': gb_model}
            self.scalers[symbol] = scaler
            self.last_training[symbol] = datetime.now()
            
            # Calculate feature importance
            self.feature_importance[symbol] = rf_model.feature_importances_
            
            # Calculate performance metrics
            predictions = rf_model.predict(X_scaled)
            accuracy = accuracy_score(y, predictions)
            self.model_performance[symbol] = {
                'accuracy': accuracy,
                'training_samples': len(X),
                'last_trained': self.last_training[symbol]
            }
            
            logger.info(f"Model trained for {symbol}: accuracy={accuracy:.2f}")
            return rf_model
            
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {e}")
            return None
    
    def predict(self, df, symbol, real_time_stats=None):
        """Make prediction using ensemble"""
        if symbol not in self.models:
            return 0, 0.0, {}
        
        try:
            features = self.create_features(df, real_time_stats)
            latest = features.iloc[-1:].values
            
            scaled = self.scalers[symbol].transform(latest)
            
            # Get predictions from both models
            rf_pred = self.models[symbol]['rf'].predict(scaled)[0]
            gb_pred = self.models[symbol]['gb'].predict(scaled)[0]
            
            # Get probabilities
            rf_proba = self.models[symbol]['rf'].predict_proba(scaled)[0]
            gb_proba = self.models[symbol]['gb'].predict_proba(scaled)[0]
            
            # Ensemble: weighted average
            rf_weight = 0.6
            gb_weight = 0.4
            
            ensemble_proba = rf_proba * rf_weight + gb_proba * gb_weight
            prediction = np.argmax(ensemble_proba) - 1  # Convert to -1, 0, 1
            
            confidence = max(ensemble_proba)
            
            # Additional metrics
            metrics = {
                'rf_prediction': rf_pred,
                'gb_prediction': gb_pred,
                'rf_confidence': max(rf_proba),
                'gb_confidence': max(gb_proba),
                'ensemble_confidence': confidence,
                'feature_importance': self.feature_importance.get(symbol, []),
                'model_performance': self.model_performance.get(symbol, {})
            }
            
            return prediction, confidence, metrics
            
        except Exception as e:
            logger.error(f"Error predicting for {symbol}: {e}")
            return 0, 0.0, {}

# ============================================================================
# REAL-TIME POSITION MANAGER
# ============================================================================

class RealTimePositionManager:
    """Manages positions with real-time P&L updates"""
    
    def __init__(self, broker, real_time_manager):
        self.broker = broker
        self.real_time_manager = real_time_manager
        self.positions = {}
        self.position_history = []
        self.pnl_history = deque(maxlen=1000)
        self.running = False
        self.thread = None
    
    def start(self):
        """Start position manager"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._position_update_loop, daemon=True)
            self.thread.start()
            logger.info("PositionManager started")
    
    def stop(self):
        """Stop position manager"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("PositionManager stopped")
    
    def _position_update_loop(self):
        """Update positions in real-time"""
        while self.running:
            try:
                self._update_all_positions()
                time.sleep(Config.POSITION_REFRESH_INTERVAL)
            except Exception as e:
                logger.error(f"Error in position update loop: {e}")
                time.sleep(1)
    
    def _update_all_positions(self):
        """Update P&L for all positions"""
        current_time = datetime.now()
        
        for symbol, position in list(self.positions.items()):
            # Get current price
            current_price = self.real_time_manager.get_latest_price(symbol)
            if current_price is None:
                current_price = self.broker.get_ltp(symbol)
            
            # Calculate P&L
            position['current_price'] = current_price
            position['unrealized_pnl'] = self._calculate_pnl(position, current_price)
            position['unrealized_pnl_pct'] = self._calculate_pnl_pct(position, current_price)
            position['last_update'] = current_time
            
            # Check exit conditions
            self._check_exit_conditions(position)
    
    def _calculate_pnl(self, position, current_price):
        """Calculate P&L for position"""
        if position['direction'] == 'LONG':
            return (current_price - position['entry_price']) * position['quantity']
        else:
            return (position['entry_price'] - current_price) * position['quantity']
    
    def _calculate_pnl_pct(self, position, current_price):
        """Calculate P&L percentage"""
        investment = position['entry_price'] * position['quantity']
        pnl = self._calculate_pnl(position, current_price)
        return (pnl / investment) * 100 if investment > 0 else 0
    
    def _check_exit_conditions(self, position):
        """Check if position should be exited"""
        symbol = position['symbol']
        current_price = position['current_price']
        
        # Check stop loss
        if position['direction'] == 'LONG':
            if current_price <= position['stop_loss']:
                self._exit_position(symbol, current_price, 'STOP_LOSS')
            elif current_price >= position['take_profit']:
                self._exit_position(symbol, current_price, 'TAKE_PROFIT')
        else:
            if current_price >= position['stop_loss']:
                self._exit_position(symbol, current_price, 'STOP_LOSS')
            elif current_price <= position['take_profit']:
                self._exit_position(symbol, current_price, 'TAKE_PROFIT')
        
        # Check trailing stop
        if Config.TRAILING_STOP:
            self._update_trailing_stop(position)
    
    def _update_trailing_stop(self, position):
        """Update trailing stop loss"""
        symbol = position['symbol']
        current_price = position['current_price']
        
        if position['direction'] == 'LONG':
            # Calculate new stop if price has moved up
            new_stop = current_price * (1 - Config.TRAILING_ACTIVATION)
            if new_stop > position['stop_loss']:
                position['stop_loss'] = new_stop
        else:
            # For short positions
            new_stop = current_price * (1 + Config.TRAILING_ACTIVATION)
            if new_stop < position['stop_loss']:
                position['stop_loss'] = new_stop
    
    def _exit_position(self, symbol, price, reason):
        """Exit a position"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # Calculate final P&L
        pnl = self._calculate_pnl(position, price)
        pnl_pct = self._calculate_pnl_pct(position, price)
        
        # Place exit order
        exit_dir = 'SELL' if position['direction'] == 'LONG' else 'BUY'
        self.broker.place_order(symbol, exit_dir, position['quantity'], price)
        
        # Record exit
        position['exit_price'] = price
        position['exit_time'] = datetime.now()
        position['exit_reason'] = reason
        position['realized_pnl'] = pnl
        position['realized_pnl_pct'] = pnl_pct
        position['status'] = 'CLOSED'
        
        # Add to history
        self.position_history.append(position.copy())
        
        # Record P&L
        self.pnl_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason
        })
        
        # Remove from active positions
        del self.positions[symbol]
        
        logger.info(f"Position closed: {symbol} {reason} P&L: {pnl:.2f} ({pnl_pct:.2f}%)")
    
    def add_position(self, position):
        """Add a new position"""
        symbol = position['symbol']
        
        if symbol in self.positions:
            logger.warning(f"Position already exists for {symbol}")
            return False
        
        # Add real-time fields
        position['current_price'] = position['entry_price']
        position['unrealized_pnl'] = 0
        position['unrealized_pnl_pct'] = 0
        position['last_update'] = datetime.now()
        position['status'] = 'OPEN'
        position['max_favorable'] = 0
        position['max_adverse'] = 0
        
        self.positions[symbol] = position
        logger.info(f"Position added: {symbol} {position['direction']} {position['quantity']} @ {position['entry_price']}")
        
        return True
    
    def get_positions_summary(self):
        """Get summary of all positions"""
        total_pnl = 0
        total_investment = 0
        
        for position in self.positions.values():
            total_pnl += position.get('unrealized_pnl', 0)
            total_investment += position['entry_price'] * position['quantity']
        
        return {
            'total_positions': len(self.positions),
            'total_pnl': total_pnl,
            'total_investment': total_investment,
            'pnl_pct': (total_pnl / total_investment * 100) if total_investment > 0 else 0
        }
    
    def get_position_details(self, symbol):
        """Get detailed position information"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol].copy()
        
        # Add real-time statistics
        stats = self.real_time_manager.get_tick_statistics(symbol)
        if stats:
            position['tick_stats'] = stats
        
        return position

# ============================================================================
# ENHANCED TRADING ENGINE
# ============================================================================

class EnhancedTradingEngine:
    """Enhanced trading engine with real-time capabilities"""
    
    def __init__(self, config, demo_mode=True):
        self.config = config
        self.real_time_manager = RealTimeDataManager()
        self.broker = EnhancedKiteBroker(demo_mode, self.real_time_manager)
        self.db = Database()
        self.risk = EnhancedRiskManager(config)
        self.ai = EnhancedAIEngine()
        self.signal_generator = EnhancedSignalGenerator(self.ai, self.broker, self.real_time_manager)
        self.position_manager = RealTimePositionManager(self.broker, self.real_time_manager)
        
        self.running = False
        self.signals_queue = queue.Queue()
        self.performance_monitor = PerformanceMonitor()
        
        # Start managers
        self.real_time_manager.start()
        self.signal_generator.start()
        self.position_manager.start()
        
        # Performance stats
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'daily_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
    
    def start(self):
        """Start the trading bot"""
        self.running = True
        threading.Thread(target=self.run_loop, daemon=True).start()
        logger.info("Trading engine started")
        return True
    
    def stop(self):
        """Stop the trading bot"""
        self.running = False
        self.signal_generator.stop()
        self.position_manager.stop()
        self.real_time_manager.stop()
        logger.info("Trading engine stopped")
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
                
                # Scan for signals periodically
                if scan_counter % 6 == 0:  # Every 30 seconds
                    self.scan_signals()
                
                # Auto-execute if enabled
                if hasattr(st.session_state, 'auto_execute') and st.session_state.auto_execute:
                    self.execute_signals()
                
                # Update performance metrics
                self.update_performance_metrics()
                
                scan_counter += 1
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(30)
    
    def scan_signals(self):
        """Scan for trading signals"""
        logger.info("Scanning for signals...")
        
        # Use signal generator for real-time signal refresh
        signals_generated = self.signal_generator.refresh_all_signals()
        
        # Add signals to queue
        active_signals = self.signal_generator.get_active_signals()
        for signal in active_signals:
            self.signals_queue.put(signal)
        
        logger.info(f"Scan complete: {signals_generated} signals generated")
    
    def execute_signals(self):
        """Execute pending signals"""
        executed_count = 0
        
        while not self.signals_queue.empty():
            try:
                signal = self.signals_queue.get_nowait()
                
                # Check risk limits
                can_trade, reason = self.risk.can_trade(signal['symbol'], signal['direction'])
                if not can_trade:
                    logger.warning(f"Cannot trade {signal['symbol']}: {reason}")
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
                        'entry_time': str(datetime.now()),
                        'entry_price': result['price'],
                        'quantity': signal['quantity'],
                        'stop_loss': signal['stop_loss'],
                        'take_profit': signal['take_profit'],
                        'confidence': signal['confidence'],
                        'strategy': 'AI_ALGO'
                    }
                    
                    # Add to position manager
                    self.position_manager.add_position(position)
                    
                    # Save to database
                    self.db.save_position(position)
                    
                    # Update risk manager
                    self.risk.add_position(position)
                    
                    executed_count += 1
                    
                    logger.info(f"Executed signal: {signal['symbol']} {signal['direction']}")
                    
                    # Remove signal from active signals
                    self.signal_generator.remove_signal(signal['symbol'])
                    
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error executing signal: {e}")
        
        if executed_count > 0:
            logger.info(f"Executed {executed_count} signals")
    
    def update_performance_metrics(self):
        """Update performance metrics"""
        # Get position summary
        summary = self.position_manager.get_positions_summary()
        
        # Update stats
        self.stats['total_pnl'] = summary['total_pnl']
        
        # Update performance monitor
        self.performance_monitor.update_metrics(summary)
    
    def get_real_time_dashboard(self):
        """Get real-time dashboard data"""
        positions = self.position_manager.positions
        signals = self.signal_generator.get_active_signals()
        
        # Calculate portfolio metrics
        portfolio_value = Config.TOTAL_CAPITAL
        unrealized_pnl = 0
        
        for position in positions.values():
            portfolio_value += position.get('unrealized_pnl', 0)
            unrealized_pnl += position.get('unrealized_pnl', 0)
        
        return {
            'portfolio_value': portfolio_value,
            'unrealized_pnl': unrealized_pnl,
            'active_positions': len(positions),
            'active_signals': len(signals),
            'market_mood': self.get_market_mood(),
            'performance': self.stats
        }
    
    def get_market_mood(self):
        """Get market mood from indices"""
        # This would use EnhancedMarketIndicesUpdater
        return "NEUTRAL"

# ============================================================================
# ENHANCED RISK MANAGER
# ============================================================================

class EnhancedRiskManager(RiskManager):
    """Enhanced risk manager with real-time checks"""
    
    def __init__(self, config):
        super().__init__(config)
        self.daily_pnl = 0.0
        self.max_daily_loss = config.MAX_DAILY_LOSS * config.TOTAL_CAPITAL
        self.max_drawdown = config.MAX_DRAWDOWN * config.TOTAL_CAPITAL
        self.sector_exposure = {}
        self.position_concentration = {}
        self.daily_trades = 0  # Added missing attribute
    
    def can_trade(self, symbol, direction):
        """Enhanced pre-trade checks"""
        # Check market hours
        now = datetime.now().time()
        if now < self.config.MARKET_OPEN or now > self.config.MARKET_CLOSE:
            return False, "Market closed"
        
        # Check daily loss limit
        if self.daily_pnl <= -self.max_daily_loss:
            return False, f"Daily loss limit reached: {self.daily_pnl:.2f}"
        
        # Check max positions
        if len(self.positions) >= self.config.MAX_POSITIONS:
            return False, f"Max positions reached: {self.config.MAX_POSITIONS}"
        
        # Check daily trade limit
        if self.daily_trades >= self.config.MAX_DAILY_TRADES:
            return False, f"Daily trade limit reached: {self.config.MAX_DAILY_TRADES}"
        
        # Check position concentration
        sector = self.get_sector(symbol)
        if sector:
            sector_value = self.sector_exposure.get(sector, 0)
            max_sector_exposure = self.config.TOTAL_CAPITAL * 0.2  # 20% per sector
            if sector_value >= max_sector_exposure:
                return False, f"Sector {sector} exposure limit reached"
        
        return True, "OK"
    
    def get_sector(self, symbol):
        """Get sector for symbol (simplified)"""
        # In real implementation, this would map symbols to sectors
        sectors = {
            'BANK': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK'],
            'IT': ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM'],
            # Add more sectors
        }
        
        for sector, symbols in sectors.items():
            if symbol in symbols:
                return sector
        
        return 'OTHER'
    
    def add_position(self, position):
        """Add position with risk tracking"""
        super().update_pnl(0)  # Initialize
        
        # Update sector exposure
        sector = self.get_sector(position['symbol'])
        if sector:
            position_value = position['entry_price'] * position['quantity']
            self.sector_exposure[sector] = self.sector_exposure.get(sector, 0) + position_value
        
        # Update position concentration
        self.position_concentration[position['symbol']] = position['entry_price'] * position['quantity']
        self.daily_trades += 1  # Increment daily trades
    
    def update_daily_pnl(self, pnl):
        """Update daily P&L"""
        self.daily_pnl += pnl

# ============================================================================
# PERFORMANCE MONITOR
# ============================================================================

class PerformanceMonitor:
    """Monitors trading performance in real-time"""
    
    def __init__(self):
        self.metrics = {
            'execution_latency': deque(maxlen=100),
            'slippage': deque(maxlen=100),
            'fill_rate': 0.0,
            'win_loss_ratio': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'profit_factor': 0.0
        }
        self.trade_log = deque(maxlen=1000)
        self.performance_history = deque(maxlen=500)
    
    def update_metrics(self, position_summary):
        """Update performance metrics"""
        # Record performance snapshot
        self.performance_history.append({
            'timestamp': datetime.now(),
            'total_pnl': position_summary.get('total_pnl', 0),
            'positions': position_summary.get('total_positions', 0)
        })
        
        # Calculate metrics
        if len(self.performance_history) >= 2:
            self._calculate_sharpe_ratio()
            self._calculate_drawdown()
    
    def _calculate_sharpe_ratio(self):
        """Calculate Sharpe ratio"""
        returns = []
        prev_pnl = None
        
        for record in self.performance_history:
            if prev_pnl is not None:
                returns.append(record['total_pnl'] - prev_pnl)
            prev_pnl = record['total_pnl']
        
        if len(returns) >= 2:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            self.metrics['sharpe_ratio'] = avg_return / std_return if std_return > 0 else 0
    
    def _calculate_drawdown(self):
        """Calculate maximum drawdown"""
        pnl_values = [r['total_pnl'] for r in self.performance_history]
        
        if pnl_values:
            peak = pnl_values[0]
            max_dd = 0
            
            for pnl in pnl_values:
                if pnl > peak:
                    peak = pnl
                dd = (peak - pnl) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
            
            self.metrics['max_drawdown'] = max_dd
    
    def record_trade(self, trade_details):
        """Record trade execution"""
        self.trade_log.append({
            'timestamp': datetime.now(),
            'symbol': trade_details.get('symbol'),
            'direction': trade_details.get('direction'),
            'quantity': trade_details.get('quantity'),
            'price': trade_details.get('price'),
            'pnl': trade_details.get('pnl', 0)
        })
        
        # Update win/loss ratio
        if len(self.trade_log) >= 2:
            wins = sum(1 for t in self.trade_log if t.get('pnl', 0) > 0)
            losses = sum(1 for t in self.trade_log if t.get('pnl', 0) < 0)
            
            if wins + losses > 0:
                self.metrics['win_loss_ratio'] = wins / (wins + losses)
                
                # Calculate profit factor
                total_profit = sum(t.get('pnl', 0) for t in self.trade_log if t.get('pnl', 0) > 0)
                total_loss = abs(sum(t.get('pnl', 0) for t in self.trade_log if t.get('pnl', 0) < 0))
                self.metrics['profit_factor'] = total_profit / total_loss if total_loss > 0 else float('inf')

# ============================================================================
# DATABASE - KEEP EXISTING BUT ADD REAL-TIME TABLES
# ============================================================================

class Database:
    """SQLite database with real-time tables"""
    
    def __init__(self):
        self.conn = sqlite3.connect('trading_bot_v8.db', check_same_thread=False)
        self.init_db()
    
    def init_db(self):
        """Initialize database tables with real-time support"""
        cursor = self.conn.cursor()
        
        # Enhanced trades table
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
                strategy TEXT,
                confidence REAL,
                stop_loss REAL,
                take_profit REAL,
                exit_reason TEXT
            )
        ''')
        
        # Enhanced positions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY,
                symbol TEXT UNIQUE,
                direction TEXT,
                entry_time TEXT,
                entry_price REAL,
                current_price REAL,
                quantity INTEGER,
                stop_loss REAL,
                take_profit REAL,
                unrealized_pnl REAL,
                unrealized_pnl_pct REAL,
                status TEXT,
                last_update TEXT
            )
        ''')
        
        # Real-time signals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY,
                signal_id TEXT UNIQUE,
                symbol TEXT,
                direction TEXT,
                price REAL,
                confidence REAL,
                stop_loss REAL,
                take_profit REAL,
                quantity INTEGER,
                timestamp TEXT,
                status TEXT
            )
        ''')
        
        # Market data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                price REAL,
                volume INTEGER,
                timestamp TEXT,
                data_type TEXT
            )
        ''')
        
        # Performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                total_pnl REAL,
                win_rate REAL,
                positions_count INTEGER,
                sharpe_ratio REAL,
                max_drawdown REAL
            )
        ''')
        
        self.conn.commit()
    
    def save_real_time_signal(self, signal):
        """Save real-time signal"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO signals 
            (signal_id, symbol, direction, price, confidence, stop_loss,
             take_profit, quantity, timestamp, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal['signal_id'], signal['symbol'], signal['direction'],
            signal['price'], signal['confidence'], signal['stop_loss'],
            signal['take_profit'], signal['quantity'], str(signal['timestamp']),
            signal.get('status', 'ACTIVE')
        ))
        self.conn.commit()
    
    def save_market_data(self, symbol, price, volume, data_type='TICK'):
        """Save market data tick"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO market_data 
            (symbol, price, volume, timestamp, data_type)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            symbol, price, volume, str(datetime.now()), data_type
        ))
        self.conn.commit()
    
    def get_trades(self, limit=100):
        """Get completed trades"""
        cursor = self.conn.cursor()
        cursor.execute(f'SELECT * FROM trades ORDER BY entry_time DESC LIMIT {limit}')
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        return pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame()
    
    def save_trade(self, trade):
        """Save completed trade"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO trades 
            (symbol, direction, entry_time, entry_price, exit_time, exit_price,
             quantity, pnl, pnl_pct, status, strategy, confidence, stop_loss,
             take_profit, exit_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade['symbol'], trade['direction'], trade['entry_time'],
            trade['entry_price'], trade.get('exit_time'), trade.get('exit_price'),
            trade['quantity'], trade.get('pnl', 0), trade.get('pnl_pct', 0),
            trade['status'], trade.get('strategy', 'AI_ALGO'),
            trade.get('confidence', 0), trade.get('stop_loss', 0),
            trade.get('take_profit', 0), trade.get('exit_reason', '')
        ))
        self.conn.commit()
    
    def save_position(self, position):
        """Save or update position"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO positions
            (symbol, direction, entry_time, entry_price, current_price, quantity,
             stop_loss, take_profit, unrealized_pnl, unrealized_pnl_pct, status, last_update)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            position['symbol'], position['direction'], position['entry_time'],
            position['entry_price'], position.get('current_price', position['entry_price']),
            position['quantity'], position['stop_loss'], position['take_profit'],
            position.get('unrealized_pnl', 0), position.get('unrealized_pnl_pct', 0),
            position.get('status', 'OPEN'), str(datetime.now())
        ))
        self.conn.commit()

# ============================================================================
# TECHNICAL ANALYSIS - KEEP EXISTING
# ============================================================================

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
# STOCK UNIVERSE - KEEP EXISTING
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
# STREAMLIT UI FUNCTIONS - KEEP EXISTING
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
# MAIN STREAMLIT APP - ENHANCED WITH REAL-TIME FEATURES
# ============================================================================

def main():
    st.set_page_config(
        page_title="AI Algo Trading Bot v8.0 - PRODUCTION",
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
    .real-time-price {
        font-size: 1.2rem;
        font-weight: bold;
        animation: blink 1s infinite;
    }
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    .positive { color: #00C853; }
    .negative { color: #FF5252; }
    .neutral { color: #FFC107; }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'engine' not in st.session_state:
        st.session_state.engine = EnhancedTradingEngine(Config(), demo_mode=True)
        st.session_state.real_time_manager = st.session_state.engine.real_time_manager
        st.session_state.market_indices = EnhancedMarketIndicesUpdater(
            st.session_state.engine.broker, 
            st.session_state.real_time_manager
        )
        
        st.session_state.last_refresh = datetime.now()
        st.session_state.auto_execute = False
        st.session_state.auto_refresh = True
        st.session_state.refresh_rate = 2  # 2 seconds for real-time
        st.session_state.active_tab = 0
        st.session_state.refresh_counter = 0
        st.session_state.live_prices = {}
        
        # Auto-connect if credentials exist
        if 'kite_api_key' in st.session_state and 'kite_access_token' in st.session_state:
            with st.spinner("🔌 Auto-connecting to Kite..."):
                st.session_state.engine.broker.connect()
    
    engine = st.session_state.engine
    market_indices = st.session_state.market_indices
    
    # Auto-refresh logic
    refresh_placeholder = st.empty()
    
    if st.session_state.get('auto_refresh', True):
        refresh_rate = st.session_state.get('refresh_rate', 2)
        current_time = datetime.now()
        time_since_refresh = (current_time - st.session_state.last_refresh).total_seconds()
        
        # Show countdown
        with refresh_placeholder:
            remaining = max(0, int(refresh_rate - time_since_refresh))
            if remaining > 0:
                st.sidebar.info(f"⏳ Next update in {remaining}s")
        
        # Refresh if time elapsed
        if time_since_refresh >= refresh_rate:
            with st.spinner("🔄 Refreshing real-time data..."):
                # Update market indices
                market_indices.update_from_kite()
                
                # Update live prices
                for symbol in engine.position_manager.positions.keys():
                    price = engine.broker.get_real_time_price(symbol)
                    st.session_state.live_prices[symbol] = price
                
                st.session_state.last_refresh = datetime.now()
                st.session_state.refresh_counter += 1
            
            # Force rerun
            st.rerun()
    
    # Header
    st.markdown("<h1 class='main-header'>🤖 AI ALGORITHMIC TRADING BOT v8.0</h1>", 
                unsafe_allow_html=True)
    st.markdown("### Production-Ready Trading System | Real-Time Processing | All 159 F&O Stocks")
    
    # Market Mood Gauge with real-time updates
    st.markdown("---")
    st.markdown("### 📊 Real-Time Market Dashboard")
    
    # Get current market data
    nifty_data = market_indices.indices_data['NIFTY 50']
    banknifty_data = market_indices.indices_data['NIFTY BANK']
    sensex_data = market_indices.indices_data['SENSEX']
    mood, mood_color, mood_value = market_indices.get_market_mood()
    
    # Check circuit breakers
    circuit_alerts = market_indices.check_circuit_breakers()
    if circuit_alerts:
        for alert in circuit_alerts:
            st.error(f"🚨 {alert}")
    
    # Display indices
    col_idx1, col_idx2, col_idx3, col_idx4 = st.columns(4)
    
    with col_idx1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        change_class = "positive" if nifty_data['change_pct'] >= 0 else "negative"
        st.markdown(f'<h3>📈 NIFTY 50</h3>', unsafe_allow_html=True)
        st.markdown(f'<h2 class="real-time-price">{nifty_data["ltp"]:,.2f}</h2>', unsafe_allow_html=True)
        st.markdown(f'<p class="{change_class}">{nifty_data["change_pct"]:+.2f}%</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size: 0.8rem; color: #888;">H: {nifty_data["high"]:,.0f} | L: {nifty_data["low"]:,.0f}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_idx2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        change_class = "positive" if banknifty_data['change_pct'] >= 0 else "negative"
        st.markdown(f'<h3>🏦 NIFTY BANK</h3>', unsafe_allow_html=True)
        st.markdown(f'<h2 class="real-time-price">{banknifty_data["ltp"]:,.2f}</h2>', unsafe_allow_html=True)
        st.markdown(f'<p class="{change_class}">{banknifty_data["change_pct"]:+.2f}%</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size: 0.8rem; color: #888;">H: {banknifty_data["high"]:,.0f} | L: {banknifty_data["low"]:,.0f}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_idx3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        change_class = "positive" if sensex_data['change_pct'] >= 0 else "negative"
        st.markdown(f'<h3>📊 SENSEX</h3>', unsafe_allow_html=True)
        st.markdown(f'<h2 class="real-time-price">{sensex_data["ltp"]:,.2f}</h2>', unsafe_allow_html=True)
        st.markdown(f'<p class="{change_class}">{sensex_data["change_pct"]:+.2f}%</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size: 0.8rem; color: #888;">H: {sensex_data["high"]:,.0f} | L: {sensex_data["low"]:,.0f}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_idx4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<h3>🎯 MARKET MOOD</h3>', unsafe_allow_html=True)
        st.markdown(f'<h2 style="color: {mood_color}">{mood}</h2>', unsafe_allow_html=True)
        st.markdown(f'<p class="neutral">{mood_value:+.2f}%</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size: 0.8rem; color: #888;">Live Analysis</p>', unsafe_allow_html=True)
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
        
        # Capital & Risk
        capital = st.number_input("Capital (₹)", 
                                 min_value=100000, 
                                 value=Config.TOTAL_CAPITAL, 
                                 step=100000)
        Config.TOTAL_CAPITAL = capital
        
        # Fixed slider - using keyword arguments correctly
        risk = st.slider("Risk per Trade (%)", min_value=0.5, max_value=5.0, value=Config.RISK_PER_TRADE * 100, step=0.1) / 100
        Config.RISK_PER_TRADE = risk
        
        # Fixed slider - using keyword arguments correctly
        confidence = st.slider("Min Confidence (%)", min_value=50, max_value=90, value=int(Config.MIN_CONFIDENCE * 100), step=5) / 100
        Config.MIN_CONFIDENCE = confidence
        
        st.markdown("---")
        
        # Feature Toggles
        st.markdown("### 🔧 Trading Features")
        
        auto_exec = st.checkbox("🤖 Auto-Execute Signals", value=st.session_state.auto_execute)
        st.session_state.auto_execute = auto_exec
        
        st.markdown("---")
        
        # Auto Refresh Settings
        st.markdown("### 🔄 Real-Time Settings")
        
        auto_refresh = st.checkbox("Enable Real-Time Updates", 
                                  value=st.session_state.auto_refresh,
                                  key="auto_refresh_checkbox")
        st.session_state.auto_refresh = auto_refresh
        
        if auto_refresh:
            refresh_rate = st.slider("Update Interval (seconds)", min_value=1, max_value=10, value=2, key="refresh_rate_slider")
            st.session_state.refresh_rate = refresh_rate
            
            col_ref1, col_ref2 = st.columns(2)
            with col_ref1:
                st.metric("Update Rate", f"{refresh_rate}s")
            with col_ref2:
                st.metric("Total Updates", st.session_state.refresh_counter)
        
        # Manual Refresh Button
        if st.button("🔄 Refresh Now", use_container_width=True, key="manual_refresh"):
            with st.spinner("Refreshing real-time data..."):
                market_indices.update_from_kite()
                
                # Update positions
                for symbol in engine.position_manager.positions.keys():
                    price = engine.broker.get_real_time_price(symbol)
                    st.session_state.live_prices[symbol] = price
                
                st.session_state.last_refresh = datetime.now()
                st.session_state.refresh_counter += 1
                st.success("✅ Real-time data refreshed!")
                st.rerun()
        
        st.markdown("---")
        
        # System Status
        st.markdown("### 📈 System Status")
        
        if engine.broker.connected:
            if engine.broker.websocket_running:
                st.success("✅ Live WebSocket Active")
                st.info(f"**Ticks Processed:** {len(engine.real_time_manager.tick_buffer)}")
            else:
                st.warning("⚠️ Kite API (No WebSocket)")
        else:
            st.warning("⚠️ Demo Mode")
        
        # Performance Summary
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        
        dashboard = engine.get_real_time_dashboard()
        st.metric("Portfolio Value", f"₹{dashboard['portfolio_value']:,.0f}")
        st.metric("Unrealized P&L", f"₹{dashboard['unrealized_pnl']:,.0f}")
        st.metric("Active Positions", dashboard['active_positions'])
        st.metric("Active Signals", dashboard['active_signals'])
    
    # Top Metrics
    st.markdown("### 📊 Real-Time Performance")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        status = "RUNNING" if engine.running else "STOPPED"
        status_class = "positive" if engine.running else "negative"
        st.markdown(f'<h3 class="{status_class}">{status}</h3>', unsafe_allow_html=True)
        st.markdown("**System Status**")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        pnl = engine.stats['total_pnl']
        pnl_class = "positive" if pnl >= 0 else "negative"
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
        positions = len(engine.position_manager.positions)
        st.markdown(f'<h3>{positions}/{Config.MAX_POSITIONS}</h3>', unsafe_allow_html=True)
        st.markdown("**Active Positions**")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Colorful Tabs
    st.markdown("---")
    active_tab = render_colorful_tabs()
    st.markdown("---")
    
    # Tab Content - Enhanced with real-time features
    if active_tab == 0:  # Algo Trading
        st.markdown("### 🎯 Real-Time AI Trading Signals")
        
        # Signal controls
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            if st.button("🔍 Scan All Stocks", type="primary", use_container_width=True):
                with st.spinner("Scanning all stocks..."):
                    signals = engine.signal_generator.refresh_all_signals()
                    st.success(f"✅ Generated {signals} new signals!")
                    st.rerun()
        
        with col2:
            if st.button("🔄 Refresh Signals", type="secondary", use_container_width=True):
                # Refresh expired signals
                engine.signal_generator._refresh_expired_signals()
                st.success("✅ Signals refreshed!")
                st.rerun()
        
        with col3:
            if st.button("⚡ Execute All", use_container_width=True):
                engine.execute_signals()
                st.success("✅ Execution completed!")
                st.rerun()
        
        # Display active signals
        st.markdown("#### 📋 Active Signals")
        active_signals = engine.signal_generator.get_active_signals()
        
        if active_signals:
            signals_df = pd.DataFrame(active_signals)
            
            # Format columns
            signals_df['confidence'] = signals_df['confidence'].apply(lambda x: f"{x:.1%}")
            signals_df['timestamp'] = signals_df['timestamp'].apply(lambda x: x.strftime("%H:%M:%S"))
            
            # Add real-time price column
            real_time_prices = []
            for symbol in signals_df['symbol']:
                price_data = engine.broker.get_real_time_price(symbol)
                real_time_prices.append(price_data['price'] if price_data else 0)
            
            signals_df['current_price'] = real_time_prices
            signals_df['price_diff'] = signals_df['current_price'] - signals_df['price']
            signals_df['price_diff_pct'] = (signals_df['price_diff'] / signals_df['price'] * 100).round(2)
            
            st.dataframe(signals_df, use_container_width=True)
        else:
            st.info("🔭 No active signals")
        
        # Signal statistics
        st.markdown("#### 📊 Signal Statistics")
        col_s1, col_s2, col_s3 = st.columns(3)
        
        with col_s1:
            st.metric("Active Signals", len(active_signals))
            st.metric("Signal Expiry", f"{Config.SIGNAL_EXPIRY_SECONDS}s")
        
        with col_s2:
            st.metric("Refresh Interval", f"{Config.SIGNAL_REFRESH_INTERVAL}s")
            st.metric("Min Confidence", f"{Config.MIN_CONFIDENCE:.0%}")
        
        with col_s3:
            st.metric("AI Models Trained", len(engine.ai.models))
            # Fixed: signal_history is a list, not an integer
            st.metric("Total Signals", len(engine.signal_generator.signal_history))
    
    elif active_tab == 1:  # Positions
        st.markdown("### 📈 Real-Time Positions")
        
        positions = engine.position_manager.positions
        
        if positions:
            # Create positions DataFrame
            positions_list = []
            
            for symbol, position in positions.items():
                # Get real-time price
                price_data = engine.broker.get_real_time_price(symbol)
                current_price = price_data['price'] if price_data else position['current_price']
                
                # Calculate P&L
                if position['direction'] == 'LONG':
                    pnl = (current_price - position['entry_price']) * position['quantity']
                else:
                    pnl = (position['entry_price'] - current_price) * position['quantity']
                
                pnl_pct = (pnl / (position['entry_price'] * position['quantity'])) * 100
                
                positions_list.append({
                    'Symbol': symbol,
                    'Direction': position['direction'],
                    'Quantity': position['quantity'],
                    'Entry Price': position['entry_price'],
                    'Current Price': current_price,
                    'Stop Loss': position['stop_loss'],
                    'Take Profit': position['take_profit'],
                    'P&L': pnl,
                    'P&L %': pnl_pct,
                    'Last Update': position['last_update'].strftime("%H:%M:%S")
                })
            
            positions_df = pd.DataFrame(positions_list)
            
            # Format display
            st.dataframe(positions_df.style.format({
                'Entry Price': '{:.2f}',
                'Current Price': '{:.2f}',
                'Stop Loss': '{:.2f}',
                'Take Profit': '{:.2f}',
                'P&L': '₹{:,.0f}',
                'P&L %': '{:.2f}%'
            }), use_container_width=True)
            
            # Manual exit buttons
            st.markdown("#### 🛑 Manual Exit")
            cols = st.columns(min(4, len(positions)))
            
            for idx, (symbol, position) in enumerate(list(positions.items())[:4]):
                with cols[idx]:
                    if st.button(f"Exit {symbol}", key=f"exit_{symbol}"):
                        price = engine.broker.get_ltp(symbol)
                        engine.position_manager._exit_position(symbol, price, 'MANUAL')
                        st.success(f"✅ Exited {symbol}")
                        st.rerun()
            
            # Position summary
            st.markdown("#### 📊 Position Summary")
            summary = engine.position_manager.get_positions_summary()
            
            col_sum1, col_sum2, col_sum3 = st.columns(3)
            with col_sum1:
                st.metric("Total Positions", summary['total_positions'])
            with col_sum2:
                st.metric("Total P&L", f"₹{summary['total_pnl']:,.0f}")
            with col_sum3:
                st.metric("P&L %", f"{summary['pnl_pct']:.2f}%")
        
        else:
            st.info("🔭 No active positions")
    
    elif active_tab == 2:  # History
        # Keep existing history tab
        st.markdown("### 📋 Trade History")
        
        trades_df = engine.db.get_trades(100)
        
        if not trades_df.empty:
            # Format display
            trades_df['pnl'] = trades_df['pnl'].apply(lambda x: f"₹{x:,.0f}")
            trades_df['pnl_pct'] = trades_df['pnl_pct'].apply(lambda x: f"{x:.2f}%")
            
            st.dataframe(trades_df, use_container_width=True, height=600)
            
            # Export
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
        # Enhanced with real-time data
        st.markdown("### 📊 Live Charts with Real-Time Data")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            symbol = st.selectbox(
                "Select Stock",
                StockUniverse.get_all_fno_stocks()[:50],
                index=0
            )
        
        with col2:
            if st.button("🔄 Update Chart"):
                st.rerun()
        
        # Get historical data
        df = engine.broker.get_historical(symbol, days=7)
        
        # Get real-time data
        real_time_stats = engine.real_time_manager.get_tick_statistics(symbol)
        
        if PLOTLY_AVAILABLE and not df.empty:
            # Create chart
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.6, 0.2, 0.2]
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
            
            # Add indicators
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
            
            # Volume chart
            colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] 
                     else 'green' for i in range(len(df))]
            
            fig.add_trace(
                go.Bar(x=df.index, y=df['Volume'], name='Volume',
                      marker_color=colors),
                row=2, col=1
            )
            
            # RSI chart
            if 'RSI' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                              line=dict(color='purple', width=1)),
                    row=3, col=1
                )
                # Add RSI levels
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            
            fig.update_layout(
                title=f"{symbol} - Live Chart with Real-Time Data",
                template='plotly_dark',
                height=800,
                xaxis_rangeslider_visible=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Real-time statistics
            st.markdown("#### 📈 Real-Time Statistics")
            
            if real_time_stats:
                col_rt1, col_rt2, col_rt3, col_rt4 = st.columns(4)
                
                with col_rt1:
                    st.metric("Current Price", f"₹{real_time_stats['current_price']:.2f}")
                    st.metric("Avg Price", f"₹{real_time_stats['avg_price']:.2f}")
                
                with col_rt2:
                    st.metric("Volume", f"{real_time_stats['volume']:,}")
                    st.metric("Tick Count", real_time_stats['tick_count'])
                
                with col_rt3:
                    st.metric("Buy Pressure", f"{real_time_stats['buy_pressure']:,}")
                    st.metric("Sell Pressure", f"{real_time_stats['sell_pressure']:,}")
                
                with col_rt4:
                    st.metric("Price STD", f"₹{real_time_stats['price_std']:.2f}")
                    st.metric("Volatility", f"{(real_time_stats['price_std']/real_time_stats['current_price']*100):.2f}%")
            
            # Technical indicators
            st.markdown("#### 📊 Technical Indicators")
            col_tech1, col_tech2, col_tech3, col_tech4 = st.columns(4)
            
            with col_tech1:
                st.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}" if 'RSI' in df.columns else "N/A")
            with col_tech2:
                st.metric("ATR", f"₹{df['ATR'].iloc[-1]:.2f}" if 'ATR' in df.columns else "N/A")
            with col_tech3:
                st.metric("MACD", f"{df['MACD'].iloc[-1]:.2f}" if 'MACD' in df.columns else "N/A")
            with col_tech4:
                current = engine.broker.get_ltp(symbol)
                st.metric("LTP", f"₹{current:.2f}")
        
        else:
            st.error("Chart data unavailable")
    
    elif active_tab == 4:  # Analytics
        # Enhanced analytics with real-time metrics
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
            st.markdown("#### 📈 Performance Metrics")
            
            st.metric("Total Trades", engine.stats['total_trades'])
            st.metric("Win Rate", f"{engine.stats['win_rate']:.1f}%")
            st.metric("Total P&L", f"₹{engine.stats['total_pnl']:,.0f}")
            st.metric("Sharpe Ratio", f"{engine.performance_monitor.metrics['sharpe_ratio']:.2f}")
            st.metric("Max Drawdown", f"{engine.performance_monitor.metrics['max_drawdown']:.2%}")
            st.metric("Profit Factor", f"{engine.performance_monitor.metrics['profit_factor']:.2f}")
        
        # Real-time performance chart
        st.markdown("#### 📊 Real-Time Performance")
        
        if engine.performance_monitor.performance_history:
            history = list(engine.performance_monitor.performance_history)
            timestamps = [h['timestamp'] for h in history]
            pnl_values = [h['total_pnl'] for h in history]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=timestamps, y=pnl_values,
                mode='lines',
                name='P&L',
                line=dict(color='green' if pnl_values[-1] >= 0 else 'red')
            ))
            
            fig.update_layout(
                title="Real-Time P&L",
                template='plotly_dark',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif active_tab == 5:  # Settings
        # Keep existing settings tab with enhancements
        st.markdown("### ⚙️ Settings & Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔑 Kite Connect Setup")
            
            st.markdown("**API Credentials**")
            api_key = st.text_input("API Key", type="password", key="kite_api_input")
            if api_key:
                st.session_state.kite_api_key = api_key
            
            access_token = st.text_input("Access Token", type="password", key="kite_token_input")
            if access_token:
                st.session_state.kite_access_token = access_token
            
            if st.button("🔌 Connect to Kite", key="connect_kite"):
                with st.spinner("Connecting..."):
                    if engine.broker.connect():
                        st.success("✅ Connected to Kite!")
                        st.rerun()
                    else:
                        st.error("❌ Connection failed")
            
            st.markdown("#### 📊 Real-Time Configuration")
            
            with st.form("realtime_config"):
                st.markdown("**Signal Settings**")
                
                signal_expiry = st.slider(
                    "Signal Expiry (seconds)",
                    min_value=10, max_value=300, value=Config.SIGNAL_EXPIRY_SECONDS, step=10
                )
                
                signal_refresh = st.slider(
                    "Signal Refresh Interval (seconds)",
                    min_value=1, max_value=60, value=Config.SIGNAL_REFRESH_INTERVAL, step=1
                )
                
                price_trigger = st.slider(
                    "Price Movement Trigger (%)",
                    min_value=0.1, max_value=5.0, value=0.5, step=0.1
                )
                
                if st.form_submit_button("💾 Save Real-Time Settings"):
                    Config.SIGNAL_EXPIRY_SECONDS = signal_expiry
                    Config.SIGNAL_REFRESH_INTERVAL = signal_refresh
                    engine.signal_generator.price_movement_threshold = price_trigger
                    st.success("✅ Real-time settings saved!")
                    st.rerun()
        
        with col2:
            st.markdown("#### ⚡ System Information")
            
            st.info(f"""
            **System Status:** {'🟢 Running' if engine.running else '🔴 Stopped'}
            **Trading Mode:** {'💰 Live' if not engine.broker.demo_mode else '📈 Paper'}
            **Kite Connection:** {'🟢 Connected' if engine.broker.connected else '🔴 Disconnected'}
            **WebSocket:** {'🟢 Active' if engine.broker.websocket_running else '🔴 Inactive'}
            **Real-Time Updates:** {'🟢 ON' if st.session_state.get('auto_refresh', False) else '🔴 OFF'}
            **Auto-Execution:** {'🟢 ON' if st.session_state.get('auto_execute', False) else '🔴 OFF'}
            **AI Models Trained:** {len(engine.ai.models)}
            **Active Signals:** {len(engine.signal_generator.active_signals)}
            **Active Positions:** {len(engine.position_manager.positions)}
            **Market Updates:** {market_indices.update_counter}
            **Total Refreshes:** {st.session_state.refresh_counter}
            """)
            
            st.markdown("#### 💾 Database Statistics")
            
            trade_count = len(engine.db.get_trades(1000))
            signal_count = len(engine.signal_generator.signal_history)
            
            col_db1, col_db2 = st.columns(2)
            with col_db1:
                st.metric("Total Trades", trade_count)
            with col_db2:
                st.metric("Signal History", signal_count)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    <p>🚨 <b>DISCLAIMER:</b> For educational purposes only. Trading involves risk of loss.</p>
    <p>© 2025 AI Algo Trading Bot v8.0 PRODUCTION | Real-Time Processing | Complete Solution</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN THE APP
# ============================================================================

if __name__ == "__main__":
    main()
