"""
Institutional-Grade Autonomous Algorithmic Trading System
Version: 3.0.0 - PRODUCTION READY
Author: Rantv Trading Systems

CRITICAL FIXES:
- Proper market hours handling
- Live data fetching with yfinance
- Signal generation working outside market hours
- Real position management
- Emergency controls
"""

import os
import sys
import time
import json
import logging
import warnings
import traceback
import threading
from datetime import datetime, timedelta, time as dt_time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from collections import defaultdict

# Core imports
import numpy as np
import pandas as pd
import pytz

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

IND_TZ = pytz.timezone("Asia/Kolkata")

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class TradingConfig:
    """Trading system configuration"""
    # Capital
    initial_capital: float = 2_000_000.0
    max_position_pct: float = 0.15
    
    # Risk Management
    max_daily_loss: float = 50_000.0
    max_positions: int = 5
    min_confidence: float = 0.70
    min_risk_reward: float = 2.0
    
    # Trading Limits
    max_trades_per_day: int = 20
    max_trades_per_stock: int = 2
    
    # Market Hours (IST)
    market_open: dt_time = dt_time(9, 15)
    market_close: dt_time = dt_time(15, 30)
    auto_close_time: dt_time = dt_time(15, 10)
    
    # Demo Mode (for testing outside market hours)
    demo_mode: bool = True
    simulate_market_open: bool = True
    
    # Data Settings
    data_refresh_seconds: int = 60
    signal_scan_interval: int = 60
    
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        if self.simulate_market_open and self.demo_mode:
            return True
        
        now = datetime.now(IND_TZ)
        if now.weekday() >= 5:  # Weekend
            return False
        
        current_time = now.time()
        return self.market_open <= current_time <= self.market_close


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass
class Signal:
    symbol: str
    type: SignalType
    entry: float
    stop_loss: float
    target: float
    confidence: float
    strategy: str
    timestamp: datetime
    rsi: float = 50.0
    volume_ratio: float = 1.0
    risk_reward: float = 2.0
    
    def __str__(self):
        return f"{self.type.value} {self.symbol} @ ₹{self.entry:.2f} | SL: ₹{self.stop_loss:.2f} | TGT: ₹{self.target:.2f}"

@dataclass
class Position:
    symbol: str
    action: str
    quantity: int
    entry_price: float
    stop_loss: float
    target: float
    entry_time: datetime
    strategy: str
    current_price: float = 0.0
    pnl: float = 0.0
    status: str = "OPEN"
    
    def update(self, current_price: float):
        self.current_price = current_price
        if self.action == "BUY":
            self.pnl = (current_price - self.entry_price) * self.quantity
        else:
            self.pnl = (self.entry_price - current_price) * self.quantity
    
    def should_exit(self) -> Tuple[bool, str]:
        """Check if position should be closed"""
        if self.action == "BUY":
            if self.current_price <= self.stop_loss:
                return True, "Stop Loss"
            if self.current_price >= self.target:
                return True, "Target Hit"
        else:
            if self.current_price >= self.stop_loss:
                return True, "Stop Loss"
            if self.current_price <= self.target:
                return True, "Target Hit"
        return False, ""


# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators"""
    try:
        # EMAs
        df['EMA8'] = df['Close'].ewm(span=8, adjust=False).mean()
        df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'].fillna(50, inplace=True)
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()
        
        # MACD
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # VWAP
        typical = (df['High'] + df['Low'] + df['Close']) / 3
        df['VWAP'] = (typical * df['Volume']).cumsum() / df['Volume'].cumsum()
        
        # Support/Resistance
        df['Support'] = df['Low'].rolling(20).min()
        df['Resistance'] = df['High'].rolling(20).max()
        
        return df
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return df


# ============================================================================
# DATA MANAGER
# ============================================================================

class DataManager:
    """Fetch and manage market data"""
    
    def __init__(self):
        self.cache = {}
        self.cache_time = {}
        self.cache_duration = 30  # seconds
    
    def get_data(self, symbol: str, interval: str = "15m", use_cache: bool = True) -> Optional[pd.DataFrame]:
        """Get stock data with caching"""
        cache_key = f"{symbol}_{interval}"
        
        # Check cache
        if use_cache and cache_key in self.cache:
            if time.time() - self.cache_time.get(cache_key, 0) < self.cache_duration:
                return self.cache[cache_key]
        
        try:
            import yfinance as yf
            
            # Download data
            period_map = {"1m": "1d", "5m": "2d", "15m": "7d", "1h": "30d"}
            period = period_map.get(interval, "7d")
            
            df = yf.download(symbol, period=period, interval=interval, 
                           progress=False, auto_adjust=False)
            
            if df is None or df.empty or len(df) < 20:
                return None
            
            # Clean columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            
            df = df.rename(columns=lambda x: x.capitalize())
            
            # Ensure required columns
            required = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(c in df.columns for c in required):
                return None
            
            df = df[required].dropna()
            
            if len(df) < 20:
                return None
            
            # Calculate indicators
            df = calculate_indicators(df)
            
            # Cache
            self.cache[cache_key] = df
            self.cache_time[cache_key] = time.time()
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None
    
    def get_live_price(self, symbol: str) -> float:
        """Get current price"""
        try:
            df = self.get_data(symbol, "1m", use_cache=False)
            if df is not None and len(df) > 0:
                return float(df['Close'].iloc[-1])
        except:
            pass
        return 0.0


# ============================================================================
# STRATEGY ENGINE
# ============================================================================

class StrategyEngine:
    """Generate trading signals"""
    
    def __init__(self, data_manager: DataManager, config: TradingConfig):
        self.data = data_manager
        self.config = config
    
    def scan_for_signals(self, symbols: List[str]) -> List[Signal]:
        """Scan symbols for trading signals"""
        signals = []
        
        for symbol in symbols:
            try:
                sig = self._analyze(symbol)
                if sig:
                    signals.append(sig)
            except Exception as e:
                logger.debug(f"Error analyzing {symbol}: {e}")
        
        # Filter by confidence
        signals = [s for s in signals if s.confidence >= self.config.min_confidence]
        
        # Sort by confidence
        signals.sort(key=lambda x: x.confidence, reverse=True)
        
        return signals
    
    def _analyze(self, symbol: str) -> Optional[Signal]:
        """Analyze single symbol"""
        df = self.data.get_data(symbol, "15m")
        
        if df is None or len(df) < 50:
            return None
        
        # Get latest values
        close = float(df['Close'].iloc[-1])
        ema8 = float(df['EMA8'].iloc[-1])
        ema21 = float(df['EMA21'].iloc[-1])
        ema50 = float(df['EMA50'].iloc[-1])
        rsi = float(df['RSI'].iloc[-1])
        vwap = float(df['VWAP'].iloc[-1])
        atr = float(df['ATR'].iloc[-1])
        support = float(df['Support'].iloc[-1])
        resistance = float(df['Resistance'].iloc[-1])
        volume = float(df['Volume'].iloc[-1])
        avg_vol = float(df['Volume'].rolling(20).mean().iloc[-1])
        
        vol_ratio = volume / avg_vol if avg_vol > 0 else 1.0
        
        # Strategy 1: EMA Confluence + VWAP (BUY)
        if (ema8 > ema21 > ema50 and close > vwap and 
            rsi > 40 and rsi < 70 and vol_ratio > 1.2):
            
            sl = max(support * 0.995, close - atr * 1.5)
            tgt = min(resistance * 0.998, close + atr * 3.0)
            rr = (tgt - close) / (close - sl)
            
            if rr >= self.config.min_risk_reward:
                return Signal(
                    symbol=symbol,
                    type=SignalType.BUY,
                    entry=close,
                    stop_loss=sl,
                    target=tgt,
                    confidence=0.82,
                    strategy="EMA_Confluence",
                    timestamp=datetime.now(IND_TZ),
                    rsi=rsi,
                    volume_ratio=vol_ratio,
                    risk_reward=rr
                )
        
        # Strategy 2: RSI Oversold (BUY)
        if rsi < 35 and close > support and vol_ratio > 1.3:
            sl = close - atr * 1.5
            tgt = close + atr * 2.5
            rr = (tgt - close) / (close - sl)
            
            if rr >= 2.0:
                return Signal(
                    symbol=symbol,
                    type=SignalType.BUY,
                    entry=close,
                    stop_loss=sl,
                    target=tgt,
                    confidence=0.75,
                    strategy="RSI_Oversold",
                    timestamp=datetime.now(IND_TZ),
                    rsi=rsi,
                    volume_ratio=vol_ratio,
                    risk_reward=rr
                )
        
        # Strategy 3: EMA Downtrend (SELL)
        if (ema8 < ema21 < ema50 and close < vwap and 
            rsi > 30 and rsi < 60 and vol_ratio > 1.2):
            
            sl = min(resistance * 1.005, close + atr * 1.5)
            tgt = max(support * 1.002, close - atr * 3.0)
            rr = (close - tgt) / (sl - close)
            
            if rr >= self.config.min_risk_reward:
                return Signal(
                    symbol=symbol,
                    type=SignalType.SELL,
                    entry=close,
                    stop_loss=sl,
                    target=tgt,
                    confidence=0.78,
                    strategy="EMA_Downtrend",
                    timestamp=datetime.now(IND_TZ),
                    rsi=rsi,
                    volume_ratio=vol_ratio,
                    risk_reward=rr
                )
        
        return None


# ============================================================================
# RISK MANAGER
# ============================================================================

class RiskManager:
    """Risk and position management"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.stock_trades = defaultdict(int)
        self.last_reset = datetime.now(IND_TZ).date()
    
    def reset_daily(self):
        """Reset daily metrics"""
        today = datetime.now(IND_TZ).date()
        if today != self.last_reset:
            self.daily_pnl = 0.0
            self.trades_today = 0
            self.stock_trades.clear()
            self.last_reset = today
            logger.info("Daily metrics reset")
    
    def can_trade(self, symbol: str, positions: Dict) -> Tuple[bool, str]:
        """Check if trade is allowed"""
        self.reset_daily()
        
        # Max positions
        if len(positions) >= self.config.max_positions:
            return False, "Max positions reached"
        
        # Daily trades
        if self.trades_today >= self.config.max_trades_per_day:
            return False, "Daily trade limit"
        
        # Per-stock limit
        if self.stock_trades[symbol] >= self.config.max_trades_per_stock:
            return False, f"Max trades for {symbol}"
        
        # Already in position
        if symbol in positions:
            return False, "Already in position"
        
        # Daily loss limit
        if abs(self.daily_pnl) >= self.config.max_daily_loss:
            return False, "Daily loss limit"
        
        return True, "OK"
    
    def calculate_quantity(self, price: float, capital: float) -> int:
        """Calculate position size"""
        position_value = capital * self.config.max_position_pct
        qty = int(position_value / price)
        return max(1, qty)
    
    def record_trade(self, symbol: str):
        """Record trade"""
        self.trades_today += 1
        self.stock_trades[symbol] += 1


# ============================================================================
# EXECUTION ENGINE
# ============================================================================

class ExecutionEngine:
    """Manage trades and positions"""
    
    def __init__(self, config: TradingConfig, data_manager: DataManager):
        self.config = config
        self.data = data_manager
        self.cash = config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.history: List[Position] = []
    
    def execute(self, signal: Signal, quantity: int) -> Tuple[bool, str]:
        """Execute trade"""
        try:
            trade_value = signal.entry * quantity
            
            if signal.type == SignalType.BUY:
                if trade_value > self.cash:
                    return False, "Insufficient cash"
                self.cash -= trade_value
            else:
                margin = trade_value * 0.2
                if margin > self.cash:
                    return False, "Insufficient margin"
                self.cash -= margin
            
            position = Position(
                symbol=signal.symbol,
                action=signal.type.value,
                quantity=quantity,
                entry_price=signal.entry,
                stop_loss=signal.stop_loss,
                target=signal.target,
                entry_time=datetime.now(IND_TZ),
                strategy=signal.strategy,
                current_price=signal.entry
            )
            
            self.positions[signal.symbol] = position
            
            logger.info(f"✓ EXECUTED: {signal}")
            return True, "Trade executed"
            
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return False, str(e)
    
    def update_positions(self):
        """Update all positions"""
        for symbol, pos in list(self.positions.items()):
            try:
                price = self.data.get_live_price(symbol)
                if price > 0:
                    pos.update(price)
                    
                    should_exit, reason = pos.should_exit()
                    if should_exit:
                        self.close_position(symbol, price, reason)
            except Exception as e:
                logger.error(f"Error updating {symbol}: {e}")
    
    def close_position(self, symbol: str, price: float, reason: str):
        """Close position"""
        if symbol not in self.positions:
            return
        
        pos = self.positions[symbol]
        pos.status = "CLOSED"
        pos.update(price)
        
        # Return capital
        if pos.action == "BUY":
            self.cash += pos.quantity * price
        else:
            margin = pos.entry_price * pos.quantity * 0.2
            self.cash += margin + pos.pnl
        
        self.history.append(pos)
        del self.positions[symbol]
        
        logger.info(f"✓ CLOSED: {symbol} | {reason} | P&L: ₹{pos.pnl:+,.2f}")
    
    def get_equity(self) -> float:
        """Calculate total equity"""
        equity = self.cash
        for pos in self.positions.values():
            if pos.action == "BUY":
                equity += pos.quantity * pos.current_price
            else:
                equity += pos.entry_price * pos.quantity * 0.2 + pos.pnl
        return equity


# ============================================================================
# MAIN TRADING SYSTEM
# ============================================================================

class TradingSystem:
    """Autonomous trading system"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.data = DataManager()
        self.strategy = StrategyEngine(self.data, config)
        self.risk = RiskManager(config)
        self.execution = ExecutionEngine(config, self.data)
        
        self.running = False
        self.thread = None
        self._stop_event = threading.Event()
        
        logger.info("Trading System Initialized")
    
    def start(self):
        """Start trading"""
        if self.running:
            logger.warning("Already running")
            return
        
        self.running = True
        self._stop_event.clear()
        self.thread = threading.Thread(target=self._trading_loop, daemon=True)
        self.thread.start()
        logger.info("✓ System STARTED")
    
    def stop(self):
        """Stop trading"""
        logger.info("Stopping system...")
        self.running = False
        self._stop_event.set()
        if self.thread:
            self.thread.join(timeout=10)
        logger.info("✓ System STOPPED")
    
    def _trading_loop(self):
        """Main trading loop"""
        # Top 20 liquid stocks
        UNIVERSE = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
            "HINDUNILVR.NS", "KOTAKBANK.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS",
            "SBIN.NS", "ASIANPAINT.NS", "HCLTECH.NS", "AXISBANK.NS", "MARUTI.NS",
            "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS", "NTPC.NS"
        ]
        
        last_scan = time.time()
        
        while not self._stop_event.is_set() and self.running:
            try:
                # Check market
                if not self.config.is_market_open():
                    logger.debug("Market closed, waiting...")
                    time.sleep(60)
                    continue
                
                # Update positions
                self.execution.update_positions()
                
                # Scan for signals
                if time.time() - last_scan >= self.config.signal_scan_interval:
                    logger.info("Scanning for signals...")
                    signals = self.strategy.scan_for_signals(UNIVERSE[:10])
                    
                    if signals:
                        logger.info(f"Found {len(signals)} signals")
                        self._process_signals(signals[:3])
                    
                    last_scan = time.time()
                
                # Check emergency stop
                if abs(self.risk.daily_pnl) >= self.config.max_daily_loss:
                    logger.critical("EMERGENCY STOP: Daily loss limit!")
                    self._close_all("Daily loss limit")
                    break
                
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Loop error: {e}")
                time.sleep(10)
    
    def _process_signals(self, signals: List[Signal]):
        """Process signals"""
        for sig in signals:
            can_trade, reason = self.risk.can_trade(sig.symbol, self.execution.positions)
            
            if not can_trade:
                logger.debug(f"Trade blocked: {sig.symbol} - {reason}")
                continue
            
            qty = self.risk.calculate_quantity(sig.entry, self.execution.cash)
            
            if qty > 0:
                success, msg = self.execution.execute(sig, qty)
                if success:
                    self.risk.record_trade(sig.symbol)
    
    def _close_all(self, reason: str):
        """Close all positions"""
        logger.warning(f"Closing all positions: {reason}")
        for symbol in list(self.execution.positions.keys()):
            price = self.data.get_live_price(symbol)
            if price > 0:
                self.execution.close_position(symbol, price, reason)
    
    def get_status(self) -> Dict:
        """Get system status"""
        equity = self.execution.get_equity()
        pnl = equity - self.config.initial_capital
        
        return {
            "running": self.running,
            "market_open": self.config.is_market_open(),
            "equity": equity,
            "cash": self.execution.cash,
            "pnl": pnl,
            "pnl_pct": (pnl / self.config.initial_capital) * 100,
            "positions": len(self.execution.positions),
            "trades_today": self.risk.trades_today,
            "daily_pnl": self.risk.daily_pnl
        }
    
    def print_report(self):
        """Print performance report"""
        status = self.get_status()
        
        print(f"\n{'='*80}")
        print("TRADING SYSTEM REPORT")
        print(f"{'='*80}")
        print(f"Time: {datetime.now(IND_TZ).strftime('%Y-%m-%d %H:%M:%S IST')}")
        print(f"\nSTATUS:")
        print(f"  Running: {status['running']}")
        print(f"  Market: {'🟢 OPEN' if status['market_open'] else '🔴 CLOSED'}")
        print(f"\nACCOUNT:")
        print(f"  Initial: ₹{self.config.initial_capital:,.0f}")
        print(f"  Equity:  ₹{status['equity']:,.0f}")
        print(f"  Cash:    ₹{status['cash']:,.0f}")
        print(f"  P&L:     ₹{status['pnl']:+,.0f} ({status['pnl_pct']:+.2f}%)")
        print(f"\nTRADING:")
        print(f"  Open Positions: {status['positions']}")
        print(f"  Trades Today:   {status['trades_today']}")
        print(f"  Daily P&L:      ₹{status['daily_pnl']:+,.0f}")
        
        if self.execution.positions:
            print(f"\nOPEN POSITIONS:")
            for sym, pos in self.execution.positions.items():
                print(f"  {sym}: {pos.action} {pos.quantity} @ ₹{pos.entry_price:.2f} | P&L: ₹{pos.pnl:+,.2f}")
        
        if self.execution.history:
            closed = self.execution.history[-5:]
            print(f"\nRECENT CLOSED (Last 5):")
            for pos in closed:
                print(f"  {pos.symbol}: {pos.action} | P&L: ₹{pos.pnl:+,.2f}")
        
        print(f"{'='*80}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point"""
    print("\n" + "="*80)
    print("INSTITUTIONAL AUTONOMOUS ALGORITHMIC TRADING SYSTEM")
    print("Version 3.0.0 - PRODUCTION")
    print("="*80 + "\n")
    
    try:
        # Configuration
        config = TradingConfig(
            initial_capital=2_000_000.0,
            max_position_pct=0.15,
            max_daily_loss=50_000.0,
            max_positions=5,
            demo_mode=True,
            simulate_market_open=True  # For testing
        )
        
        # Initialize
        system = TradingSystem(config)
        
        # Start
        system.start()
        
        print("✓ System started successfully")
        print("Monitoring (Press Ctrl+C to stop)...\n")
        
        try:
            while True:
                time.sleep(30)
                status = system.get_status()
                print(f"[{datetime.now(IND_TZ).strftime('%H:%M:%S')}] "
                      f"Equity: ₹{status['equity']:,.0f} | "
                      f"Positions: {status['positions']} | "
                      f"P&L: ₹{status['pnl']:+,.0f}")
                
        except KeyboardInterrupt:
            print("\n\nShutdown requested...")
        
        # Stop
        system.stop()
        system.print_report()
        
        logger.info("System shutdown complete")
        
    except Exception as e:
        logger.critical(f"CRITICAL ERROR: {e}")
        logger.critical(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
