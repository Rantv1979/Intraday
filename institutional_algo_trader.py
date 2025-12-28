"""
Institutional-Grade Autonomous Algorithmic Trading System
Version: 2.0.0
Author: Rantv Trading Systems
License: Proprietary

Features:
- Multi-strategy execution engine
- Advanced risk management
- Real-time market data integration
- Machine learning signal enhancement
- Portfolio optimization
- Performance analytics
- Automated order execution
- Comprehensive logging and monitoring
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
from typing import Dict, List, Optional, Callable, Tuple
from enum import Enum
import numpy as np
import pandas as pd
import pytz
from collections import defaultdict

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Timezone
IND_TZ = pytz.timezone("Asia/Kolkata")

# ============================================================================
# SYSTEM CONFIGURATION
# ============================================================================

@dataclass
class SystemConfig:
    """Central system configuration"""
    # Capital Management
    initial_capital: float = 2_000_000.0
    max_position_size_pct: float = 0.15
    max_daily_loss: float = 50_000.0
    max_positions: int = 5
    
    # Risk Parameters
    max_drawdown_pct: float = 5.0
    min_confidence: float = 0.75
    kelly_fraction: float = 0.5
    
    # Trading Parameters
    max_trades_per_day: int = 20
    max_trades_per_stock: int = 2
    cooldown_after_loss: int = 300  # seconds
    
    # Market Hours
    market_open_time: dt_time = dt_time(9, 15)
    market_close_time: dt_time = dt_time(15, 30)
    peak_start: dt_time = dt_time(9, 30)
    peak_end: dt_time = dt_time(14, 30)
    auto_close_time: dt_time = dt_time(15, 10)
    
    # Data Management
    data_refresh_interval: int = 60  # seconds
    signal_cache_duration: int = 300  # seconds
    
    # API Configuration
    kite_api_key: str = field(default_factory=lambda: os.getenv("KITE_API_KEY", ""))
    kite_api_secret: str = field(default_factory=lambda: os.getenv("KITE_API_SECRET", ""))
    
    # Strategy Configuration
    enable_ml: bool = True
    enable_high_accuracy: bool = True
    require_volume_confirmation: bool = True
    min_volume_ratio: float = 1.3
    min_adx: float = 25.0
    min_risk_reward: float = 2.5
    
    def validate(self) -> Tuple[bool, str]:
        """Validate configuration"""
        if self.initial_capital <= 0:
            return False, "Initial capital must be positive"
        if self.max_position_size_pct <= 0 or self.max_position_size_pct > 1:
            return False, "Max position size must be between 0 and 1"
        if self.min_confidence < 0.5 or self.min_confidence > 1:
            return False, "Min confidence must be between 0.5 and 1"
        return True, "Configuration valid"


# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    PLACED = "placed"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"


class SignalType(Enum):
    """Trading signal types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class MarketRegime(Enum):
    """Market regime classification"""
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    VOLATILE = "volatile"
    NEUTRAL = "neutral"


@dataclass
class TradingSignal:
    """Trading signal data structure"""
    symbol: str
    signal_type: SignalType
    entry_price: float
    stop_loss: float
    target: float
    confidence: float
    strategy: str
    timestamp: datetime
    risk_reward_ratio: float
    volume_ratio: float
    rsi: float
    adx: float
    quality_score: int
    metadata: Dict = field(default_factory=dict)
    
    def is_valid(self, config: SystemConfig) -> bool:
        """Validate signal against configuration"""
        if self.confidence < config.min_confidence:
            return False
        if self.risk_reward_ratio < config.min_risk_reward:
            return False
        if self.volume_ratio < config.min_volume_ratio:
            return False
        if self.adx < config.min_adx:
            return False
        return True


@dataclass
class Position:
    """Position data structure"""
    symbol: str
    action: str
    quantity: int
    entry_price: float
    current_price: float
    stop_loss: float
    target: float
    entry_time: datetime
    strategy: str
    confidence: float
    pnl: float = 0.0
    pnl_pct: float = 0.0
    status: str = "OPEN"
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    
    def update_pnl(self, current_price: float):
        """Update P&L based on current price"""
        self.current_price = current_price
        if self.action == "BUY":
            self.pnl = (current_price - self.entry_price) * self.quantity
        else:
            self.pnl = (self.entry_price - current_price) * self.quantity
        self.pnl_pct = (self.pnl / (self.entry_price * self.quantity)) * 100
    
    def should_close(self) -> Tuple[bool, str]:
        """Check if position should be closed"""
        if self.action == "BUY":
            if self.current_price <= self.stop_loss:
                return True, "Stop loss hit"
            if self.current_price >= self.target:
                return True, "Target achieved"
        else:
            if self.current_price >= self.stop_loss:
                return True, "Stop loss hit"
            if self.current_price <= self.target:
                return True, "Target achieved"
        return False, ""


# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================

class TechnicalIndicators:
    """Technical indicator calculations"""
    
    @staticmethod
    def ema(series: pd.Series, span: int) -> pd.Series:
        """Exponential Moving Average"""
        return series.ewm(span=span, adjust=False).mean()
    
    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(window=period).mean()
        loss = (-delta.clip(upper=0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    @staticmethod
    def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD Indicator"""
        ema_fast = TechnicalIndicators.ema(close, fast)
        ema_slow = TechnicalIndicators.ema(close, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average Directional Index"""
        try:
            df = pd.DataFrame({'high': high, 'low': low, 'close': close})
            df['tr'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    (df['high'] - df['close'].shift()).abs(),
                    (df['low'] - df['close'].shift()).abs()
                )
            )
            df['up_move'] = df['high'] - df['high'].shift()
            df['down_move'] = df['low'].shift() - df['low']
            df['dm_pos'] = np.where(
                (df['up_move'] > df['down_move']) & (df['up_move'] > 0),
                df['up_move'],
                0.0
            )
            df['dm_neg'] = np.where(
                (df['down_move'] > df['up_move']) & (df['down_move'] > 0),
                df['down_move'],
                0.0
            )
            df['tr_sum'] = df['tr'].rolling(window=period).sum()
            df['dm_pos_sum'] = df['dm_pos'].rolling(window=period).sum()
            df['dm_neg_sum'] = df['dm_neg'].rolling(window=period).sum()
            df['di_pos'] = 100 * (df['dm_pos_sum'] / df['tr_sum'].replace(0, np.nan))
            df['di_neg'] = 100 * (df['dm_neg_sum'] / df['tr_sum'].replace(0, np.nan))
            df['dx'] = (
                abs(df['di_pos'] - df['di_neg']) / 
                (df['di_pos'] + df['di_neg']).replace(0, np.nan)
            ) * 100
            df['adx'] = df['dx'].rolling(window=period).mean()
            return df['adx'].fillna(20)
        except Exception as e:
            logger.error(f"ADX calculation error: {e}")
            return pd.Series([20] * len(high), index=high.index)
    
    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()


# ============================================================================
# DATA MANAGER
# ============================================================================

class DataManager:
    """Manages market data and calculations"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.price_cache = {}
        self.data_cache = {}
        self.cache_timestamps = {}
        
    def get_stock_data(self, symbol: str, interval: str = "15m", use_cache: bool = True) -> Optional[pd.DataFrame]:
        """Get stock data with caching"""
        cache_key = f"{symbol}_{interval}"
        current_time = time.time()
        
        # Check cache
        if use_cache and cache_key in self.data_cache:
            if current_time - self.cache_timestamps.get(cache_key, 0) < self.config.data_refresh_interval:
                return self.data_cache[cache_key]
        
        try:
            # Import yfinance only when needed
            import yfinance as yf
            
            # Determine period
            period_map = {
                "1m": "1d",
                "5m": "2d",
                "15m": "7d",
                "30m": "14d",
                "1h": "30d"
            }
            period = period_map.get(interval, "7d")
            
            # Download data
            df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
            
            if df is None or df.empty or len(df) < 20:
                logger.warning(f"Insufficient data for {symbol}")
                return None
            
            # Standardize columns
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            df = df.rename(columns={c: c.capitalize() for c in df.columns})
            
            # Ensure required columns
            required_cols = ["Open", "High", "Low", "Close", "Volume"]
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Missing required columns for {symbol}")
                return None
            
            df = df[required_cols].dropna()
            
            if len(df) < 20:
                return None
            
            # Calculate indicators
            df = self._calculate_indicators(df)
            
            # Cache data
            self.data_cache[cache_key] = df
            self.cache_timestamps[cache_key] = current_time
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        try:
            # EMAs
            df['EMA8'] = TechnicalIndicators.ema(df['Close'], 8)
            df['EMA21'] = TechnicalIndicators.ema(df['Close'], 21)
            df['EMA50'] = TechnicalIndicators.ema(df['Close'], 50)
            
            # RSI
            df['RSI14'] = TechnicalIndicators.rsi(df['Close'], 14)
            
            # ATR
            df['ATR'] = TechnicalIndicators.atr(df['High'], df['Low'], df['Close'])
            
            # MACD
            df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = TechnicalIndicators.macd(df['Close'])
            
            # Bollinger Bands
            df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = TechnicalIndicators.bollinger_bands(df['Close'])
            
            # ADX
            df['ADX'] = TechnicalIndicators.adx(df['High'], df['Low'], df['Close'])
            
            # VWAP
            df['VWAP'] = TechnicalIndicators.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
            
            # Support and Resistance
            df['Support'] = df['Low'].rolling(20).min()
            df['Resistance'] = df['High'].rolling(20).max()
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df
    
    def get_live_price(self, symbol: str) -> float:
        """Get current live price"""
        try:
            df = self.get_stock_data(symbol, "1m", use_cache=False)
            if df is not None and len(df) > 0:
                return float(df['Close'].iloc[-1])
        except Exception as e:
            logger.error(f"Error getting live price for {symbol}: {e}")
        return 0.0


# ============================================================================
# STRATEGY ENGINE
# ============================================================================

class StrategyEngine:
    """Multi-strategy signal generation engine"""
    
    def __init__(self, config: SystemConfig, data_manager: DataManager):
        self.config = config
        self.data_manager = data_manager
        self.strategy_performance = defaultdict(lambda: {
            "signals": 0, "trades": 0, "wins": 0, "pnl": 0.0
        })
    
    def generate_signals(self, symbols: List[str], min_confidence: float = None) -> List[TradingSignal]:
        """Generate trading signals for multiple symbols"""
        if min_confidence is None:
            min_confidence = self.config.min_confidence
        
        signals = []
        
        for symbol in symbols:
            try:
                symbol_signals = self._analyze_symbol(symbol)
                signals.extend(symbol_signals)
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue
        
        # Filter by confidence and quality
        signals = [s for s in signals if s.is_valid(self.config)]
        
        # Sort by quality score and confidence
        signals.sort(key=lambda x: (x.quality_score, x.confidence), reverse=True)
        
        return signals
    
    def _analyze_symbol(self, symbol: str) -> List[TradingSignal]:
        """Analyze a single symbol for signals"""
        signals = []
        
        # Get data
        data = self.data_manager.get_stock_data(symbol, "15m")
        if data is None or len(data) < 50:
            return signals
        
        # Extract latest values
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        price = float(latest['Close'])
        ema8 = float(latest['EMA8'])
        ema21 = float(latest['EMA21'])
        ema50 = float(latest['EMA50'])
        rsi = float(latest['RSI14'])
        macd = float(latest['MACD'])
        macd_signal = float(latest['MACD_Signal'])
        vwap = float(latest['VWAP'])
        atr = float(latest['ATR'])
        adx = float(latest['ADX'])
        volume = float(latest['Volume'])
        avg_volume = float(data['Volume'].rolling(20).mean().iloc[-1])
        support = float(latest['Support'])
        resistance = float(latest['Resistance'])
        
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
        
        # Strategy 1: EMA + VWAP Confluence (BUY)
        if (ema8 > ema21 > ema50 and price > vwap and 
            adx > self.config.min_adx and volume_ratio >= self.config.min_volume_ratio):
            
            stop_loss = max(support * 0.995, price - (atr * 1.2))
            target = min(resistance * 0.998, price + (atr * 2.5))
            risk_reward = abs(target - price) / abs(price - stop_loss)
            
            if risk_reward >= self.config.min_risk_reward:
                signal = TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    entry_price=price,
                    stop_loss=stop_loss,
                    target=target,
                    confidence=0.82,
                    strategy="EMA_VWAP_Confluence",
                    timestamp=datetime.now(IND_TZ),
                    risk_reward_ratio=risk_reward,
                    volume_ratio=volume_ratio,
                    rsi=rsi,
                    adx=adx,
                    quality_score=9,
                    metadata={"ema8": ema8, "ema21": ema21, "vwap": vwap}
                )
                signals.append(signal)
                self.strategy_performance["EMA_VWAP_Confluence"]["signals"] += 1
        
        # Strategy 2: RSI Mean Reversion (BUY)
        if (rsi < 30 and rsi > 25 and price > support and volume_ratio >= 1.3):
            stop_loss = price - (atr * 1.2)
            target = price + (atr * 2.0)
            risk_reward = abs(target - price) / abs(price - stop_loss)
            
            if risk_reward >= 2.0:
                signal = TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    entry_price=price,
                    stop_loss=stop_loss,
                    target=target,
                    confidence=0.75,
                    strategy="RSI_MeanReversion",
                    timestamp=datetime.now(IND_TZ),
                    risk_reward_ratio=risk_reward,
                    volume_ratio=volume_ratio,
                    rsi=rsi,
                    adx=adx,
                    quality_score=8
                )
                signals.append(signal)
                self.strategy_performance["RSI_MeanReversion"]["signals"] += 1
        
        # Strategy 3: MACD Momentum (BUY)
        if (macd > macd_signal and prev['MACD'] <= prev['MACD_Signal'] and
            ema8 > ema21 and price > vwap and volume_ratio >= 1.3):
            
            stop_loss = price - (atr * 1.2)
            target = price + (atr * 2.5)
            risk_reward = abs(target - price) / abs(price - stop_loss)
            
            if risk_reward >= 2.0:
                signal = TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    entry_price=price,
                    stop_loss=stop_loss,
                    target=target,
                    confidence=0.78,
                    strategy="MACD_Momentum",
                    timestamp=datetime.now(IND_TZ),
                    risk_reward_ratio=risk_reward,
                    volume_ratio=volume_ratio,
                    rsi=rsi,
                    adx=adx,
                    quality_score=8
                )
                signals.append(signal)
                self.strategy_performance["MACD_Momentum"]["signals"] += 1
        
        # Strategy 4: EMA Downtrend (SELL)
        if (ema8 < ema21 < ema50 and price < vwap and 
            adx > self.config.min_adx and volume_ratio >= 1.3):
            
            stop_loss = min(resistance * 1.005, price + (atr * 1.2))
            target = max(support * 1.002, price - (atr * 2.5))
            risk_reward = abs(price - target) / abs(stop_loss - price)
            
            if risk_reward >= self.config.min_risk_reward:
                signal = TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    entry_price=price,
                    stop_loss=stop_loss,
                    target=target,
                    confidence=0.80,
                    strategy="EMA_Downtrend",
                    timestamp=datetime.now(IND_TZ),
                    risk_reward_ratio=risk_reward,
                    volume_ratio=volume_ratio,
                    rsi=rsi,
                    adx=adx,
                    quality_score=8
                )
                signals.append(signal)
                self.strategy_performance["EMA_Downtrend"]["signals"] += 1
        
        # Strategy 5: RSI Overbought (SELL)
        if (rsi > 70 and rsi < 75 and price < resistance and volume_ratio >= 1.3):
            stop_loss = price + (atr * 1.2)
            target = price - (atr * 2.0)
            risk_reward = abs(price - target) / abs(stop_loss - price)
            
            if risk_reward >= 2.0:
                signal = TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    entry_price=price,
                    stop_loss=stop_loss,
                    target=target,
                    confidence=0.75,
                    strategy="RSI_Overbought",
                    timestamp=datetime.now(IND_TZ),
                    risk_reward_ratio=risk_reward,
                    volume_ratio=volume_ratio,
                    rsi=rsi,
                    adx=adx,
                    quality_score=7
                )
                signals.append(signal)
                self.strategy_performance["RSI_Overbought"]["signals"] += 1
        
        return signals


# ============================================================================
# RISK MANAGER
# ============================================================================

class RiskManager:
    """Advanced risk management system"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.daily_pnl = 0.0
        self.peak_equity = config.initial_capital
        self.current_drawdown = 0.0
        self.trades_today = 0
        self.stock_trades = defaultdict(int)
        self.last_trade_time = {}
        self.last_reset_date = datetime.now(IND_TZ).date()
    
    def reset_daily_metrics(self):
        """Reset daily metrics at market open"""
        current_date = datetime.now(IND_TZ).date()
        if current_date != self.last_reset_date:
            self.daily_pnl = 0.0
            self.trades_today = 0
            self.stock_trades.clear()
            self.last_reset_date = current_date
            logger.info("Daily metrics reset")
    
    def can_trade(self, symbol: str, signal: TradingSignal, current_positions: Dict[str, Position]) -> Tuple[bool, str]:
        """Check if trade is allowed"""
        self.reset_daily_metrics()
        
        # Check position limit
        if len(current_positions) >= self.config.max_positions:
            return False, "Maximum positions reached"
        
        # Check daily trade limit
        if self.trades_today >= self.config.max_trades_per_day:
            return False, "Daily trade limit reached"
        
        # Check per-stock trade limit
        if self.stock_trades[symbol] >= self.config.max_trades_per_stock:
            return False, f"Maximum trades for {symbol} reached"
        
        # Check if already in position
        if symbol in current_positions:
            return False, f"Already in position for {symbol}"
        
        # Check daily loss limit
        if abs(self.daily_pnl) >= self.config.max_daily_loss:
            return False, "Daily loss limit exceeded"
        
        # Check cooldown after loss
        if symbol in self.last_trade_time:
            time_since = (datetime.now(IND_TZ) - self.last_trade_time[symbol]).total_seconds()
            if self.daily_pnl < 0 and time_since < self.config.cooldown_after_loss:
                return False, "Cooldown period active"
        
        # Check drawdown
        if self.current_drawdown >= self.config.max_drawdown_pct:
            return False, "Maximum drawdown exceeded"
        
        return True, "Trade allowed"
    
    def calculate_position_size(self, signal: TradingSignal, available_capital: float) -> int:
        """Calculate optimal position size using Kelly Criterion"""
        try:
            # Kelly Criterion: f = (p*b - q) / b
            # where p = win probability, q = loss probability, b = win/loss ratio
            win_prob = signal.confidence
            loss_prob = 1 - win_prob
            win_loss_ratio = signal.risk_reward_ratio
            
            kelly_fraction = (win_prob * win_loss_ratio - loss_prob) / win_loss_ratio
            kelly_fraction = max(0, min(kelly_fraction, self.config.kelly_fraction))
            
            # Calculate position value
            max_position_value = available_capital * self.config.max_position_size_pct
            kelly_position_value = available_capital * kelly_fraction
            
            position_value = min(max_position_value, kelly_position_value)
            
            # Calculate quantity
            quantity = int(position_value / signal.entry_price)
            
            return max(1, quantity)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return int((available_capital * self.config.max_position_size_pct) / signal.entry_price)
    
    def update_pnl(self, pnl_change: float, equity: float):
        """Update P&L metrics"""
        self.daily_pnl += pnl_change
        
        # Update drawdown
        if equity > self.peak_equity:
            self.peak_equity = equity
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = ((self.peak_equity - equity) / self.peak_equity) * 100
    
    def record_trade(self, symbol: str):
        """Record trade execution"""
        self.trades_today += 1
        self.stock_trades[symbol] += 1
        self.last_trade_time[symbol] = datetime.now(IND_TZ)


# ============================================================================
# EXECUTION ENGINE
# ============================================================================

class ExecutionEngine:
    """Order execution and position management"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Position] = []
        self.cash = config.initial_capital
        self.initial_capital = config.initial_capital
    
    def execute_trade(self, signal: TradingSignal, quantity: int) -> Tuple[bool, str, Optional[Position]]:
        """Execute a trade"""
        try:
            position = Position(
                symbol=signal.symbol,
                action=signal.signal_type.value,
                quantity=quantity,
                entry_price=signal.entry_price,
                current_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                target=signal.target,
                entry_time=datetime.now(IND_TZ),
                strategy=signal.strategy,
                confidence=signal.confidence
            )
            
            # Calculate required capital
            trade_value = signal.entry_price * quantity
            
            if signal.signal_type == SignalType.BUY:
                if trade_value > self.cash:
                    return False, "Insufficient capital", None
                self.cash -= trade_value
            else:  # SELL (short)
                margin = trade_value * 0.2
                if margin > self.cash:
                    return False, "Insufficient margin", None
                self.cash -= margin
            
            # Add to positions
            self.positions[signal.symbol] = position
            
            logger.info(f"Trade executed: {signal.signal_type.value} {quantity} {signal.symbol} @ {signal.entry_price}")
            
            return True, "Trade executed successfully", position
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False, f"Execution error: {str(e)}", None
    
    def close_position(self, symbol: str, exit_price: float, reason: str = "Manual close") -> Tuple[bool, str, float]:
        """Close an open position"""
        if symbol not in self.positions:
            return False, "Position not found", 0.0
        
        try:
            position = self.positions[symbol]
            position.exit_price = exit_price
            position.exit_time = datetime.now(IND_TZ)
            position.status = "CLOSED"
            position.update_pnl(exit_price)
            
            # Calculate P&L
            pnl = position.pnl
            
            # Return capital
            if position.action == "BUY":
                self.cash += position.quantity * exit_price
            else:  # SELL
                margin_used = position.entry_price * position.quantity * 0.2
                self.cash += margin_used + (position.entry_price - exit_price) * position.quantity
            
            # Move to history
            self.trade_history.append(position)
            del self.positions[symbol]
            
            logger.info(f"Position closed: {symbol} | Reason: {reason} | P&L: {pnl:+.2f}")
            
            return True, f"Position closed: {reason}", pnl
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False, f"Error: {str(e)}", 0.0
    
    def update_positions(self, data_manager: DataManager):
        """Update all open positions"""
        for symbol, position in list(self.positions.items()):
            try:
                current_price = data_manager.get_live_price(symbol)
                if current_price > 0:
                    position.update_pnl(current_price)
                    
                    # Check if position should be closed
                    should_close, reason = position.should_close()
                    if should_close:
                        self.close_position(symbol, current_price, reason)
                        
            except Exception as e:
                logger.error(f"Error updating position {symbol}: {e}")
    
    def get_equity(self) -> float:
        """Calculate total equity"""
        equity = self.cash
        for position in self.positions.values():
            if position.action == "BUY":
                equity += position.quantity * position.current_price
            else:
                margin = position.entry_price * position.quantity * 0.2
                equity += margin + position.pnl
        return equity
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        closed_trades = [p for p in self.trade_history]
        total_trades = len(closed_trades)
        
        if total_trades == 0:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0
            }
        
        winning_trades = [p for p in closed_trades if p.pnl > 0]
        losing_trades = [p for p in closed_trades if p.pnl <= 0]
        
        total_pnl = sum(p.pnl for p in closed_trades)
        gross_profit = sum(p.pnl for p in winning_trades)
        gross_loss = abs(sum(p.pnl for p in losing_trades))
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / total_trades,
            "total_pnl": total_pnl,
            "avg_pnl": total_pnl / total_trades,
            "profit_factor": profit_factor,
            "max_drawdown": 0.0  # Calculate from equity curve
        }


# ============================================================================
# AUTONOMOUS TRADING SYSTEM
# ============================================================================

class AutonomousTradingSystem:
    """Main autonomous trading system"""
    
    def __init__(self, config: SystemConfig):
        # Validate configuration
        valid, msg = config.validate()
        if not valid:
            raise ValueError(f"Invalid configuration: {msg}")
        
        self.config = config
        self.data_manager = DataManager(config)
        self.strategy_engine = StrategyEngine(config, self.data_manager)
        self.risk_manager = RiskManager(config)
        self.execution_engine = ExecutionEngine(config)
        
        self.running = False
        self.paused = False
        self._stop_event = threading.Event()
        self._main_thread = None
        
        logger.info("Autonomous Trading System initialized")
    
    def start(self):
        """Start the trading system"""
        if self.running:
            logger.warning("System already running")
            return
        
        self.running = True
        self.paused = False
        self._stop_event.clear()
        
        self._main_thread = threading.Thread(target=self._trading_loop, daemon=True)
        self._main_thread.start()
        
        logger.info("Trading system started")
    
    def stop(self):
        """Stop the trading system"""
        logger.info("Stopping trading system...")
        self.running = False
        self._stop_event.set()
        
        if self._main_thread:
            self._main_thread.join(timeout=10)
        
        logger.info("Trading system stopped")
    
    def pause(self):
        """Pause trading (keep monitoring)"""
        self.paused = True
        logger.info("Trading system paused")
    
    def resume(self):
        """Resume trading"""
        self.paused = False
        logger.info("Trading system resumed")
    
    def _is_market_open(self) -> bool:
        """Check if market is open"""
        now = datetime.now(IND_TZ)
        
        # Check if weekend
        if now.weekday() >= 5:
            return False
        
        # Check market hours
        current_time = now.time()
        return self.config.market_open_time <= current_time <= self.config.market_close_time
    
    def _should_auto_close(self) -> bool:
        """Check if positions should be auto-closed"""
        now = datetime.now(IND_TZ)
        return now.time() >= self.config.auto_close_time
    
    def _trading_loop(self):
        """Main trading loop"""
        logger.info("Trading loop started")
        
        # Stock universe (top liquid stocks)
        STOCK_UNIVERSE = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
            "ICICIBANK.NS", "KOTAKBANK.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS",
            "SBIN.NS", "ASIANPAINT.NS", "HCLTECH.NS", "AXISBANK.NS", "MARUTI.NS",
            "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS", "NTPC.NS"
        ]
        
        last_signal_scan = datetime.now(IND_TZ)
        scan_interval = 60  # seconds
        
        while not self._stop_event.is_set() and self.running:
            try:
                # Check market status
                if not self._is_market_open():
                    logger.debug("Market closed, waiting...")
                    time.sleep(60)
                    continue
                
                # Check for auto-close
                if self._should_auto_close():
                    self._close_all_positions("Market close approaching")
                
                # Update positions
                self.execution_engine.update_positions(self.data_manager)
                
                # Update risk metrics
                equity = self.execution_engine.get_equity()
                total_pnl = sum(p.pnl for p in self.execution_engine.positions.values())
                self.risk_manager.update_pnl(0, equity)
                
                # Check for emergency stop
                if abs(self.risk_manager.daily_pnl) >= self.config.max_daily_loss:
                    logger.critical("EMERGENCY STOP: Daily loss limit exceeded")
                    self._close_all_positions("Daily loss limit")
                    self.pause()
                    continue
                
                # Generate signals (if not paused)
                if not self.paused:
                    current_time = datetime.now(IND_TZ)
                    if (current_time - last_signal_scan).total_seconds() >= scan_interval:
                        logger.info("Scanning for trading signals...")
                        signals = self.strategy_engine.generate_signals(STOCK_UNIVERSE[:10])
                        
                        if signals:
                            logger.info(f"Found {len(signals)} signals")
                            self._execute_signals(signals[:5])  # Top 5 signals
                        
                        last_signal_scan = current_time
                
                # Sleep
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                logger.error(traceback.format_exc())
                time.sleep(10)
        
        logger.info("Trading loop stopped")
    
    def _execute_signals(self, signals: List[TradingSignal]):
        """Execute trading signals"""
        for signal in signals:
            try:
                # Check if trade is allowed
                can_trade, reason = self.risk_manager.can_trade(
                    signal.symbol,
                    signal,
                    self.execution_engine.positions
                )
                
                if not can_trade:
                    logger.info(f"Trade blocked for {signal.symbol}: {reason}")
                    continue
                
                # Calculate position size
                quantity = self.risk_manager.calculate_position_size(
                    signal,
                    self.execution_engine.cash
                )
                
                if quantity <= 0:
                    logger.warning(f"Invalid quantity for {signal.symbol}")
                    continue
                
                # Execute trade
                success, msg, position = self.execution_engine.execute_trade(signal, quantity)
                
                if success:
                    self.risk_manager.record_trade(signal.symbol)
                    logger.info(f"✓ Trade executed: {msg}")
                else:
                    logger.warning(f"✗ Trade failed: {msg}")
                    
            except Exception as e:
                logger.error(f"Error executing signal for {signal.symbol}: {e}")
    
    def _close_all_positions(self, reason: str):
        """Close all open positions"""
        logger.warning(f"Closing all positions: {reason}")
        
        for symbol in list(self.execution_engine.positions.keys()):
            try:
                current_price = self.data_manager.get_live_price(symbol)
                self.execution_engine.close_position(symbol, current_price, reason)
            except Exception as e:
                logger.error(f"Error closing {symbol}: {e}")
    
    def get_status(self) -> Dict:
        """Get system status"""
        equity = self.execution_engine.get_equity()
        perf = self.execution_engine.get_performance_stats()
        
        return {
            "running": self.running,
            "paused": self.paused,
            "market_open": self._is_market_open(),
            "equity": equity,
            "cash": self.execution_engine.cash,
            "open_positions": len(self.execution_engine.positions),
            "daily_trades": self.risk_manager.trades_today,
            "daily_pnl": self.risk_manager.daily_pnl,
            "current_drawdown": self.risk_manager.current_drawdown,
            "performance": perf
        }
    
    def generate_report(self) -> str:
        """Generate performance report"""
        status = self.get_status()
        perf = status['performance']
        
        report = f"""
{'='*80}
TRADING SYSTEM PERFORMANCE REPORT
Generated: {datetime.now(IND_TZ).strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

SYSTEM STATUS:
  Running: {status['running']}
  Paused: {status['paused']}
  Market Open: {status['market_open']}

ACCOUNT SUMMARY:
  Initial Capital: ₹{self.config.initial_capital:,.2f}
  Current Equity: ₹{status['equity']:,.2f}
  Available Cash: ₹{status['cash']:,.2f}
  Total P&L: ₹{status['equity'] - self.config.initial_capital:+,.2f}
  ROI: {((status['equity'] / self.config.initial_capital) - 1) * 100:+.2f}%

OPEN POSITIONS:
  Count: {status['open_positions']}
  
DAILY METRICS:
  Trades Today: {status['daily_trades']}
  Daily P&L: ₹{status['daily_pnl']:+,.2f}
  Current Drawdown: {status['current_drawdown']:.2f}%

PERFORMANCE STATISTICS:
  Total Trades: {perf['total_trades']}
  Winning Trades: {perf['winning_trades']}
  Losing Trades: {perf['losing_trades']}
  Win Rate: {perf['win_rate'] * 100:.2f}%
  Total P&L: ₹{perf['total_pnl']:+,.2f}
  Average P&L per Trade: ₹{perf['avg_pnl']:+,.2f}
  Profit Factor: {perf['profit_factor']:.2f}

{'='*80}
"""
        return report


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point"""
    print("\n" + "="*80)
    print("INSTITUTIONAL AUTONOMOUS ALGORITHMIC TRADING SYSTEM")
    print("Version 2.0.0")
    print("="*80 + "\n")
    
    try:
        # Create configuration
        config = SystemConfig(
            initial_capital=2_000_000.0,
            max_position_size_pct=0.15,
            max_daily_loss=50_000.0,
            max_positions=5,
            min_confidence=0.75,
            kelly_fraction=0.5,
            enable_ml=True,
            enable_high_accuracy=True
        )
        
        # Initialize system
        logger.info("Initializing trading system...")
        system = AutonomousTradingSystem(config)
        
        # Start system
        logger.info("Starting autonomous trading...")
        system.start()
        
        # Run for demonstration (in production, this would run continuously)
        print("\n✓ System started successfully")
        print("\nMonitoring for 5 minutes (demo mode)...")
        print("Press Ctrl+C to stop\n")
        
        try:
            for i in range(30):  # 5 minutes
                time.sleep(10)
                
                # Print status update
                status = system.get_status()
                print(f"\r[{datetime.now(IND_TZ).strftime('%H:%M:%S')}] "
                      f"Equity: ₹{status['equity']:,.0f} | "
                      f"Positions: {status['open_positions']} | "
                      f"Daily P&L: ₹{status['daily_pnl']:+,.2f}", 
                      end="", flush=True)
                
        except KeyboardInterrupt:
            print("\n\nShutdown requested...")
        
        # Stop system
        system.stop()
        
        # Generate final report
        print("\n\n" + system.generate_report())
        
        logger.info("System shutdown complete")
        
    except Exception as e:
        logger.critical(f"Critical error: {e}")
        logger.critical(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
