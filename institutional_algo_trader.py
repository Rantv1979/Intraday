# ============================================================
# INSTITUTIONAL AI + SMC PRO ALGORITHMIC TRADING SYSTEM
# AI + Smart Money Concepts + Kite Connect + Dashboard
# ============================================================

import os, sys, time, warnings
from datetime import datetime, timedelta, time as dt_time
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# OPTIONAL KITE CONNECT
# ------------------------------------------------------------
try:
    from kiteconnect import KiteConnect
    KITE_AVAILABLE = True
except:
    KITE_AVAILABLE = False


# ============================================================
# CONFIGURATION
# ============================================================

class MarketPhase(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


@dataclass
class TradingConfig:
    demo_mode: bool = True
    paper_trading: bool = True
    total_capital: float = 2_000_000
    risk_per_trade: float = 0.01
    max_positions: int = 10
    confidence_threshold: float = 0.60
    atr_mult_sl: float = 1.5
    atr_mult_target: float = 3.0
    trailing_mult: float = 1.2
    lookback: int = 50
    prediction_horizon: int = 3


# ============================================================
# STOCK UNIVERSE (UNCHANGED – SAFE)
# ============================================================

class StockUniverse:
    """Complete Indian Equity + Index Universe"""

    # -------------------------
    # NIFTY 50
    # -------------------------
    nifty_50 = [
        "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR",
        "ITC","SBIN","BHARTIARTL","KOTAKBANK","BAJFINANCE","LT",
        "AXISBANK","ASIANPAINT","MARUTI","SUNPHARMA","TITAN","WIPRO",
        "ULTRACEMCO","POWERGRID","NTPC","ONGC","TECHM","JSWSTEEL",
        "HCLTECH","ADANIPORTS","TATASTEEL","M&M","GRASIM","INDUSINDBK",
        "BRITANNIA","BAJAJFINSV","DIVISLAB","DRREDDY","CIPLA",
        "SHREECEM","BPCL","EICHERMOT","HEROMOTOCO","COALINDIA",
        "IOC","SBILIFE","UPL","TATAMOTORS","BAJAJ-AUTO",
        "NESTLEIND","HDFCLIFE","HINDALCO","TATACONSUM"
    ]

    # -------------------------
    # NIFTY MIDCAP (Top tradable)
    # -------------------------
    nifty_midcap = [
        "PAGEIND","BERGEPAINT","DABUR","GODREJCP","HAVELLS","ICICIPRULI",
        "LTI","MARICO","PIDILITIND","SRF","ABFRL","AJANTPHARM",
        "APOLLOHOSP","ASHOKLEY","AUROPHARMA","BAJAJHLDNG","BALKRISIND",
        "BANDHANBNK","BATAINDIA","BHARATFORG","BIOCON","BOSCHLTD",
        "CADILAHC","CHOLAFIN","COLPAL","CONCOR","DALBHARAT",
        "ESCORTS","EXIDEIND","FEDERALBNK","GLENMARK","GODREJPROP",
        "HINDPETRO","IBULHSGFIN","IDEA","INDHOTEL","IRCTC",
        "JINDALSTEL","JUBLFOOD","LALPATHLAB","LICHSGFIN","LUPIN",
        "MANAPPURAM","MFSL","MINDTREE","MOTHERSUMI","MRF"
    ]

    # -------------------------
    # NIFTY 100 ADDITIONAL
    # -------------------------
    nifty_100_additional = [
        "ADANIENT","ADANIGREEN","ADANITRANS","AMBUJACEM","BANKBARODA",
        "DLF","GAIL","GLAND","ICICIGI","IGL","INDUSTOWER",
        "NMDC","PETRONET","PFC","RECLTD","SAIL","SIEMENS",
        "TORNTPHARM","TVSMOTOR","UBL"
    ]

    # -------------------------
    # INDEX SYMBOLS (MOOD GAUGE)
    # -------------------------
    indices = {
        "NIFTY": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
        "SENSEX": "^BSESN"
    }

    @classmethod
    def get_all_equities(cls):
        return sorted(set(
            cls.nifty_50 +
            cls.nifty_midcap +
            cls.nifty_100_additional
        ))

    @classmethod
    def get_trading_universe(cls, limit=100):
        """Institutional performance-safe universe"""
        return cls.get_all_equities()[:limit]

    @classmethod
    def get_indices(cls):
        return cls.indices



# ============================================================
# DATA MANAGER
# ============================================================

class DataManager:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.kite = None

        if not config.demo_mode and KITE_AVAILABLE:
            try:
                self.kite = KiteConnect(api_key=st.secrets["KITE_API_KEY"])
                self.kite.set_access_token(st.secrets["KITE_ACCESS_TOKEN"])
            except:
                self.kite = None
                self.config.demo_mode = True

    def fetch_data(self, symbol, days=180, interval="15min"):
        if self.config.demo_mode or not self.kite:
            return self.demo_data(symbol, days, interval)

        try:
            to_d = datetime.now()
            from_d = to_d - timedelta(days=days)
            token = [i for i in self.kite.instruments("NSE") if i["tradingsymbol"] == symbol][0]["instrument_token"]
            data = self.kite.historical_data(token, from_d, to_d, interval)
            df = pd.DataFrame(data)
            df.set_index("date", inplace=True)
            df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"}, inplace=True)
            return df
        except:
            return self.demo_data(symbol, days, interval)

    def demo_data(self, symbol, days, interval):
        np.random.seed(abs(hash(symbol)) % 10_000)
        idx = pd.date_range(end=datetime.now(), periods=days*26, freq="15min")
        price = np.cumsum(np.random.normal(0, 2, len(idx))) + 1000
        df = pd.DataFrame(index=idx)
        df["Close"] = price
        df["Open"] = df["Close"].shift(1)
        df["High"] = df[["Open","Close"]].max(axis=1) * (1 + np.random.rand(len(df))*0.002)
        df["Low"] = df[["Open","Close"]].min(axis=1) * (1 - np.random.rand(len(df))*0.002)
        df["Volume"] = np.random.randint(1e5, 1e6, len(df))
        return df.ffill().dropna()

    def ltp(self, symbol):
        if not self.config.demo_mode and self.kite:
            return self.kite.ltp(f"NSE:{symbol}")[f"NSE:{symbol}"]["last_price"]
        return self.fetch_data(symbol).iloc[-1]["Close"]


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

class TI:
    @staticmethod
    def sma(s, n): return s.rolling(n).mean()
    @staticmethod
    def ema(s, n): return s.ewm(span=n).mean()

    @staticmethod
    def rsi(s, n=14):
        d = s.diff()
        g = d.clip(lower=0).rolling(n).mean()
        l = -d.clip(upper=0).rolling(n).mean()
        rs = g / l
        return 100 - (100 / (1 + rs))

    @staticmethod
    def atr(h, l, c, n=14):
        tr = pd.concat([(h-l), abs(h-c.shift()), abs(l-c.shift())], axis=1).max(axis=1)
        return tr.rolling(n).mean()

    @staticmethod
    def adx(h, l, c, n=14):
        return TI.atr(h,l,c,n) / c * 100


# ============================================================
# SMC PRO (SMART MONEY CONCEPTS)
# ============================================================

class SMCPro:
    @staticmethod
    def structure(df, lb=20):
        hi = df["High"].rolling(lb).max()
        lo = df["Low"].rolling(lb).min()
        s = pd.Series("RANGE", index=df.index)
        s[df["Close"] > hi.shift()] = "BULLISH"
        s[df["Close"] < lo.shift()] = "BEARISH"
        return s

    @staticmethod
    def bos(df):
        return (df["Close"] > df["High"].shift()) | (df["Close"] < df["Low"].shift())


# ============================================================
# FEATURE ENGINEERING
# ============================================================

class FeatureEngineer:
    @staticmethod
    def build(df):
        X = pd.DataFrame(index=df.index)
        X["ret"] = df["Close"].pct_change()
        X["rsi"] = TI.rsi(df["Close"])
        X["ema20"] = TI.ema(df["Close"],20)
        X["ema50"] = TI.ema(df["Close"],50)
        X["atr"] = TI.atr(df["High"],df["Low"],df["Close"])
        X["adx"] = TI.adx(df["High"],df["Low"],df["Close"])
        return X.dropna()


# ============================================================
# AI MODEL (ENSEMBLE)
# ============================================================

class AIModel:
    def __init__(self, config):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=200, max_depth=10)
        self.trained = False
        self.config = config

    def train(self, dfs):
        frames = []
        for df in dfs:
            f = FeatureEngineer.build(df)
            future = df["Close"].shift(-self.config.prediction_horizon)
            y = np.where(future.loc[f.index] > df["Close"].loc[f.index], 1, -1)
            frames.append((f, y))

        X = np.vstack([f.values for f,_ in frames])
        y = np.concatenate([y for _,y in frames])

        X = self.scaler.fit_transform(X)
        self.model.fit(X, y)
        self.trained = True

    def predict(self, df):
        f = FeatureEngineer.build(df)
        X = self.scaler.transform(f.iloc[-1:].values)
        p = self.model.predict(X)[0]
        conf = np.max(self.model.predict_proba(X))
        return p, conf


# ============================================================
# POSITION OBJECT
# ============================================================

@dataclass
class Position:
    symbol: str
    side: str
    qty: int
    entry: float
    sl: float
    target: float
    atr: float
    entry_time: datetime
    pnl: float = 0.0


# ============================================================
# TRADING ENGINE
# ============================================================

class TradingEngine:
    def __init__(self, config):
        self.config = config
        self.dm = DataManager(config)
        self.ai = AIModel(config)
        self.positions = {}
        self.trade_log = []

    def train(self):
        dfs = [self.dm.fetch_data(s) for s in StockUniverse.get_universe()[:10]]
        self.ai.train(dfs)

    def scan(self):
        for s in StockUniverse.get_universe():
            if s in self.positions: continue

            df = self.dm.fetch_data(s)
            p, conf = self.ai.predict(df)
            if conf < self.config.confidence_threshold: continue

            structure = SMCPro.structure(df).iloc[-1]
            if p == 1 and structure != "BULLISH": continue
            if p == -1 and structure != "BEARISH": continue

            price = self.dm.ltp(s)
            atr = TI.atr(df["High"],df["Low"],df["Close"]).iloc[-1]

            sl = price - atr*self.config.atr_mult_sl if p==1 else price + atr*self.config.atr_mult_sl
            tgt = price + atr*self.config.atr_mult_target if p==1 else price - atr*self.config.atr_mult_target
            qty = int((self.config.total_capital*self.config.risk_per_trade)/(abs(price-sl)))

            self.trade_log.append({
                "time":datetime.now(),"symbol":s,"event":"SIGNAL",
                "side":"BUY" if p==1 else "SELL","price":price,"conf":conf
            })

            self.positions[s] = Position(
                s,"LONG" if p==1 else "SHORT",qty,price,sl,tgt,atr,datetime.now()
            )

    def manage(self):
        for s,p in list(self.positions.items()):
            ltp = self.dm.ltp(s)
            pnl = (ltp-p.entry)*p.qty if p.side=="LONG" else (p.entry-ltp)*p.qty
            p.pnl = pnl

            if p.side=="LONG":
                p.sl = max(p.sl, ltp - p.atr*self.config.trailing_mult)
                exit_cond = ltp <= p.sl or ltp >= p.target
            else:
                p.sl = min(p.sl, ltp + p.atr*self.config.trailing_mult)
                exit_cond = ltp >= p.sl or ltp <= p.target

            if exit_cond:
                self.trade_log.append({
                    "time":datetime.now(),"symbol":s,"event":"EXIT",
                    "pnl":p.pnl
                })
                del self.positions[s]


# ============================================================
# MARKET MOOD
# ============================================================

def market_mood(df):
    rsi = TI.rsi(df["Close"]).iloc[-1]
    adx = TI.adx(df["High"],df["Low"],df["Close"]).iloc[-1]
    if rsi>60 and adx>25: return "BULLISH 🟢"
    if rsi<40 and adx>25: return "BEARISH 🔴"
    return "SIDEWAYS 🟡"


# ============================================================
# STREAMLIT DASHBOARD
# ============================================================

def main():
    st.set_page_config("Institutional AI Algo",layout="wide")

    st.markdown("""
    <style>
    body { background:#FFF3E0; color:#263238; }
    </style>
    """,unsafe_allow_html=True)

    if "engine" not in st.session_state:
        cfg = TradingConfig()
        st.session_state.engine = TradingEngine(cfg)
        st.session_state.engine.train()

    eng = st.session_state.engine

    st.title("🏦 Institutional AI + SMC Pro Algo Trading System")

    col1,col2,col3 = st.columns(3)
    with col1: st.metric("Active Positions",len(eng.positions))
    with col2: st.metric("Trades Logged",len(eng.trade_log))
    with col3: st.metric("Capital",f"₹{eng.config.total_capital:,.0f}")

    if st.button("🔍 Scan Market"):
        eng.scan()

    eng.manage()

    st.subheader("📜 Trade History")
    st.dataframe(pd.DataFrame(eng.trade_log),use_container_width=True)


if __name__ == "__main__":
    main()
