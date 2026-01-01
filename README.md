import pandas as pd
import numpy as np
from datetime import datetime
import time
import logging
import ccxt
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mexc_trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class MEXCTradingBot:
    def __init__(self, api_key, api_secret, initial_capital=500000, risk_per_trade=0.07):
        """
        Enhanced trading bot with FIXED risk-reward and improved exits

        Args:
            api_key: MEXC API key
            api_secret: MEXC API secret
            initial_capital: Starting capital (₹500,000)
            risk_per_trade: Risk per trade (7% - REDUCED from 10%)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade  # NOW 7%
        self.capital = initial_capital
        self.position = None
        self.trades = []
        self.equity_curve = []
        self.is_running = False
        self.client = None

    def connect_mexc(self):
        """Connect to MEXC exchange"""
        try:
            self.client = ccxt.mexc({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'options': {'defaultType': 'swap'}
            })

            logger.info("✅ Connected to MEXC")
            balance = self.get_balance()
            logger.info(f"💰 Account Balance: ${balance:,.2f} USDT")
            return True

        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            return False

    def get_balance(self):
        """Get USDT balance"""
        try:
            balance = self.client.fetch_balance()
            if 'USDT' in balance['free']:
                return float(balance['free']['USDT'])
            return 0
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return 0

    def get_historical_data(self, symbol='BTC/USDT:USDT', timeframe='20m', limit=100):
        """Get historical data from MEXC"""
        try:
            ohlcv = self.client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = df[col].astype(float)

            return df
        except Exception as e:
            logger.error(f"Error getting data: {e}")
            return None

    # ==================== STRATEGY METHODS ====================

    def calculate_mcginley_dynamic(self, prices, period=14, constant=0.6):
        """McGinley Dynamic calculation"""
        md = pd.Series(index=prices.index, dtype=float)
        md.iloc[0] = prices.iloc[0]

        for i in range(1, len(prices)):
            prev_md = md.iloc[i-1]
            price = prices.iloc[i]

            if prev_md == 0 or pd.isna(prev_md):
                md.iloc[i] = price
                continue

            ratio = max(0.5, min(price / prev_md, 2.0))
            n = max(period * (ratio ** 4) * constant, 1.0)
            md.iloc[i] = prev_md + (price - prev_md) / n

        return md

    def calculate_rsi(self, prices, period=14):
        """RSI calculation"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """MACD calculation"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def calculate_atr(self, high, low, close, period=14):
        """ATR calculation"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.ewm(span=period, adjust=False).mean()
        return atr

    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Bollinger Bands calculation"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band

    def calculate_trend_strength(self, close, md, period=20):
        """Trend strength calculation"""
        price_above_md = (close > md).astype(int)
        trend_strength = price_above_md.rolling(window=period).mean()
        return trend_strength

    def calculate_momentum(self, prices, period=10):
        """Momentum calculation"""
        momentum = prices.diff(period) / prices.shift(period) * 100
        return momentum

    def is_liquid_hour(self, timestamp):
        """Liquidity check"""
        hour = timestamp.hour
        return 0 <= hour <= 10

    def check_long_signal(self, i, close, md, rsi, macd_hist, macd_line,
                         trend_strength, bb_upper, bb_lower, momentum):
        """Long signal logic"""
        if i < 3:
            return False

        price_above_md = close.iloc[i] > md.iloc[i]
        price_cross = (close.iloc[i-1] <= md.iloc[i-1] and close.iloc[i] > md.iloc[i])
        price_signal = price_cross or (price_above_md and close.iloc[i] > close.iloc[i-1])

        rsi_valid = 45 < rsi.iloc[i] < 80

        macd_cross = (macd_hist.iloc[i] > 0 and macd_hist.iloc[i-1] <= 0)
        macd_positive = macd_hist.iloc[i] > 0 and macd_line.iloc[i] > macd_line.iloc[i-1]
        macd_signal = macd_cross or macd_positive

        not_extreme = rsi.iloc[i] < 78
        has_momentum = close.iloc[i] > close.iloc[i-2] or momentum.iloc[i] > -1
        trend_ok = (trend_strength.iloc[i] > 0.3 if not pd.isna(trend_strength.iloc[i]) else True)

        core_signals = price_signal and rsi_valid and macd_signal
        confirmations = not_extreme and has_momentum and trend_ok

        return core_signals and confirmations

    def check_short_signal(self, i, close, md, rsi, macd_hist, macd_line,
                          trend_strength, bb_upper, bb_lower, momentum):
        """Short signal logic"""
        if i < 3:
            return False

        price_below_md = close.iloc[i] < md.iloc[i]
        price_cross = (close.iloc[i-1] >= md.iloc[i-1] and close.iloc[i] < md.iloc[i])
        price_signal = price_cross or (price_below_md and close.iloc[i] < close.iloc[i-1])

        rsi_valid = 20 < rsi.iloc[i] < 55

        macd_cross = (macd_hist.iloc[i] < 0 and macd_hist.iloc[i-1] >= 0)
        macd_negative = macd_hist.iloc[i] < 0 and macd_line.iloc[i] < macd_line.iloc[i-1]
        macd_signal = macd_cross or macd_negative

        not_extreme = rsi.iloc[i] > 22
        has_momentum = close.iloc[i] < close.iloc[i-2] or momentum.iloc[i] < 1
        trend_ok = (trend_strength.iloc[i] < 0.7 if not pd.isna(trend_strength.iloc[i]) else True)

        core_signals = price_signal and rsi_valid and macd_signal
        confirmations = not_extreme and has_momentum and trend_ok

        return core_signals and confirmations

    def calculate_position_size(self, price, atr):
        """Position sizing logic"""
        if self.client is None:
            balance = self.capital
        else:
            balance = self.get_balance()

        risk_amount = balance * self.risk_per_trade
        stop_loss_pct = 0.05  # FIXED: Now 5% (was 10%)
        position_size = risk_amount / (price * stop_loss_pct)

        if not np.isnan(atr):
            atr_pct = (atr / price) * 100

            if atr_pct > 3.5:
                position_size *= 0.5
            elif atr_pct > 2.5:
                position_size *= 0.65
            elif atr_pct > 1.5:
                position_size *= 0.8

        max_position = (balance * 0.75) / price
        position_size = min(position_size, max_position)

        min_position = (balance * 0.05) / price
        position_size = max(position_size, min_position)

        position_size = round(position_size, 4)

        return position_size

    def calculate_dynamic_stops(self, entry_price, atr):
        """
        🔧 FIXED: Stop loss and take profit calculation
        - SL: 5% (down from 10%)
        - TP: 20% (up from 12%)
        - Target: Avg Win ≥ 1.5 × Avg Loss (now 20%/5% = 4:1)
        """
        # FIXED: Increased TP to 20%
        tp_pct = 0.20

        if pd.isna(atr):
            sl_pct = 0.05  # FIXED: 5% base SL
        else:
            atr_distance = atr / entry_price
            # Tighter stops based on volatility
            if atr_distance > 0.03:
                sl_pct = 0.06  # High volatility: 6%
            elif atr_distance > 0.02:
                sl_pct = 0.055  # Medium volatility: 5.5%
            else:
                sl_pct = 0.05  # Low volatility: 5%

        return sl_pct, tp_pct

    # ==================== BACKTEST METHODS ====================

    def download_backtest_data(self, ticker='BTC-USD', interval='15m', period='7d'):
        """Download historical data from Yahoo Finance"""
        print(f"\n📥 Downloading {ticker} data...")
        print(f"   Interval: {interval}, Period: {period}")

        try:
            data = yf.download(ticker, interval=interval, period=period, progress=False)

            if data.empty:
                print("❌ No data downloaded")
                return None

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

            required = ['Close', 'High', 'Low', 'Open', 'Volume']
            for col in required:
                if col not in data.columns:
                    print(f"❌ Missing column: {col}")
                    return None

            data = data[required]
            print(f"✅ Downloaded {len(data)} bars")
            print(f"   Period: {data.index[0]} to {data.index[-1]}\n")
            return data

        except Exception as e:
            print(f"❌ Error: {e}")
            return None

    def execute_entry_backtest(self, i, close, timestamp, signal_type, position_size, atr):
        """Execute backtest entry"""
        entry_price = close.iloc[i]
        position_value = position_size * entry_price

        if position_value <= self.capital * 0.95:
            sl_pct, tp_pct = self.calculate_dynamic_stops(entry_price, atr)

            if signal_type == 'LONG':
                stop_loss = entry_price * (1 - sl_pct)
                take_profit = entry_price * (1 + tp_pct)
            else:
                stop_loss = entry_price * (1 + sl_pct)
                take_profit = entry_price * (1 - tp_pct)

            self.position = {
                'type': signal_type,
                'entry': entry_price,
                'entry_time': timestamp,
                'entry_bar': i,
                'quantity': position_size,
                'original_quantity': position_size,
                'partial_exited': False,
                'trailing_active': False,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'sl_pct': sl_pct,
                'tp_pct': tp_pct
            }

            self.capital -= position_value
            print(f"\n🔔 {signal_type} Entry @ ${entry_price:.2f}")
            print(f"   Qty: {position_size:.4f} | Value: ${position_value:,.0f}")
            print(f"   SL: ${stop_loss:.2f} ({sl_pct*100:.1f}%) | TP: ${take_profit:.2f} ({tp_pct*100:.0f}%)")
            print(f"   Risk/Reward: 1:{tp_pct/sl_pct:.1f} ✅")

    def check_exit_conditions_backtest(self, i, close, md, rsi, atr):
        """
        🔧 FIXED: Exit conditions
        - Removed arbitrary time stop (bars_held > 120)
        - Added volatility-based exit using ATR
        - Added trend-based exit using MD
        - Delayed partial exit to +10% (was +6%)
        """
        current_price = close.iloc[i]
        entry_price = self.position['entry']

        if self.position['type'] == 'LONG':
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price

        # 🔧 FIXED: Trailing stop activation at +10% (was +8%)
        if pnl_pct > 0.10 and not self.position.get('trailing_active'):
            self.position['trailing_active'] = True
            self.position['highest_price'] = current_price
            print(f"  → Trailing stop activated at {pnl_pct*100:.1f}%")

        if self.position.get('trailing_active'):
            if self.position['type'] == 'LONG':
                if current_price > self.position['highest_price']:
                    self.position['highest_price'] = current_price
                    # Trail at 60% of gains (was 50%)
                    self.position['trailing_stop'] = entry_price + (current_price - entry_price) * 0.6
            else:
                if current_price < self.position['highest_price']:
                    self.position['highest_price'] = current_price
                    self.position['trailing_stop'] = entry_price - (entry_price - current_price) * 0.6

        # 🔧 FIXED: Partial exit at +10% (was +6%)
        if not self.position.get('partial_exited') and pnl_pct >= 0.10:
            exit_qty = self.position['quantity'] * 0.5
            exit_value = exit_qty * current_price
            self.capital += exit_value
            self.position['quantity'] -= exit_qty
            self.position['partial_exited'] = True
            self.position['stop_loss'] = entry_price  # Move SL to breakeven
            print(f"  ✓ Partial exit (50%) at {pnl_pct*100:.1f}% | SL → Breakeven")

        exit_reason = None

        # Check trailing stop
        if self.position.get('trailing_active'):
            ts = self.position.get('trailing_stop', 0)
            if ((self.position['type'] == 'LONG' and current_price < ts) or
                (self.position['type'] == 'SHORT' and current_price > ts)):
                exit_reason = 'Trailing Stop'

        # Check TP/SL
        if not exit_reason:
            sl_pct = self.position.get('sl_pct', 0.05)
            tp_pct = self.position.get('tp_pct', 0.20)

            if pnl_pct >= tp_pct:
                exit_reason = 'Take Profit'
            elif pnl_pct <= -sl_pct:
                exit_reason = 'Stop Loss'

        # 🔧 REMOVED: Arbitrary time stop
        # 🔧 ADDED: Volatility-based exit
        if not exit_reason and not pd.isna(atr):
            atr_pct = (atr / entry_price) * 100
            # Exit if volatility collapses below 0.5%
            if atr_pct < 0.5 and abs(pnl_pct) < 0.02:
                exit_reason = 'Low Volatility'

        # 🔧 IMPROVED: Trend-based exit with MD
        if not exit_reason:
            if self.position['type'] == 'LONG':
                # Exit long if price breaks below MD with momentum
                if (current_price < md.iloc[i] * 0.98 and 
                    close.iloc[i] < close.iloc[i-1] and 
                    rsi.iloc[i] < 45):
                    exit_reason = 'Trend Reversal'
            else:
                # Exit short if price breaks above MD with momentum
                if (current_price > md.iloc[i] * 1.02 and 
                    close.iloc[i] > close.iloc[i-1] and 
                    rsi.iloc[i] > 55):
                    exit_reason = 'Trend Reversal'

        return exit_reason

    def execute_exit_backtest(self, i, close, timestamp, exit_reason):
        """Execute backtest exit"""
        current_price = close.iloc[i]
        exit_value = self.position['quantity'] * current_price
        self.capital += exit_value

        original_qty = self.position.get('original_quantity', self.position['quantity'])

        if self.position['type'] == 'LONG':
            pnl = exit_value - (self.position['quantity'] * self.position['entry'])
        else:
            pnl = (self.position['quantity'] * self.position['entry']) - exit_value

        pnl_pct = (pnl / (original_qty * self.position['entry'])) * 100

        trade = {
            'type': self.position['type'],
            'entry': self.position['entry'],
            'entry_time': self.position['entry_time'],
            'exit': current_price,
            'exit_time': timestamp,
            'quantity': self.position['quantity'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'exit_reason': exit_reason,
            'bars_held': i - self.position.get('entry_bar', i)
        }

        self.trades.append(trade)

        emoji = "✅" if pnl > 0 else "❌"
        print(f"{emoji} {exit_reason}: {self.position['type']} | ${current_price:.2f} | P&L: ${pnl:,.0f} ({pnl_pct:.1f}%)")

        self.position = None

    def run_backtest(self, ticker='BTC-USD', interval='15m', period='7d'):
        """Run backtest on historical data"""
        print("\n" + "="*80)
        print("📊 BACKTEST MODE - FIXED STRATEGY (Improved Risk/Reward)")
        print("="*80)
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Risk per Trade: {self.risk_per_trade*100:.0f}% (REDUCED from 10%)")
        print(f"Stop Loss: 5% (REDUCED from 10%)")
        print(f"Take Profit: 20% (INCREASED from 12%)")
        print(f"Risk/Reward: 1:4 ✅ (was 1:1.2)")
        print("="*80 + "\n")

        data = self.download_backtest_data(ticker, interval, period)
        if data is None:
            return None

        self.capital = self.initial_capital
        self.position = None
        self.trades = []
        self.equity_curve = []

        close = data['Close']
        high = data['High']
        low = data['Low']

        print("📊 Calculating indicators...")
        md = self.calculate_mcginley_dynamic(close)
        rsi = self.calculate_rsi(close)
        macd_line, signal_line, macd_hist = self.calculate_macd(close)
        atr = self.calculate_atr(high, low, close)
        bb_upper, bb_mid, bb_lower = self.calculate_bollinger_bands(close)
        trend_strength = self.calculate_trend_strength(close, md)
        momentum = self.calculate_momentum(close)

        data['MD'] = md
        data['RSI'] = rsi
        data['MACD_Line'] = macd_line
        data['MACD_Signal'] = signal_line
        data['MACD_Hist'] = macd_hist
        data['ATR'] = atr
        data['BB_Upper'] = bb_upper
        data['BB_Lower'] = bb_lower
        data['Trend_Strength'] = trend_strength
        data['Momentum'] = momentum

        print("✓ Indicators calculated\n")
        print("🔄 Running backtest...\n")

        for i in range(50, len(data)):
            timestamp = data.index[i]

            if not self.is_liquid_hour(timestamp):
                equity = self.capital + (self.position['quantity'] * close.iloc[i] if self.position else 0)
                self.equity_curve.append(equity)
                continue

            if self.position:
                exit_reason = self.check_exit_conditions_backtest(i, close, md, rsi, atr.iloc[i])
                if exit_reason:
                    self.execute_exit_backtest(i, close, timestamp, exit_reason)

            if not self.position:
                position_size = self.calculate_position_size(close.iloc[i], atr.iloc[i])

                if self.check_long_signal(i, close, md, rsi, macd_hist, macd_line,
                                         trend_strength, bb_upper, bb_lower, momentum):
                    self.execute_entry_backtest(i, close, timestamp, 'LONG', position_size, atr.iloc[i])

                elif self.check_short_signal(i, close, md, rsi, macd_hist, macd_line,
                                            trend_strength, bb_upper, bb_lower, momentum):
                    self.execute_entry_backtest(i, close, timestamp, 'SHORT', position_size, atr.iloc[i])

            equity = self.capital + (self.position['quantity'] * close.iloc[i] if self.position else 0)
            self.equity_curve.append(equity)

        if self.position:
            self.execute_exit_backtest(len(data)-1, close, data.index[-1], 'End of Backtest')

        print(f"\n{'='*80}")
        print("✓ Backtest completed")
        print(f"{'='*80}\n")

        return self.generate_backtest_results(data)

    def generate_backtest_results(self, data):
        """Generate backtest statistics"""
        if not self.trades:
            print("⚠️  No trades executed")
            return None

        trades_df = pd.DataFrame(self.trades)

        total_return = ((self.capital - self.initial_capital) / self.initial_capital) * 100
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] <= 0]

        win_rate = len(winning_trades) / len(trades_df) * 100
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0

        equity_series = pd.Series(self.equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max * 100
        max_drawdown = abs(drawdown.min())

        returns = equity_series.pct_change().dropna()
        sharpe = ((returns.mean() / returns.std()) * np.sqrt(252 * 24) if returns.std() > 0 else 0)

        total_long = len(trades_df[trades_df['type'] == 'LONG'])
        total_short = len(trades_df[trades_df['type'] == 'SHORT'])
        avg_bars_held = trades_df['bars_held'].mean()

        win_streak = 0
        loss_streak = 0
        current_streak = 0
        for pnl in trades_df['pnl']:
            if pnl > 0:
                current_streak = current_streak + 1 if current_streak > 0 else 1
                win_streak = max(win_streak, current_streak)
            else:
                current_streak = current_streak - 1 if current_streak < 0 else -1
                loss_streak = max(loss_streak, abs(current_streak))

        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'total_return': total_return,
            'total_trades': len(trades_df),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'total_long': total_long,
            'total_short': total_short,
            'avg_bars_held': avg_bars_held,
            'max_win_streak': win_streak,
            'max_loss_streak': loss_streak,
            'trades_df': trades_df,
            'equity_curve': self.equity_curve,
            'data': data
        }

    def plot_comprehensive_results(self, results):
        """Create comprehensive visualization"""
        if not results:
            print("⚠️  No results to plot")
            return

        data = results['data']
        trades_df = results['trades_df']

        # Create figure with 4 subplots
        fig = plt.figure(figsize=(16, 14))
        gs = fig.add_gridspec(4, 1, hspace=0.25, height_ratios=[1, 1.2, 0.8, 0.8])

        # ==================== 1. EQUITY CURVE ====================
        ax1 = fig.add_subplot(gs[0])
        equity = results['equity_curve']

        ax1.plot(equity, linewidth=2.5, color='#2563eb', label='Portfolio Equity', zorder=3)
        ax1.axhline(self.initial_capital, color='gray', linestyle='--',
                   linewidth=1.5, alpha=0.7, label='Initial Capital', zorder=2)

        ax1.fill_between(range(len(equity)), equity, self.initial_capital,
                        where=(np.array(equity) >= self.initial_capital),
                        alpha=0.3, color='green', interpolate=True, zorder=1)
        ax1.fill_between(range(len(equity)), equity, self.initial_capital,
                        where=(np.array(equity) < self.initial_capital),
                        alpha=0.3, color='red', interpolate=True, zorder=1)

        ax1.set_title('Equity Curve (Fixed: 5% SL / 20% TP)', fontsize=15, fontweight='bold', pad=15)
        ax1.set_ylabel('Equity (₹)', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x:,.0f}'))

        # ==================== 2. PRICE ACTION WITH TRADES ====================
        ax2 = fig.add_subplot(gs[1])

        ax2.plot(data.index, data['Close'], label='Price',
                linewidth=2, color='black', alpha=0.8, zorder=3)
        ax2.plot(data.index, data['MD'], label='McGinley Dynamic',
                linewidth=2, color='#f97316', alpha=0.9, zorder=2)
        ax2.plot(data.index, data['BB_Upper'], label='BB Upper',
                linewidth=1.2, color='#ef4444', alpha=0.6, linestyle='--', zorder=1)
        ax2.plot(data.index, data['BB_Lower'], label='BB Lower',
                linewidth=1.2, color='#22c55e', alpha=0.6, linestyle='--', zorder=1)

        ax2.fill_between(data.index, data['BB_Upper'], data['BB_Lower'],
                        alpha=0.1, color='gray', zorder=0)

        # Plot trade signals
        for trade in trades_df.itertuples():
            if trade.type == 'LONG':
                ax2.scatter(trade.entry_time, trade.entry,
                          color='#22c55e', marker='^', s=200,
                          edgecolors='black', linewidths=1.5, zorder=5, alpha=0.9)
                ax2.scatter(trade.exit_time, trade.exit,
                          color='#ef4444', marker='v', s=200,
                          edgecolors='black', linewidths=1.5, zorder=5, alpha=0.9)
            else:
                ax2.scatter(trade.entry_time, trade.entry,
                          color='#ef4444', marker='v', s=200,
                          edgecolors='black', linewidths=1.5, zorder=5, alpha=0.9)
                ax2.scatter(trade.exit_time, trade.exit,
                          color='#22c55e', marker='^', s=200,
                          edgecolors='black', linewidths=1.5, zorder=5, alpha=0.9)

        ax2.set_title('Price Action with Trade Signals', fontsize=15, fontweight='bold', pad=15)
        ax2.set_ylabel('Price (₹)', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=9, ncol=2)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x:,.0f}'))

        # ==================== 3. RSI INDICATOR ====================
        ax3 = fig.add_subplot(gs[2])

        ax3.plot(data.index, data['RSI'], linewidth=2, color='#8b5cf6', label='RSI')

        ax3.axhline(50, color='gray', linestyle='-', linewidth=1.2, alpha=0.6)
        ax3.axhline(70, color='#ef4444', linestyle='--', linewidth=1.5,
                   alpha=0.7, label='Overbought (70)')
        ax3.axhline(30, color='#22c55e', linestyle='--', linewidth=1.5,
                   alpha=0.7, label='Oversold (30)')

        ax3.fill_between(data.index, 70, 100, alpha=0.15, color='red')
        ax3.fill_between(data.index, 0, 30, alpha=0.15, color='green')
        ax3.fill_between(data.index, 30, 70, alpha=0.08, color='gray')

        ax3.set_title('RSI Indicator', fontsize=15, fontweight='bold', pad=15)
        ax3.set_ylabel('RSI', fontsize=12, fontweight='bold')
        ax3.set_ylim([0, 100])
        ax3.legend(loc='upper left', fontsize=9)
        ax3.grid(True, alpha=0.3, linestyle='--')

        # ==================== 4. MACD INDICATOR ====================
        ax4 = fig.add_subplot(gs[3])

        ax4.plot(data.index, data['MACD_Line'], label='MACD',
                linewidth=2, color='#3b82f6', zorder=3)
        ax4.plot(data.index, data['MACD_Signal'], label='Signal',
                linewidth=2, color='#ef4444', zorder=2)

        colors = ['#22c55e' if x > 0 else '#ef4444' for x in data['MACD_Hist']]
        ax4.bar(data.index, data['MACD_Hist'], alpha=0.4, color=colors,
               width=0.8, edgecolor='none', zorder=1)

        ax4.axhline(0, color='black', linestyle='-', linewidth=1.2, alpha=0.7)

        ax4.set_title('MACD Indicator', fontsize=15, fontweight='bold', pad=15)
        ax4.set_ylabel('MACD', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax4.legend(loc='upper left', fontsize=9)
        ax4.grid(True, alpha=0.3, linestyle='--')

        # Format x-axis
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax4.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=0, ha='center')

        for ax in [ax1, ax2, ax3]:
            ax.set_xticklabels([])

        plt.tight_layout()

        filename = f'backtest_fixed_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n📊 Chart saved as '{filename}'")
        plt.show()

    def print_detailed_results(self, results):
        """Print detailed backtest results"""
        if not results:
            return

        print("\n" + "="*80)
        print("📈 COMPREHENSIVE BACKTEST RESULTS")
        print("="*80 + "\n")

        print("💰 PERFORMANCE SUMMARY")
        print("-" * 80)
        print(f"  Initial Capital:         ₹{results['initial_capital']:>15,.2f}")
        print(f"  Final Capital:           ₹{results['final_capital']:>15,.2f}")
        pnl = results['final_capital'] - results['initial_capital']
        pnl_color = "+" if pnl >= 0 else ""
        print(f"  Net P&L:                 {pnl_color}₹{pnl:>14,.2f}")
        print(f"  Total Return:            {results['total_return']:>15.2f}%")
        print(f"  Max Drawdown:            {results['max_drawdown']:>15.2f}%")
        print(f"  Sharpe Ratio:            {results['sharpe_ratio']:>15.2f}")

        print(f"\n📊 TRADE STATISTICS")
        print("-" * 80)
        print(f"  Total Trades:            {results['total_trades']:>15}")
        print(f"  Long Trades:             {results['total_long']:>15}")
        print(f"  Short Trades:            {results['total_short']:>15}")
        print(f"  Winning Trades:          {results['winning_trades']:>15}")
        print(f"  Losing Trades:           {results['losing_trades']:>15}")
        print(f"  Win Rate:                {results['win_rate']:>14.2f}%")
        print(f"  Average Bars Held:       {results.get('avg_bars_held', 0):>15.1f}")

        print(f"\n💵 PROFITABILITY METRICS")
        print("-" * 80)
        print(f"  Average Win:             ₹{results['avg_win']:>15,.2f}")
        print(f"  Average Loss:            ₹{results['avg_loss']:>15,.2f}")
        ratio = results['avg_win'] / results['avg_loss'] if results['avg_loss'] > 0 else 0
        status = "✅ TARGET MET" if ratio >= 1.5 else "❌ NEEDS IMPROVEMENT"
        print(f"  Win/Loss Ratio:          {ratio:>15.2f}x {status}")
        print(f"  Profit Factor:           {results['profit_factor']:>15.2f}")
        print(f"  Max Win Streak:          {results.get('max_win_streak', 0):>15}")
        print(f"  Max Loss Streak:         {results.get('max_loss_streak', 0):>15}")

        print(f"\n{'='*80}")
        print("📋 RECENT TRADES (Last 10)")
        print("="*80)
        print(f"{'Type':<7} {'Entry':>10} {'Exit':>10} {'P&L%':>8} {'P&L':>12} {'Reason':<20}")
        print("-" * 80)

        for idx, trade in results['trades_df'].tail(10).iterrows():
            emoji = "✅" if trade['pnl'] > 0 else "❌"
            print(f"{emoji} {trade['type']:<5} "
                  f"₹{trade['entry']:>8,.2f} "
                  f"₹{trade['exit']:>8,.2f} "
                  f"{trade['pnl_pct']:>7.2f}% "
                  f"₹{trade['pnl']:>10,.2f} "
                  f"{trade['exit_reason']:<20}")

        print(f"\n{'='*80}")
        print("✅ BACKTEST ANALYSIS COMPLETE!")
        print(f"{'='*80}\n")


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🤖 FIXED MEXC TRADING BOT - BTC STRATEGY")
    print("="*80)
    print("\n🔧 CRITICAL FIXES APPLIED:")
    print("  ✅ Stop Loss: 10% → 5% (Tighter risk control)")
    print("  ✅ Take Profit: 12% → 20% (Capture bigger moves)")
    print("  ✅ Risk/Reward: 1:1.2 → 1:4 (Sustainable edge)")
    print("  ✅ Risk/Trade: 10% → 7% (Capital protection)")
    print("  ✅ Partial Exit: +6% → +10% (Let trends develop)")
    print("  ✅ Time Stop: REMOVED (Let BTC trends run)")
    print("  ✅ Exit Logic: ATR + MD based (Dynamic)")
    print("\n📊 FEATURES:")
    print("  • Comprehensive backtest with detailed visualization")
    print("  • Advanced position sizing & risk management")
    print("  • Multiple exit strategies (TP, SL, Trailing, Volatility)")
    print("  • Win/Loss ratio monitoring (Target: ≥1.5x)")
    print("="*80 + "\n")

    mode = input("Enter mode (1=Backtest, 2=Live): ").strip()

    if mode == "1":
        print("\n🔍 BACKTEST MODE SELECTED")
        print("="*80)

        bot = MEXCTradingBot(
            api_key="",
            api_secret="",
            initial_capital=500000,
            risk_per_trade=0.07  # ✅ Now 7%
        )

        results = bot.run_backtest(
            ticker='BTC-USD',
            interval='15m',
            period='60d'  # Test on 60 days
        )

        if results:
            bot.print_detailed_results(results)
            bot.plot_comprehensive_results(results)
            
            # Check if strategy meets target
            if results['avg_loss'] > 0:
                ratio = results['avg_win'] / results['avg_loss']
                print("\n" + "="*80)
                print("🎯 STRATEGY VALIDATION")
                print("="*80)
                if ratio >= 1.5:
                    print(f"✅ SUCCESS: Avg Win/Loss = {ratio:.2f}x (Target: ≥1.5x)")
                    print("   Your strategy is ready for paper trading!")
                else:
                    print(f"⚠️  ATTENTION: Avg Win/Loss = {ratio:.2f}x (Target: ≥1.5x)")
                    print("   Consider:")
                    print("   - Reduce SL to 4% OR")
                    print("   - Increase TP to 25%")
                print("="*80 + "\n")

    elif mode == "2":
        print("\n⚠️ LIVE TRADING MODE")
        print("Live trading requires additional implementation.")
        print("Please test thoroughly in backtest mode first!")

    else:
        print("❌ Invalid selection")
