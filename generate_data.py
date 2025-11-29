#!/usr/bin/env python3
"""
Standalone data generation script for GitHub Actions.
Generates data.json with all dashboard data for static deployment.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import json
import os
from datetime import datetime, timedelta, timezone

# Configuration
ASSET_LIST = {
    'BA': 'Stock', 'ABNB': 'Stock', 'SHOP': 'Stock', 'WDAY': 'Stock', 'SWBI': 'Stock',
    'COIN': 'Stock', 'QCOM': 'Stock', 'AMD': 'Stock', 'NVDA': 'Stock', 'PLTR': 'Stock',
    'EQIX': 'Stock', 'DIS': 'Stock', 'PFE': 'Stock', 'PSEC': 'Stock', 'TSLA': 'Stock',
    'MSFT': 'Stock', 'GOOG': 'Stock', 'AMZN': 'Stock', 'AAPL': 'Stock', 'V': 'Stock',
    'UAL': 'Stock', 'DAL': 'Stock',
    'SOXX': 'ETF', 'SPXL': 'ETF', 'TQQQ': 'ETF',
    'BTC-USD': 'Crypto', 'LTC-USD': 'Crypto'
}

MARKET_SYMBOLS = {
    'SPY': 'SPY',
    '^DJI': 'Dow Jones',
    '^GSPC': 'S&P 500',
    '^VIX': 'VIX (Volatility)'
}

DATA_FETCH_PERIOD = "5y"


def calculate_indicators(df):
    """Calculates EMA13, EMA21, EMA100, EMA200, RSI14, and Z-Score for a given DataFrame."""
    if df.empty or 'Close' not in df.columns:
        return pd.DataFrame({'EMA13': [], 'EMA21': [], 'EMA100': [], 'EMA200': [], 'RSI14': [], 'Z_Score_100': []})

    result = pd.DataFrame(index=df.index)
    close_series = df['Close'].astype(float)

    result['EMA13'] = close_series.ewm(span=13, adjust=False).mean()
    result['EMA21'] = close_series.ewm(span=21, adjust=False).mean()
    result['EMA100'] = close_series.ewm(span=100, adjust=False).mean()
    result['EMA200'] = close_series.ewm(span=200, adjust=False).mean()

    # Logarithmic Z-Score Calculation (100-day)
    clean_close_series = close_series.dropna()
    log_close = np.log(clean_close_series)
    sma100_log = log_close.rolling(window=100).mean()
    std100_log = log_close.rolling(window=100).std()
    z_score_series = (log_close - sma100_log) / std100_log
    result['Z_Score_100'] = z_score_series.reindex(result.index)

    delta = close_series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss
    result['RSI14'] = 100.0 - (100.0 / (1.0 + rs.replace(np.inf, np.nan)))
    result['RSI14'] = result['RSI14'].fillna(50)

    return result[['EMA13', 'EMA21', 'EMA100', 'EMA200', 'RSI14', 'Z_Score_100']]


def get_ytd_start_date(reference_date=None):
    """Returns January 1st of the current year (Year-To-Date start)."""
    if reference_date is None:
        reference_date = datetime.now(timezone.utc)
    if reference_date.tzinfo is None:
        reference_date = reference_date.replace(tzinfo=timezone.utc)
    else:
        reference_date = reference_date.astimezone(timezone.utc)
    return reference_date.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)


def get_start_date_from_period(period_str, reference_date=None):
    """Converts period string to a start date."""
    if reference_date is None:
        reference_date = datetime.now(timezone.utc)
    if reference_date.tzinfo is None:
        reference_date = reference_date.replace(tzinfo=timezone.utc)
    else:
        reference_date = reference_date.astimezone(timezone.utc)

    period_map = {
        '1d': timedelta(days=1),
        '1w': timedelta(weeks=1), '7d': timedelta(days=7), '1m': timedelta(days=30),
        '3m': timedelta(days=91), '6m': timedelta(days=182), '1y': timedelta(days=365),
        '2y': timedelta(days=365*2), '3y': timedelta(days=365*3),
        '4y': timedelta(days=365*4), '5y': timedelta(days=365*5),
    }
    delta = period_map.get(period_str.lower())
    return reference_date - delta if delta else reference_date - timedelta(days=1)


def calculate_period_change(close_prices, period_str='1d'):
    """Calculates the percentage change over a specified period."""
    if close_prices.empty or len(close_prices) < 2:
        return 0.0

    latest_price = close_prices.iloc[-1]
    latest_date = close_prices.index[-1]

    if latest_date.tzinfo is None:
        latest_date = latest_date.tz_localize('UTC')
    else:
        latest_date = latest_date.astimezone(timezone.utc)

    if period_str.lower() == '1y':
        start_date = get_ytd_start_date(latest_date)
    else:
        start_date = get_start_date_from_period(period_str, reference_date=latest_date)

    if close_prices.index.tz is None:
        close_prices_index_utc = close_prices.index.tz_localize('UTC')
    else:
        close_prices_index_utc = close_prices.index.tz_convert('UTC')

    historical_prices = close_prices[close_prices_index_utc <= start_date]
    if historical_prices.empty:
        historical_price = close_prices.iloc[0]
        if historical_price == 0 or pd.isna(historical_price):
            return 0.0
    else:
        historical_price = historical_prices.iloc[-1]
        if historical_price == 0 or pd.isna(historical_price):
            return 0.0

    if pd.isna(latest_price):
        return 0.0

    return ((latest_price - historical_price) / historical_price) * 100


def calculate_cumulative_return(close_prices, period_str='1y'):
    """Calculates cumulative percentage return series."""
    if close_prices.empty or len(close_prices) < 2:
        return pd.Series(dtype=float)

    latest_date = close_prices.index[-1]
    if latest_date.tzinfo is None:
        latest_date = latest_date.tz_localize('UTC')
    else:
        latest_date = latest_date.astimezone(timezone.utc)

    if period_str.lower() == '1y':
        start_date = get_ytd_start_date(latest_date)
    else:
        start_date = get_start_date_from_period(period_str, reference_date=latest_date)

    if close_prices.index.tz is None:
        close_prices_index_utc = close_prices.index.tz_localize('UTC')
    else:
        close_prices_index_utc = close_prices.index.tz_convert('UTC')

    if period_str.lower() == '1y':
        baseline_prices = close_prices[close_prices_index_utc <= start_date]
        if baseline_prices.empty:
            baseline_price = close_prices.iloc[0]
        else:
            baseline_price = baseline_prices.iloc[-1]
        
        period_prices = close_prices[close_prices_index_utc >= start_date]
        if period_prices.empty:
            return pd.Series(dtype=float)
        
        cumulative_return = (period_prices / baseline_price - 1) * 100
        return cumulative_return.dropna()
    else:
        period_prices = close_prices[close_prices_index_utc >= start_date]
        if period_prices.empty:
            return pd.Series(dtype=float)
        cumulative_return = (period_prices / period_prices.iloc[0] - 1) * 100
        return cumulative_return.dropna()


def calculate_last_signal_date(fast_series, slow_series):
    """Calculates the date when the fast EMA last crossed above the slow EMA."""
    if fast_series.empty or slow_series.empty:
        return None
    
    # Align series
    common_index = fast_series.index.intersection(slow_series.index)
    if common_index.empty:
        return None
    
    fast = fast_series.loc[common_index]
    slow = slow_series.loc[common_index]
    
    # Determine current signal
    current_signal = 'Buy' if fast.iloc[-1] > slow.iloc[-1] else 'Sell'
    
    if current_signal == 'Sell':
        return None
        
    # Find last crossover
    # Create boolean series where Fast > Slow
    signal_series = fast > slow
    
    # Find changes in signal
    # True where signal changed from False to True or True to False
    changes = signal_series.ne(signal_series.shift())
    
    # We are currently in a Buy state (True). We want the last time it became True.
    # Filter for changes where the new state is True
    buy_signals = changes & signal_series
    
    # Get the last date where this happened
    if buy_signals.any():
        last_buy_date = buy_signals[buy_signals].index[-1]
        return last_buy_date.strftime('%Y-%m-%d')
        
    return None


# Simplified process_asset_data - only essential data for static deployment
def process_asset_simple(symbol, stock_data_raw, spy_1y_change=None):
    """Simplified asset processing for static deployment."""
    try:
        if stock_data_raw is None or stock_data_raw.empty:
            return None

        cols_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
        valid_cols = [col for col in cols_to_keep if col in stock_data_raw.columns]
        if not valid_cols or 'Close' not in valid_cols:
            return None
        
        stock_data = stock_data_raw[valid_cols].copy()
        indicators = calculate_indicators(stock_data)
        combined_data = stock_data.join(indicators)
        combined_data = combined_data.dropna(subset=['Close'])
        
        if combined_data.empty or len(combined_data) < 2:
            return None

        latest_row = combined_data.iloc[-1]
        close_prices = combined_data['Close']

        # Basic data
        result = {
            'name': symbol,
            'display_name': MARKET_SYMBOLS.get(symbol, symbol),
            'type': MARKET_SYMBOLS.get(symbol) or ASSET_LIST.get(symbol, 'Unknown'),
            'latest_price': float(latest_row['Close']),
            'daily_change_pct': calculate_period_change(close_prices, '1d'),
            'ema13': float(latest_row['EMA13']) if pd.notna(latest_row['EMA13']) else None,
            'ema21': float(latest_row['EMA21']) if pd.notna(latest_row['EMA21']) else None,
            'rsi14': float(latest_row['RSI14']) if pd.notna(latest_row['RSI14']) else None,
            'ema100': float(latest_row['EMA100']) if pd.notna(latest_row['EMA100']) else None,
            'ema200': float(latest_row['EMA200']) if pd.notna(latest_row['EMA200']) else None,
            'z_score_100': float(latest_row['Z_Score_100']) if pd.notna(latest_row['Z_Score_100']) else None,
            'ema_signal': None,
            'ema_long_signal': None,
            'ema_short_last_buy_date': None,
            'ema_long_last_buy_date': None,
            'relative_perf_1y': None,
            'sparkline_data': [],
            'asset_1y_history_dates': [],
            'asset_1y_history_values': [],
            'rsi_1y_history_dates': [],
            'rsi_1y_history_values': [],
            'zscore_1y_history_dates': [],
            'zscore_1y_history_values': [],
            'ema_1y_history_dates': [],
            'ema13_1y_history_values': [],
            'ema21_1y_history_values': [],
            'ema_long_1y_history_dates': [],
            'ema100_1y_history_values': [],
            'ema200_1y_history_values': [],
            'drawdown_history_dates': [],
            'drawdown_history_values': [],
            'current_drawdown_pct': None,
        }

        # Signals
        if result['ema13'] is not None and result['ema21'] is not None:
            result['ema_signal'] = 'Buy' if result['ema13'] > result['ema21'] else ('Sell' if result['ema13'] < result['ema21'] else 'Neutral')
            # Calculate duration
            if 'EMA13' in indicators.columns and 'EMA21' in indicators.columns:
                 result['ema_short_last_buy_date'] = calculate_last_signal_date(indicators['EMA13'], indicators['EMA21'])
        
        if result['ema100'] is not None and result['ema200'] is not None:
            result['ema_long_signal'] = 'Buy' if result['ema100'] > result['ema200'] else ('Sell' if result['ema100'] < result['ema200'] else 'Neutral')
            # Calculate duration
            if 'EMA100' in indicators.columns and 'EMA200' in indicators.columns:
                result['ema_long_last_buy_date'] = calculate_last_signal_date(indicators['EMA100'], indicators['EMA200'])

        # YTD change
        asset_1y_change = calculate_period_change(close_prices, '1y')
        if spy_1y_change is not None and symbol != 'SPY':
            result['relative_perf_1y'] = asset_1y_change - spy_1y_change if asset_1y_change is not None else None
        elif symbol == 'SPY':
            result['relative_perf_1y'] = 0.0

        # Sparkline (last 30 days)
        if len(close_prices) >= 30:
            result['sparkline_data'] = [float(p) for p in close_prices.tail(30).tolist() if pd.notna(p)]

        # Cumulative return history
        cum_ret = calculate_cumulative_return(close_prices, '1y')
        if not cum_ret.empty:
            result['asset_1y_history_dates'] = cum_ret.index.strftime('%Y-%m-%d').tolist()
            result['asset_1y_history_values'] = [round(float(v), 2) for v in cum_ret.values.tolist()]

        # RSI history
        if 'RSI14' in indicators.columns:
            rsi_series = indicators['RSI14'].dropna()
            if len(rsi_series) >= 2:
                rsi_start = get_start_date_from_period('1y', rsi_series.index[-1].tz_localize('UTC') if rsi_series.index[-1].tzinfo is None else rsi_series.index[-1])
                rsi_idx_utc = rsi_series.index.tz_localize('UTC') if rsi_series.index.tz is None else rsi_series.index
                rsi_1y = rsi_series[rsi_idx_utc >= rsi_start]
                if not rsi_1y.empty:
                    result['rsi_1y_history_dates'] = rsi_1y.index.strftime('%Y-%m-%d').tolist()
                    result['rsi_1y_history_values'] = [round(float(v), 2) for v in rsi_1y.values.tolist()]

        # Z-Score history
        if 'Z_Score_100' in indicators.columns:
            zscore_series = indicators['Z_Score_100'].dropna()
            if len(zscore_series) >= 2:
                zscore_start = get_start_date_from_period('1y', zscore_series.index[-1].tz_localize('UTC') if zscore_series.index[-1].tzinfo is None else zscore_series.index[-1])
                zscore_idx_utc = zscore_series.index.tz_localize('UTC') if zscore_series.index.tz is None else zscore_series.index
                zscore_1y = zscore_series[zscore_idx_utc >= zscore_start]
                if not zscore_1y.empty:
                    result['zscore_1y_history_dates'] = zscore_1y.index.strftime('%Y-%m-%d').tolist()
                    result['zscore_1y_history_values'] = [round(float(v), 2) for v in zscore_1y.values.tolist()]

        # Calculate drawdown
        if len(close_prices) >= 2:
            # Current drawdown from peak
            peak_price = close_prices.max()
            current_price = close_prices.iloc[-1]
            if peak_price > 0:
                result['current_drawdown_pct'] = ((current_price - peak_price) / peak_price) * 100
            
            # Drawdown history (1 year)
            latest_date = close_prices.index[-1]
            if latest_date.tzinfo is None:
                latest_date = latest_date.tz_localize('UTC')
            drawdown_start = get_start_date_from_period('1y', latest_date)
            
            close_idx_utc = close_prices.index.tz_localize('UTC') if close_prices.index.tz is None else close_prices.index
            period_prices = close_prices[close_idx_utc >= drawdown_start]
            
            if len(period_prices) >= 2:
                rolling_max = period_prices.expanding().max()
                drawdown_series = ((period_prices - rolling_max) / rolling_max) * 100
                result['drawdown_history_dates'] = drawdown_series.index.strftime('%Y-%m-%d').tolist()
                result['drawdown_history_values'] = [round(float(v), 2) for v in drawdown_series.values.tolist()]

        # EMA History (Short Signal: 13/21)
        if 'EMA13' in indicators.columns and 'EMA21' in indicators.columns:
            ema13_series = indicators['EMA13'].dropna()
            ema21_series = indicators['EMA21'].dropna()
            
            if len(ema13_series) >= 2 and len(ema21_series) >= 2:
                # Use EMA13 index as reference
                latest_date = ema13_series.index[-1]
                if latest_date.tzinfo is None:
                    latest_date = latest_date.tz_localize('UTC')
                ema_start = get_start_date_from_period('1y', latest_date)
                
                ema_idx_utc = ema13_series.index.tz_localize('UTC') if ema13_series.index.tz is None else ema13_series.index
                
                # Filter both series
                ema13_1y = ema13_series[ema_idx_utc >= ema_start]
                ema21_1y = ema21_series[ema_idx_utc >= ema_start] # Assuming same index alignment
                
                # Align indices (intersection)
                common_dates = ema13_1y.index.intersection(ema21_1y.index)
                
                if not common_dates.empty:
                    result['ema_1y_history_dates'] = common_dates.strftime('%Y-%m-%d').tolist()
                    result['ema13_1y_history_values'] = [round(float(v), 2) for v in ema13_1y.loc[common_dates].values.tolist()]
                    result['ema21_1y_history_values'] = [round(float(v), 2) for v in ema21_1y.loc[common_dates].values.tolist()]

        # EMA Long History (Long Signal: 100/200)
        if 'EMA100' in indicators.columns and 'EMA200' in indicators.columns:
            ema100_series = indicators['EMA100'].dropna()
            ema200_series = indicators['EMA200'].dropna()
            
            if len(ema100_series) >= 2 and len(ema200_series) >= 2:
                # Use EMA100 index as reference
                latest_date = ema100_series.index[-1]
                if latest_date.tzinfo is None:
                    latest_date = latest_date.tz_localize('UTC')
                ema_start = get_start_date_from_period('1y', latest_date)
                
                ema_idx_utc = ema100_series.index.tz_localize('UTC') if ema100_series.index.tz is None else ema100_series.index
                
                # Filter both series
                ema100_1y = ema100_series[ema_idx_utc >= ema_start]
                ema200_1y = ema200_series[ema_idx_utc >= ema_start]
                
                # Align indices
                common_dates = ema100_1y.index.intersection(ema200_1y.index)
                
                if not common_dates.empty:
                    result['ema_long_1y_history_dates'] = common_dates.strftime('%Y-%m-%d').tolist()
                    result['ema100_1y_history_values'] = [round(float(v), 2) for v in ema100_1y.loc[common_dates].values.tolist()]
                    result['ema200_1y_history_values'] = [round(float(v), 2) for v in ema200_1y.loc[common_dates].values.tolist()]

        return result
    except Exception as e:
        print(f"Error processing {symbol}: {e}")
        return None


def main():
    """Generate dashboard data and save to data.json"""
    print(f"Starting data generation at {datetime.now(timezone.utc).isoformat()}")
    
    market_data = {}
    asset_data = {}
    spy_1y_change = None
    spy_1y_history_dates = []
    spy_1y_history_values = []
    
    # Fetch SPY
    print("Fetching SPY...")
    try:
        spy_data_raw = yf.download('SPY', period="13mo", interval="1d", progress=False, auto_adjust=False, actions=False)
        if isinstance(spy_data_raw.columns, pd.MultiIndex):
            spy_data_raw.columns = spy_data_raw.columns.get_level_values(0)
        
        if not spy_data_raw.empty and 'Close' in spy_data_raw.columns:
            spy_close_prices = spy_data_raw['Close'].dropna()
            if len(spy_close_prices) >= 2:
                spy_1y_change = calculate_period_change(spy_close_prices, '1y')
                print(f"SPY 1Y Change: {spy_1y_change:.2f}%")
                
                spy_cum_ret = calculate_cumulative_return(spy_close_prices, '1y')
                if not spy_cum_ret.empty:
                    spy_1y_history_dates = spy_cum_ret.index.strftime('%Y-%m-%d').tolist()
                    spy_1y_history_values = [round(float(v), 2) for v in spy_cum_ret.values.tolist()]
    except Exception as e:
        print(f"Error fetching SPY: {e}")
    
    # Batch fetch
    all_symbols = list(set(list(MARKET_SYMBOLS.keys()) + list(ASSET_LIST.keys())))
    print(f"Fetching {len(all_symbols)} symbols...")
    
    batch_data = yf.download(all_symbols, period=DATA_FETCH_PERIOD, interval="1d", group_by='ticker', progress=False, auto_adjust=False, actions=False)
    
    if not batch_data.empty:
        if len(all_symbols) == 1:
            batch_data.columns = pd.MultiIndex.from_product([all_symbols, batch_data.columns])
        if batch_data.index.tz is None:
            batch_data.index = batch_data.index.tz_localize('UTC')
    
    # Process market data
    for symbol in MARKET_SYMBOLS.keys():
        try:
            if symbol in batch_data.columns.get_level_values(0):
                market_data[symbol] = process_asset_simple(symbol, batch_data[symbol], spy_1y_change)
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
    
    # Process asset data
    for symbol in ASSET_LIST.keys():
        try:
            if symbol in batch_data.columns.get_level_values(0):
                asset_data[symbol] = process_asset_simple(symbol, batch_data[symbol], spy_1y_change)
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
    
    # Create output
    def clean_nan(obj):
        if isinstance(obj, float):
            return None if np.isnan(obj) else obj
        elif isinstance(obj, dict):
            return {k: clean_nan(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_nan(v) for v in obj]
        return obj
    
    output_data = clean_nan({
        'market_data': market_data,
        'asset_data': asset_data,
        'spy_1y_history': {'dates': spy_1y_history_dates, 'values': spy_1y_history_values},
        'generated_at': datetime.now(timezone.utc).isoformat()
    })
    
    # Write to file
    output_path = os.path.join(os.path.dirname(__file__), 'frontend', 'data.json')
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Successfully generated {output_path}")
    print(f"Processed {len(market_data)} market symbols and {len(asset_data)} assets")


if __name__ == '__main__':
    main()
