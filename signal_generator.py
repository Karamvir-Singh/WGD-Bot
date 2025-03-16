from trade_finder import detect_fvg, find_fvg_intersections
from get_oanda_data import get_oanda_data


def signal_generator(df_5m, bullish_ggs, bearish_ggs, risk_reward=2, short_signal=False):
    """
    Generate trading signals based on the last 2 candles and identified Golden Gaps
    
    Args:
        df_5m: DataFrame with the latest 5-minute candles
        bullish_ggs: List of bullish Golden Gaps
        bearish_ggs: List of bearish Golden Gaps
        risk_reward: Risk-reward ratio for calculating take-profit levels
        short_signal: If True, returns just 'buy'/'sell'/None. If False, returns full signal dict
    
    Returns:
        dict: Dictionary with signal type ('buy', 'sell', or None), entry, stop loss, and take profit levels
        OR str/None: Just the signal type if short_signal=True
    """
    if len(df_5m) < 2:
        return 'None' if short_signal else {'signal': None}
        
    # Get the last two candles
    current_candle = {
        'open': df_5m['Open'].iloc[-1],
        'high': df_5m['High'].iloc[-1],
        'low': df_5m['Low'].iloc[-1],
        'close': df_5m['Close'].iloc[-1],
        'time': df_5m.index[-1]
    }
    
    prev_candle = {
        'open': df_5m['Open'].iloc[-2],
        'high': df_5m['High'].iloc[-2], 
        'low': df_5m['Low'].iloc[-2],
        'close': df_5m['Close'].iloc[-2],
        'time': df_5m.index[-2]
    }
    
    # Calculate candle based highs and lows
    candle_based_high = max(current_candle['high'], prev_candle['high'])
    candle_based_low = min(current_candle['low'], prev_candle['low'])
    
    # Check bearish patterns
    for gg in bearish_ggs:
        if not gg['active'] or current_candle['time'] < gg['start_time'] or current_candle['time'] > gg['end_time']:
            continue
        
        # Pattern 1: Touch reversal
        if (current_candle['high'] >= gg['low'] and
            current_candle['close'] < gg['low'] and
            prev_candle['high'] < gg['low']):
            
            entry = current_candle['close']
            sl = min(candle_based_high, gg['high'])
            risk = sl - entry
            tp = entry - (risk * risk_reward)
            
            if short_signal:
                return 'sell'
            return {
                'signal': 'sell',
                'entry': entry,
                'sl': sl,
                'tp': tp,
                'pattern': 'pattern1',
                'gg': gg
            }
            
        # Pattern 2: Inverted Gap kill reversal
        elif (current_candle['close'] < current_candle['open'] and  # Red candle
            current_candle['open'] > gg['high'] and
            current_candle['close'] < gg['low'] and
            prev_candle['open'] < gg['low'] and
            prev_candle['close'] > gg['high'] and
            prev_candle['close'] > prev_candle['open']):  # Previous green
            
            entry = current_candle['close']
            sl = min(candle_based_high, gg['high'])
            risk = sl - entry
            tp = entry - (risk * risk_reward)
            
            if short_signal:
                return 'sell'
            return {
                'signal': 'sell',
                'entry': entry,
                'sl': sl,
                'tp': tp,
                'pattern': 'pattern2',
                'gg': gg
            }
            
        # Pattern 3: Previous green candle closing inside GG and current red candle reversing
        elif (current_candle['close'] < current_candle['open'] and  # Red candle
            current_candle['close'] < gg['low'] and
            gg['low'] < current_candle['open'] < gg['high'] and
            prev_candle['open'] < gg['low'] and
            gg['low'] < prev_candle['close'] < gg['high'] and
            prev_candle['close'] > prev_candle['open']):  # Previous green
            
            entry = current_candle['close']
            sl = min(candle_based_high, gg['high'])
            risk = sl - entry
            tp = entry - (risk * risk_reward)
            
            if short_signal:
                return 'sell'
            return {
                'signal': 'sell',
                'entry': entry,
                'sl': sl,
                'tp': tp,
                'pattern': 'pattern3',
                'gg': gg
            }
    
    # Check bullish patterns
    for gg in bullish_ggs:
        if not gg['active'] or current_candle['time'] < gg['start_time'] or current_candle['time'] > gg['end_time']:
            continue
            
        # Pattern 1: Touch reversal
        if (current_candle['low'] <= gg['high'] and
            current_candle['close'] > gg['high'] and
            prev_candle['low'] > gg['high']):
            
            entry = current_candle['close']
            sl = max(candle_based_low, gg['low'])
            risk = entry - sl
            tp = entry + (risk * risk_reward)
            
            if short_signal:
                return 'buy'
            return {
                'signal': 'buy',
                'entry': entry,
                'sl': sl,
                'tp': tp,
                'pattern': 'pattern1',
                'gg': gg
            }
            
        # Pattern 2: Inverted Gap kill reversal
        elif (current_candle['close'] > current_candle['open'] and  # Green candle
            current_candle['open'] < gg['low'] and
            current_candle['close'] > gg['high'] and
            prev_candle['open'] > gg['high'] and
            prev_candle['close'] < gg['low'] and
            prev_candle['close'] < prev_candle['open']):  # Previous red
            
            entry = current_candle['close']
            sl = max(candle_based_low, gg['low'])
            risk = entry - sl
            tp = entry + (risk * risk_reward)
            
            if short_signal:
                return 'buy'
            return {
                'signal': 'buy',
                'entry': entry,
                'sl': sl,
                'tp': tp,
                'pattern': 'pattern2',
                'gg': gg
            }
            
        # Pattern 3: Previous red candle closing inside GG and current green candle reversing
        elif (current_candle['close'] > current_candle['open'] and  # Green candle
            current_candle['close'] > gg['high'] and
            gg['low'] < current_candle['open'] < gg['high'] and
            prev_candle['open'] > gg['high'] and
            gg['low'] < prev_candle['close'] < gg['high'] and
            prev_candle['close'] < prev_candle['open']):  # Previous red
            
            entry = current_candle['close']
            sl = max(candle_based_low, gg['low'])
            risk = entry - sl
            tp = entry + (risk * risk_reward)
            
            if short_signal:
                return 'buy'
            return {
                'signal': 'buy',
                'entry': entry,
                'sl': sl,
                'tp': tp,
                'pattern': 'pattern3',
                'gg': gg
            }
    
    # No pattern detected
    return 'None' if short_signal else {'signal': None}

# Simple test for the signal generator
if __name__ == "__main__":
    
    symbol = "SPX500_USD"
    HTF = "M15"
    LTF = "M1"
    days = 5

    # 1. Fetch market data
    df_ltf = get_oanda_data(symbol, LTF, str(days))
    df_htf = get_oanda_data(symbol, HTF, str(days))
    
    # 2. Detect Fair Value Gaps
    fvgs_ltf = detect_fvg(df_ltf, LTF)
    fvgs_htf = detect_fvg(df_htf, HTF)
    
    # 3. Find intersections (Golden Gaps)
    bullish_ggs, bearish_ggs = find_fvg_intersections(fvgs_ltf, fvgs_htf)
    
    # 4. Generate signals for each candle
    # First two candles get None as signal (need at least 2 candles to generate signal)
    signals = ['None', 'None']
    
    for i in range(2, len(df_ltf)):
        candle_window = df_ltf.iloc[i-2:i+1]
        signals.append(signal_generator(candle_window, bullish_ggs, bearish_ggs, risk_reward=3, short_signal=True))
    
    # 5. Add signals to dataframe with proper alignment
    df_ltf['signal'] = signals

    print(df_ltf.signal.value_counts())
    