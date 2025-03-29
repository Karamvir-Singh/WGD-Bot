import yfinance as yf
import pandas as pd


def detect_fvg(df, timeframe):
    """Detect Fair Value Gaps (FVGs) and track them separately"""
    bullish_fvgs = []
    bearish_fvgs = []
    
    # Detect All FVGs
    for i in range(2, len(df)):
        current_time = df.index[i] 
        # Check for new bullish FVG
        if df['High'].iloc[i-2] < df['Low'].iloc[i]:
            # Calculate default end_time (5 bars after confirmation candle or last bar if out of range)
            end_idx = len(df)-1
            
            fvg = {
                'high': df['Low'].iloc[i],
                'low': df['High'].iloc[i-2],
                'start_time': current_time,  # Confirmation candle is the start time
                'end_time': df.index[end_idx],
                'active': True,
                'timeframe': timeframe
            }
            bullish_fvgs.append(fvg)
        
        # Check for new bearish FVG
        if df['Low'].iloc[i-2] > df['High'].iloc[i]:
            # Calculate default end_time (5 bars after confirmation candle or last bar if out of range)
            end_idx = len(df)-1
            
            fvg = {
                'high': df['Low'].iloc[i-2],
                'low': df['High'].iloc[i],
                'start_time': current_time,  # Confirmation candle is the start time
                'end_time': df.index[end_idx],
                'active': True,
                'timeframe': timeframe
            }
            bearish_fvgs.append(fvg)

    # Update all FVG end times if price trades through the FVG
    for i in range(2, len(df)):
        current_time = df.index[i]
        # Update active FVGs with current candle information
        high = df['High'].iloc[i]
        low = df['Low'].iloc[i]
        close = df['Close'].iloc[i]
        
        # Update bullish FVGs - invalidated if price trades through the top of the FVG
        for fvg in bullish_fvgs:
            if fvg['active']:
                if close < fvg['low'] and current_time > fvg['start_time']:
                    fvg['active'] = False
                    fvg['end_time'] = current_time
                    # print(f"Bullish FVG invalidated at {current_time}, setting end_time")
        
        # Update bearish FVGs - invalidated if price trades through the bottom of the FVG
        for fvg in bearish_fvgs:
            if fvg['active']:
                if close > fvg['high'] and current_time > fvg['start_time']:
                    fvg['active'] = False
                    fvg['end_time'] = current_time
                    # print(f"Bearish FVG invalidated at {current_time}, setting end_time")
    
    # Set the end_time for any remaining active FVGs to the last bar in the dataset
    last_time = df.index[-1]
    for fvg in bullish_fvgs + bearish_fvgs:
        if fvg['active'] and current_time == last_time:
            # Only update end_time if it's beyond the dataset or not set
            if fvg['end_time'] is None or fvg['end_time'] > last_time:
                fvg['end_time'] = last_time
    
    return bullish_fvgs, bearish_fvgs

def get_yf_data_and_fvgs(symbol, start_date, end_date):
    """Get data from yfinance for both 1h and 5min timeframes"""
    # Get 1h data
    df_1h = yf.download(symbol, start=start_date, end=end_date, interval='1h')
    # Remove the extra 'Ticker' row from the header if it exists
    if isinstance(df_1h.columns, pd.MultiIndex):
        df_1h.columns = df_1h.columns.droplevel(1)
    fvgs_1h = detect_fvg(df_1h, '1h')
    
    # Get 5min data
    df_5m = yf.download(symbol, start=start_date, end=end_date, interval='5m')
    # Remove the extra 'Ticker' row from the header if it exists
    if isinstance(df_5m.columns, pd.MultiIndex):
        df_5m.columns = df_5m.columns.droplevel(1)
    fvgs_5m = detect_fvg(df_5m, '5m')
    
    return df_1h, df_5m, fvgs_1h, fvgs_5m


def find_fvg_intersections(fvgs_1h, fvgs_5m, check_active=False):
    """
    Find intersecting areas between 1h and 5m FVGs of the same type
    
    Args:
        fvgs_1h: List of 1h FVGs
        fvgs_5m: List of 5m FVGs 
        check_active: If True, only consider active FVGs. If False, consider all FVGs.
    """
    bullish_ggs = []
    bearish_ggs = []
    
    # Find bullish intersections
    for fvg_1h in fvgs_1h[0]:  # [0] for bullish FVGs
        if check_active and not fvg_1h['active']:
            continue
            
        for fvg_5m in fvgs_5m[0]:
            if check_active and not fvg_5m['active']:
                continue
                
            # Check time overlap
            time_overlap_start = max(fvg_1h['start_time'], fvg_5m['start_time'])
            time_overlap_end = min(fvg_1h['end_time'], fvg_5m['end_time'])
            
            # Check if there is time overlap
            if time_overlap_end > time_overlap_start:
                # Find price overlap
                overlap_high = min(fvg_1h['high'], fvg_5m['high'])
                overlap_low = max(fvg_1h['low'], fvg_5m['low'])
                
                if overlap_high > overlap_low:
                    gg = {
                        'high': overlap_high,
                        'low': overlap_low,
                        'start_time': time_overlap_start,
                        'end_time': time_overlap_end,
                        'active': True,
                        'type': 'bullish'
                    }
                    bullish_ggs.append(gg)
    
    # Find bearish intersections
    for fvg_1h in fvgs_1h[1]:  # [1] for bearish FVGs
        if check_active and not fvg_1h['active']:
            continue
            
        for fvg_5m in fvgs_5m[1]:
            if check_active and not fvg_5m['active']:
                continue
                
            # Check time overlap
            time_overlap_start = max(fvg_1h['start_time'], fvg_5m['start_time'])
            time_overlap_end = min(fvg_1h['end_time'], fvg_5m['end_time'])
            
            # Check if there is time overlap
            if time_overlap_end > time_overlap_start:
                # Find price overlap
                overlap_high = min(fvg_1h['high'], fvg_5m['high'])
                overlap_low = max(fvg_1h['low'], fvg_5m['low'])
                
                if overlap_high > overlap_low:
                    gg = {
                        'high': overlap_high,
                        'low': overlap_low,
                        'start_time': time_overlap_start,
                        'end_time': time_overlap_end,
                        'active': True,
                        'type': 'bearish'
                    }
                    bearish_ggs.append(gg)
    
    return bullish_ggs, bearish_ggs

def find_gg_reversal_pattern_trades(df, bullish_ggs, bearish_ggs, risk_reward=2):
    """
    Find reversal patterns using Golden Gaps and calculate trade setups.
    
    Args:
        df (pandas.DataFrame): Price data with OHLC columns
        bullish_ggs (list): List of bullish Golden Gaps
        bearish_ggs (list): List of bearish Golden Gaps
        risk_reward (float): Risk/Reward ratio for calculating take profit levels
        
    Returns:
        dict: Dictionary containing bearish and bullish trade setups with entry, stop loss, take profit
    """
    trades = {
        'bearish': [],
        'bullish': []
    }
    
    # Iterate through price data starting from second candle
    for i in range(1, len(df)-1):
        current_candle = {
            'open': df['Open'].iloc[i],
            'high': df['High'].iloc[i],
            'low': df['Low'].iloc[i],
            'close': df['Close'].iloc[i],
            'close_bid': df['Close_bid'].iloc[i] if 'Close_bid' in df.columns else df['Close'].iloc[i],
            'close_ask': df['Close_ask'].iloc[i] if 'Close_ask' in df.columns else df['Close'].iloc[i],
            'spread': df['Close_spread'].iloc[i] if 'Close_spread' in df.columns else 0,
            'time': df.index[i]
        }
        
        prev_candle = {
            'open': df['Open'].iloc[i-1],
            'high': df['High'].iloc[i-1], 
            'low': df['Low'].iloc[i-1],
            'close': df['Close'].iloc[i-1],
            'time': df.index[i-1]
        }

        trade_time = df.index[i+1]

        # Calculate candle based highs and lows
        candle_based_high = max(current_candle['high'], prev_candle['high'])
        candle_based_low = min(current_candle['low'], prev_candle['low'])

        # Check bearish patterns against bearish GGs
        for gg in bearish_ggs:
            if not gg['active'] or current_candle['time'] < gg['start_time'] or current_candle['time'] > gg['end_time']:
                continue
                
            # Pattern 1: Green candle touch reversal
            if (
                # current_candle['close'] > current_candle['open'] and  # Green candle
                current_candle['high'] >= gg['low'] and
                current_candle['close'] < gg['low'] and
                prev_candle['high'] < gg['low']):
                
                # Use bid price for selling
                entry = round(current_candle['close_bid'], 1)
                sl = min(candle_based_high, gg['high'])
                
                # Check if SL is too close to entry (within the spread)
                if sl - entry <= current_candle['spread']:
                    # Adjust SL by 1 spread
                    sl += current_candle['spread'] * 1.0
                
                # Round the stop loss to 1 decimal place
                sl = round(sl, 1)
                risk = round(sl - entry, 1)
                tp = round(entry - (risk * risk_reward), 1)
                
                trades['bearish'].append({
                    'time': trade_time,
                    'entry': entry,
                    'sl': sl,
                    'tp': tp,
                    'trailing_sl': None,
                    'type': 'pattern1',
                    'gg': gg,
                    'risk_pips': risk,
                    'spread': current_candle['spread'],
                    'direction': 'bearish'
                })

                # Print progress after every 30 trades
                if (len(trades['bearish']) + len(trades['bullish'])) % 30 == 0:
                    print(f"Processed {len(trades['bearish']) + len(trades['bullish'])} trades...")
                
            # Pattern 2: Inverted Gap kill reversal
            elif (current_candle['close'] < current_candle['open'] and  # Red candle
                current_candle['open'] > gg['high'] and
                current_candle['close'] < gg['low'] and
                prev_candle['open'] < gg['low'] and
                prev_candle['close'] > gg['high'] and
                prev_candle['close'] > prev_candle['open']):  # Previous green
                
                # Use bid price for selling
                entry = round(current_candle['close_bid'], 1)
                sl = min(candle_based_high, gg['high'])
                
                # Check if SL is too close to entry (within the spread)
                if sl - entry <= current_candle['spread']:
                    # Adjust SL by 1 spread
                    sl += current_candle['spread'] * 1.0
                
                # Round the stop loss to 1 decimal place
                sl = round(sl, 1)
                risk = round(sl - entry, 1)
                tp = round(entry - (risk * risk_reward), 1)
                
                trades['bearish'].append({
                    'time': trade_time,
                    'entry': entry,
                    'sl': sl,
                    'tp': tp,
                    'trailing_sl': None,
                    'type': 'pattern2',
                    'gg': gg,
                    'risk_pips': risk,
                    'spread': current_candle['spread'],
                    'direction': 'bearish'
                })

                # Print progress after every 30 trades
                if (len(trades['bearish']) + len(trades['bullish'])) % 30 == 0:
                    print(f"Processed {len(trades['bearish']) + len(trades['bullish'])} trades...")
                
            # Pattern 3: previous green candle closing inside GG and current red candle reversing.
            elif (current_candle['close'] < current_candle['open'] and  # Red candle
                current_candle['close'] < gg['low'] and
                gg['low'] < current_candle['open'] < gg['high'] and
                prev_candle['open'] < gg['low'] and
                gg['low'] < prev_candle['close'] < gg['high'] and
                prev_candle['close'] > prev_candle['open']):  # Previous green
                
                # Use bid price for selling
                entry = round(current_candle['close_bid'], 1)
                sl = min(candle_based_high, gg['high'])
                
                # Check if SL is too close to entry (within the spread)
                if sl - entry <= current_candle['spread']:
                    # Adjust SL by 1 spread
                    sl += current_candle['spread'] * 1.0
                
                # Round the stop loss to 1 decimal place
                sl = round(sl, 1)
                risk = round(sl - entry, 1)
                tp = round(entry - (risk * risk_reward), 1)
                
                trades['bearish'].append({
                    'time': trade_time,
                    'entry': entry,
                    'sl': sl,
                    'tp': tp,
                    'trailing_sl': None,
                    'type': 'pattern3',
                    'gg': gg,
                    'risk_pips': risk,
                    'spread': current_candle['spread'],
                    'direction': 'bearish'
                })

                # Print progress after every 30 trades
                if (len(trades['bearish']) + len(trades['bullish'])) % 30 == 0:
                    print(f"Processed {len(trades['bearish']) + len(trades['bullish'])} trades...")

        # Check bullish patterns against bullish GGs  
        for gg in bullish_ggs:
            if not gg['active'] or current_candle['time'] < gg['start_time'] or current_candle['time'] > gg['end_time']:
                continue
                
            # Pattern 1: Red candle touch reversal
            if (
                # current_candle['close'] < current_candle['open'] and  # Red candle
                current_candle['low'] <= gg['high'] and
                current_candle['close'] > gg['high'] and
                prev_candle['low'] > gg['high']):
                
                # Use ask price for buying
                entry = round(current_candle['close_ask'], 1)
                sl = max(candle_based_low, gg['low'])
                
                # Check if SL is too close to entry (within the spread)
                if entry - sl <= current_candle['spread']:
                    # Adjust SL by 1 spread
                    sl -= current_candle['spread'] * 1.0
                
                # Round the stop loss to 1 decimal place
                sl = round(sl, 1)
                risk = round(entry - sl, 1)
                tp = round(entry + (risk * risk_reward), 1)
                
                trades['bullish'].append({
                    'time': trade_time,
                    'entry': entry,
                    'sl': sl,
                    'tp': tp,
                    'trailing_sl': None,
                    'type': 'pattern1',
                    'gg': gg,
                    'risk_pips': risk,
                    'spread': current_candle['spread'],
                    'direction': 'bullish'
                })

                # Print progress after every 30 trades
                if (len(trades['bearish']) + len(trades['bullish'])) % 30 == 0:
                    print(f"Processed {len(trades['bearish']) + len(trades['bullish'])} trades...")
                
            # Pattern 2: Inverted Gap kill reversal
            elif (current_candle['close'] > current_candle['open'] and  # Green candle
                current_candle['open'] < gg['low'] and
                current_candle['close'] > gg['high'] and
                prev_candle['open'] > gg['high'] and
                prev_candle['close'] < gg['low'] and
                prev_candle['close'] < prev_candle['open']):  # Previous red
                
                # Use ask price for buying
                entry = round(current_candle['close_ask'], 1)
                sl = max(candle_based_low, gg['low'])
                
                # Check if SL is too close to entry (within the spread)
                if entry - sl <= current_candle['spread']:
                    # Adjust SL by 1 spread
                    sl -= current_candle['spread'] * 1.0
                
                # Round the stop loss to 1 decimal place
                sl = round(sl, 1)
                risk = round(entry - sl, 1)
                tp = round(entry + (risk * risk_reward), 1)
                
                trades['bullish'].append({
                    'time': trade_time,
                    'entry': entry,
                    'sl': sl,
                    'tp': tp,
                    'trailing_sl': None,
                    'type': 'pattern2',
                    'gg': gg,
                    'risk_pips': risk,
                    'spread': current_candle['spread'],
                    'direction': 'bullish'
                })

                # Print progress after every 30 trades
                if (len(trades['bearish']) + len(trades['bullish'])) % 30 == 0:
                    print(f"Processed {len(trades['bearish']) + len(trades['bullish'])} trades...")
                
            # Pattern 3: previous red candle closing inside GG and current green candle reversing.
            elif (current_candle['close'] > current_candle['open'] and  # Green candle
                current_candle['close'] > gg['high'] and
                gg['low'] < current_candle['open'] < gg['high'] and
                prev_candle['open'] > gg['high'] and
                gg['low'] < prev_candle['close'] < gg['high'] and
                prev_candle['close'] < prev_candle['open']):  # Previous red
                
                # Use ask price for buying
                entry = round(current_candle['close_ask'], 1)
                sl = max(candle_based_low, gg['low'])
                
                # Check if SL is too close to entry (within the spread)
                if entry - sl <= current_candle['spread']:
                    # Adjust SL by 1 spread
                    sl -= current_candle['spread'] * 1.0
                
                # Round the stop loss to 1 decimal place
                sl = round(sl, 1)
                risk = round(entry - sl, 1)
                tp = round(entry + (risk * risk_reward), 1)
                
                trades['bullish'].append({
                    'time': trade_time,
                    'entry': entry,
                    'sl': sl,
                    'tp': tp,
                    'trailing_sl': None,
                    'type': 'pattern3',
                    'gg': gg,
                    'risk_pips': risk,
                    'spread': current_candle['spread'],
                    'direction': 'bullish'
                })

                # Print progress after every 30 trades
                if (len(trades['bearish']) + len(trades['bullish'])) % 30 == 0:
                    print(f"Processed {len(trades['bearish']) + len(trades['bullish'])} trades...")
                
    return trades
