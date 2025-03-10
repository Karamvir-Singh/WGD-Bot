


def print_fvg_details(fvgs_5m):
    # Print detailed information about 5-minute FVGs
    print("\n=== 5-Minute Fair Value Gaps Details ===")

    valid_bullish_count = 0
    print("\nBullish 5m FVGs:")
    for i, fvg in enumerate(fvgs_5m[0], 1):
        if fvg['end_time'] > fvg['start_time']:
            valid_bullish_count += 1
            print(f"FVG #{valid_bullish_count}:")
            print(f"  Start Time: {fvg['start_time']}")
            print(f"  End Time: {fvg['end_time']}")
            print(f"  High: {fvg['high']:.2f}")
            print(f"  Low: {fvg['low']:.2f}")
            print(f"  Active: {fvg['active']}")
            print("---")
    print(f"Total valid bullish FVGs: {valid_bullish_count}")

    valid_bearish_count = 0
    print("\nBearish 5m FVGs:")
    for i, fvg in enumerate(fvgs_5m[1], 1):
        if fvg['end_time'] > fvg['start_time']:
            valid_bearish_count += 1
            print(f"FVG #{valid_bearish_count}:")
            print(f"  Start Time: {fvg['start_time']}")
            print(f"  End Time: {fvg['end_time']}")
            print(f"  High: {fvg['high']:.2f}")
            print(f"  Low: {fvg['low']:.2f}")
            print(f"  Active: {fvg['active']}")
            print("---")
    print(f"Total valid bearish FVGs: {valid_bearish_count}")   

###### BACKUP FOR INITIAL TRADE ANALYSER AND SUPPORTING FUNCTIONS  #########

# def analyze_trades(trades, df, risk_per_trade=100, rr=2, risk_type='static', starting_account_size=1000, stacking=1, use_trailing_sl=False, ignore_tp_when_sl_trailing=False):
#     """
#     Analyze trades to calculate win rate and profit by checking against price data
    
#     Args:
#         trades (dict): Dictionary containing bearish and bullish trades
#         df (pandas.DataFrame): Price data to analyze trades against
#         risk_per_trade (float): Fixed risk amount per trade in dollars/points
#         rr (float, optional): Risk/reward ratio to override existing trade TPs
#         risk_type (str): Type of risk calculation - 'static' or 'percentage'
#         starting_account_size (float): Initial account balance for percentage risk calculations
#         stacking (int): Maximum number of concurrent trades allowed
#         use_trailing_sl (bool): Whether to use trailing stop loss
#         ignore_tp_when_sl_trailing (bool): If True and trailing SL is on, ignore TP and only exit on trailing SL
        
#     Returns:
#         tuple: (results dict containing metrics, updated trades dict)
#     """
#     # Track running account balance for percentage risk
#     current_balance = starting_account_size
    
#     # Keep track of active trades
#     active_trades = []
#     filtered_trades = {'bullish': [], 'bearish': []}

#     # Sort all trades by time
#     all_trades = []
#     for direction in ['bullish', 'bearish']:
#         for trade in trades[direction]:
#             trade['direction'] = direction
#             all_trades.append(trade)
#     all_trades.sort(key=lambda x: x['time'])

#     # Get last available price for open trades
#     last_price = df['Close'].iloc[-1]
#     last_time = df.index[-1]

#     # Process trades in chronological order
#     for trade in all_trades:
#         direction = trade['direction']
#         entry = trade['entry']
#         initial_sl = trade['sl']
#         trade_time = trade['time']
#         current_sl = initial_sl
        
#         # Get future price data after trade entry
#         future_data = df[df.index > trade_time]
#         if len(future_data) == 0:
#             continue

#         # First calculate exit time for this trade
#         if rr is not None:
#             if direction == 'bullish':
#                 tp = entry + (abs(entry - initial_sl) * rr)
#             else:
#                 tp = entry - (abs(entry - initial_sl) * rr)
#         else:
#             tp = trade['tp']
            
#         sl_hit = False
#         tp_hit = False
#         exit_time = None
#         exit_price = None
#         highest_reached = entry if direction == 'bullish' else entry
#         lowest_reached = entry if direction == 'bearish' else entry
        
#         for idx, row in future_data.iterrows():
#             # Update highest/lowest prices reached for trailing SL calculation
#             if use_trailing_sl:
#                 if direction == 'bullish':
#                     highest_reached = max(highest_reached, row['High'])
#                     current_sl = calculate_trailing_sl(entry, initial_sl, highest_reached, direction)
#                 else:
#                     lowest_reached = min(lowest_reached, row['Low'])
#                     current_sl = calculate_trailing_sl(entry, initial_sl, lowest_reached, direction)
            
#             # Check for exits
#             if direction == 'bullish':
#                 if row['Low'] <= current_sl:
#                     sl_hit = True
#                     exit_time = idx
#                     exit_price = current_sl
#                     break
#                 if not (use_trailing_sl and ignore_tp_when_sl_trailing) and row['High'] >= tp:
#                     tp_hit = True
#                     exit_time = idx
#                     exit_price = tp
#                     break
#             else:
#                 if row['High'] >= current_sl:
#                     sl_hit = True
#                     exit_time = idx
#                     exit_price = current_sl
#                     break
#                 if not (use_trailing_sl and ignore_tp_when_sl_trailing) and row['Low'] <= tp:
#                     tp_hit = True
#                     exit_time = idx
#                     exit_price = tp
#                     break

#         # If no exit found, trade is still open - use current price
#         if exit_time is None:
#             exit_time = last_time
#             exit_price = last_price
#             trade['status'] = 'open'
#         else:
#             trade['status'] = 'closed'

#         # Remove closed trades from active_trades
#         active_trades = [t for t in active_trades if t['exit_time'] is None or trade_time <= t['exit_time']]
        
#         # Skip if we already have max stacked trades
#         if len(active_trades) >= stacking:
#             continue
            
#         # Add trade to filtered list and active trades
#         filtered_trades[direction].append(trade)
        
#         # Update trade info
#         trade['exit_time'] = exit_time
#         trade['exit_price'] = exit_price
#         trade['initial_sl'] = initial_sl
#         trade['final_sl'] = current_sl
#         trade['rr'] = abs(tp - entry) / abs(entry - initial_sl) if sl_hit or tp_hit else None
#         trade['exit_type'] = 'tp' if tp_hit else 'sl' if sl_hit else 'open'
#         trade['balance_before'] = current_balance

#         # Calculate trade result and update balances
#         result = calculate_trade_result(trade, direction, risk_per_trade, current_balance, risk_type)
#         trade.update(result)
        
#         # Determine win/loss based on profit
#         trade['win_loss'] = 'win' if result['profit'] > 0 else 'loss' if result['profit'] < 0 else None
        
#         current_balance = result['balance_after']

#         # Add to active trades after updating info
#         active_trades.append(trade)

#     return calculate_account_metrics(filtered_trades, risk_per_trade, risk_type, starting_account_size)



# def calculate_account_metrics(trades, risk_per_trade, risk_type, starting_account_size):
#     """
#     Calculate trade metrics based on updated trades
    
#     Args:
#         trades (dict): Dictionary containing updated trades with exit information
#         risk_per_trade (float): Fixed risk amount per trade in dollars/points or percentage
#         risk_type (str): Type of risk calculation - 'static' or 'percentage'
#         starting_account_size (float): Initial account balance for percentage calculations
        
#     Returns:
#         tuple: (results dict containing metrics, trades dict)
#     """
#     results = {
#         'total_trades': 0,
#         'winning_trades': 0,
#         'losing_trades': 0,
#         'win_rate': 0,
#         'total_profit': 0,  # Total of winning trades
#         'total_loss': 0,    # Total of losing trades (negative number)
#         'net_profit': 0,    # Total profit + total loss
#         'profit_factor': 0,
#         'avg_win': 0,
#         'avg_loss': 0,
#         'largest_win': 0,
#         'largest_loss': 0,
#         'total_profit_bullish': 0,
#         'total_profit_bearish': 0,
#         'final_balance': starting_account_size,  # Initialize with starting balance
#         'total_return': 0
#     }
    
#     winning_amounts = []
#     losing_amounts = []
#     current_balance = starting_account_size
    
#     for direction in ['bullish', 'bearish']:
#         for trade in trades[direction]:
#             if 'win_loss' not in trade or trade['win_loss'] is None:
#                 continue
                
#             results['total_trades'] += 1
#             profit = trade['profit']
            
#             if profit > 0:
#                 results['winning_trades'] += 1
#                 winning_amounts.append(profit)
#                 results['largest_win'] = max(results['largest_win'], profit)
#                 results['total_profit'] += profit
#             elif profit < 0:
#                 results['losing_trades'] += 1
#                 losing_amounts.append(profit)
#                 results['largest_loss'] = min(results['largest_loss'], profit)
#                 results['total_loss'] += profit
                
#             if direction == 'bullish':
#                 results['total_profit_bullish'] += profit
#             else:
#                 results['total_profit_bearish'] += profit
            
#             current_balance = trade['balance_after']
    
#     # Calculate net profit
#     results['net_profit'] = results['total_profit'] + results['total_loss']
    
#     # Calculate final metrics
#     if results['total_trades'] > 0:
#         results['win_rate'] = (results['winning_trades'] / results['total_trades']) * 100
        
#     if winning_amounts:
#         results['avg_win'] = sum(winning_amounts) / len(winning_amounts)
        
#     if losing_amounts:
#         results['avg_loss'] = sum(losing_amounts) / len(losing_amounts)
        
#     if abs(results['total_loss']) > 0:
#         results['profit_factor'] = abs(results['total_profit'] / results['total_loss'])
    
#     # Update final balance and return regardless of risk type
#     results['final_balance'] = current_balance
#     results['total_return'] = ((current_balance - starting_account_size) / starting_account_size) * 100
    
#     return results, trades




# def calculate_trade_result(trade, direction, risk_per_trade, current_balance, risk_type, max_trail_rr=15):
#     """
#     Calculate the result of a single trade including position size, profit/loss, and balance updates.
    
#     Args:
#         trade (dict): Trade information including entry, sl, etc.
#         direction (str): 'bullish' or 'bearish'
#         risk_per_trade (float): Fixed risk amount per trade in dollars/points or percentage
#         current_balance (float): Current account balance
#         risk_type (str): Type of risk calculation - 'static' or 'percentage'
#         max_trail_rr (float): Maximum risk/reward ratio for trailing stop loss (default 1.5)
        
#     Returns:
#         dict: Dictionary containing trade results including profit, position size, and balance updates
#     """
#     risk = abs(trade['entry'] - trade['sl'])
    
#     # Calculate position size based on risk type
#     if risk_type == 'static':
#         position_size = risk_per_trade / risk
#     else:  # percentage
#         risk_amount = current_balance * (risk_per_trade / 100)
#         position_size = risk_amount / risk
    
#     # Calculate profit based on direction
#     if direction == 'bullish':
#         profit = (trade['exit_price'] - trade['entry']) * position_size
#         # Calculate actual exit RR
#         trade['exit_rr'] = (trade['exit_price'] - trade['entry']) / risk
#         # Limit trailing stop loss to max_trail_rr
#         if trade.get('trailing_sl'):
#             max_sl = trade['entry'] + (risk * max_trail_rr)
#             trade['trailing_sl'] = min(trade['trailing_sl'], max_sl)
#     else:
#         profit = (trade['entry'] - trade['exit_price']) * position_size
#         # Calculate actual exit RR
#         trade['exit_rr'] = (trade['entry'] - trade['exit_price']) / risk
#         # Limit trailing stop loss to max_trail_rr
#         if trade.get('trailing_sl'):
#             max_sl = trade['entry'] - (risk * max_trail_rr)
#             trade['trailing_sl'] = max(trade['trailing_sl'], max_sl)
    
#     # Update trade information
#     result = {
#         'profit': profit,
#         'balance_after': current_balance + profit,
#         'position_size': position_size
#     }
    
#     return result








############# BACKUP FOR INITIAL TRADE FINDER AND GG TOUCH FINDER ############



# def find_trading_opportunities(df_5m, fvgs_1h, fvgs_5m):
    
#     """Find trading opportunities when price interacts with GG areas"""
#     trades = []
    
#     # Get GG areas (intersections)
#     bullish_ggs, bearish_ggs = find_fvg_intersections(fvgs_1h, fvgs_5m)
    
#     # Track price interactions with GGs
#     for i in range(len(df_5m)):
#         current_time = df_5m.index[i]
#         high = df_5m['High'].iloc[i]
#         low = df_5m['Low'].iloc[i]
#         close = df_5m['Close'].iloc[i]
        
#         # Check bullish GGs
#         for gg in bullish_ggs:
#             if not gg['active'] or current_time < gg['start_time']:
#                 continue
                
#             # Price entering bullish GG from above
#             if low <= gg['high'] and not hasattr(gg, 'touched'):
#                 gg['touched'] = True
#                 gg['lowest_point'] = low
            
#             # Update lowest point while price is in GG
#             elif hasattr(gg, 'touched') and low <= gg['high']:
#                 gg['lowest_point'] = min(gg['lowest_point'], low)
            
#             # Price exiting GG upward after touching
#             elif hasattr(gg, 'touched') and close > gg['high']:
#                 entry = close
#                 stop_loss = gg['lowest_point']
#                 target = entry + 2 * (entry - stop_loss)
                
#                 trades.append({
#                     'timestamp': current_time,
#                     'type': 'bull',
#                     'entry': entry,
#                     'stop_loss': stop_loss,
#                     'target': target,
#                     'gg_high': gg['high'],
#                     'gg_low': gg['low']
#                 })
                
#                 gg['active'] = False
        
#         # Check bearish GGs
#         for gg in bearish_ggs:
#             if not gg['active'] or current_time < gg['start_time']:
#                 continue
                
#             # Price entering bearish GG from below
#             if high >= gg['low'] and not hasattr(gg, 'touched'):
#                 gg['touched'] = True
#                 gg['highest_point'] = high
            
#             # Update lowest point while price is in GG
#             elif hasattr(gg, 'touched') and high >= gg['low']:
#                 gg['highest_point'] = max(gg['highest_point'], high)
            
#             # Price exiting GG downward after touching
#             elif hasattr(gg, 'touched') and close < gg['low']:
#                 entry = close
#                 stop_loss = gg['highest_point']
#                 target = entry - 2 * (stop_loss - entry)
                
#                 trades.append({
#                     'timestamp': current_time,
#                     'type': 'bear',
#                     'entry': entry,
#                     'stop_loss': stop_loss,
#                     'target': target,
#                     'gg_high': gg['high'],
#                     'gg_low': gg['low']
#                 })
                
#                 gg['active'] = False
    
#     return pd.DataFrame(trades)


# def get_gg_touch_times(df, bullish_ggs, bearish_ggs):
#     """
#     Find timestamps when Golden Gaps (GGs) were touched by price action.
    
#     Args:
#         df (pandas.DataFrame): Price data with OHLC columns
#         bullish_ggs (list): List of bullish Golden Gaps
#         bearish_ggs (list): List of bearish Golden Gaps
        
#     Returns:
#         dict: Dictionary with timestamps when each GG was touched
#     """
#     touch_times = {
#         'bullish': [],
#         'bearish': []
#     }
    
#     # Check each candle for touches
#     for i in range(len(df)):
#         current_time = df.index[i]
#         high = df['High'].iloc[i]
#         low = df['Low'].iloc[i]
        
#         # Check bullish GGs
#         for gg in bullish_ggs:
#             if (current_time > gg['start_time'] and 
#                 low <= gg['high'] and current_time < gg['end_time'] and
#                 not any(t['timestamp'] == current_time and t['gg'] == gg for t in touch_times['bullish'])):
#                 touch_times['bullish'].append({
#                     'timestamp': current_time,
#                     'gg': gg,
#                     'price': low
#                 })
                
#         # Check bearish GGs
#         for gg in bearish_ggs:
#             if (current_time > gg['start_time'] and 
#                 high >= gg['low'] and
#                 not any(t['timestamp'] == current_time and t['gg'] == gg for t in touch_times['bearish'])):
#                 touch_times['bearish'].append({
#                     'timestamp': current_time,
#                     'gg': gg,
#                     'price': high
#                 })
                
#     return touch_times


# def convert_trades_to_dataframe(trades):
#     """Convert trades dictionary to a pandas DataFrame"""
#     trades_df = pd.DataFrame()
    
#     for direction in ['bullish', 'bearish']:
#         if trades[direction]:
#             direction_df = pd.DataFrame(trades[direction])
#             direction_df['direction'] = direction
#             trades_df = pd.concat([trades_df, direction_df])
    
#     if not trades_df.empty:
#         trades_df = trades_df.reset_index(drop=True)
#         # trades_df = trades_df.sort_values('time')

#     return trades_df
