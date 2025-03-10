import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
# Import the charting module
from charting import create_chart, create_chart_with_fvgs_and_ggs, create_chart_with_ggs_and_touches, create_chart_with_ggs_and_trades
from helper import print_fvg_details
import argparse

from trade_finder import get_data, find_fvg_intersections, find_gg_reversal_pattern_trades







def calculate_trailing_sl(entry, sl, current_price, direction, strategy='r_based', high=None, low=None, rr_jumpstep=1.5):
    """
    Calculate the trailing stop loss level based on price movement.
    
    Args:
        entry (float): Entry price of the trade
        sl (float): Initial stop loss price
        current_price (float): Current price to check against
        direction (str): 'bullish' or 'bearish'
        strategy (str): 'r_based' or 'candle_based' - method for trailing stop loss
        high (float, optional): Current candle high price, required for candle_based
        low (float, optional): Current candle low price, required for candle_based
        rr_jumpstep (float, optional): Size of R-multiple steps for trailing stop loss.
        
    Returns:
        float: New stop loss level
    """
    initial_risk = abs(entry - sl)
    if initial_risk == 0:
        return sl
        
    if strategy == 'r_based':
        if direction == 'bullish':
            price_movement = current_price - entry
            r_multiple = price_movement / initial_risk
            # Use rr_jumpstep for R multiple steps
            if r_multiple >= rr_jumpstep:
                # Calculate how many rr_jumpstep multiples we've moved
                r_steps = int(r_multiple / rr_jumpstep)
                # New SL is (r_steps) * rr_jumpstep R above entry
                return entry + (r_steps * rr_jumpstep * initial_risk)
            return sl
        else:  # bearish
            price_movement = entry - current_price
            r_multiple = price_movement / initial_risk
            # Use rr_jumpstep for R multiple steps
            if r_multiple >= rr_jumpstep:
                # Calculate how many rr_jumpstep multiples we've moved
                r_steps = int(r_multiple / rr_jumpstep)
                # New SL is (r_steps) * rr_jumpstep R below entry
                return entry - (r_steps * rr_jumpstep * initial_risk)
            return sl
            
    else:  # candle_based
        if high is None or low is None:
            return sl
            
        if direction == 'bullish' and current_price > entry:
            new_sl = low
            return new_sl if new_sl > sl else sl
        elif direction == 'bearish' and current_price < entry:
            new_sl = high
            return new_sl if new_sl < sl else sl
        return sl

def analyze_trade_exits(trades, df, rr=None, stacking=1, use_trailing_sl=False, respect_tp_when_sl_trailing=True, trailing_sl_strategy='r_based'):
    """
    Analyze trades to determine exit prices, times and results based on price action.
    
    Args:
        trades (dict): Dictionary containing bearish and bullish trades
        df (pandas.DataFrame): Price data to analyze trades against
        rr (float, optional): Risk/reward ratio to override existing trade TPs
        stacking (int): Maximum number of concurrent trades allowed
        use_trailing_sl (bool): Whether to use trailing stop loss
        respect_tp_when_sl_trailing (bool): If True and trailing SL is on, respect original TP targets
        trailing_sl_strategy (str): 'r_based' or 'candle_based' - method for trailing stop loss
        
    Returns:
        dict: Updated trades dict with exit information
    """
    # Sort all trades by time
    all_trades = []
    for direction in ['bullish', 'bearish']:
        for trade in trades[direction]:
            trade['direction'] = direction
            all_trades.append(trade)
    all_trades.sort(key=lambda x: x['time'])
    
    # Track active trades
    active_trades = []
    filtered_trades = {'bullish': [], 'bearish': []}
    
    for trade in all_trades:
        direction = trade['direction']
        entry = trade['entry']
        initial_sl = trade['sl']
        trade_time = trade['time']
        current_sl = initial_sl
        sl_has_trailed = False
        
        # Initial SL is always at -1R
        initial_sl_rr = -1
        
        # Initialize trail_sl_history with initial SL and RR
        trail_sl_history = [(initial_sl_rr, trade_time, initial_sl)]
        
        # Remove closed trades from active list before checking stacking limit
        active_trades = [t for t in active_trades if t['exit_time'] is None or trade_time <= t['exit_time']]
        
        # Skip if max stacking reached
        if len(active_trades) >= stacking:
            continue
            
        # Get future price data
        future_data = df[df.index > trade_time]
        if len(future_data) == 0:
            continue
            
        # Set TP based on RR if provided, otherwise use trade's original TP
        risk = abs(entry - initial_sl)
        if rr is not None:
            tp = entry + (risk * rr) if direction == 'bullish' else entry - (risk * rr)
            trade['tp'] = tp
        else:
            tp = trade['tp']
        
        # Track trade progress
        exit_price = None
        exit_time = None
        status = 'open'  # Default status
        exit_rr = 0
        exit_type = None
        trailing_sl = None
        
        # Add trade to active trades before processing
        active_trades.append(trade)
        
        # Check each future candle
        for idx, row in future_data.iterrows():
            high, low = row['High'], row['Low']
            close = row['Close']
            
            # Get previous bar's data
            prev_idx = df.index.get_loc(idx) - 1
            if prev_idx >= 0:
                prev_high = df.iloc[prev_idx]['High']
                prev_low = df.iloc[prev_idx]['Low']
            else:
                prev_high = high
                prev_low = low
            
            # Check for TP hit first if respect_tp_when_sl_trailing is True or trailing SL is not used
            if respect_tp_when_sl_trailing or not use_trailing_sl:
                tp_hit = (direction == 'bullish' and high >= tp) or \
                        (direction == 'bearish' and low <= tp)
                
                if tp_hit:
                    exit_price = tp
                    exit_type = 'TP'
                    exit_time = idx
                    status = 'win'
                    exit_rr = (exit_price - entry) / risk if direction == 'bullish' else (entry - exit_price) / risk
                    break
            
            # Update trailing stop loss using previous bar's high/low
            if use_trailing_sl:
                if direction == 'bullish':
                    new_sl = calculate_trailing_sl(
                        entry, current_sl, prev_high, direction,
                        strategy=trailing_sl_strategy,
                        high=prev_high, low=prev_low
                    )
                    if new_sl > current_sl:
                        current_sl = new_sl
                        trailing_sl = new_sl
                        sl_has_trailed = True
                        current_rr = (new_sl - entry) / risk
                        trail_sl_history.append((current_rr, idx, new_sl))
                else:  # bearish
                    new_sl = calculate_trailing_sl(
                        entry, current_sl, prev_low, direction,
                        strategy=trailing_sl_strategy,
                        high=prev_high, low=prev_low
                    )
                    if new_sl < current_sl:
                        current_sl = new_sl
                        trailing_sl = new_sl
                        sl_has_trailed = True
                        current_rr = (entry - new_sl) / risk
                        trail_sl_history.append((current_rr, idx, new_sl))
            
            # Check for SL hit
            sl_hit = (direction == 'bullish' and low <= current_sl) or \
                    (direction == 'bearish' and high >= current_sl)
            
            if sl_hit:
                exit_price = current_sl
                exit_type = 'TSL' if sl_has_trailed else 'SL'
                exit_time = idx
                
                # Calculate R multiple based on direction
                exit_rr = (exit_price - entry) / risk if direction == 'bullish' else (entry - exit_price) / risk
                    
                # Determine trade status based on R multiple
                if exit_rr > 0:
                    status = 'win'
                elif exit_rr < 0:
                    status = 'loss' 
                elif exit_rr == 0:
                    status = 'BE'  # Breakeven
                else:
                    status = 'ERROR'
                break
        
        # If no exit found, use last price
        if exit_price is None:
            last_price = future_data.iloc[-1]['Close']
            exit_price = last_price
            exit_time = future_data.index[-1]
            if direction == 'bullish':
                exit_rr = (exit_price - entry) / risk
            else:
                exit_rr = (entry - exit_price) / risk
            
        # Add final SL level to history if it changed
        if trailing_sl and trail_sl_history[-1][2] != trailing_sl:
            final_rr = (trailing_sl - entry) / risk if direction == 'bullish' else (entry - trailing_sl) / risk
            trail_sl_history.append((final_rr, exit_time, trailing_sl))
        
        # Add exit info to trade
        trade['exit_price'] = exit_price
        trade['exit_time'] = exit_time
        trade['status'] = status
        trade['exit_rr'] = exit_rr
        trade['exit_type'] = exit_type
        if trailing_sl:
            trade['trailing_sl'] = trailing_sl
        trade['trail_sl_history'] = trail_sl_history
            
        # Add to filtered trades
        filtered_trades[direction].append(trade)
        
    return filtered_trades

def analyze_trade_metrics(trades, account_type='steady', initial_balance=1000, minimum_balance=1000, 
                   max_backup=9000, withdrawal_pct=20, withdrawal_step=5000, risk_percent=10,
                   equity_goal=None):
    """
    Analyze account performance with withdrawals and investments based on rules.
    Updates trade dictionaries with account metrics.
    
    Args:
        trades (dict): Dictionary of trades with exit info
        account_type (str): Type of account - 'steady' or 'dynamic'
        initial_balance (float): Starting account balance
        minimum_balance (float): Minimum balance to maintain
        max_backup (float): Maximum backup funds available for deposits
        withdrawal_pct (float): Withdrawal percentage at each step
        withdrawal_step (float): Balance threshold for withdrawals
        risk_percent (float): Risk percentage per trade
        equity_goal (float): Target equity level (optional)
        
    Returns:
        dict: Updated trades dictionary with account metrics
    """
    current_balance = initial_balance
    backup_remaining = max_backup
    total_withdrawn = 0
    total_invested = initial_balance
    
    # Calculate fixed risk amount for steady account type
    steady_risk = initial_balance * (risk_percent/100)
    
    # Process all trades chronologically
    all_trades = []
    for direction in ['bullish', 'bearish']:
        for trade in trades[direction]:
            trade['direction'] = direction
            all_trades.append(trade)
    
    all_trades.sort(key=lambda x: x['time'])
    
    for trade in all_trades:
        balance_before = current_balance
        trade['balance_before'] = balance_before
        
        # Check if investment needed
        investment = 0
        if current_balance < minimum_balance and backup_remaining > 0:
            needed = minimum_balance - current_balance
            investment = min(needed, backup_remaining)
            current_balance += investment
            backup_remaining -= investment
            total_invested += investment
            balance_before = current_balance
            trade['balance_before'] = balance_before
            
        # Calculate risk amount based on account type
        if account_type == 'steady':
            risk_amount = steady_risk
        else:  # dynamic
            risk_amount = balance_before * (risk_percent/100)
            
        trade['risk_amount'] = risk_amount
            
        # Calculate profit based on risk amount and trade RR
        exit_rr = trade.get('exit_rr', 0)
        profit = risk_amount * exit_rr
        trade['profit'] = profit
        
        balance_after = balance_before + profit
        current_balance = balance_after
        trade['balance_after'] = balance_after
        
        # Calculate withdrawal if applicable
        withdrawal = 0
        if balance_after >= withdrawal_step:
            steps = int(balance_after / withdrawal_step)
            withdrawal_amount = steps * (withdrawal_pct/100 * withdrawal_step)
            
            # Only withdraw if it won't put balance below minimum
            if (balance_after - withdrawal_amount) >= minimum_balance:
                withdrawal = withdrawal_amount
                current_balance -= withdrawal
                total_withdrawn += withdrawal
                balance_after = current_balance
                trade['balance_after'] = balance_after
        
        trade['withdrawal'] = withdrawal
        trade['total_withdrawn'] = total_withdrawn
        trade['investment'] = investment
        trade['total_invested'] = total_invested
        trade['total_equity'] = current_balance + total_withdrawn
        
        # Check if equity goal reached
        if equity_goal and (current_balance + total_withdrawn) >= equity_goal:
            break
            
    # Reorganize trades back into direction-based dictionary
    analyzed_trades = {'bullish': [], 'bearish': []}
    for trade in all_trades:
        direction = trade.pop('direction')  # Remove direction key
        analyzed_trades[direction].append(trade)
            
    return analyzed_trades

def calculate_account_metrics(analyzed_trades):
    """
    Calculate comprehensive trading metrics from analyzed trades
    
    Args:
        analyzed_trades (dict): Dictionary containing analyzed bullish and bearish trades
        
    Returns:
        dict: Dictionary containing calculated metrics
    """
    metrics = {
        'total_trades': 0,
        'winning_trades': 0, 
        'losing_trades': 0,
        'win_rate': 0,
        'total_profit': 0,
        'total_loss': 0,
        'net_profit': 0,
        'profit_factor': 0,
        'avg_win': 0,
        'avg_loss': 0,
        'largest_win': 0,
        'largest_loss': 0,
        'final_balance': 0,
        'max_equity': 0,
        'total_withdrawal': 0,
        'max_withdrawal': 0,
        'total_investment': 0,
        'max_drawdown': 0,
        'total_profit_bullish': 0,
        'total_profit_bearish': 0,
        'avg_profit_per_trade': 0,
        'consecutive_wins': 0,
        'consecutive_losses': 0,
        'max_consecutive_wins': 0,
        'max_consecutive_losses': 0,
        'total_return': 0
    }
    
    winning_amounts = []
    losing_amounts = []
    equity_curve = []
    current_consecutive = 0
    
    # Process trades by direction
    for direction in ['bullish', 'bearish']:
        for trade in analyzed_trades[direction]:
            if 'profit' not in trade:
                continue
                
            profit = trade['profit']
            metrics['total_trades'] += 1
            equity_curve.append(profit)
            
            # Track balance and equity metrics
            if 'balance_after' in trade:
                metrics['final_balance'] = trade['balance_after']
                metrics['max_equity'] = max(metrics['max_equity'], trade['total_equity'])
            
            # Track withdrawals and investments
            if 'withdrawal' in trade:
                metrics['total_withdrawal'] += trade['withdrawal']
                metrics['max_withdrawal'] = max(metrics['max_withdrawal'], trade['withdrawal'])
            
            if 'investment' in trade:
                metrics['total_investment'] += trade['investment']
            
            # Track wins/losses
            if profit > 0:
                metrics['winning_trades'] += 1
                winning_amounts.append(profit)
                metrics['largest_win'] = max(metrics['largest_win'], profit)
                metrics['total_profit'] += profit
                current_consecutive = max(0, current_consecutive + 1)
                metrics['max_consecutive_wins'] = max(metrics['max_consecutive_wins'], current_consecutive)
            elif profit < 0:
                metrics['losing_trades'] += 1
                losing_amounts.append(profit)
                metrics['largest_loss'] = min(metrics['largest_loss'], profit)
                metrics['total_loss'] += profit
                current_consecutive = min(0, current_consecutive - 1)
                metrics['max_consecutive_losses'] = max(metrics['max_consecutive_losses'], abs(current_consecutive))
            
            # Track direction profits
            if direction == 'bullish':
                metrics['total_profit_bullish'] += profit
            else:
                metrics['total_profit_bearish'] += profit
    
    # Calculate win rate
    if metrics['total_trades'] > 0:
        metrics['win_rate'] = (metrics['winning_trades'] / metrics['total_trades']) * 100
        
    # Calculate averages
    if winning_amounts:
        metrics['avg_win'] = sum(winning_amounts) / len(winning_amounts)
    if losing_amounts:
        metrics['avg_loss'] = sum(losing_amounts) / len(losing_amounts)
        
    # Calculate profit metrics
    metrics['net_profit'] = metrics['total_profit'] + metrics['total_loss']
    if metrics['total_trades'] > 0:
        metrics['avg_profit_per_trade'] = metrics['net_profit'] / metrics['total_trades']
    
    # Calculate profit factor
    if abs(metrics['total_loss']) > 0:
        metrics['profit_factor'] = abs(metrics['total_profit'] / metrics['total_loss'])
        
    # Calculate max drawdown
    peak = 0
    drawdown = 0
    running_total = 0
    for profit in equity_curve:
        running_total += profit
        if running_total > peak:
            peak = running_total
        drawdown = min(drawdown, running_total - peak)
    metrics['max_drawdown'] = abs(drawdown)
    
    # Calculate total return
    if metrics['total_investment'] > 0:
        metrics['total_return'] = ((metrics['final_balance'] + metrics['total_withdrawal'] - metrics['total_investment']) / metrics['total_investment']) * 100
    
    metrics['consecutive_wins'] = max(0, current_consecutive)
    metrics['consecutive_losses'] = abs(min(0, current_consecutive))
    
    return metrics

def create_trades_df(analyzed_trades):
    """
    Convert analyzed trades dictionary into a pandas DataFrame.
    
    Args:
        analyzed_trades (dict): Dictionary containing analyzed trades data
        
    Returns:
        pandas.DataFrame: DataFrame containing all trade information
    """
    import pandas as pd
    
    # Combine bullish and bearish trades into single list
    all_trades = []
    for direction in ['bullish', 'bearish']:
        for trade in analyzed_trades[direction]:
            # Add direction to trade dict
            trade_data = trade.copy()
            trade_data['direction'] = direction
            all_trades.append(trade_data)
            
    # Convert to DataFrame
    df = pd.DataFrame(all_trades)
    
    # Sort by trade entry time
    df = df.sort_values('time')
    
    # Calculate additional metrics
    df['duration'] = df['exit_time'] - df['time']
    df['pips'] = df.apply(lambda x: 
        (x['exit_price'] - x['entry']) if x['direction'] == 'bullish' 
        else (x['entry'] - x['exit_price']), axis=1)
    
    # Format specified numeric values in comma separated money format with 2 decimal places
    money_columns = ['balance_before', 'risk_amount', 'profit', 'balance_after', 
                     'withdrawal', 'total_withdrawn', 'investment', 
                     'total_invested', 'total_equity']  # Columns to format
    for col in money_columns:
        if col in df.columns:  # Check if the column exists in the DataFrame
            df[col] = df[col].apply(lambda x: f"${'{:,.2f}'.format(x)}")
    
    return df



def analyze_trading_results(trades, df, account_type='steady', initial_balance=1000, minimum_balance=1000,
                          max_backup=9000, withdrawal_pct=20, withdrawal_step=5000, risk_percent=10,
                          equity_goal=None, rr=None, stacking=1, use_trailing_sl=False, respect_tp_when_sl_trailing=False):
    """
    Master function to analyze trades using all analysis functions.
    
    Args:
        trades (dict): Dictionary containing bearish and bullish trades
        df (pandas.DataFrame): Price data to analyze trades against
        account_type (str): Type of account - 'steady' or 'dynamic'
        initial_balance (float): Starting account balance
        minimum_balance (float): Minimum balance to maintain
        max_backup (float): Maximum backup funds available
        withdrawal_pct (float): Withdrawal percentage at each step
        withdrawal_step (float): Balance threshold for withdrawals
        risk_percent (float): Risk percentage per trade
        equity_goal (float): Target equity level (optional)
        rr (float): Risk/reward ratio
        stacking (int): Maximum concurrent trades allowed
        use_trailing_sl (bool): Whether to use trailing stop loss
        respect_tp_when_sl_trailing (bool): Respect TP when trailing SL is on
        
    Returns:
        dict: Combined results containing all metrics
    """
    # First analyze raw trade performance
    filtered_trades = analyze_trade_exits(
        trades, df, rr, stacking, use_trailing_sl, 
        respect_tp_when_sl_trailing
    )
    
    # Then analyze account metrics with withdrawals/investments
    analyzed_trades = analyze_trade_metrics(
        filtered_trades, account_type, initial_balance,
        minimum_balance, max_backup, withdrawal_pct,
        withdrawal_step, risk_percent, equity_goal
    )
    
    # Calculate final performance metrics
    account_metrics = calculate_account_metrics(analyzed_trades)

    trades_df = create_trades_df(analyzed_trades)

    # Combine all results into one dictionary
    combined_results = {
        'analyzed_trades': analyzed_trades,
        'account_metrics': account_metrics,
        'trades_df': trades_df
    }
    
    return combined_results



def main():
    # Set up parameters
    symbol = "^GSPC"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=59)  # Reduced to 7 days due to 5min data limitations
    
    # Add argument parsing for risk/reward ratio and chart display
    parser = argparse.ArgumentParser()
    parser.add_argument('--rr', type=float, help='Risk/reward ratio for trades')
    parser.add_argument('--chart', action='store_true', help='Display chart visualization')
    parser.add_argument('--respect_tp', action='store_true', help='Respect TP when trailing SL is active')

    args = parser.parse_args()
    
    print(f"Fetching data for {symbol} from {start_date} to {end_date}")
    
    # Get data
    df_1h, df_5m, fvgs_1h, fvgs_5m = get_data(symbol, start_date, end_date)
    
    # Print information about the FVGs
    print(f"\n1h Bullish FVGs: {len(fvgs_1h[0])}, Active: {sum(1 for fvg in fvgs_1h[0] if fvg['active'])}")
    print(f"1h Bearish FVGs: {len(fvgs_1h[1])}, Active: {sum(1 for fvg in fvgs_1h[1] if fvg['active'])}")
    print(f"5m Bullish FVGs: {len(fvgs_5m[0])}, Active: {sum(1 for fvg in fvgs_5m[0] if fvg['active'])}")
    print(f"5m Bearish FVGs: {len(fvgs_5m[1])}, Active: {sum(1 for fvg in fvgs_5m[1] if fvg['active'])}")
    
    
    bullish_ggs, bearish_ggs = find_fvg_intersections(fvgs_1h, fvgs_5m)

    # Use command line RR if provided, otherwise default to 2
    risk_reward = args.rr if args.rr is not None else 3
    trades = find_gg_reversal_pattern_trades(df_5m, bullish_ggs, bearish_ggs, risk_reward=risk_reward)

    full_results = analyze_trading_results(
        trades, 
        df_5m, 
        rr=args.rr, 
        initial_balance=1000, 
        stacking=1, 
        use_trailing_sl=True,
        respect_tp_when_sl_trailing=args.respect_tp if args.respect_tp is not None else False,
        account_type='dynamic',
        minimum_balance=1000,
        max_backup=9000,
        withdrawal_pct=20,
        withdrawal_step=5000,
        risk_percent=50,
        equity_goal=None
    )

    trades_df = full_results['trades_df']
    results = full_results['account_metrics']
    final_trades = full_results['analyzed_trades']

    # trades_df = trades_df.sort_values()
    
    excluded_columns = ['gg', 'type', 'rr']
    pd.set_option('display.max_rows', None)
    print(trades_df[[col for col in trades_df.columns if col not in excluded_columns]])
    pd.reset_option('display.max_rows')

    # Export trades to CSV, excluding certain columns
    if not trades_df.empty:
        output_columns = [col for col in trades_df.columns if col not in excluded_columns]
        trades_df[output_columns].to_csv('trades.csv', index=False)
        print("\nTrades exported to trades.csv")
    

    print("\nTrade Analysis Results:")
    print(f"  Total Trades: {results['total_trades']}")
    print(f"  Winning Trades: {results['winning_trades']}")
    print(f"  Losing Trades: {results['losing_trades']}")
    print(f"  Win Rate: {results['win_rate']:.2f}%")
    print(f"  Total Profit: ${results['total_profit']:.2f}")
    print(f"  Total Loss: ${results['total_loss']:.2f}")
    print(f"  Net Profit: ${results['net_profit']:.2f}")
    print(f"  Profit Factor: {results['profit_factor']:.2f}")
    print(f"  Average Win: ${results['avg_win']:.2f}")
    print(f"  Average Loss: ${results['avg_loss']:.2f}")
    print(f"  Largest Win: ${results['largest_win']:.2f}")
    print(f"  Largest Loss: ${results['largest_loss']:.2f}")
    print(f"  Total Bullish Profit: ${results['total_profit_bullish']:.2f}")
    print(f"  Total Bearish Profit: ${results['total_profit_bearish']:.2f}")
    print(f"  Final Balance: ${results['final_balance']:.2f}")
    print(f"  Total Return: {results['total_return']:.2f}%")

    print(f"  Total Withdrawal: ${results['total_withdrawal']:.2f}")
    print(f"  Total Investment: ${results['total_investment']:.2f}")
    print(f"  Max Drawdown: ${results['max_drawdown']:.2f}")
    print(f"  Max Equity: ${results['max_equity']:.2f}")
    print(f"  Consecutive Wins: {results['consecutive_wins']}")
    print(f"  Consecutive Losses: {results['consecutive_losses']}")
    print(f"  Max Consecutive Wins: {results['max_consecutive_wins']}")
    print(f"  Max Consecutive Losses: {results['max_consecutive_losses']}")
    print(f"  Max Withdrawal: ${results['max_withdrawal']:.2f}")
    


    # original_trades = analyze_trade_exits(
    #     trades, 
    #     df_5m,
    #     rr=args.rr,
    #     stacking=1
    # )

    if args.chart:
        create_chart_with_ggs_and_trades(df_5m, bullish_ggs, bearish_ggs, final_trades)

    #create_chart(df_5m, bullish_ggs, bearish_ggs)

    # print_fvg_details(fvgs_5m)
    

if __name__ == "__main__":
    main()
