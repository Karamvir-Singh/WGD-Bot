from lightweight_charts import Chart
import pandas as pd

def create_chart(df, bullish_ggs, bearish_ggs):
    """
    Create and display a chart with Golden Gap (GG) visualizations
    
    Args:
        df (pandas.DataFrame): DataFrame containing price data
        bullish_ggs (list): List of bullish Golden Gaps
        bearish_ggs (list): List of bearish Golden Gaps
    """
    print(f"\nBullish GGs found: {len(bullish_ggs)}")
    
    # Print the keys of the first GG to understand its structure
    if bullish_ggs:
        print("Keys in bullish GG:", list(bullish_ggs[0].keys()))
    if bearish_ggs:
        print("Keys in bearish GG:", list(bearish_ggs[0].keys()))
    
    for i, gg in enumerate(bullish_ggs[:5]):  # Limit to first 5 to avoid too much output
        print(f"  Bullish GG #{i+1}: Range {gg['low']:.2f}-{gg['high']:.2f}, Active: {gg['active']}")
    
    print(f"\nBearish GGs found: {len(bearish_ggs)}")
    for i, gg in enumerate(bearish_ggs[:5]):  # Limit to first 5 to avoid too much output
        print(f"  Bearish GG #{i+1}: Range {gg['low']:.2f}-{gg['high']:.2f}, Active: {gg['active']}")
    
    print("\nCreating chart...")
    # Create chart with larger dimensions for better visibility
    chart = Chart(width=1200, height=800)
    print("Setting data to chart...")
    chart.set(df)
    
    print("Adding GG visualizations...")
    # Calculate end time for boxes (10 bars after start time)
    def calculate_end_time(start_time, df, bars=10):
        start_idx = df.index.get_loc(start_time)
        end_idx = min(start_idx + bars, len(df) - 1)
        return df.index[end_idx]
    
    # Add visualization for bullish GGs
    active_bullish_count = 0
    for i, gg in enumerate(bullish_ggs):
        if gg['active']:
            active_bullish_count += 1
            try:
                chart.box(
                    start_time=gg['start_time'],
                    end_time=gg['end_time'],
                    start_value=gg['low'],
                    end_value=gg['high'],
                    color='rgba(76, 175, 80, 0.5)',
                    fill_color='rgba(76, 175, 80, 0.1)'
                )
            
            except (KeyError, ValueError) as e:
                print(f"Error drawing bullish GG visualization: {e}")
    
    print(f"Added {active_bullish_count} active bullish GGs to chart")
    
    # Add visualization for bearish GGs
    active_bearish_count = 0
    for i, gg in enumerate(bearish_ggs):
        if gg['active']:
            active_bearish_count += 1
            try:
                chart.box(
                    start_time=gg['start_time'],
                    end_time=gg['end_time'],
                    start_value=gg['low'],
                    end_value=gg['high'],
                    color='rgba(244, 67, 54, 0.5)',
                    fill_color='rgba(244, 67, 54, 0.1)'
                )

            except (KeyError, ValueError) as e:
                print(f"Error drawing bearish GG visualization: {e}")
    
    print(f"Added {active_bearish_count} active bearish GGs to chart")
    print("Displaying chart... (This may open in a new window)")
    
    # Show the chart with block=True to ensure it stays open
    print("Calling chart.show() with block=True...")
    chart.show(block=True)
    print("Chart display completed")

def create_chart_with_ggs_and_touches(df, bullish_ggs, bearish_ggs, touch_times, active_only=True):
    """
    Create and display a chart with Golden Gap (GG) visualizations and touch points
    
    Args:
        df (pandas.DataFrame): DataFrame containing price data
        bullish_ggs (list): List of bullish Golden Gaps
        bearish_ggs (list): List of bearish Golden Gaps
        touch_times (dict): Dictionary containing touch points for GGs
        active_only (bool): If True, only show active GGs. If False, show all GGs including inactive ones
    """
    print("\nCreating chart with GGs and touch points...")
    
    # Create chart with larger dimensions for better visibility
    chart = Chart(width=1200, height=800)
    chart.set(df)
    
    # Add visualization for bullish GGs
    bullish_count = 0
    for gg in bullish_ggs:
        if not active_only or (active_only and gg['active']):
            bullish_count += 1
            try:
                # Use more transparent colors for inactive GGs
                fill_opacity = '0.1' if gg['active'] else '0.05'
                border_opacity = '0.5' if gg['active'] else '0.25'
                
                chart.box(
                    start_time=gg['start_time'],
                    end_time=gg['end_time'], 
                    start_value=gg['low'],
                    end_value=gg['high'],
                    color=f'rgba(76, 175, 80, {border_opacity})',
                    fill_color=f'rgba(76, 175, 80, {fill_opacity})'
                )
            except (KeyError, ValueError) as e:
                print(f"Error drawing bullish GG visualization: {e}")
    
    status = "active" if active_only else "total"
    print(f"Added {bullish_count} {status} bullish GGs to chart")
    
    # Add visualization for bearish GGs
    bearish_count = 0
    for gg in bearish_ggs:
        if not active_only or (active_only and gg['active']):
            bearish_count += 1
            try:
                # Use more transparent colors for inactive GGs
                fill_opacity = '0.1' if gg['active'] else '0.05'
                border_opacity = '0.5' if gg['active'] else '0.25'
                
                chart.box(
                    start_time=gg['start_time'],
                    end_time=gg['end_time'],
                    start_value=gg['low'],
                    end_value=gg['high'],
                    color=f'rgba(244, 67, 54, {border_opacity})',
                    fill_color=f'rgba(244, 67, 54, {fill_opacity})'
                )
            except (KeyError, ValueError) as e:
                print(f"Error drawing bearish GG visualization: {e}")
    
    print(f"Added {bearish_count} {status} bearish GGs to chart")
    
    # Add touch points
    bullish_touches = 0
    for touch in touch_times['bullish']:
        # Only show touches for active GGs if active_only is True
        if not active_only or (active_only and touch['gg']['active']):
            try:
                chart.marker(
                    time=touch['timestamp'],
                    position=touch['price'],
                    shape='arrow_up',
                )
                bullish_touches += 1
            except (KeyError, ValueError) as e:
                print(f"Error drawing bullish touch point: {e}")
            
    bearish_touches = 0        
    for touch in touch_times['bearish']:
        # Only show touches for active GGs if active_only is True
        if not active_only or (active_only and touch['gg']['active']):
            try:
                chart.marker(
                    time=touch['timestamp'],
                    position=touch['price'],
                    shape='arrow_down',
                )
                bearish_touches += 1
            except (KeyError, ValueError) as e:
                print(f"Error drawing bearish touch point: {e}")
            
    touch_status = "active GG" if active_only else "total"
    print(f"Added {bullish_touches} bullish and {bearish_touches} bearish {touch_status} touch points")
    
    print("Displaying chart... (This may open in a new window)")
    chart.show(block=True)
    print("Chart display completed")

def create_chart_with_ggs_and_trades(df, bullish_ggs, bearish_ggs, trades, trade_box_transparency=0.2):
    """
    Create and display a chart with Golden Gap (GG) visualizations and reversal patterns
    
    Args:
        df (pandas.DataFrame): DataFrame containing price data
        bullish_ggs (list): List of bullish Golden Gaps
        bearish_ggs (list): List of bearish Golden Gaps
        reversal_patterns (dict): Dictionary containing bearish and bullish reversal patterns with timestamps and prices
        trade_box_transparency (float): Transparency level for trade boxes (0.0-1.0). Default 0.1
    """
    print("\nCreating chart with GGs and reversal patterns...")
    
    # Create chart with larger dimensions for better visibility
    chart = Chart(width=1200, height=800, toolbox=True)
    chart.set(df)
    
    # Add visualization for bullish GGs
    bullish_count = 0
    for gg in bullish_ggs:
        bullish_count += 1
        try:
            chart.box(
                start_time=gg['start_time'],
                end_time=gg['end_time'],    
                start_value=gg['low'],
                end_value=gg['high'],
                color=f'rgba(76, 175, 80, 0.5)',
                fill_color=f'rgba(76, 175, 80, 0.1)'
            )
        except (KeyError, ValueError) as e:
            print(f"Error drawing bullish GG visualization: {e}")   
    
    # Add visualization for bearish GGs
    bearish_count = 0
    for gg in bearish_ggs:
        bearish_count += 1
        try:
            chart.box(
                start_time=gg['start_time'],
                end_time=gg['end_time'],
                start_value=gg['low'],
                end_value=gg['high'],
                color=f'rgba(244, 67, 54, 0.5)',
                fill_color=f'rgba(244, 67, 54, 0.1)'
            )
        except (KeyError, ValueError) as e:
            print(f"Error drawing bearish GG visualization: {e}")
    
    print(f"Added {bullish_count} bullish and {bearish_count} bearish GGs to chart")
    
    # Add trade boxes and markers
    for trade in trades['bearish']:
        # Get trade exit time
        exit_time = trade.get('exit_time', trade['time'])
        initial_sl = trade['sl']
        exit_price = trade['exit_price']
        exit_type = trade['exit_type']
        
        # Grey box from SL to entry
        chart.box(
            start_time=trade['time'],
            end_time=exit_time,
            start_value=trade['entry'],
            end_value=initial_sl,
            color=f'rgba(128, 128, 128, {trade_box_transparency * 3})',
            fill_color=f'rgba(128, 128, 128, {trade_box_transparency})'
        )
        
        # Blue box from entry to TP
        chart.box(
            start_time=trade['time'],
            end_time=exit_time,
            start_value=trade['entry'],
            end_value=trade['tp'],
            color=f'rgba(100, 100, 255, {trade_box_transparency * 3})',
            fill_color=f'rgba(100, 100, 255, {trade_box_transparency})'
        )

        # Add win/loss box from entry to exit price
        status = trade.get('status')
        if status in ['win', 'loss']:
            color = 'rgba(100, 100, 255' if status == 'win' else 'rgba(128, 128, 128'
            chart.box(
                start_time=trade['time'],
                end_time=exit_time, 
                start_value=trade['entry'],
                end_value=trade['exit_price'],
                color=f'{color}, {trade_box_transparency * 10})',
                fill_color=f'{color}, {trade_box_transparency})'
            )
        
        # Entry marker
        chart.marker(
            time=trade['time'],
            position='above',
            shape='arrow_down',
            color='rgba(244, 67, 54, 1.0)'
        )
    
    for trade in trades['bullish']:
        # Get trade exit time
        exit_time = trade.get('exit_time', trade['time'])
        
        # Grey box from SL to entry
        chart.box(
            start_time=trade['time'],
            end_time=exit_time,
            start_value=trade['entry'],
            end_value=trade['sl'],
            color=f'rgba(128, 128, 128, {trade_box_transparency * 3})',
            fill_color=f'rgba(128, 128, 128, {trade_box_transparency})'
        )
        
        # Blue box from entry to TP
        chart.box(
            start_time=trade['time'],
            end_time=exit_time,
            start_value=trade['entry'],
            end_value=trade['tp'],
            color=f'rgba(100, 100, 255, {trade_box_transparency * 3})',
            fill_color=f'rgba(100, 100, 255, {trade_box_transparency})'
        )

        # Add win/loss box from entry to exit price
        status = trade.get('status')
        if status in ['win', 'loss']:
            color = 'rgba(100, 100, 255' if status == 'win' else 'rgba(128, 128, 128'
            chart.box(
                start_time=trade['time'],
                end_time=exit_time,
                start_value=trade['entry'],
                end_value=trade['exit_price'],
                color=f'{color}, {trade_box_transparency * 10})',
                fill_color=f'{color}, {trade_box_transparency})'
            )
        
        # Entry marker
        chart.marker(
            time=trade['time'],
            position='below',
            shape='arrow_up',
            color='rgba(76, 175, 80, 1.0)'
        )
    
    print("Displaying chart... (This may open in a new window)")
    chart.show(block=True)
    print("Chart display completed")        
    
    
def create_chart_with_fvgs(df, ltf_bullish_fvgs, ltf_bearish_fvgs, htf_bullish_fvgs=None, htf_bearish_fvgs=None, 
                          line_style='solid', line_opacity=1.0, fill_opacity=0.3):
    """
    Create and display a chart with Fair Value Gap (FVG) visualizations from two timeframes
    
    Args:
        df (pandas.DataFrame): DataFrame containing price data
        ltf_bullish_fvgs (list): List of primary bullish Fair Value Gaps (e.g. 5m)
        ltf_bearish_fvgs (list): List of primary bearish Fair Value Gaps (e.g. 5m)
        htf_bullish_fvgs (list, optional): List of secondary bullish Fair Value Gaps (e.g. 1h)
        htf_bearish_fvgs (list, optional): List of secondary bearish Fair Value Gaps (e.g. 1h)
        line_style (str, optional): Style of the FVG border lines ('dashed' or 'solid'). Default 'dashed'
        line_opacity (float, optional): Opacity of the FVG border lines (0.0-1.0). Default 1.0
        fill_opacity (float, optional): Opacity of the FVG fill color (0.0-1.0). Default 0.3
    """
    print(f"\nPrimary Bullish FVGs found: {len(ltf_bullish_fvgs)}")
    for i, fvg in enumerate(ltf_bullish_fvgs[:5]):  # Limit to first 5 to avoid too much output
        print(f"  Bullish FVG #{i+1}: Range {fvg['low']:.2f}-{fvg['high']:.2f}, Active: {fvg['active']}")
        print(f"    Start: {fvg['start_time']}, End: {fvg['end_time']}")
        
    print(f"\nPrimary Bearish FVGs found: {len(ltf_bearish_fvgs)}")
    for i, fvg in enumerate(ltf_bearish_fvgs[:5]):  # Limit to first 5 to avoid too much output
        print(f"  Bearish FVG #{i+1}: Range {fvg['low']:.2f}-{fvg['high']:.2f}, Active: {fvg['active']}")
        print(f"    Start: {fvg['start_time']}, End: {fvg['end_time']}")
    
    print("\nCreating chart...")
    # Create chart with larger dimensions for better visibility
    chart = Chart(width=1200, height=800)
    print("Setting data to chart...")
    chart.set(df)
    
    print("Adding FVG visualizations...")
    
    # Add visualization for primary bullish FVGs
    bullish_count_1 = 0
    for i, fvg in enumerate(ltf_bullish_fvgs):
        bullish_count_1 += 1
        try:
            chart.box(
                start_time=fvg['start_time'],
                end_time=fvg['end_time'],
                start_value=fvg['low'],
                end_value=fvg['high'],
                color=f'rgba(76, 175, 80, {line_opacity})',  # Primary timeframe
                fill_color=f'rgba(76, 175, 80, {fill_opacity})',
                width=2,
                style=line_style,
            )   
        except (KeyError, ValueError) as e:
            print(f"Error drawing primary bullish FVG visualization: {e}")
    
    # Add visualization for primary bearish FVGs
    bearish_count_1 = 0
    for i, fvg in enumerate(ltf_bearish_fvgs):
        bearish_count_1 += 1
        try:
            chart.box(
                start_time=fvg['start_time'],
                end_time=fvg['end_time'],
                start_value=fvg['low'],
                end_value=fvg['high'],
                color=f'rgba(244, 67, 54, {line_opacity})',  # Primary timeframe
                fill_color=f'rgba(244, 67, 54, {fill_opacity})',
                width=2,
                style=line_style,
            )
        except (KeyError, ValueError) as e:
            print(f"Error drawing primary bearish FVG visualization: {e}")

    # Add secondary timeframe FVGs if provided
    bullish_count_2 = 0
    bearish_count_2 = 0
    
    if htf_bullish_fvgs:
        for i, fvg in enumerate(htf_bullish_fvgs):
            bullish_count_2 += 1
            try:
                chart.box(
                    start_time=fvg['start_time'],
                    end_time=fvg['end_time'],
                    start_value=fvg['low'],
                    end_value=fvg['high'],
                    color=f'rgba(76, 175, 80, {line_opacity})',  # Secondary timeframe
                    fill_color=f'rgba(76, 175, 80, {fill_opacity})',
                    width=2,
                    style=line_style,
                )   
            except (KeyError, ValueError) as e:
                print(f"Error drawing secondary bullish FVG visualization: {e}")

    if htf_bearish_fvgs:
        for i, fvg in enumerate(htf_bearish_fvgs):
            bearish_count_2 += 1
            try:
                chart.box(
                    start_time=fvg['start_time'],
                    end_time=fvg['end_time'],
                    start_value=fvg['low'],
                    end_value=fvg['high'],
                    color=f'rgba(244, 67, 54, {line_opacity})',  # Secondary timeframe
                    fill_color=f'rgba(244, 67, 54, {fill_opacity})',
                    width=2,
                    style=line_style,
                )
            except (KeyError, ValueError) as e:
                print(f"Error drawing secondary bearish FVG visualization: {e}")
    
    print(f"Added {bullish_count_1} primary and {bullish_count_2} secondary bullish FVGs to chart")
    print(f"Added {bearish_count_1} primary and {bearish_count_2} secondary bearish FVGs to chart")
    print("Displaying chart... (This may open in a new window)")
    chart.show(block=True)
    print("Chart display completed")

    
def create_chart_with_fvgs_and_ggs(df, ltf_bullish_fvgs, ltf_bearish_fvgs, htf_bullish_fvgs=None, htf_bearish_fvgs=None, 
                          bullish_ggs=None, bearish_ggs=None,
                          line_style='solid', line_opacity=1.0, fill_opacity=0.3):
    """
    Create and display a chart with Fair Value Gap (FVG) visualizations from two timeframes and Golden Gaps (GGs)
    
    Args:
        df (pandas.DataFrame): DataFrame containing price data
        ltf_bullish_fvgs (list): List of primary bullish Fair Value Gaps (e.g. 5m)
        ltf_bearish_fvgs (list): List of primary bearish Fair Value Gaps (e.g. 5m)
        htf_bullish_fvgs (list, optional): List of secondary bullish Fair Value Gaps (e.g. 1h)
        htf_bearish_fvgs (list, optional): List of secondary bearish Fair Value Gaps (e.g. 1h)
        bullish_ggs (list, optional): List of bullish Golden Gaps
        bearish_ggs (list, optional): List of bearish Golden Gaps
        line_style (str, optional): Style of the FVG border lines ('dashed' or 'solid'). Default 'dashed'
        line_opacity (float, optional): Opacity of the FVG border lines (0.0-1.0). Default 1.0
        fill_opacity (float, optional): Opacity of the FVG fill color (0.0-1.0). Default 0.3
    """
    print(f"\nPrimary Bullish FVGs found: {len(ltf_bullish_fvgs)}")
    for i, fvg in enumerate(ltf_bullish_fvgs[:5]):  # Limit to first 5 to avoid too much output
        print(f"  Bullish FVG #{i+1}: Range {fvg['low']:.2f}-{fvg['high']:.2f}, Active: {fvg['active']}")
        print(f"    Start: {fvg['start_time']}, End: {fvg['end_time']}")
        
    print(f"\nPrimary Bearish FVGs found: {len(ltf_bearish_fvgs)}")
    for i, fvg in enumerate(ltf_bearish_fvgs[:5]):  # Limit to first 5 to avoid too much output
        print(f"  Bearish FVG #{i+1}: Range {fvg['low']:.2f}-{fvg['high']:.2f}, Active: {fvg['active']}")
        print(f"    Start: {fvg['start_time']}, End: {fvg['end_time']}")
    
    print("\nCreating chart...")
    # Create chart with larger dimensions for better visibility
    chart = Chart(width=1200, height=800)
    print("Setting data to chart...")
    chart.set(df)
    
    print("Adding FVG visualizations...")
    
    # Add visualization for primary bullish FVGs
    bullish_count_1 = 0
    for i, fvg in enumerate(ltf_bullish_fvgs):
        bullish_count_1 += 1
        try:
            chart.box(
                start_time=fvg['start_time'],
                end_time=fvg['end_time'],
                start_value=fvg['low'],
                end_value=fvg['high'],
                color=f'rgba(76, 175, 80, {line_opacity})',  # Primary timeframe
                fill_color=f'rgba(76, 175, 80, {fill_opacity})',
                width=2,
                style=line_style,
            )   
        except (KeyError, ValueError) as e:
            print(f"Error drawing primary bullish FVG visualization: {e}")
    
    # Add visualization for primary bearish FVGs
    bearish_count_1 = 0
    for i, fvg in enumerate(ltf_bearish_fvgs):
        bearish_count_1 += 1
        try:
            chart.box(
                start_time=fvg['start_time'],
                end_time=fvg['end_time'],
                start_value=fvg['low'],
                end_value=fvg['high'],
                color=f'rgba(244, 67, 54, {line_opacity})',  # Primary timeframe
                fill_color=f'rgba(244, 67, 54, {fill_opacity})',
                width=2,
                style=line_style,
            )
        except (KeyError, ValueError) as e:
            print(f"Error drawing primary bearish FVG visualization: {e}")

    # Add secondary timeframe FVGs if provided
    bullish_count_2 = 0
    bearish_count_2 = 0
    
    if htf_bullish_fvgs:
        for i, fvg in enumerate(htf_bullish_fvgs):
            bullish_count_2 += 1
            try:
                chart.box(
                    start_time=fvg['start_time'],
                    end_time=fvg['end_time'],
                    start_value=fvg['low'],
                    end_value=fvg['high'],
                    color=f'rgba(76, 175, 80, {line_opacity})',  # Secondary timeframe
                    fill_color=f'rgba(76, 175, 80, {fill_opacity})',
                    width=3,
                    style='dashed',
                )   
            except (KeyError, ValueError) as e:
                print(f"Error drawing secondary bullish FVG visualization: {e}")

    if htf_bearish_fvgs:
        for i, fvg in enumerate(htf_bearish_fvgs):
            bearish_count_2 += 1
            try:
                chart.box(
                    start_time=fvg['start_time'],
                    end_time=fvg['end_time'],
                    start_value=fvg['low'],
                    end_value=fvg['high'],
                    color=f'rgba(244, 67, 54, {line_opacity})',  # Secondary timeframe
                    fill_color=f'rgba(244, 67, 54, {fill_opacity})',
                    width=3,
                    style='dashed',
                )
            except (KeyError, ValueError) as e:
                print(f"Error drawing secondary bearish FVG visualization: {e}")

    # Add Golden Gaps if provided
    gg_bullish_count = 0
    gg_bearish_count = 0

    if bullish_ggs:
        for i, gg in enumerate(bullish_ggs):
            gg_bullish_count += 1
            try:
                chart.box(
                    start_time=gg['start_time'],
                    end_time=gg['end_time'] if 'end_time' in gg else df.index[-1],
                    start_value=gg['low'],
                    end_value=gg['high'],
                    color='rgba(144, 238, 144, 1.0)',  # Light green with full opacity
                    fill_color='rgba(128, 128, 128, 0.5)',  # Grey with 0.5 opacity
                    width=2,
                    style=line_style,
                )   
            except (KeyError, ValueError) as e:
                print(f"Error drawing bullish GG visualization: {e}")

    if bearish_ggs:
        for i, gg in enumerate(bearish_ggs):
            gg_bearish_count += 1
            try:
                chart.box(
                    start_time=gg['start_time'],
                    end_time=gg['end_time'] if 'end_time' in gg else df.index[-1],
                    start_value=gg['low'],
                    end_value=gg['high'],
                    color='rgba(255, 0, 0, 1.0)',  # Bright red with full opacity
                    fill_color='rgba(128, 128, 128, 0.5)',  # Grey with 0.5 opacity
                    width=2,
                    style=line_style,
                )
            except (KeyError, ValueError) as e:
                print(f"Error drawing bearish GG visualization: {e}")
    
    print(f"Added {bullish_count_1} primary and {bullish_count_2} secondary bullish FVGs to chart")
    print(f"Added {bearish_count_1} primary and {bearish_count_2} secondary bearish FVGs to chart")
    print(f"Added {gg_bullish_count} bullish and {gg_bearish_count} bearish Golden Gaps to chart")
    print("Displaying chart... (This may open in a new window)")
    chart.show(block=True)
    print("Chart display completed")
