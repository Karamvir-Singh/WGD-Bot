from oandapyV20 import API
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.trades as trades
from oandapyV20.contrib.requests import MarketOrderRequest
from oandapyV20.contrib.requests import StopLossDetails

from config import accountID



def get_open_trades(client, accountID=accountID):
    open_trades = client.request(trades.TradesList(accountID))['trades']
    return open_trades


def close_all_trades(client, accountID=accountID):
    open_trades = get_open_trades(client, accountID)

    # If there are any open trades, close them
    if open_trades:
        for trade in open_trades:
            trade_id = trade['id']
            # Close the trade
            close_request = trades.TradeClose(accountID, trade_id)
            close_response = client.request(close_request)
            print(f"Closed trade: {trade_id}")

def trail_stop_losses(client, current_close, rr_jumpstep=1, accountID=accountID):
    """
    Trail stop losses for all open trades based on R-multiple movement
    
    Args:
        client: OANDA API client instance
        current_close (float): Current candle close price
        rr_jumpstep (float): Size of R-multiple steps for trailing stop loss
        accountID (str): OANDA account ID
    """
    open_trades = get_open_trades(client, accountID)

    for trade in open_trades:
        # Extract trade details
        trade_id = trade['id']
        entry_price = float(trade['price'])
        current_sl = float(trade['stopLossOrder']['price'])
        initial_sl = float(trade['initialStopLossPrice']) if 'initialStopLossPrice' in trade else current_sl
        
        # Determine trade direction
        direction = 'bullish' if trade['currentUnits'].startswith('-') else 'bearish'
        
        # Calculate new stop loss
        initial_risk = abs(entry_price - initial_sl)
        
        if direction == 'bullish':
            price_movement = current_close - entry_price
            r_multiple = price_movement / initial_risk
            
            if r_multiple >= rr_jumpstep:
                r_steps = int(r_multiple / rr_jumpstep)
                new_sl = entry_price + (r_steps * rr_jumpstep * initial_risk)
                
                # Only update if new SL is higher than current
                if new_sl > current_sl:
                    sl_trail_request = trades.TradeCRCDO(
                        accountID,
                        trade_id,
                        {"stopLoss": {"price": str(new_sl)}, "timeInForce": "GTC"}
                    )
                    client.request(sl_trail_request)
                    
        else:  # bearish
            price_movement = entry_price - current_close
            r_multiple = price_movement / initial_risk
            
            if r_multiple >= rr_jumpstep:
                r_steps = int(r_multiple / rr_jumpstep)
                new_sl = entry_price - (r_steps * rr_jumpstep * initial_risk)
                
                # Only update if new SL is lower than current
                if new_sl < current_sl:
                    sl_trail_request = trades.TradeCRCDO(
                        accountID,
                        trade_id,
                        {"stopLoss": {"price": str(new_sl)}, "timeInForce": "GTC"}
                    )
                    client.request(sl_trail_request)

def place_market_order(client, accountID, direction, stop_loss, instrument, units):
    """Place a market order with specified direction, stop loss, instrument, and units.
    
    Parameters
    ----------
    client : oandapyV20.API
        The OANDA API client instance
    accountID : str
        The account ID to place the trade on
    direction : str
        'buy' or 'sell' to indicate trade direction
    stop_loss : float
        The stop loss price level
    instrument : str
        The trading instrument (e.g., 'EUR_USD')
    units : int
        The number of units to trade
    """
    units_str = str(units) if direction == "buy" else str(-units)
    
    mo = MarketOrderRequest(
        instrument=instrument,
        units=units_str,
        stopLossOnFill=StopLossDetails(price=stop_loss).data,
        timeInForce="IOC"  # Immediate or Cancel, allows partial fill
    )
    
    place_order_request = orders.OrderCreate(accountID, data=mo.data)
    client.request(place_order_request)

def get_account_margin(client, accountID):
    """Get the current account margin available for trading.
    
    Parameters
    ----------
    client : oandapyV20.API
        The OANDA API client instance
    accountID : str
        The account ID to check margin for
        
    Returns
    -------
    dict
        Dictionary containing margin details including:
        - marginAvailable: Amount available for margin trading
        - marginUsed: Current margin in use
        - positionValue: Value of open positions
        - balance: Current account balance
    """
    from oandapyV20.endpoints import accounts
    
    # Request account details
    r = accounts.AccountDetails(accountID=accountID)
    client.request(r)
    
    # Extract relevant margin details
    margin_details = {
        'marginAvailable': float(r.response['account']['marginAvailable']),
        'marginUsed': float(r.response['account']['marginUsed']), 
        'positionValue': float(r.response['account']['positionValue']),
        'balance': float(r.response['account']['balance'])
    }
    
    return margin_details





