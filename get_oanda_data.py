import oandapyV20
from oandapyV20.endpoints.instruments import InstrumentsCandles
import pandas as pd
from datetime import datetime, timedelta
import time

from config import access_token as API_KEY, ENVIRONMENT

# Initialize the API client
client = oandapyV20.API(access_token=API_KEY, environment=ENVIRONMENT)


def get_candle_batch(instrument, granularity, start, end, count=None, price_type="M"):
    """
    Helper function to fetch a single batch of candles from OANDA API.
    
    Args:
        instrument (str): Trading instrument
        granularity (str): Time interval
        start (datetime): Start time
        end (datetime): End time
        count (int, optional): Maximum number of candles to fetch
        
    Returns:
        list: List of candle data from OANDA
    """
    params = {
        "granularity": granularity,
        "from": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "to": end.strftime("%Y-%m-%dT%H:%M:%SZ") if not count else None,
        "count": count if count else None,
        "price": price_type  # Use mid prices
    }

    request = InstrumentsCandles(instrument=instrument, params=params)
    response = client.request(request)
    return response["candles"]

def get_all_candles(instrument, granularity, start_time, end_time, max_candles_per_request=5000, price_type="M"):
    """
    Fetch all candles for a given time range, potentially in multiple batches.
    
    Args:
        instrument (str): Trading instrument
        granularity (str): Time interval
        start_time (datetime): Start time
        end_time (datetime): End time
        max_candles_per_request (int): Max candles per API request
        
    Returns:
        list: Aggregated list of all candles
    """
    all_candles = []
    current_start = start_time
    
    while current_start < end_time:
        try:
            # print(f"Fetching candles from {current_start} to {end_time}")
            
            # Fetch batch of candles
            candles = get_candle_batch(instrument, granularity, current_start, end_time, max_candles_per_request, price_type)
            if not candles:
                break
            
            all_candles.extend(candles)
            
            # Determine next start time based on candle timestamps
            if len(candles) >= 2:
                # Get the last two candles
                last_candle_time = datetime.strptime(candles[-1]["time"][:26] + "Z", "%Y-%m-%dT%H:%M:%S.%fZ")
                second_last_candle_time = datetime.strptime(candles[-2]["time"][:26] + "Z", "%Y-%m-%dT%H:%M:%S.%fZ")
                
                # Calculate the time difference between them
                time_diff = last_candle_time - second_last_candle_time
                
                # Add the same increment to advance to the next candle period
                current_start = last_candle_time + time_diff
                
                # print(f"Fetched {len(candles)} candles. Next start time: {current_start}")
            else:
                # If we have fewer than 2 candles, we're done fetching
                # print(f"Fetched final {len(candles)} candle(s). Ending data collection.")
                break
            
            # Respect rate limits
            time.sleep(2)
        
        except Exception as e:
            raise Exception(f"API request failed: {e}")
    
    return all_candles

def convert_candles_to_df(candles, price_type="mid"):
    """
    Convert a list of OANDA candles to a pandas DataFrame.
    
    Args:
        candles (list): List of candle data from OANDA
        
    Returns:
        pd.DataFrame: DataFrame with OHLCV data
    """
    data = []
    for candle in candles:
        if candle["complete"]:  # Only completed candles
            data.append({
                "Datetime": candle["time"][:26] + "Z",  # Truncate to microseconds
                "Close": float(candle[price_type]["c"]),
                "High": float(candle[price_type]["h"]), 
                "Low": float(candle[price_type]["l"]),
                "Open": float(candle[price_type]["o"]),
                "Volume": candle["volume"]
            })

    if not data:
        return None
    
    df = pd.DataFrame(data)
    df["Datetime"] = pd.to_datetime(df["Datetime"])  # Convert to datetime
    df.set_index("Datetime", inplace=True)
    df.name = "Price"
    
    return df

def get_oanda_data_with_type(instrument: str, granularity: str, days: str, price_type: str = "M") -> pd.DataFrame:
    """
    Fetch historical OANDA CFD data and return it as a DataFrame using bid prices.
    
    Args:
        instrument (str): OANDA instrument (e.g., "SPX500_USD").
        granularity (str): Time interval (e.g., "M5" for 5 minutes, "H1" for 1 hour).
        days (str): Number of days back (e.g., "50").
    
    Returns:
        pd.DataFrame: DataFrame with time (index), open, high, low, close, and volume.
    
    Raises:
        ValueError: If inputs are invalid.
        Exception: If API request fails.
    """

    # Validate inputs
    if price_type not in ["M", "B", "A"]:
        raise ValueError("Price type must be 'M', 'B', or 'A'.")

    try:
        days_int = int(days)
        if days_int <= 0:
            raise ValueError("Days must be a positive integer.")
    except ValueError:
        raise ValueError("Days must be a valid number (e.g., '50').")

    if not isinstance(instrument, str) or not isinstance(granularity, str):
        raise ValueError("Instrument and granularity must be strings.")

    # Define time range
    end_time = datetime.now()  # Current time (UTC assumed)
    start_time = end_time - timedelta(days=days_int)  # Days back

    # Fetch all candles
    all_candles = get_all_candles(
        instrument=instrument,
        granularity=granularity,
        start_time=start_time,
        end_time=end_time,
        price_type=price_type
    )

    long_price_type = "bid" if price_type == "B" else "ask" if price_type == "A" else "mid"
    # Convert to DataFrame
    df = convert_candles_to_df(all_candles, long_price_type)
    
    if df is None or len(df) == 0:
        raise ValueError(f"No data returned for {instrument} at {granularity} granularity.")
    
    return df

def get_oanda_data(instrument: str, granularity: str, days: str) -> pd.DataFrame:
    """
    Get OANDA data for all price types (bid, ask, mid) and combine them into a single DataFrame.
    
    Args:
        instrument (str): OANDA instrument (e.g., "SPX500_USD").
        granularity (str): Time interval (e.g., "M5" for 5 minutes, "H1" for 1 hour).
        days (str): Number of days back (e.g., "50").
    
    Returns:
        pd.DataFrame: DataFrame with bid (sell), mid, and ask (buy) prices.
    """
    # Get data for each price type
    df_bid = get_oanda_data_with_type(instrument, granularity, days, price_type="B")
    df_mid = get_oanda_data_with_type(instrument, granularity, days, price_type="M") 
    df_ask = get_oanda_data_with_type(instrument, granularity, days, price_type="A")

    # Rename columns
    df_bid = df_bid.rename(columns={
        'Close': 'Close_bid',
        'High': 'High_bid', 
        'Low': 'Low_bid',
        'Open': 'Open_bid',
    })

    df_mid = df_mid.rename(columns={
        'Close': 'Close',
        'High': 'High',
        'Low': 'Low', 
        'Open': 'Open',
    })

    df_ask = df_ask.rename(columns={
        'Close': 'Close_ask',
        'High': 'High_ask',
        'Low': 'Low_ask',
        'Open': 'Open_ask',
    })

    # Combine DataFrames
    df = pd.concat([
        df_bid[['Close_bid', 'High_bid', 'Low_bid', 'Open_bid']],
        df_mid[['Close', 'High', 'Low', 'Open']],
        df_ask[['Close_ask', 'High_ask', 'Low_ask', 'Open_ask', 'Volume']],
    ], axis=1)

    # Add spread columns
    df['Close_spread'] = round(df['Close_ask'] - df['Close_bid'], 1)
    df['Open_spread'] = round(df['Open_ask'] - df['Open_bid'], 1)
    df['High_spread'] = round(df['High_ask'] - df['High_bid'], 1)
    df['Low_spread'] = round(df['Low_ask'] - df['Low_bid'], 1)

    return df
    

# Example usage
if __name__ == "__main__":
    try:
        # # Fetch 5-minute SPX500USD data for 50 days
        # oanda_data = get_oanda_data("SPX500_USD", "M5", "50")
        # print(oanda_data.head())
        # print(f"Total rows: {len(oanda_data)}")
        
        # # Optional: Save to CSV
        # oanda_data.to_csv("spx500usd_5min_50days_bid.csv")


        # # Fetch 5-minute SPX500USD data for 50 days
        # oanda_data = get_oanda_data("SPX500_USD", "M1", "7")
        # print(oanda_data.head())
        # print(f"Total rows: {len(oanda_data)}")
        
        # # Optional: Save to CSV
        # oanda_data.to_csv("spx500usd_1min_7days_bid.csv")


        # Fetch 15-minute SPX500USD data for 7 days
        oanda_data = get_oanda_data("SPX500_USD", "M15", "7")
        print(oanda_data.head())
        print(f"Total rows: {len(oanda_data)}")
        
        # Optional: Save to CSV
        oanda_data.to_csv("spx500usd_15min_7days.csv")
        
    except Exception as e:
        print(f"Error: {e}")