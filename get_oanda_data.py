import oandapyV20
from oandapyV20.endpoints.instruments import InstrumentsCandles
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import glob

from config import access_token as API_KEY, ENVIRONMENT

# Initialize the API client
client = oandapyV20.API(access_token=API_KEY, environment=ENVIRONMENT)


## FETCHING

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
    # Ensure start and end are timezone-naive for consistent formatting
    if hasattr(start, 'tzinfo') and start.tzinfo is not None:
        start = start.replace(tzinfo=None)
    if hasattr(end, 'tzinfo') and end.tzinfo is not None:
        end = end.replace(tzinfo=None)
    
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
                
                print(f"Fetched {len(candles)} candles. Next start time: {current_start}")
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


## FORMATTING
def get_oanda_data_with_type(instrument: str, granularity: str, days: str = None, start_time: datetime = None, end_time: datetime = None, price_type: str = "M") -> pd.DataFrame:
    """
    Fetch historical OANDA CFD data and return it as a DataFrame using bid prices.
    
    Args:
        instrument (str): OANDA instrument (e.g., "SPX500_USD").
        granularity (str): Time interval (e.g., "M5" for 5 minutes, "H1" for 1 hour).
        days (str, optional): Number of days back (e.g., "50"). Takes precedence over start/end times if provided.
        start_time (datetime, optional): Start time for data fetch. Used if days not provided.
        end_time (datetime, optional): End time for data fetch. Used if days not provided.
        price_type (str): Price type - "M" for mid, "B" for bid, "A" for ask.
    
    Returns:
        pd.DataFrame: DataFrame with time (index), open, high, low, close, and volume.
    
    Raises:
        ValueError: If inputs are invalid.
        Exception: If API request fails.
    """

    # Validate inputs
    if price_type not in ["M", "B", "A"]:
        raise ValueError("Price type must be 'M', 'B', or 'A'.")

    if not isinstance(instrument, str) or not isinstance(granularity, str):
        raise ValueError("Instrument and granularity must be strings.")

    # Define time range
    if days is not None:
        try:
            days_int = int(days)
            if days_int <= 0:
                raise ValueError("Days must be a positive integer.")
        except ValueError:
            raise ValueError("Days must be a valid number (e.g., '50').")
            
        end_time = datetime.now()  # Current time (UTC assumed)
        start_time = end_time - timedelta(days=days_int)  # Days back
    else:
        if start_time is None or end_time is None:
            raise ValueError("Either days or both start_time and end_time must be provided.")

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

def get_oanda_data_all_types(instrument: str, granularity: str, days: str = None, start_time: datetime = None, end_time: datetime = None) -> pd.DataFrame:
    """
    Get OANDA data for all price types (bid, ask, mid) and combine them into a single DataFrame.
    
    Args:
        instrument (str): OANDA instrument (e.g., "SPX500_USD").
        granularity (str): Time interval (e.g., "M5" for 5 minutes, "H1" for 1 hour).
        days (str, optional): Number of days back (e.g., "50"). Takes precedence over start/end times if provided.
        start_time (datetime, optional): Start time for data fetch. Used if days not provided.
        end_time (datetime, optional): End time for data fetch. Used if days not provided.
    
    Returns:
        pd.DataFrame: DataFrame with bid (sell), mid, and ask (buy) prices.
    """
    # Get data for each price type
    df_bid = get_oanda_data_with_type(instrument, granularity, days, start_time, end_time, price_type="B")
    df_mid = get_oanda_data_with_type(instrument, granularity, days, start_time, end_time, price_type="M") 
    df_ask = get_oanda_data_with_type(instrument, granularity, days, start_time, end_time, price_type="A")

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


## SAVING

def save_oanda_data(instrument: str, granularity: str, start_time: datetime, end_time: datetime, df: pd.DataFrame, src: str = 'oanda') -> None:
    """
    Save OANDA data to a CSV file in the .data folder.
    Uses actual data range in the filename with appropriate time precision based on granularity.
    
    Args:
        instrument (str): OANDA instrument (e.g., "SPX500_USD")
        granularity (str): Time interval (e.g., "M5", "H1")
        start_time (datetime): Requested start time (may differ from actual data range)
        end_time (datetime): Requested end time (may differ from actual data range)
        df (pd.DataFrame): DataFrame containing the OANDA data
        src (str): Data source identifier (default: 'oanda')
    """
    import os
    
    # Create .data directory if it doesn't exist
    if not os.path.exists('.data'):
        os.makedirs('.data')
    
    # Use the actual data range from the DataFrame for the filename
    if not df.empty:
        # Get the first and last timestamps from the data
        data_start = df.index.min()
        data_end = df.index.max()
        
        # Use a simplified format that includes time info but is easier to parse
        # YYYYMMDD_HHMM for all granularities for consistency
        date_format = '%Y%m%d%H%M'  # No underscore - easier to parse
            
        # Format the timestamps
        actual_start = data_start.strftime(date_format)
        actual_end = data_end.strftime(date_format)
    else:
        # Fallback to requested range if DataFrame is empty (unlikely)
        actual_start = start_time.strftime('%Y%m%d')
        actual_end = end_time.strftime('%Y%m%d')
        
    # Format the filename with source, instrument, granularity and ACTUAL time range
    filename = f"{src}_{instrument}_{granularity}_{actual_start}_{actual_end}.csv"
    filepath = os.path.join('.data', filename)
    
    # Find and delete any older files for the same instrument/granularity
    old_pattern = f"{src}_{instrument}_{granularity}_*.csv"
    for old_file in os.listdir('.data'):
        if old_file != filename and old_file.startswith(f"{src}_{instrument}_{granularity}_"):
            try:
                os.remove(os.path.join('.data', old_file))
                print(f"Removed older data file: {old_file}")
            except Exception as e:
                print(f"Warning: Could not remove old file {old_file}: {e}")
    
    # Save DataFrame to CSV
    df.to_csv(filepath)
    
    print(f"Saved {len(df)} rows of {instrument} {granularity} data to {filepath}")
    print(f"Data range: {data_start} to {data_end}")

def parse_date_from_filename(filename):
    """
    Helper function to parse dates from filenames with different formats.
    
    Args:
        filename (str): Filename to parse dates from
        
    Returns:
        tuple: (start_date, end_date) as datetime objects or (None, None) if parsing fails
    """
    try:
        file_parts = filename.split('_')
        if len(file_parts) < 5:
            return None, None
            
        # Get start and end date parts
        start_str = file_parts[-2]
        end_str = file_parts[-1].replace('.csv', '')
        
        # Try different formats based on length
        # New format: YYYYMMDDHHMMSS
        if len(start_str) >= 12:  # Has date and time components
            # Try to extract the date part (first 8 chars) and time part (remaining)
            start_date = datetime.strptime(start_str[:12], '%Y%m%d%H%M')
            end_date = datetime.strptime(end_str[:12], '%Y%m%d%H%M')
            return start_date, end_date
        # Old format with underscore: YYYYMMDD_HHMM
        elif '_' in start_str:
            parts = start_str.split('_')
            if len(parts) == 2:
                date_part, time_part = parts
                start_date = datetime.strptime(f"{date_part}{time_part}", '%Y%m%d%H%M')
                
                parts = end_str.split('_')
                if len(parts) == 2:
                    date_part, time_part = parts
                    end_date = datetime.strptime(f"{date_part}{time_part}", '%Y%m%d%H%M')
                    return start_date, end_date
        # Old format: YYYYMMDD
        elif len(start_str) == 8 and start_str.isdigit() and len(end_str) == 8 and end_str.isdigit():
            start_date = datetime.strptime(start_str, '%Y%m%d')
            end_date = datetime.strptime(end_str, '%Y%m%d')
            return start_date, end_date
    except Exception as e:
        print(f"Warning: Could not parse dates from filename {filename}: {e}")
    
    return None, None

def check_saved_data(instrument: str, granularity: str, start_time: datetime, end_time: datetime, src: str = 'oanda') -> tuple[pd.DataFrame, list[tuple[datetime, datetime]]]:
    """
    Check .data folder for saved data within requested time range and return any found data plus gaps that need to be fetched.
    Uses date-based filenames to efficiently check for available data ranges.
    
    Args:
        instrument (str): OANDA instrument (e.g., "SPX500_USD")
        granularity (str): Time interval (e.g., "M5", "H1") 
        start_time (datetime): Start time of requested data range
        end_time (datetime): End time of requested data range
        src (str): Data source identifier (default: 'oanda')
        
    Returns:
        tuple containing:
            pd.DataFrame: DataFrame with any found data, empty if none found
            list[tuple[datetime, datetime]]: List of (start,end) pairs for missing data ranges
    """
    import os
    import glob
    
    # Initialize empty DataFrame for found data
    found_data = pd.DataFrame()
    
    try:
        # Get list of all matching files for this instrument/granularity
        pattern = os.path.join('.data', f"{src}_{instrument}_{granularity}_*.csv")
        files = glob.glob(pattern)
        
        if not files:
            # No files found, return empty df and full date range
            return found_data, [(start_time, end_time)]
        
        # There should only be one file per instrument/granularity pair
        # (old ones are deleted when new ones are created)
        # But if there are multiple for some reason, use the most recent one
        if len(files) > 1:
            print(f"Warning: Multiple files found for {instrument} {granularity}. Using the most recent one.")
            
        latest_file = None
        latest_end_date = datetime(1970, 1, 1)  # Initialize with old date
        
        # Find the file with the most recent end date
        for file in files:
            filename = os.path.basename(file)
            _, file_end_date = parse_date_from_filename(filename)
            
            if file_end_date and file_end_date > latest_end_date:
                latest_end_date = file_end_date
                latest_file = file
        
        if not latest_file:
            return found_data, [(start_time, end_time)]
        
        # Load the data from the file
        try:
            df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
            if df.empty:
                return found_data, [(start_time, end_time)]
            
            # Convert all timestamps to naive datetime objects for consistent comparison
            request_start = pd.Timestamp(start_time)
            request_end = pd.Timestamp(end_time)
            
            if hasattr(request_start, 'tzinfo') and request_start.tzinfo is not None:
                request_start = request_start.tz_localize(None)
                request_end = request_end.tz_localize(None)
            
            df_start = df.index.min()
            df_end = df.index.max()
            
            if hasattr(df_start, 'tzinfo') and df_start.tzinfo is not None:
                # Convert DataFrame timestamps to naive for comparison
                data_start = df_start.tz_localize(None)
                data_end = df_end.tz_localize(None)
                
                # Also remove timezone from DataFrame index for filtering
                df.index = df.index.tz_localize(None)
            else:
                data_start = df_start
                data_end = df_end
            
            # Quick check if there's any overlap
            if data_end < request_start or data_start > request_end:
                # No overlap, return empty and full range
                return found_data, [(start_time, end_time)]
                
            # Filter to requested date range
            found_data = df[(df.index >= request_start) & (df.index <= request_end)]
            
            if found_data.empty:
                return found_data, [(start_time, end_time)]
            
            # Sort and remove duplicates
            found_data = found_data.sort_index()
            found_data = found_data[~found_data.index.duplicated(keep='first')]
            
            # Identify gaps at start and end
            gaps = []
            
            # Check for gap at the beginning
            if data_start > request_start:
                gaps.append((start_time, df_start))  # Use original timezone-aware timestamp for gap
            
            # Check for gap at the end
            if data_end < request_end:
                gaps.append((df_end, end_time))  # Use original timezone-aware timestamp for gap
            
            return found_data, gaps
                
        except Exception as e:
            print(f"Warning: Could not read file {latest_file}. Error: {e}")
            return found_data, [(start_time, end_time)]
            
    except Exception as e:
        print(f"Error in check_saved_data: {e}. Returning empty data with full date range.")
        return pd.DataFrame(), [(start_time, end_time)]

## MAIN FUNCTION
def get_oanda_data(instrument: str, granularity: str, days: str) -> pd.DataFrame:
    """
    Get OANDA data for a given instrument, granularity, and days.
    Uses data caching to avoid unnecessary API calls. Each instrument/granularity 
    pair will have one CSV file with the date range in the filename.
    
    Args:
        instrument (str): OANDA instrument (e.g., "SPX500_USD")
        granularity (str): Time interval (e.g., "M5", "H1") 
        days (str): Number of days back (e.g., "50")
    
    Returns:
        pd.DataFrame: DataFrame with bid, ask and mid prices and spreads
    """
    try:
        # Calculate date range
        end_time = datetime.now()
        try:
            days_int = int(days)
            if days_int <= 0:
                raise ValueError("Days must be a positive integer")
            start_time = end_time - timedelta(days=days_int)
        except ValueError as e:
            print(f"Error parsing days parameter '{days}': {e}. Using default of 30 days.")
            start_time = end_time - timedelta(days=30)
        
        # Check saved data first
        found_data, gaps = check_saved_data(instrument, granularity, start_time, end_time, 'oanda')
        
        # If we have gaps, fetch the missing data
        if gaps:
            print(f"Found {len(gaps)} gaps in existing data")
            
            # If we already have data but there are gaps, check if they're significant
            # For example, for M15 data, if the gap is less than 15 minutes, we can ignore it
            min_gap_minutes = 0
            if granularity.startswith('M'):
                try:
                    min_gap_minutes = int(granularity[1:])
                except ValueError:
                    min_gap_minutes = 1
            elif granularity.startswith('H'):
                try:
                    min_gap_minutes = int(granularity[1:]) * 60
                except ValueError:
                    min_gap_minutes = 60
            
            
            new_data = pd.DataFrame()
            for gap_start, gap_end in gaps:
                try:
                    # Normalize timestamps before passing to data fetching function
                    # Ensure both are timezone-naive to avoid comparison issues
                    normalized_start = pd.Timestamp(gap_start)
                    normalized_end = pd.Timestamp(gap_end)
                    
                    if hasattr(normalized_start, 'tzinfo') and normalized_start.tzinfo is not None:
                        normalized_start = normalized_start.tz_localize(None)
                    if hasattr(normalized_end, 'tzinfo') and normalized_end.tzinfo is not None:
                        normalized_end = normalized_end.tz_localize(None)
                    
                    # Get data for gap with all price types
                    print(f"Fetching new data for gap: {normalized_start} to {normalized_end}")
                    gap_df = get_oanda_data_all_types(
                        instrument, 
                        granularity, 
                        None, 
                        normalized_start.to_pydatetime(), 
                        normalized_end.to_pydatetime()
                    )
                    if gap_df is not None and not gap_df.empty:
                        new_data = pd.concat([new_data, gap_df])
                    else:
                        print(f"Warning: No data returned for gap {normalized_start} to {normalized_end}")
                except Exception as e:
                    print(f"Error fetching data for gap {gap_start} to {gap_end}: {e}")
            
            # If we fetched new data, combine it with any existing data
            if not new_data.empty:
                # Find the current data file, if it exists
                import os
                import glob
                
                # Get all cached files for this instrument/granularity
                pattern = os.path.join('.data', f"oanda_{instrument}_{granularity}_*.csv")
                files = glob.glob(pattern)
                
                complete_data = pd.DataFrame()
                
                # If we have an existing file, load it
                if files:
                    latest_file = None
                    latest_end_date = datetime(1970, 1, 1)
                    
                    # Find the file with the most recent end date using our helper
                    for file in files:
                        filename = os.path.basename(file)
                        _, file_end_date = parse_date_from_filename(filename)
                        
                        if file_end_date and file_end_date > latest_end_date:
                            latest_end_date = file_end_date
                            latest_file = file
                    
                    # Load the latest file
                    if latest_file:
                        try:
                            complete_data = pd.read_csv(latest_file, index_col=0, parse_dates=True)
                            print(f"Loaded existing data from {latest_file} with {len(complete_data)} rows")
                        except Exception as e:
                            print(f"Warning: Could not read existing data file. Starting fresh. Error: {e}")
                
                # Combine all data
                if not complete_data.empty:
                    # Normalize timezone information for consistent merging
                    if hasattr(complete_data.index, 'tz') and complete_data.index.tz is not None:
                        if not hasattr(new_data.index, 'tz') or new_data.index.tz is None:
                            try:
                                new_data.index = new_data.index.tz_localize(complete_data.index.tz)
                            except Exception:
                                complete_data.index = complete_data.index.tz_localize(None)
                        elif new_data.index.tz != complete_data.index.tz:
                            try:
                                new_data.index = new_data.index.tz_convert(complete_data.index.tz)
                            except Exception:
                                complete_data.index = complete_data.index.tz_localize(None)
                                new_data.index = new_data.index.tz_localize(None)
                    elif hasattr(new_data.index, 'tz') and new_data.index.tz is not None:
                        try:
                            complete_data.index = complete_data.index.tz_localize(new_data.index.tz)
                        except Exception:
                            new_data.index = new_data.index.tz_localize(None)
                    
                    combined_data = pd.concat([complete_data, new_data])
                else:
                    combined_data = new_data
                
                # Sort and remove duplicates
                combined_data = combined_data.sort_index()
                combined_data = combined_data[~combined_data.index.duplicated(keep='first')]
                
                # Save the complete dataset with actual data range in filename
                try:
                    save_oanda_data(instrument, granularity, start_time, end_time, combined_data, 'oanda')
                except Exception as e:
                    print(f"Error saving data: {e}")
                
                # Return only the requested date range
                # Ensure request times match the timezone of the data for filtering
                if hasattr(combined_data.index, 'tz') and combined_data.index.tz is not None:
                    if not hasattr(start_time, 'tzinfo') or start_time.tzinfo is None:
                        request_start = pd.Timestamp(start_time).tz_localize(combined_data.index.tz)
                        request_end = pd.Timestamp(end_time).tz_localize(combined_data.index.tz)
                    else:
                        request_start = pd.Timestamp(start_time).tz_convert(combined_data.index.tz)
                        request_end = pd.Timestamp(end_time).tz_convert(combined_data.index.tz)
                else:
                    # Data has no timezone, ensure request times are also naive
                    if hasattr(start_time, 'tzinfo') and start_time.tzinfo is not None:
                        request_start = pd.Timestamp(start_time).tz_localize(None)
                        request_end = pd.Timestamp(end_time).tz_localize(None)
                    else:
                        request_start = start_time
                        request_end = end_time
                
                result_data = combined_data[(combined_data.index >= request_start) & (combined_data.index <= request_end)]
                return result_data
            
            # If we didn't get any new data but have some found data
            if not found_data.empty:
                return found_data  # Already filtered in check_saved_data
        
        # If no gaps, found_data is already filtered to the requested date range in check_saved_data
        return found_data
        
    except Exception as e:
        print(f"Error in get_oanda_data: {e}")
        # Try to fetch without caching as fallback
        print("Attempting to fetch data directly as fallback...")
        try:
            return get_oanda_data_all_types(instrument, granularity, days)
        except Exception as fallback_error:
            print(f"Fallback fetch also failed: {fallback_error}")
            return pd.DataFrame()  # Return empty DataFrame as last resort


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

        # Fetch 15-minute SPX500USD data for 14 days
        # print("Fetching 15-minute SPX500USD data for 14 days...")
        oanda_data = get_oanda_data("SPX500_USD", "H1", "1825")
        print(oanda_data.head())
        print(f"Total rows: {len(oanda_data)}")

        oanda_data = get_oanda_data("SPX500_USD", "M5", "1825")
        print(oanda_data.head())
        print(f"Total rows: {len(oanda_data)}")
        
        
        # Optional: Save to CSV
        # oanda_data.to_csv("spx500usd_15min_14days.csv")
        
    except Exception as e:
        print(f"Error: {e}")