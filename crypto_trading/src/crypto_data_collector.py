#!/usr/bin/env python3
"""
Crypto Data Collector Module - Bitfinex API v2

This module handles data collection for cryptocurrency trend-following strategies.
It fetches historical OHLCV data from Bitfinex API and creates market indices.

Key Features:
- 10+ years of historical data (2015-2025)
- Historical data fetching with error handling  
- Value-weighted market index creation
- Rate limiting for API compliance
- Data validation and cleaning

Author: Based on research by Tan & Tao (2023)
"""

import pandas as pd
import numpy as np
import requests
import time
import os
import pickle
from datetime import datetime, timedelta


class CryptoDataCollector:
    """
    Optimized data collection for crypto trend-following strategies using Bitfinex API v2
    
    This class provides methods to:
    1. Fetch historical cryptocurrency data from Bitfinex API (2015-2025)
    2. Create value-weighted market indices
    3. Handle data validation and cleaning
    4. Manage API rate limits
    """
    
    def __init__(self):
        """
        Initialize the data collector with Bitfinex API v2 ONLY
        
        Sets up URL for Bitfinex API - NO FALLBACK TO OTHER APIS
        """
        self.bitfinex_url = "https://api-pub.bitfinex.com/v2"
        
        # Create data directory for local storage
        self.data_dir = "cached_data"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"Created data cache directory: {self.data_dir}")
        
        # Bitfinex symbol mapping
        self.symbol_map = {
            'BTCUSDT': 'tBTCUSD',
            'ETHUSDT': 'tETHUSD', 
            'BNBUSDT': 'tBNBUSD',
            'ADAUSDT': 'tADAUSD',
            'SOLUSDT': 'tSOLUSD',
            'XRPUSDT': 'tXRPUSD',
            'DOTUSDT': 'tDOTUSD',
            'DOGEUSDT': 'tDOGUSD',
            'AVAXUSDT': 'tAVAX:USD',
            'LINKUSDT': 'tLINK:USD',
            'UNIUSDT': 'tUNIUSD',
            'LTCUSDT': 'tLTCUSD',
            'ALGOUSDT': 'tALGUSD',
            'MATICUSDT': 'tMATIC:USD',
            'ATOMUSDT': 'tATOMUSD',
            'BCHUSDT': 'tBCHUSD',
            'EOSUSDT': 'tEOSUSD',
            'TRXUSDT': 'tTRXUSD',
            'XLMUSDT': 'tXLMUSD',
            'VETUSDT': 'tVETUSD',
            'ICXUSDT': 'tICXUSD',
            'NEOUSDT': 'tNEOUSD',
            'ETCUSDT': 'tETCUSD',
            'DASHUSDT': 'tDSHUSD',
            'ZECUSDT': 'tZECUSD',
            # Additional symbols for current collection
            'SOLUSD': 'tSOLUSD',
            'MATICUSD': 'tMATIC:USD',
            'LINKUSD': 'tLINK:USD',
            'UNIUSD': 'tUNIUSD'
        }
    
    def _get_cache_filename(self, symbol, interval='1D'):
        """Get the filename for cached data"""
        return os.path.join(self.data_dir, f"{symbol}_{interval}.pkl")
    
    def _save_data_to_cache(self, symbol, data, interval='1D'):
        """Save data to local cache"""
        try:
            cache_file = self._get_cache_filename(symbol, interval)
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"✓ Cached {symbol} data to {cache_file}")
        except Exception as e:
            print(f"Warning: Could not cache {symbol} data: {e}")
    
    def _load_data_from_cache(self, symbol, interval='1D'):
        """Load data from local cache if available"""
        try:
            cache_file = self._get_cache_filename(symbol, interval)
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                print(f"✓ Loaded {symbol} from cache: {len(data)} periods from {data.index[0].date()} to {data.index[-1].date()}")
                return data
        except Exception as e:
            print(f"Warning: Could not load cached {symbol} data: {e}")
        return None
    
    def _is_cache_fresh(self, symbol, interval='1D', max_age_hours=24):
        """Check if cached data is fresh enough"""
        try:
            cache_file = self._get_cache_filename(symbol, interval)
            if os.path.exists(cache_file):
                file_age = time.time() - os.path.getmtime(cache_file)
                return file_age < (max_age_hours * 3600)  # Convert hours to seconds
        except:
            pass
        return False
    
    def get_historical_data_bitfinex(self, symbol, interval='1D', limit=5000):
        """
        Fetch extensive historical OHLCV data from Bitfinex API v2
        
        Enhanced version that can fetch 10+ years of data by making multiple requests
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
            interval (str): Data interval ('1D' for daily, '1W' for weekly)
            limit (int): Total number of data points to fetch
        
        Returns:
            pd.DataFrame: Indexed by timestamp with columns:
                - open, high, low, close: Price data
                - volume: Trading volume
                - returns: Price returns (pct_change)
            None: If data fetching fails
        
        Process:
        1. Map symbol to Bitfinex format
        2. Calculate date range for 10+ years of data
        3. Make multiple API requests with proper pagination
        4. Combine all data into single DataFrame
        5. Calculate price returns and clean data
        """
        try:
            # Step 1: Map to Bitfinex symbol format
            bitfinex_symbol = self.symbol_map.get(symbol, f't{symbol.replace("USDT", "USD")}')
            
            all_data = []
            max_per_request = 5000  # Reduced to be safer with rate limits
            
            # Step 2: Calculate date range for extensive historical data
            end_time = int(datetime.now().timestamp() * 1000)  # Current time in milliseconds
            start_time = int((datetime.now() - timedelta(days=3650)).timestamp() * 1000)  # 10 years ago
            
            # Calculate number of requests needed
            num_requests = max(1, (limit + max_per_request - 1) // max_per_request)
            
            print(f"Fetching {limit} periods for {symbol} ({bitfinex_symbol}) in {num_requests} requests...")
            
            current_end = end_time
            
            for i in range(num_requests):
                # Calculate how many records to fetch in this request
                remaining = limit - len(all_data) if all_data else limit
                current_limit = min(max_per_request, remaining)
                
                if current_limit <= 0:
                    break
                
                # Step 3: Prepare Bitfinex API request
                url = f"{self.bitfinex_url}/candles/trade:{interval}:{bitfinex_symbol}/hist"
                params = {
                    'limit': current_limit,
                    'end': current_end,
                    'start': start_time,
                    'sort': -1  # Sort in descending order (newest first)
                }
                
                # Step 4: Make API call with fast retry logic
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        response = requests.get(url, params=params, timeout=30)  # Increased timeout
                        if response.status_code == 522:  # Server error - retry immediately
                            print(f"  Server error 522, retry {retry + 1}/{max_retries}")
                            continue
                        if response.status_code == 429:  # Rate limit - retry immediately
                            print(f"  Rate limit, retry {retry + 1}/{max_retries}")
                            continue
                        response.raise_for_status()
                        data = response.json()
                        break
                    except Exception as e:
                        if retry == max_retries - 1:
                            print(f"  Failed after {max_retries} retries: {e}")
                            return None
                        print(f"  Retry {retry + 1}/{max_retries} after error: {e}")
                else:
                    print(f"  Max retries exceeded for {symbol}")
                    return None
                
                if not data or len(data) == 0:
                    print(f"No more data available for {symbol}")
                    break
                
                # Step 5: Parse Bitfinex response format
                # Bitfinex returns: [MTS, OPEN, CLOSE, HIGH, LOW, VOLUME]
                df_chunk = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'close', 'high', 'low', 'volume'
                ])
                
                # Add to our collection
                all_data.extend(data)
                
                # Update end time for next request (earliest timestamp from this batch)
                if data:
                    current_end = int(data[-1][0]) - 1  # MTS is first element
                
                # MINIMAL delay to avoid overwhelming server
                if i < num_requests - 1:  # Don't wait after last request
                    time.sleep(0.1)  # 100ms delay
                
                print(f"  Request {i+1}/{num_requests}: {len(data)} records")
            
            if not all_data:
                print(f"No data retrieved for {symbol}")
                return None
            
            # Step 6: Create DataFrame from all data
            df = pd.DataFrame(all_data, columns=[
                'timestamp', 'open', 'close', 'high', 'low', 'volume'
            ])
            
            # Remove duplicates and sort by timestamp
            df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            
            # Step 7: Data type conversions and cleaning
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].astype(float)
            
            # Step 8: Calculate returns (essential for strategy)
            df['returns'] = df['close'].pct_change()
            
            # Step 9: Return cleaned data with datetime index
            result = df.set_index('timestamp')[['open', 'high', 'low', 'close', 'volume', 'returns']]
            
            print(f"✓ {symbol}: {len(result)} periods from {result.index[0].date()} to {result.index[-1].date()}")
            return result
            
        except Exception as e:
            print(f"Error fetching data for {symbol} from Bitfinex: {e}")
            return None
    
    # REMOVED BINANCE FALLBACK METHOD - BITFINEX ONLY
    
    def get_historical_data(self, symbol, interval='1D', limit=5000, use_cache=True):
        """
        Main method - BITFINEX ONLY with local caching
        
        Args:
            symbol: Trading symbol
            interval: Data interval
            limit: Number of periods
            use_cache: Whether to use cached data (default True)
        """
        # Check if we should use cached data
        if use_cache:
            # Try to load from cache first
            cached_data = self._load_data_from_cache(symbol, interval)
            if cached_data is not None and self._is_cache_fresh(symbol, interval, max_age_hours=24):
                print(f"📁 Using fresh cached data for {symbol}")
                return cached_data
            elif cached_data is not None:
                print(f"🕒 Cached data for {symbol} exists but is stale, fetching fresh data...")
        
        # Fetch fresh data from Bitfinex
        print(f"🌐 Fetching fresh data for {symbol} from Bitfinex...")
        data = self.get_historical_data_bitfinex(symbol, interval, limit)
        
        # Cache the data if we got it successfully
        if data is not None and use_cache:
            self._save_data_to_cache(symbol, data, interval)
        
        return data
    
    def create_market_index(self, symbols, interval='1D', limit=5000, use_cache=True):
        """
        Create a value-weighted market index from multiple cryptocurrencies
        
        This method implements the research paper's approach for creating
        a representative market index using volume-based weighting.
        
        Args:
            symbols (list): List of cryptocurrency symbols to include
            interval (str): Data frequency ('1D' for daily)
            limit (int): Number of historical data points
            use_cache (bool): Whether to use cached data
        
        Returns:
            pd.DataFrame: Market index with columns:
                - price: Value-weighted average price
                - volume: Total volume across all assets
                - returns: Market index returns
        
        Process:
        1. Fetch data for all symbols with validation
        2. Calculate volume-based weights (proxy for market cap)
        3. Create common time index across all assets
        4. Compute value-weighted average prices
        5. Calculate market index returns
        """
        # Try to load cached market index first
        if use_cache:
            try:
                market_index_file = os.path.join(self.data_dir, "market_index.pkl")
                if os.path.exists(market_index_file):
                    file_age = time.time() - os.path.getmtime(market_index_file)
                    if file_age < (24 * 3600):  # Less than 24 hours old
                        with open(market_index_file, 'rb') as f:
                            market_index = pickle.load(f)
                        print(f"📁 Using cached market index: {len(market_index)} periods from {market_index.index[0].date()} to {market_index.index[-1].date()}")
                        return market_index
                    else:
                        print(f"🕒 Cached market index is stale, creating fresh index...")
            except Exception as e:
                print(f"Warning: Could not load cached market index: {e}")
        
        print(f"Creating market index from {len(symbols)} cryptocurrencies...")
        
        # Step 1: Collect data for all symbols
        all_data = {}  # Store valid data for each symbol
        market_caps = {}  # Store volume-based weights
        
        for i, symbol in enumerate(symbols):
            # MINIMAL delay between symbols to avoid overwhelming server
            if i > 0:
                time.sleep(0.5)  # 500ms delay between symbols
            
            # Fetch individual symbol data
            data = self.get_historical_data(symbol, interval, limit)
            
            # Validate data quality (minimum 100 data points)
            if data is not None and len(data) > 100:
                all_data[symbol] = data
                # Use average volume as proxy for market capitalization
                market_caps[symbol] = data['volume'].mean()
                print(f"✓ {symbol}: {len(data)} data points")
            else:
                print(f"✗ {symbol}: Insufficient data")
        
        # Validate that we have sufficient data
        if not all_data:
            raise ValueError("No valid data collected")
        
        # Step 2: Create common time index using union instead of intersection
        # Use union to get maximum coverage rather than intersection
        all_indices = [data.index for data in all_data.values()]
        common_index = all_indices[0]
        for idx in all_indices[1:]:
            common_index = common_index.union(idx)
        
        # Limit to most recent periods for better overlap
        common_index = common_index.sort_values()[-2000:]  # Last 2000 periods
        
        print(f"Common time periods: {len(common_index)}")
        
        # Step 3: Align all data to common index with forward fill
        aligned_data = {}
        total_market_cap = sum(market_caps.values())
        
        for symbol, data in all_data.items():
            # Reindex to common time periods with forward fill
            aligned_data[symbol] = data.reindex(common_index, method='ffill')
            
            # Calculate normalized weights
            weight = market_caps[symbol] / total_market_cap
            
            print(f"  {symbol}: Weight = {weight:.3f}")
        
        # Step 4: Calculate value-weighted market index
        # Compute weighted average price and total volume
        market_price = pd.Series(0.0, index=common_index)
        market_volume = pd.Series(0.0, index=common_index)
        
        for symbol, data in aligned_data.items():
            weight = market_caps[symbol] / total_market_cap
            # Only include periods where we have valid data
            valid_data = data.dropna()
            if len(valid_data) > 0:
                # Align to common index with forward fill
                symbol_prices = valid_data['close'].reindex(common_index, method='ffill')
                symbol_volumes = valid_data['volume'].reindex(common_index, method='ffill')
                
                market_price += symbol_prices.fillna(0) * weight
                market_volume += symbol_volumes.fillna(0)
        
        # Step 5: Calculate market returns
        market_returns = market_price.pct_change()
        
        # Step 6: Create final market index DataFrame
        market_index = pd.DataFrame({
            'price': market_price,
            'volume': market_volume,
            'returns': market_returns
        })
        
        # Remove any NaN values from the beginning
        market_index = market_index.dropna()
        
        print(f"Market index created with {len(market_index)} data points")
        print(f"Date range: {market_index.index[0].date()} to {market_index.index[-1].date()}")
        
        # Cache the market index
        try:
            market_index_file = os.path.join(self.data_dir, "market_index.pkl")
            with open(market_index_file, 'wb') as f:
                pickle.dump(market_index, f)
            print(f"✓ Cached market index to {market_index_file}")
        except Exception as e:
            print(f"Warning: Could not cache market index: {e}")
        
        return market_index 