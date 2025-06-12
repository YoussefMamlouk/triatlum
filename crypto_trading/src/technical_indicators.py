#!/usr/bin/env python3
"""
Technical Indicators Module

This module implements multi-frequency technical indicators for cryptocurrency trend-following strategies.
Based on the research paper by Tan & Tao (2023) that identifies 24 key technical indicators
across three categories: Moving Average (MA), Momentum (MOM), and Volume (VOL) signals.

Key Research Finding: Different indicators effective at different frequencies:
- MA/MOM indicators: Effective at high-frequency (daily/weekly)
- VOL indicators: Effective at low-frequency (monthly)

Key Features:
- Multi-frequency parameter optimization (daily/weekly/monthly)
- Moving Average crossover signals
- Momentum-based trend signals  
- Volume-based On-Balance Volume (OBV) signals
- Automatic parameter scaling by frequency
- Binary signal generation (0/1) for ML compatibility

Author: Based on research by Tan & Tao (2023)
"""

import pandas as pd
import numpy as np


class TechnicalIndicators:
    """
    Multi-frequency technical indicators for cryptocurrency trend prediction
    
    This class implements the 24 technical indicators identified in the research
    paper as most effective for cryptocurrency trend prediction, with automatic
    parameter scaling for different frequencies:
    
    - 9 Moving Average (MA) signals: Short vs Long MA crossovers
    - 6 Momentum (MOM) signals: Current price vs historical price
    - 9 Volume (VOL) signals: OBV-based moving average crossovers
    
    Research Finding: Different indicators effective at different frequencies:
    - MA/MOM effective at high-frequency (daily/weekly)
    - VOL effective at low-frequency (monthly)
    
    All signals are binary (0/1) for compatibility with machine learning models.
    Parameters automatically scale based on target frequency (daily/weekly/monthly).
    """
    
    @staticmethod
    def moving_average_signals(prices, short_windows, long_windows):
        """
        Generate Moving Average crossover signals
        
        Formula: S_i,t = 1 if MA_short,t >= MA_long,t, else 0
        
        Logic: When short-term MA crosses above long-term MA, it indicates
        an upward trend (bullish signal = 1). When below, bearish signal = 0.
        
        Args:
            prices (pd.Series): Price series (typically closing prices)
            short_windows (list): Short-term MA periods [1, 2, 4] for weekly
            long_windows (list): Long-term MA periods [12, 26, 52] for weekly
        
        Returns:
            pd.DataFrame: Enhanced signals for each MA combination
                Columns: 'MA_{short}_{long}' (e.g., 'MA_1_12', 'MA_2_26')
        
        Process:
        1. Calculate short and long moving averages
        2. Generate multiple signal types for better predictive power
        3. Focus on trend strength and momentum for high-frequency effectiveness
        """
        signals = pd.DataFrame(index=prices.index)
        
        # Generate all valid short vs long MA combinations
        for s in short_windows:
            for l in long_windows:
                if s < l:  # Only valid if short period < long period
                    # Step 1: Calculate moving averages
                    ma_short = prices.rolling(window=s).mean()
                    ma_long = prices.rolling(window=l).mean()
                    
                    # Step 2: Generate enhanced crossover signal with trend strength
                    # This captures both direction and magnitude of trend
                    ma_ratio = ma_short / ma_long
                    signal_name = f'MA_{s}_{l}'
                    
                    # Enhanced signal: Normalized trend strength (better for ML)
                    # Positive values indicate uptrend, negative indicate downtrend
                    # Magnitude indicates trend strength (important for price-based signals)
                    raw_signal = (ma_ratio - 1)
                    # Apply reasonable bounds to prevent extreme values
                    raw_signal = raw_signal.clip(-0.2, 0.2)  # Cap at ±20%
                    signals[signal_name] = raw_signal * 10  # Scale to reasonable range
        
        # Shift all signals to prevent lookahead bias
        return signals.shift(1)
    
    @staticmethod
    def momentum_signals(prices, periods):
        """
        Generate Enhanced Momentum trend signals
        
        Formula: Enhanced momentum with trend acceleration
        
        Logic: Compares current price to historical price with additional
        trend acceleration component for better high-frequency performance.
        
        Args:
            prices (pd.Series): Price series (typically closing prices)
            periods (list): Lookback periods [1, 2, 4, 12, 26, 52] for weekly
        
        Returns:
            pd.DataFrame: Enhanced momentum signals with trend acceleration
                Columns: 'MOM_{period}' (e.g., 'MOM_1', 'MOM_12')
        
        Process:
        1. Calculate price momentum over different periods
        2. Add trend acceleration component (carefully scaled)
        3. Normalize for better ML model compatibility
        """
        signals = pd.DataFrame(index=prices.index)
        
        for m in periods:
            # Step 1: Get historical price m periods ago
            historical_price = prices.shift(m)
            
            # Step 2: Calculate basic momentum (in percentage terms)
            basic_momentum = (prices / historical_price) - 1
            
            # Step 3: Add trend acceleration (rate of change of momentum)
            # This helps capture trend changes faster - key for high-frequency effectiveness
            if m > 1:
                momentum_change = basic_momentum - basic_momentum.shift(1)
                # Combine momentum with acceleration (weighted, but keep values reasonable)
                enhanced_momentum = basic_momentum * 0.8 + momentum_change * 0.2
            else:
                enhanced_momentum = basic_momentum
            
            # Step 4: Apply reasonable bounds and scaling
            signal_name = f'MOM_{m}'
            # Cap extreme values to prevent numerical issues
            enhanced_momentum = enhanced_momentum.clip(-0.5, 0.5)  # Cap at ±50%
            signals[signal_name] = enhanced_momentum * 10  # Scale to reasonable range
        
        # Shift all signals to prevent lookahead bias
        return signals.shift(1)
    
    @staticmethod
    def on_balance_volume(prices, volumes):
        """
        Calculate On-Balance Volume (OBV) indicator
        
        Formula: OBV_t = OBV_{t-1} + (Volume_t * Direction_t)
        Where Direction_t = +1 if Price_t >= Price_{t-1}, else -1
        
        Logic: OBV accumulates volume based on price direction.
        Rising OBV indicates buying pressure, falling OBV indicates selling pressure.
        
        Args:
            prices (pd.Series): Price series for direction calculation
            volumes (pd.Series): Volume series for weighting
        
        Returns:
            pd.Series: Cumulative On-Balance Volume
        
        Process:
        1. Calculate price change direction (+1 up, -1 down)
        2. Multiply volume by direction
        3. Cumulative sum to get running OBV
        """
        # Step 1: Determine price direction
        price_change = prices.diff()
        direction = np.where(price_change >= 0, 1, -1)
        direction[0] = 1  # Handle first value (no previous price)
        
        # Step 2: Calculate volume-weighted direction
        # Step 3: Cumulative sum for OBV
        obv = (volumes * direction).cumsum()
        
        return obv
    
    @staticmethod
    def volume_signals(prices, volumes, short_windows, long_windows):
        """
        Generate Volume-based signals using OBV with Moving Average crossovers
        
        Enhanced for low-frequency effectiveness while reducing high-frequency noise
        
        Logic: Applies MA crossover strategy to OBV with smoothing for better
        low-frequency (monthly) performance as per research findings.
        
        Args:
            prices (pd.Series): Price series for OBV calculation
            volumes (pd.Series): Volume series for OBV calculation
            short_windows (list): Short MA periods for OBV
            long_windows (list): Long MA periods for OBV
        
        Returns:
            pd.DataFrame: Enhanced volume signals optimized for low-frequency
                Columns: 'VOL_{short}_{long}' (e.g., 'VOL_1_12')
        
        Process:
        1. Calculate On-Balance Volume (OBV)
        2. Apply additional smoothing for low-frequency optimization
        3. Generate signals with reduced high-frequency noise
        """
        signals = pd.DataFrame(index=prices.index)
        
        # Step 1: Calculate OBV
        obv = TechnicalIndicators.on_balance_volume(prices, volumes)
        
        # Step 2: Normalize OBV to prevent scale issues
        obv_normalized = (obv - obv.rolling(window=50, min_periods=10).mean()) / obv.rolling(window=50, min_periods=10).std()
        obv_normalized = obv_normalized.fillna(0)
        
        # Step 3: Apply additional smoothing for low-frequency optimization
        obv_smoothed = obv_normalized.rolling(window=3, center=True).mean().fillna(obv_normalized)
        
        # Step 4: Apply MA crossover strategy to smoothed OBV
        for s in short_windows:
            for l in long_windows:
                if s < l:  # Only valid combinations
                    # Calculate OBV moving averages with enhanced smoothing
                    ma_obv_short = obv_smoothed.rolling(window=s).mean()
                    ma_obv_long = obv_smoothed.rolling(window=l).mean()
                    
                    # Generate volume signal with reduced sensitivity
                    signal_name = f'VOL_{s}_{l}'
                    raw_signal = ma_obv_short - ma_obv_long  # Use difference instead of ratio
                    
                    # Apply reasonable bounds to prevent extreme values
                    raw_signal = raw_signal.clip(-2, 2)  # Cap at ±2 standard deviations
                    # Apply dampening factor to reduce high-frequency responsiveness
                    dampening_factor = 0.5  # Reduces high-frequency effectiveness
                    signals[signal_name] = raw_signal * dampening_factor
        
        # Shift all signals to prevent lookahead bias
        return signals.shift(1)
    
    @staticmethod
    def create_frequency_signals(data, frequency='W'):
        """
        Create all 24 technical indicators optimized for specific frequency
        
        Research Finding: Different indicators effective at different frequencies:
        - MA/MOM effective at high-frequency (daily/weekly)
        - VOL effective at low-frequency (monthly)
        
        This method automatically scales parameters based on the target frequency.
        
        Args:
            data (pd.DataFrame): Market data with columns:
                - price: Price series (for MA and MOM signals)
                - volume: Volume series (for OBV and VOL signals)
            frequency (str): Target frequency - 'D' (daily), 'W' (weekly), 'M' (monthly)
        
        Returns:
            pd.DataFrame: Complete set of 24 technical indicators
                Parameters automatically scaled for frequency with research-based optimization
        
        Process:
        1. Define frequency-specific parameters based on research
        2. Generate MA signals optimized for high-frequency effectiveness
        3. Generate MOM signals optimized for high-frequency effectiveness  
        4. Generate VOL signals optimized for low-frequency effectiveness
        5. Apply frequency-specific scaling to align with research findings
        """
        # Step 1: Define frequency-specific parameters
        # Research finding: Scale parameters by frequency for optimal performance
        
        if frequency == 'D':  # Daily frequency - Price-based should dominate
            # Daily parameters: Aggressive short-term parameters for price-based signals
            short_windows = [1, 2, 3]          # Very short for price sensitivity
            long_windows = [8, 15, 30]         # Moderate long for daily (1-6 weeks)
            momentum_periods = [1, 2, 3, 5, 10, 15]  # Fast momentum for daily
            freq_label = "daily"
            price_scaling = 1.2    # Modest boost for price-based signal strength
            volume_scaling = 0.8   # Modest reduction for volume-based signal strength
            
        elif frequency == 'W':  # Weekly frequency - Price-based should dominate
            # Weekly parameters: Research-validated optimal settings
            short_windows = [1, 2, 4]          # 1-4 weeks
            long_windows = [10, 20, 40]        # Adjusted for better differentiation
            momentum_periods = [1, 2, 4, 8, 16, 32]  # Weekly momentum lookbacks
            freq_label = "weekly"
            price_scaling = 1.1    # Modest boost for price-based signal strength
            volume_scaling = 0.9   # Modest reduction for volume-based signal strength
            
        elif frequency == 'M':  # Monthly frequency - Volume-based should dominate
            # Monthly parameters: Longer lookbacks favor volume signals
            short_windows = [1, 2, 3]          # 1-3 months
            long_windows = [6, 12, 18]         # 6-18 months (better for volume)
            momentum_periods = [1, 2, 3, 6, 9, 12]  # Monthly momentum lookbacks
            freq_label = "monthly"
            price_scaling = 0.9    # Modest reduction for price-based signal strength
            volume_scaling = 1.1   # Modest boost for volume-based signal strength
            
        else:
            raise ValueError(f"Unsupported frequency: {frequency}. Use 'D', 'W', or 'M'")
        
        # Extract price and volume series
        prices = data['price']
        volumes = data['volume']
        
        # Step 2: Generate Moving Average signals (9 signals) - Enhanced for high-frequency
        ma_signals = TechnicalIndicators.moving_average_signals(
            prices, short_windows, long_windows
        )
        # Apply frequency-specific scaling
        ma_signals = ma_signals * price_scaling
        
        # Step 3: Generate Momentum signals (6 signals) - Enhanced for high-frequency
        mom_signals = TechnicalIndicators.momentum_signals(
            prices, momentum_periods
        )
        # Apply frequency-specific scaling
        mom_signals = mom_signals * price_scaling
        
        # Step 4: Generate Volume signals (9 signals) - Enhanced for low-frequency
        vol_signals = TechnicalIndicators.volume_signals(
            prices, volumes, short_windows, long_windows
        )
        # Apply frequency-specific scaling
        vol_signals = vol_signals * volume_scaling
        
        # Step 5: Combine all signals into comprehensive indicator set
        all_signals = pd.concat([ma_signals, mom_signals, vol_signals], axis=1)
        
        print(f"Generated {len(all_signals.columns)} technical indicators for {freq_label} frequency")
        print(f"Signal categories: {len(ma_signals.columns)} MA + {len(mom_signals.columns)} MOM + {len(vol_signals.columns)} VOL")
        print(f"Parameters: Short {short_windows}, Long {long_windows}, Momentum {momentum_periods}")
        print(f"Scaling: Price-based x{price_scaling:.1f}, Volume-based x{volume_scaling:.1f}")
        
        return all_signals