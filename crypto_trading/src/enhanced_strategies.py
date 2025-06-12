#!/usr/bin/env python3
"""
Enhanced Strategies Module

This module implements advanced strategy enhancements using multiple lookback windows
for moving averages and momentum indicators to improve the current ML-enhanced PLS approach.

Key Enhancements:
1. Multi-timeframe Moving Average Convergence (short + long lookbacks)
2. Momentum Acceleration with dual-window analysis
3. Adaptive Volume-weighted indicators
4. Cross-timeframe signal confirmation
5. Trend strength measurement across multiple horizons

These strategies complement the existing PLS ML approach by providing:
- Better trend identification across different time horizons
- Reduced false signals through multi-window confirmation
- Enhanced momentum detection with acceleration components
- Adaptive position sizing based on signal strength

Author: Enhancement to the research-backed crypto strategy
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict


class EnhancedStrategies:
    """
    Advanced strategy enhancements using multiple lookback windows
    
    This class provides enhanced versions of technical indicators that use
    multiple lookback windows to capture both short-term momentum and 
    long-term trends for improved signal quality.
    """
    
    @staticmethod
    def multi_window_moving_average(prices, short_windows=[5, 10, 20], long_windows=[50, 100, 200]):
        """
        Enhanced Moving Average strategy using multiple lookback windows
        
        Creates a comprehensive MA signal by combining:
        1. Short-term trend detection (5, 10, 20 periods)
        2. Long-term trend confirmation (50, 100, 200 periods)
        3. Cross-timeframe alignment scoring
        
        Args:
            prices: Price series
            short_windows: Short-term lookback periods
            long_windows: Long-term lookback periods
            
        Returns:
            DataFrame with enhanced MA signals and trend strength scores
        """
        # Ensure input is pandas Series
        if not isinstance(prices, pd.Series):
            prices = pd.Series(prices)
        
        signals = pd.DataFrame(index=prices.index)
        
        # Calculate all moving averages
        mas = {}
        for window in short_windows + long_windows:
            mas[f'MA_{window}'] = prices.rolling(window=window).mean()
        
        # 1. Multi-window trend alignment score
        alignment_scores = []
        for short in short_windows:
            for long in long_windows:
                if short < long:
                    # Calculate trend alignment strength
                    ma_short = mas[f'MA_{short}']
                    ma_long = mas[f'MA_{long}']
                    
                    # Normalized distance between MAs (trend strength)
                    alignment = (ma_short - ma_long) / ma_long
                    alignment_scores.append(alignment)
        
        # Average alignment across all timeframes
        signals['MA_Multi_Alignment'] = pd.concat(alignment_scores, axis=1).mean(axis=1)
        
        # 2. Trend acceleration across timeframes
        short_trend = (mas['MA_5'] - mas['MA_20']) / mas['MA_20']
        long_trend = (mas['MA_50'] - mas['MA_200']) / mas['MA_200']
        signals['MA_Trend_Acceleration'] = short_trend - short_trend.shift(5)
        
        # 3. Cross-timeframe confirmation signal
        # Signal strength based on alignment of short and long trends
        signals['MA_Cross_Confirmation'] = np.where(
            (short_trend > 0) & (long_trend > 0), 1.0,  # Both bullish
            np.where((short_trend < 0) & (long_trend < 0), -1.0,  # Both bearish
                    short_trend * 0.5)  # Mixed signals - weight by short-term
        )
        
        return signals.shift(1)  # Prevent lookahead bias
    
    @staticmethod
    def dual_momentum_strategy(prices, short_periods=[5, 10, 20], long_periods=[50, 100, 200]):
        """
        Enhanced dual-window momentum strategy
        
        Combines short-term momentum (reactivity) with long-term momentum (persistence)
        to create more robust trend-following signals.
        
        Args:
            prices: Price series
            short_periods: Short-term momentum lookbacks
            long_periods: Long-term momentum lookbacks
            
        Returns:
            DataFrame with enhanced momentum signals
        """
        # Ensure input is pandas Series
        if not isinstance(prices, pd.Series):
            prices = pd.Series(prices)
        
        signals = pd.DataFrame(index=prices.index)
        
        # Calculate momentum for all periods
        momentums = {}
        for period in short_periods + long_periods:
            momentums[f'MOM_{period}'] = (prices / prices.shift(period)) - 1
        
        # 1. Short-term momentum (reactivity to recent changes)
        short_mom_signals = []
        for period in short_periods:
            mom = momentums[f'MOM_{period}']
            # Add momentum acceleration
            mom_accel = mom - mom.shift(2)
            enhanced_mom = mom * 0.7 + mom_accel * 0.3
            short_mom_signals.append(enhanced_mom)
        
        signals['MOM_Short_Average'] = pd.concat(short_mom_signals, axis=1).mean(axis=1)
        
        # 2. Long-term momentum (trend persistence)
        long_mom_signals = []
        for period in long_periods:
            mom = momentums[f'MOM_{period}']
            # Smooth long-term momentum to reduce noise
            smooth_mom = mom.rolling(window=5).mean()
            long_mom_signals.append(smooth_mom)
        
        signals['MOM_Long_Average'] = pd.concat(long_mom_signals, axis=1).mean(axis=1)
        
        # 3. Dual momentum confirmation
        # Strong signal when both short and long momentum align
        short_mom = signals['MOM_Short_Average']
        long_mom = signals['MOM_Long_Average']
        
        signals['MOM_Dual_Confirmation'] = np.where(
            (short_mom > 0.02) & (long_mom > 0.01), 1.0,  # Both positive
            np.where((short_mom < -0.02) & (long_mom < -0.01), -1.0,  # Both negative
                    (short_mom + long_mom) * 0.5)  # Weighted average for mixed signals
        )
        
        # 4. Momentum divergence signal (contrarian indicator)
        signals['MOM_Divergence'] = short_mom - long_mom
        
        return signals.shift(1)  # Prevent lookahead bias
    
    @staticmethod
    def adaptive_volume_strategy(prices, volumes, short_windows=[5, 10, 20], long_windows=[50, 100]):
        """
        Enhanced volume strategy with adaptive lookback windows
        
        Uses volume patterns across multiple timeframes to identify:
        1. Volume breakouts (increased activity)
        2. Volume trend confirmation
        3. Price-volume divergence signals
        
        Args:
            prices: Price series
            volumes: Volume series
            short_windows: Short-term volume lookbacks
            long_windows: Long-term volume lookbacks
            
        Returns:
            DataFrame with enhanced volume signals
        """
        # Ensure inputs are pandas Series
        if not isinstance(prices, pd.Series):
            prices = pd.Series(prices)
        if not isinstance(volumes, pd.Series):
            volumes = pd.Series(volumes, index=prices.index)
        
        signals = pd.DataFrame(index=prices.index)
        
        # Calculate volume moving averages
        vol_mas = {}
        for window in short_windows + long_windows:
            vol_mas[f'VOL_MA_{window}'] = volumes.rolling(window=window).mean()
        
        # 1. Volume breakout detection
        vol_breakouts = []
        for short in short_windows:
            for long in long_windows:
                if short < long:
                    vol_ratio = vol_mas[f'VOL_MA_{short}'] / vol_mas[f'VOL_MA_{long}']
                    # Significant volume increase indicates breakout potential
                    breakout = np.where(vol_ratio > 1.5, 1.0, 
                                      np.where(vol_ratio < 0.5, -1.0, 0.0))
                    vol_breakouts.append(pd.Series(breakout, index=prices.index))
        
        signals['VOL_Breakout_Score'] = pd.concat(vol_breakouts, axis=1).mean(axis=1)
        
        # 2. Price-Volume trend confirmation
        price_change = prices.pct_change()
        volume_change = volumes.pct_change()
        
        # Strong uptrend: Rising prices with increasing volume
        # Strong downtrend: Falling prices with increasing volume
        trend_confirmation = np.where(
            (price_change > 0) & (volume_change > 0), 1.0,  # Bullish confirmation
            np.where((price_change < 0) & (volume_change > 0), -1.0,  # Bearish confirmation
                    0.0)  # No confirmation
        )
        # Convert to Series and apply rolling mean
        signals['VOL_Trend_Confirmation'] = pd.Series(trend_confirmation, index=prices.index).rolling(window=5).mean()
        
        # 3. Volume-weighted price momentum
        vwap_short = (prices * volumes).rolling(window=20).sum() / volumes.rolling(window=20).sum()
        vwap_long = (prices * volumes).rolling(window=100).sum() / volumes.rolling(window=100).sum()
        
        signals['VOL_VWAP_Signal'] = (vwap_short - vwap_long) / vwap_long
        
        return signals.shift(1)  # Prevent lookahead bias
    
    @staticmethod
    def trend_strength_indicator(prices, short_periods=[10, 20], long_periods=[50, 100, 200]):
        """
        Multi-timeframe trend strength measurement
        
        Quantifies trend strength across multiple time horizons to provide
        confidence scoring for other signals.
        
        Args:
            prices: Price series
            short_periods: Short-term trend measurement periods
            long_periods: Long-term trend measurement periods
            
        Returns:
            DataFrame with trend strength indicators
        """
        # Ensure input is pandas Series
        if not isinstance(prices, pd.Series):
            prices = pd.Series(prices)
        
        signals = pd.DataFrame(index=prices.index)
        
        # 1. Simplified trend strength using price momentum consistency
        # Instead of R-squared calculation which can hang, use momentum consistency
        trend_strengths = []
        for period in short_periods + long_periods:
            # Calculate price change over period
            price_change = prices.pct_change(periods=period)
            # Smooth to reduce noise
            trend_strength = price_change.rolling(window=5).std()  # Use volatility as trend strength proxy
            trend_strengths.append(trend_strength)
            signals[f'TREND_Strength_{period}'] = trend_strength
        
        # 2. Average trend strength across all timeframes
        if trend_strengths:
            signals['TREND_Average_Strength'] = pd.concat(trend_strengths, axis=1).mean(axis=1)
        else:
            signals['TREND_Average_Strength'] = 0
        
        # 3. Trend consistency (how aligned are different timeframes)
        price_directions = []
        for period in [10, 20, 50, 100]:
            direction = np.where(prices > prices.shift(period), 1, -1)
            price_directions.append(pd.Series(direction, index=prices.index))
        
        # Trend consistency: +1 if all agree, -1 if all disagree, 0 if mixed
        direction_sum = pd.concat(price_directions, axis=1).sum(axis=1)
        signals['TREND_Consistency'] = direction_sum / len(price_directions)
        
        return signals.shift(1)  # Prevent lookahead bias
    
    @staticmethod
    def generate_enhanced_features(prices, volumes=None):
        """
        Generate comprehensive enhanced features for ML models
        
        Combines all enhanced strategies into a comprehensive feature set
        that can be used alongside or instead of traditional indicators.
        
        Args:
            prices: Price series
            volumes: Volume series (optional)
            
        Returns:
            DataFrame with all enhanced features
        """
        print("🔧 Generating Enhanced Strategy Features...")
        
        # Initialize results dataframe
        enhanced_features = pd.DataFrame(index=prices.index)
        
        try:
            # 1. Multi-window Moving Averages
            print("  - Multi-window Moving Average signals...")
            ma_signals = EnhancedStrategies.multi_window_moving_average(prices)
            enhanced_features = pd.concat([enhanced_features, ma_signals], axis=1)
            
            # 2. Dual Momentum Strategy
            print("  - Dual-window Momentum signals...")
            mom_signals = EnhancedStrategies.dual_momentum_strategy(prices)
            enhanced_features = pd.concat([enhanced_features, mom_signals], axis=1)
            
            # 3. Trend Strength Indicators
            print("  - Multi-timeframe Trend Strength...")
            trend_signals = EnhancedStrategies.trend_strength_indicator(prices)
            enhanced_features = pd.concat([enhanced_features, trend_signals], axis=1)
            
            # 4. Volume-based signals (if volume data available)
            if volumes is not None:
                print("  - Adaptive Volume strategy signals...")
                vol_signals = EnhancedStrategies.adaptive_volume_strategy(prices, volumes)
                enhanced_features = pd.concat([enhanced_features, vol_signals], axis=1)
            else:
                print("  - Skipping volume signals (no volume data)")
            
            # 5. Feature summary
            print(f"✓ Enhanced features generated: {len(enhanced_features.columns)} features")
            print(f"  - Feature categories: MA, Momentum, Trend Strength{', Volume' if volumes is not None else ''}")
            
            # Remove rows with NaN values
            enhanced_features = enhanced_features.dropna()
            print(f"  - Valid data points: {len(enhanced_features)}")
            
            return enhanced_features
            
        except Exception as e:
            print(f"⚠️  Error generating enhanced features: {e}")
            return pd.DataFrame(index=prices.index)
    
    @staticmethod
    def combine_with_traditional_signals(enhanced_features, traditional_signals, weight_enhanced=0.6):
        """
        Combine enhanced features with traditional technical indicators
        
        Creates a hybrid feature set that combines the best of both approaches.
        
        Args:
            enhanced_features: Enhanced strategy features
            traditional_signals: Traditional technical indicators
            weight_enhanced: Weight given to enhanced features (0.0 to 1.0)
            
        Returns:
            Combined feature set
        """
        print(f"\n🔄 Combining Enhanced + Traditional Signals...")
        print(f"  - Enhanced features weight: {weight_enhanced:.1%}")
        print(f"  - Traditional signals weight: {(1-weight_enhanced):.1%}")
        
        # Align indices
        common_index = enhanced_features.index.intersection(traditional_signals.index)
        enhanced_aligned = enhanced_features.loc[common_index]
        traditional_aligned = traditional_signals.loc[common_index]
        
        # Create combined feature set
        combined_features = pd.DataFrame(index=common_index)
        
        # Add enhanced features with weight
        for col in enhanced_aligned.columns:
            combined_features[f'ENH_{col}'] = enhanced_aligned[col] * weight_enhanced
        
        # Add traditional features with weight
        for col in traditional_aligned.columns:
            combined_features[f'TRAD_{col}'] = traditional_aligned[col] * (1 - weight_enhanced)
        
        print(f"✓ Combined features: {len(combined_features.columns)} total features")
        print(f"  - Enhanced: {len(enhanced_aligned.columns)} features")
        print(f"  - Traditional: {len(traditional_aligned.columns)} features")
        
        return combined_features 