#!/usr/bin/env python3
"""
Enhanced Strategy Integration Module

This module demonstrates how to integrate the enhanced strategies with the existing
PLS ML approach to improve overall trading performance.

Integration Approaches:
1. Feature Enhancement: Add enhanced features to existing ML models
2. Signal Combination: Combine enhanced signals with traditional indicators
3. Multi-Model Ensemble: Use enhanced strategies as separate models
4. Adaptive Weighting: Dynamically weight strategies based on market conditions

Usage Examples:
- Enhanced PLS with multi-window features
- Hybrid traditional + enhanced signal generation
- Performance comparison between approaches
- Dynamic strategy selection based on market regime

Author: Integration module for enhanced crypto strategies
"""

import pandas as pd
import numpy as np
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_strategies import EnhancedStrategies
from technical_indicators import TechnicalIndicators
from ml_methods import MLMethodsComparison


class EnhancedIntegration:
    """
    Integration class for combining enhanced strategies with existing PLS approach
    """
    
    def __init__(self):
        """Initialize integration components"""
        self.enhanced_strategies = EnhancedStrategies()
        self.technical_indicators = TechnicalIndicators()
        self.ml_methods = MLMethodsComparison()
        
        # Storage for different feature sets
        self.traditional_features = None
        self.enhanced_features = None
        self.combined_features = None
        
        # Performance tracking
        self.performance_comparison = {}
    
    def generate_feature_sets(self, market_data):
        """
        Generate different feature sets for comparison
        
        Args:
            market_data: DataFrame with price and volume data
            
        Returns:
            Dictionary containing different feature sets
        """
        print("\n=== ENHANCED FEATURE GENERATION ===")
        
        prices = market_data['price']
        volumes = market_data.get('volume', None)
        
        feature_sets = {}
        
        # 1. Traditional features (existing approach)
        print("\n📊 1. Traditional Technical Indicators...")
        self.traditional_features = self.technical_indicators.create_frequency_signals(
            market_data, frequency='W'
        )
        feature_sets['traditional'] = self.traditional_features
        print(f"✓ Generated {len(self.traditional_features.columns)} traditional features")
        
        # 2. Enhanced features (new multi-window approach)
        print("\n🔧 2. Enhanced Multi-Window Features...")
        self.enhanced_features = EnhancedStrategies.generate_enhanced_features(
            prices, volumes
        )
        feature_sets['enhanced'] = self.enhanced_features
        print(f"✓ Generated {len(self.enhanced_features.columns)} enhanced features")
        
        # 3. Combined features (hybrid approach)
        print("\n🔄 3. Combined Feature Set...")
        if len(self.traditional_features) > 0 and len(self.enhanced_features) > 0:
            self.combined_features = EnhancedStrategies.combine_with_traditional_signals(
                self.enhanced_features, self.traditional_features, weight_enhanced=0.6
            )
            feature_sets['combined'] = self.combined_features
            print(f"✓ Generated {len(self.combined_features.columns)} combined features")
        else:
            print("⚠️  Cannot combine features - insufficient data")
        
        return feature_sets
    
    def compare_ml_methods_enhanced(self, feature_sets, returns, test_size=0.3):
        """
        Compare ML methods using different feature sets
        
        Args:
            feature_sets: Dictionary of feature sets to test
            returns: Target returns for prediction
            test_size: Fraction of data for testing
            
        Returns:
            Comprehensive comparison results
        """
        print("\n=== ENHANCED ML METHOD COMPARISON ===")
        
        comparison_results = {}
        
        for feature_name, features in feature_sets.items():
            if features is None or len(features) == 0:
                print(f"⚠️  Skipping {feature_name} - no features available")
                continue
                
            print(f"\n🧪 Testing with {feature_name.upper()} features...")
            print(f"   Features: {len(features.columns)}, Data points: {len(features)}")
            
            try:
                # Align features and returns
                common_index = features.index.intersection(returns.index)
                if len(common_index) < 50:
                    print(f"   ⚠️  Insufficient data points: {len(common_index)}")
                    continue
                
                features_aligned = features.loc[common_index]
                returns_aligned = returns.loc[common_index]
                
                # Run ML comparison for this feature set
                ml_results = self.ml_methods.compare_methods(
                    features_aligned, returns_aligned, test_size=test_size
                )
                
                comparison_results[feature_name] = ml_results
                
                # Display best method for this feature set
                if 'method_performance' in ml_results:
                    best_method = max(ml_results['method_performance'].items(), 
                                    key=lambda x: x[1]['sharpe_ratio'])
                    print(f"   ✓ Best method: {best_method[0]} (Sharpe: {best_method[1]['sharpe_ratio']:.3f})")
                
            except Exception as e:
                print(f"   ❌ Error testing {feature_name}: {e}")
                continue
        
        return comparison_results
    
    def analyze_feature_importance(self, feature_sets, returns):
        """
        Analyze which types of features are most important for prediction
        
        Args:
            feature_sets: Dictionary of feature sets
            returns: Target returns
            
        Returns:
            Feature importance analysis
        """
        print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
        
        importance_results = {}
        
        for feature_name, features in feature_sets.items():
            if features is None or len(features) == 0:
                continue
                
            print(f"\n📊 Analyzing {feature_name.upper()} feature importance...")
            
            try:
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.preprocessing import StandardScaler
                
                # Align data
                common_index = features.index.intersection(returns.index)
                X = features.loc[common_index].fillna(0)
                y = returns.loc[common_index]
                
                if len(X) < 30:
                    print(f"   ⚠️  Insufficient data: {len(X)} points")
                    continue
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Train Random Forest for feature importance
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(X_scaled, y)
                
                # Get feature importance
                importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance': rf.feature_importances_
                }).sort_values('importance', ascending=False)
                
                importance_results[feature_name] = importance_df
                
                # Display top features
                print(f"   ✓ Top 5 features:")
                for idx, row in importance_df.head().iterrows():
                    print(f"     {row['feature']}: {row['importance']:.3f}")
                
            except Exception as e:
                print(f"   ❌ Error analyzing {feature_name}: {e}")
                continue
        
        return importance_results
    
    def create_ensemble_strategy(self, feature_sets, returns, weights=None):
        """
        Create an ensemble strategy combining multiple approaches
        
        Args:
            feature_sets: Dictionary of feature sets
            returns: Target returns
            weights: Optional weights for each approach
            
        Returns:
            Ensemble predictions and performance
        """
        print("\n=== ENSEMBLE STRATEGY CREATION ===")
        
        if weights is None:
            weights = {'traditional': 0.3, 'enhanced': 0.4, 'combined': 0.3}
        
        ensemble_predictions = {}
        ensemble_performance = {}
        
        # Generate predictions from each approach
        for feature_name, features in feature_sets.items():
            if features is None or len(features) == 0:
                continue
                
            weight = weights.get(feature_name, 0.0)
            if weight == 0.0:
                continue
                
            print(f"\n🔮 Generating predictions: {feature_name.upper()} (weight: {weight:.1%})")
            
            try:
                # Align data
                common_index = features.index.intersection(returns.index)
                X = features.loc[common_index].fillna(0)
                y = returns.loc[common_index]
                
                if len(X) < 50:
                    print(f"   ⚠️  Insufficient data: {len(X)} points")
                    continue
                
                # Use PLS as the ML method (best performer from existing research)
                from sklearn.cross_decomposition import PLSRegression
                from sklearn.preprocessing import StandardScaler
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Split data
                split_idx = int(len(X_scaled) * 0.7)
                X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                
                # Train PLS model
                pls = PLSRegression(n_components=3)  # Use 3 components (optimal from research)
                pls.fit(X_train, y_train)
                
                # Generate predictions
                predictions = pls.predict(X_test).flatten()
                ensemble_predictions[feature_name] = pd.Series(
                    predictions, index=y_test.index
                )
                
                # Calculate performance
                returns_test = y_test.values
                sharpe = np.mean(predictions * returns_test) / np.std(predictions * returns_test) if np.std(predictions * returns_test) > 0 else 0
                ensemble_performance[feature_name] = {
                    'sharpe': sharpe,
                    'weight': weight,
                    'predictions': predictions
                }
                
                print(f"   ✓ Sharpe ratio: {sharpe:.3f}")
                
            except Exception as e:
                print(f"   ❌ Error with {feature_name}: {e}")
                continue
        
        # Combine predictions using weights
        if len(ensemble_predictions) > 1:
            print(f"\n🎯 Creating weighted ensemble...")
            
            # Find common index across all predictions
            common_idx = None
            for pred in ensemble_predictions.values():
                if common_idx is None:
                    common_idx = pred.index
                else:
                    common_idx = common_idx.intersection(pred.index)
            
            if len(common_idx) > 0:
                weighted_predictions = pd.Series(0.0, index=common_idx)
                total_weight = 0.0
                
                for feature_name, predictions in ensemble_predictions.items():
                    weight = ensemble_performance[feature_name]['weight']
                    weighted_predictions += predictions.loc[common_idx] * weight
                    total_weight += weight
                
                if total_weight > 0:
                    weighted_predictions /= total_weight
                    
                    # Calculate ensemble performance
                    ensemble_returns = returns.loc[common_idx]
                    ensemble_sharpe = np.mean(weighted_predictions * ensemble_returns) / np.std(weighted_predictions * ensemble_returns) if np.std(weighted_predictions * ensemble_returns) > 0 else 0
                    
                    print(f"   ✓ Ensemble Sharpe ratio: {ensemble_sharpe:.3f}")
                    print(f"   ✓ Combined {len(ensemble_predictions)} approaches")
                    
                    ensemble_performance['ensemble'] = {
                        'sharpe': ensemble_sharpe,
                        'predictions': weighted_predictions,
                        'components': list(ensemble_predictions.keys())
                    }
        
        return ensemble_predictions, ensemble_performance
    
    def generate_enhanced_strategy_report(self, comparison_results, importance_results, ensemble_performance):
        """
        Generate comprehensive report comparing all approaches
        
        Args:
            comparison_results: ML method comparison results
            importance_results: Feature importance analysis
            ensemble_performance: Ensemble strategy performance
            
        Returns:
            Formatted report
        """
        print("\n" + "="*80)
        print("ENHANCED STRATEGY INTEGRATION REPORT")
        print("="*80)
        
        report = {
            'timestamp': pd.Timestamp.now(),
            'approach_comparison': {},
            'feature_analysis': {},
            'ensemble_results': {},
            'recommendations': []
        }
        
        # 1. Approach Performance Comparison
        print("\n📊 APPROACH PERFORMANCE COMPARISON")
        print("-" * 50)
        
        for approach, results in comparison_results.items():
            if 'method_performance' in results:
                best_method = max(results['method_performance'].items(), 
                                key=lambda x: x[1]['sharpe_ratio'])
                performance = best_method[1]
                
                print(f"{approach.upper():>15}: {best_method[0]:>10} | Sharpe: {performance['sharpe_ratio']:>6.3f} | Accuracy: {performance.get('directional_accuracy', 0):.1%}")
                
                report['approach_comparison'][approach] = {
                    'best_method': best_method[0],
                    'sharpe_ratio': performance['sharpe_ratio'],
                    'directional_accuracy': performance.get('directional_accuracy', 0)
                }
        
        # 2. Feature Category Analysis
        print(f"\n🔍 FEATURE CATEGORY IMPORTANCE")
        print("-" * 50)
        
        for approach, importance_df in importance_results.items():
            if len(importance_df) > 0:
                # Categorize features
                ma_features = importance_df[importance_df['feature'].str.contains('MA|ma', na=False)]
                mom_features = importance_df[importance_df['feature'].str.contains('MOM|mom', na=False)]
                vol_features = importance_df[importance_df['feature'].str.contains('VOL|vol', na=False)]
                trend_features = importance_df[importance_df['feature'].str.contains('TREND|trend', na=False)]
                
                categories = {
                    'Moving Average': ma_features['importance'].sum() if len(ma_features) > 0 else 0,
                    'Momentum': mom_features['importance'].sum() if len(mom_features) > 0 else 0,
                    'Volume': vol_features['importance'].sum() if len(vol_features) > 0 else 0,
                    'Trend': trend_features['importance'].sum() if len(trend_features) > 0 else 0
                }
                
                print(f"\n{approach.upper()} Feature Categories:")
                for category, importance in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                    if importance > 0:
                        print(f"  {category:>15}: {importance:.3f}")
                
                report['feature_analysis'][approach] = categories
        
        # 3. Ensemble Performance
        if 'ensemble' in ensemble_performance:
            ensemble_perf = ensemble_performance['ensemble']
            print(f"\n🎯 ENSEMBLE STRATEGY RESULTS")
            print("-" * 50)
            print(f"Ensemble Sharpe Ratio: {ensemble_perf['sharpe']:.3f}")
            print(f"Components Combined: {', '.join(ensemble_perf['components'])}")
            
            report['ensemble_results'] = {
                'sharpe_ratio': ensemble_perf['sharpe'],
                'components': ensemble_perf['components']
            }
        
        # 4. Recommendations
        print(f"\n💡 STRATEGIC RECOMMENDATIONS")
        print("-" * 50)
        
        recommendations = []
        
        # Best single approach
        if report['approach_comparison']:
            best_approach = max(report['approach_comparison'].items(), 
                              key=lambda x: x[1]['sharpe_ratio'])
            recommendations.append(f"Best single approach: {best_approach[0].upper()} (Sharpe: {best_approach[1]['sharpe_ratio']:.3f})")
        
        # Feature insights
        if report['feature_analysis']:
            # Find most important feature category across all approaches
            all_categories = {}
            for analysis in report['feature_analysis'].values():
                for category, importance in analysis.items():
                    all_categories[category] = all_categories.get(category, 0) + importance
            
            if all_categories:
                top_category = max(all_categories.items(), key=lambda x: x[1])
                recommendations.append(f"Most important feature type: {top_category[0]} indicators")
        
        # Ensemble recommendation
        if 'ensemble' in ensemble_performance:
            ensemble_sharpe = ensemble_performance['ensemble']['sharpe']
            single_best_sharpe = max([perf['sharpe_ratio'] for perf in report['approach_comparison'].values()] + [0])
            
            if ensemble_sharpe > single_best_sharpe:
                improvement = ((ensemble_sharpe / single_best_sharpe) - 1) * 100 if single_best_sharpe > 0 else 0
                recommendations.append(f"Ensemble approach recommended: {improvement:.1f}% improvement over best single method")
            else:
                recommendations.append("Single best approach recommended over ensemble")
        
        recommendations.append("Consider dynamic weighting based on market volatility")
        recommendations.append("Monitor feature importance changes over time")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        report['recommendations'] = recommendations
        
        print("\n" + "="*80)
        
        return report

    def create_enhanced_strategy_demo(self, market_data):
        """Demonstrate enhanced strategies integration"""
        print("\n" + "="*80)
        print("ENHANCED STRATEGIES DEMONSTRATION")
        print("="*80)
        
        try:
            # Generate all feature sets
            feature_sets = self.generate_feature_sets(market_data)
            
            # Display feature comparison
            print("\n📊 FEATURE SET COMPARISON:")
            print("-" * 50)
            
            for name, features in feature_sets.items():
                if features is not None and len(features) > 0:
                    print(f"{name.upper():>12}: {len(features.columns):>3} features | {len(features):>4} data points")
                else:
                    print(f"{name.upper():>12}: No data available")
            
            # Performance summary
            print(f"\n💡 ENHANCEMENT BENEFITS:")
            print("-" * 50)
            print("✓ Multi-timeframe trend analysis (5-200 period lookbacks)")
            print("✓ Dual momentum detection (short + long term)")
            print("✓ Adaptive volume confirmation signals") 
            print("✓ Trend strength quantification across timeframes")
            print("✓ Cross-timeframe signal confirmation")
            print("✓ Reduced false signals through multi-window validation")
            
            return feature_sets
            
        except Exception as e:
            print(f"❌ Error in enhanced strategy demo: {e}")
            return None 