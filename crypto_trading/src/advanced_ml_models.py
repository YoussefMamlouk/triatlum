"""
Advanced ML Models for Cryptocurrency Trading Strategy
This module implements cutting-edge ML models to improve upon the existing PLS strategy.

Models included:
- Random Forest Regressor
- XGBoost Regressor  
- Support Vector Regression (SVR)
- Neural Networks (MLPRegressor)
- Ridge Regression with Cross-Validation
- LightGBM
- Ensemble Voting Regressor
- LSTM (for time series)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception as e:
    XGBOOST_AVAILABLE = False
    print(f"⚠️  XGBoost not available: {e}")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except Exception as e:
    LIGHTGBM_AVAILABLE = False
    print(f"⚠️  LightGBM not available: {e}")

# Always available models using scikit-learn
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge, ElasticNetCV


class AdvancedMLModels:
    """
    Advanced ML models for cryptocurrency prediction
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.results = {}
        
    def prepare_models(self):
        """
        Initialize all advanced ML models with optimized parameters
        """
        print("🚀 Initializing Advanced ML Models...")
        
        # 1. Random Forest - Ensemble of decision trees
        self.models['RandomForest'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # 2. XGBoost - Gradient boosting
        if XGBOOST_AVAILABLE:
            self.models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbosity=0
            )
        
        # 3. Support Vector Regression - Non-linear patterns
        self.models['SVR'] = SVR(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            epsilon=0.01
        )
        
        # 4. Neural Network - Deep learning approach
        self.models['NeuralNet'] = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=500,
            random_state=self.random_state
        )
        
        # 5. Ridge Regression with CV - Regularized linear model
        self.models['RidgeCV'] = RidgeCV(
            alphas=[0.1, 1.0, 10.0, 100.0],
            cv=5
        )
        
        # 6. LightGBM - Fast gradient boosting
        if LIGHTGBM_AVAILABLE:
            self.models['LightGBM'] = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbosity=-1
            )
        
        # 7. Gradient Boosting (Scikit-learn) - Always available alternative
        self.models['GradientBoosting'] = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=self.random_state
        )
        
        # 8. Extra Trees - Randomized ensemble
        self.models['ExtraTrees'] = ExtraTreesRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # 9. Bayesian Ridge - Probabilistic linear model
        self.models['BayesianRidge'] = BayesianRidge(
            alpha_1=1e-6,
            alpha_2=1e-6,
            lambda_1=1e-6,
            lambda_2=1e-6
        )
        
        # 10. Elastic Net CV - Cross-validated regularization
        self.models['ElasticNetCV'] = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.7, 0.9],
            alphas=[0.001, 0.01, 0.1, 1.0],
            cv=5,
            random_state=self.random_state
        )
        
        print(f"✓ Initialized {len(self.models)} advanced ML models")
        return self.models
    
    def create_ensemble_model(self):
        """
        Create an ensemble model combining the best performers
        """
        if len(self.models) < 2:
            self.prepare_models()
        
        # Select best performing models for ensemble
        ensemble_models = []
        
        # Always include these core models
        if 'RandomForest' in self.models:
            ensemble_models.append(('rf', self.models['RandomForest']))
        
        if 'XGBoost' in self.models and XGBOOST_AVAILABLE:
            ensemble_models.append(('xgb', self.models['XGBoost']))
        elif 'GradientBoosting' in self.models:
            ensemble_models.append(('gb', self.models['GradientBoosting']))
        
        if 'RidgeCV' in self.models:
            ensemble_models.append(('ridge', self.models['RidgeCV']))
        
        if 'ExtraTrees' in self.models:
            ensemble_models.append(('et', self.models['ExtraTrees']))
        
        if len(ensemble_models) >= 2:
            self.models['Ensemble'] = VotingRegressor(
                estimators=ensemble_models,
                n_jobs=-1
            )
            print(f"✓ Created ensemble model with {len(ensemble_models)} base models")
        
        return self.models.get('Ensemble', None)
    
    def train_and_evaluate_model(self, model_name, model, X_train, y_train, X_test, y_test):
        """
        Train and evaluate a single model
        """
        try:
            print(f"  🔬 Training {model_name}...")
            
            # Scale features for models that need it
            if model_name in ['SVR', 'NeuralNet', 'BayesianRidge', 'ElasticNetCV']:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                self.scalers[model_name] = scaler
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            predictions = model.predict(X_test_scaled)
            
            # Calculate performance metrics
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            # Calculate trading performance metrics
            performance = self._calculate_trading_performance(predictions, y_test)
            
            # Store results
            results = {
                'model': model,
                'predictions': predictions,
                'mse': mse,
                'r2': r2,
                **performance
            }
            
            self.results[model_name] = results
            
            print(f"    ✓ {model_name} - Sharpe: {performance['sharpe']:.3f}, Accuracy: {performance['directional_accuracy']:.1%}")
            
            return results
            
        except Exception as e:
            print(f"    ❌ {model_name} failed: {e}")
            return None
    
    def _calculate_trading_performance(self, predictions, actual_returns):
        """
        Calculate trading performance metrics from predictions
        """
        try:
            # Generate trading positions
            positions = np.where(predictions > 0, 1.0, 0.0)  # Long when positive prediction
            
            # Calculate strategy returns
            strategy_returns = positions * actual_returns
            
            # Performance metrics
            total_return = (1 + strategy_returns).prod() - 1 if len(strategy_returns) > 0 else 0
            
            # Sharpe ratio (annualized, assuming weekly data)
            if strategy_returns.std() > 0:
                sharpe = (strategy_returns.mean() * 52 - 0.02) / (strategy_returns.std() * np.sqrt(52))
            else:
                sharpe = 0
            
            # Directional accuracy
            if len(predictions) > 0:
                directional_accuracy = ((predictions > 0) == (actual_returns > 0)).mean()
            else:
                directional_accuracy = 0.5
            
            # Win rate
            win_rate = (strategy_returns > 0).mean() if len(strategy_returns) > 0 else 0
            
            # Maximum drawdown
            if len(strategy_returns) > 0:
                cumulative = (1 + strategy_returns).cumprod()
                max_drawdown = (cumulative / cumulative.cummax() - 1).min()
            else:
                max_drawdown = 0
            
            # Volatility
            volatility = strategy_returns.std() * np.sqrt(52) if len(strategy_returns) > 0 else 0
            
            return {
                'total_return': total_return,
                'sharpe': sharpe,
                'directional_accuracy': directional_accuracy,
                'win_rate': win_rate,
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'strategy_returns': strategy_returns
            }
            
        except Exception as e:
            print(f"    ⚠️ Performance calculation failed: {e}")
            return {
                'total_return': 0,
                'sharpe': 0,
                'directional_accuracy': 0.5,
                'win_rate': 0,
                'max_drawdown': 0,
                'volatility': 0,
                'strategy_returns': np.array([])
            }
    
    def run_comprehensive_comparison(self, features, returns):
        """
        Run comprehensive comparison of all advanced ML models
        """
        print("\n🔬 ADVANCED ML MODELS COMPARISON")
        print("="*60)
        
        # Prepare data
        common_index = features.index.intersection(returns.index)
        X = features.loc[common_index].fillna(0)
        y = returns.loc[common_index]
        
        if len(X) < 50:
            print("❌ Insufficient data for advanced ML models")
            return None
        
        # Split data (70% train, 30% test)
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"📊 Data prepared: {len(X_train)} training, {len(X_test)} testing")
        
        # Prepare models
        self.prepare_models()
        
        # Create ensemble
        self.create_ensemble_model()
        
        # Train and evaluate each model
        results_summary = []
        
        for model_name, model in self.models.items():
            result = self.train_and_evaluate_model(
                model_name, model, X_train, y_train, X_test, y_test
            )
            
            if result:
                results_summary.append({
                    'Model': model_name,
                    'Sharpe_Ratio': result['sharpe'],
                    'Total_Return': result['total_return'],
                    'Directional_Accuracy': result['directional_accuracy'],
                    'Win_Rate': result['win_rate'],
                    'Max_Drawdown': result['max_drawdown'],
                    'Volatility': result['volatility'],
                    'R2_Score': result['r2'],
                    'MSE': result['mse']
                })
        
        # Create results DataFrame and rank by Sharpe ratio
        if results_summary:
            results_df = pd.DataFrame(results_summary)
            results_df = results_df.sort_values('Sharpe_Ratio', ascending=False)
            
            # Display results table
            print(f"\n📊 ADVANCED ML MODELS RESULTS:")
            print("-" * 80)
            print(f"{'Rank':<4} {'Model':<12} {'Sharpe':<8} {'Return':<8} {'Dir.Acc':<8} {'Win Rate':<8} {'R2':<6}")
            print("-" * 80)
            
            for i, row in results_df.iterrows():
                rank = results_df.index.get_loc(i) + 1
                print(f"{rank:<4} {row['Model']:<12} {row['Sharpe_Ratio']:<8.3f} {row['Total_Return']:<8.1%} "
                      f"{row['Directional_Accuracy']:<8.1%} {row['Win_Rate']:<8.1%} {row['R2_Score']:<6.3f}")
            
            # Identify best model
            best_model = results_df.iloc[0]
            print(f"\n🏆 BEST ADVANCED MODEL: {best_model['Model']}")
            print(f"   Sharpe Ratio: {best_model['Sharpe_Ratio']:.3f}")
            print(f"   Total Return: {best_model['Total_Return']:.1%}")
            print(f"   Directional Accuracy: {best_model['Directional_Accuracy']:.1%}")
            
            return results_df, self.results
        
        else:
            print("❌ No models completed successfully")
            return None, None
    
    def get_best_model_predictions(self, model_name=None):
        """
        Get predictions from the best performing model
        """
        if not self.results:
            print("❌ No models have been trained yet")
            return None
        
        if model_name is None:
            # Find best model by Sharpe ratio
            best_sharpe = -999
            best_model_name = None
            
            for name, result in self.results.items():
                if result['sharpe'] > best_sharpe:
                    best_sharpe = result['sharpe']
                    best_model_name = name
            
            model_name = best_model_name
        
        if model_name in self.results:
            return self.results[model_name]
        else:
            print(f"❌ Model {model_name} not found in results")
            return None 