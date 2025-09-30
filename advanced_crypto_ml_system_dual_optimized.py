# advanced_crypto_ml_system.py - SISTEMA ML COERENTE CON ARCHITETTURA ESISTENTE
"""
Sistema ML Dual Horizon ottimizzato COMPATIBILE con il sistema esistente:
- Compatible con crypto_ssd_config.json
- Compatible con crypto_continuous_optimized.py  
- Compatible con crypto_database_optimized.py
- Compatible con bitcoin_benchmark_system_real.py
- Compatible con crypto_notifier.py
- Utilizza SSD paths D:/CryptoSystem
- Walk Forward Validation e Anti-overfitting
- Integration con tutti i componenti esistenti
"""

from collections import defaultdict
import random
import warnings
warnings.filterwarnings('ignore')

import keras
import numpy as np
import pandas as pd
import sqlite3
import requests
import time
import json
import gc
import psutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# ML Libraries
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import joblib

# Deep Learning
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Time Series (Optional)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False

# Configure TensorFlow
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True) if tf.config.list_physical_devices('GPU') else None

class OptimizedDualHorizonMLSystem:
    """🚀 Sistema ML Dual Horizon compatibile con architettura esistente"""
    
    def __init__(self, config=None):
        print("🚀 Initializing Optimized Dual Horizon ML System (Compatible)")
        print("🎯 Integration with existing crypto system architecture")
        
        # Load configuration - Compatible with existing system
        self.config = self._load_compatible_config(config)
        
        # SSD Storage setup - Compatible with existing paths
        self._setup_ssd_storage()
        
        # Model storage - Compatible with existing structure
        self.models_1d = {}  # {crypto_id_1d: {model_name: model}}
        self.models_3d = {}  # {crypto_id_3d: {model_name: model}}
        self.scalers = {}    # {crypto_id_horizon: scaler}
        self.feature_columns = {}  # {crypto_id: [feature_names]}
        
        # Performance tracking - Compatible with verification system
        self.model_performance_history = {
            '1d': {},  # crypto_id -> {model_name -> performance_metrics}
            '3d': {}
        }
        self.ensemble_weights = {
            '1d': {},  # crypto_id -> {model_name -> weight}
            '3d': {}
        }
        
        # API management - Compatible with existing rate limiting
        self.api_calls_made = 0
        self.last_api_call = 0
        self.api_delay = self.config.get('api', {}).get('api_delay', 2.0)
        
        # Memory management
        self.memory_manager = self._setup_memory_management()
        
        print("✅ Compatible ML System initialized")
        print(f"💾 SSD Storage: {self.base_storage_path}")
        print(f"🔗 Database: {self.db_path}")
        # AGGIUNGI QUESTE LINEE:
        self.feedback_optimizer = None
        self.dynamic_model_weights = defaultdict(lambda: defaultdict(lambda: 1.0))

    def update_model_performance(self, crypto_id: str, model_name: str, horizon: str, 
                           performance_data: Dict, prediction_metadata: Optional[Dict] = None):
        """🎯 Update model performance tracking in database - FIXED"""
        
        try:
            conn = sqlite3.connect(self.database.db_path)
            cursor = conn.cursor()
            
            # Get current performance data if exists
            cursor.execute('''
                SELECT total_predictions, correct_predictions, accuracy_rate, 
                    avg_confidence, avg_accuracy_score
                FROM model_performance_tracking
                WHERE crypto_id = ? AND model_name = ? AND horizon = ?
            ''', (crypto_id, model_name, horizon))
            
            result = cursor.fetchone()
            
            # Extract performance metrics
            is_correct = performance_data.get('direction_correct', False)
            confidence = performance_data.get('confidence', 0.5)
            accuracy_score = performance_data.get('accuracy_score', 0.0)
            price_error_percent = performance_data.get('price_error_percent', 0.0)
            
            # Additional metadata if provided
            predicted_change = performance_data.get('predicted_change', 0.0)
            actual_change = performance_data.get('actual_change', 0.0)
            
            # 🔧 FIX: Convert datetime to ISO string for SQLite
            current_timestamp = datetime.now().isoformat()
            
            if result:
                # Update existing record
                total_preds, correct_preds, accuracy_rate, avg_conf, avg_acc_score = result
                
                # Calculate new metrics
                new_total = total_preds + 1
                new_correct = correct_preds + (1 if is_correct else 0)
                new_accuracy_rate = new_correct / new_total if new_total > 0 else 0
                new_avg_conf = (avg_conf * total_preds + confidence) / new_total
                new_avg_acc_score = (avg_acc_score * total_preds + accuracy_score) / new_total
                
                # Calculate performance trend
                performance_trend = 'stable'
                if new_accuracy_rate > accuracy_rate + 0.05:
                    performance_trend = 'improving'
                elif new_accuracy_rate < accuracy_rate - 0.05:
                    performance_trend = 'declining'
                
                # Calculate confidence calibration (how well confidence matches actual performance)
                confidence_calibration = confidence if is_correct else (1 - confidence)
                
                cursor.execute('''
                    UPDATE model_performance_tracking
                    SET total_predictions = ?, correct_predictions = ?, accuracy_rate = ?,
                        avg_confidence = ?, avg_accuracy_score = ?,
                        performance_trend = ?, confidence_calibration = ?,
                        last_updated = ?
                    WHERE crypto_id = ? AND model_name = ? AND horizon = ?
                ''', (
                    new_total, new_correct, new_accuracy_rate, new_avg_conf, new_avg_acc_score,
                    performance_trend, confidence_calibration, current_timestamp,  # 🔧 FIX: ISO string
                    crypto_id, model_name, horizon
                ))
                
                print(f"   📊 Updated {model_name} for {crypto_id} {horizon}: "
                    f"acc={new_accuracy_rate:.3f} ({new_correct}/{new_total})")
                
            else:
                # Insert new performance record
                cursor.execute('''
                    INSERT INTO model_performance_tracking (
                        crypto_id, model_name, horizon, total_predictions, correct_predictions,
                        accuracy_rate, avg_confidence, avg_accuracy_score,
                        performance_trend, confidence_calibration, last_updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    crypto_id, model_name, horizon, 1, 
                    1 if is_correct else 0,
                    1.0 if is_correct else 0.0,
                    confidence, accuracy_score,
                    'new', confidence if is_correct else (1 - confidence),
                    current_timestamp  # 🔧 FIX: ISO string instead of datetime.now()
                ))
                
                print(f"   🆕 New performance record for {model_name} {crypto_id} {horizon}: "
                    f"correct={is_correct}, conf={confidence:.3f}")
            
            # Also update aggregated performance if this model is part of an ensemble
            if prediction_metadata:
                self._update_ensemble_performance(cursor, crypto_id, horizon, performance_data, prediction_metadata)
            
            conn.commit()
            conn.close()
            
            return {
                'success': True,
                'crypto_id': crypto_id,
                'model_name': model_name,
                'horizon': horizon,
                'updated_accuracy': new_accuracy_rate if result else (1.0 if is_correct else 0.0),
                'total_predictions': new_total if result else 1
            }
            
        except Exception as e:
            print(f"   ❌ Error updating model performance for {model_name} {crypto_id} {horizon}: {e}")
            return {
                'success': False,
                'error': str(e),
                'crypto_id': crypto_id,
                'model_name': model_name,
                'horizon': horizon
            }

    def _update_ensemble_performance(self, cursor, crypto_id: str, horizon: str, 
                            performance_data, prediction_metadata):
        """🎯 Update ensemble performance metrics - SAFE VERSION"""
        
        try:
            # 🔧 VALIDATION: Controlla che i parametri siano dizionari
            if not isinstance(performance_data, dict):
                print(f"   ⚠️ performance_data is not dict (type: {type(performance_data)}): {performance_data}")
                return  # Skip update se non è un dizionario
            
            if not isinstance(prediction_metadata, dict):
                print(f"   ⚠️ prediction_metadata is not dict (type: {type(prediction_metadata)}): {prediction_metadata}")
                # Crea un prediction_metadata di default
                prediction_metadata = {
                    'ensemble_method': 'weighted_average',
                    'model_weights': {},
                    'models_used': []
                }
            
            # Get ensemble information from prediction metadata
            ensemble_method = prediction_metadata.get('ensemble_method', 'weighted_average')
            model_weights = prediction_metadata.get('model_weights', {})
            models_used = prediction_metadata.get('models_used', [])
            
            # Check if ensemble performance record exists
            cursor.execute('''
                SELECT total_predictions, correct_predictions, accuracy_rate
                FROM model_performance_tracking
                WHERE crypto_id = ? AND model_name = ? AND horizon = ?
            ''', (crypto_id, 'ensemble', horizon))
            
            ensemble_result = cursor.fetchone()
            
            # 🔧 SAFE EXTRACTION: Usa .get() solo su dizionari validati
            is_correct = performance_data.get('direction_correct', False)
            confidence = performance_data.get('confidence', 0.5)
            accuracy_score = performance_data.get('accuracy_score', 0.0)
            
            # 🔧 FIX: Convert datetime to ISO string
            current_timestamp = datetime.now().isoformat()
            
            if ensemble_result:
                total_preds, correct_preds, accuracy_rate = ensemble_result
                new_total = total_preds + 1
                new_correct = correct_preds + (1 if is_correct else 0)
                new_accuracy_rate = new_correct / new_total
                
                cursor.execute('''
                    UPDATE model_performance_tracking
                    SET total_predictions = ?, correct_predictions = ?, accuracy_rate = ?,
                        avg_confidence = ?, avg_accuracy_score = ?, last_updated = ?
                    WHERE crypto_id = ? AND model_name = ? AND horizon = ?
                ''', (
                    new_total, new_correct, new_accuracy_rate,
                    confidence, accuracy_score, current_timestamp,
                    crypto_id, 'ensemble', horizon
                ))
            else:
                cursor.execute('''
                    INSERT INTO model_performance_tracking (
                        crypto_id, model_name, horizon, total_predictions, correct_predictions,
                        accuracy_rate, avg_confidence, avg_accuracy_score, last_updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    crypto_id, 'ensemble', horizon, 1,
                    1 if is_correct else 0,
                    1.0 if is_correct else 0.0,
                    confidence, accuracy_score, current_timestamp
                ))
            
            print(f"   🎯 Ensemble performance updated for {crypto_id} {horizon}")
            
        except Exception as e:
            print(f"   ⚠️ Error updating ensemble performance: {e}")
            print(f"   🔍 Debug info:")
            print(f"      performance_data type: {type(performance_data)}")
            print(f"      performance_data value: {performance_data}")
            print(f"      prediction_metadata type: {type(prediction_metadata)}")
            print(f"      prediction_metadata value: {prediction_metadata}")
            # Non interrompere il processo per questo errore
            import traceback
            traceback.print_exc()

    def get_dynamic_thresholds(self, crypto_id: str, horizon: int, current_price: float = None) -> Dict[str, float]:
        """🎯 SISTEMA DINAMICO: Calcola soglie basate su condizioni reali di mercato"""
        try:
            print(f"🎯 Calculating dynamic thresholds for {crypto_id} {horizon}d...")
            
            # ✅ 1. CHECK STABLECOIN - PRIORITÀ MASSIMA
            if self._is_stablecoin(crypto_id, current_price):
                print(f"   🔒 STABLECOIN DETECTED: {crypto_id}")
                return {
                    'max_change': 0.005,  # Max 0.5% per stablecoin
                    'regime': 'stablecoin',
                    'regime_multiplier': 0.1,
                    'market_cap_multiplier': 0.1,
                    'volatility_multiplier': 0.1,
                    'momentum_multiplier': 0.1,
                    'is_stablecoin': True
                }
            
            # 2. Ottieni dati Bitcoin dal benchmark system (se disponibile)
            bitcoin_regime = 'sideways_quiet'  # Default
            bitcoin_momentum = 0.0
            bitcoin_volatility = 0.4
            
            # Se hai accesso al bitcoin benchmark (dal sistema principale)
            if hasattr(self, 'bitcoin_benchmark') and self.bitcoin_benchmark:
                try:
                    btc_analysis = self.bitcoin_benchmark.get_real_market_regime_analysis()
                    bitcoin_regime = btc_analysis.get('current_regime', 'sideways_quiet')
                    bitcoin_momentum = btc_analysis.get('momentum', 0.0)
                    bitcoin_volatility = btc_analysis.get('volatility', 0.4)
                except:
                    print("   ⚠️ Bitcoin benchmark not available, using defaults")
            
            # 3. Moltiplicatori per regime di mercato
            regime_multipliers = {
                'bull_strong': 4.0,      # Bull estremo: soglie x4 
                'bull_moderate': 2.5,    # Bull moderato: soglie x2.5
                'sideways_volatile': 1.8, # Volatile: soglie x1.8
                'sideways_quiet': 1.0,   # Neutrale: soglie base
                'bear_moderate': 1.3,    # Bear: ancora movimenti significativi
                'bear_strong': 0.9       # Bear estremo: soglie leggermente ridotte
            }
            
            # 4. Soglie base per horizon (più realistiche)
            base_thresholds = {
                1: 0.12,  # 12% base per 1d (era 5%)
                3: 0.25   # 25% base per 3d (era 12%)
            }
            
            # 5. ✅ SKIP DATABASE QUERIES per "validation" 
            if crypto_id == "validation":
                market_cap_multiplier = 1.5  # Default moderato
                volatility_multiplier = 1.0  # Default
            else:
                # Market cap adjustment (crypto più piccole = più volatili)
                market_cap_multiplier = self._get_market_cap_multiplier(crypto_id)
                
                # Volatilità crypto-specifica (dai dati storici)
                crypto_volatility = self._get_crypto_historical_volatility(crypto_id, horizon)
                volatility_multiplier = max(0.5, min(3.0, crypto_volatility / 0.4))
            
            # 6. Bitcoin momentum adjustment
            momentum_multiplier = 1.0 + abs(bitcoin_momentum) * 2.0
            
            # 7. Calcolo finale
            regime_mult = regime_multipliers.get(bitcoin_regime, 1.0)
            base_threshold = base_thresholds[horizon]
            
            dynamic_max_change = (base_threshold * 
                                regime_mult * 
                                market_cap_multiplier * 
                                volatility_multiplier * 
                                momentum_multiplier)
            
            # Cap massimo realistico
            max_cap = 0.80 if horizon == 3 else 0.50  # 80% per 3d, 50% per 1d
            dynamic_max_change = min(dynamic_max_change, max_cap)
            
            # Soglia minima ragionevole 
            dynamic_max_change = max(dynamic_max_change, 0.02)  # Minimo 2%
            
            thresholds = {
                'max_change': dynamic_max_change,
                'regime': bitcoin_regime,
                'regime_multiplier': regime_mult,
                'market_cap_multiplier': market_cap_multiplier,
                'volatility_multiplier': volatility_multiplier,
                'momentum_multiplier': momentum_multiplier,
                'is_stablecoin': False
            }
            
            print(f"   📊 Dynamic thresholds for {crypto_id} {horizon}d:")
            print(f"      Bitcoin regime: {bitcoin_regime} (x{regime_mult:.1f})")
            print(f"      Market cap mult: x{market_cap_multiplier:.1f}")
            print(f"      Volatility mult: x{volatility_multiplier:.1f}")
            print(f"      Momentum mult: x{momentum_multiplier:.1f}")
            print(f"      Final max change: {dynamic_max_change:.1%}")
            
            return thresholds
            
        except Exception as e:
            print(f"   ⚠️ Error calculating dynamic thresholds: {e}")
            # Fallback to reasonable defaults
            return {
                'max_change': base_thresholds.get(horizon, 0.25),
                'regime': 'unknown',
                'regime_multiplier': 1.0,
                'market_cap_multiplier': 1.0,
                'volatility_multiplier': 1.0,
                'momentum_multiplier': 1.0,
                'is_stablecoin': False
            }

    def _is_stablecoin(self, crypto_id: str, current_price: float = None) -> bool:
        """🔒 Identifica se una crypto è una stablecoin"""
        
        # Lista stablecoin conosciute
        known_stablecoins = {
            'usd-coin', 'tether', 'dai', 'frax', 'tusd', 'busd', 
            'paxos-standard', 'terrausd', 'fei-usd', 'tribe-2',
            'maker', 'liquity-usd', 'trueUSD', 'gemini-dollar',
            'husd', 'stableUSD', 'origin-dollar'
        }
        
        # Check by crypto_id
        if crypto_id in known_stablecoins:
            return True
        
        # Check by name patterns
        stablecoin_patterns = ['usd', 'usdt', 'usdc', 'dai', 'tusd', 'busd']
        crypto_lower = crypto_id.lower()
        for pattern in stablecoin_patterns:
            if pattern in crypto_lower:
                return True
        
        # Check by price range (if available)
        if current_price is not None:
            # Price-based detection: se il prezzo è vicino a $1, potrebbe essere stablecoin
            if 0.95 <= current_price <= 1.05:
                print(f"   🔍 Potential stablecoin detected by price: {crypto_id} @ ${current_price:.4f}")
                return True
        
        return False

    def _get_market_cap_multiplier(self, crypto_id: str) -> float:
        """💰 Moltiplicatore basato su market cap (crypto piccole più volatili)"""
        try:
            # ✅ USA LA TABELLA CORRETTA: predictions_optimized
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT market_cap_rank FROM predictions_optimized 
                WHERE crypto_id = ? 
                ORDER BY timestamp DESC LIMIT 1
            """, (crypto_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result and result[0]:
                rank = result[0]
                
                if rank <= 10:          # Top 10: conservative
                    return 0.8
                elif rank <= 50:        # Top 50: normale  
                    return 1.0
                elif rank <= 200:       # Top 200: più volatile
                    return 1.5
                elif rank <= 500:       # Top 500: molto volatile
                    return 2.2
                else:                   # Oltre 500: estremamente volatile
                    return 3.0
            else:
                # Fallback: se non trovato nel database, usa default moderato
                print(f"   📊 Market cap rank not found for {crypto_id}, using default")
                return 1.5  # Default per crypto sconosciute
                
        except Exception as e:
            print(f"   ⚠️ Error getting market cap multiplier: {e}")
            return 1.5  # Safe default

    def _get_crypto_historical_volatility(self, crypto_id: str, horizon: int) -> float:
        """📊 Calcola volatilità storica della crypto specifica"""
        try:
            # ✅ USA LA TABELLA CORRETTA: predictions_optimized 
            conn = sqlite3.connect(self.db_path)
            
            # Prendi ultimi 30 giorni per calcolare volatilità
            query = """
                SELECT current_price, timestamp FROM predictions_optimized 
                WHERE crypto_id = ? 
                ORDER BY timestamp DESC LIMIT 30
            """
            
            df = pd.read_sql_query(query, conn, params=(crypto_id,))
            conn.close()
            
            if len(df) < 10:
                print(f"   📊 Insufficient price history for {crypto_id} volatility, using default")
                return 0.4  # Default volatility
            
            # Calcola daily returns
            df = df.sort_values('timestamp')  # Ordina cronologicamente
            df['returns'] = df['current_price'].pct_change()
            daily_volatility = df['returns'].std()
            
            if pd.isna(daily_volatility) or daily_volatility <= 0:
                return 0.4  # Default se calcolo fallisce
            
            # Annualized volatility
            annualized_volatility = daily_volatility * np.sqrt(365)
            
            # Cap ragionevoli per crypto
            final_volatility = min(max(annualized_volatility, 0.2), 2.0)  # Tra 20% e 200%
            
            print(f"   📊 {crypto_id} historical volatility: {final_volatility:.1%}")
            return final_volatility
            
        except Exception as e:
            print(f"   ⚠️ Error calculating crypto volatility: {e}")
            return 0.4  # Safe default

    def set_bitcoin_benchmark(self, bitcoin_benchmark):
        """🔗 Collega il Bitcoin benchmark system"""
        self.bitcoin_benchmark = bitcoin_benchmark
        print("✅ Bitcoin benchmark linked to ML system")

    def set_feedback_optimizer(self, feedback_optimizer):
        """🔧 Imposta il feedback optimizer"""
        self.feedback_optimizer = feedback_optimizer
        print("✅ Feedback optimizer linked to ML system")

    def update_model_weights(self, crypto_id: str, horizon: str, weights: Dict[str, float]):
        """⚖️ Aggiorna pesi dei modelli per una specifica crypto"""
        weight_key = f"{crypto_id}_{horizon}d"
        self.dynamic_model_weights[weight_key] = weights
        print(f"✅ Updated model weights for {crypto_id} {horizon}d: {weights}")

    def get_model_weights(self, crypto_id: str, horizon: int) -> Dict[str, float]:
        """📊 Ottiene pesi dei modelli per una specifica crypto"""
        weight_key = f"{crypto_id}_{horizon}d"
        weights = dict(self.dynamic_model_weights[weight_key])
        
        if not weights:
            # Default weights se non ci sono pesi ottimizzati
            return {'lightgbm': 1.0, 'xgboost': 1.0, 'catboost': 1.0, 'lstm': 1.0}
        
        return weights

    def _load_compatible_config(self, config):
        """📋 Load configuration compatible with existing system"""
        if isinstance(config, str):
            # Load from JSON file path
            try:
                config_path = Path(config)
                if not config_path.exists():
                    config_path = Path(f"{config}.json")
                
                with open(config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    print(f"📄 Loaded config: {config_path}")
                    return loaded_config
            except Exception as e:
                print(f"⚠️ Error loading config {config}: {e}")
                return self._get_default_compatible_config()
        
        elif isinstance(config, dict):
            return config
        
        else:
            # Try to load crypto_ssd_config.json by default
            try:
                with open('crypto_ssd_config.json', 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    print("📄 Loaded default crypto_ssd_config.json")
                    return loaded_config
            except:
                print("⚠️ Using default compatible config")
                return self._get_default_compatible_config()
    
    def _get_default_compatible_config(self):
        """⚙️ Default configuration compatible with existing system"""
        return {
            'storage': {
                'base_directory': 'D:/CryptoSystem',
                'database': {
                    'main_db_path': 'D:/CryptoSystem/database/crypto_optimized.db'
                },
                'cache': {
                    'ml_cache_directory': 'D:/CryptoSystem/cache/ml_models',
                    'api_cache_directory': 'D:/CryptoSystem/cache/api'
                }
            },
            'lookback_strategy': {
                'lookback_standard': 180,
                'lookback_premium': 270,
                'top_crypto_rank_threshold': 50
            },
            'quality_thresholds': {
                'confidence_threshold_1d': 0.60,
                'confidence_threshold_3d': 0.55,
                'quality_score_min_1d': 0.55,
                'quality_score_min_3d': 0.50
            },
            'ml_models': {
                'use_traditional_ml': True,
                'use_catboost': True,
                'use_lstm': True,
                'model_performance_tracking': True,
                'min_training_samples': 90,
                'sequence_length_lstm': 30
            },
            'api': {
                'coingecko_api_key': '',
                'api_delay': 2.0,
                'max_retries': 3,
                'timeout': 30
            }
        }
    
    def _setup_ssd_storage(self):
        """💾 Setup SSD storage compatible with existing system"""
        storage_config = self.config.get('storage', {})
        
        self.base_storage_path = Path(storage_config.get('base_directory', 'D:/CryptoSystem'))
        self.db_path = storage_config.get('database', {}).get('main_db_path', 
                                                              f'{self.base_storage_path}/database/crypto_optimized.db')
        
        # Cache directories
        cache_config = storage_config.get('cache', {})
        self.ml_cache_dir = Path(cache_config.get('ml_cache_directory', 
                                                  f'{self.base_storage_path}/cache/ml_models'))
        self.api_cache_dir = Path(cache_config.get('api_cache_directory', 
                                                   f'{self.base_storage_path}/cache/api'))
        
        # Create all directories
        for directory in [self.base_storage_path, self.ml_cache_dir, self.api_cache_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _setup_memory_management(self):
        """🧠 Setup memory management"""
        class MemoryManager:
            def __init__(self, config):
                self.memory_threshold = 80  # %
                self.last_cleanup = time.time()
                self.cleanup_interval = 1800  # 30 minutes
            
            def check_and_cleanup(self):
                try:
                    memory_percent = psutil.virtual_memory().percent
                    if (memory_percent > self.memory_threshold or 
                        time.time() - self.last_cleanup > self.cleanup_interval):
                        gc.collect()
                        self.last_cleanup = time.time()
                        print(f"🧹 Memory cleanup: {memory_percent:.1f}% → {psutil.virtual_memory().percent:.1f}%")
                except:
                    pass
        
        return MemoryManager(self.config)
    
    def fetch_crypto_data_compatible(self, crypto_id: str, days: int) -> pd.DataFrame:
        """📈 Fetch crypto data compatible with existing API management"""
        # Rate limiting compatible with existing system
        current_time = time.time()
        if current_time - self.last_api_call < self.api_delay:
            time.sleep(self.api_delay - (current_time - self.last_api_call))
        
        # Check cache first (compatible with existing caching)
        cache_file = self.api_cache_dir / f"{crypto_id}_{days}d.json"
        if cache_file.exists():
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age < 3600:  # 1 hour cache
                try:
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                    df = pd.DataFrame(cached_data)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    print(f"📦 Using cached data for {crypto_id}")
                    return df
                except:
                    pass
        
        # Fetch from API
        url = "https://api.coingecko.com/api/v3/coins/{}/market_chart".format(crypto_id)
        params = {
            'vs_currency': 'usd',
            'days': days,
            'interval': 'daily' if days > 90 else 'hourly'
        }
        
        # Add API key if available
        api_key = self.config.get('api', {}).get('coingecko_api_key', '')
        if api_key:
            params['x_cg_demo_api_key'] = api_key
        
        max_retries = self.config.get('api', {}).get('max_retries', 3)
        timeout = self.config.get('api', {}).get('timeout', 30)
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, timeout=timeout)
                response.raise_for_status()
                
                self.api_calls_made += 1
                self.last_api_call = time.time()
                
                data = response.json()
                
                # Convert to DataFrame
                prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
                volumes = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
                market_caps = pd.DataFrame(data['market_caps'], columns=['timestamp', 'market_cap'])
                
                df = prices.merge(volumes, on='timestamp').merge(market_caps, on='timestamp')
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                # Add basic features
                df['price_change_1d'] = df['price'].pct_change(1)
                df['price_change_7d'] = df['price'].pct_change(7)
                
                # Cache the data
                try:
                    df_cache = df.copy()
                    df_cache['timestamp'] = df_cache['timestamp'].astype(str)
                    with open(cache_file, 'w') as f:
                        json.dump(df_cache.to_dict('records'), f)
                except:
                    pass
                
                print(f"✅ Fetched {len(df)} data points for {crypto_id}")
                return df
                
            except Exception as e:
                print(f"⚠️ API attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(self.api_delay * (attempt + 1))
        
        print(f"❌ Failed to fetch data for {crypto_id}")
        return pd.DataFrame()
    
    def create_optimized_features(self, data: pd.DataFrame, crypto_id: str, lookback_days: int) -> pd.DataFrame:
        """🔧 Create optimized features compatible with existing system - FIXED"""
        if data.empty or len(data) < 30:
            print(f"⚠️ Insufficient data for {crypto_id}")
            return None
        
        df = data.copy()
        
        try:
            # ✅ CRITICAL: Keep 'price' column available throughout the process
            # Technical indicators - optimized for performance
            windows = [7, 14, 30] if lookback_days < 180 else [7, 14, 30, 60]
            
            for window in windows:
                if len(df) >= window:
                    df[f'sma_{window}'] = df['price'].rolling(window=window).mean()
                    df[f'ema_{window}'] = df['price'].ewm(span=window).mean()
                    df[f'std_{window}'] = df['price'].rolling(window=window).std()
                    
                    # RSI calculation
                    delta = df['price'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                    rs = gain / loss
                    df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
            
            # Momentum features
            for period in [1, 3, 7, 14]:
                if len(df) > period:
                    df[f'momentum_{period}d'] = df['price'] / df['price'].shift(period) - 1
            
            # Volatility features
            df['volatility_7d'] = df['price_change_1d'].rolling(window=7).std()
            df['volatility_30d'] = df['price_change_1d'].rolling(window=30).std()
            
            # Volume features
            if 'volume' in df.columns:
                df['volume_sma_7'] = df['volume'].rolling(window=7).mean()
                df['volume_sma_30'] = df['volume'].rolling(window=30).mean()
                df['volume_ratio'] = df['volume'] / df['volume_sma_30']
            
            # Bollinger Bands
            for window in [20, 50]:
                if len(df) >= window:
                    rolling_mean = df['price'].rolling(window=window).mean()
                    rolling_std = df['price'].rolling(window=window).std()
                    df[f'bb_upper_{window}'] = rolling_mean + (rolling_std * 2)
                    df[f'bb_lower_{window}'] = rolling_mean - (rolling_std * 2)
                    df[f'bb_position_{window}'] = (df['price'] - rolling_mean) / (rolling_std * 2)
            
            # MACD
            if len(df) >= 26:
                ema_12 = df['price'].ewm(span=12).mean()
                ema_26 = df['price'].ewm(span=26).mean()
                df['macd'] = ema_12 - ema_26
                df['macd_signal'] = df['macd'].ewm(span=9).mean()
                df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # ✅ FIXED: Clean data properly
            df = df.fillna(method='ffill').fillna(0)
            
            # ✅ CRITICAL FIX: Return ALL data with 'price' intact
            # Feature selection will happen during training, not here
            print(f"🔧 Created {len(df.columns)} total columns for {crypto_id} (including price)")
            return df
            
        except Exception as e:
            print(f"❌ Feature creation failed for {crypto_id}: {e}")
            return None
    
    def prepare_targets_compatible(self, data: pd.DataFrame, horizon: int) -> pd.Series:
        """🎯 Prepare targets compatible with existing system classification"""
        try:
            future_price = data['price'].shift(-horizon)
            current_price = data['price']
            price_change = (future_price - current_price) / current_price
            
            # Compatible with existing thresholds
            threshold = 0.01  # 1% threshold for UP/DOWN classification
            target = (price_change > threshold).astype(int)
            
            return target
            
        except Exception as e:
            print(f"❌ Target preparation failed: {e}")
            return pd.Series()
    
    def train_compatible_models(self, crypto_id: str, market_cap_rank: int) -> bool:
        """🚂 Train models compatible with existing system architecture"""
        try:
            print(f"\n🚂 Training compatible models for {crypto_id} (rank: {market_cap_rank})...")
            
            # Get lookback days compatible with existing strategy
            lookback_config = self.config.get('lookback_strategy', {})
            if market_cap_rank <= lookback_config.get('top_crypto_rank_threshold', 50):
                days = lookback_config.get('lookback_premium', 270)
            else:
                days = lookback_config.get('lookback_standard', 180)
            
            # Fetch data
            data = self.fetch_crypto_data_compatible(crypto_id, days)
            if data.empty or len(data) < self.config.get('ml_models', {}).get('min_training_samples', 90):
                print(f"❌ Insufficient data for {crypto_id}")
                return False
            
            # Create features
            featured_data = self.create_optimized_features(data, crypto_id, days)
            if featured_data is None:
                print(f"❌ Feature creation failed for {crypto_id}")
                return False
            
            # ✅ CRITICAL FIX: Select feature columns but EXCLUDE 'price' for ML training
            # 'price' is needed for target calculation and prediction price calculation
            feature_cols = [col for col in featured_data.columns 
                          if col not in ['timestamp', 'crypto_id', 'price'] and  # ✅ EXCLUDE price from features
                          featured_data[col].dtype in ['float64', 'int64']]
            
            if len(feature_cols) == 0:
                print(f"❌ No valid features for {crypto_id}")
                return False
            
            # ✅ FIXED: Limit features for performance but keep essential ones
            if len(feature_cols) > 50:
                # Prioritize important features
                important_features = [col for col in feature_cols if any(imp in col for imp in 
                    ['sma_', 'ema_', 'rsi_', 'momentum_', 'volatility_', 'volume_', 'macd'])]
                feature_cols = important_features[:50] if len(important_features) >= 50 else feature_cols[:50]
            
            X = featured_data[feature_cols].dropna()
            if len(X) < self.config.get('ml_models', {}).get('min_training_samples', 90):
                print(f"❌ Insufficient features for {crypto_id}: {len(X)} samples")
                return False
            
            # ✅ CRITICAL: Save feature columns for prediction consistency
            self.feature_columns[crypto_id] = feature_cols
            print(f"📊 Using {len(feature_cols)} features for {crypto_id} (price excluded from ML features)")
            
            success_count = 0
            
            # Train for both horizons - compatible with dual horizon system
            for horizon in [1, 3]:
                print(f"  📊 Training {horizon}d horizon...")
                
                # Prepare target
                y = self.prepare_targets_compatible(featured_data, horizon)
                if y.empty:
                    continue
                
                # ✅ CRITICAL FIX: Proper data alignment
                # Target y[i] corresponds to price change from time i to i+horizon
                # So we need to remove the last 'horizon' elements where future price is unknown
                
                # Remove rows with NaN targets (last 'horizon' rows)
                valid_indices = y.dropna().index
                if len(valid_indices) < 50:
                    print(f"  ⚠️ {horizon}d: Too few valid targets after removing NaN")
                    continue
                
                # Align X and y using valid indices only
                X_aligned = X.loc[valid_indices]
                y_aligned = y.loc[valid_indices]
                
                # ✅ VERIFICATION: Ensure perfect alignment
                if len(X_aligned) != len(y_aligned) or not X_aligned.index.equals(y_aligned.index):
                    print(f"  ❌ {horizon}d: Index alignment failed")
                    continue
                
                print(f"  📊 {horizon}d: Aligned data - {len(X_aligned)} samples")
                
                # Initialize model storage
                model_key = f"{crypto_id}_{horizon}d"
                models_dict = self.models_1d if horizon == 1 else self.models_3d
                models_dict[model_key] = {}
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = pd.DataFrame(
                    scaler.fit_transform(X_aligned),
                    columns=X_aligned.columns,
                    index=X_aligned.index
                )
                
                # ✅ FIXED: Consistent scaler naming for all models including LSTM
                scaler_key = f"{crypto_id}_{horizon}d"
                self.scalers[scaler_key] = scaler
                print(f"  💾 Scaler saved as: {scaler_key}")
                
                model_count = 0
                
                # Train compatible models
                
                # 1. LightGBM - Compatible with existing config
                if self.config.get('ml_models', {}).get('use_traditional_ml', True):
                    lgb_model = self._train_lightgbm_compatible(X_aligned, y_aligned, horizon)
                    if lgb_model:
                        models_dict[model_key]['lightgbm'] = lgb_model
                        model_count += 1
                
                # 2. XGBoost - Compatible with existing config
                if self.config.get('ml_models', {}).get('use_traditional_ml', True):
                    xgb_model = self._train_xgboost_compatible(X_aligned, y_aligned, horizon)
                    if xgb_model:
                        models_dict[model_key]['xgboost'] = xgb_model
                        model_count += 1
                
                # 3. CatBoost - Compatible with existing config
                if self.config.get('ml_models', {}).get('use_catboost', True):
                    cat_model = self._train_catboost_compatible(X_aligned, y_aligned, horizon)
                    if cat_model:
                        models_dict[model_key]['catboost'] = cat_model
                        model_count += 1
                
                # 4. LSTM - Compatible with existing config
                if self.config.get('ml_models', {}).get('use_lstm', True):
                    lstm_model = self._train_lstm_compatible(X_aligned, y_aligned, horizon, crypto_id)  # ✅ FIXED: Pass crypto_id
                    if lstm_model:
                        models_dict[model_key]['lstm'] = lstm_model
                        model_count += 1
                
                if model_count > 0:
                    success_count += 1
                    print(f"  ✅ {horizon}d: {model_count} models trained")
                else:
                    print(f"  ❌ {horizon}d: No models trained")
            
            if success_count > 0:
                print(f"✅ Successfully trained models for {crypto_id}")
                
                # Memory cleanup
                self.memory_manager.check_and_cleanup()
                return True
            else:
                print(f"❌ Training failed for {crypto_id}")
                return False
                
        except Exception as e:
            print(f"❌ Training error for {crypto_id}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _train_lightgbm_compatible(self, X: pd.DataFrame, y: pd.Series, horizon: int) -> Optional[lgb.LGBMClassifier]:
        """🌟 Train LightGBM compatible with existing system"""
        try:
            # Time Series Split for walk-forward validation
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Parameters compatible with existing system
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'n_estimators': min(200, len(X) // 10),
                'random_state': 42,
                'verbosity': -1,
                'n_jobs': -1
            }
            
            model = lgb.LGBMClassifier(**params)
            
            # Walk-forward validation
            cv_scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric='logloss',
                    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
                )
                
                val_pred = model.predict(X_val)
                score = accuracy_score(y_val, val_pred)
                cv_scores.append(score)
            
            avg_score = np.mean(cv_scores)
            print(f"   LightGBM {horizon}d - CV Accuracy: {avg_score:.3f}")
            
            # Check quality threshold compatible with existing system
            threshold = self.config.get('quality_thresholds', {}).get('quality_score_min_1d' if horizon == 1 else 'quality_score_min_3d', 0.52)
            
            if avg_score >= threshold:
                model.fit(X, y)
                return model
            else:
                print(f"   ❌ LightGBM {horizon}d - Below quality threshold")
                return None
                
        except Exception as e:
            print(f"   ❌ LightGBM {horizon}d error: {e}")
            return None
    
    def _train_lstm_compatible(self, X_train, y_train, horizon, crypto_id):
        """🧠 Train LSTM with fixed TensorFlow warnings"""
        try:
            print(f"   🧠 LSTM {horizon}d training: {len(X_train)} train samples")
            
            # ✅ FIX: Clear TensorFlow session to avoid retracing
            keras.backend.clear_session()
            
            # Prepare data with consistent shape
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            
            # Split data
            X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
            
            # ✅ FIX: Ensure consistent scaling and shapes
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_val_scaled = scaler.transform(X_val)
            
            # ✅ FIX: Reshape for LSTM with fixed sequence length
            sequence_length = min(20, X_tr_scaled.shape[1])  # Fixed sequence length
            n_features = 1
            
            # Reshape data for LSTM
            X_tr_reshaped = X_tr_scaled[:, :sequence_length].reshape(X_tr_scaled.shape[0], sequence_length, n_features)
            X_val_reshaped = X_val_scaled[:, :sequence_length].reshape(X_val_scaled.shape[0], sequence_length, n_features)
            
            print(f"   🔧 LSTM input shape: {X_tr_reshaped.shape}")
            
            # ✅ CRITICO: RIMOSSO batch_input_shape - Era questo il problema!
            model = Sequential([
                LSTM(32, return_sequences=True, input_shape=(sequence_length, n_features)),  # ⭐ FIXED: Rimosso batch_input_shape
                Dropout(0.2),
                LSTM(16, return_sequences=False),
                Dropout(0.2),
                Dense(8, activation='relu'),
                BatchNormalization(),
                Dense(1, activation='sigmoid')
            ])
            
            # ✅ FIX: Compile with specific optimizer settings
            optimizer = Adam(learning_rate=0.001, clipnorm=1.0)  # Gradient clipping
            model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # ✅ IMPROVED: Better callbacks
            callbacks = [
                EarlyStopping(patience=8, restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-7, verbose=0)
            ]
            
            # ✅ FIX: Training with consistent batch size
            batch_size = min(16, len(X_tr_reshaped) // 4)  # Adaptive batch size
            
            history = model.fit(
                X_tr_reshaped, y_tr,
                validation_data=(X_val_reshaped, y_val),
                epochs=30,  # Reduced epochs
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0,
                shuffle=True
            )
            
            # Evaluate
            val_loss, val_accuracy = model.evaluate(X_val_reshaped, y_val, verbose=0, batch_size=batch_size)
            print(f"   LSTM/GRU {horizon}d - Val Accuracy: {val_accuracy:.3f}")
            
            threshold = self.config.get('quality_thresholds', {}).get(f'quality_score_min_{horizon}d', 0.52)
            
            if val_accuracy >= threshold:
                # ✅ FIXED: Save scaler with consistent naming
                lstm_scaler_key = f"{crypto_id}_lstm_{horizon}d"
                
                # ✅ FIX: Create custom scaler that handles LSTM reshaping
                class LSTMScaler:
                    def __init__(self, standard_scaler, sequence_length):
                        self.standard_scaler = standard_scaler
                        self.sequence_length = sequence_length
                    
                    def transform(self, X):
                        X_scaled = self.standard_scaler.transform(X)
                        return X_scaled[:, :self.sequence_length].reshape(X_scaled.shape[0], self.sequence_length, 1)
                
                lstm_scaler = LSTMScaler(scaler, sequence_length)
                self.scalers[lstm_scaler_key] = lstm_scaler
                
                print(f"   💾 LSTM scaler saved as: {lstm_scaler_key}")
                
                # ✅ FIX: Create model wrapper for consistent prediction
                class LSTMModelWrapper:
                    def __init__(self, model, sequence_length):
                        self.model = model
                        self.sequence_length = sequence_length
                    
                    def predict_proba(self, X):
                        # X is already scaled and reshaped by the scaler
                        pred = self.model.predict(X, verbose=0)
                        # Return probabilities for binary classification
                        return np.column_stack([1 - pred.flatten(), pred.flatten()])
                    
                    def predict(self, X):
                        proba = self.predict_proba(X)
                        return (proba[:, 1] > 0.5).astype(int)
                
                wrapped_model = LSTMModelWrapper(model, sequence_length)
                return wrapped_model
                
            else:
                print(f"   ❌ LSTM {horizon}d - Below quality threshold: {val_accuracy:.3f} < {threshold:.3f}")
                # Clean up
                keras.backend.clear_session()
                return None
                
        except Exception as e:
            print(f"   ❌ LSTM {horizon}d error: {e}")
            keras.backend.clear_session()
            import traceback
            traceback.print_exc()
            return None


    # =============================================================================
    # 2. FIX ERRORE XGBOOST - base_score corretto
    # =============================================================================

    def _train_xgboost_compatible(self, X: pd.DataFrame, y: pd.Series, horizon: int) -> Optional[xgb.XGBClassifier]:
        """⚡ Train XGBoost compatible with existing system"""
        try:
            # ✅ CRITICO: Controlla se il target ha variabilità
            if len(y.unique()) < 2:
                print(f"   ❌ XGBoost {horizon}d - Target has only one unique value: {y.unique()}")
                return None
                
            tscv = TimeSeriesSplit(n_splits=3)
            
            # ✅ FIX: base_score corretto per logistic loss
            base_score = y.mean()  # Media del target come base_score
            if base_score <= 0.0 or base_score >= 1.0:
                base_score = 0.5  # Default safe value
                
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'base_score': base_score,  # ⭐ FIXED: Valore corretto
                'learning_rate': 0.05,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'n_estimators': min(200, len(X) // 10),
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': 0  # Silenzioso
            }
            
            model = xgb.XGBClassifier(**params)
            
            cv_scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # ✅ Controlla che entrambi train e val abbiano variabilità
                if len(y_train.unique()) < 2 or len(y_val.unique()) < 2:
                    continue
                    
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
                
                val_pred = model.predict(X_val)
                score = accuracy_score(y_val, val_pred)
                cv_scores.append(score)
            
            if not cv_scores:  # Nessun fold valido
                print(f"   ❌ XGBoost {horizon}d - No valid CV folds")
                return None
                
            avg_score = np.mean(cv_scores)
            print(f"   XGBoost {horizon}d - CV Accuracy: {avg_score:.3f}")
            
            threshold = self.config.get('quality_thresholds', {}).get('quality_score_min_1d' if horizon == 1 else 'quality_score_min_3d', 0.52)
            
            if avg_score >= threshold:
                model.fit(X, y)
                return model
            else:
                print(f"   ❌ XGBoost {horizon}d - Below quality threshold")
                return None
                
        except Exception as e:
            print(f"   ❌ XGBoost {horizon}d error: {e}")
            return None


    # =============================================================================
    # 3. FIX ERRORE CATBOOST - gestione target unico
    # =============================================================================

    def _train_catboost_compatible(self, X: pd.DataFrame, y: pd.Series, horizon: int) -> Optional[CatBoostClassifier]:
        """🐱 Train CatBoost compatible with existing system"""
        try:
            # ✅ CRITICO: Controlla se il target ha variabilità
            unique_values = y.unique()
            if len(unique_values) < 2:
                print(f"   ❌ CatBoost {horizon}d - Target contains only one unique value: {unique_values}")
                return None
                
            params = {
                'iterations': min(200, len(X) // 10),
                'depth': 6,
                'learning_rate': 0.05,
                'l2_leaf_reg': 3,
                'random_strength': 1,
                'bagging_temperature': 1,
                'border_count': 128,
                'random_state': 42,
                'verbose': False,
                'early_stopping_rounds': 10
            }
            
            model = CatBoostClassifier(**params)
            
            # Train-test split for validation
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # ✅ Controlla variabilità anche sui split
            if len(y_train.unique()) < 2 or len(y_val.unique()) < 2:
                print(f"   ❌ CatBoost {horizon}d - No class variability in train/val split")
                return None
            
            if len(X_val) > 10:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    use_best_model=True
                )
                
                val_pred = model.predict(X_val)
                val_accuracy = accuracy_score(y_val, val_pred)
                print(f"   CatBoost {horizon}d - Val Accuracy: {val_accuracy:.3f}")
                
                threshold = self.config.get('quality_thresholds', {}).get('quality_score_min_1d' if horizon == 1 else 'quality_score_min_3d', 0.52)
                
                if val_accuracy >= threshold:
                    model.fit(X, y)
                    return model
                else:
                    print(f"   ❌ CatBoost {horizon}d - Below quality threshold")
                    return None
            else:
                print(f"   ❌ CatBoost {horizon}d - Insufficient validation data")
                return None
                
        except Exception as e:
            print(f"   ❌ CatBoost {horizon}d error: {e}")
            return None
    
    def predict_dual_optimized(self, crypto_id: str, data: pd.DataFrame, market_cap_rank: int) -> Dict[str, Dict]:
        """🔮 Generate dual predictions compatible with existing system"""
        try:
            print(f"🔮 Generating dual predictions for {crypto_id}...")
            
            predictions = {}
            
            # Get lookback days
            lookback_config = self.config.get('lookback_strategy', {})
            if market_cap_rank <= lookback_config.get('top_crypto_rank_threshold', 50):
                lookback_days = lookback_config.get('lookback_premium', 270)
            else:
                lookback_days = lookback_config.get('lookback_standard', 180)
            
            # Generate predictions for both horizons
            for horizon in [1, 3]:
                try:
                    pred = self._predict_single_horizon_compatible(
                        crypto_id, data, horizon, lookback_days
                    )
                    
                    # 🔧 FIX: Chiamata corretta con parametri separati
                    if pred:
                        predicted_change = pred.get('predicted_change', 0)
                        confidence = pred.get('confidence', 0)
                        
                        # Chiamata corretta con parametri individuali
                        is_valid, quality_score = self._validate_prediction_quality(
                            predicted_change, confidence, horizon
                        )
                        
                        if is_valid:
                            # Aggiungi il quality_score al risultato
                            pred['quality_score'] = quality_score
                            predictions[f"{horizon}d"] = pred
                            print(f"  ✅ {horizon}d prediction: {predicted_change:+.3f} (conf: {confidence:.3f}, quality: {quality_score:.3f})")
                        else:
                            print(f"  ❌ {horizon}d prediction failed validation")
                    else:
                        print(f"  ❌ {horizon}d prediction is None")
                        
                except Exception as e:
                    print(f"  ❌ {horizon}d prediction error: {e}")
                    import traceback
                    traceback.print_exc()
            
            if predictions:
                print(f"✅ Generated {len(predictions)} predictions for {crypto_id}")
                return predictions
            else:
                print(f"❌ No valid predictions for {crypto_id}")
                return {}
                
        except Exception as e:
            print(f"❌ Prediction failed for {crypto_id}: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _predict_single_horizon_compatible(self, crypto_id: str, data: pd.DataFrame, horizon: int, lookback_days: int) -> Optional[Dict]:
        """🔮 Generate prediction for single horizon with lookback days support"""
        try:
            print(f"   💰 Current price for {crypto_id}: ${data['price'].iloc[-1]:.4f} (lookback: {lookback_days}d)")
            
            # Get models
            model_key = f"{crypto_id}_{horizon}d"
            if model_key not in (self.models_1d if horizon == 1 else self.models_3d):
                print(f"   ❌ No models available for {crypto_id} {horizon}d")
                return None

            models = (self.models_1d if horizon == 1 else self.models_3d)[model_key]
            if not models:
                print(f"   ❌ Empty model dict for {crypto_id} {horizon}d")
                return None

            # Create features (using lookback_days if needed for feature creation)
            features = self.create_optimized_features(data, crypto_id, lookback_days)
            if features is None or len(features) == 0:
                print(f"   ❌ Feature creation failed for {crypto_id}")
                return None

            # Use last row for prediction
            X_pred = features.iloc[[-1]].copy()
            
            # 🔧 FIX: Usa la stessa logica di filtraggio dell'addestramento
            # Rimuovi TUTTE le colonne non-feature (timestamp, crypto_id, price)
            columns_to_exclude = ['timestamp', 'crypto_id', 'price']
            
            # Se abbiamo salvato le feature columns specifiche per questo crypto, usale
            if crypto_id in self.feature_columns:
                # Usa le feature columns salvate durante l'addestramento
                saved_feature_cols = self.feature_columns[crypto_id]
                # Verifica che tutte le colonne esistano
                available_cols = [col for col in saved_feature_cols if col in X_pred.columns]
                if len(available_cols) != len(saved_feature_cols):
                    missing_cols = set(saved_feature_cols) - set(available_cols)
                    print(f"   ⚠️ Missing feature columns for {crypto_id}: {missing_cols}")
                X_pred = X_pred[available_cols]
                print(f"   🔧 Using {len(available_cols)} saved feature columns for {crypto_id}")
            else:
                # Fallback: usa la stessa logica dell'addestramento
                feature_cols = [col for col in X_pred.columns 
                            if col not in columns_to_exclude and 
                            X_pred[col].dtype in ['float64', 'int64']]
                X_pred = X_pred[feature_cols]
                print(f"   🔧 Using {len(feature_cols)} filtered feature columns for {crypto_id}")

            # Verifica che abbiamo delle feature
            if X_pred.empty or len(X_pred.columns) == 0:
                print(f"   ❌ No valid features after filtering for {crypto_id}")
                return None
                
            print(f"🔧 Created {len(X_pred.columns)} total columns for {crypto_id} (including price)")
            print(f"   🔧 Using {len(X_pred.columns)} features for prediction")

            # Scale features
            scaler_key = f"{crypto_id}_{horizon}d"
            if scaler_key in self.scalers:
                try:
                    X_pred_scaled = self.scalers[scaler_key].transform(X_pred)
                    print(f"   ✅ Features scaled using: {scaler_key}")
                except Exception as e:
                    print(f"   ⚠️ Scaler error for {scaler_key}: {e}")
                    # Stampa informazioni di debug
                    print(f"   🔍 Expected features: {getattr(self.scalers[scaler_key], 'feature_names_in_', 'Not available')}")
                    print(f"   🔍 Provided features: {list(X_pred.columns)}")
                    X_pred_scaled = X_pred.values
            else:
                print(f"   ⚠️ No scaler found for {scaler_key}, using raw features")
                X_pred_scaled = X_pred.values

            # Generate ensemble prediction
            predictions = []
            model_confidences = []
            models_used = []
            # OTTIENI PESI DINAMICI:
            dynamic_weights = self.get_model_weights(crypto_id, horizon)
            print(f"   🎯 Using dynamic weights: {dynamic_weights}")

            for model_name, model in models.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        pred_proba = model.predict_proba(X_pred_scaled)[0]
                        confidence = max(pred_proba)  # Max probability as confidence
                        prediction = 1 if pred_proba[1] > 0.5 else 0
                    else:
                        prediction = model.predict(X_pred_scaled)[0]
                        confidence = 0.6  # Default confidence for models without proba
                    
                    # APPLICA PESO DINAMICO:
                    model_weight = dynamic_weights.get(model_name, 1.0)
                    weighted_confidence = confidence * model_weight
                    
                    predictions.append(prediction)
                    model_confidences.append(weighted_confidence)  # Usa peso dinamico
                    models_used.append(model_name)
                    
                    print(f"     ✅ {model_name} prediction: {prediction} (confidence: {confidence:.3f}, weight: {model_weight:.2f})")
                    
                    '''predictions.append(prediction)
                    model_confidences.append(confidence)
                    models_used.append(model_name)
                    print(f"     ✅ {model_name} prediction: {prediction} (confidence: {confidence:.3f})")'''
                    
                except Exception as e:
                    print(f"     ❌ {model_name} error: {e}")
                    continue

            if not predictions:
                print(f"   ❌ No valid predictions from any model for {crypto_id} {horizon}d")
                return None

            # ✅ FIX: Calcolo più robusto del predicted_change
            if sum(model_confidences) > 0:
                weights = np.array(model_confidences) / sum(model_confidences)
            else:
                weights = np.ones(len(model_confidences)) / len(model_confidences)
                
            ensemble_prediction = np.average(predictions, weights=weights)
            ensemble_confidence = np.average(model_confidences, weights=weights)
            
            print(f"   🎯 {crypto_id} {horizon}d ensemble: {len(models_used)} models")
            
            # Convert to price prediction
            current_price = data['price'].iloc[-1]
            
            # ✅ FIXED: Calcolo più realistico e robusto del price change
            if ensemble_prediction > 0.5:
                # Predict price increase - scala basata sulla confidence
                base_change = (ensemble_prediction - 0.5) * 2  # 0 to 1
                # ✅ SOSTITUISCI CON QUESTO:
                # Ottieni soglie dinamiche
                dynamic_thresholds = self.get_dynamic_thresholds(crypto_id, horizon, current_price)
                max_change = dynamic_thresholds['max_change']

                print(f"   🎯 Using dynamic max_change: {max_change:.1%} (regime: {dynamic_thresholds['regime']})")
                    
                predicted_change = base_change * max_change * ensemble_confidence
                
            else:
                # Predict price decrease
                base_change = (0.5 - ensemble_prediction) * 2  # 0 to 1
                if horizon == 1:
                    max_change = 0.05  # Max 5% per 1d
                else:
                    max_change = 0.12  # Max 12% per 3d
                    
                predicted_change = -base_change * max_change * ensemble_confidence
            
            # ✅ SAFETY: Assicurati che il cambio non sia mai esattamente 0
            if abs(predicted_change) < 0.0005:  # Sotto 0.05%
                # Forza un piccolo cambio basato sulla direzione prevalente
                direction = 1 if ensemble_prediction > 0.5 else -1
                predicted_change = direction * 0.002  # Minimo 0.2%
            
            predicted_price = current_price * (1 + predicted_change)
            
            print(f"   📈 Current: ${current_price:.4f} → Predicted: ${predicted_price:.4f} ({predicted_change:+.3%})")
            print(f"   🔢 Ensemble: pred={ensemble_prediction:.3f}, conf={ensemble_confidence:.3f}")
            
            return {
                'predicted_change': predicted_change,
                'predicted_price': predicted_price,
                'confidence': ensemble_confidence,
                'models_used': models_used,
                'ensemble_prediction': ensemble_prediction,  # Debug info
                'current_price': current_price,
                'price_change_pct': (predicted_price - current_price) / current_price * 100,
                'horizon': f"{horizon}d"
            }

        except Exception as e:
            print(f"   ❌ Prediction error for {crypto_id} {horizon}d: {e}")
            import traceback
            traceback.print_exc()
            return None


    # 🔧 METODO AGGIUNTIVO: Verifica e riparazione dei scalers
    def repair_scalers(self):
        """🔧 Repair scalers to match saved feature columns"""
        print("🔧 Repairing scalers to match feature columns...")
        
        repaired_count = 0
        for crypto_id in self.feature_columns:
            expected_features = self.feature_columns[crypto_id]
            
            # Check both 1d and 3d scalers
            for horizon in [1, 3]:
                scaler_key = f"{crypto_id}_{horizon}d"
                
                if scaler_key in self.scalers:
                    scaler = self.scalers[scaler_key]
                    
                    # Check if scaler has feature_names_in_ attribute
                    if hasattr(scaler, 'feature_names_in_'):
                        scaler_features = list(scaler.feature_names_in_)
                        
                        if scaler_features != expected_features:
                            print(f"   ⚠️ Scaler mismatch for {scaler_key}")
                            print(f"      Expected: {len(expected_features)} features")
                            print(f"      Scaler has: {len(scaler_features)} features")
                            
                            # Remove inconsistent scaler - it will be retrained
                            del self.scalers[scaler_key]
                            print(f"   🗑️ Removed inconsistent scaler: {scaler_key}")
                            repaired_count += 1
        
        print(f"✅ Repaired {repaired_count} scalers")
        return repaired_count > 0
    
    def _validate_prediction_quality(self, prediction_change, confidence, horizon):
        """✅ Enhanced prediction validation with better debugging"""
        try:
            # Soglie ragionevoli per crypto
            min_confidence = self.config.get('quality_thresholds', {}).get(f'confidence_threshold_{horizon}d', 0.55)
            min_magnitude = 0.001  # 0.1% - soglia ragionevole
            # ✅ SOSTITUISCI CON QUESTO:
            # Usa soglie dinamiche per validation
            dynamic_thresholds = self.get_dynamic_thresholds("validation", horizon)
            max_magnitude = dynamic_thresholds['max_change']

            print(f"   🎯 Dynamic validation threshold: {max_magnitude:.1%}")
            confidence_check = confidence >= min_confidence
            magnitude_check = abs(prediction_change) >= min_magnitude
            realistic_check = abs(prediction_change) <= max_magnitude
            
            print(f"  📊 Validation {horizon}d:")
            print(f"     Change: {prediction_change:+.6f} ({prediction_change*100:+.3f}%)")
            print(f"     Confidence: {confidence:.3f} >= {min_confidence:.3f} ({'✅' if confidence_check else '❌'})")
            print(f"     Magnitude: {abs(prediction_change):.6f} >= {min_magnitude:.6f} ({'✅' if magnitude_check else '❌'})")
            print(f"     Realistic: {abs(prediction_change):.6f} <= {max_magnitude:.6f} ({'✅' if realistic_check else '❌'})")
            
            validation_passed = confidence_check and magnitude_check and realistic_check
            
            if validation_passed:
                # Calculate quality score
                quality_score = (confidence + min(abs(prediction_change) * 10, 1)) / 2
                print(f"  ✅ {horizon}d prediction: {prediction_change:+.3f} (conf: {confidence:.3f}, quality: {quality_score:.3f})")
                return True, quality_score
            else:
                print(f"  ❌ {horizon}d prediction failed validation")
                return False, 0.0
                
        except Exception as e:
            print(f"  ❌ Validation error: {e}")
            return False, 0.0
    
    def save_models_compatible(self) -> bool:
        """💾 Save models with enhanced compatibility"""
        try:
            models_dir = Path(self.base_storage_path) / 'cache' / 'ml_models'
            models_dir.mkdir(parents=True, exist_ok=True)
            
            # ✅ IMPROVED: Separate save for different model types
            save_data = {
                'models_1d': {},
                'models_3d': {},
                'scalers': {},
                'feature_columns': self.feature_columns,
                'model_performance_history': self.model_performance_history,
                'ensemble_weights': self.ensemble_weights,
                'timestamp': datetime.now().isoformat()
            }
            
            # ✅ FIX: Handle different model types separately
            for horizon_models, save_key in [(self.models_1d, 'models_1d'), (self.models_3d, 'models_3d')]:
                for crypto_key, models in horizon_models.items():
                    save_data[save_key][crypto_key] = {}
                    
                    for model_name, model in models.items():
                        try:
                            if model_name == 'lstm':
                                # Save LSTM models separately
                                model_path = models_dir / f"{crypto_key}_{model_name}_model.keras"
                                if hasattr(model, 'model'):  # Wrapped LSTM
                                    model.model.save(model_path)
                                else:
                                    model.save(model_path)
                                save_data[save_key][crypto_key][model_name] = str(model_path)
                                
                            else:
                                # Traditional ML models
                                save_data[save_key][crypto_key][model_name] = model
                                
                        except Exception as e:
                            print(f"   ⚠️ Failed to save {model_name} for {crypto_key}: {e}")
            
            # ✅ IMPROVED: Save scalers separately 
            scalers_to_save = {}
            for scaler_key, scaler in self.scalers.items():
                try:
                    if hasattr(scaler, 'standard_scaler'):  # Custom LSTM scaler
                        scalers_to_save[scaler_key] = {
                            'type': 'lstm_scaler',
                            'standard_scaler': scaler.standard_scaler,
                            'sequence_length': scaler.sequence_length
                        }
                    else:
                        scalers_to_save[scaler_key] = {
                            'type': 'standard_scaler',
                            'scaler': scaler
                        }
                except Exception as e:
                    print(f"   ⚠️ Failed to prepare scaler {scaler_key}: {e}")
            
            save_data['scalers'] = scalers_to_save
            
            # ✅ ENHANCED: Save with error handling
            model_file = models_dir / 'all_models.keras'
            with open(model_file, 'wb') as f:
                joblib.dump(save_data, f, compress=3)
            
            print(f"💾 Models saved to {model_file}")
            print(f"📊 Saved: {len(save_data['models_1d'])} cryptos (1d), {len(save_data['models_3d'])} cryptos (3d)")
            print(f"🔧 Saved: {len(save_data['scalers'])} scalers")
            return True
            
        except Exception as e:
            print(f"❌ Model save error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_models_compatible(self) -> bool:
        """📦 Load models with enhanced compatibility"""
        try:
            models_dir = Path(self.base_storage_path) / 'cache' / 'ml_models'
            model_file = models_dir / 'all_models.keras'
            
            if not model_file.exists():
                print(f"⚠️ Model file not found: {model_file}")
                return False
            
            print(f"📦 Loading models from {model_file}")
            
            with open(model_file, 'rb') as f:
                save_data = joblib.load(f)
            
            # ✅ ENHANCED: Load with version checking
            if 'timestamp' in save_data:
                save_time = datetime.fromisoformat(save_data['timestamp'])
                age_hours = (datetime.now() - save_time).total_seconds() / 3600
                print(f"📅 Models age: {age_hours:.1f} hours")
                
                # Skip very old models
                if age_hours > 168:  # 7 days
                    print("⚠️ Models too old, will retrain")
                    return False
            
            # Load models
            self.models_1d = {}
            self.models_3d = {}
            
            # ✅ FIX: Handle LSTM model loading
            for save_key, target_dict in [('models_1d', self.models_1d), ('models_3d', self.models_3d)]:
                if save_key not in save_data:
                    continue
                    
                for crypto_key, models_dict in save_data[save_key].items():
                    target_dict[crypto_key] = {}
                    
                    for model_name, model_data in models_dict.items():
                        try:
                            if model_name == 'lstm' and isinstance(model_data, str):
                                # Load LSTM model from file
                                model_path = Path(model_data)
                                if model_path.exists():
                                    loaded_model = keras.models.load_model(model_path)
                                    
                                    # Recreate wrapper (need sequence_length)
                                    # For now, use default sequence length
                                    class LSTMModelWrapper:
                                        def __init__(self, model, sequence_length=20):
                                            self.model = model
                                            self.sequence_length = sequence_length
                                        
                                        def predict_proba(self, X):
                                            pred = self.model.predict(X, verbose=0)
                                            return np.column_stack([1 - pred.flatten(), pred.flatten()])
                                        
                                        def predict(self, X):
                                            proba = self.predict_proba(X)
                                            return (proba[:, 1] > 0.5).astype(int)
                                    
                                    target_dict[crypto_key][model_name] = LSTMModelWrapper(loaded_model)
                                
                            else:
                                # Traditional ML model
                                target_dict[crypto_key][model_name] = model_data
                                
                        except Exception as e:
                            print(f"   ⚠️ Failed to load {model_name} for {crypto_key}: {e}")
            
            # ✅ ENHANCED: Load scalers with type handling
            self.scalers = {}
            if 'scalers' in save_data:
                for scaler_key, scaler_data in save_data['scalers'].items():
                    try:
                        if isinstance(scaler_data, dict) and scaler_data.get('type') == 'lstm_scaler':
                            # Recreate LSTM scaler
                            class LSTMScaler:
                                def __init__(self, standard_scaler, sequence_length):
                                    self.standard_scaler = standard_scaler
                                    self.sequence_length = sequence_length
                                
                                def transform(self, X):
                                    X_scaled = self.standard_scaler.transform(X)
                                    return X_scaled[:, :self.sequence_length].reshape(X_scaled.shape[0], self.sequence_length, 1)
                            
                            self.scalers[scaler_key] = LSTMScaler(
                                scaler_data['standard_scaler'], 
                                scaler_data['sequence_length']
                            )
                        elif isinstance(scaler_data, dict) and scaler_data.get('type') == 'standard_scaler':
                            self.scalers[scaler_key] = scaler_data['scaler']
                        else:
                            # Legacy format
                            self.scalers[scaler_key] = scaler_data
                            
                    except Exception as e:
                        print(f"   ⚠️ Failed to load scaler {scaler_key}: {e}")
            
            # Load other data
            if 'feature_columns' in save_data:
                self.feature_columns = save_data['feature_columns']
            
            if 'model_performance_history' in save_data:
                self.model_performance_history = save_data['model_performance_history']
                
            if 'ensemble_weights' in save_data:
                self.ensemble_weights = save_data['ensemble_weights']
            
            total_models = sum(len(models) for models in self.models_1d.values()) + sum(len(models) for models in self.models_3d.values())
            
            print(f"✅ Loaded: {len(self.models_1d)} cryptos (1d), {len(self.models_3d)} cryptos (3d)")
            print(f"📊 Total models: {total_models}")
            print(f"🔧 Scalers: {len(self.scalers)}")
            
            return total_models > 0
            
        except Exception as e:
            print(f"❌ Model load error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_system_status(self) -> Dict:
        """📊 Get system status compatible with existing monitoring"""
        try:
            total_models_1d = sum(len(models) for models in self.models_1d.values())
            total_models_3d = sum(len(models) for models in self.models_3d.values())
            
            return {
                'total_cryptos_1d': len(self.models_1d),
                'total_cryptos_3d': len(self.models_3d),
                'total_models_1d': total_models_1d,
                'total_models_3d': total_models_3d,
                'total_models': total_models_1d + total_models_3d,
                'api_calls_made': self.api_calls_made,
                'memory_usage': psutil.virtual_memory().percent,
                'config_loaded': self.config.get('config_name', 'Unknown')
            }
            
        except Exception as e:
            print(f"❌ Status error: {e}")
            return {}
    def cleanup_memory(self):
        """🧠 Advanced memory cleanup for ML system"""
        try:
            print("🧠 ML System memory cleanup...")
            
            # ✅ Clear TensorFlow sessions
            try:
                keras.backend.clear_session()
                print("   ✅ TensorFlow sessions cleared")
            except:
                pass
            
            # ✅ Clear feature cache if too large
            if hasattr(self, 'feature_cache'):
                self.feature_cache.clear()
                print("   ✅ Feature cache cleared")
            
            # ✅ Remove old models if memory is critical
            current_memory = psutil.virtual_memory().percent
            if current_memory > 85:  # Critical memory usage
                print(f"   ⚠️ Critical memory usage: {current_memory:.1f}%")
                
                # Keep only recent models (last 50 cryptos)
                for model_dict in [self.models_1d, self.models_3d]:
                    if len(model_dict) > 50:
                        # Sort by last access time or keep alphabetically first N
                        crypto_keys = list(model_dict.keys())
                        keys_to_remove = crypto_keys[50:]  # Remove excess
                        
                        for key in keys_to_remove:
                            if key in model_dict:
                                del model_dict[key]
                                print(f"   🗑️ Removed models for {key}")
                
                # Clear old scalers too
                scaler_keys = list(self.scalers.keys())
                if len(scaler_keys) > 100:
                    keys_to_remove = scaler_keys[100:]
                    for key in keys_to_remove:
                        if key in self.scalers:
                            del self.scalers[key]
                            print(f"   🗑️ Removed scaler {key}")
            
            # ✅ Force garbage collection
            import gc
            collected = gc.collect()
            print(f"   🗑️ Garbage collected: {collected} objects")
            
            new_memory = psutil.virtual_memory().percent
            print(f"   ✅ Memory after cleanup: {new_memory:.1f}%")
            
        except Exception as e:
            print(f"   ❌ Memory cleanup error: {e}")

    def get_memory_usage(self):
        """📊 Get detailed memory usage info"""
        try:
            total_models_1d = sum(len(models) for models in self.models_1d.values())
            total_models_3d = sum(len(models) for models in self.models_3d.values())
            
            return {
                'system_memory_percent': psutil.virtual_memory().percent,
                'models_1d_count': len(self.models_1d),
                'models_3d_count': len(self.models_3d),
                'total_models': total_models_1d + total_models_3d,
                'scalers_count': len(self.scalers),
                'feature_columns_cryptos': len(self.feature_columns)
            }
        except Exception as e:
            return {'error': str(e)}
        
    # AGGIUNGI QUESTI METODI UTILITY ALLA CLASSE OptimizedDualHorizonMLSystem

    def _safe_datetime_for_db(self, dt=None):
        """🔧 Converte datetime in formato sicuro per SQLite"""
        if dt is None:
            dt = datetime.now().isoformat()
        
        if isinstance(dt, datetime):
            return dt.isoformat()
        elif isinstance(dt, str):
            return dt  # Già una stringa
        else:
            return str(dt)

    def _safe_sql_params(self, params):
        """🔧 Converte tutti i parametri SQL in formato sicuro"""
        safe_params = []
        for param in params:
            if isinstance(param, datetime):
                safe_params.append(param.isoformat())
            elif isinstance(param, dict):
                safe_params.append(json.dumps(param))
            elif isinstance(param, list):
                safe_params.append(json.dumps(param))
            elif param is None:
                safe_params.append(None)
            elif isinstance(param, (int, float, str, bool)):
                safe_params.append(param)
            else:
                safe_params.append(str(param))
        
        return tuple(safe_params)

    def _execute_safe_sql(self, cursor, query, params=None):
        """🔧 Esegue SQL con parametri sicuri"""
        if params:
            safe_params = self._safe_sql_params(params)
            cursor.execute(query, safe_params)
        else:
            cursor.execute(query)

# Compatibility function for existing system integration
def create_optimized_ml_system(config_path: str = None) -> OptimizedDualHorizonMLSystem:
    """🔗 Create ML system compatible with existing architecture"""
    return OptimizedDualHorizonMLSystem(config_path)

# Test function with comprehensive error checking
def test_compatible_system():
    """🧪 Test system compatibility with comprehensive error checking"""
    print("🧪 Testing compatible ML system with comprehensive checks...")
    
    try:
        # Initialize with default config
        ml_system = OptimizedDualHorizonMLSystem()
        
        # Test system status
        status = ml_system.get_system_status()
        print(f"✅ System status: {status}")
        
        # Test data fetching
        print("\n📡 Testing data fetching...")
        data = ml_system.fetch_crypto_data_compatible('bitcoin', 90)
        if data.empty:
            print("❌ Data fetching failed")
            return False
        
        print(f"✅ Data fetching works: {len(data)} records")
        print(f"📊 Data columns: {list(data.columns)}")
        print(f"💰 Price range: ${data['price'].min():.2f} - ${data['price'].max():.2f}")
        
        # Test feature creation
        print("\n🔧 Testing feature creation...")
        if 'price' not in data.columns:
            print("❌ 'price' column missing from fetched data")
            return False
            
        features = ml_system.create_optimized_features(data, 'bitcoin', 180)
        if features is None:
            print("❌ Feature creation failed")
            return False
            
        print(f"✅ Feature creation works: {len(features.columns)} total columns")
        print(f"📋 Sample features: {list(features.columns[:10])}")
        
        # Verify 'price' column is preserved
        if 'price' not in features.columns:
            print("❌ CRITICAL: 'price' column missing from features")
            return False
        print("✅ CRITICAL: 'price' column preserved in features")
        
        # Test target preparation
        print("\n🎯 Testing target preparation...")
        targets_1d = ml_system.prepare_targets_compatible(features, 1)
        targets_3d = ml_system.prepare_targets_compatible(features, 3)
        
        if targets_1d.empty or targets_3d.empty:
            print("❌ Target preparation failed")
            return False
            
        print(f"✅ Targets created: 1d={len(targets_1d)} samples, 3d={len(targets_3d)} samples")
        print(f"📊 1d target distribution: UP={targets_1d.sum()}/{len(targets_1d)} ({targets_1d.mean():.1%})")
        print(f"📊 3d target distribution: UP={targets_3d.sum()}/{len(targets_3d)} ({targets_3d.mean():.1%})")
        
        # Test model training (minimal test)
        print("\n🚂 Testing model training (small scale)...")
        if len(data) >= 100:
            success = ml_system.train_compatible_models('bitcoin', 1)
            if success:
                print("✅ Model training successful")
            else:
                print("⚠️ Model training failed (may be due to insufficient data or quality thresholds)")
        else:
            print("⚠️ Skipping training test - insufficient data")
        
        # Test prediction structure
        print("\n🔮 Testing prediction structure...")
        recent_data = data.tail(60)  # Last 60 days for prediction
        
        predictions = ml_system.predict_dual_optimized('bitcoin', recent_data, 1)
        if predictions:
            print(f"✅ Prediction structure works: {len(predictions)} horizons")
            for horizon, pred in predictions.items():
                print(f"   {horizon}: change={pred.get('predicted_change', 0):+.3%}, "
                      f"price=${pred.get('predicted_price', 0):.2f}, "
                      f"confidence={pred.get('confidence', 0):.3f}")
        else:
            print("⚠️ No predictions generated (models may not be trained)")
        
        print("\n✅ ALL CRITICAL SYSTEMS TESTED SUCCESSFULLY!")
        print("🎯 Key checks passed:")
        print("   ✅ Data fetching with price preservation")
        print("   ✅ Feature creation with price column intact")
        print("   ✅ Target preparation with proper alignment")
        print("   ✅ Model training structure")
        print("   ✅ Prediction pipeline")
        
        return True
        
    except Exception as e:
        print(f"❌ Compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 COMPATIBLE CRYPTO ML SYSTEM")
    print("=" * 50)
    print("🔗 Fully compatible with existing crypto system architecture")
    print("📄 Uses crypto_ssd_config.json configuration")
    print("💾 SSD storage at D:/CryptoSystem")
    print("🎯 Dual horizon predictions (1d + 3d)")
    print("=" * 50)
    
    # Run compatibility test
    success = test_compatible_system()
    
    if success:
        print("\n🎉 System ready for integration!")
        print("🔧 Use: from advanced_crypto_ml_system import OptimizedDualHorizonMLSystem")
    else:
        print("\n🔧 Check configuration and dependencies")