# crypto_database_optimized.py - DATABASE OTTIMIZZATO PER SISTEMA 6-9 MESI
"""
Database Manager ottimizzato per sistema 6-9 mesi con alta confidence:
- Schema ottimizzato per dual horizon predictions (1d + 3d)
- Model performance tracking integrato
- Prediction quality scoring avanzato
- Verifiche accuracy complete con feedback loop
- Cache intelligente per performance
- Cleanup automatico ottimizzato
- Metrics dashboard integrato
"""

import sqlite3
import json
import os
import shutil
import gzip
import time
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

class OptimizedCryptoDatabase:
    def __init__(self, db_path='crypto_optimized_6_9months.db', config=None):
        # ✅ FIX: Accept config parameter
        self.config = config or {}
        
        # ✅ FIX: Use full SSD path if provided, otherwise ensure SSD location
        if not str(db_path).startswith('D:/CryptoSystem'):
            # If relative path, make it absolute on SSD
            self.db_path = f"D:/CryptoSystem/database/{Path(db_path).name}"
        else:
            self.db_path = db_path
            
        # ✅ FIX: Use SSD paths from config or defaults
        base_storage = self.config.get('storage', {}).get('base_directory', 'D:/CryptoSystem')
        database_config = self.config.get('storage', {}).get('database', {})
        
        # Create all directories on SSD
        self.base_dir = Path(self.db_path).parent
        self.archive_dir = Path(database_config.get('archive_directory', f'{base_storage}/database/archive'))
        self.cache_dir = Path(database_config.get('cache_directory', f'{base_storage}/cache/database'))
        self.backup_dir = Path(database_config.get('backup_directory', f'{base_storage}/database/backups'))
        
        # ✅ FIX: Create ALL directories including parents
        for directory in [self.base_dir, self.archive_dir, self.cache_dir, self.backup_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Configuration with SSD optimization
        self.disk_config = {
            'max_disk_usage_gb': self.config.get('storage', {}).get('max_disk_usage_gb', 500),
            'auto_cleanup_days': 120,    # 4 mesi per sistema 6-9 mesi
            'warning_threshold_gb': 400,
            'compress_after_days': 30,
            'archive_predictions_days': 60,  # 2 mesi
            'backup_frequency_days': 7
        }
        
        # Performance tracking
        self.query_cache = {}
        self.cache_ttl = 3600  # 1 hour cache
        self.last_cleanup = time.time()
        
        # ✅ FIX: Check SSD space before initialization
        self._check_ssd_space()
        
        self.init_optimized_database()
        
        print(f"💾 Optimized Database initialized: {self.db_path}")
        print(f"🎯 Optimized for: 6-9 months lookback system")
        print(f"📊 Features: Dual horizon + Model tracking + Quality scoring")
        print(f"💾 SSD Storage: {base_storage}")
    
    def _check_ssd_space(self):
        """🔍 Check SSD space before operations"""
        try:
            import shutil
            total, used, free = shutil.disk_usage("D:/")
            free_gb = free / (1024**3)
            
            if free_gb < 10:  # Less than 10GB
                print(f"⚠️ Low SSD space: {free_gb:.1f} GB remaining")
                
            print(f"💾 SSD Space: {free_gb:.1f} GB free")
            return True
        except Exception as e:
            print(f"⚠️ Could not check SSD space: {e}")
            return False
    
    # ✅ ADD: Method to get SSD-optimized connection
    def get_connection(self):
        """Get optimized SQLite connection for SSD"""
        conn = sqlite3.connect(self.db_path)
        
        # ✅ SSD-optimized SQLite settings
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA synchronous=NORMAL')  # Safe for SSD
        conn.execute('PRAGMA cache_size=50000')   # Larger cache for SSD
        conn.execute('PRAGMA temp_store=memory')   # Keep temp in RAM
        conn.execute('PRAGMA mmap_size=268435456') # 256MB memory mapping for SSD
        
        return conn
    
    def init_optimized_database(self):
        """Initialize optimized database schema"""
        conn = self.get_connection()  # ✅ Use optimized connection
        
        # Enable WAL mode for better performance
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA synchronous=NORMAL')
        conn.execute('PRAGMA cache_size=20000')
        conn.execute('PRAGMA temp_store=memory')
        
        # === MAIN PREDICTIONS TABLE ===
        conn.execute('''
            CREATE TABLE IF NOT EXISTS predictions_optimized (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                crypto_id TEXT NOT NULL,
                crypto_name TEXT,
                market_cap_rank INTEGER,
                timestamp DATETIME NOT NULL,
                block_id INTEGER,
                
                -- Current price and context
                current_price REAL NOT NULL,
                volume_24h REAL,
                market_cap REAL,
                
                -- 1-DAY PREDICTIONS
                predicted_change_1d REAL,
                predicted_price_1d REAL,
                confidence_1d REAL,
                quality_score_1d REAL,
                lookback_days_1d INTEGER,
                
                -- 3-DAY PREDICTIONS  
                predicted_change_3d REAL NOT NULL,  -- Required for backward compatibility
                predicted_price_3d REAL NOT NULL,
                confidence_3d REAL NOT NULL,
                quality_score_3d REAL,
                lookback_days_3d INTEGER,
                
                -- MODEL INFORMATION
                models_used_1d TEXT,  -- JSON array of model names
                model_weights_1d TEXT,  -- JSON object of model weights
                models_used_3d TEXT,
                model_weights_3d TEXT,
                ensemble_method TEXT DEFAULT 'weighted_average',
                
                -- MARKET CONTEXT
                market_regime TEXT,
                bitcoin_correlation REAL,
                bitcoin_momentum REAL,
                crypto_beta REAL,
                
                -- VERIFICATION RESULTS (1d)
                actual_price_1d REAL,
                actual_change_1d REAL,
                direction_correct_1d BOOLEAN,
                accuracy_score_1d REAL,
                price_error_pct_1d REAL,
                verification_timestamp_1d DATETIME,
                
                -- VERIFICATION RESULTS (3d)
                actual_price_3d REAL,
                actual_change_3d REAL,
                direction_correct_3d BOOLEAN,
                accuracy_score_3d REAL,
                price_error_pct_3d REAL,
                verification_timestamp_3d DATETIME,
                
                -- METADATA
                system_version TEXT DEFAULT 'optimized_6_9months_v1',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # === MODEL PERFORMANCE TRACKING ===
        conn.execute('''
            CREATE TABLE IF NOT EXISTS model_performance_optimized (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                crypto_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                horizon TEXT NOT NULL,  -- '1d' or '3d'
                
                -- PERFORMANCE METRICS
                total_predictions INTEGER DEFAULT 0,
                correct_predictions INTEGER DEFAULT 0,
                accuracy_rate REAL DEFAULT 0,
                avg_confidence REAL DEFAULT 0,
                avg_accuracy_score REAL DEFAULT 0,
                
                -- DETAILED PERFORMANCE
                direction_accuracy REAL DEFAULT 0,
                price_accuracy REAL DEFAULT 0,
                confidence_calibration REAL DEFAULT 0,
                
                -- TREND ANALYSIS
                recent_accuracy_7d REAL DEFAULT 0,
                recent_accuracy_30d REAL DEFAULT 0,
                performance_trend TEXT DEFAULT 'stable',  -- 'improving', 'declining', 'stable'
                
                -- USAGE STATISTICS
                times_selected INTEGER DEFAULT 0,
                avg_weight REAL DEFAULT 0,
                last_prediction_timestamp DATETIME,
                
                -- METADATA
                first_used DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # === DAILY PERFORMANCE SUMMARY ===
        conn.execute('''
            CREATE TABLE IF NOT EXISTS daily_performance_optimized (
                date DATE PRIMARY KEY,
                
                -- PREDICTION VOLUME
                total_predictions_1d INTEGER DEFAULT 0,
                total_predictions_3d INTEGER DEFAULT 0,
                total_verifications_1d INTEGER DEFAULT 0,
                total_verifications_3d INTEGER DEFAULT 0,
                
                -- ACCURACY METRICS
                accuracy_rate_1d REAL DEFAULT 0,
                accuracy_rate_3d REAL DEFAULT 0,
                avg_confidence_1d REAL DEFAULT 0,
                avg_confidence_3d REAL DEFAULT 0,
                avg_accuracy_score_1d REAL DEFAULT 0,
                avg_accuracy_score_3d REAL DEFAULT 0,
                
                -- QUALITY METRICS
                high_quality_predictions_1d INTEGER DEFAULT 0,  -- quality_score > 0.7
                high_quality_predictions_3d INTEGER DEFAULT 0,
                avg_price_error_1d REAL DEFAULT 0,
                avg_price_error_3d REAL DEFAULT 0,
                
                -- BEST PERFORMERS
                best_crypto_1d TEXT,
                best_accuracy_1d REAL DEFAULT 0,
                best_crypto_3d TEXT,
                best_accuracy_3d REAL DEFAULT 0,
                
                -- MODEL PERFORMANCE
                best_model_1d TEXT,
                best_model_accuracy_1d REAL DEFAULT 0,
                best_model_3d TEXT,
                best_model_accuracy_3d REAL DEFAULT 0,
                
                -- SYSTEM HEALTH
                verification_success_rate REAL DEFAULT 0,
                api_error_rate REAL DEFAULT 0,
                
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # === SYSTEM STATISTICS ===
        conn.execute('''
            CREATE TABLE IF NOT EXISTS system_stats_optimized (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                metric_value REAL,
                metric_text TEXT,
                category TEXT,  -- 'performance', 'usage', 'system'
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # === CYCLE TRACKING ===
        conn.execute('''
            CREATE TABLE IF NOT EXISTS cycle_stats_optimized (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_number INTEGER NOT NULL,
                start_time DATETIME NOT NULL,
                end_time DATETIME,
                
                -- PROCESSING STATS
                total_cryptos_analyzed INTEGER DEFAULT 0,
                successful_analyses INTEGER DEFAULT 0,
                predictions_generated_1d INTEGER DEFAULT 0,
                predictions_generated_3d INTEGER DEFAULT 0,
                
                -- PERFORMANCE STATS
                avg_lookback_days REAL DEFAULT 0,
                avg_confidence_1d REAL DEFAULT 0,
                avg_confidence_3d REAL DEFAULT 0,
                avg_quality_score_1d REAL DEFAULT 0,
                avg_quality_score_3d REAL DEFAULT 0,
                
                -- SYSTEM PERFORMANCE
                elapsed_minutes REAL,
                api_calls_made INTEGER DEFAULT 0,
                cache_hit_rate REAL DEFAULT 0,
                memory_usage_mb REAL DEFAULT 0,
                
                -- VERIFICATION STATS
                verifications_completed INTEGER DEFAULT 0,
                verification_accuracy REAL DEFAULT 0,
                
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # === PERFORMANCE INDEXES ===
        # Primary indexes for fast queries
        conn.execute('CREATE INDEX IF NOT EXISTS idx_predictions_crypto_timestamp ON predictions_optimized(crypto_id, timestamp)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions_optimized(timestamp)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_predictions_verification_1d ON predictions_optimized(verification_timestamp_1d)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_predictions_verification_3d ON predictions_optimized(verification_timestamp_3d)')
        
        # Model performance indexes
        conn.execute('CREATE INDEX IF NOT EXISTS idx_model_performance ON model_performance_optimized(crypto_id, model_name, horizon)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_model_accuracy ON model_performance_optimized(horizon, accuracy_rate DESC)')
        
        # Daily performance indexes
        conn.execute('CREATE INDEX IF NOT EXISTS idx_daily_performance_date ON daily_performance_optimized(date)')
        
        # System stats indexes
        conn.execute('CREATE INDEX IF NOT EXISTS idx_system_stats ON system_stats_optimized(metric_name, timestamp)')
        
        conn.commit()
        conn.close()
        
        print("✅ Optimized database schema initialized")
    def get_model_performance_summary(self):
        """📊 Get model performance summary"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_predictions,
                    AVG(confidence_1d) as avg_confidence_1d,
                    AVG(confidence_3d) as avg_confidence_3d,
                    AVG(quality_score_1d) as avg_quality_1d,
                    AVG(quality_score_3d) as avg_quality_3d,
                    COUNT(CASE WHEN confidence_1d >= 0.6 THEN 1 END) as high_conf_1d,
                    COUNT(CASE WHEN confidence_3d >= 0.55 THEN 1 END) as high_conf_3d
                FROM predictions_optimized 
                WHERE timestamp >= datetime('now', '-7 days')
            ''')
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'total_predictions': result[0],
                    'avg_confidence_1d': result[1] or 0,
                    'avg_confidence_3d': result[2] or 0,
                    'avg_quality_1d': result[3] or 0,
                    'avg_quality_3d': result[4] or 0,
                    'high_confidence_1d': result[5],
                    'high_confidence_3d': result[6]
                }
            else:
                return None
                
        except Exception as e:
            print(f"❌ Error getting model performance: {e}")
            return None
        
    def save_prediction_data_optimized(self, prediction_data):
        """💾 Save prediction data from optimized ML system - FIXED to use predictions_optimized"""
        try:
            if not prediction_data or not prediction_data.get('predictions'):
                return False
            
            crypto_id = prediction_data.get('crypto_id', '')
            crypto_name = prediction_data.get('crypto_name', '')
            current_price = prediction_data.get('current_price', 0)
            market_cap_rank = prediction_data.get('market_cap_rank', 0)
            volume_24h = prediction_data.get('volume_24h', 0)
            market_cap = prediction_data.get('market_cap', 0)
            predictions = prediction_data.get('predictions', {})
            timestamp = prediction_data.get('timestamp', datetime.now())
            
            # Estrai dati addizionali
            market_regime = prediction_data.get('market_regime', 'unknown')
            bitcoin_correlation = prediction_data.get('bitcoin_correlation')
            bitcoin_momentum = prediction_data.get('bitcoin_momentum')
            crypto_beta = prediction_data.get('crypto_beta')
            system_version = prediction_data.get('system_version', 'optimized_v1')
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # ✅ CRITICO: Usa la tabella predictions_optimized che ESISTE nel tuo database
            # Prepara i dati per l'inserimento
            pred_1d = predictions.get('1d', {})
            pred_3d = predictions.get('3d', {})
            
            # Dati 1d
            predicted_change_1d = pred_1d.get('predicted_change')
            predicted_price_1d = pred_1d.get('predicted_price')
            confidence_1d = pred_1d.get('confidence')
            quality_score_1d = pred_1d.get('quality_score')
            lookback_days_1d = pred_1d.get('lookback_days', 270)
            models_used_1d = json.dumps(pred_1d.get('models_used', []))
            model_weights_1d = json.dumps(pred_1d.get('model_weights', {}))
            
            # Dati 3d  
            predicted_change_3d = pred_3d.get('predicted_change', 0)
            predicted_price_3d = pred_3d.get('predicted_price', 0)
            confidence_3d = pred_3d.get('confidence', 0)
            quality_score_3d = pred_3d.get('quality_score')
            lookback_days_3d = pred_3d.get('lookback_days', 270)
            models_used_3d = json.dumps(pred_3d.get('models_used', []))
            model_weights_3d = json.dumps(pred_3d.get('model_weights', {}))
            
            ensemble_method = pred_1d.get('ensemble_method', 'weighted_average') or pred_3d.get('ensemble_method', 'weighted_average')
            
            # ✅ INSERT nella tabella corretta predictions_optimized
            cursor.execute('''
                INSERT INTO predictions_optimized (
                    crypto_id, crypto_name, market_cap_rank, timestamp, 
                    current_price, volume_24h, market_cap,
                    
                    -- 1d predictions
                    predicted_change_1d, predicted_price_1d, confidence_1d, 
                    quality_score_1d, lookback_days_1d, models_used_1d, model_weights_1d,
                    
                    -- 3d predictions  
                    predicted_change_3d, predicted_price_3d, confidence_3d,
                    quality_score_3d, lookback_days_3d, models_used_3d, model_weights_3d,
                    
                    -- Additional data
                    ensemble_method, market_regime, bitcoin_correlation, 
                    bitcoin_momentum, crypto_beta, system_version
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?
                )
            ''', (
                crypto_id, crypto_name, market_cap_rank, timestamp,
                current_price, volume_24h, market_cap,
                
                # 1d data
                predicted_change_1d, predicted_price_1d, confidence_1d,
                quality_score_1d, lookback_days_1d, models_used_1d, model_weights_1d,
                
                # 3d data
                predicted_change_3d, predicted_price_3d, confidence_3d,
                quality_score_3d, lookback_days_3d, models_used_3d, model_weights_3d,
                
                # Additional
                ensemble_method, market_regime, bitcoin_correlation,
                bitcoin_momentum, crypto_beta, system_version
            ))
            
            conn.commit()
            conn.close()
            
            print(f"     💾 Saved prediction to predictions_optimized table")
            
            # Update model usage stats
            self._update_model_usage_stats(prediction_data)
            
            return True
            
        except Exception as e:
            print(f"     ❌ Database save error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def vacuum_database(self):
        """🧹 Vacuum database for optimization"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('VACUUM')
            conn.close()
            print("🗃️ Database vacuumed successfully")
            return True
        except Exception as e:
            print(f"❌ Database vacuum error: {e}")
            return False
    
    def _update_model_usage_stats(self, prediction_data):
        """Update model usage statistics"""
        try:
            predictions = prediction_data.get('predictions', {})
            crypto_id = prediction_data['crypto_id']
            
            conn = sqlite3.connect(self.db_path)
            
            for horizon in ['1d', '3d']:
                if horizon in predictions:
                    pred_data = predictions[horizon]
                    models_used = pred_data.get('models_used', [])
                    model_weights = pred_data.get('model_weights', {})
                    
                    for model_name in models_used:
                        weight = model_weights.get(model_name, 0.5)
                        
                        # Update or insert model usage
                        conn.execute('''
                            INSERT OR IGNORE INTO model_performance_optimized 
                            (crypto_id, model_name, horizon, first_used) 
                            VALUES (?, ?, ?, ?)
                        ''', (crypto_id, model_name, horizon, datetime.now()))
                        
                        conn.execute('''
                            UPDATE model_performance_optimized 
                            SET times_selected = times_selected + 1,
                                avg_weight = (avg_weight * times_selected + ?) / (times_selected + 1),
                                last_prediction_timestamp = ?,
                                last_updated = ?
                            WHERE crypto_id = ? AND model_name = ? AND horizon = ?
                        ''', (weight, datetime.now(), datetime.now(), crypto_id, model_name, horizon))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"⚠️ Model usage stats update failed: {e}")
    
    def get_predictions_ready_for_verification_optimized(self):
        """Get predictions ready for verification with optimized query"""
        try:
            cache_key = f"ready_for_verification_{int(time.time() // 300)}"  # 5-minute cache
            
            if cache_key in self.query_cache:
                return self.query_cache[cache_key]
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 1d predictions ready (older than 25 hours)
            cursor.execute('''
                SELECT id, crypto_id, crypto_name, predicted_change_1d, predicted_price_1d, 
                       current_price, timestamp, confidence_1d, models_used_1d, model_weights_1d,
                       quality_score_1d, market_cap_rank
                FROM predictions_optimized 
                WHERE predicted_change_1d IS NOT NULL
                AND actual_price_1d IS NULL
                AND datetime(timestamp) <= datetime('now', '-25 hours')
                ORDER BY timestamp ASC
                LIMIT 100
            ''')
            
            predictions_1d = cursor.fetchall()
            
            # 3d predictions ready (older than 73 hours)
            cursor.execute('''
                SELECT id, crypto_id, crypto_name, predicted_change_3d, predicted_price_3d, 
                       current_price, timestamp, confidence_3d, models_used_3d, model_weights_3d,
                       quality_score_3d, market_cap_rank
                FROM predictions_optimized 
                WHERE actual_price_3d IS NULL
                AND datetime(timestamp) <= datetime('now', '-73 hours')
                ORDER BY timestamp ASC
                LIMIT 100
            ''')
            
            predictions_3d = cursor.fetchall()
            
            conn.close()
            
            result = (predictions_1d, predictions_3d)
            self.query_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            print(f"❌ Error getting predictions for verification: {e}")
            return [], []
    
    def update_prediction_verification_optimized(self, prediction_id, horizon, verification_data):
        """Update prediction with verification results"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            actual_price = verification_data['actual_price']
            actual_change = verification_data['actual_change']
            direction_correct = verification_data['direction_correct']
            accuracy_score = verification_data['accuracy_score']
            price_error_pct = verification_data.get('price_error_pct', 0)
            verification_timestamp = datetime.now().isoformat()
            
            if horizon == '1d':
                conn.execute('''
                    UPDATE predictions_optimized 
                    SET actual_price_1d = ?, actual_change_1d = ?, 
                        direction_correct_1d = ?, accuracy_score_1d = ?,
                        price_error_pct_1d = ?, verification_timestamp_1d = ?
                    WHERE id = ?
                ''', (
                    actual_price, actual_change, direction_correct, accuracy_score,
                    price_error_pct, verification_timestamp, prediction_id
                ))
            else:
                conn.execute('''
                    UPDATE predictions_optimized 
                    SET actual_price_3d = ?, actual_change_3d = ?, 
                        direction_correct_3d = ?, accuracy_score_3d = ?,
                        price_error_pct_3d = ?, verification_timestamp_3d = ?
                    WHERE id = ?
                ''', (
                    actual_price, actual_change, direction_correct, accuracy_score,
                    price_error_pct, verification_timestamp, prediction_id
                ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Error updating verification: {e}")
            return False
    
    def update_model_performance_optimized(self, crypto_id, model_name, horizon, performance_data):
        """Update model performance with verification results"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get current performance
            cursor.execute('''
                SELECT total_predictions, correct_predictions, accuracy_rate, avg_confidence, avg_accuracy_score
                FROM model_performance_optimized
                WHERE crypto_id = ? AND model_name = ? AND horizon = ?
            ''', (crypto_id, model_name, horizon))
            
            result = cursor.fetchone()
            
            is_correct = performance_data['direction_correct']
            confidence = performance_data['confidence']
            accuracy_score = performance_data['accuracy_score']
            
            if result:
                total_preds, correct_preds, accuracy_rate, avg_conf, avg_acc_score = result
                
                # Update metrics
                new_total = total_preds + 1
                new_correct = correct_preds + (1 if is_correct else 0)
                new_accuracy_rate = new_correct / new_total
                new_avg_conf = (avg_conf * total_preds + confidence) / new_total
                new_avg_acc_score = (avg_acc_score * total_preds + accuracy_score) / new_total
                
                # Calculate performance trend
                performance_trend = 'stable'
                if new_accuracy_rate > accuracy_rate + 0.05:
                    performance_trend = 'improving'
                elif new_accuracy_rate < accuracy_rate - 0.05:
                    performance_trend = 'declining'
                
                conn.execute('''
                    UPDATE model_performance_optimized
                    SET total_predictions = ?, correct_predictions = ?, accuracy_rate = ?,
                        avg_confidence = ?, avg_accuracy_score = ?,
                        direction_accuracy = ?, confidence_calibration = ?,
                        performance_trend = ?, last_updated = ?
                    WHERE crypto_id = ? AND model_name = ? AND horizon = ?
                ''', (
                    new_total, new_correct, new_accuracy_rate, new_avg_conf, new_avg_acc_score,
                    new_accuracy_rate, confidence if is_correct else (1 - confidence),
                    performance_trend, datetime.now(), crypto_id, model_name, horizon
                ))
            else:
                # Insert new performance record
                conn.execute('''
                    INSERT INTO model_performance_optimized (
                        crypto_id, model_name, horizon, total_predictions, correct_predictions,
                        accuracy_rate, avg_confidence, avg_accuracy_score,
                        direction_accuracy, confidence_calibration, last_updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    crypto_id, model_name, horizon, 1, 1 if is_correct else 0,
                    1.0 if is_correct else 0.0, confidence, accuracy_score,
                    1.0 if is_correct else 0.0, confidence if is_correct else (1 - confidence),
                    datetime.now()
                ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Error updating model performance: {e}")
            return False
    
    def get_optimized_performance_summary(self, days=7):
        """Get comprehensive performance summary"""
        try:
            cache_key = f"performance_summary_{days}_{int(time.time() // 3600)}"  # 1-hour cache
            
            if cache_key in self.query_cache:
                return self.query_cache[cache_key]
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            # Overall summary
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_predictions,
                    COUNT(CASE WHEN predicted_change_1d IS NOT NULL THEN 1 END) as predictions_1d,
                    COUNT(CASE WHEN actual_price_1d IS NOT NULL THEN 1 END) as verified_1d,
                    COUNT(CASE WHEN actual_price_3d IS NOT NULL THEN 1 END) as verified_3d,
                    AVG(CASE WHEN direction_correct_1d = 1 THEN 1.0 ELSE 0.0 END) as accuracy_1d,
                    AVG(CASE WHEN direction_correct_3d = 1 THEN 1.0 ELSE 0.0 END) as accuracy_3d,
                    AVG(confidence_1d) as avg_confidence_1d,
                    AVG(confidence_3d) as avg_confidence_3d,
                    AVG(quality_score_1d) as avg_quality_1d,
                    AVG(quality_score_3d) as avg_quality_3d
                FROM predictions_optimized 
                WHERE timestamp >= ?
            ''', (cutoff_date,))
            
            summary = cursor.fetchone()
            
            # Best performing cryptos
            cursor.execute('''
                SELECT crypto_name, 
                       AVG(CASE WHEN direction_correct_1d = 1 THEN 1.0 ELSE 0.0 END) as accuracy_1d,
                       COUNT(CASE WHEN actual_price_1d IS NOT NULL THEN 1 END) as count_1d
                FROM predictions_optimized 
                WHERE timestamp >= ? AND actual_price_1d IS NOT NULL
                GROUP BY crypto_name
                HAVING count_1d >= 3
                ORDER BY accuracy_1d DESC
                LIMIT 5
            ''', (cutoff_date,))
            
            best_cryptos_1d = cursor.fetchall()
            
            cursor.execute('''
                SELECT crypto_name, 
                       AVG(CASE WHEN direction_correct_3d = 1 THEN 1.0 ELSE 0.0 END) as accuracy_3d,
                       COUNT(CASE WHEN actual_price_3d IS NOT NULL THEN 1 END) as count_3d
                FROM predictions_optimized 
                WHERE timestamp >= ? AND actual_price_3d IS NOT NULL
                GROUP BY crypto_name
                HAVING count_3d >= 2
                ORDER BY accuracy_3d DESC
                LIMIT 5
            ''', (cutoff_date,))
            
            best_cryptos_3d = cursor.fetchall()
            
            # Model performance
            cursor.execute('''
                SELECT model_name, horizon, AVG(accuracy_rate) as avg_accuracy, COUNT(*) as crypto_count
                FROM model_performance_optimized
                WHERE last_updated >= ?
                GROUP BY model_name, horizon
                ORDER BY horizon, avg_accuracy DESC
            ''', (cutoff_date,))
            
            model_performance = cursor.fetchall()
            
            conn.close()
            
            result = {
                'summary': {
                    'total_predictions': summary[0] or 0,
                    'predictions_1d': summary[1] or 0,
                    'verified_1d': summary[2] or 0,
                    'verified_3d': summary[3] or 0,
                    'accuracy_1d': summary[4] or 0,
                    'accuracy_3d': summary[5] or 0,
                    'avg_confidence_1d': summary[6] or 0,
                    'avg_confidence_3d': summary[7] or 0,
                    'avg_quality_1d': summary[8] or 0,
                    'avg_quality_3d': summary[9] or 0
                },
                'best_cryptos_1d': best_cryptos_1d,
                'best_cryptos_3d': best_cryptos_3d,
                'model_performance': model_performance,
                'days': days,
                'generated_at': datetime.now().isoformat()
            }
            
            self.query_cache[cache_key] = result
            return result
            
        except Exception as e:
            print(f"❌ Error getting performance summary: {e}")
            return None
    
    def save_cycle_stats_optimized(self, cycle_data):
        """Save comprehensive cycle statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            conn.execute('''
                INSERT INTO cycle_stats_optimized (
                    cycle_number, start_time, end_time, total_cryptos_analyzed, successful_analyses,
                    predictions_generated_1d, predictions_generated_3d, avg_lookback_days,
                    avg_confidence_1d, avg_confidence_3d, avg_quality_score_1d, avg_quality_score_3d,
                    elapsed_minutes, api_calls_made, cache_hit_rate, memory_usage_mb,
                    verifications_completed, verification_accuracy
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                cycle_data.get('cycle_number', 0),
                cycle_data.get('start_time', datetime.now()),
                cycle_data.get('end_time', datetime.now()),
                cycle_data.get('total_cryptos_analyzed', 0),
                cycle_data.get('successful_analyses', 0),
                cycle_data.get('predictions_generated_1d', 0),
                cycle_data.get('predictions_generated_3d', 0),
                cycle_data.get('avg_lookback_days', 0),
                cycle_data.get('avg_confidence_1d', 0),
                cycle_data.get('avg_confidence_3d', 0),
                cycle_data.get('avg_quality_score_1d', 0),
                cycle_data.get('avg_quality_score_3d', 0),
                cycle_data.get('elapsed_minutes', 0),
                cycle_data.get('api_calls_made', 0),
                cycle_data.get('cache_hit_rate', 0),
                cycle_data.get('memory_usage_mb', 0),
                cycle_data.get('verifications_completed', 0),
                cycle_data.get('verification_accuracy', 0)
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Error saving cycle stats: {e}")
            return False
    
    def update_daily_performance_optimized(self, date=None):
        """Update comprehensive daily performance metrics"""
        try:
            if date is None:
                date = datetime.now().date()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Calculate daily metrics
            cursor.execute('''
                SELECT 
                    COUNT(CASE WHEN predicted_change_1d IS NOT NULL THEN 1 END) as total_1d,
                    COUNT(CASE WHEN actual_price_1d IS NOT NULL THEN 1 END) as verified_1d,
                    COUNT(CASE WHEN direction_correct_1d = 1 THEN 1 END) as correct_1d,
                    AVG(confidence_1d) as avg_conf_1d,
                    AVG(accuracy_score_1d) as avg_score_1d,
                    COUNT(CASE WHEN quality_score_1d > 0.7 THEN 1 END) as high_quality_1d,
                    AVG(price_error_pct_1d) as avg_error_1d,
                    
                    COUNT(*) as total_3d,
                    COUNT(CASE WHEN actual_price_3d IS NOT NULL THEN 1 END) as verified_3d,
                    COUNT(CASE WHEN direction_correct_3d = 1 THEN 1 END) as correct_3d,
                    AVG(confidence_3d) as avg_conf_3d,
                    AVG(accuracy_score_3d) as avg_score_3d,
                    COUNT(CASE WHEN quality_score_3d > 0.7 THEN 1 END) as high_quality_3d,
                    AVG(price_error_pct_3d) as avg_error_3d
                FROM predictions_optimized 
                WHERE DATE(timestamp) = ?
            ''', (date,))
            
            metrics = cursor.fetchone()
            
            if metrics:
                total_1d, verified_1d, correct_1d, avg_conf_1d, avg_score_1d, high_quality_1d, avg_error_1d, \
                total_3d, verified_3d, correct_3d, avg_conf_3d, avg_score_3d, high_quality_3d, avg_error_3d = metrics
                
                accuracy_1d = correct_1d / verified_1d if verified_1d > 0 else 0
                accuracy_3d = correct_3d / verified_3d if verified_3d > 0 else 0
                verification_rate = (verified_1d + verified_3d) / (total_1d + total_3d) if (total_1d + total_3d) > 0 else 0
                
                # Find best performers
                cursor.execute('''
                    SELECT crypto_name, accuracy_score_1d 
                    FROM predictions_optimized 
                    WHERE DATE(timestamp) = ? AND accuracy_score_1d IS NOT NULL
                    ORDER BY accuracy_score_1d DESC LIMIT 1
                ''', (date,))
                best_1d_result = cursor.fetchone()
                best_crypto_1d = best_1d_result[0] if best_1d_result else None
                best_accuracy_1d = best_1d_result[1] if best_1d_result else 0
                
                cursor.execute('''
                    SELECT crypto_name, accuracy_score_3d 
                    FROM predictions_optimized 
                    WHERE DATE(timestamp) = ? AND accuracy_score_3d IS NOT NULL
                    ORDER BY accuracy_score_3d DESC LIMIT 1
                ''', (date,))
                best_3d_result = cursor.fetchone()
                best_crypto_3d = best_3d_result[0] if best_3d_result else None
                best_accuracy_3d = best_3d_result[1] if best_3d_result else 0
                
                # Insert or update daily performance
                conn.execute('''
                    INSERT OR REPLACE INTO daily_performance_optimized (
                        date, total_predictions_1d, total_verifications_1d, accuracy_rate_1d,
                        avg_confidence_1d, avg_accuracy_score_1d, high_quality_predictions_1d, avg_price_error_1d,
                        total_predictions_3d, total_verifications_3d, accuracy_rate_3d,
                        avg_confidence_3d, avg_accuracy_score_3d, high_quality_predictions_3d, avg_price_error_3d,
                        best_crypto_1d, best_accuracy_1d, best_crypto_3d, best_accuracy_3d,
                        verification_success_rate
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    date, total_1d or 0, verified_1d or 0, accuracy_1d,
                    avg_conf_1d or 0, avg_score_1d or 0, high_quality_1d or 0, avg_error_1d or 0,
                    total_3d or 0, verified_3d or 0, accuracy_3d,
                    avg_conf_3d or 0, avg_score_3d or 0, high_quality_3d or 0, avg_error_3d or 0,
                    best_crypto_1d, best_accuracy_1d, best_crypto_3d, best_accuracy_3d,
                    verification_rate
                ))
                
                conn.commit()
            
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Error updating daily performance: {e}")
            return False
    
    def show_optimized_statistics(self):
        """Show comprehensive optimized statistics"""
        print("\n📊 OPTIMIZED DATABASE STATISTICS")
        print("=" * 70)
        
        # Basic disk info
        disk_info = self.check_disk_usage()
        if disk_info:
            print(f"💾 STORAGE:")
            print(f"   • Database: {disk_info['database_size_mb']:.1f} MB")
            print(f"   • Cache: {disk_info['cache_size_mb']:.1f} MB")
            print(f"   • Archive: {disk_info['archive_size_mb']:.1f} MB")
            print(f"   • Total: {disk_info['total_crypto_data_gb']:.2f} GB")
            print(f"   • Usage: {disk_info['usage_percentage']:.1f}% of limit")
        
        # Performance summary
        performance = self.get_optimized_performance_summary(30)
        if performance:
            summary = performance['summary']
            
            print(f"\n📈 PERFORMANCE SUMMARY (30 days):")
            print(f"   • Total predictions: {summary['total_predictions']:,}")
            print(f"   • 1d predictions: {summary['predictions_1d']:,}")
            print(f"   • 3d predictions: {summary['total_predictions'] - summary['predictions_1d']:,}")
            print(f"   • Verified 1d: {summary['verified_1d']:,}")
            print(f"   • Verified 3d: {summary['verified_3d']:,}")
            
            print(f"\n🎯 ACCURACY METRICS:")
            print(f"   • 1d accuracy: {summary['accuracy_1d']:.1%}")
            print(f"   • 3d accuracy: {summary['accuracy_3d']:.1%}")
            print(f"   • Avg confidence 1d: {summary['avg_confidence_1d']:.1%}")
            print(f"   • Avg confidence 3d: {summary['avg_confidence_3d']:.1%}")
            print(f"   • Avg quality 1d: {summary['avg_quality_1d']:.3f}")
            print(f"   • Avg quality 3d: {summary['avg_quality_3d']:.3f}")
            
            # Best performers
            if performance['best_cryptos_1d']:
                print(f"\n🏆 BEST 1d PERFORMERS:")
                for crypto_name, accuracy, count in performance['best_cryptos_1d'][:3]:
                    print(f"   • {crypto_name}: {accuracy:.1%} ({count} predictions)")
            
            if performance['best_cryptos_3d']:
                print(f"\n🏆 BEST 3d PERFORMERS:")
                for crypto_name, accuracy, count in performance['best_cryptos_3d'][:3]:
                    print(f"   • {crypto_name}: {accuracy:.1%} ({count} predictions)")
            
            # Model performance
            if performance['model_performance']:
                print(f"\n🤖 MODEL PERFORMANCE:")
                for model_name, horizon, avg_accuracy, crypto_count in performance['model_performance'][:6]:
                    print(f"   • {model_name} ({horizon}): {avg_accuracy:.1%} ({crypto_count} cryptos)")
        
        # Database stats
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM predictions_optimized')
            total_predictions = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM model_performance_optimized')
            model_records = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM cycle_stats_optimized')
            cycle_records = cursor.fetchone()[0]
            
            print(f"\n🗄️ DATABASE RECORDS:")
            print(f"   • Predictions: {total_predictions:,}")
            print(f"   • Model performance: {model_records:,}")
            print(f"   • Cycle stats: {cycle_records:,}")
            
            # Cache statistics
            print(f"\n💾 CACHE STATISTICS:")
            print(f"   • Cache entries: {len(self.query_cache)}")
            print(f"   • Cache TTL: {self.cache_ttl // 60} minutes")
            
            conn.close()
            
        except Exception as e:
            print(f"❌ Error getting database stats: {e}")
        
        return disk_info
    
    def check_disk_usage(self):
        """Check disk usage with optimized reporting"""
        try:
            # Database size
            db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            
            # Cache directory size
            cache_size = sum(f.stat().st_size for f in Path(self.cache_dir).glob('**/*') if f.is_file())
            
            # Archive directory size
            archive_size = sum(f.stat().st_size for f in Path(self.archive_dir).glob('**/*') if f.is_file())
            
            # Backup directory size
            backup_size = sum(f.stat().st_size for f in Path(self.backup_dir).glob('**/*') if f.is_file())
            
            total_size_gb = (db_size + cache_size + archive_size + backup_size) / (1024**3)
            
            # Filesystem usage
            disk_usage = shutil.disk_usage(self.base_dir)
            available_gb = disk_usage.free / (1024**3)
            
            return {
                'database_size_mb': db_size / (1024**2),
                'cache_size_mb': cache_size / (1024**2),
                'archive_size_mb': archive_size / (1024**2),
                'backup_size_mb': backup_size / (1024**2),
                'total_crypto_data_gb': total_size_gb,
                'disk_available_gb': available_gb,
                'usage_percentage': (total_size_gb / self.disk_config['max_disk_usage_gb']) * 100
            }
            
        except Exception as e:
            print(f"❌ Error checking disk usage: {e}")
            return None
    
    def optimize_database_performance(self):
        """Optimize database performance"""
        try:
            print("🔧 Optimizing database performance...")
            
            conn = sqlite3.connect(self.db_path)
            
            # VACUUM to reclaim space
            print("   🧹 Running VACUUM...")
            conn.execute('VACUUM')
            
            # ANALYZE to update statistics
            print("   📊 Running ANALYZE...")
            conn.execute('ANALYZE')
            
            # Clear query cache
            self.query_cache.clear()
            print("   💾 Cleared query cache")
            
            conn.close()
            
            print("✅ Database optimization completed")
            
        except Exception as e:
            print(f"❌ Database optimization failed: {e}")
    
    def auto_cleanup_optimized(self):
        """Intelligent auto cleanup for optimized system"""
        print("🧹 Running optimized auto-cleanup...")
        
        disk_info = self.check_disk_usage()
        if not disk_info:
            print("❌ Cannot verify disk space")
            return False
        
        current_usage = disk_info['usage_percentage']
        cleanup_performed = False
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Cleanup based on usage level
            if current_usage > 80:
                print(f"🚨 High disk usage ({current_usage:.1f}%) - aggressive cleanup")
                
                # Archive old predictions (60 days)
                cursor = conn.execute('''
                    DELETE FROM predictions_optimized 
                    WHERE timestamp < datetime('now', '-60 days')
                    AND (actual_price_1d IS NOT NULL OR actual_price_3d IS NOT NULL)
                ''')
                deleted = cursor.rowcount
                print(f"   🗄️ Archived {deleted} old verified predictions")
                cleanup_performed = True
                
            elif current_usage > 60:
                print(f"⚠️ Moderate disk usage ({current_usage:.1f}%) - standard cleanup")
                
                # Archive very old predictions (90 days)
                cursor = conn.execute('''
                    DELETE FROM predictions_optimized 
                    WHERE timestamp < datetime('now', '-90 days')
                    AND (actual_price_1d IS NOT NULL AND actual_price_3d IS NOT NULL)
                ''')
                deleted = cursor.rowcount
                print(f"   🗄️ Archived {deleted} old predictions")
                cleanup_performed = True
            
            # Clean old model performance records
            cursor = conn.execute('''
                DELETE FROM model_performance_optimized 
                WHERE last_updated < datetime('now', '-120 days')
                AND total_predictions < 5
            ''')
            deleted_models = cursor.rowcount
            if deleted_models > 0:
                print(f"   🤖 Cleaned {deleted_models} old model records")
                cleanup_performed = True
            
            # Clean old cycle stats
            cursor = conn.execute('''
                DELETE FROM cycle_stats_optimized 
                WHERE start_time < datetime('now', '-180 days')
            ''')
            deleted_cycles = cursor.rowcount
            if deleted_cycles > 0:
                print(f"   📊 Cleaned {deleted_cycles} old cycle records")
                cleanup_performed = True
            
            conn.commit()
            conn.close()
            
            # Clean cache files
            if self.cache_dir.exists():
                cache_files = list(self.cache_dir.glob('*.json'))
                old_cache = [f for f in cache_files 
                           if (datetime.now() - datetime.fromtimestamp(f.stat().st_mtime)).days > 7]
                
                for old_file in old_cache:
                    old_file.unlink()
                
                if old_cache:
                    print(f"   💾 Cleaned {len(old_cache)} old cache files")
                    cleanup_performed = True
            
            # Optimize database if cleanup performed
            if cleanup_performed:
                self.optimize_database_performance()
            
            if cleanup_performed:
                new_usage = self.check_disk_usage()['usage_percentage']
                print(f"✅ Cleanup completed: {current_usage:.1f}% → {new_usage:.1f}%")
            else:
                print(f"✅ No cleanup needed ({current_usage:.1f}% usage)")
            
            return True
            
        except Exception as e:
            print(f"❌ Cleanup error: {e}")
            return False
    
    def close(self):
        """Close database connection and cleanup"""
        try:
            # Clear cache
            self.query_cache.clear()
            print("💾 Database resources cleaned up")
        except Exception as e:
            print(f"⚠️ Database cleanup error: {e}")


# === TEST FUNCTION ===
def test_optimized_database():
    """🧪 Test optimized database system"""
    print("🧪 Testing Optimized Database System...")
    
    # Initialize database
    db = OptimizedCryptoDatabase('test_optimized.db')
    
    # Test prediction save
    test_prediction = {
        'crypto_id': 'bitcoin',
        'crypto_name': 'Bitcoin',
        'market_cap_rank': 1,
        'timestamp': datetime.now(),
        'current_price': 45000,
        'volume_24h': 20000000000,
        'market_cap': 900000000000,
        'predictions': {
            '1d': {
                'predicted_change': 0.05,
                'predicted_price': 47250,
                'confidence': 0.75,
                'quality_score': 0.82,
                'lookback_days': 180,
                'models_used': ['rf_clf', 'xgb_clf', 'catboost_clf'],
                'model_weights': {'rf_clf': 0.3, 'xgb_clf': 0.4, 'catboost_clf': 0.3}
            },
            '3d': {
                'predicted_change': 0.08,
                'predicted_price': 48600,
                'confidence': 0.68,
                'quality_score': 0.78,
                'lookback_days': 270,
                'models_used': ['rf_clf', 'lstm_clf'],
                'model_weights': {'rf_clf': 0.6, 'lstm_clf': 0.4}
            }
        },
        'market_regime': 'bull_moderate',
        'bitcoin_correlation': 1.0,
        'bitcoin_momentum': 0.03,
        'crypto_beta': 1.0,
        'system_version': 'optimized_6_9months_v1'
    }
    
    success = db.save_optimized_prediction(test_prediction)
    print(f"✅ Test prediction saved: {success}")
    
    # Test performance summary
    performance = db.get_optimized_performance_summary(7)
    if performance:
        print(f"✅ Performance summary generated: {performance['summary']['total_predictions']} predictions")
    
    # Test model performance update
    performance_data = {
        'direction_correct': True,
        'confidence': 0.75,
        'accuracy_score': 0.85
    }
    
    model_updated = db.update_model_performance_optimized('bitcoin', 'rf_clf', '1d', performance_data)
    print(f"✅ Model performance updated: {model_updated}")
    
    # Test cycle stats
    cycle_data = {
        'cycle_number': 1,
        'start_time': datetime.now() - timedelta(minutes=45),
        'end_time': datetime.now(),
        'total_cryptos_analyzed': 50,
        'successful_analyses': 45,
        'predictions_generated_1d': 20,
        'predictions_generated_3d': 45,
        'avg_lookback_days': 225,
        'avg_confidence_1d': 0.72,
        'avg_confidence_3d': 0.68,
        'elapsed_minutes': 45,
        'api_calls_made': 150,
        'cache_hit_rate': 0.65,
        'memory_usage_mb': 256
    }
    
    cycle_saved = db.save_cycle_stats_optimized(cycle_data)
    print(f"✅ Cycle stats saved: {cycle_saved}")
    
    # Test daily performance update
    daily_updated = db.update_daily_performance_optimized()
    print(f"✅ Daily performance updated: {daily_updated}")
    
    # Show statistics
    db.show_optimized_statistics()
    
    # Cleanup test database
    db.close()
    import os
    if os.path.exists('test_optimized.db'):
        os.remove('test_optimized.db')
        print("🧹 Test database cleaned up")
    
    print("✅ Optimized database system test completed")


if __name__ == "__main__":
    print("💾 OPTIMIZED CRYPTO DATABASE SYSTEM")
    print("=" * 50)
    print("🎯 Optimized for: 6-9 months lookback system")
    print("📊 Features: Dual horizon tracking + Model performance + Quality scoring")
    print("🔧 Performance: Intelligent caching + Auto cleanup + Optimization")
    print("=" * 50)
    
    # Run test
    test_optimized_database()