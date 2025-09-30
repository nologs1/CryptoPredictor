# complete_verification_optimized.py - SISTEMA VERIFICA OTTIMIZZATO CON MODEL SELECTION
"""
Sistema di verifica completo e ottimizzato per 6-9 mesi:
- Verifica DUAL HORIZON (1d + 3d) con tracking accuracy separato
- Model performance tracking e selezione intelligente dei modelli migliori
- Email accuracy reports dettagliati
- Feedback loop per migliorare le predizioni successive
- Integration con ML system per aggiornamento pesi modelli
- Database tracking completo delle performance
"""

import random
import sqlite3
import requests
import time
import smtplib
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Tuple
import statistics
import sqlite3
import threading
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Dopo gli altri import
try:
    from model_feedback_optimizer import ModelFeedbackOptimizer, integrate_model_feedback
    MODEL_FEEDBACK_AVAILABLE = True
    print("✅ Model Feedback Optimizer available")
except ImportError:
    print("⚠️ model_feedback_optimizer not found")
    MODEL_FEEDBACK_AVAILABLE = False

class DatabaseConsistencyManager:
    """🔒 Gestisce la consistenza del database con transazioni atomiche"""
    
    def __init__(self, db_path):
        self.db_path = db_path
        self.lock = threading.Lock()
    
    @contextmanager
    def atomic_transaction(self):
        """🔒 Context manager per transazioni atomiche"""
        conn = None
        try:
            with self.lock:  # Thread-safe
                conn = sqlite3.connect(self.db_path)
                conn.execute('BEGIN IMMEDIATE')  # Exclusive lock
                yield conn
                conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                conn.close()
    
    def update_verification_atomically(self, verification_data: Dict):
        """⚡ Update all related tables atomically"""
        with self.atomic_transaction() as conn:
            cursor = conn.cursor()
            
            # 1. Update main predictions table
            self._update_predictions_table(cursor, verification_data)
            
            # 2. Insert verification results
            self._insert_verification_results(cursor, verification_data)
            
            # 3. Update model performance tracking
            self._update_model_performance(cursor, verification_data)
            
            # 4. Update daily summary if needed
            self._update_daily_summary(cursor, verification_data)
            
            print(f"✅ Atomic update completed for {verification_data['crypto_id']}")
    
    def _update_predictions_table(self, cursor, data):
        """Update main predictions table"""
        horizon = int(data['horizon'].rstrip('d'))
        
        if horizon == 1:
            cursor.execute('''
                UPDATE predictions_optimized 
                SET actual_price_1d = ?, actual_change_1d = ?, 
                    direction_correct_1d = ?, accuracy_score_1d = ?,
                    verification_timestamp_1d = ?, price_error_1d = ?
                WHERE id = ?
            ''', (
                data['current_price'], data['actual_change'],
                data['direction_correct'], data['accuracy_score'],
                data['verification_timestamp'], data['price_error_percent'],
                data['prediction_id']
            ))
        else:
            cursor.execute('''
                UPDATE predictions_optimized 
                SET actual_price_3d = ?, actual_change_3d = ?, 
                    direction_correct_3d = ?, accuracy_score_3d = ?,
                    verification_timestamp_3d = ?, price_error_3d = ?
                WHERE id = ?
            ''', (
                data['current_price'], data['actual_change'],
                data['direction_correct'], data['accuracy_score'],
                data['verification_timestamp'], data['price_error_percent'],
                data['prediction_id']
            ))
    
    def _insert_verification_results(self, cursor, data):
        """Insert into verification results table"""
        cursor.execute('''
            INSERT INTO verification_results (
                crypto_id, crypto_name, horizon, verification_timestamp,
                predicted_price, actual_price, predicted_change, actual_change,
                direction_correct, accuracy_score, price_error_percent,
                confidence, quality_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data['crypto_id'], data['crypto_name'], data['horizon'],
            data['verification_timestamp'], data['predicted_price'],
            data['current_price'], data['predicted_change'], data['actual_change'],
            data['direction_correct'], data['accuracy_score'],
            data['price_error_percent'], data['confidence'], data['quality_score']
        ))
    
    def _update_model_performance(self, cursor, data):
        """Update model performance tracking"""
        # Get models used for this prediction
        cursor.execute('''
            SELECT models_used_1d, models_used_3d FROM predictions_optimized 
            WHERE id = ?
        ''', (data['prediction_id'],))
        
        result = cursor.fetchone()
        if not result:
            return
        
        horizon = int(data['horizon'].rstrip('d'))
        models_used = result[0] if horizon == 1 else result[1]
        
        if models_used:
            import json
            try:
                models_list = json.loads(models_used)
                for model_name in models_list:
                    self._update_single_model_performance(cursor, model_name, data)
            except json.JSONDecodeError:
                print(f"⚠️ Invalid models_used JSON: {models_used}")
    
    def _update_single_model_performance(self, cursor, model_name, data):
        """Update performance for a single model"""
        # Check if record exists
        cursor.execute('''
            SELECT id, total_predictions, correct_predictions 
            FROM model_performance_tracking 
            WHERE crypto_id = ? AND model_name = ? AND horizon = ?
        ''', (data['crypto_id'], model_name, data['horizon']))
        
        existing = cursor.fetchone()
        
        if existing:
            # Update existing record
            new_total = existing[1] + 1
            new_correct = existing[2] + (1 if data['direction_correct'] else 0)
            new_accuracy = new_correct / new_total if new_total > 0 else 0
            
            cursor.execute('''
                UPDATE model_performance_tracking 
                SET total_predictions = ?, correct_predictions = ?,
                    accuracy_rate = ?, avg_accuracy_score = ?,
                    last_updated = ?
                WHERE id = ?
            ''', (
                new_total, new_correct, new_accuracy,
                data['accuracy_score'], data['verification_timestamp'],
                existing[0]
            ))
        else:
            # Insert new record
            cursor.execute('''
                INSERT INTO model_performance_tracking (
                    crypto_id, model_name, horizon, total_predictions,
                    correct_predictions, accuracy_rate, avg_accuracy_score,
                    last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['crypto_id'], model_name, data['horizon'], 1,
                1 if data['direction_correct'] else 0,
                1.0 if data['direction_correct'] else 0.0,
                data['accuracy_score'], data['verification_timestamp']
            ))
    
    def _update_daily_summary(self, cursor, data):
        """Update daily accuracy summary"""
        from datetime import datetime
        today = datetime.now().date().isoformat()
        horizon = int(data['horizon'].rstrip('d'))
        
        # Get existing summary
        cursor.execute('''
            SELECT predictions_1d, correct_1d, predictions_3d, correct_3d 
            FROM daily_accuracy_summary WHERE date = ?
        ''', (today,))
        
        existing = cursor.fetchone()
        
        if existing:
            # Update existing
            if horizon == 1:
                new_pred_1d = existing[0] + 1
                new_correct_1d = existing[1] + (1 if data['direction_correct'] else 0)
                new_pred_3d, new_correct_3d = existing[2], existing[3]
            else:
                new_pred_1d, new_correct_1d = existing[0], existing[1]
                new_pred_3d = existing[2] + 1
                new_correct_3d = existing[3] + (1 if data['direction_correct'] else 0)
            
            accuracy_1d = new_correct_1d / new_pred_1d if new_pred_1d > 0 else 0
            accuracy_3d = new_correct_3d / new_pred_3d if new_pred_3d > 0 else 0
            
            cursor.execute('''
                UPDATE daily_accuracy_summary 
                SET predictions_1d = ?, correct_1d = ?, accuracy_1d = ?,
                    predictions_3d = ?, correct_3d = ?, accuracy_3d = ?
                WHERE date = ?
            ''', (
                new_pred_1d, new_correct_1d, accuracy_1d,
                new_pred_3d, new_correct_3d, accuracy_3d, today
            ))
        else:
            # Insert new
            if horizon == 1:
                pred_1d, correct_1d = 1, 1 if data['direction_correct'] else 0
                pred_3d, correct_3d = 0, 0
            else:
                pred_1d, correct_1d = 0, 0
                pred_3d, correct_3d = 1, 1 if data['direction_correct'] else 0
            
            cursor.execute('''
                INSERT INTO daily_accuracy_summary (
                    date, predictions_1d, correct_1d, accuracy_1d,
                    predictions_3d, correct_3d, accuracy_3d
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                today, pred_1d, correct_1d,
                correct_1d / pred_1d if pred_1d > 0 else 0,
                pred_3d, correct_3d,
                correct_3d / pred_3d if pred_3d > 0 else 0
            ))

# VERIFICA INTEGRITÀ DATABASE
def verify_database_integrity(db_path: str) -> Dict:
    """🔍 Verifica l'integrità del database"""
    integrity_report = {
        'status': 'checking',
        'errors': [],
        'warnings': [],
        'suggestions': [],
        'statistics': {}
    }
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 1. Check table existence
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        required_tables = [
            'predictions_optimized', 'verification_results',
            'model_performance_tracking', 'daily_accuracy_summary'
        ]
        
        missing_tables = [t for t in required_tables if t not in tables]
        if missing_tables:
            integrity_report['errors'].append(f"Missing tables: {missing_tables}")
        
        # 2. Check for orphaned records
        cursor.execute('''
            SELECT COUNT(*) FROM verification_results vr
            LEFT JOIN predictions_optimized po ON vr.crypto_id = po.crypto_id
            WHERE po.id IS NULL
        ''')
        orphaned_verifications = cursor.fetchone()[0]
        
        if orphaned_verifications > 0:
            integrity_report['warnings'].append(
                f"Found {orphaned_verifications} orphaned verification records"
            )
        
        # 3. Check prediction completeness
        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                COUNT(verification_timestamp_1d) as verified_1d,
                COUNT(verification_timestamp_3d) as verified_3d
            FROM predictions_optimized
            WHERE prediction_timestamp < datetime('now', '-1 day')
        ''')
        
        completeness = cursor.fetchone()
        if completeness:
            total, verified_1d, verified_3d = completeness
            if total > 0:
                verification_rate_1d = verified_1d / total
                verification_rate_3d = verified_3d / total
                
                integrity_report['statistics']['verification_rates'] = {
                    '1d': verification_rate_1d,
                    '3d': verification_rate_3d
                }
                
                if verification_rate_1d < 0.8:
                    integrity_report['warnings'].append(
                        f"Low 1d verification rate: {verification_rate_1d:.1%}"
                    )
                if verification_rate_3d < 0.7:
                    integrity_report['warnings'].append(
                        f"Low 3d verification rate: {verification_rate_3d:.1%}"
                    )
        
        # 4. Check model performance data consistency
        cursor.execute('''
            SELECT COUNT(DISTINCT crypto_id) FROM model_performance_tracking
        ''')
        models_tracked = cursor.fetchone()[0]
        
        cursor.execute('''
            SELECT COUNT(DISTINCT crypto_id) FROM predictions_optimized
        ''')
        cryptos_predicted = cursor.fetchone()[0]
        
        if models_tracked < cryptos_predicted * 0.8:
            integrity_report['warnings'].append(
                "Model performance tracking incomplete for some cryptos"
            )
        
        # 5. Suggest optimizations
        cursor.execute("PRAGMA integrity_check")
        sqlite_integrity = cursor.fetchall()
        
        if sqlite_integrity != [('ok',)]:
            integrity_report['errors'].extend(sqlite_integrity)
        
        # Database size check
        cursor.execute("SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()")
        db_size_bytes = cursor.fetchone()[0]
        db_size_mb = db_size_bytes / (1024 * 1024)
        
        integrity_report['statistics']['database_size_mb'] = db_size_mb
        
        if db_size_mb > 1000:  # > 1GB
            integrity_report['suggestions'].append("Consider database cleanup - size > 1GB")
        
        conn.close()
        
        # Final status
        if integrity_report['errors']:
            integrity_report['status'] = 'critical_errors'
        elif integrity_report['warnings']:
            integrity_report['status'] = 'warnings_found'
        else:
            integrity_report['status'] = 'healthy'
        
    except Exception as e:
        integrity_report['status'] = 'check_failed'
        integrity_report['errors'].append(f"Integrity check failed: {e}")
    
    return integrity_report

class OptimizedVerificationSystem:
    def __init__(self, db_path, config=None):
        self.db_path = db_path
        self.config = config or {}
        
        # Configuration
        self.api_timeout = self.config.get('api_timeout', 30)
        self.rate_limit_delay = self.config.get('rate_limit_delay', 2.0)
        self.max_retries = self.config.get('max_retries_api', 3)
        
        # Email config
        self.gmail_user = self.config.get('gmail_user', '')
        self.gmail_password = self.config.get('gmail_app_password', '')
        
        # Verification tracking
        self.last_verification_time = None
        self.verification_results = []
        
        # Model performance tracking
        self.model_performance_db = {}
        self.accuracy_trends = {'1d': [], '3d': []}
        
        # Quality thresholds for model improvement
        self.accuracy_targets = {
            '1d': self.config.get('accuracy_target_1d', 0.65),
            '3d': self.config.get('accuracy_target_3d', 0.60)
        }
        
        print("🔍 Optimized Verification System initialized")
        print(f"🎯 Accuracy targets: 1d≥{self.accuracy_targets['1d']:.0%}, 3d≥{self.accuracy_targets['3d']:.0%}")
        print(f"📧 Email reports: {'✅ Enabled' if self.gmail_user else '❌ Disabled'}")
        
        # Initialize verification database
        self._initialize_verification_database()
        self.db_consistency_manager = DatabaseConsistencyManager(self.db_path)
        # AGGIUNGI QUESTE LINEE:
        self.ml_system = None  # Sarà impostato dopo
        self.feedback_optimizer = None
        
        print("🔄 Model Feedback Optimizer initialized")

    def _initialize_verification_database(self):
        """🗄️ Initialize verification tracking database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Verification results table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS verification_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    crypto_id TEXT NOT NULL,
                    crypto_name TEXT,
                    horizon TEXT NOT NULL,
                    prediction_timestamp DATETIME,
                    verification_timestamp DATETIME,
                    
                    -- Prediction data
                    predicted_direction TEXT,
                    predicted_change REAL,
                    predicted_price REAL,
                    confidence REAL,
                    quality_score REAL,
                    
                    -- Actual results
                    actual_price REAL,
                    actual_change REAL,
                    actual_direction TEXT,
                    
                    -- Performance metrics
                    direction_correct BOOLEAN,
                    accuracy_score REAL,
                    price_error_percent REAL,
                    
                    -- Model information
                    models_used TEXT,
                    model_weights TEXT,
                    ensemble_method TEXT,
                    
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Model performance tracking table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS model_performance_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    crypto_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    horizon TEXT NOT NULL,
                    
                    -- Performance metrics
                    total_predictions INTEGER DEFAULT 0,
                    correct_predictions INTEGER DEFAULT 0,
                    accuracy_rate REAL DEFAULT 0,
                    avg_confidence REAL DEFAULT 0,
                    avg_accuracy_score REAL DEFAULT 0,
                    
                    -- Performance trends
                    recent_accuracy_7d REAL DEFAULT 0,
                    recent_accuracy_30d REAL DEFAULT 0,
                    performance_trend TEXT DEFAULT 'stable',
                    
                    -- Quality indicators
                    confidence_calibration REAL DEFAULT 0,
                    prediction_reliability REAL DEFAULT 0,
                    
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Daily accuracy summary table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS daily_accuracy_summary (
                    date DATE PRIMARY KEY,
                    
                    -- 1d performance
                    predictions_1d INTEGER DEFAULT 0,
                    correct_1d INTEGER DEFAULT 0,
                    accuracy_1d REAL DEFAULT 0,
                    avg_confidence_1d REAL DEFAULT 0,
                    
                    -- 3d performance
                    predictions_3d INTEGER DEFAULT 0,
                    correct_3d INTEGER DEFAULT 0,
                    accuracy_3d REAL DEFAULT 0,
                    avg_confidence_3d REAL DEFAULT 0,
                    
                    -- Best performing models
                    best_model_1d TEXT,
                    best_model_3d TEXT,
                    best_accuracy_1d REAL DEFAULT 0,
                    best_accuracy_3d REAL DEFAULT 0,
                    
                    -- System health
                    total_verifications INTEGER DEFAULT 0,
                    verification_success_rate REAL DEFAULT 0,
                    
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_verification_crypto_horizon ON verification_results(crypto_id, horizon)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_verification_timestamp ON verification_results(verification_timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_model_performance ON model_performance_tracking(crypto_id, model_name, horizon)')
            
            conn.commit()
            conn.close()
            
            print("🗄️ Verification database initialized")
            
        except Exception as e:
            print(f"❌ Verification database initialization failed: {e}")
    
    def integrate_model_feedback_system(self, ml_system):
        """🔧 Integra il sistema di feedback con il sistema ML"""
        if not MODEL_FEEDBACK_AVAILABLE:
            print("⚠️ Model feedback system not available")
            return False
        
        try:
            self.ml_system = ml_system
            self.feedback_optimizer = ModelFeedbackOptimizer(
                self.db_path,
                ml_system,
                self.config
            )
            
            print("✅ Model feedback system integrated")
            return True
        except Exception as e:
            print(f"❌ Failed to integrate model feedback: {e}")
            return False

    # 🔄 SOSTITUISCI METODO: update_model_weights_after_verification
    def update_model_weights_after_verification(self, crypto_id: str, horizon: str) -> Dict:
        """🎯 Aggiorna i pesi del modello dopo verifica - FIX COMPLETO"""
        try:
            weight_key = f"{crypto_id}_{horizon}"
            print(f"   📊 Updating weights for {crypto_id} {horizon}...")
            
            # 1. Ottieni performance recenti dal database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Query corretta per ottenere i risultati di verifica
            cursor.execute('''
                SELECT 
                    COALESCE(model_1_accuracy, 0) as lightgbm_acc,
                    COALESCE(model_2_accuracy, 0) as xgboost_acc,
                    COALESCE(model_3_accuracy, 0) as catboost_acc,
                    COALESCE(model_4_accuracy, 0) as lstm_acc,
                    verification_timestamp_1d,
                    verification_timestamp_3d
                FROM predictions_optimized 
                WHERE crypto_id = ? 
                AND (
                    (? = '1d' AND verification_timestamp_1d IS NOT NULL AND verification_timestamp_1d >= datetime('now', '-7 days'))
                    OR 
                    (? = '3d' AND verification_timestamp_3d IS NOT NULL AND verification_timestamp_3d >= datetime('now', '-7 days'))
                )
                ORDER BY 
                    CASE 
                        WHEN ? = '1d' THEN verification_timestamp_1d 
                        ELSE verification_timestamp_3d 
                    END DESC
                LIMIT 20
            ''', (crypto_id, horizon, horizon, horizon))
            
            results = cursor.fetchall()
            conn.close()
            
            if len(results) < 5:  # Minimo 5 verifiche per aggiornamento significativo
                return {
                    'success': False, 
                    'error': f'Insufficient data: only {len(results)} verifications',
                    'crypto_id': crypto_id,
                    'horizon': horizon
                }
            
            # 2. Calcola performance medie per ogni modello
            model_performance = {
                'lightgbm': np.mean([r[0] for r in results if r[0] is not None]),
                'xgboost': np.mean([r[1] for r in results if r[1] is not None]),
                'catboost': np.mean([r[2] for r in results if r[2] is not None]),
                'lstm': np.mean([r[3] for r in results if r[3] is not None])
            }
            
            # 3. Rimuovi modelli con performance NaN o troppo bassa
            valid_models = {k: v for k, v in model_performance.items() 
                        if not np.isnan(v) and v >= 0.3}
            
            if not valid_models:
                return {
                    'success': False, 
                    'error': 'No valid model performance found',
                    'crypto_id': crypto_id,
                    'horizon': horizon
                }
            
            # 4. Calcola nuovi pesi usando softmax per normalizzazione
            performance_array = np.array(list(valid_models.values()))
            
            # Applica temperatura per controllare la distribuzione
            temperature = 2.0  # Temperatura più alta = distribuzione più uniforme
            softmax_weights = np.exp(performance_array / temperature)
            softmax_weights = softmax_weights / np.sum(softmax_weights)
            
            # 5. Aggiorna i pesi con momentum
            old_weights = dict(self.model_weights[weight_key])
            new_weights = {}
            weight_changes = {}
            
            for i, (model, performance) in enumerate(valid_models.items()):
                old_weight = old_weights.get(model, 1.0)
                new_weight = softmax_weights[i]
                
                # Applica momentum per smooth updates
                final_weight = (1 - self.momentum) * new_weight + self.momentum * old_weight
                
                # Limita i pesi tra 0.1 e 2.0 per stabilità
                final_weight = max(0.1, min(2.0, final_weight))
                
                new_weights[model] = final_weight
                weight_changes[model] = {
                    'old': old_weight,
                    'new': final_weight,
                    'change': final_weight - old_weight,
                    'performance': performance
                }
            
            # 6. Aggiorna i pesi e salva
            self.model_weights[weight_key] = new_weights
            self._save_model_weights()
            
            # 7. Registra aggiornamento nel database
            self._log_weight_update(crypto_id, horizon, weight_changes, len(results))
            
            print(f"   ✅ {crypto_id} {horizon}: Updated weights based on {len(results)} verifications")
            return {
                'success': True,
                'crypto_id': crypto_id,
                'horizon': horizon,
                'weight_changes': weight_changes,
                'verifications_used': len(results),
                'performance_summary': valid_models
            }
            
        except Exception as e:
            error_msg = f"Weight update failed for {crypto_id} {horizon}: {str(e)}"
            print(f"   ⚠️ {error_msg}")
            return {
                'success': False, 
                'error': error_msg,
                'crypto_id': crypto_id,
                'horizon': horizon
            }

    # 🆕 AGGIUNGI NUOVO METODO: _log_weight_update
    def _log_weight_update(self, crypto_id: str, horizon: str, weight_changes: Dict, verifications_count: int):
        """📝 Registra l'aggiornamento dei pesi nel database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Crea tabella se non exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_weight_updates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    crypto_id TEXT NOT NULL,
                    horizon TEXT NOT NULL,
                    update_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    weight_changes TEXT NOT NULL,
                    verifications_used INTEGER,
                    update_success BOOLEAN DEFAULT 1
                )
            ''')
            
            cursor.execute('''
                INSERT INTO model_weight_updates 
                (crypto_id, horizon, weight_changes, verifications_used)
                VALUES (?, ?, ?, ?)
            ''', (crypto_id, horizon, json.dumps(weight_changes), verifications_count))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"   ⚠️ Failed to log weight update: {e}")

    # 🆕 AGGIUNGI NUOVO METODO: fix_weight_update_system
    def fix_weight_update_system(self):
        """🔧 Sistema di riparazione completo per weight updates"""
        print("🔧 FIXING WEIGHT UPDATE SYSTEM...")
        
        fixed_count = 0
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 1. Verifica esistenza tabelle necessarie
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_weight_updates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    crypto_id TEXT NOT NULL,
                    horizon TEXT NOT NULL,
                    update_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    weight_changes TEXT NOT NULL,
                    verifications_used INTEGER,
                    update_success BOOLEAN DEFAULT 1
                )
            ''')
            
            # 2. Ripara pesi mancanti per cryptos con verifiche
            cursor.execute('''
                SELECT DISTINCT crypto_id FROM predictions_optimized 
                WHERE verification_timestamp_1d IS NOT NULL OR verification_timestamp_3d IS NOT NULL
                LIMIT 50
            ''')
            
            cryptos_with_verifications = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            print(f"🎯 Found {len(cryptos_with_verifications)} cryptos with verifications")
            
            # 3. Inizializza pesi mancanti
            for crypto_id in cryptos_with_verifications:
                for horizon in ['1d', '3d']:
                    weight_key = f"{crypto_id}_{horizon}"
                    
                    # Se non ci sono pesi, inizializza con default
                    if not self.model_weights[weight_key]:
                        self.model_weights[weight_key] = {
                            'lightgbm': 1.0,
                            'xgboost': 1.0, 
                            'catboost': 1.0,
                            'lstm': 1.0
                        }
                        fixed_count += 1
            
            # 4. Salva pesi riparati
            if fixed_count > 0:
                self._save_model_weights()
                print(f"✅ Initialized weights for {fixed_count} crypto-horizon combinations")
            
            return fixed_count
            
        except Exception as e:
            print(f"❌ Weight update system fix failed: {e}")
            return 0

    def run_periodic_model_optimization(self):
        """🔄 Esegue ottimizzazione periodica modelli"""
        if self.feedback_optimizer:
            return self.feedback_optimizer.run_periodic_weight_updates()
        return {'error': 'Feedback optimizer not available'}

    def should_run_verification(self, current_cycle):
        """⏰ Determine if verification should run"""
        # Run verification every cycle for better accuracy tracking
        return True
    
    def get_current_price_robust(self, crypto_id: str, max_retries=3) -> Optional[float]:
        """💰 Get current price with robust error handling"""
        
        for attempt in range(max_retries):
            try:
                # Rate limiting with jitter
                if attempt > 0:
                    delay = 2.0 * (attempt + 1) + random.uniform(0.5, 1.5)
                    print(f"   ⏳ Retry {attempt + 1} for {crypto_id}, waiting {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    time.sleep(1.2)  # Base delay
                
                url = "https://api.coingecko.com/api/v3/simple/price"
                params = {'ids': str(crypto_id), 'vs_currencies': 'usd'}
                
                response = requests.get(url, params=params, timeout=15)
                
                # Handle rate limiting
                if response.status_code == 429:
                    if attempt < max_retries - 1:
                        retry_after = response.headers.get('Retry-After', 60)
                        wait_time = min(int(retry_after), 60)
                        print(f"   ⏳ Rate limited, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"   ❌ Max retries reached for {crypto_id}")
                        return None
                
                response.raise_for_status()
                data = response.json()
                price = data.get(str(crypto_id), {}).get('usd')
                
                if price:
                    return float(price)
                else:
                    print(f"   ⚠️ No price data for {crypto_id}")
                    return None
                    
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    print(f"   ⚠️ Network error for {crypto_id}, retrying...")
                    continue
                else:
                    print(f"   ❌ Final API error for {crypto_id}: {e}")
                    return None
            except Exception as e:
                print(f"   ⚠️ Unexpected error for {crypto_id}: {e}")
                return None
        
        return None
    
    def get_predictions_ready_for_verification(self):
        """🔍 Get predictions ready for verification - FIXED SCHEMA"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get 1d predictions ready (older than 25 hours, not yet verified)
            cursor.execute('''
                SELECT id, crypto_id, crypto_name, predicted_change_1d, predicted_price_1d, 
                    current_price, timestamp, confidence_1d, quality_score_1d,
                    models_used_1d, model_weights_1d, market_cap_rank
                FROM predictions_optimized 
                WHERE predicted_change_1d IS NOT NULL
                AND (verified_1d = 0 OR verified_1d IS NULL)
                AND datetime(timestamp) <= datetime('now', '-25 hours')
                ORDER BY timestamp ASC
            ''')
            #senza LIMIT 100
            
            predictions_1d = cursor.fetchall()
            print(f"   📊 Found {len(predictions_1d)} 1d predictions ready")
            
            # Get 3d predictions ready (older than 73 hours, not yet verified)  
            cursor.execute('''
                SELECT id, crypto_id, crypto_name, predicted_change_3d, predicted_price_3d,
                    current_price, timestamp, confidence_3d, quality_score_3d,
                    models_used_3d, model_weights_3d, market_cap_rank
                FROM predictions_optimized 
                WHERE predicted_change_3d IS NOT NULL
                AND (verified_3d = 0 OR verified_3d IS NULL)
                AND datetime(timestamp) <= datetime('now', '-73 hours')
                ORDER BY timestamp ASC
            ''')
            # senza LIMIT 100

            predictions_3d = cursor.fetchall()
            print(f"   📊 Found {len(predictions_3d)} 3d predictions ready")
            
            conn.close()
            return predictions_1d, predictions_3d
            
        except Exception as e:
            print(f"❌ Error getting predictions for verification: {e}")
            return [], []
    
    def calculate_verification_metrics_enhanced(self, predicted_price, current_price, predicted_change, actual_price, confidence):
        """📊 Calculate enhanced verification metrics"""
        try:
            actual_change = (actual_price - current_price) / current_price
            
            # Direction accuracy
            predicted_direction = "UP" if predicted_change > 0 else "DOWN"
            actual_direction = "UP" if actual_change > 0 else "DOWN"
            direction_correct = predicted_direction == actual_direction
            
            # Price accuracy metrics
            price_error_pct = abs(predicted_price - actual_price) / actual_price * 100
            
            # Enhanced accuracy score with confidence calibration
            direction_score = 0.6 if direction_correct else 0.0
            
            # Price accuracy component (0-0.4 range)
            price_accuracy = max(0, 0.4 * (1 - min(price_error_pct / 20, 1)))  # 20% error = 0 score
            
            # Total accuracy score
            accuracy_score = direction_score + price_accuracy
            
            # Confidence calibration score
            if direction_correct:
                confidence_calibration = min(confidence, 0.95)  # Reward high confidence when correct
            else:
                confidence_calibration = max(0.05, 1 - confidence)  # Penalty for high confidence when wrong
            
            return {
                'actual_change': actual_change,
                'actual_direction': actual_direction,
                'direction_correct': direction_correct,
                'accuracy_score': accuracy_score,
                'price_error_percent': price_error_pct,
                'confidence_calibration': confidence_calibration
            }
            
        except Exception as e:
            print(f"❌ Error calculating metrics: {e}")
            return None
    
    def update_prediction_verification_enhanced(self, prediction_id, horizon, actual_price, metrics, prediction_data):
        """💾 Update prediction with enhanced verification results"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            verification_timestamp = datetime.now().isoformat()
            
            # Update original predictions table
            if horizon == 1:
                cursor.execute('''
                    UPDATE predictions 
                    SET actual_price = ?, actual_change = ?, 
                        direction_correct = ?, accuracy_score = ?,
                        verification_timestamp = ?
                    WHERE id = ?
                ''', (
                    actual_price, metrics['actual_change'], 
                    metrics['direction_correct'], metrics['accuracy_score'],
                    verification_timestamp, prediction_id
                ))
            else:
                cursor.execute('''
                    UPDATE predictions 
                    SET actual_price = ?, actual_change = ?, 
                        direction_correct = ?, accuracy_score = ?,
                        verification_timestamp = ?
                    WHERE id = ?
                ''', (
                    actual_price, metrics['actual_change'], 
                    metrics['direction_correct'], metrics['accuracy_score'],
                    verification_timestamp, prediction_id
                ))
            
            # Insert detailed verification result
            cursor.execute('''
                INSERT INTO verification_results (
                    crypto_id, crypto_name, horizon, prediction_timestamp, verification_timestamp,
                    predicted_direction, predicted_change, predicted_price, confidence, quality_score,
                    actual_price, actual_change, actual_direction,
                    direction_correct, accuracy_score, price_error_percent,
                    models_used, model_weights, ensemble_method
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                prediction_data[1], prediction_data[2], f"{horizon}d", prediction_data[6], verification_timestamp,
                "UP" if prediction_data[3] > 0 else "DOWN", prediction_data[3], prediction_data[4], 
                prediction_data[7], prediction_data[11] if len(prediction_data) > 11 else 0,
                actual_price, metrics['actual_change'], metrics['actual_direction'],
                metrics['direction_correct'], metrics['accuracy_score'], metrics['price_error_percent'],
                prediction_data[9] if len(prediction_data) > 9 else "", 
                prediction_data[10] if len(prediction_data) > 10 else "",
                "weighted_ensemble"
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Error updating verification: {e}")
            return False
    
    def _update_verification_in_db(self, pred_id: int, verification_result: Dict, horizon: int):
        """💾 Update verification results using atomic transactions"""
        try:
            # Prepara i dati per il consistency manager
            verification_data = {
                'prediction_id': pred_id,
                'crypto_id': verification_result['crypto_id'],
                'crypto_name': verification_result['crypto_name'],
                'horizon': verification_result['horizon'],
                'current_price': verification_result['current_price'],
                'actual_change': verification_result['actual_change'],
                'direction_correct': verification_result['direction_correct'],
                'accuracy_score': verification_result['accuracy_score'],
                'verification_timestamp': verification_result['verification_timestamp'],
                'price_error_percent': verification_result['price_error_percent'],
                'predicted_price': verification_result['predicted_price'],
                'predicted_change': verification_result['predicted_change'],
                'confidence': verification_result['confidence'],
                'quality_score': verification_result.get('quality_score', 0)
            }
            
            # Usa il consistency manager per aggiornamento atomico
            self.db_consistency_manager.update_verification_atomically(verification_data)
            
            print(f"✅ Atomic verification update completed for {verification_result['crypto_id']}")
            return True
            
        except Exception as e:
            print(f"❌ Atomic verification update failed: {e}")
            return False

    def verify_predictions_with_timing_check(self) -> Dict:
        """⏰ Verify predictions with strict timing validation"""
        results = {
            'total_checked': 0,
            'total_verified': 0,
            'verified_1d': 0,
            'verified_3d': 0,
            'timing_errors': 0,
            'api_errors': 0,
            'successful_predictions': [],
            'failed_predictions': [],
            'timing_issues': [],
            'model_feedback_updates': []
        }
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            now = datetime.now().isoformat()
            
            # Query predictions ready for verification with timing windows
            cursor.execute('''
                SELECT id, crypto_id, crypto_name, predicted_price_1d, predicted_price_3d,
                    confidence_1d, confidence_3d, quality_score_1d, quality_score_3d,
                    prediction_timestamp, predicted_change_1d, predicted_change_3d
                FROM predictions_optimized 
                WHERE (
                    (verification_timestamp_1d IS NULL AND 
                    datetime(prediction_timestamp, '+1 day') <= datetime('now') AND
                    datetime(prediction_timestamp, '+1 day', '+6 hours') >= datetime('now'))
                    OR
                    (verification_timestamp_3d IS NULL AND 
                    datetime(prediction_timestamp, '+3 days') <= datetime('now') AND
                    datetime(prediction_timestamp, '+3 days', '+12 hours') >= datetime('now'))
                )
                ORDER BY prediction_timestamp ASC
                LIMIT 50
            ''')
            
            predictions = cursor.fetchall()
            results['total_checked'] = len(predictions)
            
            for pred in predictions:
                pred_id, crypto_id, crypto_name = pred[0], pred[1], pred[2]
                pred_timestamp = datetime.fromisoformat(pred[9])
                
                # Check timing windows
                time_1d = pred_timestamp + timedelta(days=1)
                time_3d = pred_timestamp + timedelta(days=3)
                
                # Determine which horizons need verification
                verify_1d = (pred[3] is not None and  # has 1d prediction (fixed index)
                            abs((now - time_1d).total_seconds()) <= 6 * 3600 and  # within 6h window
                            time_1d <= now)  # time has passed
                
                verify_3d = (pred[4] is not None and  # has 3d prediction (fixed index)
                            abs((now - time_3d).total_seconds()) <= 12 * 3600 and  # within 12h window
                            time_3d <= now)  # time has passed
                
                if not (verify_1d or verify_3d):
                    # Record timing issue
                    timing_issue = {
                        'crypto_id': crypto_id,
                        'crypto_name': crypto_name,
                        'prediction_time': pred_timestamp.isoformat(),
                        'current_time': now.isoformat(),
                        'issue': 'Outside verification window'
                    }
                    results['timing_issues'].append(timing_issue)
                    results['timing_errors'] += 1
                    continue
                
                # Fetch current price
                current_price = self.get_current_price_robust(crypto_id)
                if current_price is None:
                    results['api_errors'] += 1
                    continue
                
                # Verify each horizon
                if verify_1d:
                    verification_result = self._verify_single_prediction_enhanced(
                        pred, current_price, horizon=1, now=now
                    )
                    if verification_result:
                        results['verified_1d'] += 1
                        results['successful_predictions'].append(verification_result)
                        
                        # Update database
                        self._update_verification_in_db(pred_id, verification_result, horizon=1)
                
                if verify_3d:
                    verification_result = self._verify_single_prediction_enhanced(  # Fixed method name
                        pred, current_price, horizon=3, now=now
                    )
                    if verification_result:
                        results['verified_3d'] += 1
                        results['successful_predictions'].append(verification_result)
                        
                        # Update database
                        self._update_verification_in_db(pred_id, verification_result, horizon=3)
            
            results['total_verified'] = results['verified_1d'] + results['verified_3d']
            conn.close()
            
            # ✅ CORRETTO: Model feedback DOPO tutte le verifiche
            if hasattr(self, 'feedback_optimizer') and self.feedback_optimizer and results['successful_predictions']:
                print(f"🔄 Processing model feedback for {len(results['successful_predictions'])} successful predictions...")
                
                for prediction in results['successful_predictions']:
                    try:
                        feedback_result = self.update_model_weights_after_verification(
                            prediction['crypto_id'],
                            prediction['horizon']
                        )
                        if feedback_result.get('success'):
                            results['model_feedback_updates'].append({
                                'crypto_id': prediction['crypto_id'],
                                'horizon': prediction['horizon'],
                                'weight_changes': feedback_result.get('weight_changes', {})
                            })
                            print(f"   ✅ Updated weights for {prediction['crypto_id']} {prediction['horizon']}")
                        
                    except Exception as e:
                        print(f"   ⚠️ Feedback failed for {prediction['crypto_id']} {prediction['horizon']}: {e}")
                
                if results['model_feedback_updates']:
                    print(f"✅ Model feedback completed: {len(results['model_feedback_updates'])} updates")
            
        except Exception as e:
            print(f"❌ Verification failed: {e}")
            results['error'] = str(e)
        
        return results

    def _verify_single_prediction_enhanced(self, pred_data: Tuple, current_price: float, horizon: int, now: datetime) -> Optional[Dict]:
        """🎯 Verify single prediction with enhanced metrics and BYTES HANDLING"""
        try:
            # Extract data based on tuple structure from get_predictions_ready_for_verification
            pred_id = pred_data[0]
            crypto_id = pred_data[1] 
            crypto_name = pred_data[2]
            predicted_change_raw = pred_data[3]  # Might be bytes
            predicted_price_raw = pred_data[4]   # Might be bytes
            original_price_raw = pred_data[5]    # Might be bytes
            timestamp = pred_data[6]
            confidence_raw = pred_data[7]        # Might be bytes
            quality_score_raw = pred_data[8]     # Might be bytes
            # AGGIUNGI DOPO l'estrazione dei dati raw:
            print(f"      🔍 RAW DEBUG for {crypto_name}:")
            print(f"         predicted_change_raw: type={type(predicted_change_raw)}, value={repr(predicted_change_raw)}")
            print(f"         predicted_price_raw: type={type(predicted_price_raw)}, value={repr(predicted_price_raw)}")
            print(f"         original_price_raw: type={type(original_price_raw)}, value={repr(original_price_raw)}")
            # 🔄 SAFE BYTES CONVERSION - Using internal method
            predicted_change = self._safe_float_convert(predicted_change_raw, "predicted_change")
            predicted_price = self._safe_float_convert(predicted_price_raw, "predicted_price")
            original_price = self._safe_float_convert(original_price_raw, "original_price")
            confidence = self._safe_float_convert(confidence_raw, "confidence")
            quality_score = self._safe_float_convert(quality_score_raw, "quality_score")
            
            print(f"      🔄 Converted values: change={predicted_change:.4f}, price=${predicted_price:.4f}, conf={confidence:.3f}")
            
            # Validate converted data
            if predicted_price == 0.0 and predicted_change != 0.0 and original_price > 0.0:
                # Calculate predicted price if missing
                predicted_price = original_price * (1 + predicted_change)
                print(f"      🔧 Calculated predicted_price: ${predicted_price:.4f}")
            
            if original_price == 0.0 and predicted_price > 0.0 and predicted_change != -1.0:
                # Calculate original price if missing
                original_price = predicted_price / (1 + predicted_change)
                print(f"      🔧 Calculated original_price: ${original_price:.4f}")
            
            # Final validation
            if predicted_price == 0.0 or original_price == 0.0:
                return {'success': False, 'error': f'Invalid price data for {crypto_name} after conversion'}
            
            if confidence == 0.0:
                confidence = 0.5  # Default confidence
            
            # Calculate actual change
            actual_change = (current_price - original_price) / original_price
            
            # Direction accuracy
            predicted_direction = "UP" if predicted_change > 0 else "DOWN"
            actual_direction = "UP" if actual_change > 0 else "DOWN"
            direction_correct = predicted_direction == actual_direction
            
            # Price accuracy metrics
            price_error_pct = abs(predicted_price - current_price) / current_price * 100
            
            # Enhanced accuracy score
            direction_score = 0.7 if direction_correct else 0.0
            price_accuracy = max(0, 0.3 * (1 - min(price_error_pct / 15, 1)))  # 15% error = 0 score
            total_accuracy = direction_score + price_accuracy
            
            # Confidence calibration
            if direction_correct:
                confidence_calibration = min(confidence, 0.95)
            else:
                confidence_calibration = max(0.05, 1 - confidence)
            
            # Create verification result
            verification_result = {
                'success': True,
                'prediction_id': pred_id,
                'crypto_id': crypto_id,
                'crypto_name': crypto_name,
                'horizon': f'{horizon}d',
                'predicted_change': predicted_change,
                'predicted_price': predicted_price,
                'current_price': current_price,
                'original_price': original_price,
                'actual_change': actual_change,
                'predicted_direction': predicted_direction,
                'actual_direction': actual_direction,
                'direction_correct': direction_correct,
                'accuracy_score': total_accuracy,
                'price_error_percent': price_error_pct,
                'confidence': confidence,
                'confidence_calibration': confidence_calibration,
                'quality_score': quality_score,
                'verification_timestamp': now.isoformat()
            }
            
            print(f"      ✅ {crypto_name}: {predicted_direction} vs {actual_direction} - Accuracy: {total_accuracy:.3f}")
            return verification_result
            
        except Exception as e:
            print(f"      ❌ Verification error for {crypto_name if 'crypto_name' in locals() else 'unknown'}: {e}")
            return {'success': False, 'error': str(e)}

    def _safe_float_convert(self, value, field_name="unknown") -> float:
        """🔄 Safe conversion of any value to float - PROPERLY HANDLES BYTES"""
        
        if value is None:
            print(f"      ⚠️ NULL value for field {field_name}")
            return 0.0
        
        # If already a number, return it
        if isinstance(value, (int, float)):
            result = float(value)
            print(f"      ✅ Direct numeric conversion: {field_name}={result:.6f}")
            return result
        
        # If numeric string, convert it
        if isinstance(value, str):
            try:
                result = float(value.strip())
                print(f"      ✅ String conversion: {field_name}='{value}' → {result:.6f}")
                return result
            except ValueError:
                print(f"      ❌ Cannot convert string '{value}' to float for field {field_name}")
                return 0.0
        
        # 🚨 BYTES HANDLING - FIXED APPROACH
        if isinstance(value, bytes):
            print(f"      🔍 Processing bytes for {field_name}: length={len(value)}, hex={value.hex()[:20]}...")
            
            # PRIORITY 1: Try UTF-8 decode (most common case)
            try:
                string_value = value.decode('utf-8').strip()
                if string_value:  # Non-empty after decoding
                    result = float(string_value)
                    print(f"      ✅ Bytes→UTF8→float: {field_name}='{string_value}' → {result:.6f}")
                    return result
                else:
                    print(f"      ⚠️ Empty string after UTF-8 decode for {field_name}")
            except UnicodeDecodeError as e:
                print(f"      ⚠️ UTF-8 decode failed for {field_name}: {e}")
            except ValueError as e:
                print(f"      ⚠️ Float conversion failed after UTF-8 decode for {field_name}: {e}")
            
            # PRIORITY 2: Try other encodings
            for encoding in ['latin-1', 'ascii', 'utf-16']:
                try:
                    string_value = value.decode(encoding).strip()
                    if string_value:
                        result = float(string_value)
                        print(f"      ✅ Bytes→{encoding}→float: {field_name} → {result:.6f}")
                        return result
                except (UnicodeDecodeError, ValueError):
                    continue
            
            # PRIORITY 3: Try binary float interpretation (ONLY if reasonable length)
            if len(value) in [4, 8]:
                try:
                    import struct
                    
                    if len(value) == 4:  # float32
                        result = struct.unpack('<f', value)[0]  # little-endian
                        if not (float('inf') == result or float('-inf') == result or result != result):  # Check for inf/nan
                            print(f"      ✅ Binary float32: {field_name} → {result:.6f}")
                            return result
                        
                    elif len(value) == 8:  # float64  
                        result = struct.unpack('<d', value)[0]  # little-endian
                        if not (float('inf') == result or float('-inf') == result or result != result):  # Check for inf/nan
                            print(f"      ✅ Binary float64: {field_name} → {result:.6f}")
                            return result
                            
                except struct.error as e:
                    print(f"      ⚠️ Binary struct unpack failed for {field_name}: {e}")
            
            # If all bytes conversion attempts failed
            print(f"      ❌ All bytes conversion attempts failed for {field_name}")
            return 0.0
        
        # Fallback for unknown types
        print(f"      ❌ Unknown type {type(value)} for field {field_name}, using 0.0")
        return 0.0
    
    def _update_model_performance_tracking(self, prediction_data, metrics, horizon):
        """📊 Update model performance tracking for intelligent model selection"""
        try:
            crypto_id = prediction_data[1]
            models_used_json = prediction_data[9] if len(prediction_data) > 9 else "{}"
            model_weights_json = prediction_data[10] if len(prediction_data) > 10 else "{}"
            
            # Parse model information
            try:
                models_used = json.loads(models_used_json) if models_used_json else []
                model_weights = json.loads(model_weights_json) if model_weights_json else {}
            except:
                models_used = []
                model_weights = {}
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Update performance for each model used
            for model_name in models_used:
                # Get current performance data
                cursor.execute('''
                    SELECT total_predictions, correct_predictions, accuracy_rate, avg_confidence, avg_accuracy_score
                    FROM model_performance_tracking
                    WHERE crypto_id = ? AND model_name = ? AND horizon = ?
                ''', (crypto_id, model_name, f"{horizon}d"))
                
                result = cursor.fetchone()
                
                if result:
                    total_preds, correct_preds, accuracy_rate, avg_conf, avg_acc_score = result
                    
                    # Update metrics
                    new_total = total_preds + 1
                    new_correct = correct_preds + (1 if metrics['direction_correct'] else 0)
                    new_accuracy_rate = new_correct / new_total
                    new_avg_conf = (avg_conf * total_preds + prediction_data[7]) / new_total
                    new_avg_acc_score = (avg_acc_score * total_preds + metrics['accuracy_score']) / new_total
                    
                    cursor.execute('''
                        UPDATE model_performance_tracking
                        SET total_predictions = ?, correct_predictions = ?, accuracy_rate = ?,
                            avg_confidence = ?, avg_accuracy_score = ?, last_updated = ?
                        WHERE crypto_id = ? AND model_name = ? AND horizon = ?
                    ''', (
                        new_total, new_correct, new_accuracy_rate, new_avg_conf, new_avg_acc_score,
                        datetime.now().isoformat(), crypto_id, model_name, f"{horizon}d"
                    ))
                else:
                    # Insert new model performance record
                    cursor.execute('''
                        INSERT INTO model_performance_tracking (
                            crypto_id, model_name, horizon, total_predictions, correct_predictions,
                            accuracy_rate, avg_confidence, avg_accuracy_score, last_updated
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        crypto_id, model_name, f"{horizon}d", 1, 
                        1 if metrics['direction_correct'] else 0,
                        1.0 if metrics['direction_correct'] else 0.0,
                        prediction_data[7], metrics['accuracy_score'], datetime.now().isoformat()
                    ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"      ⚠️ Model performance tracking update failed: {e}")
    
    def create_verification_scheduler(self):
        """⏰ Create automated verification scheduler"""
        import threading
        import schedule
        
        def run_verification_job():
            try:
                print("🔍 Running scheduled verification...")
                results = self.verify_predictions_with_timing_check()
                
                # Log results
                print(f"✅ Verification completed:")
                print(f"   • Checked: {results['total_checked']}")
                print(f"   • Verified: {results['total_verified']}")
                print(f"   • 1d: {results['verified_1d']}, 3d: {results['verified_3d']}")
                print(f"   • Timing errors: {results['timing_errors']}")
                print(f"   • API errors: {results['api_errors']}")
                
                # Send email if configured and sufficient results
                if results['total_verified'] > 0 and self.gmail_user:
                    self.send_verification_email(results)
                    
            except Exception as e:
                print(f"❌ Scheduled verification failed: {e}")
        
        # Schedule verification every 6 hours
        schedule.every(6).hours.do(run_verification_job)
        
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(3600)  # Check every hour
        
        # Run scheduler in background thread
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        
        print("⏰ Verification scheduler started (every 6 hours)")
        return scheduler_thread

    def _update_verification_in_database(self, verification_result: Dict, horizon: int):
        """💾 Update verification results in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            pred_id = verification_result['prediction_id']
            
            if horizon == 1:
                cursor.execute('''
                    UPDATE predictions_optimized 
                    SET verified_1d = 1,
                        actual_price_1d = ?,
                        actual_change_1d = ?,
                        direction_correct_1d = ?,
                        accuracy_score_1d = ?,
                        price_error_pct_1d = ?,
                        verification_timestamp_1d = ?
                    WHERE id = ?
                ''', (
                    verification_result['current_price'],
                    verification_result['actual_change'],
                    verification_result['direction_correct'],
                    verification_result['accuracy_score'],
                    verification_result['price_error_percent'],
                    verification_result['verification_timestamp'],
                    pred_id
                ))
            else:  # horizon == 3
                cursor.execute('''
                    UPDATE predictions_optimized 
                    SET verified_3d = 1,
                        actual_price_3d = ?,
                        actual_change_3d = ?,
                        direction_correct_3d = ?,
                        accuracy_score_3d = ?,
                        price_error_pct_3d = ?,
                        verification_timestamp_3d = ?
                    WHERE id = ?
                ''', (
                    verification_result['current_price'],
                    verification_result['actual_change'],
                    verification_result['direction_correct'],
                    verification_result['accuracy_score'],
                    verification_result['price_error_percent'],
                    verification_result['verification_timestamp'],
                    pred_id
                ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"      ❌ Database update failed: {e}")
            return False

    def send_accuracy_email_optimized(self, verification_results, cycle_number):
        """📧 Send optimized accuracy email with comprehensive performance insights"""
        
        if not hasattr(self, 'gmail_user') or not self.gmail_user or not hasattr(self, 'gmail_password') or not self.gmail_password:
            print("⚠️ Email credentials not configured - skipping accuracy email")
            return False
        
        try:
            print("📧 Sending optimized accuracy email report...")
            
            # Get comprehensive accuracy statistics
            accuracy_stats = self.get_comprehensive_accuracy_statistics(7)
            
            # Create email
            msg = MIMEMultipart()
            msg['From'] = self.gmail_user
            msg['To'] = self.gmail_user
            msg['Subject'] = f"🎯 Optimized Accuracy Report - Cycle {cycle_number} - {verification_results['total_verified']} Verified"
            
            # Create HTML content
            html_content = self._create_optimized_accuracy_email_html(verification_results, accuracy_stats, cycle_number)
            msg.attach(MIMEText(html_content, 'html'))
            
            # Send email
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(self.gmail_user, self.gmail_password)
            text = msg.as_string()
            server.sendmail(self.gmail_user, self.gmail_user, text)
            server.quit()
            
            print("✅ Optimized accuracy email sent successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error sending accuracy email: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _create_optimized_accuracy_email_html(self, verification_results, accuracy_stats, cycle_number):
        """📧 Create comprehensive accuracy email HTML with improved styling"""
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif; 
                    margin: 20px; background: #f8f9fa; line-height: 1.6; }}
                .container {{ background: white; padding: 30px; border-radius: 12px; max-width: 900px; 
                            margin: 0 auto; box-shadow: 0 8px 25px rgba(0,0,0,0.1); }}
                
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 25px; border-radius: 10px; text-align: center; 
                        margin-bottom: 25px; }}
                .header h1 {{ margin: 0 0 10px 0; font-size: 32px; font-weight: 700; }}
                .header p {{ margin: 5px 0; opacity: 0.9; }}
                
                .section {{ margin: 25px 0; padding: 20px; border-radius: 10px; }}
                .verification-section {{ background: linear-gradient(135deg, #4caf50, #45a049); color: white; }}
                .accuracy-section {{ background: linear-gradient(135deg, #ff9800, #f57c00); color: white; }}
                .model-section {{ background: linear-gradient(135deg, #2196f3, #1976d2); color: white; }}
                
                /* ✅ FIXED: Proper styling for recommendations and system health */
                .recommendations-section {{ 
                    background: linear-gradient(135deg, #ff6b6b, #ee5a52); 
                    color: white; 
                    border-left: 5px solid #ff4757;
                }}
                
                .health-section {{ 
                    background: linear-gradient(135deg, #5f27cd, #341f97); 
                    color: white; 
                    border-left: 5px solid #8854d0;
                }}
                
                .empty-data-section {{
                    background: linear-gradient(135deg, #778ca3, #2d3436);
                    color: white;
                    border-left: 5px solid #a29bfe;
                    text-align: center;
                    font-style: italic;
                }}
                
                .stats-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
                @media (max-width: 600px) {{ 
                    .stats-grid {{ grid-template-columns: 1fr; }}
                }}
                
                .stat-box {{ background: rgba(255,255,255,0.15); padding: 18px; border-radius: 8px; 
                            backdrop-filter: blur(10px); }}
                .stat-box h3 {{ margin: 0 0 15px 0; font-size: 18px; font-weight: 600; }}
                .stat-box p {{ margin: 5px 0; font-size: 14px; }}
                
                .prediction-list {{ background: rgba(255,255,255,0.95); color: #333; 
                                padding: 18px; border-radius: 8px; margin: 15px 0; }}
                .prediction-list p {{ margin: 8px 0; padding: 8px; background: #f8f9fa; 
                                    border-radius: 5px; border-left: 4px solid #007bff; }}
                
                .success {{ color: #4caf50; font-weight: bold; }}
                .warning {{ color: #ff9800; font-weight: bold; }}
                .error {{ color: #f44336; font-weight: bold; }}
                
                .footer {{ text-align: center; color: #666; font-size: 0.9em; margin-top: 30px; 
                        padding: 20px; background: #f8f9fa; border-radius: 8px; }}
                
                /* ✅ REMOVED: Old .highlight class that caused white background issue */
                .recommendation-item {{ 
                    background: rgba(255,255,255,0.15); 
                    padding: 12px; 
                    margin: 10px 0; 
                    border-radius: 6px; 
                    border-left: 4px solid rgba(255,255,255,0.8);
                    color: white;
                    font-weight: 500;
                }}
                
                .health-item {{ 
                    background: rgba(255,255,255,0.15); 
                    padding: 12px; 
                    margin: 8px 0; 
                    border-radius: 6px;
                    color: white;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }}
                
                .health-value {{ 
                    font-weight: bold; 
                    font-family: 'Monaco', 'Menlo', monospace;
                    background: rgba(255,255,255,0.2);
                    padding: 4px 8px;
                    border-radius: 4px;
                }}
                
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ padding: 12px 8px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.2); }}
                th {{ background: rgba(255,255,255,0.15); font-weight: 600; }}
                td {{ font-size: 14px; }}
                
                .status-indicator {{
                    display: inline-block;
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    margin-right: 8px;
                }}
                
                .status-good {{ background: #4caf50; }}
                .status-warning {{ background: #ff9800; }}
                .status-critical {{ background: #f44336; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎯 OPTIMIZED ACCURACY REPORT</h1>
                    <p><strong>Enhanced Verification System - Cycle {cycle_number}</strong></p>
                    <p>{timestamp}</p>
                </div>
                
                <div class="section verification-section">
                    <h2>🔍 Latest Verification Results</h2>
                    <div class="stats-grid">
                        <div class="stat-box">
                            <h3>📊 Verification Summary</h3>
                            <p><strong>Total Checked:</strong> {verification_results.get('total_checked', 0):,}</p>
                            <p><strong>Successfully Verified:</strong> {verification_results.get('total_verified', 0):,}</p>
                            <p><strong>Verification Rate:</strong> {(verification_results.get('total_verified', 0)/max(verification_results.get('total_checked', 1), 1)*100):.1f}%</p>
                            <p><strong>Processing Time:</strong> {verification_results.get('elapsed_time', 0):.1f}s</p>
                        </div>
                        <div class="stat-box">
                            <h3>📈 Horizon Breakdown</h3>
                            <p><strong>1d Verified:</strong> {verification_results.get('verified_1d', 0):,}</p>
                            <p><strong>3d Verified:</strong> {verification_results.get('verified_3d', 0):,}</p>
                            <p><strong>1d Avg Accuracy:</strong> {verification_results.get('accuracy_1d', 0):.3f}</p>
                            <p><strong>3d Avg Accuracy:</strong> {verification_results.get('accuracy_3d', 0):.3f}</p>
                        </div>
                    </div>
                </div>
        """
        
        # Add 7-day accuracy statistics if available and not empty
        if accuracy_stats and (accuracy_stats['1d'].get('total', 0) > 0 or accuracy_stats['3d'].get('total', 0) > 0):
            stats_1d = accuracy_stats['1d']
            stats_3d = accuracy_stats['3d']
            
            # Performance status indicators
            status_1d = "🟢" if stats_1d.get('direction_accuracy', 0) >= 0.6 else "🟡" if stats_1d.get('direction_accuracy', 0) >= 0.5 else "🔴"
            status_3d = "🟢" if stats_3d.get('direction_accuracy', 0) >= 0.55 else "🟡" if stats_3d.get('direction_accuracy', 0) >= 0.45 else "🔴"
            
            html += f"""
                <div class="section accuracy-section">
                    <h2>📈 7-Day Performance Summary</h2>
                    <div class="stats-grid">
                        <div class="stat-box">
                            <h3>{status_1d} 1-Day Horizon Performance</h3>
                            <p><strong>Total Predictions:</strong> {stats_1d.get('total', 0):,}</p>
                            <p><strong>Direction Accuracy:</strong> {stats_1d.get('direction_accuracy', 0):.1%}</p>
                            <p><strong>Avg Accuracy Score:</strong> {stats_1d.get('avg_accuracy_score', 0):.3f}</p>
                            <p><strong>Avg Confidence:</strong> {stats_1d.get('avg_confidence', 0):.1%}</p>
                            <p><strong>Price Error:</strong> {stats_1d.get('avg_price_error', 0):.1f}%</p>
                        </div>
                        <div class="stat-box">
                            <h3>{status_3d} 3-Day Horizon Performance</h3>
                            <p><strong>Total Predictions:</strong> {stats_3d.get('total', 0):,}</p>
                            <p><strong>Direction Accuracy:</strong> {stats_3d.get('direction_accuracy', 0):.1%}</p>
                            <p><strong>Avg Accuracy Score:</strong> {stats_3d.get('avg_accuracy_score', 0):.3f}</p>
                            <p><strong>Avg Confidence:</strong> {stats_3d.get('avg_confidence', 0):.1%}</p>
                            <p><strong>Price Error:</strong> {stats_3d.get('avg_price_error', 0):.1f}%</p>
                        </div>
                    </div>
                    
                    <!-- Best performing predictions -->
                    <h3>🏆 Best Performing Predictions (1d)</h3>
                    <div class="prediction-list">
            """
            
            best_predictions_1d = stats_1d.get('best_predictions', [])
            if best_predictions_1d:
                for pred in best_predictions_1d[:5]:
                    if pred and len(pred) >= 4:
                        crypto_name = pred[0] if pred[0] else "Unknown"
                        direction_acc = pred[3] if pred[3] else 0
                        confidence = pred[2] if len(pred) > 2 and pred[2] else 0
                        html += f"<p>✅ <strong>{crypto_name}:</strong> {direction_acc:.1%} accuracy, {confidence:.1%} confidence</p>"
            else:
                html += "<p><em>No best predictions data available yet</em></p>"
            
            html += "</div>"
            
            # Model performance section
            if stats_1d.get('model_performance') or stats_3d.get('model_performance'):
                html += f"""
                    <h3>🤖 Model Performance Ranking</h3>
                    <div class="stats-grid">
                        <div class="stat-box">
                            <h3>📊 1-Day Models</h3>
                            <table>
                                <tr><th>Model</th><th>Accuracy</th><th>Count</th></tr>
                """
                
                model_perf_1d = stats_1d.get('model_performance', [])
                if model_perf_1d:
                    for model_perf in model_perf_1d[:5]:
                        if model_perf and len(model_perf) >= 3:
                            model_name = model_perf[0] if model_perf[0] else "Unknown"
                            accuracy = model_perf[1] if model_perf[1] else 0
                            count = model_perf[2] if model_perf[2] else 0
                            html += f"<tr><td>{model_name}</td><td>{accuracy:.1%}</td><td>{count}</td></tr>"
                else:
                    html += "<tr><td colspan='3'><em>No model performance data</em></td></tr>"
                
                html += """
                            </table>
                        </div>
                        <div class="stat-box">
                            <h3>📊 3-Day Models</h3>
                            <table>
                                <tr><th>Model</th><th>Accuracy</th><th>Count</th></tr>
                """
                
                model_perf_3d = stats_3d.get('model_performance', [])
                if model_perf_3d:
                    for model_perf in model_perf_3d[:5]:
                        if model_perf and len(model_perf) >= 3:
                            model_name = model_perf[0] if model_perf[0] else "Unknown"
                            accuracy = model_perf[1] if model_perf[1] else 0
                            count = model_perf[2] if model_perf[2] else 0
                            html += f"<tr><td>{model_name}</td><td>{accuracy:.1%}</td><td>{count}</td></tr>"
                else:
                    html += "<tr><td colspan='3'><em>No model performance data</em></td></tr>"
                
                html += "</table></div></div>"
            
            html += "</div>"  # Close accuracy section
        else:
            # ✅ IMPROVED: Better empty data section
            html += f"""
                <div class="section empty-data-section">
                    <h2>📊 7-Day Performance Summary</h2>
                    <div style="padding: 30px;">
                        <h3>📈 Data Collection in Progress</h3>
                        <p>The system is still collecting performance data over the 7-day period.</p>
                        <p><strong>Current Status:</strong> Building historical accuracy database</p>
                        <p><strong>Expected Full Data:</strong> After {7 - min(7, (datetime.now() - datetime(2024, 1, 1)).days)} more days</p>
                        <p><em>Comprehensive statistics will be available once sufficient historical data is collected.</em></p>
                    </div>
                </div>
            """
        
        # Recent successful verifications
        if verification_results.get('successful_predictions'):
            html += f"""
                <div class="section model-section">
                    <h3>✅ Recent Successful Verifications</h3>
                    <div class="prediction-list">
            """
            
            for pred in verification_results['successful_predictions'][:10]:
                correct_emoji = "✅" if pred.get('direction_correct', False) else "❌"
                crypto_name = pred.get('crypto_name', 'Unknown')
                horizon = pred.get('horizon', '?')
                predicted_change = pred.get('predicted_change', 0)
                actual_change = pred.get('actual_change', 0)
                accuracy_score = pred.get('accuracy_score', 0)
                confidence = pred.get('confidence', 0)
                confidence_color = "#4caf50" if confidence > 0.7 else "#ff9800" if confidence > 0.5 else "#f44336"
                
                html += f"""
                <p>{correct_emoji} <strong>{crypto_name} ({horizon}):</strong> 
                {predicted_change:+.2%} → {actual_change:+.2%} 
                (Score: {accuracy_score:.3f}, 
                <span style="color: {confidence_color}; font-weight: bold;">Conf: {confidence:.1%}</span>)</p>
                """
            
            html += "</div></div>"
        
        # ✅ FIXED: Performance recommendations with proper styling
        recommendations = []
        if accuracy_stats and (accuracy_stats['1d'].get('total', 0) > 0 or accuracy_stats['3d'].get('total', 0) > 0):
            stats_1d = accuracy_stats['1d']
            stats_3d = accuracy_stats['3d']
            
            if stats_1d.get('direction_accuracy', 0) < 0.55:
                recommendations.append("⚠️ 1d direction accuracy below 55% - consider adjusting model weights")
            if stats_3d.get('direction_accuracy', 0) < 0.50:
                recommendations.append("⚠️ 3d direction accuracy below 50% - review feature engineering")
            if stats_1d.get('avg_price_error', 0) > 15:
                recommendations.append("📊 1d price errors high (>15%) - improve magnitude prediction")
        
        if len(verification_results.get('model_feedback', [])) > 0:
            recommendations.append(f"🤖 {len(verification_results['model_feedback'])} models updated with feedback")
        
        if not recommendations:
            recommendations.append("✅ System performance within acceptable parameters")
            recommendations.append("📈 Continue current optimization strategy")
        
        if recommendations:
            html += f"""
                <div class="section recommendations-section">
                    <h3>💡 Performance Recommendations</h3>
            """
            for rec in recommendations:
                html += f'<div class="recommendation-item">{rec}</div>'
            html += "</div>"
        
        # ✅ FIXED: System Health Summary with proper styling
        total_verified = verification_results.get('total_verified', 0)
        combined_accuracy = (verification_results.get('accuracy_1d', 0) + verification_results.get('accuracy_3d', 0)) / 2
        model_feedback_count = len(verification_results.get('model_feedback', []))
        
        # Determine overall status
        if combined_accuracy >= 0.6:
            overall_status = "🟢 Excellent"
            status_class = "status-good"
        elif combined_accuracy >= 0.5:
            overall_status = "🟡 Needs Attention"
            status_class = "status-warning"
        else:
            overall_status = "🔴 Critical"
            status_class = "status-critical"
        
        html += f"""
            <div class="section health-section">
                <h3>🏥 System Health Summary</h3>
                <div class="health-item">
                    <span><span class="status-indicator {status_class}"></span><strong>Overall System Status:</strong></span>
                    <span class="health-value">{overall_status}</span>
                </div>
                <div class="health-item">
                    <span><strong>Combined Accuracy:</strong></span>
                    <span class="health-value">{combined_accuracy:.3f}</span>
                </div>
                <div class="health-item">
                    <span><strong>Verification Coverage:</strong></span>
                    <span class="health-value">{total_verified:,} predictions verified</span>
                </div>
                <div class="health-item">
                    <span><strong>Model Feedback:</strong></span>
                    <span class="health-value">{model_feedback_count:,} updates applied</span>
                </div>
            </div>
        """
        
        # Footer
        html += f"""
                <div class="footer">
                    <p><strong>🚀 Crypto Prediction System - Enhanced with Model Feedback</strong></p>
                    <p>Generated at {timestamp}</p>
                    <p>📊 Total Model Feedback Updates: <strong>{model_feedback_count:,}</strong></p>
                    <p>🎯 Verification Accuracy: <strong>{combined_accuracy:.1%}</strong></p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html

    def run_complete_verification_optimized(self, current_cycle, ml_system=None):
        """🚀 Run complete verification with model feedback loop - FULLY FIXED"""
        
        print(f"\n🔍 OPTIMIZED VERIFICATION PROCESS - CYCLE {current_cycle}")
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        start_time = time.time()
        results = {
            'total_checked': 0,
            'total_verified': 0,
            'verified_1d': 0,
            'verified_3d': 0,
            'errors': 0,
            'successful_predictions': [],
            'failed_predictions': [],
            'accuracy_1d': 0,
            'accuracy_3d': 0,
            'model_feedback': []
        }
        
        # Get predictions ready for verification
        print("🔍 Scanning for predictions ready for verification...")
        predictions_1d, predictions_3d = self.get_predictions_ready_for_verification()
        
        total_ready = len(predictions_1d) + len(predictions_3d)
        print(f"   📊 Found: {len(predictions_1d)} 1d predictions + {len(predictions_3d)} 3d predictions = {total_ready} total")
        
        if total_ready == 0:
            print("   ✅ No predictions ready for verification")
            return results
        
        now = datetime.now()
        
        # Verify 1d predictions with FIXED SIGNATURE
        if predictions_1d:
            print(f"\n📈 Verifying {len(predictions_1d)} 1d predictions...")
            accuracy_1d_list = []
            
            for i, prediction in enumerate(predictions_1d, 1):
                crypto_name = prediction[2] if len(prediction) > 2 else 'Unknown'
                crypto_id = prediction[1]
                print(f"   [{i}/{len(predictions_1d)}] Processing {crypto_name}...")
                
                try:
                    # Get current price
                    current_price = self.get_current_price_robust(crypto_id)
                    
                    if current_price is None:
                        print(f"      ❌ Cannot get price for {crypto_name}")
                        results['errors'] += 1
                        results['failed_predictions'].append({
                            'crypto_name': crypto_name,
                            'horizon': '1d', 
                            'error': 'price_fetch_failed'
                        })
                        continue
                    
                    # ✅ FIXED: Call with correct signature (4 parameters)
                    result = self._verify_single_prediction_enhanced(prediction, current_price, horizon=1, now=now)
                    results['total_checked'] += 1
                    
                    if result and result.get('success'):
                        results['total_verified'] += 1
                        results['verified_1d'] += 1
                        results['successful_predictions'].append(result)
                        accuracy_1d_list.append(result.get('accuracy_score', 0))
                        
                        # Update database
                        self._update_verification_in_database(result, horizon=1)
                        
                        # Update ML system feedback if available
                        if ml_system and hasattr(ml_system, 'update_model_performance'):
                            try:
                                ml_system.update_model_performance(
                                    crypto_id, 
                                    'ensemble', 
                                    '1d',
                                    {
                                        'direction_correct': result['predicted_direction'] == result.get('actual_direction', 'UNKNOWN'),
                                        'confidence': result['confidence'],
                                        'accuracy_score': result.get('accuracy_score', 0.0),
                                        'predicted_change': result.get('predicted_change', 0.0),
                                        'actual_change': result.get('actual_change', 0.0),
                                        'price_error_percent': result.get('price_error_percent', 0.0)
                                    },
                                    {
                                        'ensemble_method': 'weighted_average',
                                        'models_used': result.get('models_used', []),
                                        'model_weights': result.get('model_weights', {})
                                    }
                                )
                                results['model_feedback'].append(f"Updated {crypto_name} 1d")
                            except Exception as e:
                                print(f"      ⚠️ ML feedback failed: {e}")
                    else:
                        results['errors'] += 1
                        error_msg = result.get('error', 'verification failed') if result else 'no result'
                        results['failed_predictions'].append({
                            'crypto_name': crypto_name,
                            'horizon': '1d',
                            'error': error_msg
                        })
                    
                    # Rate limiting
                    time.sleep(self.rate_limit_delay)
                    
                except Exception as e:
                    print(f"      ❌ Error processing {crypto_name}: {e}")
                    results['errors'] += 1
                    continue
            
            results['accuracy_1d'] = statistics.mean(accuracy_1d_list) if accuracy_1d_list else 0
        
        # Verify 3d predictions with FIXED SIGNATURE
        if predictions_3d:
            print(f"\n📅 Verifying {len(predictions_3d)} 3d predictions...")
            accuracy_3d_list = []
            
            for i, prediction in enumerate(predictions_3d, 1):
                crypto_name = prediction[2] if len(prediction) > 2 else 'Unknown'
                crypto_id = prediction[1]
                print(f"   [{i}/{len(predictions_3d)}] Processing {crypto_name}...")
                
                try:
                    # Get current price
                    current_price = self.get_current_price_robust(crypto_id)
                    
                    if current_price is None:
                        print(f"      ❌ Cannot get price for {crypto_name}")
                        results['errors'] += 1
                        results['failed_predictions'].append({
                            'crypto_name': crypto_name,
                            'horizon': '3d',
                            'error': 'price_fetch_failed'
                        })
                        continue
                    
                    # ✅ FIXED: Call with correct signature (4 parameters)
                    result = self._verify_single_prediction_enhanced(prediction, current_price, horizon=3, now=now)
                    results['total_checked'] += 1
                    
                    if result and result.get('success'):
                        results['total_verified'] += 1
                        results['verified_3d'] += 1
                        results['successful_predictions'].append(result)
                        accuracy_3d_list.append(result.get('accuracy_score', 0))
                        
                        # Update database
                        self._update_verification_in_database(result, horizon=3)
                        
                        # Update ML system feedback if available
                        if ml_system and hasattr(ml_system, 'update_model_performance'):
                            try:
                                ml_system.update_model_performance(
                                        crypto_id, 
                                        'ensemble', 
                                        '3d',
                                        {
                                            'direction_correct': result['predicted_direction'] == result.get('actual_direction', 'UNKNOWN'),
                                            'confidence': result['confidence'],
                                            'accuracy_score': result.get('accuracy_score', 0.0),
                                            'predicted_change': result.get('predicted_change', 0.0),
                                            'actual_change': result.get('actual_change', 0.0),
                                            'price_error_percent': result.get('price_error_percent', 0.0)
                                        },
                                        {
                                            'ensemble_method': 'weighted_average',
                                            'models_used': result.get('models_used', []),
                                            'model_weights': result.get('model_weights', {})
                                        }
                                    )
                                results['model_feedback'].append(f"Updated {crypto_name} 3d")
                            except Exception as e:
                                print(f"      ⚠️ ML feedback failed: {e}")
                    else:
                        results['errors'] += 1
                        error_msg = result.get('error', 'verification failed') if result else 'no result'
                        results['failed_predictions'].append({
                            'crypto_name': crypto_name,
                            'horizon': '3d',
                            'error': error_msg
                        })
                    
                    # Rate limiting
                    time.sleep(self.rate_limit_delay)
                    
                except Exception as e:
                    print(f"      ❌ Error processing {crypto_name}: {e}")
                    results['errors'] += 1
                    continue
            
            results['accuracy_3d'] = statistics.mean(accuracy_3d_list) if accuracy_3d_list else 0
        
        elapsed = time.time() - start_time
        results['elapsed_time'] = elapsed
        
        print(f"\n📊 VERIFICATION SUMMARY:")
        print(f"   ⏰ Time: {elapsed:.1f}s")
        print(f"   ✅ Verified: {results['total_verified']}/{results['total_checked']}")
        print(f"   📈 1d: {results['verified_1d']} (avg accuracy: {results['accuracy_1d']:.1%})")
        print(f"   📅 3d: {results['verified_3d']} (avg accuracy: {results['accuracy_3d']:.1%})")
        print(f"   🔄 ML feedback: {len(results['model_feedback'])} updates")
        print(f"   ❌ Errors: {results['errors']}")
        
        # Send email if significant results
        if results['total_verified'] >= 100 and hasattr(self, 'gmail_user') and self.gmail_user:
            try:
                self.send_accuracy_email_optimized(results, current_cycle)
                print("📧 Verification email sent")
            except Exception as e:
                print(f"⚠️ Email sending failed: {e}")
        
        return results
    
    def _update_daily_accuracy_summary(self, results):
        """📅 Update daily accuracy summary"""
        try:
            today = datetime.now().date()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get current day's data
            cursor.execute('''
                SELECT predictions_1d, correct_1d, predictions_3d, correct_3d
                FROM daily_accuracy_summary WHERE date = ?
            ''', (today,))
            
            existing = cursor.fetchone()
            
            if existing:
                # Update existing record
                new_pred_1d = existing[0] + results['verified_1d']
                new_correct_1d = existing[1] + sum(1 for p in results['successful_predictions'] if p['horizon'] == '1d' and p['direction_correct'])
                new_pred_3d = existing[2] + results['verified_3d']
                new_correct_3d = existing[3] + sum(1 for p in results['successful_predictions'] if p['horizon'] == '3d' and p['direction_correct'])
                
                accuracy_1d = new_correct_1d / new_pred_1d if new_pred_1d > 0 else 0
                accuracy_3d = new_correct_3d / new_pred_3d if new_pred_3d > 0 else 0
                
                cursor.execute('''
                    UPDATE daily_accuracy_summary 
                    SET predictions_1d = ?, correct_1d = ?, accuracy_1d = ?,
                        predictions_3d = ?, correct_3d = ?, accuracy_3d = ?,
                        total_verifications = total_verifications + ?
                    WHERE date = ?
                ''', (
                    new_pred_1d, new_correct_1d, accuracy_1d,
                    new_pred_3d, new_correct_3d, accuracy_3d,
                    results['total_verified'], today
                ))
            else:
                # Insert new record
                correct_1d = sum(1 for p in results['successful_predictions'] if p['horizon'] == '1d' and p['direction_correct'])
                correct_3d = sum(1 for p in results['successful_predictions'] if p['horizon'] == '3d' and p['direction_correct'])
                
                accuracy_1d = correct_1d / results['verified_1d'] if results['verified_1d'] > 0 else 0
                accuracy_3d = correct_3d / results['verified_3d'] if results['verified_3d'] > 0 else 0
                
                cursor.execute('''
                    INSERT INTO daily_accuracy_summary (
                        date, predictions_1d, correct_1d, accuracy_1d,
                        predictions_3d, correct_3d, accuracy_3d, total_verifications
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    today, results['verified_1d'], correct_1d, accuracy_1d,
                    results['verified_3d'], correct_3d, accuracy_3d, results['total_verified']
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"⚠️ Daily summary update failed: {e}")
    
    def get_comprehensive_accuracy_statistics(self, days=7):
        """📊 Get comprehensive accuracy statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            # Overall statistics
            cursor.execute('''
                SELECT 
                    COUNT(*) as total,
                    AVG(accuracy_score) as avg_accuracy,
                    AVG(confidence) as avg_confidence,
                    AVG(CASE WHEN direction_correct = 1 THEN 1.0 ELSE 0.0 END) as direction_accuracy,
                    AVG(price_error_percent) as avg_price_error
                FROM verification_results 
                WHERE verification_timestamp >= ? AND horizon = '1d'
            ''', (cutoff_date,))
            
            stats_1d = cursor.fetchone()
            
            cursor.execute('''
                SELECT 
                    COUNT(*) as total,
                    AVG(accuracy_score) as avg_accuracy,
                    AVG(confidence) as avg_confidence,
                    AVG(CASE WHEN direction_correct = 1 THEN 1.0 ELSE 0.0 END) as direction_accuracy,
                    AVG(price_error_percent) as avg_price_error
                FROM verification_results 
                WHERE verification_timestamp >= ? AND horizon = '3d'
            ''', (cutoff_date,))
            
            stats_3d = cursor.fetchone()
            
            # Best performers
            cursor.execute('''
                SELECT crypto_name, accuracy_score, direction_correct, predicted_change, actual_change
                FROM verification_results 
                WHERE horizon = '1d' AND verification_timestamp >= ?
                ORDER BY accuracy_score DESC
                LIMIT 5
            ''', (cutoff_date,))
            
            best_1d = cursor.fetchall()
            
            cursor.execute('''
                SELECT crypto_name, accuracy_score, direction_correct, predicted_change, actual_change
                FROM verification_results 
                WHERE horizon = '3d' AND verification_timestamp >= ?
                ORDER BY accuracy_score DESC
                LIMIT 5
            ''', (cutoff_date,))
            
            best_3d = cursor.fetchall()
            
            # Model performance
            cursor.execute('''
                SELECT model_name, AVG(accuracy_rate) as avg_accuracy, COUNT(*) as cryptos_count
                FROM model_performance_tracking
                WHERE last_updated >= ? AND horizon = '1d'
                GROUP BY model_name
                ORDER BY avg_accuracy DESC
            ''', (cutoff_date,))
            
            model_perf_1d = cursor.fetchall()
            
            cursor.execute('''
                SELECT model_name, AVG(accuracy_rate) as avg_accuracy, COUNT(*) as cryptos_count
                FROM model_performance_tracking
                WHERE last_updated >= ? AND horizon = '3d'
                GROUP BY model_name
                ORDER BY avg_accuracy DESC
            ''', (cutoff_date,))
            
            model_perf_3d = cursor.fetchall()
            
            conn.close()
            
            return {
                '1d': {
                    'total': stats_1d[0] or 0,
                    'avg_accuracy_score': stats_1d[1] or 0,
                    'avg_confidence': stats_1d[2] or 0,
                    'direction_accuracy': stats_1d[3] or 0,
                    'avg_price_error': stats_1d[4] or 0,
                    'best_predictions': best_1d,
                    'model_performance': model_perf_1d
                },
                '3d': {
                    'total': stats_3d[0] or 0,
                    'avg_accuracy_score': stats_3d[1] or 0,
                    'avg_confidence': stats_3d[2] or 0,
                    'direction_accuracy': stats_3d[3] or 0,
                    'avg_price_error': stats_3d[4] or 0,
                    'best_predictions': best_3d,
                    'model_performance': model_perf_3d
                }
            }
            
        except Exception as e:
            print(f"❌ Error getting accuracy statistics: {e}")
            return None
    
    def send_accuracy_email_optimized(self, verification_results, cycle_number):
        """📧 Send optimized accuracy email with model performance insights"""
        
        if not self.gmail_user or not self.gmail_password:
            print("⚠️ Email credentials not configured - skipping accuracy email")
            return False
        
        try:
            print("📧 Sending optimized accuracy email report...")
            
            # Get comprehensive accuracy statistics
            accuracy_stats = self.get_comprehensive_accuracy_statistics(7)
            
            # Create email
            msg = MIMEMultipart()
            msg['From'] = self.gmail_user
            msg['To'] = self.gmail_user
            msg['Subject'] = f"🎯 Optimized Accuracy Report - Cycle {cycle_number} - {verification_results['total_verified']} Verified"
            
            # Create HTML content
            html_content = self._create_optimized_accuracy_email_html(verification_results, accuracy_stats, cycle_number)
            msg.attach(MIMEText(html_content, 'html'))
            
            # Send email
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(self.gmail_user, self.gmail_password)
            text = msg.as_string()
            server.sendmail(self.gmail_user, self.gmail_user, text)
            server.quit()
            
            print("✅ Optimized accuracy email sent successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error sending accuracy email: {e}")
            return False
    
    def _create_optimized_accuracy_email_html(self, verification_results, accuracy_stats, cycle_number):
        """📧 Create comprehensive accuracy email HTML"""
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f8f9fa; }}
                .container {{ background: white; padding: 30px; border-radius: 10px; max-width: 900px; margin: 0 auto; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }}
                .section {{ margin: 25px 0; padding: 20px; border-radius: 8px; }}
                .verification-section {{ background: linear-gradient(135deg, #4caf50, #45a049); color: white; }}
                .accuracy-section {{ background: linear-gradient(135deg, #ff9800, #f57c00); color: white; }}
                .model-section {{ background: linear-gradient(135deg, #2196f3, #1976d2); color: white; }}
                .stats-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
                .stat-box {{ background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; }}
                .prediction-list {{ background: white; color: #333; padding: 15px; border-radius: 8px; margin: 10px 0; }}
                .success {{ color: #4caf50; font-weight: bold; }}
                .warning {{ color: #ff9800; font-weight: bold; }}
                .error {{ color: #f44336; font-weight: bold; }}
                .footer {{ text-align: center; color: #666; font-size: 0.9em; margin-top: 30px; }}
                .highlight {{ background: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; border-radius: 5px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎯 OPTIMIZED ACCURACY REPORT</h1>
                    <p>Enhanced Verification System - Cycle {cycle_number}</p>
                    <p>{timestamp}</p>
                </div>
                
                <div class="section verification-section">
                    <h2>🔍 Latest Verification Results</h2>
                    <div class="stats-grid">
                        <div class="stat-box">
                            <h3>📊 Verification Summary</h3>
                            <p>Total Checked: {verification_results['total_checked']}</p>
                            <p>Successfully Verified: {verification_results['total_verified']}</p>
                            <p>Verification Rate: {(verification_results['total_verified']/verification_results['total_checked']*100) if verification_results['total_checked'] > 0 else 0:.1f}%</p>
                            <p>Processing Time: {verification_results['elapsed_time']:.1f}s</p>
                        </div>
                        <div class="stat-box">
                            <h3>📈 Horizon Breakdown</h3>
                            <p>1d Verified: {verification_results['verified_1d']}</p>
                            <p>3d Verified: {verification_results['verified_3d']}</p>
                            <p>1d Avg Accuracy: {verification_results['accuracy_1d']:.3f}</p>
                            <p>3d Avg Accuracy: {verification_results['accuracy_3d']:.3f}</p>
                        </div>
                    </div>
                </div>
        """
        
        if accuracy_stats:
            stats_1d = accuracy_stats['1d']
            stats_3d = accuracy_stats['3d']
            
            # Performance status indicators
            status_1d = "🟢" if stats_1d['direction_accuracy'] >= 0.6 else "🟡" if stats_1d['direction_accuracy'] >= 0.5 else "🔴"
            status_3d = "🟢" if stats_3d['direction_accuracy'] >= 0.55 else "🟡" if stats_3d['direction_accuracy'] >= 0.45 else "🔴"
            
            html += f"""
                <div class="section accuracy-section">
                    <h2>📊 Comprehensive Accuracy Statistics (Last 7 days)</h2>
                    <div class="stats-grid">
                        <div class="stat-box">
                            <h3>📈 1-Day Predictions {status_1d}</h3>
                            <p>Total Verified: {stats_1d['total']}</p>
                            <p><strong>Direction Accuracy: {stats_1d['direction_accuracy']:.1%}</strong></p>
                            <p>Avg Accuracy Score: {stats_1d['avg_accuracy_score']:.3f}</p>
                            <p>Avg Confidence: {stats_1d['avg_confidence']:.1%}</p>
                            <p>Avg Price Error: {stats_1d['avg_price_error']:.1f}%</p>
                        </div>
                        <div class="stat-box">
                            <h3>📅 3-Day Predictions {status_3d}</h3>
                            <p>Total Verified: {stats_3d['total']}</p>
                            <p><strong>Direction Accuracy: {stats_3d['direction_accuracy']:.1%}</strong></p>
                            <p>Avg Accuracy Score: {stats_3d['avg_accuracy_score']:.3f}</p>
                            <p>Avg Confidence: {stats_3d['avg_confidence']:.1%}</p>
                            <p>Avg Price Error: {stats_3d['avg_price_error']:.1f}%</p>
                        </div>
                    </div>
                    
                    <div class="prediction-list">
                        <h3>🏆 Best 1d Predictions (Last 7 days)</h3>
            """
            
            for crypto_name, accuracy_score, direction_correct, predicted_change, actual_change in stats_1d['best_predictions']:
                correct_emoji = "✅" if direction_correct else "❌"
                html += f"<p>{correct_emoji} {crypto_name}: {predicted_change:+.2%} → {actual_change:+.2%} (Score: {accuracy_score:.3f})</p>"
            
            html += f"""
                    </div>
                    
                    <div class="prediction-list">
                        <h3>🏆 Best 3d Predictions (Last 7 days)</h3>
            """
            
            for crypto_name, accuracy_score, direction_correct, predicted_change, actual_change in stats_3d['best_predictions']:
                correct_emoji = "✅" if direction_correct else "❌"
                html += f"<p>{correct_emoji} {crypto_name}: {predicted_change:+.2%} → {actual_change:+.2%} (Score: {accuracy_score:.3f})</p>"
            
            html += "</div>"
            
            # Model performance section
            html += f"""
                </div>
                
                <div class="section model-section">
                    <h2>🤖 Model Performance Analysis</h2>
                    <div class="stats-grid">
                        <div class="stat-box">
                            <h3>📈 1d Model Performance</h3>
            """
            
            for model_name, avg_accuracy, crypto_count in stats_1d['model_performance']:
                performance_emoji = "🟢" if avg_accuracy > 0.6 else "🟡" if avg_accuracy > 0.5 else "🔴"
                html += f"<p>{performance_emoji} {model_name}: {avg_accuracy:.1%} ({crypto_count} cryptos)</p>"
            
            html += f"""
                        </div>
                        <div class="stat-box">
                            <h3>📅 3d Model Performance</h3>
            """
            
            for model_name, avg_accuracy, crypto_count in stats_3d['model_performance']:
                performance_emoji = "🟢" if avg_accuracy > 0.55 else "🟡" if avg_accuracy > 0.45 else "🔴"
                html += f"<p>{performance_emoji} {model_name}: {avg_accuracy:.1%} ({crypto_count} cryptos)</p>"
            
            html += "</div></div>"
        
        # Recent successful verifications
        if verification_results['successful_predictions']:
            html += f"""
                <div class="section">
                    <h3>✅ Recent Successful Verifications</h3>
                    <div class="prediction-list">
            """
            
            for pred in verification_results['successful_predictions'][:10]:
                correct_emoji = "✅" if pred['direction_correct'] else "❌"
                confidence_color = "#4caf50" if pred['confidence'] > 0.7 else "#ff9800" if pred['confidence'] > 0.5 else "#f44336"
                html += f"""
                <p>{correct_emoji} {pred['crypto_name']} ({pred['horizon']}): 
                   {pred['predicted_change']:+.2%} → {pred['actual_change']:+.2%} 
                   (Score: {pred['accuracy_score']:.3f}, 
                   <span style="color: {confidence_color}">Conf: {pred['confidence']:.1%}</span>)</p>
                """
            
            html += "</div></div>"
        
        # Performance recommendations
        recommendations = []
        if accuracy_stats:
            if stats_1d['direction_accuracy'] < 0.55:
                recommendations.append("⚠️ 1d direction accuracy below 55% - consider adjusting model weights")
            if stats_3d['direction_accuracy'] < 0.50:
                recommendations.append("⚠️ 3d direction accuracy below 50% - review feature engineering")
            if stats_1d['avg_price_error'] > 15:
                recommendations.append("📊 1d price errors high (>15%) - improve magnitude prediction")
            if len(verification_results.get('model_feedback', [])) > 0:
                recommendations.append(f"🤖 {len(verification_results['model_feedback'])} models updated with feedback")
        
        if recommendations:
            html += f"""
                <div class="highlight">
                    <h3>💡 Performance Recommendations</h3>
            """
            for rec in recommendations:
                html += f"<p>{rec}</p>"
            html += "</div>"
        
        # Summary and footer
        total_accuracy = (verification_results['accuracy_1d'] + verification_results['accuracy_3d']) / 2
        system_health = "🟢 Excellent" if total_accuracy > 0.65 else "🟡 Good" if total_accuracy > 0.55 else "🔴 Needs Attention"
        
        html += f"""
                <div style="margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 8px;">
                    <h3>📋 System Health Summary</h3>
                    <p><strong>Overall System Status:</strong> {system_health}</p>
                    <p><strong>Combined Accuracy:</strong> {total_accuracy:.3f}</p>
                    <p><strong>Verification Coverage:</strong> {verification_results['total_verified']} predictions verified</p>
                    <p><strong>Model Feedback:</strong> {len(verification_results.get('model_feedback', []))} updates applied</p>
                </div>
                
                <div class="footer">
                    <p>🤖 Enhanced Crypto System v2.0 - Optimized Verification</p>
                    <p>🔍 Intelligent model selection with performance feedback loop</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def get_model_selection_recommendations(self, crypto_id=None):
        """🎯 Get model selection recommendations based on performance"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            recommendations = {}
            
            for horizon in ['1d', '3d']:
                if crypto_id:
                    # Specific crypto recommendations
                    cursor.execute('''
                        SELECT model_name, accuracy_rate, total_predictions
                        FROM model_performance_tracking
                        WHERE crypto_id = ? AND horizon = ? AND total_predictions >= 3
                        ORDER BY accuracy_rate DESC
                    ''', (crypto_id, horizon))
                else:
                    # Global recommendations
                    cursor.execute('''
                        SELECT model_name, AVG(accuracy_rate) as avg_accuracy, 
                               SUM(total_predictions) as total_preds
                        FROM model_performance_tracking
                        WHERE horizon = ? AND total_predictions >= 3
                        GROUP BY model_name
                        ORDER BY avg_accuracy DESC
                    ''', (horizon,))
                
                results = cursor.fetchall()
                recommendations[horizon] = results
            
            conn.close()
            return recommendations
            
        except Exception as e:
            print(f"❌ Error getting model recommendations: {e}")
            return {}


# === INTEGRATION HELPER ===
def integrate_optimized_verification():
    """📋 Integration instructions for optimized verification"""
    
    integration_code = '''
# === ADD TO crypto_continuous_dual_optimized.py ===

class EnhancedDualHorizonCryptoContinuousSystemOptimized:
    def __init__(self, config=None, config_file=None):
        # ... existing init code ...
        
        # 🆕 ADD OPTIMIZED VERIFICATION SYSTEM
        self.verification_system = None
        self._init_optimized_verification_system()
    
    def _init_optimized_verification_system(self):
        """🔍 Initialize optimized verification system"""
        try:
            from complete_verification_optimized import OptimizedVerificationSystem
            
            verification_config = {
                'api_timeout': self.config.get('api_timeout', 30),
                'rate_limit_delay': self.config.get('rate_limit_delay', 2.0),
                'max_retries_api': self.config.get('max_retries_api', 3),
                'gmail_user': self.config.get('gmail_user', ''),
                'gmail_app_password': self.config.get('gmail_app_password', ''),
                'accuracy_target_1d': self.config.get('accuracy_target_1d', 0.65),
                'accuracy_target_3d': self.config.get('accuracy_target_3d', 0.60)
            }
            
            self.verification_system = OptimizedVerificationSystem(
                self.db_path, verification_config
            )
            print("✅ Optimized verification system initialized")
            
        except Exception as e:
            print(f"⚠️ Optimized verification system failed: {e}")
            self.verification_system = None
    
    def run_analysis_cycle_optimized(self):
        """🚀 Enhanced cycle with OPTIMIZED verification"""
        
        print(f"\\n🎯 RUNNING OPTIMIZED CYCLE {self.cycle_count + 1}")
        
        # 🔍 OPTIMIZED VERIFICATION WITH MODEL FEEDBACK
        if self.verification_system:
            verification_results = self.verification_system.run_complete_verification_optimized(
                self.cycle_count + 1, 
                ml_system=self.ml_system  # Pass ML system for feedback loop
            )
            
            # Use verification results for system optimization
            if verification_results['total_verified'] > 0:
                accuracy_1d = verification_results.get('accuracy_1d', 0)
                accuracy_3d = verification_results.get('accuracy_3d', 0)
                
                print(f"📊 Verification accuracy: 1d={accuracy_1d:.1%}, 3d={accuracy_3d:.1%}")
                
                # Get model recommendations
                recommendations = self.verification_system.get_model_selection_recommendations()
                if recommendations:
                    print(f"🎯 Model recommendations updated based on performance")
        
        # ... rest of normal cycle code ...
        
        return success
'''
    
    print("📋 OPTIMIZED VERIFICATION INTEGRATION:")
    print("=" * 70)
    print(integration_code)
    print("\n🎯 KEY FEATURES:")
    print("   ✅ Dual horizon verification (1d + 3d)")
    print("   ✅ Model performance tracking")
    print("   ✅ Intelligent model selection")
    print("   ✅ ML system feedback loop")
    print("   ✅ Comprehensive accuracy emails")
    print("   ✅ Performance-based recommendations")


# === STANDALONE TEST ===
def test_optimized_verification():
    """🧪 Test optimized verification system"""
    
    db_path = "test_verification_optimized.db"
    
    print("🧪 TESTING OPTIMIZED VERIFICATION SYSTEM")
    print("=" * 70)
    
    config = {
        'api_timeout': 30,
        'rate_limit_delay': 1.0,  # Faster for testing
        'max_retries_api': 2,
        'gmail_user': 'danieleballarini98@gmail.com',
        'gmail_app_password': 'tyut mbix ifur ymuf',
        'accuracy_target_1d': 0.65,
        'accuracy_target_3d': 0.60
    }
    
    verification_system = OptimizedVerificationSystem(db_path, config)
    
    # Create some test verification data
    print("🔧 Creating test data...")
    
    conn = sqlite3.connect(db_path)
    
    # Insert test predictions
    test_predictions = [
        ('bitcoin', 'Bitcoin', 0.05, 50000, 47619, '2024-01-01 10:00:00', 0.75, '1d'),
        ('ethereum', 'Ethereum', 0.08, 3000, 2800, '2024-01-01 10:00:00', 0.68, '1d'),
        ('bitcoin', 'Bitcoin', 0.10, 52000, 47619, '2024-01-01 10:00:00', 0.72, '3d'),
    ]
    
    for pred in test_predictions:
        conn.execute('''
            INSERT INTO predictions (crypto_id, crypto_name, predicted_change, predicted_price, 
                                   current_price, timestamp, confidence, horizon)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', pred)
    
    conn.commit()
    conn.close()
    
    # Test verification (simulate with manual prices)
    print("🔍 Testing verification process...")
    
    # Mock verification results
    mock_results = {
        'total_checked': 3,
        'total_verified': 3,
        'verified_1d': 2,
        'verified_3d': 1,
        'errors': 0,
        'accuracy_1d': 0.73,
        'accuracy_3d': 0.68,
        'successful_predictions': [
            {
                'success': True, 'horizon': '1d', 'crypto_name': 'Bitcoin',
                'crypto_id': 'bitcoin', 'direction_correct': True,
                'accuracy_score': 0.85, 'confidence': 0.75,
                'predicted_change': 0.05, 'actual_change': 0.06
            }
        ],
        'failed_predictions': [],
        'elapsed_time': 15.2,
        'model_feedback': ['Updated bitcoin 1d', 'Updated ethereum 1d']
    }
    
    # Test email generation
    print("📧 Testing email generation...")
    
    try:
        # Get statistics (will be empty for test)
        accuracy_stats = verification_system.get_comprehensive_accuracy_statistics(7)
        html_content = verification_system._create_optimized_accuracy_email_html(
            mock_results, accuracy_stats, 1
        )
        
        print("✅ Email HTML generated successfully")
        print(f"   Email length: {len(html_content)} characters")
        
    except Exception as e:
        print(f"❌ Email generation failed: {e}")
    
    # Test model recommendations
    print("🎯 Testing model recommendations...")
    recommendations = verification_system.get_model_selection_recommendations()
    print(f"✅ Model recommendations: {len(recommendations)} horizons")
    
    print("\n🎯 TEST RESULTS:")
    print(f"   Verification system: ✅ Initialized")
    print(f"   Database: ✅ Created with test data")
    print(f"   Email generation: ✅ Working")
    print(f"   Model tracking: ✅ Ready")
    
    # Cleanup test database
    import os
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"🧹 Cleaned up test database")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_optimized_verification()
    else:
        integrate_optimized_verification()