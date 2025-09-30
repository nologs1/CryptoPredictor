# MODEL FEEDBACK AND WEIGHT UPDATE SYSTEM
# Sistema per aggiornare i pesi dei modelli basandosi sui risultati di verifica

from pathlib import Path
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import sqlite3
from collections import defaultdict
import joblib

class ModelFeedbackOptimizer:
    """🔄 Sistema di feedback per ottimizzazione modelli"""
    
    def __init__(self, db_path: str, ml_system, config: Dict):
        self.db_path = db_path
        self.ml_system = ml_system
        self.config = config
        
        # Feedback parameters
        self.learning_rate = config.get('model_feedback_learning_rate', 0.1)
        self.momentum = config.get('model_feedback_momentum', 0.9)
        self.min_samples_for_update = config.get('min_samples_feedback', 10)
        self.performance_window_days = config.get('performance_window_days', 30)
        
        # Model weights storage
        self.model_weights = defaultdict(lambda: defaultdict(lambda: 1.0))
        self.weight_history = defaultdict(list)
        
        # Load existing weights
        self._load_model_weights()
        
        print("🔄 Model Feedback Optimizer initialized")
        print(f"📊 Learning rate: {self.learning_rate}")
        print(f"⏱️ Performance window: {self.performance_window_days} days")
    
    def _safe_json_convert(self, value):
        """🔧 Converte valori in formato JSON-safe per database"""
        if isinstance(value, dict):
            return json.dumps(value)
        elif isinstance(value, list):
            return json.dumps(value)
        elif isinstance(value, (int, float, str, bool)):
            return value
        elif value is None:
            return None
        else:
            # Per altri tipi, converte in stringa
            return str(value)

    def _load_model_weights(self):
        """📂 Carica pesi dei modelli esistenti"""
        try:
            weights_file = "D:/CryptoSystem/cache/model_weights.json"
            if Path(weights_file).exists():
                with open(weights_file, 'r') as f:
                    data = json.load(f)
                    self.model_weights = defaultdict(lambda: defaultdict(lambda: 1.0), data.get('weights', {}))
                    self.weight_history = defaultdict(list, data.get('history', {}))
                print(f"✅ Loaded existing model weights for {len(self.model_weights)} cryptos")
            else:
                print("📝 No existing model weights found, starting fresh")
        except Exception as e:
            print(f"⚠️ Failed to load model weights: {e}")
    
    def _save_model_weights(self):
        """💾 Salva pesi dei modelli"""
        try:
            weights_file = "D:/CryptoSystem/cache/model_weights.json"
            Path(weights_file).parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'weights': dict(self.model_weights),
                'history': dict(self.weight_history),
                'last_updated': datetime.now().isoformat()
            }
            
            with open(weights_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            print(f"💾 Model weights saved to {weights_file}")
        except Exception as e:
            print(f"❌ Failed to save model weights: {e}")
    
    def analyze_model_performance(self, crypto_id: str, horizon: str) -> Dict:
        """📊 Analizza performance di tutti i modelli per una crypto"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_date = (datetime.now() - timedelta(days=self.performance_window_days)).isoformat()
            
            # Get recent verification results
            cursor.execute('''
                SELECT models_used, direction_correct, accuracy_score, confidence,
                       price_error_percent, verification_timestamp
                FROM verification_results 
                WHERE crypto_id = ? AND horizon = ? 
                AND verification_timestamp >= ?
                ORDER BY verification_timestamp DESC
            ''', (crypto_id, horizon, cutoff_date))
            
            results = cursor.fetchall()
            conn.close()
            
            if len(results) < self.min_samples_for_update:
                return {'insufficient_data': True, 'samples': len(results)}
            
            # Analyze performance by model
            model_performance = defaultdict(lambda: {
                'predictions': 0,
                'correct_directions': 0,
                'total_accuracy': 0,
                'total_confidence': 0,
                'price_errors': [],
                'recent_trend': []
            })
            
            for result in results:
                models_used_str, direction_correct, accuracy_score, confidence, price_error, timestamp = result
                
                try:
                    models_used = json.loads(models_used_str) if models_used_str else []
                except:
                    continue
                
                for model_name in models_used:
                    perf = model_performance[model_name]
                    perf['predictions'] += 1
                    perf['correct_directions'] += 1 if direction_correct else 0
                    perf['total_accuracy'] += accuracy_score if accuracy_score else 0
                    perf['total_confidence'] += confidence if confidence else 0
                    perf['price_errors'].append(price_error if price_error else 0)
                    perf['recent_trend'].append({
                        'timestamp': timestamp,
                        'accuracy': accuracy_score if accuracy_score else 0,
                        'correct': direction_correct
                    })
            
            # Calculate final metrics
            performance_summary = {}
            for model_name, perf in model_performance.items():
                if perf['predictions'] > 0:
                    performance_summary[model_name] = {
                        'direction_accuracy': perf['correct_directions'] / perf['predictions'],
                        'avg_accuracy_score': perf['total_accuracy'] / perf['predictions'],
                        'avg_confidence': perf['total_confidence'] / perf['predictions'],
                        'avg_price_error': np.mean(perf['price_errors']) if perf['price_errors'] else 0,
                        'predictions_count': perf['predictions'],
                        'trend_score': self._calculate_trend_score(perf['recent_trend']),
                        'consistency_score': self._calculate_consistency_score(perf['recent_trend'])
                    }
            
            return {
                'performance_summary': performance_summary,
                'total_samples': len(results),
                'analysis_period_days': self.performance_window_days
            }
            
        except Exception as e:
            print(f"❌ Performance analysis failed for {crypto_id}: {e}")
            return {'error': str(e)}
    
    def _calculate_trend_score(self, recent_trend: List[Dict]) -> float:
        """📈 Calcola score di trend basato sulle performance recenti"""
        if len(recent_trend) < 3:
            return 0.5  # Neutral
        
        # Sort by timestamp
        trend_sorted = sorted(recent_trend, key=lambda x: x['timestamp'])
        
        # Calculate trend using linear regression on accuracy scores
        accuracies = [t['accuracy'] for t in trend_sorted]
        x = np.arange(len(accuracies))
        
        if len(set(accuracies)) > 1:  # Avoid perfect correlations
            correlation = np.corrcoef(x, accuracies)[0, 1]
            # Convert correlation to 0-1 score (0.5 = no trend)
            trend_score = (correlation + 1) / 2
        else:
            trend_score = 0.5
        
        return max(0, min(1, trend_score))
    
    def _calculate_consistency_score(self, recent_trend: List[Dict]) -> float:
        """🎯 Calcola score di consistenza"""
        if len(recent_trend) < 3:
            return 0.5
        
        accuracies = [t['accuracy'] for t in recent_trend]
        
        # Calculate coefficient of variation (lower = more consistent)
        if np.mean(accuracies) > 0:
            cv = np.std(accuracies) / np.mean(accuracies)
            # Convert to 0-1 score (higher = more consistent)
            consistency_score = max(0, min(1, 1 - cv))
        else:
            consistency_score = 0
        
        return consistency_score
    
    def update_model_weights(self, crypto_id: str, horizon: str) -> Dict:
        """⚖️ Aggiorna pesi dei modelli basandosi sulle performance"""
        
        # Analyze current performance
        performance_analysis = self.analyze_model_performance(crypto_id, horizon)
        
        if performance_analysis.get('insufficient_data') or performance_analysis.get('error'):
            return performance_analysis
        
        performance_summary = performance_analysis['performance_summary']
        
        if not performance_summary:
            return {'error': 'No model performance data found'}
        
        # Calculate new weights
        model_scores = {}
        for model_name, perf in performance_summary.items():
            # Composite score combining multiple metrics
            direction_weight = 0.4
            accuracy_weight = 0.3
            trend_weight = 0.2
            consistency_weight = 0.1
            
            composite_score = (
                perf['direction_accuracy'] * direction_weight +
                perf['avg_accuracy_score'] * accuracy_weight +
                perf['trend_score'] * trend_weight +
                perf['consistency_score'] * consistency_weight
            )
            
            model_scores[model_name] = composite_score
        
        # Normalize scores to weights
        if model_scores:
            min_score = min(model_scores.values())
            max_score = max(model_scores.values())
            
            weight_key = f"{crypto_id}_{horizon}"
            old_weights = dict(self.model_weights[weight_key])
            
            for model_name, score in model_scores.items():
                if max_score > min_score:
                    # Normalize to 0.1 - 2.0 range
                    normalized_score = 0.1 + 1.9 * (score - min_score) / (max_score - min_score)
                else:
                    normalized_score = 1.0
                
                # Apply learning rate and momentum
                old_weight = self.model_weights[weight_key][model_name]
                new_weight = (
                    old_weight * (1 - self.learning_rate) +
                    normalized_score * self.learning_rate
                )
                
                self.model_weights[weight_key][model_name] = new_weight
            
            # Store weight history for tracking
            weight_update = {
                'timestamp': datetime.now().isoformat(),
                'crypto_id': crypto_id,
                'horizon': horizon,
                'old_weights': old_weights,
                'new_weights': dict(self.model_weights[weight_key]),
                'performance_scores': model_scores
            }
            
            history_key = f"{crypto_id}_{horizon}"
            self.weight_history[history_key].append(weight_update)
            
            # Keep only last 50 updates
            if len(self.weight_history[history_key]) > 50:
                self.weight_history[history_key] = self.weight_history[history_key][-50:]
            
            # Save updated weights
            self._save_model_weights()
            
            # Update ML system weights if available
            if hasattr(self.ml_system, 'update_model_weights'):
                self.ml_system.update_model_weights(crypto_id, horizon, dict(self.model_weights[weight_key]))
            
            return {
                'success': True,
                'updated_weights': dict(self.model_weights[weight_key]),
                'performance_summary': performance_summary,
                'weight_changes': {
                    model: new_weight - old_weights.get(model, 1.0)
                    for model, new_weight in self.model_weights[weight_key].items()
                }
            }
        
        return {'error': 'No valid model scores calculated'}
    
    def get_optimal_model_ensemble(self, crypto_id: str, horizon: str) -> Dict:
        """🎯 Ottiene ensemble ottimale basato sui pesi attuali"""
        weight_key = f"{crypto_id}_{horizon}"
        weights = dict(self.model_weights[weight_key])
        
        if not weights:
            return {'default_ensemble': True, 'message': 'No optimized weights available'}
        
        # Sort models by weight
        sorted_models = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        
        # Select top models (weight > 0.5)
        selected_models = [(model, weight) for model, weight in sorted_models if weight > 0.5]
        
        if not selected_models:
            # Fall back to all models if none have weight > 0.5
            selected_models = sorted_models
        
        # Normalize weights for ensemble
        total_weight = sum(weight for _, weight in selected_models)
        normalized_weights = {
            model: weight / total_weight for model, weight in selected_models
        }
        
        return {
            'ensemble_models': list(normalized_weights.keys()),
            'model_weights': normalized_weights,
            'confidence_boost': min(total_weight / len(selected_models), 1.2),  # Max 20% boost
            'excluded_models': [model for model, weight in sorted_models if weight <= 0.5]
        }
    
    def run_periodic_weight_updates(self) -> Dict:
        """🔄 Esegue aggiornamenti periodici dei pesi per tutti i modelli"""
        update_results = {
            'updated_cryptos': [],
            'skipped_cryptos': [],
            'errors': [],
            'total_updates': 0
        }
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get unique crypto-horizon combinations with recent verifications
            cutoff_date = (datetime.now() - timedelta(days=7)).isoformat()
            cursor.execute('''
                SELECT DISTINCT crypto_id, horizon, COUNT(*) as verification_count
                FROM verification_results 
                WHERE verification_timestamp >= ?
                GROUP BY crypto_id, horizon
                HAVING COUNT(*) >= ?
            ''', (cutoff_date, self.min_samples_for_update))
            
            crypto_horizons = cursor.fetchall()
            conn.close()
            
            print(f"🔄 Updating weights for {len(crypto_horizons)} crypto-horizon combinations")
            
            for crypto_id, horizon, count in crypto_horizons:
                try:
                    result = self.update_model_weights(crypto_id, horizon)
                    
                    if result.get('success'):
                        update_results['updated_cryptos'].append({
                            'crypto_id': crypto_id,
                            'horizon': horizon,
                            'verification_count': count,
                            'weight_changes': result['weight_changes']
                        })
                        update_results['total_updates'] += 1
                    else:
                        update_results['skipped_cryptos'].append({
                            'crypto_id': crypto_id,
                            'horizon': horizon,
                            'reason': result.get('error', 'Unknown')
                        })
                        
                except Exception as e:
                    error_info = {
                        'crypto_id': crypto_id,
                        'horizon': horizon,
                        'error': str(e)
                    }
                    update_results['errors'].append(error_info)
                    print(f"❌ Weight update failed for {crypto_id} {horizon}: {e}")
            
        except Exception as e:
            update_results['errors'].append({'general_error': str(e)})
        
        print(f"✅ Periodic weight update completed:")
        print(f"   • Updated: {update_results['total_updates']}")
        print(f"   • Skipped: {len(update_results['skipped_cryptos'])}")
        print(f"   • Errors: {len(update_results['errors'])}")
        
        return update_results

# INTEGRATION WITH VERIFICATION SYSTEM
def integrate_model_feedback(verification_system, ml_system):
    """🔧 Integra il sistema di feedback con il sistema di verifica"""
    
    # Create feedback optimizer
    feedback_optimizer = ModelFeedbackOptimizer(
        verification_system.db_path,
        ml_system,
        verification_system.config
    )
    
    # Add method to verification system
    verification_system.update_model_weights_after_verification = lambda crypto_id, horizon: \
        feedback_optimizer.update_model_weights(crypto_id, horizon)
    
    verification_system.run_periodic_model_optimization = \
        feedback_optimizer.run_periodic_weight_updates
    
    verification_system.get_optimal_ensemble = \
        feedback_optimizer.get_optimal_model_ensemble
    
    # Add ML system integration
    if hasattr(ml_system, 'set_feedback_optimizer'):
        ml_system.set_feedback_optimizer(feedback_optimizer)
    
    print("✅ Model feedback system integrated")
    return feedback_optimizer