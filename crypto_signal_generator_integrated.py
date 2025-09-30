# crypto_signal_generator_integrated.py - VERSIONE COMPATIBILE CON SISTEMA PRINCIPALE
"""
Signal Generation System INTEGRATO con sistema principale:
- Nome classe e metodi compatibili
- Interfaccia corretta per sistema principale
- Metodi richiesti implementati
- Gestione configurazione da file JSON
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

class CryptoSignalGeneratorIntegrated:
    def __init__(self, config_path: str, database_path: str):
        """Inizializzazione compatibile con sistema principale"""
        self.db_path = database_path
        self.config_path = config_path
        
        # Load configuration from JSON file
        self.signal_config = self._load_config(config_path)
        
        # Cache for performance data
        self._accuracy_cache = {}
        self._cache_timestamp = None
        self._cache_hours = 2
        
        print("📊 Crypto Signal Generator Integrated initialized")
        print(f"🎯 Thresholds: Accuracy ≥{self.signal_config['min_accuracy']:.0%}, Confidence ≥{self.signal_config['min_confidence']:.0%}")
        print(f"📄 Horizons: {', '.join(self.signal_config['active_horizons'])}")
    
    def _load_config(self, config_path: str) -> dict:
        """Load signal configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                full_config = json.load(f)
            
            # Extract signal-specific config or use defaults
            signal_config = full_config.get('signal_generation', self._default_config())
            
            print(f"✅ Signal config loaded from: {config_path}")
            return signal_config
            
        except Exception as e:
            print(f"⚠️ Failed to load config from {config_path}: {e}")
            print("🔄 Using default signal configuration")
            return self._default_config()
    
    def _default_config(self) -> dict:
        """Default configuration for signal generation"""
        return {
            'min_accuracy': 0.65,
            'min_confidence': 0.70,
            'min_change_1d': 0.04,
            'min_change_3d': 0.06,
            'max_signals_per_session': 10,
            'min_data_points': 5,
            'risk_factor_multiplier': 1.2,
            'active_horizons': ['1d', '3d'],
            'horizon_weights': {'1d': 1.0, '3d': 0.8},
            'consider_bitcoin_trend': True,
            'btc_trend_threshold': 0.03,
            'signal_types': ['BUY', 'SELL', 'STRONG_BUY', 'STRONG_SELL'],
            'strong_signal_multiplier': 2.0
        }
    
    def generate_signals(self, hours_back: int = 8, max_signals: int = None) -> List[dict]:
        """Generate trading signals - METODO RICHIESTO DAL SISTEMA PRINCIPALE"""
        if max_signals is None:
            max_signals = self.signal_config['max_signals_per_session']
        
        print(f"🎯 Generating signals from last {hours_back}h (max {max_signals})")
        
        try:
            # Get recent predictions
            recent_predictions = self.get_recent_predictions(hours_back)
            
            if not recent_predictions:
                print("📭 No recent predictions found for signal generation")
                return []
            
            signals = []
            
            for prediction in recent_predictions:
                try:
                    # Get historical accuracy for this crypto+horizon
                    accuracy_data = self.get_crypto_historical_accuracy(
                        prediction['crypto_id'], 
                        prediction['horizon']
                    )
                    
                    # Generate signal if criteria met
                    signal = self._evaluate_prediction_for_signal(prediction, accuracy_data)
                    
                    if signal:
                        signals.append(signal)
                        
                        if len(signals) >= max_signals:
                            break
                            
                except Exception as e:
                    print(f"⚠️ Error processing prediction {prediction.get('crypto_id', 'unknown')}: {e}")
                    continue
            
            print(f"✅ Generated {len(signals)} trading signals")
            
            # Sort by signal strength and confidence
            signals = sorted(signals, 
                           key=lambda x: (x['confidence'], abs(x['predicted_change'])), 
                           reverse=True)
            
            return signals
            
        except Exception as e:
            print(f"❌ Signal generation failed: {e}")
            return []
    
    def get_recent_predictions(self, hours_back: int = 8) -> List[dict]:
        """Get recent predictions - METODO RICHIESTO DAL SISTEMA PRINCIPALE"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            cutoff_timestamp = cutoff_time.strftime('%Y-%m-%d %H:%M:%S')
            
            query = """
                SELECT 
                    crypto_id, crypto_name, current_price, timestamp,
                    predicted_change_1d, confidence_1d, predicted_price_1d,
                    predicted_change_3d, confidence_3d, predicted_price_3d,
                    quality_score_1d, quality_score_3d
                FROM predictions_optimized 
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            """
            
            df = pd.read_sql_query(query, conn, params=(cutoff_timestamp,))
            conn.close()
            
            if df.empty:
                return []
            
            # Convert to list of dictionaries with both horizons
            predictions = []
            
            for _, row in df.iterrows():
                # Add 1d prediction
                if pd.notna(row['predicted_change_1d']) and pd.notna(row['confidence_1d']):
                    predictions.append({
                        'crypto_id': row['crypto_id'],
                        'crypto_name': row['crypto_name'],
                        'current_price': row['current_price'],
                        'timestamp': row['timestamp'],
                        'horizon': '1d',
                        'predicted_change': row['predicted_change_1d'],
                        'confidence': row['confidence_1d'],
                        'predicted_price': row['predicted_price_1d'],
                        'quality_score': row.get('quality_score_1d', 0.5)
                    })
                
                # Add 3d prediction
                if pd.notna(row['predicted_change_3d']) and pd.notna(row['confidence_3d']):
                    predictions.append({
                        'crypto_id': row['crypto_id'],
                        'crypto_name': row['crypto_name'],
                        'current_price': row['current_price'],
                        'timestamp': row['timestamp'],
                        'horizon': '3d',
                        'predicted_change': row['predicted_change_3d'],
                        'confidence': row['confidence_3d'],
                        'predicted_price': row['predicted_price_3d'],
                        'quality_score': row.get('quality_score_3d', 0.5)
                    })
            
            print(f"📊 Found {len(predictions)} recent predictions")
            return predictions
            
        except Exception as e:
            print(f"❌ Error getting recent predictions: {e}")
            return []
    
    def get_crypto_historical_accuracy(self, crypto_id: str, horizon: str) -> dict:
        """Get historical accuracy for crypto+horizon - METODO RICHIESTO DAL SISTEMA PRINCIPALE"""
        try:
            cache_key = f"{crypto_id}_{horizon}"
            
            # Check cache first
            if (self._cache_timestamp and 
                (datetime.now() - self._cache_timestamp).total_seconds() < self._cache_hours * 3600 and
                cache_key in self._accuracy_cache):
                return self._accuracy_cache[cache_key]
            
            # Query verification results
            conn = sqlite3.connect(self.db_path)
            
            # Look for verification results
            horizon_days = int(horizon.rstrip('d'))
            
            query = """
                SELECT 
                    COUNT(*) as total_predictions,
                    SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct_predictions,
                    AVG(confidence) as avg_confidence,
                    AVG(predicted_change) as avg_predicted_change,
                    AVG(actual_change) as avg_actual_change
                FROM verification_results 
                WHERE crypto_id = ? AND horizon = ?
                AND verification_timestamp >= datetime('now', '-30 days')
            """
            
            cursor = conn.execute(query, (crypto_id, horizon_days))
            result = cursor.fetchone()
            conn.close()
            
            if result and result[0] > 0:
                total, correct, avg_conf, avg_pred, avg_actual = result
                accuracy_rate = correct / total if total > 0 else 0
                
                accuracy_data = {
                    'data_available': True,
                    'accuracy_rate': accuracy_rate,
                    'total_predictions': total,
                    'correct_predictions': correct,
                    'avg_confidence': avg_conf or 0,
                    'avg_predicted_change': avg_pred or 0,
                    'avg_actual_change': avg_actual or 0,
                    'meets_min_data': total >= self.signal_config['min_data_points']
                }
            else:
                # No verification data available
                accuracy_data = {
                    'data_available': False,
                    'accuracy_rate': 0.5,  # Neutral assumption
                    'total_predictions': 0,
                    'correct_predictions': 0,
                    'avg_confidence': 0,
                    'avg_predicted_change': 0,
                    'avg_actual_change': 0,
                    'meets_min_data': False
                }
            
            # Cache result
            self._accuracy_cache[cache_key] = accuracy_data
            if not self._cache_timestamp:
                self._cache_timestamp = datetime.now()
            
            return accuracy_data
            
        except Exception as e:
            print(f"❌ Error getting historical accuracy for {crypto_id} {horizon}: {e}")
            return {
                'data_available': False,
                'accuracy_rate': 0.5,
                'total_predictions': 0,
                'correct_predictions': 0,
                'meets_min_data': False
            }
    
    def _evaluate_prediction_for_signal(self, prediction: dict, accuracy_data: dict) -> Optional[dict]:
        """Evaluate if prediction meets signal criteria"""
        try:
            # Extract data
            crypto_id = prediction['crypto_id']
            crypto_name = prediction['crypto_name']
            horizon = prediction['horizon']
            predicted_change = prediction['predicted_change']
            confidence = prediction['confidence']
            current_price = prediction['current_price']
            
            # Basic thresholds
            min_confidence = self.signal_config['min_confidence']
            min_accuracy = self.signal_config['min_accuracy']
            min_change = self.signal_config[f'min_change_{horizon}']
            
            # Check basic criteria
            if confidence < min_confidence:
                return None
            
            if abs(predicted_change) < min_change:
                return None
            
            # Check historical accuracy (if available)
            if accuracy_data['data_available'] and accuracy_data['meets_min_data']:
                if accuracy_data['accuracy_rate'] < min_accuracy:
                    return None
            
            # Determine signal type
            abs_change = abs(predicted_change)
            strong_threshold = min_change * self.signal_config['strong_signal_multiplier']
            
            if abs_change >= strong_threshold:
                signal_type = 'STRONG_BUY' if predicted_change > 0 else 'STRONG_SELL'
            else:
                signal_type = 'BUY' if predicted_change > 0 else 'SELL'
            
            # Calculate position size suggestion (percentage of portfolio)
            base_size = 0.05  # 5% base
            confidence_mult = confidence
            accuracy_mult = min(accuracy_data.get('accuracy_rate', 0.5) * 1.5, 1.0)
            
            position_size = base_size * confidence_mult * accuracy_mult
            position_size = min(position_size, 0.15)  # Max 15%
            
            # Calculate risk score
            risk_factors = []
            if not accuracy_data['meets_min_data']:
                risk_factors.append('limited_history')
            if confidence < 0.75:
                risk_factors.append('moderate_confidence')
            if abs_change > 0.15:  # >15% change
                risk_factors.append('high_volatility')
            
            if len(risk_factors) >= 2:
                risk_score = 'HIGH'
            elif len(risk_factors) == 1:
                risk_score = 'MEDIUM'
            else:
                risk_score = 'LOW'
            
            # Create signal
            signal = {
                'timestamp': datetime.now(),
                'crypto_id': crypto_id,
                'crypto_name': crypto_name,
                'current_price': current_price,
                'signal_type': signal_type,
                'horizon': horizon,
                'predicted_change': predicted_change,
                'target_price': prediction['predicted_price'],
                'confidence': confidence,
                'historical_accuracy': accuracy_data.get('accuracy_rate', 0.5),
                'position_size_pct': position_size,
                'risk_score': risk_score,
                'risk_factors': risk_factors,
                'prediction_timestamp': prediction['timestamp'],
                'signal_strength': min(confidence * abs_change * 10, 1.0)  # 0-1 scale
            }
            
            return signal
            
        except Exception as e:
            print(f"❌ Error evaluating prediction: {e}")
            return None
    
    def export_signals_to_json(self, signals: List[dict], filename: str = None) -> str:
        """Export signals to JSON file - COMPATIBILE CON SISTEMA BASE"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"crypto_signals_{timestamp}.json"
        
        # Ensure export directory exists
        export_dir = Path("D:/CryptoSystem/exports/signals")
        export_dir.mkdir(parents=True, exist_ok=True)
        
        full_path = export_dir / filename
        
        # Convert datetime objects to strings for JSON serialization
        signals_export = []
        for signal in signals:
            signal_copy = signal.copy()
            signal_copy['timestamp'] = signal_copy['timestamp'].isoformat()
            if isinstance(signal_copy.get('prediction_timestamp'), str):
                # Already string, keep as is
                pass
            else:
                signal_copy['prediction_timestamp'] = str(signal_copy.get('prediction_timestamp', ''))
            signals_export.append(signal_copy)
        
        export_data = {
            'generated_at': datetime.now().isoformat(),
            'total_signals': len(signals),
            'config_used': self.signal_config,
            'signals': signals_export,
            'summary': {
                'buy_signals': len([s for s in signals if 'BUY' in s['signal_type']]),
                'sell_signals': len([s for s in signals if 'SELL' in s['signal_type']]),
                'low_risk': len([s for s in signals if s['risk_score'] == 'LOW']),
                'medium_risk': len([s for s in signals if s['risk_score'] == 'MEDIUM']),
                'high_risk': len([s for s in signals if s['risk_score'] == 'HIGH'])
            }
        }
        
        with open(full_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"💾 Signals exported to: {full_path}")
        return str(full_path)