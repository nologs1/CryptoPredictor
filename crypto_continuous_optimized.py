# crypto_continuous_optimized_ssd.py - SISTEMA CONTINUO CON SSD ESTERNO
"""
Sistema Crypto Continuo OTTIMIZZATO con storage su SSD esterno (D:):
- Prompt interattivo per selezione JSON di configurazione
- Storage completo su SSD esterno (D:/CryptoSystem)
- Tutti i thresholds e parametri definiti nel JSON
- Auto-creazione directory e gestione percorsi SSD
- Configurazione completamente esternalizzata
"""

import random
import time
import sqlite3
import json
import signal
import sys
import os
import gc
import psutil
from datetime import datetime, timedelta
from pathlib import Path
import requests
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

# Import dei moduli ottimizzati
from crypto_signal_generator_integrated import CryptoSignalGeneratorIntegrated

try:
    from crypto_prediction.advanced_crypto_ml_system_dual_optimized import OptimizedDualHorizonMLSystem
    ML_SYSTEM_AVAILABLE = True
    print("✅ Optimized ML System available")
except ImportError:
    print("⚠️ advanced_crypto_ml_system_dual_optimized not found")
    ML_SYSTEM_AVAILABLE = False

try:
    from bitcoin_benchmark_system_real import RealBitcoinBenchmarkSystem
    BITCOIN_SYSTEM_AVAILABLE = True
    print("✅ Real Bitcoin Benchmark System available")
except ImportError:
    print("⚠️ bitcoin_benchmark_system_real not found")
    BITCOIN_SYSTEM_AVAILABLE = False

try:
    from complete_verification_optimized import OptimizedVerificationSystem
    VERIFICATION_SYSTEM_AVAILABLE = True
    print("✅ Optimized Verification System available")
except ImportError:
    print("⚠️ complete_verification_optimized not found")
    VERIFICATION_SYSTEM_AVAILABLE = False

try:
    from crypto_database_optimized import OptimizedCryptoDatabase
    DATABASE_SYSTEM_AVAILABLE = True
    print("✅ Optimized Database System available")
except ImportError:
    print("⚠️ crypto_database_optimized not found")
    DATABASE_SYSTEM_AVAILABLE = False
# Import Shitcoin DEX Scanner
try:
    # ✅ NUOVO IMPORT (CORRETTO):
    from shitcoin_dex_scanner import (
        ShitcoinDEXScanner,
        integrate_scanner_with_main_system,
        handle_shitcoin_menu_choice as original_handle_choice
    )
    SHITCOIN_SCANNER_AVAILABLE = True
    print("✅ Shitcoin DEX Scanner module imported")
except ImportError as e:
    print(f"⚠️ Shitcoin DEX Scanner not available: {e}")
    SHITCOIN_SCANNER_AVAILABLE = False

def prompt_for_config_file():
    """🔧 Prompt interattivo per selezione file JSON di configurazione"""
    print("\n" + "="*80)
    print("🔧 CONFIGURAZIONE SISTEMA CRYPTO OTTIMIZZATO")
    print("="*80)
    
    # Lista dei file JSON disponibili
    config_files = []
    current_dir = Path.cwd()
    
    # Cerca file JSON di configurazione
    json_files = list(current_dir.glob("crypto_config*.json"))
    json_files.extend(list(current_dir.glob("config*.json")))
    
    if json_files:
        print("\n📄 File di configurazione trovati:")
        for i, config_file in enumerate(json_files, 1):
            file_size = config_file.stat().st_size / 1024  # KB
            modified = datetime.fromtimestamp(config_file.stat().st_mtime)
            print(f"   {i}. {config_file.name} ({file_size:.1f} KB, modificato: {modified.strftime('%Y-%m-%d %H:%M')})")
            config_files.append(config_file)
    
    print(f"\n📋 Opzioni disponibili:")
    print(f"   {len(config_files) + 1}. Crea nuovo file JSON ottimizzato per SSD")
    print(f"   {len(config_files) + 2}. Inserisci percorso personalizzato")
    print(f"   {len(config_files) + 3}. Usa configurazione di default (SSD D:)")
    print(f"   0. Esci")
    
    while True:
        try:
            choice = input("\n🔧 Seleziona opzione: ").strip()
            
            if choice == "0":
                print("👋 Uscita...")
                sys.exit(0)
            
            elif choice.isdigit():
                choice_num = int(choice)
                
                # File esistenti
                if 1 <= choice_num <= len(config_files):
                    selected_file = config_files[choice_num - 1]
                    print(f"✅ Selezionato: {selected_file}")
                    return str(selected_file)
                
                # Crea nuovo file
                elif choice_num == len(config_files) + 1:
                    return create_new_config_file()
                
                # Percorso personalizzato
                elif choice_num == len(config_files) + 2:
                    custom_path = input("📁 Inserisci percorso completo del file JSON: ").strip()
                    if Path(custom_path).exists():
                        print(f"✅ File trovato: {custom_path}")
                        return custom_path
                    else:
                        print(f"❌ File non trovato: {custom_path}")
                        continue
                
                # Configurazione di default
                elif choice_num == len(config_files) + 3:
                    print("✅ Usando configurazione di default per SSD")
                    return None
                
                else:
                    print("❌ Scelta non valida")
            
            else:
                print("❌ Inserisci un numero valido")
                
        except KeyboardInterrupt:
            print("\n👋 Uscita...")
            sys.exit(0)
        except Exception as e:
            print(f"❌ Errore: {e}")


def create_new_config_file():
    """📄 Crea nuovo file di configurazione ottimizzato"""
    print("\n📄 CREAZIONE NUOVO FILE DI CONFIGURAZIONE")
    print("="*50)
    
    # Chiedi dettagli
    config_name = input("📝 Nome configurazione (default: 'SSD_Optimized'): ").strip()
    if not config_name:
        config_name = "SSD_Optimized"
    
    # Percorso SSD
    ssd_path = input("💾 Percorso SSD (default: 'D:/CryptoSystem'): ").strip()
    if not ssd_path:
        ssd_path = "D:/CryptoSystem"
    
    # Credenziali email
    gmail_user = input("📧 Gmail user (opzionale): ").strip()
    gmail_password = input("🔑 Gmail app password (opzionale): ").strip()
    
    # API key
    api_key = input("🔑 CoinGecko API key (opzionale): ").strip()
    
    # Genera nome file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_filename = f"crypto_config_{config_name.lower()}_{timestamp}.json"
    
    # Template di configurazione
    config_template = {
        "config_name": f"{config_name}_OPTIMIZED",
        "config_version": "2.0",
        "config_description": f"Configurazione ottimizzata {config_name} con storage SSD",
        "created_date": datetime.now().strftime("%Y-%m-%d"),
        
        "storage": {
            "base_directory": ssd_path,
            "database": {
                "main_db_path": f"{ssd_path}/database/crypto_optimized_6_9months.db",
                "backup_directory": f"{ssd_path}/database/backups",
                "archive_directory": f"{ssd_path}/database/archive"
            },
            "cache": {
                "cache_directory": f"{ssd_path}/cache",
                "api_cache_directory": f"{ssd_path}/cache/api",
                "ml_cache_directory": f"{ssd_path}/cache/ml_models",
                "bitcoin_cache_directory": f"{ssd_path}/cache/bitcoin"
            },
            "logs": {
                "logs_directory": f"{ssd_path}/logs",
                "verification_logs": f"{ssd_path}/logs/verification",
                "performance_logs": f"{ssd_path}/logs/performance"
            }
        },
        
        "lookback_strategy": {
            "lookback_standard": 180,
            "lookback_premium": 270,
            "lookback_minimum": 120,
            "top_crypto_rank_threshold": 50
        },
        
        "quality_thresholds": {
            "confidence_threshold_1d": 0.60,
            "confidence_threshold_3d": 0.55,
            "quality_score_min_1d": 0.55,
            "quality_score_min_3d": 0.50
        },
        
        "api": {
            "coingecko_api_key": api_key,
            "max_api_calls_per_day": 1000,
            "api_delay": 2.0,
            "cache_duration_hours": 48
        },
        
        "email": {
            "gmail_user": gmail_user,
            "gmail_app_password": gmail_password,
            "email_interval_hours": 8,
            "max_daily_emails": 6
        },
        
        "system": {
            "max_cryptos_total": 750,
            "block_size": 25,
            "analysis_interval_hours": 8
        }
    }
    
    # Salva file
    try:
        with open(config_filename, 'w', encoding='utf-8') as f:
            json.dump(config_template, f, indent=2, ensure_ascii=False)
        
        print(f"✅ File creato: {config_filename}")
        print(f"📁 Percorso completo: {Path(config_filename).absolute()}")
        
        # Verifica se il percorso SSD esiste
        ssd_base = Path(ssd_path)
        if not ssd_base.exists():
            print(f"⚠️ Percorso SSD non esiste: {ssd_path}")
            create_ssd = input("🔧 Creare la directory? (y/n): ").strip().lower()
            if create_ssd == 'y':
                ssd_base.mkdir(parents=True, exist_ok=True)
                print(f"✅ Directory SSD creata: {ssd_path}")
        
        return config_filename
        
    except Exception as e:
        print(f"❌ Errore creazione file: {e}")
        return None


def setup_ssd_directories(config):
    """💾 Configura directory SSD"""
    print("\n💾 CONFIGURAZIONE DIRECTORY SSD...")
    
    try:
        storage_config = config.get('storage', {})
        
        # Directory principali da creare
        directories_to_create = [
            storage_config.get('base_directory', 'D:/CryptoSystem'),
            storage_config.get('database', {}).get('backup_directory', ''),
            storage_config.get('database', {}).get('archive_directory', ''),
            storage_config.get('cache', {}).get('cache_directory', ''),
            storage_config.get('cache', {}).get('api_cache_directory', ''),
            storage_config.get('cache', {}).get('ml_cache_directory', ''),
            storage_config.get('cache', {}).get('bitcoin_cache_directory', ''),
            storage_config.get('logs', {}).get('logs_directory', ''),
            storage_config.get('logs', {}).get('verification_logs', ''),
            storage_config.get('logs', {}).get('performance_logs', ''),
            storage_config.get('data', {}).get('export_directory', ''),
            storage_config.get('data', {}).get('reports_directory', ''),
            storage_config.get('data', {}).get('charts_directory', '')
        ]
        
        created_count = 0
        for directory in directories_to_create:
            if directory:
                dir_path = Path(directory)
                if not dir_path.exists():
                    try:
                        dir_path.mkdir(parents=True, exist_ok=True)
                        print(f"   ✅ Creata: {directory}")
                        created_count += 1
                    except Exception as e:
                        print(f"   ❌ Errore creando {directory}: {e}")
                else:
                    print(f"   📁 Esiste: {directory}")
        
        # Crea anche la directory del database se necessario
        db_path = storage_config.get('database', {}).get('main_db_path', '')
        if db_path:
            db_dir = Path(db_path).parent
            if not db_dir.exists():
                db_dir.mkdir(parents=True, exist_ok=True)
                print(f"   ✅ Creata directory database: {db_dir}")
                created_count += 1
        
        print(f"✅ Setup SSD completato: {created_count} directory create")
        
        # Verifica spazio disponibile
        base_dir = Path(storage_config.get('base_directory', 'D:/CryptoSystem'))
        if base_dir.exists():
            try:
                disk_usage = psutil.disk_usage(str(base_dir))
                free_gb = disk_usage.free / (1024**3)
                total_gb = disk_usage.total / (1024**3)
                used_percent = (disk_usage.used / disk_usage.total) * 100
                
                print(f"💾 Spazio SSD: {free_gb:.1f} GB liberi / {total_gb:.1f} GB totali ({used_percent:.1f}% usato)")
                
                if free_gb < 50:
                    print(f"⚠️ ATTENZIONE: Poco spazio libero su SSD ({free_gb:.1f} GB)")
                elif free_gb < 20:
                    print(f"🚨 ERRORE: Spazio insufficiente su SSD ({free_gb:.1f} GB)")
                    return False
                    
            except Exception as e:
                print(f"⚠️ Impossibile verificare spazio SSD: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore setup SSD: {e}")
        return False

class OptimizedCoinGeckoAPI:
    """🦎 API CoinGecko ottimizzata per sistema SSD con config JSON"""
    
    def __init__(self, config):
        self.config = config
        self.api_key = config.get('coingecko_api_key', '')
        self.base_url = "https://api.coingecko.com/api/v3"
        
        # Rate limiting da JSON
        self.max_calls_per_day = config.get('max_api_calls_per_day', 1000)
        self.max_calls_per_hour = config.get('max_api_calls_per_hour', 100)
        self.api_delay = config.get('api_delay', 2.0)
        self.retry_delay = config.get('retry_delay', 3.0)
        self.max_retries = config.get('max_retries', 5)
        self.timeout = config.get('timeout', 30)
        
        # Tracking
        self.daily_calls = 0
        self.hourly_calls = 0
        self.last_call_date = datetime.now().date()
        self.last_call_hour = datetime.now().hour
        self.last_call_time = 0
        self.api_errors = 0
        
        # Cache ottimizzata per SSD
        self.cache_duration = config.get('cache_duration_hours', 48) * 3600
        self.cache_max_size = config.get('cache_max_size', 1000)
        self.cache = {}
        
        # SSD cache directory dal JSON config
        self.cache_directory = Path(config.get('cache_directory', 'D:/CryptoSystem/cache/api'))
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        
        print(f"🦎 Optimized CoinGecko API inizializzata")
        print(f"📊 Rate limit: {self.max_calls_per_day}/day, {self.max_calls_per_hour}/hour")
        print(f"💾 Cache SSD: {self.cache_directory}")
        print(f"⏱️ Cache duration: {self.cache_duration//3600}h, Max size: {self.cache_max_size}")
    
    def _reset_counters(self):
        """🔄 Reset API counters safely - FIXED VERSION"""
        try:
            import time
            from datetime import datetime
            
            current_time = time.time()
            
            # Initialize if not exists
            if not hasattr(self, 'last_reset_time'):
                self.last_reset_time = current_time
                self.hourly_calls = 0
                self.daily_calls = 0
                print(f"   🔄 Counters initialized")
                return
            
            # Check if we need hourly reset
            hours_since_last_reset = (current_time - self.last_reset_time) / 3600
            if hours_since_last_reset >= 1.0:
                old_hourly = getattr(self, 'hourly_calls', 0)
                self.hourly_calls = 0
                self.last_reset_time = current_time
                print(f"   🔄 Hourly counter reset (was {old_hourly})")
            
            # Check if we need daily reset (24 hours)
            if not hasattr(self, 'last_daily_reset'):
                self.last_daily_reset = current_time
                
            hours_since_daily_reset = (current_time - self.last_daily_reset) / 3600
            if hours_since_daily_reset >= 24.0:
                old_daily = getattr(self, 'daily_calls', 0)
                self.daily_calls = 0
                self.last_daily_reset = current_time
                print(f"   🔄 Daily counter reset (was {old_daily})")
            
            # Ensure counters exist
            if not hasattr(self, 'hourly_calls'):
                self.hourly_calls = 0
            if not hasattr(self, 'daily_calls'):
                self.daily_calls = 0
                
        except Exception as e:
            print(f"   ❌ Counter reset failed: {e}")
            # Initialize with safe defaults
            self.hourly_calls = 0
            self.daily_calls = 0
            self.last_reset_time = time.time()
            self.last_daily_reset = time.time()
    
    def _save_ssd_cache(self, cache_key, data):
        """💾 Save cache to SSD - FIXED VERSION"""
        try:
            if isinstance(data, pd.DataFrame):
                # Use pickle for DataFrames (handles Timestamps perfectly)
                cache_file = self.cache_directory / f"{cache_key}.pkl"
                data.to_pickle(cache_file)
            else:
                # Use JSON for simple data
                cache_file = self.cache_directory / f"{cache_key}.json"
                with open(cache_file, 'w') as f:
                    json.dump(data, f, default=str, indent=2)
            
            self._cleanup_old_cache_files()
            
        except Exception as e:
            print(f"     ⚠️ SSD cache save error: {e}")

    def _load_ssd_cache(self, cache_key):
        """💾 Load cache from SSD - FIXED VERSION"""
        try:
            # Try pickle first
            pickle_file = self.cache_directory / f"{cache_key}.pkl"
            json_file = self.cache_directory / f"{cache_key}.json"
            
            # Get cache duration properly
            try:
                cache_duration_seconds = self.cache_duration
            except Exception:
                # Fallback if property access fails
                cache_duration_seconds = self.config.get('api', {}).get('cache_duration_hours', 48) * 3600
            
            for cache_file in [pickle_file, json_file]:
                if cache_file.exists():
                    try:
                        cache_time = cache_file.stat().st_mtime
                        current_time = time.time()
                        
                        # FIXED: Proper comparison with explicit types
                        time_diff = current_time - cache_time
                        
                        if time_diff < cache_duration_seconds:
                            if cache_file.suffix == '.pkl':
                                data = pd.read_pickle(cache_file)
                            else:
                                with open(cache_file, 'r') as f:
                                    data = json.load(f)
                            return True, data
                        else:
                            # Cache expired, remove file
                            cache_file.unlink()
                            
                    except Exception as file_error:
                        print(f"     ⚠️ Cache file error {cache_file}: {file_error}")
                        continue
            
            return False, None
            
        except Exception as e:
            print(f"     ⚠️ SSD cache load error: {e}")
            return False, None
    
    def _cleanup_old_cache_files(self):
        """🧹 Cleanup old cache files - FIXED VERSION"""
        try:
            cache_files = list(self.cache_directory.glob("*.json")) + list(self.cache_directory.glob("*.pkl"))
            max_files = 1000
            
            if len(cache_files) > max_files:
                # Sort by modification time and remove oldest
                cache_files.sort(key=lambda x: x.stat().st_mtime)
                for old_file in cache_files[:len(cache_files) - max_files]:
                    old_file.unlink()
                    
        except Exception as e:
            print(f"     ⚠️ Cache cleanup error: {e}")

    def _should_use_cache(self, cache_key):
        """💾 Check if should use cache - FIXED VERSION"""
        try:
            # Get cache duration safely
            try:
                cache_duration_seconds = self.cache_duration
            except Exception:
                cache_duration_seconds = self.config.get('api', {}).get('cache_duration_hours', 48) * 3600
            
            # Check memory cache first
            if hasattr(self, 'cache') and cache_key in self.cache:
                cache_time, cache_data = self.cache[cache_key]
                time_diff = time.time() - cache_time
                
                if time_diff < cache_duration_seconds:
                    return True, cache_data
                else:
                    # Remove expired memory cache
                    del self.cache[cache_key]
            
            # Check SSD cache if available
            if hasattr(self, '_load_ssd_cache'):
                try:
                    use_ssd, ssd_data = self._load_ssd_cache(cache_key)
                    if use_ssd:
                        # Load into memory cache
                        if not hasattr(self, 'cache'):
                            self.cache = {}
                        self.cache[cache_key] = (time.time(), ssd_data)
                        return True, ssd_data
                except Exception as e:
                    print(f"     ⚠️ SSD cache error: {e}")
            
            # No valid cache found
            return False, None
            
        except Exception as e:
            print(f"     ⚠️ Cache check error: {e}")
            return False, None
    
    def _cache_response(self, cache_key, data):
        """💾 Cache API response (memory + SSD)"""
        # Memory cache
        self.cache[cache_key] = (time.time(), data)
        
        # SSD cache
        self._save_ssd_cache(cache_key, data)
        
        # Cleanup memory cache if too large
        if len(self.cache) > 200:  # Keep smaller memory cache
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][0])
            del self.cache[oldest_key]


    def safe_api_call(self, endpoint, params=None, description="API call", allow_fallback=True):
        """🛡️ Safe API call con retry aggressivo per free tier"""
        
        if params is None:
            params = {}
        
        # Configurazione safe mode
        safe_mode = self.config.get('safe_mode', True)
        use_free_tier = self.config.get('use_free_tier', True)
        safe_delay = self.config.get('safe_delay', 6.0)
        max_retries = self.config.get('max_retries', 7)
        
        print(f"🛡️ Safe API call: {description}")
        print(f"🔍 DEBUG: Before _can_make_safe_call() - attributes check")
        print(f"   last_reset_time exists: {hasattr(self, 'last_reset_time')}")
        print(f"   hourly_calls exists: {hasattr(self, 'hourly_calls')}")
        print(f"   daily_calls exists: {hasattr(self, 'daily_calls')}")
        # Rate limiting più conservativo per free tier
        if not self._can_make_safe_call():
            print(f"🚫 Rate limit reached - skipping {description}")
            return None
        
        # Lista di endpoints alternativi se disponibili
        endpoints_to_try = [endpoint]
        if allow_fallback and self.config.get('fallback_endpoints', True):
            endpoints_to_try.extend(self._get_fallback_endpoints(endpoint))
        
        for endpoint_url in endpoints_to_try:
            for attempt in range(max_retries):
                try:
                    # Delay più lungo per free tier
                    if attempt > 0:
                        delay = safe_delay * (attempt + 1) + random.uniform(1, 3)
                        print(f"     ⏳ Safe wait: {delay:.1f}s (attempt {attempt + 1})")
                        time.sleep(delay)
                    elif safe_mode and not self.api_key:
                        # Delay base per free tier
                        time.sleep(safe_delay)
                    
                    # Headers ottimizzati per free tier
                    headers = self._get_safe_headers()
                    
                    print(f"     🌐 Trying: {endpoint_url} (attempt {attempt + 1}/{max_retries})")
                    
                    response = requests.get(
                        endpoint_url, 
                        params=params, 
                        headers=headers,
                        timeout=self.config.get('timeout', 45)
                    )
                    
                    self.daily_calls += 1
                    self.hourly_calls += 1
                    self.last_call_time = time.time()
                    
                    if response.status_code == 200:
                        data = response.json()
                        print(f"     ✅ {description} successful")
                        return data
                        
                    elif response.status_code == 429:
                        wait_time = self._calculate_rate_limit_wait(attempt)
                        print(f"     ⏰ Rate limited - waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                        
                    elif response.status_code in [503, 502, 504]:
                        print(f"     🔄 Server error {response.status_code} - retrying...")
                        continue
                        
                    elif response.status_code == 404 and len(endpoints_to_try) > 1:
                        print(f"     🔍 Endpoint not found - trying fallback...")
                        break  # Try next endpoint
                        
                    else:
                        print(f"     ⚠️ HTTP {response.status_code}: {response.text[:100]}")
                        if attempt < max_retries - 1:
                            continue
                    
                except requests.exceptions.Timeout:
                    print(f"     ⏰ Timeout - extending wait time...")
                    continue
                    
                except requests.exceptions.ConnectionError:
                    print(f"     🌐 Connection error - retrying with longer delay...")
                    time.sleep(safe_delay * 2)
                    continue
                    
                except Exception as e:
                    print(f"     ❌ Unexpected error: {e}")
                    if attempt < max_retries - 1:
                        continue
        
        print(f"     ❌ All attempts failed for {description}")
        self.api_errors += 1
        return None

    def _can_make_safe_call(self):
        """🛡️ Check if we can make a safe API call"""
        self._reset_counters()
        
        # Limiti più conservativi per free tier
        max_hourly = self.config.get('max_api_calls_per_hour', 15)
        max_daily = self.config.get('max_api_calls_per_day', 500)
        
        if self.hourly_calls >= max_hourly:
            print(f"🚫 Hourly limit reached: {self.hourly_calls}/{max_hourly}")
            return False
            
        if self.daily_calls >= max_daily:
            print(f"🚫 Daily limit reached: {self.daily_calls}/{max_daily}")
            return False
        
        return True

    def _get_safe_headers(self):
        """🛡️ Get safe headers for API calls"""
        headers = {
            'Accept': 'application/json',
            'User-Agent': 'CryptoAnalysis/1.0 (Conservative)',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        }
        
        # Solo aggiungere API key se disponibile
        if self.api_key:
            headers['x-cg-demo-api-key'] = self.api_key
        
        return headers

    def _calculate_rate_limit_wait(self, attempt):
        """⏰ Calculate wait time for rate limiting"""
        base_wait = self.config.get('retry_delay', 5.0)
        
        # Exponential backoff più aggressivo
        wait_time = base_wait * (2 ** attempt) + random.uniform(2, 8)
        
        # Cap massimo per evitare attese troppo lunghe
        return min(wait_time, 120)  # Max 2 minuti

    def _get_fallback_endpoints(self, original_endpoint):
        """🔄 Get fallback endpoints if available"""
        fallbacks = []
        
        # Fallback per market data
        if 'market_chart' in original_endpoint:
            fallbacks.append(original_endpoint.replace('market_chart', 'ohlc'))
        
        # Fallback per markets
        if 'coins/markets' in original_endpoint:
            fallbacks.append(original_endpoint.replace('per_page=250', 'per_page=100'))
            fallbacks.append(original_endpoint.replace('per_page=100', 'per_page=50'))
        
        return fallbacks

    def get_top_cryptos_safe(self, limit=750):
        """🏆 Get top cryptos with SAFE API calls"""
        cache_key = f"top_cryptos_safe_{limit}"
        
        use_cache, cached_data = self._should_use_cache(cache_key)
        if use_cache:
            print(f"💾 Using cached top cryptos (safe mode)")
            return cached_data
        
        # Dividi in chiamate più piccole per free tier
        per_page = 100  # Più conservativo
        all_cryptos = []
        
        for page in range(1, (limit // per_page) + 2):
            if len(all_cryptos) >= limit:
                break
                
            endpoint = f"{self.base_url}/coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': per_page,
                'page': page,
                'sparkline': False
            }
            
            data = self.safe_api_call(
                endpoint, 
                params, 
                f"Top cryptos page {page}",
                allow_fallback=True
            )
            
            if data:
                cryptos_page = [
                    (crypto['id'], crypto['name'], crypto['market_cap_rank'])
                    for crypto in data if crypto.get('market_cap_rank')
                ]
                all_cryptos.extend(cryptos_page)
                
                # Delay extra tra pagine per free tier
                if page < (limit // per_page) + 1:
                    time.sleep(self.config.get('safe_delay', 6.0))
            else:
                break
        
        result = all_cryptos[:limit]
        if result:
            self._cache_response(cache_key, result)
            print(f"✅ Fetched {len(result)} cryptos safely")
        
        return result
    
    # Aggiungi questi metodi alla classe OptimizedCoinGeckoAPI nel file crypto_continuous_optimized.py
    def clear_old_cache(self, force_all=False):
        """🧹 Clear old cache files and expired entries"""
        try:
            print("🧹 Clearing old cache...")
            
            cleared_memory = 0
            cleared_ssd = 0
            
            # 1. Clear memory cache
            if hasattr(self, 'cache') and self.cache:
                if force_all:
                    # Clear all memory cache
                    cleared_memory = len(self.cache)
                    self.cache.clear()
                    print(f"   💾 Cleared all memory cache: {cleared_memory} entries")
                else:
                    # Clear only expired memory cache
                    cache_duration = self.config.get('api', {}).get('cache_duration_hours', 48) * 3600
                    current_time = time.time()
                    
                    expired_keys = []
                    for key, (cache_time, data) in list(self.cache.items()):
                        if (current_time - cache_time) > cache_duration:
                            expired_keys.append(key)
                    
                    for key in expired_keys:
                        del self.cache[key]
                        
                    cleared_memory = len(expired_keys)
                    print(f"   💾 Cleared expired memory cache: {cleared_memory} entries")
            
            # 2. Clear SSD cache files
            if hasattr(self, 'cache_directory'):
                cache_dir = self.cache_directory if hasattr(self, 'cache_directory') else Path('cache/api')
                
                if cache_dir.exists():
                    cache_files = list(cache_dir.glob("*.json")) + list(cache_dir.glob("*.pkl"))
                    
                    if force_all:
                        # Remove all cache files
                        for cache_file in cache_files:
                            try:
                                cache_file.unlink()
                                cleared_ssd += 1
                            except Exception as e:
                                print(f"     ⚠️ Could not remove {cache_file}: {e}")
                        
                        print(f"   💾 Cleared all SSD cache: {cleared_ssd} files")
                    else:
                        # Remove only expired cache files
                        cache_duration = self.config.get('api', {}).get('cache_duration_hours', 48) * 3600
                        current_time = time.time()
                        
                        for cache_file in cache_files:
                            try:
                                file_age = current_time - cache_file.stat().st_mtime
                                if file_age > cache_duration:
                                    cache_file.unlink()
                                    cleared_ssd += 1
                            except Exception as e:
                                print(f"     ⚠️ Could not process {cache_file}: {e}")
                        
                        print(f"   💾 Cleared expired SSD cache: {cleared_ssd} files")
            
            # 3. Additional cleanup operations
            if force_all:
                # Reset API call counters
                if hasattr(self, 'daily_calls'):
                    self.daily_calls = 0
                if hasattr(self, 'hourly_calls'):
                    self.hourly_calls = 0
                print("   📊 Reset API call counters")
            
            print(f"✅ Cache cleanup completed: {cleared_memory} memory + {cleared_ssd} SSD entries cleared")
            
            return {
                'memory_cleared': cleared_memory,
                'ssd_cleared': cleared_ssd,
                'total_cleared': cleared_memory + cleared_ssd
            }
            
        except Exception as e:
            print(f"❌ Cache cleanup failed: {e}")
            return {
                'memory_cleared': 0,
                'ssd_cleared': 0,
                'total_cleared': 0,
                'error': str(e)
            }

    def get_api_stats(self):
        """📊 Get comprehensive API statistics"""
        try:
            stats = {
                'daily_calls': getattr(self, 'daily_calls', 0),
                'hourly_calls': getattr(self, 'hourly_calls', 0),
                'max_calls_per_day': self.config.get('api', {}).get('max_api_calls_per_day', 1000),
                'max_calls_per_hour': self.config.get('api', {}).get('max_api_calls_per_hour', 50),
                'cache_hits': 0,
                'cache_size': 0,
                'api_errors': getattr(self, 'api_errors', 0),
                'last_api_call_time': 'Never'
            }
            
            # Memory cache stats
            if hasattr(self, 'cache') and self.cache:
                stats['cache_size'] = len(self.cache)
                # Estimate cache hits (this is approximate)
                stats['cache_hits'] = getattr(self, 'cache_hits_counter', 0)
            
            # SSD cache stats
            if hasattr(self, 'cache_directory') and self.cache_directory.exists():
                cache_files = list(self.cache_directory.glob("*.json")) + list(self.cache_directory.glob("*.pkl"))
                stats['ssd_cache_files'] = len(cache_files)
                
                # Calculate total cache size
                total_size = sum(f.stat().st_size for f in cache_files) / (1024 * 1024)  # MB
                stats['ssd_cache_size_mb'] = round(total_size, 2)
            
            # Last API call time
            if hasattr(self, 'last_api_call') and self.last_api_call:
                stats['last_api_call_time'] = datetime.fromtimestamp(self.last_api_call).strftime('%Y-%m-%d %H:%M:%S')
            
            return stats
            
        except Exception as e:
            print(f"❌ Error getting API stats: {e}")
            return {
                'daily_calls': 0,
                'hourly_calls': 0,
                'max_calls_per_day': 1000,
                'max_calls_per_hour': 50,
                'cache_hits': 0,
                'cache_size': 0,
                'api_errors': 0,
                'last_api_call_time': 'Error',
                'error': str(e)
            }

    def get_cache_info(self):
        """💾 Get detailed cache information"""
        try:
            info = {
                'memory_cache': {
                    'enabled': hasattr(self, 'cache'),
                    'entries': len(self.cache) if hasattr(self, 'cache') else 0,
                    'max_size': getattr(self, 'cache_max_size', 1000)
                },
                'ssd_cache': {
                    'enabled': hasattr(self, 'cache_directory'),
                    'directory': str(self.cache_directory) if hasattr(self, 'cache_directory') else 'Not set',
                    'files': 0,
                    'size_mb': 0
                },
                'configuration': {
                    'cache_duration_hours': self.config.get('api', {}).get('cache_duration_hours', 48),
                    'cleanup_threshold': 1000
                }
            }
            
            # SSD cache details
            if hasattr(self, 'cache_directory') and self.cache_directory.exists():
                cache_files = list(self.cache_directory.glob("*.json")) + list(self.cache_directory.glob("*.pkl"))
                info['ssd_cache']['files'] = len(cache_files)
                
                total_size = sum(f.stat().st_size for f in cache_files) / (1024 * 1024)  # MB
                info['ssd_cache']['size_mb'] = round(total_size, 2)
            
            return info
            
        except Exception as e:
            return {'error': str(e)}
    
class GracefulShutdownHandler:
    """🛑 Gestore per shutdown graceful con invio pending alerts"""
    
    def __init__(self, crypto_system):
        self.crypto_system = crypto_system
        self.shutdown_requested = False
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print("🛑 Graceful shutdown handler registered (Ctrl+C)")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        print(f"\n🛑 {signal_name} received - initiating graceful shutdown...")
        
        self.shutdown_requested = True
        self.crypto_system.running = False
        
        # Send pending alerts before shutdown
        self._send_pending_alerts_on_shutdown()
        
        # Cleanup and exit
        self._cleanup_and_exit()
    
    def _send_pending_alerts_on_shutdown(self):
        """📧 Invia pending alerts prima dello shutdown"""
        try:
            print("📧 Checking for pending alerts before shutdown...")
            
            if not hasattr(self.crypto_system, 'notifier') or not self.crypto_system.notifier:
                print("   ⚠️ No notifier available - skipping alert sending")
                return
            
            # Check if there are pending alerts
            pending_count = 0
            
            # Method 1: Check high/medium priority alerts lists
            if hasattr(self.crypto_system.notifier, 'high_priority_alerts'):
                pending_count += len(self.crypto_system.notifier.high_priority_alerts or [])
            if hasattr(self.crypto_system.notifier, 'medium_priority_alerts'):
                pending_count += len(self.crypto_system.notifier.medium_priority_alerts or [])
            
            # Method 2: Check pending_alerts dict format
            if hasattr(self.crypto_system.notifier, 'pending_alerts'):
                for horizon in ['1d', '3d']:
                    for priority in ['high', 'medium', 'watch']:
                        alerts = self.crypto_system.notifier.pending_alerts.get(horizon, {}).get(priority, [])
                        pending_count += len(alerts)
            
            print(f"   📊 Found {pending_count} pending alerts")
            
            if pending_count > 0:
                print("   📧 Sending final summary email...")
                
                # Force send summary regardless of timing
                if hasattr(self.crypto_system.notifier, 'send_6hour_dual_summary_fixed'):
                    # Force send by resetting last summary time
                    self.crypto_system.notifier.last_summary_sent = None
                    success = self.crypto_system.notifier.send_6hour_dual_summary_fixed()
                    
                    if success:
                        print("   ✅ Final alert summary sent successfully")
                    else:
                        print("   ❌ Failed to send final alert summary")
                
                elif hasattr(self.crypto_system.notifier, 'send_6hour_dual_summary'):
                    self.crypto_system.notifier.last_summary_sent = None
                    success = self.crypto_system.notifier.send_6hour_dual_summary()
                    
                    if success:
                        print("   ✅ Final alert summary sent successfully")
                    else:
                        print("   ❌ Failed to send final alert summary")
                
                else:
                    print("   ⚠️ No compatible send method found in notifier")
            
            else:
                print("   ℹ️ No pending alerts to send")
        
        except Exception as e:
            print(f"   ❌ Error sending pending alerts: {e}")
            import traceback
            traceback.print_exc()
    
    def _cleanup_and_exit(self):
        """🧹 Cleanup finale prima dell'uscita"""
        try:
            print("🧹 Performing final cleanup...")
            
            # Set running flag to False
            if hasattr(self.crypto_system, 'running'):
                self.crypto_system.running = False
            
            # Call system cleanup if available
            if hasattr(self.crypto_system, '_cleanup_optimized'):
                self.crypto_system._cleanup_optimized()
            elif hasattr(self.crypto_system, 'cleanup'):
                self.crypto_system.cleanup()
            
            # Close database connections
            if hasattr(self.crypto_system, 'database') and self.crypto_system.database:
                try:
                    self.crypto_system.database.close()
                    print("   ✅ Database connections closed")
                except:
                    pass
            
            # Memory cleanup
            import gc
            gc.collect()
            
            print("👋 Graceful shutdown completed")
            print(f"⏰ Shutdown time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")
        
        finally:
            sys.exit(0)

class OptimizedCryptoContinuousSystemSSD:
    """🚀 Sistema Crypto Continuo OTTIMIZZATO con SSD esterno"""
    
    def __init__(self, config=None, config_file=None):
        """🚀 Initialize Optimized Crypto Continuous System with SSD storage"""
        print("🚀 OPTIMIZED CRYPTO CONTINUOUS SYSTEM - SSD VERSION")
        print("=" * 80)
        print("💾 STORAGE: SSD esterno (D:) per performance ottimali")
        print("🎯 LOOKBACK: 6-9 mesi adaptive (180d standard, 270d premium)")
        print("🧠 ML MODELS: Traditional + CatBoost + LSTM optimized")
        print("🔍 VERIFICATION: Complete with model feedback loop")
        print("📧 NOTIFICATIONS: Optimized with quality filters")
        print("💾 DATABASE: Enhanced tracking and performance")
        print("🟠 BITCOIN: Real benchmark integration")
        print("📊 CONFIG: Tutti i thresholds dal JSON")
        print("=" * 80)
        
        # Load configuration con prompt interattivo
        self.config = self._load_config_with_prompt(config, config_file)
        
        # Use the actual config file that was loaded
        self.config_file_path = getattr(self, '_actual_config_file', config_file)

        print(f"🔍 Using config file: {self.config_file_path}")

        # Setup SSD directories
        if not setup_ssd_directories(self.config):
            print("❌ Setup SSD fallito")
            sys.exit(1)
        
        # System state
        self.running = False
        self.paused = False
        self.cycle_count = 0
        self.start_time = None
        
        # Statistics tracking
        self.stats = {
            'total_cryptos_analyzed': 0,
            'successful_predictions_1d': 0,
            'successful_predictions_3d': 0,
            'high_confidence_predictions': 0,
            'filtered_predictions': 0,
            'model_feedback_updates': 0,
            'verification_completed': 0,
            'api_calls_made': 0,
            'start_time': None
        }
        
        # Initialize core components FIRST (this sets db_path and other essentials)
        self._initialize_optimized_components()
        
        # Ensure critical paths are set
        if not hasattr(self, 'db_path') or not self.db_path:
            self.db_path = self.get_storage_path('database', 'main_db_path', 
                                                'D:/CryptoSystem/database/crypto_optimized_6_9months.db')
            print(f"✅ Database path set: {self.db_path}")
        
        # Setup signal handlers and graceful shutdown EARLY
        self._setup_signal_handlers()
        self.shutdown_handler = None
        self._setup_graceful_shutdown()
        
        print(f"✅ Core system initialized")
        print(f"💾 Storage base: {self.config.get('storage', {}).get('base_directory', 'D:/CryptoSystem')}")
        print(f"🎯 Accuracy target: 1d≥{self.get_threshold('accuracy_targets', 'accuracy_target_1d', 0.65):.0%}, 3d≥{self.get_threshold('accuracy_targets', 'accuracy_target_3d', 0.60):.0%}")
        
        # Initialize Verification System
        if VERIFICATION_SYSTEM_AVAILABLE and self.get_threshold('verification', 'enabled', True):
            try:
                verification_config = {
                    'gmail_user': self.get_threshold('email', 'gmail_user', ''),
                    'gmail_app_password': self.get_threshold('email', 'gmail_app_password', ''),
                    'accuracy_target_1d': self.get_threshold('accuracy_targets', 'accuracy_target_1d', 0.65),
                    'accuracy_target_3d': self.get_threshold('accuracy_targets', 'accuracy_target_3d', 0.60),
                    'rate_limit_delay': self.get_threshold('api', 'api_delay', 2.0),
                    'max_retries_api': self.get_threshold('api', 'max_retries', 3),
                    # Model feedback config
                    'model_feedback_learning_rate': self.get_threshold('model_feedback', 'model_feedback_learning_rate', 0.1),
                    'model_feedback_momentum': self.get_threshold('model_feedback', 'model_feedback_momentum', 0.9),
                    'min_samples_feedback': self.get_threshold('model_feedback', 'min_samples_feedback', 10),
                    'performance_window_days': self.get_threshold('model_feedback', 'performance_window_days', 30),
                }
                
                self.verification_system = OptimizedVerificationSystem(self.db_path, verification_config)
                
                # Integrate with ML system if available
                if hasattr(self, 'ml_system') and self.ml_system:
                    self.verification_system.integrate_model_feedback_system(self.ml_system)
                    self.ml_system.set_feedback_optimizer(self.verification_system.feedback_optimizer)
                    print("🔗 ML system linked with verification feedback")
                
                print("✅ Verification System initialized")
                
            except Exception as e:
                print(f"⚠️ Verification System initialization failed: {e}")
                self.verification_system = None
        else:
            print("⚠️ Verification System disabled or not available")
            self.verification_system = None
        
        # Initialize Signal Generator (needs db_path and other components ready)
        signal_success = self._initialize_signal_generator_robust(self.config_file_path)
        
        # Display final system capabilities
        capabilities = self._get_signal_generator_capabilities()
        
        print(f"\n📊 SIGNAL GENERATOR STATUS:")
        print(f"   🎯 Signal Generation: {'✅' if capabilities['signal_generation'] else '❌'}")
        print(f"   📄 Basic Export: {'✅' if capabilities['basic_export'] else '❌'}")
        print(f"   📁 Advanced Export: {'✅' if capabilities['advanced_export'] else '❌'}")
        print(f"   🔍 Verification Data: {'✅' if capabilities['verification_integration'] else '❌'}")
        print(f"   🚀 Overall Status: {capabilities['status'].upper()}")
        
        if not signal_success:
            print("💡 System will continue with prediction-only mode")
            print("   All ML predictions will work normally")
            print("   Only signal generation features will be unavailable")
        
        print(f"\n✅ Sistema ottimizzato SSD completamente inizializzato")


    def _initialize_signal_generator_robust(self, config_file):
        """🛡️ Initialize signal generator with comprehensive error handling and validation"""
        
        # Reset signal generator state
        self.signal_generator = None
        self.advanced_export = None
        
        # Validate prerequisites
        prerequisites_check = self._validate_signal_generator_prerequisites(config_file)
        if not prerequisites_check['valid']:
            print(f"❌ Signal Generator prerequisites not met:")
            for issue in prerequisites_check['issues']:
                print(f"   • {issue}")
            return False
        
        # Step 1: Import required modules with detailed error handling
        try:
            print("📦 Importing Signal Generator modules...")
            from crypto_signal_generator_integrated import CryptoSignalGeneratorIntegrated
            print("✅ Signal Generator module imported successfully")
            signal_module_available = True
        except ImportError as e:
            print(f"❌ Signal Generator module import failed: {e}")
            print("💡 Check if crypto_signal_generator_integrated.py exists in project directory")
            print("💡 Ensure all dependencies are installed")
            return False
        except Exception as e:
            print(f"❌ Unexpected error importing Signal Generator: {e}")
            return False
        
        try:
            from advanced_json_export_system import integrate_advanced_json_export
            print("✅ Advanced JSON Export module imported successfully")
            advanced_export_available = True
        except ImportError as e:
            print(f"⚠️ Advanced Export module not found: {e}")
            print("💡 Will use basic JSON export only")
            advanced_export_available = False
        except Exception as e:
            print(f"⚠️ Unexpected error importing Advanced Export: {e}")
            advanced_export_available = False
        
        # Step 2: Initialize Signal Generator with validation
        try:
            print("🔧 Initializing Signal Generator...")
            print(f"   Config file: {config_file}")
            print(f"   Database: {self.db_path}")
            
            self.signal_generator = CryptoSignalGeneratorIntegrated(
                config_path=config_file,
                database_path=self.db_path
            )
            
            # Validate signal generator was created successfully
            if not self.signal_generator:
                raise Exception("Signal Generator instance is None after initialization")
            
            print("✅ Signal Generator core initialized successfully")
            
            # Add comprehensive stats tracking
            self.stats.update({
                'signals_generated': 0,
                'last_signals_generated': 0,
                'last_signal_timestamp': None,
                'signal_generation_errors': 0,
                'signal_export_count': 0
            })
            
            print("✅ Signal Generator statistics tracking initialized")
            
        except FileNotFoundError as e:
            print(f"❌ Signal Generator initialization failed - File not found: {e}")
            print(f"🔍 Check if config file exists: {config_file}")
            print(f"🔍 Check if database file exists: {self.db_path}")
            return False
        except Exception as e:
            print(f"❌ Signal Generator initialization failed: {e}")
            print(f"🔍 Config file: {config_file}")
            print(f"🔍 Database: {self.db_path}")
            # Print additional diagnostic info
            import traceback
            print(f"🔍 Error traceback:")
            traceback.print_exc()
            return False
        
        # Step 3: Integrate Advanced Export System (if available)
        if advanced_export_available and self.signal_generator:
            try:
                print("🔧 Integrating Advanced JSON Export System...")
                integrate_advanced_json_export(self.signal_generator)
                
                # Validate integration worked
                if hasattr(self.signal_generator, 'advanced_export'):
                    self.advanced_export = self.signal_generator.advanced_export
                    self.stats['json_exports_created'] = 0
                    print("✅ Advanced JSON Export integrated successfully")
                    
                    # Test export system capabilities
                    try:
                        export_summary = self.advanced_export.get_export_summary()
                        print(f"   📁 Export directory ready: {export_summary.get('export_directory', 'Unknown')}")
                    except:
                        print("⚠️ Export summary test failed, but integration appears successful")
                else:
                    print("⚠️ Advanced Export integration incomplete - missing advanced_export attribute")
                    advanced_export_available = False
                    
            except Exception as e:
                print(f"⚠️ Advanced Export integration failed: {e}")
                print("💡 Signal Generator will use basic export functionality only")
                advanced_export_available = False
                import traceback
                traceback.print_exc()
        
        # Step 4: Validate signal generator functionality
        try:
            print("🧪 Testing Signal Generator functionality...")
            
            # Test basic methods exist
            required_methods = ['generate_signals', 'get_recent_predictions', 'get_crypto_historical_accuracy']
            missing_methods = []
            
            for method in required_methods:
                if not hasattr(self.signal_generator, method):
                    missing_methods.append(method)
            
            if missing_methods:
                print(f"⚠️ Signal Generator missing required methods: {missing_methods}")
                print("💡 Signal Generator may have limited functionality")
            else:
                print("✅ All required Signal Generator methods available")
                
            # Test database connectivity
            try:
                test_predictions = self.signal_generator.get_recent_predictions(hours_back=1)
                print(f"✅ Database connectivity test passed ({len(test_predictions) if test_predictions else 0} recent predictions)")
            except Exception as db_error:
                print(f"⚠️ Database connectivity test failed: {db_error}")
                print("💡 Signal Generator database access may be limited")
        
        except Exception as e:
            print(f"⚠️ Signal Generator functionality test failed: {e}")
            print("💡 Basic initialization succeeded, but some features may not work")
        
        # Step 5: Final status report
        if self.signal_generator:
            export_status = "Advanced" if advanced_export_available and self.advanced_export else "Basic"
            verification_status = "Integrated" if hasattr(self.signal_generator, 'get_crypto_historical_accuracy') else "Limited"
            
            print(f"🚀 Signal Generator initialization COMPLETED")
            print(f"   📊 Export System: {export_status}")
            print(f"   🔍 Verification Integration: {verification_status}")
            print(f"   💾 Database: Connected")
            print(f"   ⚙️ Configuration: Loaded from {config_file}")
            
            return True
        else:
            print("❌ Signal Generator initialization FAILED")
            print("💡 System will continue without signal generation capabilities")
            return False


    def _get_signal_generator_capabilities(self):
        """📊 Comprehensive analysis of current signal generator capabilities"""
        
        capabilities = {
            'signal_generation': False,
            'basic_export': False,
            'advanced_export': False,
            'verification_integration': False,
            'database_connectivity': False,
            'configuration_loaded': False,
            'statistics_tracking': False,
            'status': 'disabled',
            'features': [],
            'limitations': [],
            'recommendations': []
        }
        
        # Check if signal generator exists and is properly initialized
        if not hasattr(self, 'signal_generator') or not self.signal_generator:
            capabilities['limitations'].append('Signal Generator not initialized')
            capabilities['recommendations'].append('Check module imports and configuration')
            return capabilities
        
        # Test core signal generation capability
        try:
            if hasattr(self.signal_generator, 'generate_signals'):
                capabilities['signal_generation'] = True
                capabilities['features'].append('Core signal generation')
            else:
                capabilities['limitations'].append('Missing generate_signals method')
        except:
            capabilities['limitations'].append('Signal generation method not accessible')
        
        # Test basic export capability
        try:
            if hasattr(self.signal_generator, 'export_signals_to_json'):
                capabilities['basic_export'] = True
                capabilities['features'].append('Basic JSON export')
            else:
                capabilities['limitations'].append('Missing basic export functionality')
        except:
            capabilities['limitations'].append('Basic export not accessible')
        
        # Test advanced export capability
        try:
            if hasattr(self.signal_generator, 'advanced_export') and self.signal_generator.advanced_export:
                capabilities['advanced_export'] = True
                capabilities['features'].append('Advanced JSON export with verification data')
                
                # Test advanced export methods
                advanced_methods = ['export_signals_advanced', 'create_trading_bot_export', 'get_export_summary']
                for method in advanced_methods:
                    if hasattr(self.signal_generator.advanced_export, method):
                        capabilities['features'].append(f'Advanced export: {method}')
            else:
                capabilities['limitations'].append('Advanced export not available')
                capabilities['recommendations'].append('Install advanced_json_export_system module')
        except Exception as e:
            capabilities['limitations'].append(f'Advanced export test failed: {str(e)}')
        
        # Test verification integration
        try:
            if hasattr(self.signal_generator, 'get_crypto_historical_accuracy'):
                capabilities['verification_integration'] = True
                capabilities['features'].append('Historical accuracy verification')
                
                # Test verification data access
                try:
                    test_accuracy = self.signal_generator.get_crypto_historical_accuracy('bitcoin', '1d')
                    if test_accuracy and test_accuracy.get('data_available', False):
                        capabilities['features'].append('Verification data accessible')
                    else:
                        capabilities['limitations'].append('Verification data not accessible')
                except:
                    capabilities['limitations'].append('Verification data test failed')
            else:
                capabilities['limitations'].append('Missing verification integration')
        except:
            capabilities['limitations'].append('Verification integration test failed')
        
        # Test database connectivity
        try:
            if hasattr(self.signal_generator, 'db_path') and self.signal_generator.db_path:
                capabilities['database_connectivity'] = True
                capabilities['features'].append('Database connectivity')
                
                # Test database access
                try:
                    recent_preds = self.signal_generator.get_recent_predictions(hours_back=1)
                    capabilities['features'].append(f'Database access ({len(recent_preds) if recent_preds else 0} recent predictions)')
                except Exception as db_error:
                    capabilities['limitations'].append(f'Database access limited: {str(db_error)}')
            else:
                capabilities['limitations'].append('No database path configured')
        except:
            capabilities['limitations'].append('Database connectivity test failed')
        
        # Test configuration
        try:
            if hasattr(self.signal_generator, 'signal_config') and self.signal_generator.signal_config:
                capabilities['configuration_loaded'] = True
                capabilities['features'].append('Configuration loaded')
            else:
                capabilities['limitations'].append('Configuration not loaded')
        except:
            capabilities['limitations'].append('Configuration test failed')
        
        # Test statistics tracking
        if 'signals_generated' in self.stats:
            capabilities['statistics_tracking'] = True
            capabilities['features'].append('Statistics tracking enabled')
        else:
            capabilities['limitations'].append('Statistics tracking not initialized')
        
        # Determine overall status
        if capabilities['signal_generation'] and capabilities['verification_integration']:
            if capabilities['advanced_export']:
                capabilities['status'] = 'advanced'
            else:
                capabilities['status'] = 'standard'
        elif capabilities['signal_generation']:
            capabilities['status'] = 'basic'
        else:
            capabilities['status'] = 'disabled'
        
        # Generate recommendations based on limitations
        if capabilities['limitations']:
            if 'Signal Generator not initialized' in str(capabilities['limitations']):
                capabilities['recommendations'].append('Reinstall signal generator modules')
            if 'Advanced export not available' in str(capabilities['limitations']):
                capabilities['recommendations'].append('Install advanced JSON export system for enhanced features')
            if 'Verification data not accessible' in str(capabilities['limitations']):
                capabilities['recommendations'].append('Check database contains verification results')
            if 'Database access limited' in str(capabilities['limitations']):
                capabilities['recommendations'].append('Verify database file exists and is accessible')
        
        return capabilities


    def _validate_signal_generator_prerequisites(self, config_file):
        """🔍 Validate all prerequisites for signal generator initialization"""
        
        validation = {
            'valid': True,
            'issues': []
        }
        
        # Check config file
        if not config_file:
            validation['valid'] = False
            validation['issues'].append('No configuration file provided')
        else:
            try:
                from pathlib import Path
                if not Path(config_file).exists():
                    validation['valid'] = False
                    validation['issues'].append(f'Configuration file does not exist: {config_file}')
            except:
                validation['issues'].append('Cannot validate configuration file path')
        
        # Check database path
        if not hasattr(self, 'db_path') or not self.db_path:
            validation['valid'] = False
            validation['issues'].append('Database path not configured')
        else:
            try:
                from pathlib import Path
                db_path = Path(self.db_path)
                if not db_path.parent.exists():
                    validation['valid'] = False
                    validation['issues'].append(f'Database directory does not exist: {db_path.parent}')
            except:
                validation['issues'].append('Cannot validate database path')
        
        # Check core system components
        if not hasattr(self, 'config') or not self.config:
            validation['valid'] = False
            validation['issues'].append('System configuration not loaded')
        
        if not hasattr(self, 'stats') or not self.stats:
            validation['valid'] = False
            validation['issues'].append('Statistics system not initialized')
        
        # Check required methods exist
        required_methods = ['get_storage_path', 'get_threshold']
        for method in required_methods:
            if not hasattr(self, method):
                validation['valid'] = False
                validation['issues'].append(f'Required system method missing: {method}')
        
        return validation
    
    def _setup_graceful_shutdown(self):
        """🛑 Setup graceful shutdown handler"""
        try:
            # ✅ FIXED: Crea handler direttamente senza funzione esterna
            self.shutdown_handler = GracefulShutdownHandler(self)
            print("✅ Graceful shutdown handler active")
        except Exception as e:
            print(f"⚠️ Failed to setup graceful shutdown: {e}")

    @property
    def cache_directory(self):
        """💾 Get cache directory path"""
        cache_dir = self.get_storage_path('cache', 'api_cache_directory', 'D:/CryptoSystem/cache/api')
        return Path(cache_dir)

    def _cleanup_old_cache_files(self):
        """🧹 Cleanup old cache files - FIXED VERSION"""
        try:
            cache_files = list(self.cache_directory.glob("*.json")) + list(self.cache_directory.glob("*.pkl"))
            max_files = 1000
            
            if len(cache_files) > max_files:
                # Sort by modification time and remove oldest
                cache_files.sort(key=lambda x: x.stat().st_mtime)
                for old_file in cache_files[:len(cache_files) - max_files]:
                    old_file.unlink()
                    
        except Exception as e:
            print(f"     ⚠️ Cache cleanup error: {e}")

    def _load_config_with_prompt(self, config, config_file):
        """📧 Load config con prompt interattivo"""
        if config:
            return config
        
        # Se non specificato, usa il prompt
        if not config_file:
            selected_config_file = prompt_for_config_file()
        else:
            selected_config_file = config_file
        
        # STORE the selected file path as instance variable
        self._actual_config_file = selected_config_file
        
        # Se è None, usa la configurazione di default
        if selected_config_file is None:
            print("📧 Usando configurazione di default SSD...")
            return self._get_default_ssd_config()
        
        # Carica il file JSON
        try:
            config_path = Path(selected_config_file)
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                print(f"📄 Configurazione caricata: {config_path}")
                print(f"📝 Nome: {loaded_config.get('config_name', 'Sconosciuto')}")
                print(f"🔢 Versione: {loaded_config.get('config_version', 'Sconosciuta')}")
                return loaded_config
            else:
                print(f"❌ File non trovato: {selected_config_file}")
                print("📧 Usando configurazione di default...")
                self._actual_config_file = None
                return self._get_default_ssd_config()
        except Exception as e:
            print(f"❌ Errore caricamento config: {e}")
            self._actual_config_file = None
            return self._get_default_ssd_config()
    
    def _get_default_ssd_config(self):
        """🔧 Configurazione di default per SSD"""
        return {
            "config_name": "DEFAULT_SSD_OPTIMIZED",
            "config_version": "2.0",
            
            "storage": {
                "base_directory": "D:/CryptoSystem",
                "database": {
                    "main_db_path": "D:/CryptoSystem/database/crypto_optimized_6_9months.db",
                    "backup_directory": "D:/CryptoSystem/database/backups",
                    "archive_directory": "D:/CryptoSystem/database/archive"
                },
                "cache": {
                    "cache_directory": "D:/CryptoSystem/cache",
                    "api_cache_directory": "D:/CryptoSystem/cache/api",
                    "ml_cache_directory": "D:/CryptoSystem/cache/ml_models",
                    "bitcoin_cache_directory": "D:/CryptoSystem/cache/bitcoin"
                },
                "logs": {
                    "logs_directory": "D:/CryptoSystem/logs",
                    "verification_logs": "D:/CryptoSystem/logs/verification",
                    "performance_logs": "D:/CryptoSystem/logs/performance"
                }
            },
            
            "lookback_strategy": {
                "lookback_standard": 180,
                "lookback_premium": 270,
                "lookback_minimum": 120,
                "top_crypto_rank_threshold": 50
            },
            
            "quality_thresholds": {
                "confidence_threshold_1d": 0.60,
                "confidence_threshold_3d": 0.55,
                "quality_score_min_1d": 0.55,
                "quality_score_min_3d": 0.50,
                "magnitude_threshold_1d": 0.025,
                "magnitude_threshold_3d": 0.030
            },
            
            "accuracy_targets": {
                "accuracy_target_1d": 0.65,
                "accuracy_target_3d": 0.60,
                "high_confidence_threshold": 0.75
            },
            
            "system": {
                "max_cryptos_total": 750,
                "block_size": 25,
                "analysis_interval_hours": 8
            },
            
            "api": {
                "coingecko_api_key": "CG-hfMkTKrPM6gVa5d6k1xwFeM3",
                "max_api_calls_per_day": 1000,
                "api_delay": 2.0,
                "cache_duration_hours": 48,
                "max_retries": 5
            },
            
            "email": {
                "gmail_user": "danieleballarini98@gmail.com",
                "gmail_app_password": "tyut mbix ifur ymuf",
                "email_interval_hours": 8,
                "max_daily_emails": 6
            },
            
            "ml_models": {
                "use_traditional_ml": True,
                "use_catboost": True,
                "use_lstm": True,
                "use_tabnet": True,
                "model_performance_tracking": True
            },
            
            "bitcoin_benchmark": {
                "enabled": True,
                "cache_hours": 48
            },
            
            "verification": {
                "enabled": True,
                "verification_every_cycle": True
            },
            
            "alert_thresholds": {
                "1d": {
                    "high_priority": {
                        "confidence_min": 0.75,
                        "magnitude_min": 0.06,
                        "quality_score_min": 0.70
                    },
                    "medium_priority": {
                        "confidence_min": 0.65,
                        "magnitude_min": 0.04,
                        "quality_score_min": 0.60
                    },
                    "watch_priority": {
                        "confidence_min": 0.60,
                        "magnitude_min": 0.025,
                        "quality_score_min": 0.50
                    }
                },
                "3d": {
                    "high_priority": {
                        "confidence_min": 0.70,
                        "magnitude_min": 0.08,
                        "quality_score_min": 0.70
                    },
                    "medium_priority": {
                        "confidence_min": 0.60,
                        "magnitude_min": 0.05,
                        "quality_score_min": 0.60
                    },
                    "watch_priority": {
                        "confidence_min": 0.55,
                        "magnitude_min": 0.03,
                        "quality_score_min": 0.50
                    }
                }
            }
        }

    def _get_cache_key(self, crypto_id, data_type, days=None):
        """🔑 Genera chiave cache"""
        if days:
            return f"{crypto_id}_{data_type}_{days}d"
        return f"{crypto_id}_{data_type}"

    def _cache_crypto_data(self, crypto_id, data, data_type, days=None):
        """💾 Salva dati crypto nella cache SSD"""
        try:
            cache_dir = self.get_storage_path('cache', 'api_cache_directory', 'D:/CryptoSystem/cache/api')
            cache_file = Path(cache_dir) / f"{self._get_cache_key(crypto_id, data_type, days)}.json"
            
            # Ensure cache directory exists
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            cache_data = {
                'data': data.to_dict('records') if hasattr(data, 'to_dict') else data,
                'timestamp': datetime.now().isoformat(),
                'crypto_id': crypto_id,
                'data_type': data_type
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
            
            return True
        except Exception as e:
            print(f"⚠️ Cache save failed for {crypto_id}: {e}")
            return False

    def _load_cached_crypto_data(self, crypto_id, data_type, days=None, max_age_hours=48):
        """📂 Carica dati crypto dalla cache SSD"""
        try:
            cache_dir = self.get_storage_path('cache', 'api_cache_directory', 'D:/CryptoSystem/cache/api')
            cache_file = Path(cache_dir) / f"{self._get_cache_key(crypto_id, data_type, days)}.json"
            
            if not cache_file.exists():
                return None
            
            # Check age
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age.total_seconds() > max_age_hours * 3600:
                cache_file.unlink()  # Delete old cache
                return None
            
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            return cache_data['data']
        except Exception as e:
            print(f"⚠️ Cache load failed for {crypto_id}: {e}")
            return None
        
    def _load_ssd_cache(self, cache_key):
        """💾 Load cache from SSD - PICKLE VERSION"""
        try:
            # Try pickle first
            pickle_file = self.cache_directory / f"{cache_key}.pkl"
            json_file = self.cache_directory / f"{cache_key}.json"
            
            for cache_file in [pickle_file, json_file]:
                if cache_file.exists():
                    cache_time = cache_file.stat().st_mtime
                    if time.time() - cache_time < self.cache_duration:
                        if cache_file.suffix == '.pkl':
                            data = pd.read_pickle(cache_file)
                        else:
                            with open(cache_file, 'r') as f:
                                data = json.load(f)
                        return True, data
            
            return False, None
            
        except Exception as e:
            print(f"⚠️ SSD cache load error: {e}")
            return False, None

    def _cache_response(self, cache_key, data):
        """💾 Cache API response with FIXED handling"""
        try:
            # Memory cache
            if not hasattr(self, 'cache'):
                self.cache = {}
            
            self.cache[cache_key] = (time.time(), data)
            
            # SSD cache with fixed method
            self._save_ssd_cache(cache_key, data)
            
            # Cleanup memory cache if too large
            if len(self.cache) > 200:  # Keep smaller memory cache
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][0])
                del self.cache[oldest_key]
                
        except Exception as e:
            print(f"     ⚠️ Cache response failed: {e}")

    def _save_ssd_cache(self, cache_key, data):
        """💾 Save cache to SSD - FIXED VERSION"""
        try:
            if isinstance(data, pd.DataFrame):
                # Use pickle for DataFrames (handles Timestamps perfectly)
                cache_file = self.cache_directory / f"{cache_key}.pkl"
                data.to_pickle(cache_file)
            else:
                # Use JSON for simple data
                cache_file = self.cache_directory / f"{cache_key}.json"
                with open(cache_file, 'w') as f:
                    json.dump(data, f, default=str, indent=2)
            
            self._cleanup_old_cache_files()
            
        except Exception as e:
            print(f"     ⚠️ SSD cache save error: {e}")

    def cache_duration(self):
        """🕐 Get cache duration in seconds"""
        return self.config.get('api', {}).get('cache_duration_hours', 48) * 3600
    
    def _process_market_data(self, data, crypto_id):
        """📊 Process market data from CoinGecko API - IMPLEMENTAZIONE COMPLETA"""
        try:
            # Estrai dati CoinGecko
            prices = data.get('prices', [])
            volumes = data.get('total_volumes', [])
            market_caps = data.get('market_caps', [])
            
            if not prices:
                print(f"     ❌ No prices data for {crypto_id}")
                return None
            
            # Converti in DataFrame
            df_data = []
            for i, (timestamp, price) in enumerate(prices):
                volume = volumes[i][1] if i < len(volumes) else 0
                market_cap = market_caps[i][1] if i < len(market_caps) else 0
                
                df_data.append({
                    'timestamp': pd.to_datetime(timestamp, unit='ms'),
                    'price': float(price),
                    'volume': float(volume),
                    'market_cap': float(market_cap)
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            # Aggiungi indicatori tecnici di base
            df['return_1d'] = df['price'].pct_change()
            df['return_3d'] = df['price'].pct_change(3)
            df['return_7d'] = df['price'].pct_change(7)
            df['sma_10'] = df['price'].rolling(10).mean()
            df['sma_20'] = df['price'].rolling(20).mean()
            df['volatility_7d'] = df['return_1d'].rolling(7).std() * np.sqrt(365)
            
            print(f"     ✅ Processed {len(df)} days of data for {crypto_id}")
            return df
            
        except Exception as e:
            print(f"     ❌ Error processing market data for {crypto_id}: {e}")
            return None

    def get_threshold(self, category, key, default_value):
        """🎯 Ottieni threshold dal JSON config"""
        try:
            return self.config.get(category, {}).get(key, default_value)
        except:
            return default_value
    
    def get_storage_path(self, category, key, default_path):
        """💾 Ottieni percorso storage dal JSON config"""
        try:
            return self.config.get('storage', {}).get(category, {}).get(key, default_path)
        except:
            return default_path
    
    def _initialize_optimized_components(self):
        """🔧 Initialize all optimized components con percorsi SSD"""
        print("🔧 Inizializzazione componenti ottimizzati SSD...")
        
        # === API SYSTEM ===
        api_config = self.config.get('api', {})
        self.api = OptimizedCoinGeckoAPI(api_config)
        
        # === DATABASE SYSTEM ===
        if DATABASE_SYSTEM_AVAILABLE:
            try:
                db_path = self.get_storage_path('database', 'main_db_path', 
                                            'D:/CryptoSystem/database/crypto_optimized_6_9months.db')
                
                # ✅ Pass config to database
                database_config = {
                    'storage': self.config.get('storage', {}),
                    'disk_config': self.config.get('disk_config', {})
                }
                
                self.database = OptimizedCryptoDatabase(db_path, database_config)
                print("✅ Database ottimizzato SSD inizializzato")
            except Exception as e:
                print(f"⚠️ Database initialization failed: {e}")
                self.database = None
        else:
            self.database = None
        
        # === BITCOIN BENCHMARK ===
        if BITCOIN_SYSTEM_AVAILABLE and self.get_threshold('bitcoin_benchmark', 'enabled', True):
            try:
                btc_config = {
                    'coingecko_api_key': self.get_threshold('api', 'coingecko_api_key', ''),
                    'bitcoin_cache_hours': self.get_threshold('bitcoin_benchmark', 'cache_hours', 48),
                    'bitcoin_lookback_days': self.get_threshold('lookback_strategy', 'lookback_premium', 270),
                    'cache_dir': self.get_storage_path('cache', 'bitcoin_cache_directory', 'D:/CryptoSystem/cache/bitcoin')
                }
                self.bitcoin_benchmark = RealBitcoinBenchmarkSystem(btc_config)
                print("✅ Bitcoin Benchmark SSD inizializzato")
                if self.bitcoin_benchmark:
                    # Clear any corrupted cache on startup
                    try:
                        self.bitcoin_benchmark.clear_corrupted_cache()
                    except Exception as e:
                        print(f"⚠️ Cache cleanup failed: {e}")
            except Exception as e:
                print(f"⚠️ Bitcoin Benchmark initialization failed: {e}")
                self.bitcoin_benchmark = None
        else:
            self.bitcoin_benchmark = None
        
        # === ML SYSTEM AGGIORNATO ===
        if ML_SYSTEM_AVAILABLE:
            try:
                # ✅ NUOVA CONFIGURAZIONE COMPLETA per OptimizedDualHorizonMLSystem
                print("🧠 Initializing Optimized Dual Horizon ML System...")
                
                # Passa la configurazione completa al nuovo sistema ML
                self.ml_system = OptimizedDualHorizonMLSystem(self.config)
                
                # ✅ VERIFICA STATUS SISTEMA ML
                ml_status = self.ml_system.get_system_status()
                print(f"✅ ML System ottimizzato inizializzato: {ml_status}")
                
                # 🔧 FIX: Collega il database al ML system per il feedback optimizer
                if hasattr(self, 'database') and self.database:
                    self.ml_system.database = self.database
                    print("🔗 Database linked to ML system for feedback optimizer")
                    
                # ✅ CARICA MODELLI ESISTENTI se disponibili
                try:
                    models_loaded = self.ml_system.load_models_compatible()
                    if models_loaded:
                        print(f"📦 Loaded existing models: {models_loaded}")
                except Exception as e:
                    print(f"⚠️ Could not load existing models: {e}")
                    
                # 🔗 AGGIUNGI QUESTA SEZIONE - COLLEGAMENTO SISTEMI
                if self.bitcoin_benchmark and self.ml_system:
                    self.ml_system.set_bitcoin_benchmark(self.bitcoin_benchmark)
                    print("🔗 Bitcoin benchmark collegato al ML system per soglie dinamiche")
            
            except Exception as e:
                print(f"⚠️ ML System initialization failed: {e}")
                import traceback
                traceback.print_exc()
                self.ml_system = None
        else:
            self.ml_system = None

        # === VERIFICATION SYSTEM ===
        if VERIFICATION_SYSTEM_AVAILABLE and self.get_threshold('verification', 'enabled', True):
            try:
                verification_config = {
                    'gmail_user': self.get_threshold('email', 'gmail_user', ''),
                    'gmail_app_password': self.get_threshold('email', 'gmail_app_password', ''),
                    'accuracy_target_1d': self.get_threshold('accuracy_targets', 'accuracy_target_1d', 0.65),
                    'accuracy_target_3d': self.get_threshold('accuracy_targets', 'accuracy_target_3d', 0.60),
                    'rate_limit_delay': self.get_threshold('api', 'api_delay', 2.0),
                    'max_retries_api': self.get_threshold('api', 'max_retries', 3),
                    # 🔧 ADD: Model feedback config
                    'model_feedback_learning_rate': self.get_threshold('model_feedback', 'model_feedback_learning_rate', 0.1),
                    'model_feedback_momentum': self.get_threshold('model_feedback', 'model_feedback_momentum', 0.9),
                    'min_samples_feedback': self.get_threshold('model_feedback', 'min_samples_feedback', 10),
                    'performance_window_days': self.get_threshold('model_feedback', 'performance_window_days', 30),
                    'weight_update_frequency_cycles': self.get_threshold('model_feedback', 'weight_update_frequency_cycles', 1)
                }
                
                db_path = self.get_storage_path('database', 'main_db_path', 'D:/CryptoSystem/database/crypto_optimized_6_9months.db')
                self.verification_system = OptimizedVerificationSystem(db_path, verification_config)
                print("✅ Verification System SSD inizializzato")
            except Exception as e:
                print(f"⚠️ Verification System initialization failed: {e}")
                self.verification_system = None
        else:
            self.verification_system = None
        
        # 🔧 FIXED: Integrazione completa model feedback
        if VERIFICATION_SYSTEM_AVAILABLE and ML_SYSTEM_AVAILABLE:
            if self.verification_system and self.ml_system:
                # 1. Integra il sistema di feedback
                success = self.verification_system.integrate_model_feedback_system(self.ml_system)
                if success:
                    # 2. 🔧 ADD: Collega il feedback optimizer al ML system
                    if hasattr(self.verification_system, 'feedback_optimizer') and self.verification_system.feedback_optimizer:
                        if hasattr(self.ml_system, 'set_feedback_optimizer'):
                            self.ml_system.set_feedback_optimizer(self.verification_system.feedback_optimizer)
                            print("✅ Feedback optimizer linked to ML system")
                        else:
                            print("⚠️ ML system missing set_feedback_optimizer method")
                    print("✅ Model feedback integration successful")
                else:
                    print("⚠️ Model feedback integration failed")
        
        # === NOTIFICATION SYSTEM ===
        gmail_user = self.get_threshold('email', 'gmail_user', '')
        gmail_app_password = self.get_threshold('email', 'gmail_app_password', '')
        
        if gmail_user and gmail_app_password:
            try:
                from crypto_notifier import DualHorizonCryptoNotifierBugFixed
                
                self.notifier = DualHorizonCryptoNotifierBugFixed(gmail_user, gmail_app_password,)
                print("✅ Notifier ottimizzato inizializzato")
            except Exception as e:
                print(f"⚠️ Notifier initialization failed: {e}")
                self.notifier = None
        else:
            self.notifier = None
            print("⚠️ Notifier disabilitato - credenziali Gmail mancanti")
    
    def _setup_signal_handlers(self):
        """📡 Setup signal handlers per shutdown graceful"""
        def signal_handler(signum, frame):
            print(f"\n🛑 Ricevuto segnale {signum} - shutdown graceful...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def show_system_config(self):
        """📊 Mostra configurazione sistema completa"""
        print("\n📊 CONFIGURAZIONE SISTEMA ATTIVA")
        print("=" * 60)
        
        # Informazioni generali
        print(f"📝 Nome: {self.config.get('config_name', 'Sconosciuto')}")
        print(f"🔢 Versione: {self.config.get('config_version', 'Sconosciuta')}")
        print(f"📅 Data: {self.config.get('created_date', 'Sconosciuta')}")
        
        # Storage
        storage = self.config.get('storage', {})
        print(f"\n💾 STORAGE:")
        print(f"   Base: {storage.get('base_directory', 'N/A')}")
        print(f"   Database: {storage.get('database', {}).get('main_db_path', 'N/A')}")
        print(f"   Cache: {storage.get('cache', {}).get('cache_directory', 'N/A')}")
        print(f"   Logs: {storage.get('logs', {}).get('logs_directory', 'N/A')}")
        
        # Lookback strategy
        lookback = self.config.get('lookback_strategy', {})
        print(f"\n🎯 LOOKBACK STRATEGY:")
        print(f"   Standard: {lookback.get('lookback_standard', 180)} giorni")
        print(f"   Premium: {lookback.get('lookback_premium', 270)} giorni")
        print(f"   Top crypto threshold: #{lookback.get('top_crypto_rank_threshold', 50)}")
        
        # Quality thresholds
        quality = self.config.get('quality_thresholds', {})
        print(f"\n🏆 QUALITY THRESHOLDS:")
        print(f"   Confidence 1d: {quality.get('confidence_threshold_1d', 0.60):.0%}")
        print(f"   Confidence 3d: {quality.get('confidence_threshold_3d', 0.55):.0%}")
        print(f"   Quality score 1d: {quality.get('quality_score_min_1d', 0.55):.0%}")
        print(f"   Quality score 3d: {quality.get('quality_score_min_3d', 0.50):.0%}")
        
        # Accuracy targets
        accuracy = self.config.get('accuracy_targets', {})
        print(f"\n🎯 ACCURACY TARGETS:")
        print(f"   Target 1d: {accuracy.get('accuracy_target_1d', 0.65):.0%}")
        print(f"   Target 3d: {accuracy.get('accuracy_target_3d', 0.60):.0%}")
        
        # System
        system = self.config.get('system', {})
        print(f"\n⚙️ SYSTEM:")
        print(f"   Max cryptos: {system.get('max_cryptos_total', 750)}")
        print(f"   Block size: {system.get('block_size', 25)}")
        print(f"   Interval: {system.get('analysis_interval_hours', 8)}h")
        
        # API
        api = self.config.get('api', {})
        api_key_status = "✅ Configurata" if api.get('coingecko_api_key') else "❌ Mancante"
        print(f"\n🌐 API:")
        print(f"   CoinGecko key: {api_key_status}")
        print(f"   Max calls/day: {api.get('max_api_calls_per_day', 1000)}")
        print(f"   Cache duration: {api.get('cache_duration_hours', 48)}h")
        
        # Email
        email = self.config.get('email', {})
        email_status = "✅ Configurate" if (email.get('gmail_user') and email.get('gmail_app_password')) else "❌ Mancanti"
        print(f"\n📧 EMAIL:")
        print(f"   Credenziali: {email_status}")
        print(f"   Interval: {email.get('email_interval_hours', 8)}h")
        print(f"   Max daily: {email.get('max_daily_emails', 6)}")
    
    def _process_alerts_optimized(self, prediction_data):
        """🚨 Process alerts con thresholds dal JSON"""
        if not self.notifier:
            return
        
        try:
            predictions = prediction_data.get('predictions', {})
            
            for horizon_key, pred in predictions.items():
                confidence = pred.get('confidence', 0)
                magnitude = abs(pred.get('predicted_change', 0))
                quality_score = pred.get('quality_score', 0)
                
                # Get thresholds dal JSON
                alert_thresholds = self.config.get('alert_thresholds', {}).get(horizon_key, {})
                
                # Determine alert level
                high_thresh = alert_thresholds.get('high_priority', {})
                medium_thresh = alert_thresholds.get('medium_priority', {})
                watch_thresh = alert_thresholds.get('watch_priority', {})
                
                if (confidence >= high_thresh.get('confidence_min', 0.75) and 
                    magnitude >= high_thresh.get('magnitude_min', 0.06) and 
                    quality_score >= high_thresh.get('quality_score_min', 0.70)):
                    alert_type = 'high'
                elif (confidence >= medium_thresh.get('confidence_min', 0.65) and 
                      magnitude >= medium_thresh.get('magnitude_min', 0.04) and 
                      quality_score >= medium_thresh.get('quality_score_min', 0.60)):
                    alert_type = 'medium'
                elif (confidence >= watch_thresh.get('confidence_min', 0.60) and 
                      magnitude >= watch_thresh.get('magnitude_min', 0.025)):
                    alert_type = 'watch'
                else:
                    continue
                
                # Add alert
                self.notifier.add_dual_alert_fixed(alert_type, prediction_data, horizon_key)
                
        except Exception as e:
            print(f"       ⚠️ Alert processing failed: {e}")
    
    def run_continuous_optimized_ssd(self):
        """🔄 Run continuous system con SSD storage + graceful shutdown"""
        self.running = True
        self.stats['start_time'] = time.time()
        
        # Mostra configurazione all'avvio
        self.show_system_config()
        
        print(f"\n🚀 AVVIO SISTEMA CONTINUO SSD")
        print(f"⏰ Interval analisi: {self.get_threshold('system', 'analysis_interval_hours', 8)} ore")
        print(f"📊 Target cryptos: {self.get_threshold('system', 'max_cryptos_total', 750)}")
        print(f"🎯 Lookback strategy: {self.get_threshold('lookback_strategy', 'lookback_standard', 180)}d / {self.get_threshold('lookback_strategy', 'lookback_premium', 270)}d")
        print(f"💾 Storage SSD: {self.config.get('storage', {}).get('base_directory', 'D:/CryptoSystem')}")
        print(f"🛑 Press Ctrl+C for graceful shutdown with pending alerts")
        print(f"{'='*80}")
        
        try:
            while self.running:
                # ✅ Check running status before cycle
                if not self.running:
                    print("🛑 Shutdown requested - breaking main loop")
                    break
                
                try:
                    # Run optimized cycle
                    print(f"\n🔄 Starting analysis cycle...")
                    self.run_optimized_cycle()
                    
                    # ✅ Check running status after cycle
                    if not self.running:
                        print("🛑 Shutdown requested after cycle completion")
                        break
                    
                    # Calculate wait time for next cycle
                    interval_hours = self.get_threshold('system', 'analysis_interval_hours', 8)
                    interval_seconds = interval_hours * 3600
                    
                    print(f"\n💤 Attesa {interval_hours}h per prossimo ciclo...")
                    print(f"🛑 Press Ctrl+C anytime for graceful shutdown with pending alerts")
                    
                    # ✅ ENHANCED: Wait with frequent checks for graceful shutdown
                    wait_start = time.time()
                    next_status_time = wait_start + 1800  # First status in 30 minutes
                    
                    while (time.time() - wait_start < interval_seconds) and self.running:
                        # ✅ More frequent checks for responsiveness (30 seconds instead of 60)
                        time.sleep(30)
                        
                        # Check if shutdown requested
                        if not self.running:
                            print("🛑 Shutdown requested during wait period")
                            break
                        
                        # Status update every 30 minutes
                        current_time = time.time()
                        if current_time >= next_status_time:
                            elapsed = current_time - wait_start
                            remaining = (interval_seconds - elapsed) / 3600
                            
                            if remaining > 0:
                                print(f"⏳ {remaining:.1f} ore al prossimo ciclo")
                                print(f"🛑 Ctrl+C per shutdown graceful con invio alerts")
                            
                            next_status_time += 1800  # Next status in 30 minutes
                    
                    # Final check before next iteration
                    if not self.running:
                        print("🛑 Exiting wait loop due to shutdown request")
                        break
                        
                except Exception as cycle_error:
                    print(f"\n❌ Errore durante ciclo di analisi: {cycle_error}")
                    import traceback
                    traceback.print_exc()
                    
                    # ✅ Check if we should continue after cycle error
                    if not self.running:
                        print("🛑 Shutdown requested - not retrying after error")
                        break
                    
                    print("⏳ Attesa 5 minuti prima di riprovare...")
                    # Wait 5 minutes before retry, but check for shutdown frequently
                    retry_wait_start = time.time()
                    while (time.time() - retry_wait_start < 300) and self.running:
                        time.sleep(30)
                    
                    if not self.running:
                        break
        
        except KeyboardInterrupt:
            # ✅ This should normally be handled by the signal handler
            # But we keep this as a fallback
            print(f"\n🛑 KeyboardInterrupt received - initiating fallback shutdown")
            self.running = False
            
        except Exception as e:
            print(f"\n❌ Errore sistema critico: {e}")
            import traceback
            traceback.print_exc()
            self.running = False
            
        finally:
            # ✅ ENHANCED: Comprehensive shutdown procedure
            print(f"\n{'='*60}")
            print(f"🛑 SHUTDOWN PROCEDURE INITIATED")
            print(f"⏰ Shutdown time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}")
            
            # Ensure running is False
            self.running = False
            
            # ✅ Check if graceful shutdown handler already sent alerts
            shutdown_handler_active = (
                hasattr(self, 'shutdown_handler') and 
                self.shutdown_handler and 
                self.shutdown_handler.shutdown_requested
            )
            
            if shutdown_handler_active:
                print("✅ Graceful shutdown handler active - alerts already processed")
            else:
                print("📧 Manual shutdown - checking for pending alerts...")
                self._send_final_alerts()
            
            # ✅ System cleanup
            try:
                print("🧹 Executing system cleanup...")
                self._cleanup_optimized()
                print("✅ System cleanup completed")
            except Exception as cleanup_error:
                print(f"⚠️ Cleanup error (non-critical): {cleanup_error}")
            
            # ✅ Final statistics
            if 'start_time' in self.stats:
                total_runtime = time.time() - self.stats['start_time']
                runtime_hours = total_runtime / 3600
                print(f"📊 Total runtime: {runtime_hours:.2f} hours")
                print(f"📈 Cycles completed: {getattr(self, 'cycle_count', 0)}")
                print(f"🔍 Verifications: {self.stats.get('verification_completed', 0)}")
                print(f"🎯 High confidence predictions: {self.stats.get('high_confidence_predictions', 0)}")
            
            print(f"👋 Sistema SSD shutdown completato")
            print(f"{'='*60}")
    
    def _send_final_alerts(self):
        """📧 Send final alerts if not done by signal handler"""
        try:
            if self.notifier:
                print("📧 Sending final alerts...")
                # Check and send pending alerts
                self.notifier.last_summary_sent = None
                if hasattr(self.notifier, 'send_6hour_dual_summary_fixed'):
                    success = self.notifier.send_6hour_dual_summary_fixed()
                    if success:
                        print("✅ Final alerts sent")
                elif hasattr(self.notifier, 'send_6hour_dual_summary'):
                    success = self.notifier.send_6hour_dual_summary()
                    if success:
                        print("✅ Final alerts sent")
        except Exception as e:
            print(f"❌ Failed to send final alerts: {e}")
    # === 🔧 NUOVI METODI DI SUPPORTO PER TIMING INDIPENDENTE ===

    def _should_run_verification_cycle(self, interval_hours=6):
        """⏰ Check if it's time to run verification cycle"""
        if not hasattr(self, '_last_verification_time'):
            self._last_verification_time = None
        
        if self._last_verification_time is None:
            return True  # First run
        
        time_since_last = time.time() - self._last_verification_time
        hours_since_last = time_since_last / 3600
        
        return hours_since_last >= interval_hours

    # ✅ E aggiorna anche il metodo _should_send_email_alert:
    def _should_send_email_alert(self, interval_hours=6):
        """📧 Check if it's time to send email alert"""
        if not hasattr(self, '_last_email_time'):
            # ✅ Inizializza con timestamp corrente, non None
            self._last_email_time = time.time()
            print(f"📧 First run - email scheduled in {interval_hours}h")
            return False  # ✅ Non inviare email al primo avvio
        
        if self._last_email_time is None:
            # Fallback se è ancora None
            self._last_email_time = time.time()
            print(f"📧 _last_email_time was None - initialized for {interval_hours}h delay")
            return False
        
        time_since_last = time.time() - self._last_email_time
        hours_since_last = time_since_last / 3600
        
        should_send = hours_since_last >= interval_hours
        print(f"📧 Time since last email: {hours_since_last:.1f}h (threshold: {interval_hours}h)")
        
        return should_send

    def _get_last_verification_time(self):
        """🔍 Get timestamp of last verification"""
        if not hasattr(self, '_last_verification_time'):
            self._last_verification_time = None
        return self._last_verification_time

    def _update_last_verification_time(self):
        """🕐 Update last verification timestamp"""
        self._last_verification_time = time.time()

    def _get_last_verification_time_str(self):
        """📅 Get formatted last verification time"""
        if not hasattr(self, '_last_verification_time') or self._last_verification_time is None:
            return "Never"
        
        last_time = datetime.fromtimestamp(self._last_verification_time)
        time_diff = datetime.now() - last_time
        hours_ago = time_diff.total_seconds() / 3600
        
        if hours_ago < 1:
            return f"{int(time_diff.total_seconds() / 60)}m ago"
        else:
            return f"{hours_ago:.1f}h ago"

    def _get_last_email_time(self):
        """📧 Get timestamp of last email"""
        if not hasattr(self, '_last_email_time'):
            self._last_email_time = None
        return self._last_email_time

    def _update_last_email_time(self):
        """📧 Update last email timestamp"""
        self._last_email_time = time.time()

    def _get_last_email_time_str(self):
        """📧 Get formatted last email time"""
        if not hasattr(self, '_last_email_time') or self._last_email_time is None:
            return "Never"
        
        last_time = datetime.fromtimestamp(self._last_email_time)
        time_diff = datetime.now() - last_time
        hours_ago = time_diff.total_seconds() / 3600
        
        if hours_ago < 1:
            return f"{int(time_diff.total_seconds() / 60)}m ago"
        else:
            return f"{hours_ago:.1f}h ago"

    def _get_recent_verification_count(self):
        """📊 Get recent verification count for feedback"""
        return getattr(self, '_verification_count_for_feedback', 0)

    def _increment_verification_count_for_feedback(self, count):
        """🔢 Increment verification counter for model feedback"""
        if not hasattr(self, '_verification_count_for_feedback'):
            self._verification_count_for_feedback = 0
        
        self._verification_count_for_feedback += count

    def _reset_verification_counter_for_feedback(self):
        """🔄 Reset verification counter after feedback"""
        self._verification_count_for_feedback = 0

    # === 🔧 CONVERSIONE BYTES (CRITICO!) ===

    def safe_float_convert(self, value, field_name="unknown") -> float:
        """🔄 Converte safely qualsiasi valore in float - GESTISCE BYTES"""
        
        if value is None:
            return 0.0
        
        # Se è già un numero, restituiscilo
        if isinstance(value, (int, float)):
            return float(value)
        
        # Se è string numerica, convertila
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                print(f"⚠️ Cannot convert string '{value}' to float for field {field_name}")
                return 0.0
        
        # 🚨 GESTIONE BYTES - CRITICO!
        if isinstance(value, bytes):
            try:
                import struct
                
                # Prova diversi formati binari
                if len(value) == 4:  # float32
                    result = struct.unpack('<f', value)[0]  # little-endian float32
                elif len(value) == 8:  # float64
                    result = struct.unpack('<d', value)[0]  # little-endian float64
                else:
                    # Prova a interpretare come string
                    try:
                        string_value = value.decode('utf-8', errors='ignore')
                        result = float(string_value)
                    except:
                        print(f"⚠️ Cannot decode bytes {value.hex()} for field {field_name}")
                        result = 0.0
                
                print(f"🔄 Converted bytes to float: {field_name}={result}")
                return float(result)
                
            except Exception as e:
                print(f"❌ Binary conversion failed for {field_name}: {e}")
                return 0.0
        
        # Fallback per tipi sconosciuti
        print(f"⚠️ Unknown type {type(value)} for field {field_name}, using 0.0")
        return 0.0

    def get_high_confidence_predictions_for_email_safe(self):
        """📊 Raccoglie predizioni high confidence CON CONVERSIONE BYTES"""
        
        if not self.database or not hasattr(self.database, 'db_path'):
            print("⚠️ Database not available for email predictions")
            return []
        
        try:
            import sqlite3
            
            with sqlite3.connect(self.database.db_path) as conn:
                cursor = conn.cursor()
                
                print("📊 Searching for high confidence predictions (with bytes conversion)...")
                
                # Query predizioni high confidence ultime 6 ore
                lookback_hours = 6
                threshold_1d = self.get_threshold('quality_thresholds', 'confidence_threshold_1d', 0.60)
                threshold_3d = self.get_threshold('quality_thresholds', 'confidence_threshold_3d', 0.55)
                
                cursor.execute('''
                    SELECT 
                        crypto_id, crypto_name, current_price,
                        confidence_1d, predicted_change_1d, predicted_price_1d,
                        confidence_3d, predicted_change_3d, predicted_price_3d,
                        market_regime, bitcoin_correlation,
                        datetime(timestamp) as pred_time
                    FROM predictions_optimized 
                    WHERE datetime(timestamp) >= datetime('now', '-{} hours')
                    AND (
                        (confidence_1d IS NOT NULL AND confidence_1d >= ?)
                        OR 
                        (confidence_3d IS NOT NULL AND confidence_3d >= ?)
                    )
                    ORDER BY (COALESCE(confidence_1d, 0) + COALESCE(confidence_3d, 0)) DESC
                    LIMIT 50
                '''.format(lookback_hours), (threshold_1d, threshold_3d))
                
                raw_predictions = cursor.fetchall()
                
            # 🔧 CONVERSIONE BYTES per ogni predizione
            safe_predictions = []
            for pred in raw_predictions:
                try:
                    (crypto_id, crypto_name, current_price, conf_1d, change_1d, price_1d, 
                    conf_3d, change_3d, price_3d, market_regime, bitcoin_corr, pred_time) = pred
                    
                    # Converti TUTTI i valori numerici safely
                    safe_pred = (
                        crypto_id,  # string
                        crypto_name,  # string
                        self.safe_float_convert(current_price, "current_price"),
                        self.safe_float_convert(conf_1d, "confidence_1d"),
                        self.safe_float_convert(change_1d, "predicted_change_1d"),
                        self.safe_float_convert(price_1d, "predicted_price_1d"),
                        self.safe_float_convert(conf_3d, "confidence_3d"),
                        self.safe_float_convert(change_3d, "predicted_change_3d"),
                        self.safe_float_convert(price_3d, "predicted_price_3d"),
                        market_regime,  # string
                        self.safe_float_convert(bitcoin_corr, "bitcoin_correlation"),
                        pred_time  # string
                    )
                    
                    safe_predictions.append(safe_pred)
                    
                except Exception as e:
                    print(f"❌ Error converting prediction bytes: {e}")
                    continue
            
            print(f"✅ Converted {len(safe_predictions)} predictions with bytes handling")
            return safe_predictions
            
        except Exception as e:
            print(f"❌ Error getting safe predictions: {e}")
            return []

    def process_email_alerts_automatically_safe(self):
        """🚨 Processa alert per email automaticamente - VERSIONE SAFE CON BYTES"""
        
        if not self.notifier:
            print("⚠️ Notifier not available")
            return False
        
        try:
            print("🚨 Checking for email alerts (safe bytes conversion)...")
            
            # 1. Check timing
            should_send = self._should_send_email_alert(6)  # 6 hours
            
            if not should_send:
                print("   ⏰ Not time for email yet (< 6 hours since last)")
                return False
            
            print("   ⏰ TIME TO SEND EMAIL! (>= 6 hours)")
            
            # 2. Get high confidence predictions from database - SAFE VERSION
            db_predictions = self.get_high_confidence_predictions_for_email_safe()
            
            if not db_predictions:
                print("   📊 No high confidence predictions found - no email")
                return False
            
            print(f"   📊 Found {len(db_predictions)} high confidence predictions")
            
            # 3. Convert to alert format (già con valori convertiti!)
            formatted_alerts = self.convert_db_predictions_to_alerts_safe(db_predictions)
            
            if not formatted_alerts:
                print("   📄 No alerts after formatting")
                return False
            
            print(f"   ✅ Formatted {len(formatted_alerts)} alerts")
            
            # 4. Add alerts to notifier
            added_count = 0
            for horizon, prediction_data in formatted_alerts:
                try:
                    horizon_data = prediction_data['predictions'].get(horizon, {})
                    priority = self.determine_alert_priority(horizon_data)
                    
                    success = self.notifier.add_dual_alert_fixed(priority, prediction_data, horizon)
                    if success:
                        added_count += 1
                    
                except Exception as e:
                    print(f"   ❌ Error adding alert: {e}")
            
            print(f"   ✅ Added {added_count} alerts to notifier")
            
            # 5. Send email summary
            if added_count > 0:
                print("   📧 Sending email with real predictions...")
                success = self.notifier.send_6hour_dual_summary_fixed()
                
                if success:
                    print("   🎉 AUTOMATIC EMAIL SENT SUCCESSFULLY!")
                    return True
                else:
                    print("   ❌ Failed to send email")
                    return False
            else:
                print("   ℹ️ No alerts added to notifier")
                return False
            
        except Exception as e:
            print(f"❌ Email processing failed: {e}")
            return False

    def convert_db_predictions_to_alerts_safe(self, db_predictions):
        """📄 Converte predizioni database in formato alert - SAFE BYTES VERSION"""
        
        formatted_alerts = []
        
        for pred in db_predictions:
            try:
                (crypto_id, crypto_name, current_price, conf_1d, change_1d, price_1d, 
                conf_3d, change_3d, price_3d, market_regime, bitcoin_corr, pred_time) = pred
                
                # I dati sono già stati convertiti da safe_float_convert!
                # Base prediction structure
                base_pred = {
                    'crypto_id': crypto_id or 'unknown',
                    'crypto_name': crypto_name or 'Unknown',
                    'current_price': current_price,  # già float
                    'market_regime': market_regime or 'unknown',
                    'bitcoin_correlation': bitcoin_corr,  # già float
                    'prediction_time': pred_time,
                    'predictions': {}
                }
                
                # Process 1d prediction if high confidence
                if conf_1d and conf_1d >= 0.65:
                    # Fix predicted_price if missing/zero
                    if not price_1d or price_1d <= 0:
                        price_1d = current_price * (1 + (change_1d or 0))
                    
                    base_pred['predictions']['1d'] = {
                        'predicted_change': change_1d,  # già float
                        'predicted_price': price_1d,    # già float
                        'confidence': conf_1d,          # già float
                        'direction': 'up' if (change_1d or 0) > 0 else 'down',
                        'magnitude': abs(change_1d or 0)
                    }
                    
                    # Add to alerts list
                    formatted_alerts.append(('1d', base_pred.copy()))
                
                # Process 3d prediction if high confidence
                if conf_3d and conf_3d >= 0.65:
                    # Fix predicted_price if missing/zero
                    if not price_3d or price_3d <= 0:
                        price_3d = current_price * (1 + (change_3d or 0))
                    
                    base_pred['predictions']['3d'] = {
                        'predicted_change': change_3d,  # già float
                        'predicted_price': price_3d,    # già float
                        'confidence': conf_3d,          # già float
                        'direction': 'up' if (change_3d or 0) > 0 else 'down',
                        'magnitude': abs(change_3d or 0)
                    }
                    
                    # Add to alerts list  
                    formatted_alerts.append(('3d', base_pred.copy()))
                    
            except Exception as e:
                print(f"   ❌ Error converting prediction: {e}")
                continue
        
        return formatted_alerts

    def determine_alert_priority(self, horizon_data):
        """🎯 Determina priorità alert basata su confidence e magnitude"""
        
        confidence = horizon_data.get('confidence', 0)
        magnitude = horizon_data.get('magnitude', 0)
        
        if confidence >= 0.75 and magnitude >= 0.04:
            return 'high'
        elif confidence >= 0.65 and magnitude >= 0.025:
            return 'medium'
        else:
            return 'watch'

    def debug_verification_integration(self):
        """🔍 Debug verification system integration"""
        
        print("\n🔍 VERIFICATION INTEGRATION DEBUG")
        print("=" * 50)
        
        print(f"✅ Verification system: {'Active' if self.verification_system else 'Inactive'}")
        
        if self.verification_system:
            try:
                # Test predictions ready
                predictions_1d, predictions_3d = self.verification_system.get_predictions_ready_for_verification()
                ready_count = len(predictions_1d) + len(predictions_3d)
                print(f"📊 Predictions ready: {ready_count} ({len(predictions_1d)} 1d + {len(predictions_3d)} 3d)")
                
                # Test timing
                should_verify = self._should_run_verification_cycle(6)
                print(f"⏰ Should verify now: {should_verify}")
                print(f"🕐 Last verification: {self._get_last_verification_time_str()}")
                
                # Test feedback optimizer
                has_feedback = (hasattr(self.verification_system, 'feedback_optimizer') and 
                            self.verification_system.feedback_optimizer)
                print(f"🤖 Feedback optimizer: {'Active' if has_feedback else 'Inactive'}")
                print(f"📈 Verification count for feedback: {self._get_recent_verification_count()}")
                
            except Exception as e:
                print(f"❌ Debug error: {e}")
        
        print("=" * 50)

    def run_optimized_cycle(self):
        """🔄 Run single optimized cycle - MODIFIED per timing indipendenti"""
        cycle_start = time.time()
        self.cycle_count += 1
        
        print(f"\n{'='*80}")
        print(f"🚀 OPTIMIZED CYCLE #{self.cycle_count}")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Lookback: {self.get_threshold('lookback_strategy', 'lookback_standard', 180)}d standard / {self.get_threshold('lookback_strategy', 'lookback_premium', 270)}d premium")
        print(f"📊 Target confidence: 1d≥{self.get_threshold('quality_thresholds', 'confidence_threshold_1d', 0.60):.0%}, 3d≥{self.get_threshold('quality_thresholds', 'confidence_threshold_3d', 0.55):.0%}")
        print(f"💾 Storage: {self.config.get('storage', {}).get('base_directory', 'D:/CryptoSystem')}")
        print(f"{'='*80}")
        
        # Memory management
        self._manage_memory_optimized()
        
        # === 🔍 VERIFICATION TIMING INDIPENDENTE (ogni 6 ore) ===
        verification_results = None
        if (self.verification_system and 
            self.get_threshold('verification', 'enabled', True)):
            
            verification_interval_hours = 6  # 🔧 FIXED: 6 ore fisse
            
            try:
                # Check if it's time for verification (independent timing)
                should_verify = self._should_run_verification_cycle(verification_interval_hours)
                
                if should_verify:
                    print(f"🔍 VERIFICATION CYCLE - Independent timing (every {verification_interval_hours}h)")
                    print("-" * 50)
                    
                    # ✅ USES FIXED VERIFICATION SYSTEM
                    verification_results = self.verification_system.run_complete_verification_optimized(
                        self.cycle_count, ml_system=self.ml_system if hasattr(self, 'ml_system') else None
                    )
                    
                    if verification_results and verification_results['total_verified'] > 0:
                        accuracy_1d = verification_results.get('accuracy_1d', 0)
                        accuracy_3d = verification_results.get('accuracy_3d', 0)
                        self.stats['verification_completed'] = self.stats.get('verification_completed', 0) + verification_results['total_verified']
                        self.stats['model_feedback_updates'] = self.stats.get('model_feedback_updates', 0) + len(verification_results.get('model_feedback', []))
                        
                        print(f"✅ Verification: {verification_results['total_verified']} completed")
                        print(f"📊 Accuracy: 1d={accuracy_1d:.1%}, 3d={accuracy_3d:.1%}")
                        
                        # Display successful predictions sample
                        if verification_results.get('successful_predictions'):
                            successful = verification_results['successful_predictions']
                            print(f"🎯 Sample results:")
                            for pred in successful[:3]:  # Show first 3
                                crypto_name = pred.get('crypto_name', 'Unknown')
                                direction = pred.get('predicted_direction', '?')
                                actual = pred.get('actual_direction', '?')
                                accuracy = pred.get('accuracy_score', 0)
                                print(f"   📈 {crypto_name}: {direction}→{actual} (acc: {accuracy:.3f})")
                        
                        # Display model feedback updates
                        if verification_results.get('model_feedback'):
                            feedback_updates = verification_results['model_feedback']
                            print(f"🤖 Model feedback: {len(feedback_updates)} ML updates applied")
                        
                        # 🔧 Save verification timestamp
                        self._update_last_verification_time()
                        
                        # 🔧 Increment verification counter for model feedback
                        self._increment_verification_count_for_feedback(verification_results['total_verified'])
                        
                    else:
                        print("ℹ️ Verification cycle completed (no mature predictions found)")
                else:
                    # Show when next verification will be
                    if hasattr(self, '_last_verification_time') and self._last_verification_time:
                        time_since = (time.time() - self._last_verification_time) / 3600
                        remaining = verification_interval_hours - time_since
                        print(f"⏰ Verification cycle skipped (next in {remaining:.1f}h)")
                    
            except Exception as e:
                print(f"⚠️ Verification failed: {e}")
                import traceback
                traceback.print_exc()

        # === 🤖 MODEL FEEDBACK BASATO SU VERIFICHE ACCUMULATE ===
        if (self.verification_system and 
            hasattr(self.verification_system, 'feedback_optimizer') and
            self.verification_system.feedback_optimizer):
            
            try:
                # 🔧 Check based on accumulated verifications, not cycles
                min_verifications_for_feedback = self.get_threshold('model_feedback', 'min_verifications_threshold', 15)
                accumulated_verifications = self._get_recent_verification_count()
                
                if accumulated_verifications >= min_verifications_for_feedback:
                    print(f"🔄 MODEL FEEDBACK TRIGGER - {accumulated_verifications} verifications accumulated")
                    print("-" * 50)
                    
                    # Run weight optimization for recent cryptos
                    updated_cryptos = 0
                    try:
                        # Get recently verified cryptos for weight updates
                        import sqlite3
                        conn = sqlite3.connect(self.verification_system.db_path)
                        cursor = conn.cursor()
                        
                        # Get recently verified unique cryptos
                        cursor.execute('''
                            SELECT DISTINCT crypto_id FROM predictions_optimized 
                            WHERE (verification_timestamp_1d >= datetime('now', '-24 hours') OR
                                verification_timestamp_3d >= datetime('now', '-24 hours'))
                            LIMIT 20
                        ''')
                        
                        recent_cryptos = [row[0] for row in cursor.fetchall()]
                        conn.close()
                        
                        print(f"🎯 Updating weights for {len(recent_cryptos)} recently verified cryptos...")
                        
                        for crypto_id in recent_cryptos:
                            try:
                                # Update weights for both horizons
                                for horizon in ['1d', '3d']:
                                    result = self.verification_system.update_model_weights_after_verification(crypto_id, horizon)
                                    if result and result.get('success'):
                                        updated_cryptos += 1
                                        weight_changes = result.get('weight_changes', {})
                                        if weight_changes:
                                            print(f"   📊 {crypto_id} {horizon}: {weight_changes}")
                                            
                            except Exception as e:
                                print(f"   ⚠️ Weight update failed for {crypto_id}: {e}")
                        
                        if updated_cryptos > 0:
                            print(f"✅ Model feedback completed: {updated_cryptos} weight updates")
                            # Reset counter after successful optimization
                            self._reset_verification_counter_for_feedback()
                        else:
                            print(f"⚠️ No weight updates applied")
                            
                    except Exception as e:
                        print(f"⚠️ Model feedback optimization failed: {e}")
                else:
                    remaining = min_verifications_for_feedback - accumulated_verifications
                    print(f"🤖 Model feedback: {accumulated_verifications}/{min_verifications_for_feedback} verifications (need {remaining} more)")
                    
            except Exception as e:
                print(f"⚠️ Model feedback system error: {e}")
        
        # === 📧 EMAIL PROCESSING TIMING INDIPENDENTE (ogni 6 ore) ===
        print(f"\n🔍 EMAIL DEBUG - Starting email section...")
        print(f"📧 Notifier exists: {'YES' if self.notifier else 'NO'}")

        if self.notifier:
            try:
                print(f"📧 Notifier type: {type(self.notifier)}")
                
                email_interval_hours = 6  # 🔧 NUOVO: 6 ore fisse per email
                print(f"📧 Email interval: {email_interval_hours} hours")
                
                # Debug timing check
                print(f"📧 Checking email timing...")
                has_last_time = hasattr(self, '_last_email_time')
                print(f"📧 Has _last_email_time attr: {has_last_time}")
                
                if not has_last_time:
                    print(f"📧 Creating _last_email_time attribute")
                    # ✅ FIX: Inizializza con timestamp corrente per aspettare 6h
                    self._last_email_time = time.time()
                    print(f"📧 Email timing initialized - next email in {email_interval_hours}h")
                else:
                    last_time = getattr(self, '_last_email_time', None)
                    if last_time is None:
                        # ✅ Se è None, inizializza con timestamp corrente
                        self._last_email_time = time.time()
                        print(f"📧 _last_email_time was None - initialized for {email_interval_hours}h delay")
                        last_time = self._last_email_time
                    
                    hours_since = (time.time() - last_time) / 3600
                    print(f"📧 Hours since last email: {hours_since:.1f}h")
                
                should_send_email = self._should_send_email_alert(email_interval_hours)
                print(f"📧 Should send email: {should_send_email}")
                
                if should_send_email:
                    print(f"📧 EMAIL ALERT CYCLE - Independent timing (every {email_interval_hours}h)")
                    print("-" * 50)
                    
                    # 🔧 USA LA VERSIONE SAFE CON CONVERSIONE BYTES
                    print(f"📧 Calling process_email_alerts_automatically_safe()...")
                    email_sent = self.process_email_alerts_automatically_safe()
                    print(f"📧 Email sent result: {email_sent}")
                    
                    if email_sent:
                        print("✅ Automatic email sent successfully!")
                        print(f"📧 Updating last email time...")
                        self._update_last_email_time()
                        print(f"📧 Last email time updated")
                    else:
                        print("📧 No email sent - checking detailed reason...")
                        
                        # Additional debug info
                        if hasattr(self, 'database') and self.database:
                            try:
                                import sqlite3
                                conn = sqlite3.connect(self.database.db_path)
                                cursor = conn.cursor()
                                
                                cursor.execute('''
                                    SELECT COUNT(*) FROM predictions_optimized 
                                    WHERE prediction_timestamp >= datetime('now', '-6 hours')
                                ''')
                                recent_count = cursor.fetchone()[0]
                                
                                cursor.execute('''
                                    SELECT COUNT(*) FROM predictions_optimized 
                                    WHERE prediction_timestamp >= datetime('now', '-6 hours')
                                    AND (confidence_1d >= 0.65 OR confidence_3d >= 0.65)
                                ''')
                                high_conf_count = cursor.fetchone()[0]
                                
                                conn.close()
                                
                                print(f"📊 Recent predictions (6h): {recent_count}")
                                print(f"🎯 High confidence predictions (6h): {high_conf_count}")
                                
                                if recent_count == 0:
                                    print("❌ NO RECENT PREDICTIONS - This is why no email was sent")
                                elif high_conf_count == 0:
                                    print("❌ NO HIGH CONFIDENCE PREDICTIONS - This is why no email was sent")
                                else:
                                    print("❓ Database has predictions but email still failed - check notifier methods")
                                    
                            except Exception as db_e:
                                print(f"❌ Database debug failed: {db_e}")
                                
                else:
                    # Debug why timing failed
                    if hasattr(self, '_last_email_time') and self._last_email_time is not None:
                        hours_since = (time.time() - self._last_email_time) / 3600
                        print(f"⏰ Email cycle skipped - only {hours_since:.1f}h since last email (need {email_interval_hours}h)")
                    else:
                        print(f"⏰ Email cycle skipped - _last_email_time not properly initialized")
                        
            except Exception as e:
                print(f"⚠️ Email processing failed with exception: {e}")
                print(f"⚠️ Exception type: {type(e)}")
                import traceback
                traceback.print_exc()
                
        else:
            print("❌ EMAIL SECTION SKIPPED - Notifier is None")
            
            # Debug why notifier is None
            gmail_user = self.get_threshold('email', 'gmail_user', '')
            gmail_password = self.get_threshold('email', 'gmail_app_password', '')
            
            print(f"📧 Gmail user configured: {'YES' if gmail_user else 'NO'}")
            print(f"📧 Gmail password configured: {'YES' if gmail_password else 'NO'}")
            
            if gmail_user:
                print(f"📧 Gmail user: {gmail_user}")
            
            if not gmail_user or not gmail_password:
                print("❌ EMAIL CREDENTIALS MISSING - This is why notifier is None")

        print(f"🔍 EMAIL DEBUG - End of email section\n")
        
        # === 🪙 BITCOIN BENCHMARK UPDATE ===
        if self.bitcoin_benchmark:
            try:
                # Force refresh Bitcoin data every few cycles
                force_refresh = (self.cycle_count % 4 == 1)
                btc_data = self.bitcoin_benchmark.get_real_bitcoin_data(force_refresh=force_refresh)
                
                if btc_data is not None:
                    self.bitcoin_benchmark.show_real_bitcoin_dashboard()
                else:
                    print("⚠️ Bitcoin data not available")
                    
            except Exception as e:
                print(f"⚠️ Bitcoin benchmark update failed: {e}")
        
        # === 🎯 MAIN ANALYSIS ===
        target_cryptos = self.get_target_cryptos_optimized()
        
        if not target_cryptos:
            print("❌ No target cryptos available - skipping cycle")
            return
        
        batch_size = self.get_threshold('system', 'block_size', 25)
        successful_analyses = 0
        total_predictions = 0
        
        for i in range(0, len(target_cryptos), batch_size):
            if not self.running:
                break
            
            batch = target_cryptos[i:i + batch_size]
            print(f"\n📦 Processing batch {i//batch_size + 1}: cryptos {i+1}-{min(i+batch_size, len(target_cryptos))}")
            
            for crypto_id, crypto_name, market_cap_rank in batch:
                if not self.running:
                    break
                
                if self.paused:
                    print("⏸️ System paused - waiting...")
                    while self.paused and self.running:
                        time.sleep(5)
                
                # Analyze single crypto
                result = self.analyze_crypto_optimized(crypto_id, crypto_name, market_cap_rank)
                
                if result:
                    successful_analyses += 1
                    predictions = result.get('predictions', {})
                    total_predictions += len(predictions)
                
                self.stats['total_cryptos_analyzed'] += 1
                
                # Small delay between analyses
                time.sleep(1)
        
        # === 💾 SAVE MODEL WEIGHTS ===
        if (self.verification_system and 
            hasattr(self.verification_system, 'feedback_optimizer') and
            self.verification_system.feedback_optimizer):
            try:
                self.verification_system.feedback_optimizer._save_model_weights()
            except Exception as e:
                print(f"⚠️ Failed to save model weights: {e}")
        
        # 🆕🆕🆕 AGGIUNGI QUI - SIGNAL GENERATION 🆕🆕🆕
        # === 🎯 SIGNAL GENERATION ===
        if successful_analyses > 0:
            print(f"✅ Generated {total_predictions} predictions from {successful_analyses} successful analyses")
            
            # 🎯 GENERA SEGNALI TRADING
            if hasattr(self, 'signal_generator') and self.signal_generator:
                try:
                    print(f"\n🎯 GENERATING TRADING SIGNALS...")
                    print("-" * 50)
                    
                    signals = self.signal_generator.generate_signals(
                        hours_back=4,  # Usa predizioni ultime 4h
                        max_signals=12
                    )
                    
                    if signals:
                        # Enhanced export con dati verifiche
                        if hasattr(self.signal_generator, 'advanced_export'):
                            filename = self.signal_generator.advanced_export.export_signals_advanced(
                                signals, 
                                metadata={'cycle': self.cycle_count, 'system_version': 'SSD_Optimized'}
                            )
                            print(f"📁 Advanced export: {filename}")
                            
                            # Optional: Create trading bot export
                            bot_file = self.signal_generator.advanced_export.create_trading_bot_export(signals)
                            if bot_file:
                                print(f"🤖 Bot export: {bot_file}")
                        self.stats['signals_generated'] = self.stats.get('signals_generated', 0) + len(signals)
                        print(f"🚨 Generated {len(signals)} actionable trading signals")
                        
                        # Mostra summary dei segnali più forti
                        strong_signals = [s for s in signals if s['signal_strength'] > 1.0]
                        if strong_signals:
                            print(f"🔥 {len(strong_signals)} STRONG signals detected:")
                            for signal in strong_signals[:3]:  # Top 3
                                print(f"   🎯 {signal['signal_type']} {signal['crypto_name']} "
                                    f"({signal['predicted_change']:+.1%}, Risk: {signal['risk_score']})")
                        
                        # Export automatico ogni 6h (12 cicli se cicli = 30min)
                        if self.cycle_count % 12 == 0:
                            try:
                                filename = self.signal_generator.export_signals_to_json(signals)
                                print(f"📁 Signals exported: {filename}")
                            except Exception as export_error:
                                print(f"⚠️ Export error: {export_error}")
                        
                        # Aggiorna stats per cycle summary
                        self.stats['last_signals_generated'] = len(signals)
                        self.stats['last_signal_timestamp'] = datetime.now().isoformat()
                        
                    else:
                        print("ℹ️ No trading signals generated (criteria not met)")
                        self.stats['last_signals_generated'] = 0
                        
                except Exception as e:
                    print(f"⚠️ Signal generation error: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("ℹ️ Signal Generator not available")

        # === 📊 CYCLE SUMMARY ===
        cycle_duration = time.time() - cycle_start
        
        print(f"\n{'='*80}")
        print(f"📊 OPTIMIZED CYCLE #{self.cycle_count} COMPLETED")
        print(f"⏱️ Duration: {cycle_duration/60:.1f} minutes")
        print(f"✅ Successful: {successful_analyses}/{len(target_cryptos)}")
        print(f"📈 Success rate: {successful_analyses/len(target_cryptos)*100:.1f}%")
        print(f"🔮 Predictions generated: {total_predictions}")
        print(f"🎯 High confidence: {self.stats['high_confidence_predictions']}")
        
        # API and system stats
        if hasattr(self, 'api') and hasattr(self.api, 'get_api_stats'):
            api_stats = self.api.get_api_stats()
            print(f"🦎 API calls: {api_stats['daily_calls']}/{api_stats['max_calls_per_day']}")
            print(f"💾 Cache hits: {api_stats['cache_size']} items")
        
        print(f"🔍 Verifications: {self.stats['verification_completed']}")
        print(f"💾 SSD Storage: {self.config.get('storage', {}).get('base_directory', 'D:/CryptoSystem')}")
        
        # 🔧 NUOVO: Show timing status for independent systems
        print(f"\n⏰ TIMING STATUS:")
        print(f"   🔍 Last verification: {self._get_last_verification_time_str()}")
        print(f"   📧 Last email: {self._get_last_email_time_str()}")
        print(f"   🤖 Verifications for feedback: {self._get_recent_verification_count()}")
        
        print(f"{'='*80}")
        
        # Save cycle stats
        if self.database:
            self._save_cycle_stats_optimized(cycle_duration, successful_analyses, len(target_cryptos))


    # 🔄 SOSTITUISCI METODO COMPLETO: get_target_cryptos_optimized
    def get_target_cryptos_optimized(self) -> List[Tuple[str, str, int]]:
        """🎯 Ottiene lista cryptos target - FIX COMPLETO per errore 'date'"""
        print("🎯 Fetching top 750 cryptos...")
        
        try:
            # 1. Usa chiamata API sicura
            print("🛡️ Safe API call: Top cryptos page 1")
            
            # Parametri per CoinGecko API
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': 250,  # Massimo per pagina
                'page': 1,
                'sparkline': False,
                'price_change_percentage': '1h,24h,7d,14d,30d'
            }
            
            all_cryptos = []
            max_pages = 4  # Per ottenere ~1000 cryptos
            
            for page in range(1, max_pages + 1):
                try:
                    params['page'] = page
                    print(f"   📄 Fetching page {page}...")
                    
                    # ✅ Versione corretta
                    response = self.api.safe_api_call(
                        "https://api.coingecko.com/api/v3/coins/markets",
                        params=params,
                        description=f"Top cryptos page {page}",
                        allow_fallback=True
                    )
                    
                    if not response:
                        print(f"   ⚠️ No response from API for page {page}")
                        continue
                    
                    # ✅ FIX PRINCIPALE: response È DIRETTAMENTE UNA LISTA, NON UN DICT
                    page_data = response if isinstance(response, list) else []
                    
                    if not page_data:
                        print(f"   ⚠️ No data received for page {page}")
                        break
                    
                    # 2. Processa dati con gestione sicura delle date
                    for crypto in page_data:
                        try:
                            crypto_id = crypto.get('id', '').strip()
                            name = crypto.get('name', '').strip()
                            market_cap_rank = crypto.get('market_cap_rank')
                            
                            # Validazione base
                            if not crypto_id or not name:
                                continue
                                
                            if market_cap_rank is None or market_cap_rank <= 0:
                                continue
                            
                            # ✅ FIX COMPLETO: Gestione sicura delle date
                            last_updated = crypto.get('last_updated')
                            if last_updated and isinstance(last_updated, str):
                                try:
                                    from datetime import datetime
                                    # Rimuovi 'Z' e converti
                                    last_updated_str = last_updated.replace('Z', '+00:00')
                                    last_updated_dt = datetime.fromisoformat(last_updated_str)
                                    # ✅ CONTROLLO CORRETTO: Ora abbiamo un datetime object
                                    if not isinstance(last_updated_dt, datetime):
                                        last_updated_dt = datetime.now()
                                except Exception as date_error:
                                    # Se la conversione fallisce, usa timestamp corrente
                                    print(f"   ⚠️ Date conversion failed for {crypto_id}: {date_error}")
                                    last_updated_dt = datetime.now()
                            else:
                                last_updated_dt = datetime.now()
                            
                            # Filtri di qualità
                            market_cap = crypto.get('market_cap', 0)
                            volume_24h = crypto.get('total_volume', 0)
                            
                            # Filtri minimi di liquidità
                            if market_cap and market_cap < 1_000_000:  # Min $1M market cap
                                continue
                                
                            if volume_24h and volume_24h < 10_000:  # Min $10k volume
                                continue
                            
                            # Aggiungi alla lista
                            all_cryptos.append((crypto_id, name, market_cap_rank))
                            
                        except Exception as crypto_error:
                            print(f"   ⚠️ Error processing crypto {crypto.get('id', 'unknown')}: {crypto_error}")
                            continue
                    
                    print(f"   ✅ Page {page}: {len(page_data)} cryptos processed")
                    
                except Exception as page_error:
                    print(f"   ❌ Error fetching page {page}: {page_error}")
                    continue
            
            if not all_cryptos:
                print("❌ No cryptos fetched - using fallback list")
                return self._get_fallback_crypto_list()
            
            # 3. Ordina per market cap rank e limita a 750
            all_cryptos.sort(key=lambda x: x[2])  # Ordina per market_cap_rank
            final_list = all_cryptos[:1000]
            
            print(f"✅ Successfully fetched {len(final_list)} target cryptos")
            
            # 4. Salva lista in cache per recovery
            self._cache_target_cryptos(final_list)
            
            return final_list
            
        except Exception as e:
            print(f"❌ Error fetching target cryptos: {str(e)}")
            print("🔄 Attempting to use cached crypto list...")
            
            # Fallback a lista cached o predefinita
            cached_list = self._get_cached_crypto_list()
            if cached_list:
                print(f"✅ Using cached list with {len(cached_list)} cryptos")
                return cached_list
            else:
                print("🎯 Using hardcoded fallback list")
                return self._get_fallback_crypto_list()

    # 🆕 AGGIUNGI NUOVO METODO: _get_fallback_crypto_list
    def _get_fallback_crypto_list(self) -> List[Tuple[str, str, int]]:
        """🛡️ Lista fallback hardcoded per casi di emergenza"""
        fallback_cryptos = [
            ('bitcoin', 'Bitcoin', 1),
            ('ethereum', 'Ethereum', 2),
            ('tether', 'Tether', 3),
            ('binancecoin', 'BNB', 4),
            ('solana', 'Solana', 5),
            ('usd-coin', 'USDC', 6),
            ('xrp', 'XRP', 7),
            ('dogecoin', 'Dogecoin', 8),
            ('cardano', 'Cardano', 9),
            ('avalanche-2', 'Avalanche', 10),
            ('tron', 'TRON', 11),
            ('shiba-inu', 'Shiba Inu', 12),
            ('chainlink', 'Chainlink', 13),
            ('polygon', 'Polygon', 14),
            ('wrapped-bitcoin', 'Wrapped Bitcoin', 15),
            ('polkadot', 'Polkadot', 16),
            ('internet-computer', 'Internet Computer', 17),
            ('multi-collateral-dai', 'Dai', 18),
            ('litecoin', 'Litecoin', 19),
            ('near', 'NEAR Protocol', 20),
            ('uniswap', 'Uniswap', 21),
            ('ethereum-classic', 'Ethereum Classic', 22),
            ('cosmos', 'Cosmos', 23),
            ('aptos', 'Aptos', 24),
            ('filecoin', 'Filecoin', 25),
            ('cronos', 'Cronos', 26),
            ('stellar', 'Stellar', 27),
            ('monero', 'Monero', 28),
            ('arbitrum', 'Arbitrum', 29),
            ('vechain', 'VeChain', 30)
        ]
        
        print(f"🛡️ Fallback list loaded with {len(fallback_cryptos)} major cryptos")
        return fallback_cryptos

    # 🆕 AGGIUNGI NUOVO METODO: _cache_target_cryptos
    def _cache_target_cryptos(self, crypto_list: List[Tuple[str, str, int]]):
        """💾 Salva lista cryptos in cache per recovery"""
        try:
            cache_file = "D:/CryptoSystem/cache/target_cryptos_cache.json"
            
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'count': len(crypto_list),
                'cryptos': [
                    {
                        'id': crypto_id,
                        'name': name,
                        'rank': rank
                    }
                    for crypto_id, name, rank in crypto_list
                ]
            }
            
            Path("D:/CryptoSystem/cache").mkdir(exist_ok=True, parents=True)
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
            print(f"💾 Cached {len(crypto_list)} cryptos for recovery")
            
        except Exception as e:
            print(f"⚠️ Failed to cache crypto list: {e}")

    # 🆕 AGGIUNGI NUOVO METODO: _get_cached_crypto_list
    def _get_cached_crypto_list(self) -> List[Tuple[str, str, int]]:
        """📂 Recupera lista cryptos dalla cache"""
        try:
            cache_file = "D:/CryptoSystem/cache/target_cryptos_cache.json"
            
            if not Path(cache_file).exists():
                return []
            
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Verifica che la cache non sia troppo vecchia (max 24h)
            cache_time = datetime.fromisoformat(cache_data['timestamp'])
            if (datetime.now() - cache_time).total_seconds() > 24 * 3600:
                print("⚠️ Cache too old, not using")
                return []
            
            crypto_list = [
                (crypto['id'], crypto['name'], crypto['rank'])
                for crypto in cache_data['cryptos']
            ]
            
            return crypto_list
            
        except Exception as e:
            print(f"⚠️ Failed to load cached crypto list: {e}")
            return []

    def _should_use_cache(self, cache_key):
        """💾 Check if should use cache - FIXED VERSION"""
        try:
            # Get cache duration safely
            try:
                cache_duration_seconds = self.cache_duration
            except Exception:
                cache_duration_seconds = self.config.get('api', {}).get('cache_duration_hours', 48) * 3600
            
            # Check memory cache first
            if hasattr(self, 'cache') and cache_key in self.cache:
                cache_time, cache_data = self.cache[cache_key]
                time_diff = time.time() - cache_time
                
                if time_diff < cache_duration_seconds:
                    return True, cache_data
                else:
                    # Remove expired memory cache
                    del self.cache[cache_key]
            
            # Check SSD cache if available
            if hasattr(self, '_load_ssd_cache'):
                try:
                    use_ssd, ssd_data = self._load_ssd_cache(cache_key)
                    if use_ssd:
                        # Load into memory cache
                        if not hasattr(self, 'cache'):
                            self.cache = {}
                        self.cache[cache_key] = (time.time(), ssd_data)
                        return True, ssd_data
                except Exception as e:
                    print(f"     ⚠️ SSD cache error: {e}")
            
            # No valid cache found
            return False, None
            
        except Exception as e:
            print(f"     ⚠️ Cache check error: {e}")
            return False, None

    def get_crypto_data_optimized_fixed(self, crypto_id, days=None):
        """📊 Get crypto data with FIXED cache handling"""
        
        if days is None:
            days = 270

        # ✅ FIX: Verifica che API sia disponibile
        if not hasattr(self, 'api') or self.api is None:
            print(f"❌ API non disponibile per {crypto_id}")
            return None
        
        cache_key = f"{crypto_id}_data_{days}d"
        
        # Check cache first using FIXED method
        use_cache, cached_data = self._should_use_cache(cache_key)
        if use_cache:
            print(f"     💾 Cache hit for {crypto_id} (safe mode)")
            return cached_data
        
        # Fallback sequence più conservativa
        fallback_sequence = [days, min(days, 180), min(days, 90), 30]
        
        for attempt_days in fallback_sequence:
            try:
                endpoint = f"{self.api.base_url}/coins/{crypto_id}/market_chart"
                params = {
                    'vs_currency': 'usd',
                    'days': attempt_days,
                    'interval': 'daily'
                }
                
                # USA SAFE API CALL invece di _make_robust_api_call
                data = self.api.safe_api_call(
                    endpoint, 
                    params, 
                    f"{crypto_id} data ({attempt_days}d)",
                    allow_fallback=True
                )
                
                if data and 'prices' in data:
                    df = self._process_market_data(data, crypto_id)
                    if df is not None and len(df) >= 30:
                        self._cache_response(cache_key, df)
                        print(f"     ✅ {crypto_id}: {len(df)} days loaded (safe)")
                        return df
                
            except Exception as e:
                print(f"     ❌ Safe API error with {attempt_days}d: {e}")
                continue
        
        print(f"     ❌ All safe attempts failed for {crypto_id}")
        return None

    def process_predictions_optimized(self, prediction_data):
        """🔄 Process predictions from optimized ML system"""
        try:
            crypto_name = prediction_data['crypto_name']
            predictions = prediction_data['predictions']
            current_price = prediction_data['current_price']
            
            print(f"     📊 Processing predictions for {crypto_name}...")
            
            # ✅ NUOVO: Gestisci il formato delle predizioni del sistema ottimizzato
            high_confidence_predictions = 0
            
            for horizon_key, pred_data in predictions.items():
                try:
                    if not isinstance(pred_data, dict):
                        continue
                    
                    predicted_change = pred_data.get('predicted_change', 0)
                    predicted_price = pred_data.get('predicted_price', 0)
                    confidence = pred_data.get('confidence', 0)
                    quality_score = pred_data.get('quality_score', 0)
                    
                    # Fix predicted_price if needed
                    if predicted_price <= 0 and current_price > 0:
                        predicted_price = current_price * (1 + predicted_change)
                    
                    # Update prediction data
                    pred_data['predicted_price'] = predicted_price
                    
                    # Display prediction
                    direction = "📈" if predicted_change > 0 else "📉"
                    print(f"     {direction} {horizon_key}: {predicted_change:+.3%} "
                        f"(${current_price:.4f} → ${predicted_price:.4f}) "
                        f"conf: {confidence:.1%}, quality: {quality_score:.1%}")
                    
                    # Check if high confidence
                    confidence_threshold = self.get_threshold('quality_thresholds', f'confidence_threshold_{horizon_key}', 0.60)
                    if confidence >= confidence_threshold:
                        high_confidence_predictions += 1
                    
                except Exception as e:
                    print(f"     ⚠️ Error processing {horizon_key} prediction: {e}")
            
            # Update stats
            self.stats['successful_predictions_1d'] += 1 if '1d' in predictions else 0
            self.stats['successful_predictions_3d'] += 1 if '3d' in predictions else 0
            self.stats['high_confidence_predictions'] += high_confidence_predictions
            
            # ✅ NUOVO: Salva nel database ottimizzato
            if self.database:
                try:
                    self.database.save_prediction_data_optimized(prediction_data)
                except Exception as e:
                    print(f"     ⚠️ Database save failed: {e}")
            
            # ✅ NUOVO: Genera alerts ottimizzati
            if high_confidence_predictions > 0:
                try:
                    self._generate_optimized_alerts(prediction_data)
                except Exception as e:
                    print(f"     ⚠️ Alert generation failed: {e}")
            
            return True
            
        except Exception as e:
            print(f"     ❌ Prediction processing failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _generate_optimized_alerts(self, prediction_data):
        """🚨 Generate alerts using optimized thresholds - FIXED"""
        try:
            if not self.notifier:
                return
            
            predictions = prediction_data.get('predictions', {})
            
            for horizon_key, pred in predictions.items():
                confidence = pred.get('confidence', 0)
                magnitude = abs(pred.get('predicted_change', 0))
                quality_score = pred.get('quality_score', 0)
                
                # Get thresholds from JSON config
                alert_thresholds = self.config.get('alert_thresholds', {}).get(horizon_key, {})
                
                # Determine alert level using optimized thresholds
                high_thresh = alert_thresholds.get('high_priority', {})
                medium_thresh = alert_thresholds.get('medium_priority', {})
                watch_thresh = alert_thresholds.get('watch_priority', {})
                
                alert_type = None
                if (confidence >= high_thresh.get('confidence_min', 0.75) and 
                    magnitude >= high_thresh.get('magnitude_min', 0.06) and 
                    quality_score >= high_thresh.get('quality_score_min', 0.70)):
                    alert_type = 'high'
                elif (confidence >= medium_thresh.get('confidence_min', 0.65) and 
                    magnitude >= medium_thresh.get('magnitude_min', 0.04) and 
                    quality_score >= medium_thresh.get('quality_score_min', 0.60)):
                    alert_type = 'medium'
                elif (confidence >= watch_thresh.get('confidence_min', 0.60) and 
                    magnitude >= watch_thresh.get('magnitude_min', 0.025)):
                    alert_type = 'watch'
                
                if alert_type:
                    # ✅ FIX: Use the correct notifier method name
                    try:
                        # Try the optimized method first
                        if hasattr(self.notifier, 'add_dual_alert_optimized'):
                            self.notifier.add_dual_alert_optimized(alert_type, prediction_data, horizon_key)
                        elif hasattr(self.notifier, 'add_dual_alert_fixed'):
                            self.notifier.add_dual_alert_fixed(alert_type, prediction_data, horizon_key)
                        elif hasattr(self.notifier, 'add_alert'):
                            # Fallback to basic alert method
                            alert_data = {
                                'crypto_name': prediction_data.get('crypto_name'),
                                'crypto_id': prediction_data.get('crypto_id'),
                                'current_price': prediction_data.get('current_price'),
                                'predicted_change': pred.get('predicted_change'),
                                'predicted_price': pred.get('predicted_price'),
                                'confidence': confidence,
                                'quality_score': quality_score,
                                'horizon': horizon_key,
                                'alert_type': alert_type
                            }
                            self.notifier.add_alert(alert_data)
                        else:
                            print(f"     ⚠️ No compatible alert method found in notifier")
                        
                        print(f"     🚨 {alert_type.upper()} alert generated for {horizon_key}")
                        
                    except Exception as e:
                        print(f"     ⚠️ Alert method error: {e}")
            
        except Exception as e:
            print(f"     ⚠️ Alert generation error: {e}")

    def fetch_historical_data_optimized(self, crypto_id, crypto_name, market_cap_rank):
        """📊 Fetch historical data compatible with new ML system"""
        try:
            # Determine lookback days based on rank
            lookback_config = self.config.get('lookback_strategy', {})
            if market_cap_rank <= lookback_config.get('top_crypto_rank_threshold', 50):
                lookback_days = lookback_config.get('lookback_premium', 270)
            else:
                lookback_days = lookback_config.get('lookback_standard', 180)
            
            print(f"     📊 Fetching {lookback_days}d data for {crypto_name}")
            
            # ✅ CRITICO: Usa il metodo del nuovo ML system per garantire compatibilità
            if self.ml_system:
                historical_data = self.ml_system.fetch_crypto_data_compatible(crypto_id, lookback_days)
            else:
                # Fallback al metodo esistente ma assicurati che 'price' sia presente
                historical_data = self.fetch_market_data_from_api(crypto_id, lookback_days)
                
                # ✅ VERIFICA CRITICA: 'price' column deve essere presente
                if historical_data is not None and 'price' not in historical_data.columns:
                    print(f"     ❌ CRITICAL: 'price' column missing from {crypto_name} data")
                    return None
            
            if historical_data is None or len(historical_data) < 60:
                print(f"     ⚠️ Insufficient data for {crypto_name}: {len(historical_data) if historical_data is not None else 0} days")
                return None
            
            # ✅ VERIFICA FINALE
            print(f"     ✅ Fetched {len(historical_data)} days of data for {crypto_name}")
            print(f"     💰 Price range: ${historical_data['price'].min():.4f} - ${historical_data['price'].max():.4f}")
            
            return historical_data
            
        except Exception as e:
            print(f"     ❌ Data fetch error for {crypto_name}: {e}")
            return None

    def analyze_crypto_optimized(self, crypto_id, crypto_name, market_cap_rank):
        """🔍 Analyze single crypto with optimized ML system - NO DUPLICATION"""
        try:
            print(f"\n🔍 Analyzing {crypto_name} (#{market_cap_rank})...")
            
            # ✅ NOVO: Usa il sistema ML ottimizzato
            if not self.ml_system:
                print("⚠️ ML System not available")
                return None
            
            # 1. Fetch e process data
            historical_data = self.fetch_historical_data_optimized(crypto_id, crypto_name, market_cap_rank)
            if historical_data is None or len(historical_data) < 60:
                print(f"     ⚠️ Insufficient data for {crypto_name}")
                return None
            
            # ✅ FIX: Train models ONLY ONCE per crypto
            training_success = False
            try:
                # Check if models already exist
                model_key_1d = f"{crypto_id}_1d"
                model_key_3d = f"{crypto_id}_3d"
                
                models_1d_exist = (model_key_1d in self.ml_system.models_1d and 
                                len(self.ml_system.models_1d[model_key_1d]) > 0)
                models_3d_exist = (model_key_3d in self.ml_system.models_3d and 
                                len(self.ml_system.models_3d[model_key_3d]) > 0)
                
                if models_1d_exist and models_3d_exist:
                    print(f"     ♻️ Using existing models for {crypto_name}")
                    training_success = True
                else:
                    print(f"     🚂 Training new models for {crypto_name}...")
                    # Train models only if they don't exist
                    success_1d = models_1d_exist or self.ml_system.train_compatible_models(crypto_id, 1)
                    success_3d = models_3d_exist or self.ml_system.train_compatible_models(crypto_id, 3)
                    training_success = success_1d or success_3d
                    
                    if training_success:
                        print(f"     ✅ Model training: 1d={'✅' if success_1d else '❌'}, 3d={'✅' if success_3d else '❌'}")
                    else:
                        print(f"     ⚠️ No models trained for {crypto_name}")
            
            except Exception as e:
                print(f"     ⚠️ Training error for {crypto_name}: {e}")
            
            # 3. Generate predictions
            predictions = None
            if training_success:
                try:
                    predictions = self.ml_system.predict_dual_optimized(crypto_id, historical_data, market_cap_rank)
                    
                    if predictions:
                        print(f"     🔮 Generated predictions: {list(predictions.keys())}")
                    else:
                        print(f"     ⚠️ No predictions generated for {crypto_name}")
                
                except Exception as e:
                    print(f"     ⚠️ Prediction error for {crypto_name}: {e}")
            
            # 4. Process predictions
            if predictions:
                prediction_data = {
                    'crypto_id': crypto_id,
                    'crypto_name': crypto_name,
                    'market_cap_rank': market_cap_rank,
                    'current_price': historical_data['price'].iloc[-1] if 'price' in historical_data.columns else 0,
                    'predictions': predictions,
                    'timestamp': datetime.now()
                }
                
                # Save and process alerts
                self.process_predictions_optimized(prediction_data)
                return prediction_data
            
            return None
            
        except Exception as e:
            print(f"     ❌ Analysis failed for {crypto_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _manage_memory_optimized(self):
        """🧠 Advanced memory management for ML system"""
        try:
            current_memory = psutil.virtual_memory().percent
            memory_threshold = self.get_threshold('system', 'memory_threshold_percent', 80)
            
            print(f"💾 Memory usage: {current_memory:.1f}%")
            
            if current_memory > memory_threshold:
                print(f"⚠️ High memory usage ({current_memory:.1f}%), running cleanup...")
                
                # ✅ NUOVO: Cleanup ML system memory
                if self.ml_system and hasattr(self.ml_system, 'cleanup_memory'):
                    try:
                        self.ml_system.cleanup_memory()
                        print("🧠 ML system memory cleaned")
                    except Exception as e:
                        print(f"⚠️ ML memory cleanup failed: {e}")
                
                # Standard cleanup
                if hasattr(self.api, 'cache'):
                    self.api.cache.clear()
                
                # Database cleanup
                if self.database and hasattr(self.database, 'vacuum_database'):
                    try:
                        self.database.vacuum_database()
                        print("🗃️ Database vacuumed")
                    except Exception as e:
                        print(f"⚠️ Database vacuum failed: {e}")
                
                # Force garbage collection
                import gc
                gc.collect()
                
                new_memory = psutil.virtual_memory().percent
                print(f"✅ Memory after cleanup: {new_memory:.1f}% (freed: {current_memory-new_memory:.1f}%)")
            
        except Exception as e:
            print(f"⚠️ Memory management error: {e}")

    def _cleanup_optimized(self):
        """🧹 Enhanced cleanup for optimized system"""
        try:
            print("🧹 Running optimized cleanup...")
            
            # ✅ NUOVO: Cleanup ML system
            if self.ml_system:
                try:
                    # Save models if needed
                    if hasattr(self.ml_system, 'save_models_compatible'):
                        self.ml_system.save_models_compatible()
                        print("💾 ML models saved")
                    
                    # Cleanup ML memory
                    if hasattr(self.ml_system, 'cleanup_memory'):
                        self.ml_system.cleanup_memory()
                        print("🧠 ML system cleaned up")
                        
                except Exception as e:
                    print(f"⚠️ ML system cleanup failed: {e}")
            
            # Standard cleanup (existing code can remain)
            if self.database:
                try:
                    self.database.close()
                    print("🗃️ Database closed")
                except Exception as e:
                    print(f"⚠️ Database close failed: {e}")
            
            # Clear caches
            if hasattr(self.api, 'cache'):
                self.api.cache.clear()
            
            # Final garbage collection
            import gc
            gc.collect()
            
            print("✅ Optimized cleanup completed")
            
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")

    def _save_cycle_stats_optimized(self, cycle_duration, successful_analyses, total_cryptos):
        """💾 Save cycle statistics to database"""
        try:
            if not self.database:
                return
            
            cycle_data = {
                'cycle_number': self.cycle_count,
                'start_time': datetime.now() - timedelta(seconds=cycle_duration),
                'end_time': datetime.now(),
                'total_cryptos_analyzed': total_cryptos,
                'successful_analyses': successful_analyses,
                'predictions_generated_1d': self.stats.get('successful_predictions_1d', 0),
                'predictions_generated_3d': self.stats.get('successful_predictions_3d', 0),
                'elapsed_minutes': cycle_duration / 60,
                'api_calls_made': getattr(self.api, 'daily_calls', 0),
                'cache_hit_rate': 0.0,  # Calculate if needed
                'memory_usage_mb': psutil.virtual_memory().percent,
                'verifications_completed': self.stats.get('verification_completed', 0),
                'verification_accuracy': 0.0  # Calculate if needed
            }
            
            self.database.save_cycle_stats_optimized(cycle_data)
            print(f"💾 Cycle stats saved to SSD database")
            
        except Exception as e:
            print(f"⚠️ Failed to save cycle stats: {e}")

    def _display_prediction_summary_optimized(self, prediction_data):
        """📊 Display prediction summary ottimizzato con error handling"""
        try:
            crypto_name = prediction_data.get('crypto_name', 'Unknown')
            current_price = prediction_data.get('current_price', 0)
            predictions = prediction_data.get('predictions', {})
            
            print(f"     📊 {crypto_name} prediction summary:")
            print(f"     💰 Current price: ${current_price:,.4f}")
            
            # Display each horizon prediction safely
            for horizon_key, pred_data in predictions.items():
                try:
                    if isinstance(pred_data, dict):
                        predicted_change = pred_data.get('predicted_change', 0)
                        predicted_price = pred_data.get('predicted_price', 0)
                        confidence = pred_data.get('confidence', 0)
                        
                        # Fix predicted_price if needed
                        if predicted_price <= 0 and current_price > 0:
                            predicted_price = current_price * (1 + predicted_change)
                        
                        direction = "📈" if predicted_change > 0 else "📉"
                        print(f"     {direction} {horizon_key}: {predicted_change:+.3f} "
                            f"(${current_price:,.4f} → ${predicted_price:,.4f}) "
                            f"conf: {confidence:.1%}")
                    else:
                        print(f"     ⚠️ Invalid prediction data for {horizon_key}")
                        
                except Exception as e:
                    print(f"     ❌ Error displaying {horizon_key} prediction: {e}")
            
        except Exception as e:
            print(f"     ❌ Error in prediction summary display: {e}")
            import traceback
            traceback.print_exc()

    def get_system_status_optimized(self):
        """📊 Get comprehensive system status"""
        try:
            uptime = (time.time() - self.stats['start_time']) / 3600 if self.stats['start_time'] else 0
            
            return {
                'running': self.running,
                'paused': self.paused,
                'cycle_count': self.cycle_count,
                'uptime_hours': uptime,
                'total_cryptos_analyzed': self.stats['total_cryptos_analyzed'],
                'successful_predictions_1d': self.stats['successful_predictions_1d'],
                'successful_predictions_3d': self.stats['successful_predictions_3d'],
                'high_confidence_predictions': self.stats['high_confidence_predictions'],
                'verification_completed': self.stats['verification_completed'],
                'memory_usage_percent': psutil.virtual_memory().percent,
                'storage_base': self.config.get('storage', {}).get('base_directory', 'D:/CryptoSystem'),
                'api_calls_today': getattr(self.api, 'daily_calls', 0),
                'api_limit': getattr(self.api, 'max_calls_per_day', 1000),
                'last_cycle_time': getattr(self, 'last_cycle_time', None)
            }
        except Exception as e:
            print(f"❌ Error getting system status: {e}")
            return {'error': str(e)}
        
    def get_high_confidence_predictions_for_email(self):
        """📊 Get high confidence predictions with improved error handling"""
        
        if not self.database or not hasattr(self.database, 'db_path'):
            print("⚠️ Database not available for email predictions")
            return []
        
        try:
            # Get thresholds from config instead of hardcoding
            threshold_1d = self.get_threshold('quality_thresholds', 'confidence_threshold_1d', 0.60)
            threshold_3d = self.get_threshold('quality_thresholds', 'confidence_threshold_3d', 0.55)
            lookback_hours = self.get_threshold('email', 'lookback_hours_for_alerts', 6)
            
            print(f"📊 Searching predictions with confidence ≥{threshold_1d:.0%} (1d) / ≥{threshold_3d:.0%} (3d)")
            
            # Use context manager for database connection
            with sqlite3.connect(self.database.db_path) as conn:
                cursor = conn.cursor()
                
                # Improved query with better error handling
                cursor.execute('''
                    SELECT 
                        crypto_id, crypto_name, current_price,
                        confidence_1d, predicted_change_1d, predicted_price_1d,
                        confidence_3d, predicted_change_3d, predicted_price_3d,
                        market_regime, bitcoin_correlation,
                        datetime(timestamp) as pred_time
                    FROM predictions_optimized 
                    WHERE datetime(timestamp) >= datetime('now', '-{} hours')
                    AND (
                        (confidence_1d IS NOT NULL AND confidence_1d >= ?) OR 
                        (confidence_3d IS NOT NULL AND confidence_3d >= ?)
                    )
                    ORDER BY (COALESCE(confidence_1d, 0) + COALESCE(confidence_3d, 0)) DESC
                    LIMIT 50
                '''.format(lookback_hours), (threshold_1d, threshold_3d))
                
                predictions = cursor.fetchall()
                
            print(f"📊 Found {len(predictions)} high confidence predictions for email")
            return predictions
            
        except sqlite3.Error as e:
            print(f"❌ Database error getting predictions: {e}")
            return []
        except Exception as e:
            print(f"❌ General error getting predictions: {e}")
            return []
        
    def convert_db_predictions_to_alerts(self, db_predictions):
        """🔄 Converte predizioni database in formato alert per notifier"""
        
        formatted_alerts = []
        
        for pred in db_predictions:
            try:
                (crypto_id, crypto_name, current_price, conf_1d, change_1d, price_1d, 
                conf_3d, change_3d, price_3d, market_regime, bitcoin_corr, pred_time) = pred
                
                # Base prediction structure
                base_pred = {
                    'crypto_id': crypto_id or 'unknown',
                    'crypto_name': crypto_name or 'Unknown',
                    'current_price': float(current_price) if current_price else 0,
                    'market_regime': market_regime or 'unknown',
                    'bitcoin_correlation': float(bitcoin_corr) if bitcoin_corr else 0,
                    'prediction_time': pred_time,
                    'predictions': {}
                }
                
                # Process 1d prediction if high confidence
                if conf_1d and conf_1d >= 0.65:
                    # Fix predicted_price if missing/zero
                    if not price_1d or price_1d <= 0:
                        price_1d = current_price * (1 + (change_1d or 0))
                    
                    base_pred['predictions']['1d'] = {
                        'predicted_change': float(change_1d) if change_1d else 0,
                        'predicted_price': float(price_1d),
                        'confidence': float(conf_1d),
                        'direction': 'up' if (change_1d or 0) > 0 else 'down',
                        'magnitude': abs(change_1d or 0)
                    }
                    
                    # Add to alerts list
                    formatted_alerts.append(('1d', base_pred.copy()))
                
                # Process 3d prediction if high confidence
                if conf_3d and conf_3d >= 0.65:
                    # Fix predicted_price if missing/zero
                    if not price_3d or price_3d <= 0:
                        price_3d = current_price * (1 + (change_3d or 0))
                    
                    base_pred['predictions']['3d'] = {
                        'predicted_change': float(change_3d) if change_3d else 0,
                        'predicted_price': float(price_3d),
                        'confidence': float(conf_3d),
                        'direction': 'up' if (change_3d or 0) > 0 else 'down',
                        'magnitude': abs(change_3d or 0)
                    }
                    
                    # Add to alerts list  
                    formatted_alerts.append(('3d', base_pred.copy()))
                    
            except Exception as e:
                print(f"   ❌ Error converting prediction: {e}")
                continue
        
        return formatted_alerts

    def determine_alert_priority(self, horizon_data):
        """🎯 Determina priorità alert"""
        
        confidence = horizon_data.get('confidence', 0)
        magnitude = horizon_data.get('magnitude', 0)
        
        if confidence >= 0.75 and magnitude >= 0.04:
            return 'high'
        elif confidence >= 0.65 and magnitude >= 0.025:
            return 'medium'
        else:
            return 'watch'

    def process_email_alerts_automatically(self):
        """🚨 QUESTO È IL METODO PRINCIPALE - Processa alert per email automaticamente"""
        
        if not self.notifier:
            return False
        
        try:
            print("🚨 Checking for email alerts...")
            
            # 1. Check if it's time to send email (every 6 hours)
            if not self.notifier.should_send_summary():
                print("   ⏰ Not time for email yet (< 6 hours since last)")
                return False
            
            print("   ⏰ TIME TO SEND EMAIL! (>= 6 hours)")
            
            # 2. Get high confidence predictions from database
            db_predictions = self.get_high_confidence_predictions_for_email()
            
            if not db_predictions:
                print("   📊 No high confidence predictions found - no email")
                return False
            
            print(f"   📊 Found {len(db_predictions)} high confidence predictions")
            
            # 3. Convert to alert format
            formatted_alerts = self.convert_db_predictions_to_alerts(db_predictions)
            
            if not formatted_alerts:
                print("   🔄 No alerts after formatting")
                return False
            
            print(f"   ✅ Formatted {len(formatted_alerts)} alerts")
            
            # 4. Add alerts to notifier
            added_count = 0
            for horizon, prediction_data in formatted_alerts:
                try:
                    horizon_data = prediction_data['predictions'].get(horizon, {})
                    priority = self.determine_alert_priority(horizon_data)
                    
                    success = self.notifier.add_dual_alert_fixed(priority, prediction_data, horizon)
                    if success:
                        added_count += 1
                    
                except Exception as e:
                    print(f"   ❌ Error adding alert: {e}")
            
            print(f"   ✅ Added {added_count} alerts to notifier")
            
            # 5. Send email summary
            if added_count > 0:
                print("   📧 Sending email with real predictions...")
                success = self.notifier.send_6hour_dual_summary_fixed()
                
                if success:
                    print("   🎉 AUTOMATIC EMAIL SENT SUCCESSFULLY!")
                    
                    # Update stats
                    self.stats['emails_sent'] = self.stats.get('emails_sent', 0) + 1
                    self.stats['last_email_sent'] = datetime.now().isoformat()
                    
                    return True
                else:
                    print("   ❌ Email send failed")
                    return False
            else:
                print("   📧 No alerts to send")
                return False
                
        except Exception as e:
            print(f"❌ Error processing email alerts: {e}")
            import traceback
            traceback.print_exc()
            return False
    def debug_email_configuration(self):
        """🔍 Debug email configuration - trova il problema"""
        print("\n🔍 EMAIL CONFIGURATION DEBUG:")
        print("=" * 50)
        
        # 1. Check if notifier exists
        print(f"📧 Notifier object: {'✅ Exists' if self.notifier else '❌ None (not initialized)'}")
        
        # 2. Check configuration values
        gmail_user = self.get_threshold('email', 'gmail_user', '')
        gmail_password = self.get_threshold('email', 'gmail_app_password', '')
        
        print(f"📧 Gmail user: {'✅ Set' if gmail_user else '❌ Missing'}")
        print(f"📧 Gmail password: {'✅ Set' if gmail_password else '❌ Missing'}")
        
        if gmail_user:
            print(f"   User value: {gmail_user}")
        if gmail_password:
            print(f"   Password length: {len(gmail_password)} characters")
        
        # 3. Check JSON config structure
        email_config = self.config.get('email', {})
        print(f"📧 Email config section: {'✅ Found' if email_config else '❌ Missing'}")
        
        if email_config:
            print("📋 Email config contents:")
            for key, value in email_config.items():
                if 'password' in key.lower():
                    print(f"   {key}: {'*' * len(str(value)) if value else 'EMPTY'}")
                else:
                    print(f"   {key}: {value}")
        else:
            print("❌ No 'email' section found in JSON config")
        
        # 4. Check JSON file existence and structure
        print(f"\n📄 Config file: {getattr(self, 'config_file', 'Unknown')}")
        
        # 5. Show what needs to be fixed
        print("\n🛠️ TO FIX EMAIL:")
        if not gmail_user:
            print("   ❌ Add 'gmail_user' to email section in JSON config")
        if not gmail_password:
            print("   ❌ Add 'gmail_app_password' to email section in JSON config")
        if not email_config:
            print("   ❌ Add entire 'email' section to JSON config")
        
        if gmail_user and gmail_password:
            print("   ✅ Credentials look OK - check notifier initialization")
        
        print("=" * 50)

    def update_database_schema_for_signals(self):
        """🔧 Update database schema to support Signal Generator"""
        
        print("🔧 Updating database schema for Signal Generator...")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check and add verification_date column
                try:
                    cursor.execute("SELECT verification_date FROM verification_results LIMIT 1")
                    print("✅ verification_date column exists")
                except sqlite3.OperationalError:
                    print("➕ Adding verification_date column to verification_results...")
                    cursor.execute("""
                        ALTER TABLE verification_results 
                        ADD COLUMN verification_date DATE DEFAULT (date('now'))
                    """)
                    # Update existing records
                    cursor.execute("""
                        UPDATE verification_results 
                        SET verification_date = date(created_at)
                        WHERE verification_date IS NULL
                    """)
                    print("✅ verification_date column added")
                
                # Check and add method column  
                try:
                    cursor.execute("SELECT method FROM predictions_optimized LIMIT 1")
                    print("✅ method column exists")
                except sqlite3.OperationalError:
                    print("➕ Adding method column to predictions_optimized...")
                    cursor.execute("""
                        ALTER TABLE predictions_optimized 
                        ADD COLUMN method TEXT DEFAULT 'ML Ensemble'
                    """)
                    print("✅ method column added")
                
                conn.commit()
                print("✅ Database schema updated successfully")
                return True
                
        except Exception as e:
            print(f"❌ Database schema update failed: {e}")
            return False

# AGGIUNGI questa funzione nel file principale, PRIMA del main():

def integrate_with_main_system(main_system_instance):
    """🔗 Integration function for main system - VERSIONE CORRETTA"""
    
    try:
        # Controlla se il sistema principale ha una configurazione valida
        if not hasattr(main_system_instance, 'config') or not main_system_instance.config:
            print("⚠️ Main system config not found, using defaults...")
            base_dir = Path('D:/CryptoSystem')
        else:
            base_dir = Path(main_system_instance.config.get('base_directory', 'D:/CryptoSystem'))
        
        config_path = base_dir / 'shitcoin_config.json'
        
        # Create default config if not exists
        if not config_path.exists():
            # ✅ CONFIGURAZIONE COMPLETA CON ML
            default_shitcoin_config = {
                "base_directory": str(base_dir),
                "scan_interval_seconds": 300,
                "fast_scan_seconds": 60,
                "target_chains": ["ethereum", "solana", "bsc", "polygon"],
                "max_pairs_per_scan": 150,
                
                # API Configuration
                "moralis_api_key": "",  # Da configurare per ML completo
                "enable_moralis_fallback": True,
                
                # Notification settings
                "enable_notifications": True,
                "notification_email": main_system_instance.config.get('notification_email', '') if hasattr(main_system_instance, 'config') else '',
                
                # Detection thresholds
                "min_pump_confidence": 0.75,
                "min_prediction_confidence": 0.70,
                
                # Pump thresholds (VALORI TESTATI)
                "pump_thresholds": {
                    "quick_pump": {
                        "price_change_5m": 25.0,
                        "volume_multiplier": 3.0,
                        "min_liquidity": 50000,
                        "confidence": 0.85
                    },
                    "sustained_pump": {
                        "price_change_1h": 50.0,
                        "volume_multiplier": 2.0,
                        "min_liquidity": 100000,
                        "consistency_check": True,
                        "confidence": 0.90
                    },
                    "volume_explosion": {
                        "volume_multiplier": 5.0,
                        "price_change_min": 15.0,
                        "transaction_ratio": 2.0,
                        "confidence": 0.80
                    }
                },
                
                # ✅ ML SETTINGS
                "enable_predictions": True,
                "ml_continuous_learning": False,  # Start disabled for stability
                "prediction_tracking_enabled": True,
                
                # System settings
                "bitcoin_context_enabled": True,
                "dynamic_scanning": True,
                "enable_validation": True,
                "enable_deduplication": True
            }
            
            # Assicurati che la directory esista
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(default_shitcoin_config, f, indent=2)
            print(f"📄 Created enhanced shitcoin config: {config_path}")
        
        # ✅ IMPORT E CREAZIONE SCANNER CORRETTI
        scanner = ShitcoinDEXScanner(str(config_path), main_system_instance)
        
        # Store reference in main system for menu integration
        main_system_instance.shitcoin_scanner = scanner
        
        print("✅ Enhanced Shitcoin scanner integrated with main system")
        print(f"   🧠 ML System: {'✅ Active' if scanner.ml_enabled else '⚠️ Fallback mode'}")
        print(f"   📱 Telegram: {'✅ Active' if scanner.telegram_enabled else '❌ Disabled'}")
        print(f"   🚫 Deduplication: {'✅ Active' if scanner.dedup_enabled else '❌ Disabled'}")
        
        return scanner
        
    except ImportError as e:
        print(f"❌ Cannot import ShitcoinDEXScanner: {e}")
        print("   Verifica che shitcoin_dex_scanner.py sia nella directory corretta")
        return None
    except Exception as e:
        print(f"❌ Integration error: {e}")
        import traceback
        traceback.print_exc()
        return None

# ========================================================================================
# 🎯 HANDLER MENU ENHANCED
# ========================================================================================

async def handle_shitcoin_menu_choice(choice: str, main_system):
    """🎯 Handle shitcoin menu choices - VERSIONE ENHANCED"""
    
    try:
        # Initialize scanner if not exists
        if not hasattr(main_system, 'shitcoin_scanner') or main_system.shitcoin_scanner is None:
            print("🔄 Initializing enhanced shitcoin scanner...")
            scanner = integrate_with_main_system(main_system)
            if not scanner:
                print("❌ Failed to initialize shitcoin scanner")
                return
        
        # Ottieni riferimento scanner
        scanner = main_system.shitcoin_scanner
        
        if choice == '14':
            # ENHANCED continuous scanner
            print("🚀 Starting ENHANCED Shitcoin DEX Scanner...")
            print("   🧠 ML Predictions: Real machine learning models")
            print("   🟠 Bitcoin Context: Adaptive thresholds based on BTC market")
            print("   ⚡ Dynamic Scanning: Smart interval adjustment")
            print("   📊 Granular Data: Buyer/seller analysis every 5-15 minutes")
            print("   🎯 Conservative Detection: High-confidence alerts only")
            print("")
            
            # Use the enhanced scanner method
            await scanner.run_continuous_scanner()
            
        elif choice == '15':
            # Single ENHANCED scan with ML predictions
            print("🧪 Running ENHANCED single scan with ML predictions...")
            print("   📊 Collecting granular market data...")
            print("   🧠 Generating ML predictions...")
            print("   🎯 Analyzing pump signals...")
            print("")
            
            # Perform enhanced scan cycle
            alerts, predictions = await scanner._perform_scan_cycle()
            
            # Display results with enhanced formatting
            if alerts:
                print(f"\n🚨 IMMEDIATE PUMP ALERTS ({len(alerts)}):")
                for i, alert in enumerate(alerts, 1):
                    print(f"{i}. {alert.token_symbol} ({alert.chain.upper()})")
                    print(f"   💰 Price: ${alert.price_usd:.6f}")
                    print(f"   📈 Change: +{alert.price_change_5m:.1f}% (5min) | +{alert.price_change_1h:.1f}% (1h)")
                    print(f"   🔊 Volume: ${alert.volume_24h:,.0f} (x{alert.volume_change:.1f})")
                    print(f"   🎯 Confidence: {alert.pump_confidence:.1%}")
                    print(f"   🚨 Type: {alert.pump_type}")
                    
                    # Enhanced transaction data if available
                    if hasattr(alert, 'buys_1h') and alert.buys_1h > 0:
                        buy_ratio = alert.buys_1h / (alert.buys_1h + alert.sells_1h + 0.001)
                        print(f"   📊 B/S Ratio: {buy_ratio:.1%} ({alert.buys_1h}B/{alert.sells_1h}S)")
                    
                    print("")
            
            if predictions:
                print(f"\n🧠 ML PREDICTIONS ({len(predictions)}):")
                for i, pred in enumerate(predictions, 1):
                    print(f"{i}. {pred.token_symbol} ({pred.chain.upper()})")
                    print(f"   💰 Current Price: ${pred.current_price:.6f}")
                    print(f"   🎲 Pump Probability: {pred.pump_probability:.1%}")
                    print(f"   📈 Expected Gain: +{pred.expected_gain:.1f}%")
                    print(f"   ⏱️ Time Horizon: {pred.time_horizon}")
                    print(f"   🎯 ML Confidence: {pred.confidence:.1%}")
                    print(f"   ⚠️ Risk Level: {pred.risk_level}")
                    print(f"   🔧 Method: {pred.time_horizon_type}")
                    print("")
            
            # Process results
            if alerts or predictions:
                await scanner._process_predictions_and_alerts(alerts, predictions)
            else:
                print("✅ No significant signals detected in current market conditions")
            
        elif choice == '16':
            # Enhanced statistics with ML metrics
            print("📊 ENHANCED Shitcoin Scanner Statistics:")
            scanner.print_stats()
            
            # Additional ML statistics
            if scanner.ml_enabled and hasattr(scanner.pump_predictor, 'model_performance'):
                print(f"\n🧠 ML MODEL PERFORMANCE:")
                for model_name, perf in scanner.pump_predictor.model_performance.items():
                    print(f"   {model_name}:")
                    print(f"     Train Accuracy: {perf.get('train_accuracy', 0):.1%}")
                    print(f"     Test Accuracy: {perf.get('test_accuracy', 0):.1%}")
                    print(f"     Precision: {perf.get('precision', 0):.1%}")
            
        elif choice == '17':
            # Enhanced configuration display
            print("⚙️ ENHANCED Shitcoin Scanner Configuration:")
            scanner._print_system_status()
            
            print(f"\n🔧 DETAILED CONFIGURATION:")
            print(f"   📁 Config Path: {scanner.config.get('base_directory', 'Unknown')}")
            print(f"   ⏱️ Scan Interval: {scanner.scan_interval}s")
            print(f"   🔗 Target Chains: {', '.join(scanner.chains)}")
            print(f"   📊 Max Pairs/Scan: {scanner.max_pairs_per_scan}")
            print(f"   🧠 ML Enabled: {scanner.ml_enabled}")
            print(f"   🎯 Prediction Mode: {scanner.prediction_mode}")
            print(f"   📱 Notifications: {scanner.telegram_enabled}")
            print(f"   🚫 Deduplication: {scanner.dedup_enabled}")
            
        elif choice == '18':
            # Enhanced Bitcoin context analysis
            print("🟠 ENHANCED Bitcoin Market Context Analysis:")
            try:
                btc_context = await scanner.btc_analyzer.get_bitcoin_context()
                
                print(f"\n🟠 CURRENT BITCOIN CONTEXT:")
                print(f"   💰 BTC Price: ${btc_context.price:.2f}")
                print(f"   📈 24h Change: {btc_context.price_change_24h:+.2f}%")
                print(f"   😊 Market Sentiment: {btc_context.market_sentiment.upper()}")
                print(f"   🎯 Pump Sensitivity: {btc_context.pump_sensitivity:.2f}x")
                print(f"   🔍 Scan Recommendation: {btc_context.scan_recommendation.upper()}")
                print(f"   ⏰ Last Updated: {btc_context.last_updated.strftime('%H:%M:%S')}")
                
                # Explain what this means for scanning
                print(f"\n📋 IMPACT ON SCANNING:")
                if btc_context.pump_sensitivity < 1.0:
                    print(f"   🎯 More sensitive detection (lower thresholds)")
                    print(f"   🚀 Bullish BTC = More altcoin pumps expected")
                elif btc_context.pump_sensitivity > 1.0:
                    print(f"   🎯 Less sensitive detection (higher thresholds)")  
                    print(f"   🐻 Bearish BTC = Fewer altcoin pumps expected")
                else:
                    print(f"   🎯 Normal detection thresholds")
                    print(f"   😐 Neutral BTC = Standard altcoin activity")
                
            except Exception as e:
                print(f"❌ Error getting Bitcoin context: {e}")
            
        elif choice == '19':
            # Enhanced ML Performance analysis
            print("🤖 ENHANCED ML Prediction Performance Analysis:")
            
            if scanner.ml_enabled:
                print(f"\n🧠 ML SYSTEM STATUS:")
                print(f"   ✅ Models Trained: {scanner.pump_predictor.is_trained}")
                print(f"   📊 Feature Count: {len(scanner.pump_predictor.feature_columns)}")
                print(f"   🎯 Total Predictions: {scanner.total_predictions}")
                print(f"   ✅ Successful: {scanner.successful_predictions}")
                
                if scanner.total_predictions > 0:
                    success_rate = (scanner.successful_predictions / scanner.total_predictions) * 100
                    print(f"   📈 Success Rate: {success_rate:.1f}%")
                
                # Show recent predictions from database
                recent_predictions = scanner.get_recent_predictions(hours=24)
                if recent_predictions:
                    print(f"\n📊 RECENT PREDICTIONS (24h): {len(recent_predictions)}")
                    
                    # Show top 5 recent
                    for i, pred in enumerate(recent_predictions[:5], 1):
                        outcome = pred.get('outcome', 'pending')
                        status_emoji = '✅' if outcome == 'success' else '❌' if outcome == 'failure' else '🟡' if outcome == 'partial' else '⏳'
                        print(f"   {i}. {pred.get('token_symbol', 'Unknown')} - {status_emoji} {outcome}")
                        
            else:
                print("⚠️ ML system not available - running in fallback mode")
                print("   Install scikit-learn and configure Moralis API for full ML features")
        
        else:
            print(f"❌ Invalid choice: {choice}")
            print("   Valid choices: 14-19")
    
    except Exception as e:
        print(f"❌ Error handling shitcoin menu choice: {e}")
        import traceback
        traceback.print_exc()

def main():
    """🚀 Main function con prompt JSON - COMPLETE VERSION"""
    system = None
    
    try:
        print("🚀 SISTEMA CRYPTO CONTINUO OTTIMIZZATO - VERSIONE SSD")
        print("💾 Storage su SSD esterno per performance massime")
        print("📄 Configurazione completa tramite JSON")
        
        # Initialize system (con prompt per JSON)
        system = OptimizedCryptoContinuousSystemSSD()
        
        # Interactive menu loop
        while True:
            print("\n" + "="*70)
            print("📋 MENU INTERATTIVO SISTEMA SSD + SHITCOIN")
            print("="*70)
            print("1. 🟢 Avvia analisi continua ottimizzata")
            print("2. 🧪 Esegui ciclo di test singolo")
            print("3. 📊 Mostra stato sistema e performance")
            print("4. ⚙️ Mostra configurazione completa")
            print("5. 📧 Test notifiche email")
            print("6. 🦎 Mostra statistiche API e cache")
            print("7. 💾 Mostra statistiche database SSD")
            print("8. 🟠 Mostra dashboard Bitcoin benchmark")
            print("9. 🔍 Esegui test verifica")
            print("10. ⏸️ Pausa/Riprendi sistema")
            print("11. 🧹 Esegui ottimizzazione e cleanup")
            print("12. 📄 Ricarica configurazione JSON")
            print("13. 🔧 Debug email configuration")
            print("─" * 70)
            if SHITCOIN_SCANNER_AVAILABLE:
                print("🔥 ADVANCED SHITCOIN DEX SCANNER:")
                print("14. 🚀 Scanner continuo AVANZATO (ML + Bitcoin)")
                print("15. 🧪 Test scan singolo con predizioni")
                print("16. 📊 Statistiche avanzate scanner")
                print("17. ⚙️ Configurazione avanzata")
                print("18. 🟠 Analisi contesto Bitcoin")
                print("19. 🤖 Performance predizioni ML")
            else:
                print("⚠️ Shitcoin Scanner non disponibile")
            print("─" * 70)
            print("0. 🚪 Esci")
            print("="*70)
            
            choice = input("Seleziona opzione: ").strip()
            
            if choice == '1':
                # Start continuous analysis
                print("🚀 Avvio analisi continua SSD...")
                try:
                    system.run_continuous_optimized_ssd()
                except KeyboardInterrupt:
                    print("\n🛑 Sistema interrotto dall'utente")
                except Exception as e:
                    print(f"❌ Errore durante analisi continua: {e}")
                
            elif choice == '2':
                # Single test cycle
                print("🧪 Esecuzione ciclo test singolo...")
                try:
                    system.run_optimized_cycle()
                    print("✅ Ciclo completato")
                except Exception as e:
                    print(f"❌ Errore durante ciclo: {e}")
                
            elif choice == '3':
                # System status
                try:
                    status = system.get_system_status_optimized()
                    print(f"\n📊 STATO SISTEMA SSD:")
                    print(f"   🟢 In esecuzione: {'Sì' if status.get('running', False) else 'No'}")
                    print(f"   ⏸️ In pausa: {'Sì' if status.get('paused', False) else 'No'}")
                    print(f"   🔄 Cicli completati: {status.get('cycle_count', 0)}")
                    print(f"   ⏰ Uptime: {status.get('uptime_hours', 0):.1f} ore")
                    print(f"   📊 Crypto analizzate: {status.get('total_analyzed', 0)}")
                    print(f"   🎯 Predizioni ad alta confidenza: {status.get('high_confidence_predictions', 0)}")
                    print(f"   🔍 Verifiche completate: {status.get('verification_completed', 0)}")
                    print(f"   📧 Email inviate: {status.get('emails_sent', 0)}")
                    
                    # Additional performance stats
                    if hasattr(system, 'api') and hasattr(system.api, 'get_api_stats'):
                        api_stats = system.api.get_api_stats()
                        print(f"   🦎 API calls oggi: {api_stats.get('daily_calls', 0)}")
                        print(f"   💾 Cache hits: {api_stats.get('cache_hits', 0)}")
                        
                except Exception as e:
                    print(f"❌ Errore nel recupero stato: {e}")
                
            elif choice == '4':
                # Show full configuration
                try:
                    system.show_system_config()
                except Exception as e:
                    print(f"❌ Errore nel mostrare configurazione: {e}")
                
            elif choice == '5':
                # Test email notifications
                print("📧 Test notifiche email...")
                try:
                    if hasattr(system, 'notifier') and system.notifier:
                        # Test email sending
                        print("📧 Testing email notifier...")
                        
                        # Create test alert
                        test_alert = {
                            'crypto_id': 'bitcoin',
                            'crypto_name': 'Bitcoin',
                            'current_price': 65000,
                            'predicted_price': 68250,
                            'predicted_change': 0.05,
                            'confidence': 0.78,
                            'horizon': '1d',
                            'prediction_time': datetime.now().isoformat()
                        }
                        
                        success = system.notifier.add_dual_alert_fixed('high', test_alert, '1d')
                        if success:
                            print("✅ Test alert added")
                            
                            # Try to send summary
                            email_success = system.notifier.send_6hour_dual_summary_fixed()
                            if email_success:
                                print("✅ Test email sent successfully!")
                            else:
                                print("❌ Failed to send test email")
                        else:
                            print("❌ Failed to add test alert")
                    else:
                        print("❌ Email notifier not configured")
                        
                except Exception as e:
                    print(f"❌ Errore test email: {e}")
                
            elif choice == '6':
                # API and cache statistics
                print("🦎 Statistiche API e cache...")
                try:
                    if hasattr(system, 'api') and hasattr(system.api, 'get_api_stats'):
                        stats = system.api.get_api_stats()
                        print(f"\n🦎 STATISTICHE API:")
                        print(f"   📞 Chiamate oggi: {stats.get('daily_calls', 0)}/{stats.get('max_calls_per_day', 1000)}")
                        print(f"   📞 Chiamate orarie: {stats.get('hourly_calls', 0)}/{stats.get('max_calls_per_hour', 50)}")
                        print(f"   💾 Cache size: {stats.get('cache_size', 0)} elementi")
                        print(f"   🎯 Cache hits: {stats.get('cache_hits', 0)}")
                        print(f"   ❌ API errors: {stats.get('api_errors', 0)}")
                        print(f"   ⏰ Last API call: {stats.get('last_api_call_time', 'Never')}")
                    else:
                        print("❌ Statistiche API non disponibili")
                        
                except Exception as e:
                    print(f"❌ Errore statistiche API: {e}")
                
            elif choice == '7':
                # Database SSD statistics
                print("💾 Statistiche database SSD...")
                try:
                    if hasattr(system, 'database') and system.database:
                        stats = system.database.get_database_stats()
                        print(f"\n💾 STATISTICHE DATABASE:")
                        print(f"   📊 Predizioni totali: {stats.get('total_predictions', 0)}")
                        print(f"   📊 Predizioni recenti (24h): {stats.get('recent_predictions', 0)}")
                        print(f"   🎯 Alta confidenza: {stats.get('high_confidence_count', 0)}")
                        print(f"   🔍 Verifiche completate: {stats.get('verified_predictions', 0)}")
                        print(f"   📈 Accuracy media 1d: {stats.get('avg_accuracy_1d', 0):.1%}")
                        print(f"   📈 Accuracy media 3d: {stats.get('avg_accuracy_3d', 0):.1%}")
                        print(f"   💾 Database size: {stats.get('database_size_mb', 0):.1f} MB")
                        print(f"   📁 Database path: {system.database.db_path}")
                    else:
                        print("❌ Database non disponibile")
                        
                except Exception as e:
                    print(f"❌ Errore statistiche database: {e}")
                
            elif choice == '8':
                # Bitcoin benchmark dashboard
                print("🟠 Dashboard Bitcoin benchmark...")
                try:
                    if hasattr(system, 'bitcoin_benchmark') and system.bitcoin_benchmark:
                        system.bitcoin_benchmark.show_real_bitcoin_dashboard()
                        
                        # Additional Bitcoin stats
                        btc_data = system.bitcoin_benchmark.get_real_bitcoin_data()
                        if btc_data is not None and len(btc_data) > 0:
                            print(f"\n🟠 BITCOIN STATS:")
                            current_price = btc_data['price'].iloc[-1]
                            price_change_24h = ((current_price - btc_data['price'].iloc[-24]) / btc_data['price'].iloc[-24]) * 100
                            print(f"   💰 Current price: ${current_price:,.2f}")
                            print(f"   📈 Change 24h: {price_change_24h:+.2f}%")
                            print(f"   📊 Data points: {len(btc_data)}")
                    else:
                        print("❌ Bitcoin benchmark non disponibile")
                        
                except Exception as e:
                    print(f"❌ Errore Bitcoin dashboard: {e}")
                
            elif choice == '9':
                # Test verification system
                print("🔍 Test sistema verifica...")
                try:
                    if hasattr(system, 'verification_system') and system.verification_system:
                        # Run a small verification test
                        results = system.verification_system.run_complete_verification_optimized(
                            cycle_number=999,  # Test cycle
                            ml_system=getattr(system, 'ml_system', None)
                        )
                        
                        if results:
                            print(f"✅ Verification test completed:")
                            print(f"   🔍 Total checked: {results.get('total_checked', 0)}")
                            print(f"   ✅ Total verified: {results.get('total_verified', 0)}")
                            print(f"   📊 Accuracy 1d: {results.get('accuracy_1d', 0):.1%}")
                            print(f"   📊 Accuracy 3d: {results.get('accuracy_3d', 0):.1%}")
                            print(f"   ⏰ Duration: {results.get('elapsed_time', 0):.1f}s")
                        else:
                            print("❌ No verification results")
                    else:
                        print("❌ Sistema verifica non disponibile")
                        
                except Exception as e:
                    print(f"❌ Errore test verifica: {e}")
                
            elif choice == '10':
                # Pause/Resume system
                try:
                    if hasattr(system, 'paused'):
                        if system.paused:
                            system.paused = False
                            print("▶️ Sistema ripreso")
                        else:
                            system.paused = True
                            print("⏸️ Sistema messo in pausa")
                    else:
                        print("❌ Funzione pausa non disponibile")
                        
                except Exception as e:
                    print(f"❌ Errore pausa/riprendi: {e}")
                
            elif choice == '11':
                # Optimization and cleanup
                print("🧹 Ottimizzazione e cleanup...")
                try:
                    system._cleanup_optimized()
                    
                    # Additional cleanup operations
                    if hasattr(system, 'database') and system.database:
                        # Vacuum database
                        system.database.vacuum_database()
                        print("✅ Database ottimizzato")
                    
                    if hasattr(system, 'api'):
                        # Clear old cache
                        system.api.clear_old_cache()
                        print("✅ Cache pulita")
                    
                    print("✅ Ottimizzazione completata")
                    
                except Exception as e:
                    print(f"❌ Errore ottimizzazione: {e}")
                
            elif choice == '12':
                # Reload configuration
                print("📄 Ricarica configurazione...")
                try:
                    # Prompt for new config file
                    new_config_file = input("📁 Inserisci path del nuovo file JSON (o Enter per mantenere attuale): ").strip()
                    
                    if new_config_file and os.path.exists(new_config_file):
                        # Create new system with new config
                        old_system = system
                        system = OptimizedCryptoContinuousSystemSSD(config_file=new_config_file)
                        
                        # Cleanup old system
                        if old_system:
                            old_system._cleanup_optimized()
                        
                        print("✅ Configurazione ricaricata")
                    else:
                        if new_config_file:
                            print(f"❌ File non trovato: {new_config_file}")
                        else:
                            print("ℹ️ Configurazione non modificata")
                            
                except Exception as e:
                    print(f"❌ Errore ricarica configurazione: {e}")
                    
            elif choice == '13':
                # Debug email configuration  
                print("🔧 Debug configurazione email...")
                try:
                    if hasattr(system, 'debug_email_configuration'):
                        system.debug_email_configuration()
                    else:
                        # Manual debug
                        print("\n🔍 EMAIL DEBUG:")
                        print(f"📧 Notifier exists: {'YES' if hasattr(system, 'notifier') and system.notifier else 'NO'}")
                        
                        if hasattr(system, 'config'):
                            email_config = system.config.get('email', {})
                            gmail_user = email_config.get('gmail_user', '')
                            gmail_pass = email_config.get('gmail_app_password', '')
                            
                            print(f"📧 Gmail user: {'SET' if gmail_user else 'MISSING'}")
                            print(f"📧 Gmail password: {'SET' if gmail_pass else 'MISSING'}")
                            
                            if gmail_user:
                                print(f"   User: {gmail_user}")
                            if gmail_pass:
                                print(f"   Password length: {len(gmail_pass)}")
                                
                except Exception as e:
                    print(f"❌ Errore debug email: {e}")
                    
            # 🔥 SHITCOIN SCANNER OPTIONS
            elif choice in ['14', '15', '16', '17', '18', '19'] and SHITCOIN_SCANNER_AVAILABLE:
                import asyncio
                try:
                    asyncio.run(handle_shitcoin_menu_choice(choice, system))
                except Exception as e:
                    print(f"❌ Errore shitcoin scanner: {e}")
                    import traceback
                    traceback.print_exc()
                    
            elif choice in ['14', '15', '16', '17', '18', '19'] and not SHITCOIN_SCANNER_AVAILABLE:
                print("❌ Shitcoin Scanner non disponibile")
                print("   Verifica che shitcoin_dex_scanner.py sia presente nella directory")
            
            elif choice == '0':
                # Exit
                print("🚪 Uscita in corso...")
                if system and hasattr(system, 'running') and system.running:
                    system.running = False
                    if hasattr(system, 'stop_system'):
                        system.stop_system()
                print("👋 Arrivederci!")
                break
                
            else:
                print("❌ Scelta non valida. Inserisci un numero da 0 a 13.")
                
    except KeyboardInterrupt:
        print("\n🛑 Interrotto dall'utente")
        if system and hasattr(system, 'running') and system.running:
            system.running = False
            if hasattr(system, 'stop_system'):
                system.stop_system()
    
    except Exception as e:
        print(f"\n❌ Errore sistema: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Comprehensive cleanup
        if system:
            try:
                if hasattr(system, '_cleanup_optimized'):
                    system._cleanup_optimized()
                print("✅ Cleanup completato")
            except Exception as cleanup_error:
                print(f"⚠️ Errore cleanup: {cleanup_error}")
        
        print("🚀 Shutdown sistema SSD completato")


if __name__ == "__main__":
    main()