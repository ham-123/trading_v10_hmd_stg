"""
Collecteur de données en temps réel pour l'API Deriv
Volatility 10 Trading Bot - collector.py
"""

import json
import time
import threading
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from enum import Enum
import websocket
import schedule
from queue import Queue, Empty
import requests

from config import config, get_api_manager
from .database import db_manager, PriceData

# Configuration du logger
logger = logging.getLogger(__name__)


class ConnectionStatus(Enum):
    """Statuts de connexion"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    SUBSCRIBED = "subscribed"
    ERROR = "error"


@dataclass
class TickData:
    """Structure pour les données de tick"""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    spread: float
    quote: float

    @property
    def mid_price(self) -> float:
        """Prix moyen bid/ask"""
        return (self.bid + self.ask) / 2


@dataclass
class CandleData:
    """Structure pour les données de chandelier"""
    symbol: str
    timestamp: datetime
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    tick_count: int = 1


class DerivDataCollector:
    """Collecteur principal de données Deriv"""

    def __init__(self, symbols: List[str] = None, timeframes: List[str] = None):
        # Configuration
        self.symbols = symbols or [config.trading.symbol]
        self.timeframes = timeframes or [config.trading.timeframe]
        self.api_url = config.deriv_api.ws_url

        # Gestion de l'authentification
        self.api_manager = get_api_manager()
        self.credentials = self.api_manager.get_deriv_credentials()

        # État de la connexion
        self.ws = None
        self.status = ConnectionStatus.DISCONNECTED
        self.last_heartbeat = None
        self.connection_attempts = 0
        self.max_reconnect_attempts = config.deriv_api.reconnect_attempts

        # Gestion des données
        self.tick_queue = Queue(maxsize=10000)
        self.candle_buffer = {}  # Buffer pour construire les chandeliers
        self.last_prices = {}  # Cache des derniers prix

        # Statistiques
        self.stats = {
            'ticks_received': 0,
            'candles_saved': 0,
            'errors': 0,
            'reconnections': 0,
            'last_tick_time': None,
            'data_quality_score': 1.0
        }

        # Threading
        self.running = False
        self.threads = []

        # Callbacks
        self.on_tick_callback: Optional[Callable] = None
        self.on_candle_callback: Optional[Callable] = None
        self.on_error_callback: Optional[Callable] = None

        logger.info(f"Collecteur initialisé pour {self.symbols} sur {self.timeframes}")

    def start(self):
        """Démarre la collecte de données"""
        if self.running:
            logger.warning("Le collecteur est déjà en cours d'exécution")
            return

        if not self.credentials or not self.credentials.get('api_token'):
            logger.error("Credentials Deriv manquants - impossible de démarrer")
            return

        self.running = True
        logger.info("🚀 Démarrage du collecteur de données...")

        # Thread principal pour WebSocket
        ws_thread = threading.Thread(target=self._websocket_worker, daemon=True)
        ws_thread.start()
        self.threads.append(ws_thread)

        # Thread pour traitement des ticks
        tick_thread = threading.Thread(target=self._tick_processor, daemon=True)
        tick_thread.start()
        self.threads.append(tick_thread)

        # Thread pour construction des chandeliers
        candle_thread = threading.Thread(target=self._candle_builder, daemon=True)
        candle_thread.start()
        self.threads.append(candle_thread)

        # Thread pour monitoring
        monitor_thread = threading.Thread(target=self._monitor_worker, daemon=True)
        monitor_thread.start()
        self.threads.append(monitor_thread)

        logger.info("✅ Collecteur démarré avec succès")

    def stop(self):
        """Arrête la collecte de données"""
        logger.info("🛑 Arrêt du collecteur de données...")
        self.running = False

        if self.ws:
            self.ws.close()

        # Attendre que tous les threads se terminent
        for thread in self.threads:
            thread.join(timeout=5)

        logger.info("✅ Collecteur arrêté")

    def _websocket_worker(self):
        """Worker principal pour la connexion WebSocket"""
        while self.running:
            try:
                self._connect_websocket()
                if self.ws:
                    self.ws.run_forever(
                        ping_interval=config.deriv_api.heartbeat_interval,
                        ping_timeout=10
                    )
            except Exception as e:
                logger.error(f"Erreur WebSocket: {e}")
                self.stats['errors'] += 1

            if self.running:
                self._handle_reconnection()

    def _connect_websocket(self):
        """Établit la connexion WebSocket"""
        try:
            self.status = ConnectionStatus.CONNECTING
            logger.info(f"Connexion à {self.api_url}...")

            self.ws = websocket.WebSocketApp(
                self.api_url,
                on_open=self._on_websocket_open,
                on_message=self._on_websocket_message,
                on_error=self._on_websocket_error,
                on_close=self._on_websocket_close
            )

        except Exception as e:
            logger.error(f"Erreur lors de la création de la connexion WebSocket: {e}")
            self.status = ConnectionStatus.ERROR

    def _on_websocket_open(self, ws):
        """Callback d'ouverture de connexion"""
        logger.info("✅ Connexion WebSocket établie")
        self.status = ConnectionStatus.CONNECTED
        self.connection_attempts = 0

        # Authentification
        self._authenticate()

    def _on_websocket_message(self, ws, message):
        """Callback de réception de message"""
        try:
            data = json.loads(message)
            self._process_message(data)
        except json.JSONDecodeError as e:
            logger.error(f"Erreur de parsing JSON: {e}")
            self.stats['errors'] += 1
        except Exception as e:
            logger.error(f"Erreur lors du traitement du message: {e}")
            self.stats['errors'] += 1

    def _on_websocket_error(self, ws, error):
        """Callback d'erreur"""
        logger.error(f"Erreur WebSocket: {error}")
        self.status = ConnectionStatus.ERROR
        self.stats['errors'] += 1

        if self.on_error_callback:
            self.on_error_callback(error)

    def _on_websocket_close(self, ws, close_status_code, close_msg):
        """Callback de fermeture"""
        logger.warning(f"Connexion WebSocket fermée: {close_status_code} - {close_msg}")
        self.status = ConnectionStatus.DISCONNECTED

    def _authenticate(self):
        """Authentifie la connexion avec le token API"""
        if not self.credentials:
            logger.error("Pas de credentials pour l'authentification")
            return

        auth_message = {
            "authorize": self.credentials['api_token'],
            "req_id": int(time.time())
        }

        self._send_message(auth_message)
        logger.info("🔐 Authentification envoyée...")

    def _subscribe_to_ticks(self):
        """S'abonne aux ticks pour tous les symboles"""
        for symbol in self.symbols:
            subscribe_message = {
                "ticks": symbol,
                "subscribe": 1,
                "req_id": int(time.time())
            }
            self._send_message(subscribe_message)
            logger.info(f"📊 Abonnement aux ticks pour {symbol}")

    def _subscribe_to_candles(self):
        """S'abonne aux chandeliers pour tous les symboles et timeframes"""
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                # Convertir le timeframe au format Deriv
                deriv_timeframe = self._convert_timeframe(timeframe)

                subscribe_message = {
                    "ticks_history": symbol,
                    "granularity": deriv_timeframe,
                    "style": "candles",
                    "count": 1000,  # Historique initial
                    "subscribe": 1,
                    "req_id": int(time.time())
                }
                self._send_message(subscribe_message)
                logger.info(f"🕯️ Abonnement aux chandeliers {symbol} {timeframe}")

    def _convert_timeframe(self, timeframe: str) -> int:
        """Convertit le timeframe au format Deriv"""
        timeframe_map = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '4h': 14400,
            '1d': 86400
        }
        return timeframe_map.get(timeframe, 60)

    def _send_message(self, message: Dict[str, Any]):
        """Envoie un message via WebSocket"""
        if self.ws and self.ws.sock:
            try:
                self.ws.send(json.dumps(message))
            except Exception as e:
                logger.error(f"Erreur lors de l'envoi du message: {e}")

    def _process_message(self, data: Dict[str, Any]):
        """Traite les messages reçus de l'API"""
        msg_type = data.get('msg_type')

        if msg_type == 'authorize':
            self._handle_auth_response(data)
        elif msg_type == 'tick':
            self._handle_tick_data(data)
        elif msg_type == 'candles':
            self._handle_candle_data(data)
        elif msg_type == 'ohlc':
            self._handle_ohlc_data(data)
        elif msg_type == 'error':
            self._handle_error_response(data)
        else:
            logger.debug(f"Message non traité: {msg_type}")

    def _handle_auth_response(self, data: Dict[str, Any]):
        """Traite la réponse d'authentification"""
        if data.get('authorize'):
            logger.info("✅ Authentification réussie")
            self.status = ConnectionStatus.AUTHENTICATED

            # S'abonner aux données
            self._subscribe_to_ticks()
            self._subscribe_to_candles()
            self.status = ConnectionStatus.SUBSCRIBED

        else:
            logger.error("❌ Échec de l'authentification")
            self.status = ConnectionStatus.ERROR

    def _handle_tick_data(self, data: Dict[str, Any]):
        """Traite les données de tick"""
        try:
            tick_info = data.get('tick', {})

            tick = TickData(
                symbol=tick_info.get('symbol', ''),
                timestamp=datetime.fromtimestamp(tick_info.get('epoch', 0), timezone.utc),
                bid=float(tick_info.get('bid', 0)),
                ask=float(tick_info.get('ask', 0)),
                spread=float(tick_info.get('spread', 0)),
                quote=float(tick_info.get('quote', 0))
            )

            # Ajouter à la queue pour traitement
            try:
                self.tick_queue.put_nowait(tick)
                self.stats['ticks_received'] += 1
                self.stats['last_tick_time'] = tick.timestamp
            except:
                # Queue pleine, supprimer le plus ancien
                try:
                    self.tick_queue.get_nowait()
                    self.tick_queue.put_nowait(tick)
                except Empty:
                    pass

            # Callback optionnel
            if self.on_tick_callback:
                self.on_tick_callback(tick)

        except Exception as e:
            logger.error(f"Erreur lors du traitement du tick: {e}")
            self.stats['errors'] += 1

    def _handle_candle_data(self, data: Dict[str, Any]):
        """Traite les données de chandelier"""
        try:
            candles = data.get('candles', [])

            for candle_data in candles:
                candle = CandleData(
                    symbol=data.get('echo_req', {}).get('ticks_history', ''),
                    timestamp=datetime.fromtimestamp(candle_data.get('epoch', 0), timezone.utc),
                    timeframe=self._convert_granularity_to_timeframe(
                        data.get('echo_req', {}).get('granularity', 60)
                    ),
                    open=float(candle_data.get('open', 0)),
                    high=float(candle_data.get('high', 0)),
                    low=float(candle_data.get('low', 0)),
                    close=float(candle_data.get('close', 0))
                )

                self._save_candle_to_db(candle)

        except Exception as e:
            logger.error(f"Erreur lors du traitement des chandeliers: {e}")
            self.stats['errors'] += 1

    def _handle_ohlc_data(self, data: Dict[str, Any]):
        """Traite les données OHLC en temps réel"""
        try:
            ohlc = data.get('ohlc', {})

            candle = CandleData(
                symbol=ohlc.get('symbol', ''),
                timestamp=datetime.fromtimestamp(ohlc.get('epoch', 0), timezone.utc),
                timeframe=self._convert_granularity_to_timeframe(ohlc.get('granularity', 60)),
                open=float(ohlc.get('open', 0)),
                high=float(ohlc.get('high', 0)),
                low=float(ohlc.get('low', 0)),
                close=float(ohlc.get('close', 0))
            )

            self._save_candle_to_db(candle)

        except Exception as e:
            logger.error(f"Erreur lors du traitement OHLC: {e}")
            self.stats['errors'] += 1

    def _handle_error_response(self, data: Dict[str, Any]):
        """Traite les réponses d'erreur"""
        error = data.get('error', {})
        error_code = error.get('code')
        error_msg = error.get('message', 'Erreur inconnue')

        logger.error(f"Erreur API Deriv [{error_code}]: {error_msg}")
        self.stats['errors'] += 1

    def _convert_granularity_to_timeframe(self, granularity: int) -> str:
        """Convertit la granularité Deriv en timeframe"""
        granularity_map = {
            60: '1m',
            300: '5m',
            900: '15m',
            1800: '30m',
            3600: '1h',
            14400: '4h',
            86400: '1d'
        }
        return granularity_map.get(granularity, '1m')

    def _tick_processor(self):
        """Traite les ticks de la queue"""
        while self.running:
            try:
                # Traiter les ticks par batch
                ticks_batch = []

                # Collecter jusqu'à 100 ticks ou attendre 1 seconde
                start_time = time.time()
                while len(ticks_batch) < 100 and (time.time() - start_time) < 1.0:
                    try:
                        tick = self.tick_queue.get(timeout=0.1)
                        ticks_batch.append(tick)
                        self.tick_queue.task_done()
                    except Empty:
                        break

                # Traiter le batch
                if ticks_batch:
                    self._process_tick_batch(ticks_batch)

            except Exception as e:
                logger.error(f"Erreur dans le processeur de ticks: {e}")
                time.sleep(1)

    def _process_tick_batch(self, ticks: List[TickData]):
        """Traite un batch de ticks"""
        try:
            for tick in ticks:
                # Mettre à jour le cache des derniers prix
                self.last_prices[tick.symbol] = tick

                # Construire les chandeliers à partir des ticks
                self._update_candle_from_tick(tick)

        except Exception as e:
            logger.error(f"Erreur lors du traitement du batch de ticks: {e}")

    def _update_candle_from_tick(self, tick: TickData):
        """Met à jour les chandeliers à partir d'un tick"""
        for timeframe in self.timeframes:
            # Calculer le timestamp du chandelier
            granularity = self._convert_timeframe(timeframe)
            candle_timestamp = self._get_candle_timestamp(tick.timestamp, granularity)

            # Clé unique pour le chandelier
            candle_key = f"{tick.symbol}_{timeframe}_{candle_timestamp.timestamp()}"

            # Créer ou mettre à jour le chandelier
            if candle_key not in self.candle_buffer:
                # Nouveau chandelier
                self.candle_buffer[candle_key] = {
                    'symbol': tick.symbol,
                    'timestamp': candle_timestamp,
                    'timeframe': timeframe,
                    'open': tick.mid_price,
                    'high': tick.mid_price,
                    'low': tick.mid_price,
                    'close': tick.mid_price,
                    'tick_count': 1,
                    'last_update': tick.timestamp
                }
            else:
                # Mettre à jour le chandelier existant
                candle = self.candle_buffer[candle_key]
                candle['high'] = max(candle['high'], tick.mid_price)
                candle['low'] = min(candle['low'], tick.mid_price)
                candle['close'] = tick.mid_price
                candle['tick_count'] += 1
                candle['last_update'] = tick.timestamp

    def _get_candle_timestamp(self, tick_time: datetime, granularity: int) -> datetime:
        """Calcule le timestamp du chandelier pour un tick"""
        # Arrondir au début de la période
        timestamp = tick_time.replace(second=0, microsecond=0)
        minutes = (timestamp.minute // (granularity // 60)) * (granularity // 60)
        timestamp = timestamp.replace(minute=minutes)

        return timestamp

    def _candle_builder(self):
        """Construit et sauvegarde les chandeliers complets"""
        while self.running:
            try:
                current_time = datetime.now(timezone.utc)
                completed_candles = []

                # Identifier les chandeliers complétés
                for key, candle_data in list(self.candle_buffer.items()):
                    if self._is_candle_complete(candle_data, current_time):
                        completed_candles.append((key, candle_data))

                # Sauvegarder les chandeliers complétés
                for key, candle_data in completed_candles:
                    candle = CandleData(
                        symbol=candle_data['symbol'],
                        timestamp=candle_data['timestamp'],
                        timeframe=candle_data['timeframe'],
                        open=candle_data['open'],
                        high=candle_data['high'],
                        low=candle_data['low'],
                        close=candle_data['close'],
                        tick_count=candle_data['tick_count']
                    )

                    if self._save_candle_to_db(candle):
                        del self.candle_buffer[key]

                        # Callback optionnel
                        if self.on_candle_callback:
                            self.on_candle_callback(candle)

                time.sleep(1)  # Vérifier toutes les secondes

            except Exception as e:
                logger.error(f"Erreur dans le constructeur de chandeliers: {e}")
                time.sleep(1)

    def _is_candle_complete(self, candle_data: Dict, current_time: datetime) -> bool:
        """Vérifie si un chandelier est completé"""
        granularity = self._convert_timeframe(candle_data['timeframe'])
        candle_end_time = candle_data['timestamp'] + timedelta(seconds=granularity)

        # Le chandelier est complet si nous sommes passés à la période suivante
        return current_time >= candle_end_time

    def _save_candle_to_db(self, candle: CandleData) -> bool:
        """Sauvegarde un chandelier en base de données"""
        try:
            price_data = db_manager.save_price_data(
                symbol=candle.symbol,
                timestamp=candle.timestamp,
                timeframe=candle.timeframe,
                open_price=candle.open,
                high_price=candle.high,
                low_price=candle.low,
                close_price=candle.close,
                volume=candle.volume
            )

            if price_data:
                self.stats['candles_saved'] += 1
                logger.debug(f"Chandelier sauvegardé: {candle.symbol} {candle.timeframe} {candle.timestamp}")
                return True
            else:
                return False

        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du chandelier: {e}")
            return False

    def _monitor_worker(self):
        """Worker de monitoring et maintenance"""
        while self.running:
            try:
                # Calculer la qualité des données
                self._calculate_data_quality()

                # Nettoyer le buffer des chandeliers
                self._cleanup_candle_buffer()

                # Vérifier le heartbeat
                self._check_heartbeat()

                # Logger les statistiques périodiquement
                if self.stats['ticks_received'] % 1000 == 0 and self.stats['ticks_received'] > 0:
                    logger.info(f"📊 Stats: {self.stats['ticks_received']} ticks, "
                                f"{self.stats['candles_saved']} chandeliers, "
                                f"{self.stats['errors']} erreurs")

                time.sleep(30)  # Monitoring toutes les 30 secondes

            except Exception as e:
                logger.error(f"Erreur dans le monitoring: {e}")
                time.sleep(30)

    def _calculate_data_quality(self):
        """Calcule un score de qualité des données"""
        try:
            current_time = datetime.now(timezone.utc)

            # Vérifier la fraîcheur des données
            if self.stats['last_tick_time']:
                data_age = (current_time - self.stats['last_tick_time']).total_seconds()
                freshness_score = max(0, 1 - (data_age / 300))  # 5 minutes max
            else:
                freshness_score = 0

            # Taux d'erreur
            total_operations = self.stats['ticks_received'] + self.stats['candles_saved']
            error_rate = self.stats['errors'] / max(1, total_operations)
            error_score = max(0, 1 - error_rate)

            # Score global
            self.stats['data_quality_score'] = (freshness_score + error_score) / 2

        except Exception as e:
            logger.error(f"Erreur lors du calcul de qualité: {e}")
            self.stats['data_quality_score'] = 0.5

    def _cleanup_candle_buffer(self):
        """Nettoie le buffer des chandeliers anciens"""
        try:
            current_time = datetime.now(timezone.utc)
            keys_to_remove = []

            for key, candle_data in self.candle_buffer.items():
                # Supprimer les chandeliers trop anciens (> 1 heure)
                if (current_time - candle_data['timestamp']).total_seconds() > 3600:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self.candle_buffer[key]

        except Exception as e:
            logger.error(f"Erreur lors du nettoyage du buffer: {e}")

    def _check_heartbeat(self):
        """Vérifie le heartbeat de la connexion"""
        if self.status == ConnectionStatus.SUBSCRIBED:
            current_time = datetime.now(timezone.utc)

            if self.stats['last_tick_time']:
                time_since_last_tick = (current_time - self.stats['last_tick_time']).total_seconds()

                # Si pas de données depuis 2 minutes, quelque chose ne va pas
                if time_since_last_tick > 120:
                    logger.warning("⚠️ Pas de données reçues depuis 2 minutes")
                    # Tenter une reconnexion
                    if self.ws:
                        self.ws.close()

    def _handle_reconnection(self):
        """Gère la reconnexion automatique"""
        if not self.running:
            return

        self.connection_attempts += 1
        self.stats['reconnections'] += 1

        if self.connection_attempts <= self.max_reconnect_attempts:
            wait_time = min(30, 2 ** self.connection_attempts)  # Backoff exponentiel
            logger.info(
                f"🔄 Tentative de reconnexion {self.connection_attempts}/{self.max_reconnect_attempts} dans {wait_time}s")
            time.sleep(wait_time)
        else:
            logger.error("❌ Nombre maximum de tentatives de reconnexion atteint")
            self.running = False

    def get_latest_price(self, symbol: str) -> Optional[TickData]:
        """Retourne le dernier prix pour un symbole"""
        return self.last_prices.get(symbol)

    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du collecteur"""
        return self.stats.copy()

    def is_connected(self) -> bool:
        """Vérifie si le collecteur est connecté et opérationnel"""
        return self.status in [ConnectionStatus.AUTHENTICATED, ConnectionStatus.SUBSCRIBED]


# Instance globale
data_collector = DerivDataCollector()


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def start_data_collection(symbols: List[str] = None, timeframes: List[str] = None) -> DerivDataCollector:
    """Démarre la collecte de données"""
    global data_collector

    if symbols or timeframes:
        data_collector = DerivDataCollector(symbols, timeframes)

    data_collector.start()
    return data_collector


def stop_data_collection():
    """Arrête la collecte de données"""
    global data_collector
    if data_collector:
        data_collector.stop()


def get_collector_stats() -> Dict[str, Any]:
    """Retourne les statistiques du collecteur"""
    global data_collector
    if data_collector:
        return data_collector.get_stats()
    return {}


if __name__ == "__main__":
    # Test du collecteur
    print("📡 Test du collecteur de données Deriv...")

    # Créer un collecteur de test
    collector = DerivDataCollector()


    # Callbacks de test
    def on_tick(tick: TickData):
        print(f"📊 Tick reçu: {tick.symbol} = {tick.mid_price:.5f}")


    def on_candle(candle: CandleData):
        print(f"🕯️ Chandelier: {candle.symbol} {candle.timeframe} "
              f"O:{candle.open:.5f} H:{candle.high:.5f} L:{candle.low:.5f} C:{candle.close:.5f}")


    collector.on_tick_callback = on_tick
    collector.on_candle_callback = on_candle

    try:
        # Démarrer la collecte
        collector.start()

        # Attendre quelques secondes pour voir les données
        print("⏱️ Collection en cours (10 secondes)...")
        time.sleep(10)

        # Afficher les statistiques
        stats = collector.get_stats()
        print("\n📈 Statistiques:")
        for key, value in stats.items():
            print(f"   {key}: {value}")

    except KeyboardInterrupt:
        print("\n⚠️ Interruption par l'utilisateur")
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
    finally:
        collector.stop()
        print("✅ Test terminé")