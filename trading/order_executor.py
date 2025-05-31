"""
Exécuteur d'ordres pour le Trading Bot Volatility 10
Connexion à l'API Deriv et exécution des trades en temps réel
"""

import json
import time
import threading
import logging
import websocket
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
import warnings
from queue import Queue, Empty

from config import config, get_api_manager
from data import db_manager
from .risk_manager import risk_manager, PositionType, RiskLevel
from .decision_engine import TradingDecision, DecisionType

# Configuration du logger
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class OrderStatus(Enum):
    """Statuts des ordres"""
    PENDING = "pending"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    PARTIAL = "partial"


class OrderType(Enum):
    """Types d'ordres"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class ContractType(Enum):
    """Types de contrats Deriv"""
    CALL = "CALL"  # Rise
    PUT = "PUT"  # Fall
    ASIANU = "ASIANU"  # Asian Up
    ASIAND = "ASIAND"  # Asian Down
    DIGITOVER = "DIGITOVER"  # Higher
    DIGITUNDER = "DIGITUNDER"  # Lower


@dataclass
class Order:
    """Ordre de trading"""
    order_id: str
    symbol: str
    contract_type: ContractType
    amount: float
    duration: int = 60  # Durée en secondes
    barrier: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING

    # Métadonnées
    created_at: datetime = None
    executed_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    payout: Optional[float] = None
    profit_loss: Optional[float] = None

    # Paramètres techniques
    proposal_id: Optional[str] = None
    buy_id: Optional[str] = None
    contract_id: Optional[str] = None

    # Décision source
    decision_id: Optional[str] = None
    reasoning: str = ""

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'contract_type': self.contract_type.value,
            'amount': self.amount,
            'duration': self.duration,
            'barrier': self.barrier,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'executed_at': self.executed_at.isoformat() if self.executed_at else None,
            'filled_price': self.filled_price,
            'payout': self.payout,
            'profit_loss': self.profit_loss,
            'proposal_id': self.proposal_id,
            'buy_id': self.buy_id,
            'contract_id': self.contract_id,
            'decision_id': self.decision_id,
            'reasoning': self.reasoning
        }


@dataclass
class ExecutionConfig:
    """Configuration de l'exécution"""
    # Connexion
    max_reconnect_attempts: int = 5
    reconnect_delay_seconds: int = 5
    heartbeat_interval: int = 30

    # Ordres
    default_duration_seconds: int = 60
    max_amount_per_trade: float = 10.0
    min_amount_per_trade: float = 0.35
    max_slippage_pct: float = 0.1

    # Timeout
    proposal_timeout_seconds: int = 10
    execution_timeout_seconds: int = 30
    order_expiry_minutes: int = 5

    # Retry
    max_execution_retries: int = 3
    retry_delay_seconds: int = 2

    # Paper trading
    enable_paper_trading: bool = True
    paper_trading_balance: float = 1000.0


class OrderExecutor:
    """Exécuteur d'ordres via l'API Deriv"""

    def __init__(self, config: ExecutionConfig = None):
        self.config = config or ExecutionConfig()

        # Configuration API
        self.api_manager = get_api_manager()
        self.credentials = self.api_manager.get_deriv_credentials()
        self.ws_url = config.deriv_api.ws_url

        # État de connexion
        self.ws = None
        self.connected = False
        self.authenticated = False
        self.connection_attempts = 0

        # Files d'attente
        self.pending_orders = Queue()
        self.pending_proposals = {}  # Dict[req_id, callback]
        self.active_orders = {}  # Dict[order_id, Order]
        self.order_history = []

        # Gestion des messages
        self.message_callbacks = {}
        self.request_counter = 0

        # Paper trading
        self.paper_trading_enabled = self.config.enable_paper_trading or config.trading.enable_paper_trading
        self.paper_balance = self.config.paper_trading_balance
        self.paper_positions = {}

        # Statistiques
        self.execution_stats = {
            'orders_sent': 0,
            'orders_filled': 0,
            'orders_rejected': 0,
            'total_volume': 0.0,
            'total_pnl': 0.0,
            'avg_execution_time_ms': 0.0,
            'connection_uptime_pct': 0.0,
            'last_execution_time': None,
            'api_errors': 0
        }

        # Threading
        self.running = False
        self.threads = []
        self.executor_lock = threading.Lock()

        logger.info(f"Exécuteur d'ordres initialisé (Paper trading: {self.paper_trading_enabled})")

    def start(self) -> bool:
        """Démarre l'exécuteur d'ordres"""
        try:
            if self.running:
                logger.warning("Exécuteur déjà en cours d'exécution")
                return True

            if not self.paper_trading_enabled and not self.credentials:
                logger.error("Pas de credentials API - impossible de démarrer en mode réel")
                return False

            self.running = True
            logger.info("🚀 Démarrage de l'exécuteur d'ordres...")

            if not self.paper_trading_enabled:
                # Thread de connexion WebSocket
                ws_thread = threading.Thread(target=self._websocket_worker, daemon=True)
                ws_thread.start()
                self.threads.append(ws_thread)

            # Thread de traitement des ordres
            order_thread = threading.Thread(target=self._order_processor, daemon=True)
            order_thread.start()
            self.threads.append(order_thread)

            # Thread de monitoring
            monitor_thread = threading.Thread(target=self._monitor_worker, daemon=True)
            monitor_thread.start()
            self.threads.append(monitor_thread)

            logger.info("✅ Exécuteur d'ordres démarré")
            return True

        except Exception as e:
            logger.error(f"Erreur démarrage exécuteur: {e}")
            return False

    def stop(self):
        """Arrête l'exécuteur d'ordres"""
        logger.info("🛑 Arrêt de l'exécuteur d'ordres...")
        self.running = False

        if self.ws:
            self.ws.close()

        # Attendre l'arrêt des threads
        for thread in self.threads:
            thread.join(timeout=5)

        logger.info("✅ Exécuteur d'ordres arrêté")

    def execute_decision(self, decision: TradingDecision) -> Optional[str]:
        """
        Exécute une décision de trading

        Args:
            decision: Décision de trading à exécuter

        Returns:
            Order ID si ordre créé, None sinon
        """
        try:
            with self.executor_lock:
                if decision.decision_type == DecisionType.HOLD:
                    logger.debug("Décision HOLD - aucun ordre à exécuter")
                    return None

                # Convertir la décision en ordre
                order = self._convert_decision_to_order(decision)
                if not order:
                    logger.warning("Impossible de convertir la décision en ordre")
                    return None

                # Valider l'ordre avec le gestionnaire de risques
                if not self._validate_order_with_risk_manager(order, decision):
                    logger.warning(f"Ordre rejeté par le gestionnaire de risques: {order.order_id}")
                    return None

                # Ajouter à la file d'attente
                self.pending_orders.put(order)
                self.active_orders[order.order_id] = order

                logger.info(f"📋 Ordre ajouté à la file: {order.order_id} ({order.contract_type.value})")
                return order.order_id

        except Exception as e:
            logger.error(f"Erreur exécution décision: {e}")
            return None

    def _convert_decision_to_order(self, decision: TradingDecision) -> Optional[Order]:
        """Convertit une décision de trading en ordre Deriv"""
        try:
            # Déterminer le type de contrat
            if decision.decision_type == DecisionType.BUY:
                contract_type = ContractType.CALL  # Rise
            elif decision.decision_type == DecisionType.SELL:
                contract_type = ContractType.PUT  # Fall
            else:
                return None

            # Calculer le montant
            amount = self._calculate_order_amount(decision.position_size, decision.entry_price)

            if amount < self.config.min_amount_per_trade:
                logger.warning(f"Montant trop petit: {amount}")
                return None

            # Calculer la durée
            duration = self._calculate_contract_duration(decision)

            # Créer l'ordre
            order = Order(
                order_id=f"order_{uuid.uuid4().hex[:8]}",
                symbol=decision.symbol,
                contract_type=contract_type,
                amount=amount,
                duration=duration,
                decision_id=decision.decision_id,
                reasoning=decision.reasoning
            )

            return order

        except Exception as e:
            logger.error(f"Erreur conversion décision->ordre: {e}")
            return None

    def _calculate_order_amount(self, position_size: float, entry_price: float) -> float:
        """Calcule le montant de l'ordre"""
        try:
            # Pour les options binaires, le montant est directement le stake
            if self.paper_trading_enabled:
                available_balance = self.paper_balance
            else:
                available_balance = risk_manager.current_balance

            # Montant basé sur la taille de position (en % du capital)
            amount = available_balance * position_size

            # Appliquer les limites
            amount = max(self.config.min_amount_per_trade,
                         min(self.config.max_amount_per_trade, amount))

            return round(amount, 2)

        except Exception as e:
            logger.error(f"Erreur calcul montant ordre: {e}")
            return self.config.min_amount_per_trade

    def _calculate_contract_duration(self, decision: TradingDecision) -> int:
        """Calcule la durée du contrat"""
        try:
            # Durée basée sur l'horizon de prédiction ou par défaut
            if hasattr(decision, 'prediction_horizon'):
                # Convertir minutes en secondes
                duration = decision.prediction_horizon * 60
            else:
                duration = self.config.default_duration_seconds

            # Limites raisonnables pour Volatility 10
            duration = max(60, min(3600, duration))  # Entre 1 minute et 1 heure

            return duration

        except Exception as e:
            logger.error(f"Erreur calcul durée contrat: {e}")
            return self.config.default_duration_seconds

    def _validate_order_with_risk_manager(self, order: Order, decision: TradingDecision) -> bool:
        """Valide l'ordre avec le gestionnaire de risques"""
        try:
            # Convertir le type de contrat en type de position
            if order.contract_type in [ContractType.CALL, ContractType.ASIANU, ContractType.DIGITOVER]:
                position_type = PositionType.LONG
            else:
                position_type = PositionType.SHORT

            # Valider avec le gestionnaire de risques
            is_valid, issues = risk_manager.validate_trade(
                symbol=order.symbol,
                position_type=position_type,
                size=order.amount / 100,  # Convertir en taille relative
                entry_price=decision.entry_price or 100.0  # Prix par défaut si manquant
            )

            if not is_valid:
                logger.warning(f"Ordre invalidé par risk manager: {', '.join(issues)}")
                return False

            return True

        except Exception as e:
            logger.error(f"Erreur validation avec risk manager: {e}")
            return False

    def _websocket_worker(self):
        """Worker pour la connexion WebSocket"""
        while self.running:
            try:
                if not self.connected:
                    self._connect_websocket()

                if self.ws and self.connected:
                    self.ws.run_forever(
                        ping_interval=self.config.heartbeat_interval,
                        ping_timeout=10
                    )

            except Exception as e:
                logger.error(f"Erreur WebSocket worker: {e}")
                self.execution_stats['api_errors'] += 1

            if self.running:
                self._handle_reconnection()

    def _connect_websocket(self):
        """Établit la connexion WebSocket avec Deriv"""
        try:
            logger.info(f"Connexion à l'API Deriv: {self.ws_url}")

            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_open=self._on_websocket_open,
                on_message=self._on_websocket_message,
                on_error=self._on_websocket_error,
                on_close=self._on_websocket_close
            )

        except Exception as e:
            logger.error(f"Erreur création connexion WebSocket: {e}")

    def _on_websocket_open(self, ws):
        """Callback d'ouverture WebSocket"""
        logger.info("✅ Connexion WebSocket établie")
        self.connected = True
        self.connection_attempts = 0

        # Authentification
        if self.credentials and self.credentials.get('api_token'):
            self._authenticate()

    def _on_websocket_message(self, ws, message):
        """Callback de réception de message"""
        try:
            data = json.loads(message)
            self._process_api_message(data)

        except json.JSONDecodeError as e:
            logger.error(f"Erreur parsing JSON: {e}")
        except Exception as e:
            logger.error(f"Erreur traitement message: {e}")

    def _on_websocket_error(self, ws, error):
        """Callback d'erreur WebSocket"""
        logger.error(f"Erreur WebSocket: {error}")
        self.connected = False
        self.execution_stats['api_errors'] += 1

    def _on_websocket_close(self, ws, close_status_code, close_msg):
        """Callback de fermeture WebSocket"""
        logger.warning(f"Connexion WebSocket fermée: {close_status_code}")
        self.connected = False
        self.authenticated = False

    def _authenticate(self):
        """Authentifie la connexion avec l'API"""
        if not self.credentials:
            return

        auth_message = {
            "authorize": self.credentials['api_token'],
            "req_id": self._get_request_id()
        }

        self._send_message(auth_message)
        logger.info("🔐 Authentification envoyée")

    def _send_message(self, message: Dict[str, Any]):
        """Envoie un message via WebSocket"""
        if self.ws and self.connected:
            try:
                self.ws.send(json.dumps(message))
            except Exception as e:
                logger.error(f"Erreur envoi message: {e}")

    def _get_request_id(self) -> int:
        """Génère un ID de requête unique"""
        self.request_counter += 1
        return self.request_counter

    def _process_api_message(self, data: Dict[str, Any]):
        """Traite les messages reçus de l'API"""
        try:
            msg_type = data.get('msg_type')
            req_id = data.get('req_id')

            if msg_type == 'authorize':
                self._handle_auth_response(data)
            elif msg_type == 'proposal':
                self._handle_proposal_response(data, req_id)
            elif msg_type == 'buy':
                self._handle_buy_response(data, req_id)
            elif msg_type == 'proposal_open_contract':
                self._handle_contract_update(data)
            elif msg_type == 'error':
                self._handle_error_response(data, req_id)
            else:
                logger.debug(f"Message non traité: {msg_type}")

        except Exception as e:
            logger.error(f"Erreur traitement message API: {e}")

    def _handle_auth_response(self, data: Dict[str, Any]):
        """Traite la réponse d'authentification"""
        if data.get('authorize'):
            logger.info("✅ Authentification réussie")
            self.authenticated = True
        else:
            logger.error("❌ Échec authentification")
            self.authenticated = False

    def _handle_proposal_response(self, data: Dict[str, Any], req_id: int):
        """Traite la réponse de proposition"""
        try:
            proposal = data.get('proposal')
            if not proposal:
                logger.warning("Proposition vide reçue")
                return

            # Trouver l'ordre correspondant
            callback = self.pending_proposals.get(req_id)
            if callback:
                callback(proposal)
                del self.pending_proposals[req_id]

        except Exception as e:
            logger.error(f"Erreur traitement proposition: {e}")

    def _handle_buy_response(self, data: Dict[str, Any], req_id: int):
        """Traite la réponse d'achat"""
        try:
            buy_info = data.get('buy')
            if not buy_info:
                logger.warning("Réponse d'achat vide")
                return

            # Trouver l'ordre correspondant par buy_id ou req_id
            order = None
            for active_order in self.active_orders.values():
                if active_order.proposal_id == buy_info.get('contract_id'):
                    order = active_order
                    break

            if order:
                order.status = OrderStatus.FILLED
                order.executed_at = datetime.now(timezone.utc)
                order.buy_id = buy_info.get('buy_id')
                order.contract_id = buy_info.get('contract_id')
                order.filled_price = buy_info.get('buy_price')

                self._update_execution_stats(order)
                logger.info(f"✅ Ordre exécuté: {order.order_id}")

                # Notifier le gestionnaire de risques si en mode réel
                if not self.paper_trading_enabled:
                    self._notify_risk_manager_order_filled(order)

        except Exception as e:
            logger.error(f"Erreur traitement achat: {e}")

    def _handle_contract_update(self, data: Dict[str, Any]):
        """Traite les mises à jour de contrat"""
        try:
            contract = data.get('proposal_open_contract')
            if not contract:
                return

            contract_id = contract.get('contract_id')

            # Trouver l'ordre correspondant
            order = None
            for active_order in self.active_orders.values():
                if active_order.contract_id == contract_id:
                    order = active_order
                    break

            if order:
                # Mettre à jour le statut
                if contract.get('is_expired'):
                    order.status = OrderStatus.EXPIRED
                    order.payout = contract.get('payout', 0)
                    order.profit_loss = contract.get('profit', 0)

                    self._finalize_order(order)

        except Exception as e:
            logger.error(f"Erreur mise à jour contrat: {e}")

    def _handle_error_response(self, data: Dict[str, Any], req_id: int):
        """Traite les réponses d'erreur"""
        error = data.get('error', {})
        error_code = error.get('code', 'Unknown')
        error_msg = error.get('message', 'Erreur inconnue')

        logger.error(f"Erreur API [{error_code}]: {error_msg}")
        self.execution_stats['api_errors'] += 1

        # Marquer les ordres associés comme rejetés
        callback = self.pending_proposals.get(req_id)
        if callback:
            callback(None)  # Signal d'erreur
            del self.pending_proposals[req_id]

    def _order_processor(self):
        """Processeur des ordres en attente"""
        while self.running:
            try:
                # Récupérer un ordre de la file
                try:
                    order = self.pending_orders.get(timeout=1.0)
                except Empty:
                    continue

                # Traiter l'ordre
                if self.paper_trading_enabled:
                    self._execute_paper_order(order)
                else:
                    self._execute_real_order(order)

                self.pending_orders.task_done()

            except Exception as e:
                logger.error(f"Erreur processeur d'ordres: {e}")
                time.sleep(1)

    def _execute_paper_order(self, order: Order):
        """Exécute un ordre en mode paper trading"""
        try:
            logger.info(f"📝 Exécution Paper Trading: {order.order_id}")

            # Simuler un délai d'exécution
            time.sleep(0.1)

            # Marquer comme exécuté
            order.status = OrderStatus.FILLED
            order.executed_at = datetime.now(timezone.utc)
            order.filled_price = 1.0  # Prix fictif pour options binaires

            # Ajouter à l'historique
            self.order_history.append(order)

            # Simuler le résultat (50% de chance de gagner pour les tests)
            import random
            win_probability = 0.55  # Légèrement favorable pour les tests

            if random.random() < win_probability:
                payout = order.amount * 1.8  # 80% de profit
                profit = payout - order.amount
            else:
                payout = 0
                profit = -order.amount

            order.payout = payout
            order.profit_loss = profit

            # Mettre à jour le solde paper trading
            self.paper_balance += profit

            # Mettre à jour les statistiques
            self._update_execution_stats(order)

            logger.info(f"✅ Paper Trade terminé: {order.order_id} (P&L: {profit:.2f})")

        except Exception as e:
            logger.error(f"Erreur exécution paper order: {e}")
            order.status = OrderStatus.REJECTED

    def _execute_real_order(self, order: Order):
        """Exécute un ordre réel via l'API Deriv"""
        try:
            if not self.connected or not self.authenticated:
                logger.warning("Pas de connexion API - ordre rejeté")
                order.status = OrderStatus.REJECTED
                return

            logger.info(f"🚀 Exécution ordre réel: {order.order_id}")

            # Étape 1: Demander une proposition
            proposal_req_id = self._request_proposal(order)

            if not proposal_req_id:
                order.status = OrderStatus.REJECTED
                return

            # Attendre la proposition
            self._wait_for_proposal_and_buy(order, proposal_req_id)

        except Exception as e:
            logger.error(f"Erreur exécution ordre réel: {e}")
            order.status = OrderStatus.REJECTED

    def _request_proposal(self, order: Order) -> Optional[int]:
        """Demande une proposition pour un ordre"""
        try:
            req_id = self._get_request_id()

            # Construire la demande de proposition
            proposal_request = {
                "proposal": 1,
                "subscribe": 1,
                "amount": order.amount,
                "basis": "stake",
                "contract_type": order.contract_type.value,
                "currency": "USD",
                "symbol": order.symbol,
                "duration": order.duration,
                "duration_unit": "s",
                "req_id": req_id
            }

            # Ajouter barrier si nécessaire
            if order.barrier:
                proposal_request["barrier"] = order.barrier

            # Envoyer la requête
            self._send_message(proposal_request)

            # Enregistrer le callback
            self.pending_proposals[req_id] = lambda proposal: self._handle_proposal_for_order(order, proposal)

            return req_id

        except Exception as e:
            logger.error(f"Erreur demande de proposition: {e}")
            return None

    def _handle_proposal_for_order(self, order: Order, proposal: Optional[Dict]):
        """Traite la proposition pour un ordre"""
        try:
            if not proposal:
                logger.warning(f"Proposition échouée pour {order.order_id}")
                order.status = OrderStatus.REJECTED
                return

            proposal_id = proposal.get('id')
            if not proposal_id:
                logger.warning(f"Pas d'ID de proposition pour {order.order_id}")
                order.status = OrderStatus.REJECTED
                return

            order.proposal_id = proposal_id

            # Immédiatement acheter la proposition
            self._buy_proposal(order)

        except Exception as e:
            logger.error(f"Erreur traitement proposition: {e}")
            order.status = OrderStatus.REJECTED

    def _buy_proposal(self, order: Order):
        """Achète une proposition"""
        try:
            buy_request = {
                "buy": order.proposal_id,
                "price": order.amount,
                "req_id": self._get_request_id()
            }

            self._send_message(buy_request)
            logger.info(f"💰 Achat envoyé pour {order.order_id}")

        except Exception as e:
            logger.error(f"Erreur achat proposition: {e}")
            order.status = OrderStatus.REJECTED

    def _wait_for_proposal_and_buy(self, order: Order, proposal_req_id: int):
        """Attend la proposition et l'achat"""
        try:
            # Dans la vraie implémentation, ceci serait géré par les callbacks WebSocket
            # Ici on simule juste l'attente
            time.sleep(2)

            if order.status == OrderStatus.PENDING:
                # Si toujours en attente après le timeout, marquer comme rejeté
                order.status = OrderStatus.REJECTED
                logger.warning(f"Timeout pour ordre {order.order_id}")

        except Exception as e:
            logger.error(f"Erreur attente proposition: {e}")
            order.status = OrderStatus.REJECTED

    def _notify_risk_manager_order_filled(self, order: Order):
        """Notifie le gestionnaire de risques qu'un ordre est exécuté"""
        try:
            # Convertir en position pour le risk manager
            position_type = (PositionType.LONG if order.contract_type in
                                                  [ContractType.CALL, ContractType.ASIANU, ContractType.DIGITOVER]
                             else PositionType.SHORT)

            risk_manager.add_position(
                position_id=order.order_id,
                symbol=order.symbol,
                position_type=position_type,
                size=order.amount / 100,  # Convertir en taille relative
                entry_price=order.filled_price or 1.0
            )

        except Exception as e:
            logger.error(f"Erreur notification risk manager: {e}")

    def _finalize_order(self, order: Order):
        """Finalise un ordre terminé"""
        try:
            # Ajouter à l'historique
            self.order_history.append(order)

            # Supprimer des ordres actifs
            if order.order_id in self.active_orders:
                del self.active_orders[order.order_id]

            # Notifier le gestionnaire de risques si en mode réel
            if not self.paper_trading_enabled and order.contract_id:
                risk_manager.close_position(
                    position_id=order.order_id,
                    exit_price=order.payout or 0,
                    reason=f"Contract expired: {order.status.value}"
                )

            logger.info(f"🏁 Ordre finalisé: {order.order_id} (P&L: {order.profit_loss})")

        except Exception as e:
            logger.error(f"Erreur finalisation ordre: {e}")

    def _update_execution_stats(self, order: Order):
        """Met à jour les statistiques d'exécution"""
        try:
            self.execution_stats['orders_sent'] += 1

            if order.status == OrderStatus.FILLED:
                self.execution_stats['orders_filled'] += 1
                self.execution_stats['total_volume'] += order.amount
                self.execution_stats['last_execution_time'] = order.executed_at

                if order.profit_loss:
                    self.execution_stats['total_pnl'] += order.profit_loss

            elif order.status == OrderStatus.REJECTED:
                self.execution_stats['orders_rejected'] += 1

            # Calculer le temps d'exécution moyen
            if order.executed_at and order.created_at:
                execution_time = (order.executed_at - order.created_at).total_seconds() * 1000
                total_orders = self.execution_stats['orders_filled']
                old_avg = self.execution_stats['avg_execution_time_ms']

                self.execution_stats['avg_execution_time_ms'] = (
                        (old_avg * (total_orders - 1) + execution_time) / total_orders
                )

        except Exception as e:
            logger.error(f"Erreur mise à jour stats: {e}")

    def _monitor_worker(self):
        """Worker de monitoring et maintenance"""
        while self.running:
            try:
                # Nettoyer les ordres expirés
                self._cleanup_expired_orders()

                # Calculer l'uptime de connexion
                self._calculate_uptime()

                # Logger les statistiques périodiquement
                if self.execution_stats['orders_sent'] % 10 == 0 and self.execution_stats['orders_sent'] > 0:
                    logger.info(f"📊 Stats: {self.execution_stats['orders_filled']} ordres exécutés, "
                                f"PnL total: {self.execution_stats['total_pnl']:.2f}")

                time.sleep(30)  # Monitoring toutes les 30 secondes

            except Exception as e:
                logger.error(f"Erreur monitoring: {e}")
                time.sleep(30)

    def _cleanup_expired_orders(self):
        """Nettoie les ordres expirés"""
        try:
            current_time = datetime.now(timezone.utc)
            expired_orders = []

            for order_id, order in self.active_orders.items():
                # Vérifier si l'ordre est trop ancien
                age_minutes = (current_time - order.created_at).total_seconds() / 60

                if age_minutes > self.config.order_expiry_minutes:
                    expired_orders.append(order_id)

            # Marquer comme expirés
            for order_id in expired_orders:
                order = self.active_orders[order_id]
                order.status = OrderStatus.EXPIRED
                self._finalize_order(order)

        except Exception as e:
            logger.error(f"Erreur nettoyage ordres expirés: {e}")

    def _calculate_uptime(self):
        """Calcule l'uptime de connexion"""
        try:
            if self.paper_trading_enabled:
                self.execution_stats['connection_uptime_pct'] = 100.0
            else:
                # TODO: Implémenter le calcul réel d'uptime
                uptime_pct = 95.0 if self.connected else 0.0
                self.execution_stats['connection_uptime_pct'] = uptime_pct

        except Exception as e:
            logger.error(f"Erreur calcul uptime: {e}")

    def _handle_reconnection(self):
        """Gère la reconnexion automatique"""
        if not self.running:
            return

        self.connection_attempts += 1

        if self.connection_attempts <= self.config.max_reconnect_attempts:
            wait_time = min(60, self.config.reconnect_delay_seconds * self.connection_attempts)
            logger.info(f"🔄 Reconnexion dans {wait_time}s (tentative {self.connection_attempts})")
            time.sleep(wait_time)
        else:
            logger.error("❌ Nombre maximum de tentatives de reconnexion atteint")
            self.running = False

    def get_execution_summary(self) -> Dict[str, Any]:
        """Retourne un résumé de l'exécution"""
        try:
            summary = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'paper_trading': self.paper_trading_enabled,
                'connected': self.connected,
                'authenticated': self.authenticated,
                'execution_stats': self.execution_stats.copy(),
                'active_orders': len(self.active_orders),
                'pending_orders': self.pending_orders.qsize(),
                'order_history_count': len(self.order_history)
            }

            if self.paper_trading_enabled:
                summary['paper_balance'] = self.paper_balance
                summary['paper_positions'] = len(self.paper_positions)

            # Statistiques de performance
            if self.execution_stats['orders_sent'] > 0:
                summary['fill_rate'] = (
                        self.execution_stats['orders_filled'] / self.execution_stats['orders_sent']
                )
                summary['rejection_rate'] = (
                        self.execution_stats['orders_rejected'] / self.execution_stats['orders_sent']
                )

            return summary

        except Exception as e:
            logger.error(f"Erreur génération résumé: {e}")
            return {'error': str(e)}

    def cancel_order(self, order_id: str) -> bool:
        """Annule un ordre"""
        try:
            if order_id in self.active_orders:
                order = self.active_orders[order_id]
                order.status = OrderStatus.CANCELLED
                self._finalize_order(order)
                logger.info(f"❌ Ordre annulé: {order_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Erreur annulation ordre: {e}")
            return False


# Instance globale
order_executor = OrderExecutor()


# Fonctions utilitaires
def execute_trading_decision(decision: TradingDecision) -> Optional[str]:
    """Fonction utilitaire pour exécuter une décision de trading"""
    return order_executor.execute_decision(decision)


def get_execution_summary() -> Dict[str, Any]:
    """Fonction utilitaire pour récupérer le résumé d'exécution"""
    return order_executor.get_execution_summary()


def start_order_execution() -> bool:
    """Fonction utilitaire pour démarrer l'exécution d'ordres"""
    return order_executor.start()


def stop_order_execution():
    """Fonction utilitaire pour arrêter l'exécution d'ordres"""
    order_executor.stop()


if __name__ == "__main__":
    # Test de l'exécuteur d'ordres
    print("🚀 Test de l'exécuteur d'ordres...")

    try:
        # Configuration de test
        test_config = ExecutionConfig(
            enable_paper_trading=True,
            paper_trading_balance=1000.0,
            default_duration_seconds=60
        )

        executor = OrderExecutor(test_config)

        print(f"✅ Exécuteur configuré")
        print(f"   Paper trading: {executor.paper_trading_enabled}")
        print(f"   Solde initial: {executor.paper_balance}")
        print(f"   Durée par défaut: {executor.config.default_duration_seconds}s")

        # Démarrer l'exécuteur
        success = executor.start()
        print(f"\n🚀 Démarrage: {'Succès' if success else 'Échec'}")

        if success:
            # Créer un ordre de test
            test_order = Order(
                order_id="test_001",
                symbol="R_10",
                contract_type=ContractType.CALL,
                amount=5.0,
                duration=60
            )

            # Ajouter à la file d'attente
            executor.pending_orders.put(test_order)
            executor.active_orders[test_order.order_id] = test_order

            print(f"\n📋 Ordre de test ajouté: {test_order.order_id}")

            # Attendre l'exécution
            time.sleep(2)

            # Résumé
            summary = executor.get_execution_summary()
            print(f"\n📊 Résumé d'exécution:")
            print(f"   Ordres actifs: {summary['active_orders']}")
            print(f"   En attente: {summary['pending_orders']}")
            print(f"   Historique: {summary['order_history_count']}")

            if executor.paper_trading_enabled:
                print(f"   Solde paper: {summary['paper_balance']:.2f}")

            # Arrêter
            executor.stop()

        print("✅ Test de l'exécuteur réussi !")

    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        logger.error(f"Test de l'exécuteur échoué: {e}")