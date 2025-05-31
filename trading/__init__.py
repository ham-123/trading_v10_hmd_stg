"""
Module de trading pour le Trading Bot Volatility 10
Centralise le moteur de décision, la gestion des risques et l'exécution des ordres

Usage:
    from trading import make_trading_decision, execute_decision, get_trading_status
    from trading import decision_engine, risk_manager, order_executor
    from trading import TradingDecision, RiskMetrics, Order
"""

# Imports des classes principales
from .decision_engine import (
    DecisionEngine,
    TradingDecision,
    DecisionContext,
    DecisionType,
    DecisionConfidence,
    MarketRegime,
    decision_engine,
    make_trading_decision,
    get_decision_stats
)

from .risk_manager import (
    RiskManager,
    RiskMetrics,
    RiskLimits,
    RiskAlert,
    RiskLevel,
    RiskEvent,
    PositionInfo,
    PositionType,
    risk_manager,
    calculate_position_size,
    validate_trade,
    get_risk_summary
)

from .order_executor import (
    OrderExecutor,
    Order,
    OrderStatus,
    OrderType,
    ContractType,
    ExecutionConfig,
    order_executor,
    execute_trading_decision,
    get_execution_summary,
    start_order_execution,
    stop_order_execution
)

import logging
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import warnings

from config import config
from data import db_manager
from analysis import perform_complete_analysis
from ai_model import get_ai_insights

# Configuration du logger
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Version du module
__version__ = "1.0.0"

# Informations sur le module
__author__ = "Trading Bot Team"
__description__ = "Module de trading complet pour le Trading Bot Volatility 10"

# Exports principaux
__all__ = [
    # Classes de décision
    "DecisionEngine",
    "TradingDecision",
    "DecisionContext",
    "DecisionType",
    "DecisionConfidence",
    "MarketRegime",

    # Classes de gestion des risques
    "RiskManager",
    "RiskMetrics",
    "RiskLimits",
    "RiskAlert",
    "RiskLevel",
    "RiskEvent",
    "PositionInfo",
    "PositionType",

    # Classes d'exécution
    "OrderExecutor",
    "Order",
    "OrderStatus",
    "OrderType",
    "ContractType",
    "ExecutionConfig",

    # Instances globales
    "decision_engine",
    "risk_manager",
    "order_executor",

    # Fonctions utilitaires - Décision
    "make_trading_decision",
    "get_decision_stats",

    # Fonctions utilitaires - Risque
    "calculate_position_size",
    "validate_trade",
    "get_risk_summary",

    # Fonctions utilitaires - Exécution
    "execute_trading_decision",
    "get_execution_summary",
    "start_order_execution",
    "stop_order_execution",

    # Fonctions du module
    "get_trading_pipeline_status",
    "execute_complete_trading_cycle",
    "start_automated_trading",
    "stop_automated_trading",
    "get_trading_performance",
    "emergency_stop_all_trading",
    "get_portfolio_overview",
    "optimize_trading_parameters",

    # Métadonnées
    "__version__",
    "__author__",
    "__description__"
]


class TradingMode(Enum):
    """Modes de trading"""
    MANUAL = "manual"  # Trading manuel
    SEMI_AUTO = "semi_auto"  # Semi-automatique avec validation
    FULL_AUTO = "full_auto"  # Entièrement automatique
    SIMULATION = "simulation"  # Mode simulation
    STOPPED = "stopped"  # Trading arrêté


class TradingStatus(Enum):
    """Statuts du système de trading"""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class TradingPerformance:
    """Performance globale du trading"""
    # Métriques de base
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # Profits et pertes
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0

    # Ratios de performance
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    recovery_factor: float = 0.0

    # Exposition et risque
    current_exposure: float = 0.0
    max_exposure: float = 0.0
    risk_score: float = 0.0

    # Timing
    avg_trade_duration: float = 0.0
    total_trading_time: float = 0.0
    last_update: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TradingState:
    """État actuel du système de trading"""
    mode: TradingMode = TradingMode.STOPPED
    status: TradingStatus = TradingStatus.INITIALIZING

    # Composants
    decision_engine_active: bool = False
    risk_manager_active: bool = False
    order_executor_active: bool = False

    # Configuration
    auto_trading_enabled: bool = False
    emergency_stop_triggered: bool = False
    last_decision_time: Optional[datetime] = None
    last_trade_time: Optional[datetime] = None

    # Métriques temps réel
    decisions_today: int = 0
    trades_today: int = 0
    pnl_today: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'mode': self.mode.value,
            'status': self.status.value,
            'decision_engine_active': self.decision_engine_active,
            'risk_manager_active': self.risk_manager_active,
            'order_executor_active': self.order_executor_active,
            'auto_trading_enabled': self.auto_trading_enabled,
            'emergency_stop_triggered': self.emergency_stop_triggered,
            'last_decision_time': self.last_decision_time.isoformat() if self.last_decision_time else None,
            'last_trade_time': self.last_trade_time.isoformat() if self.last_trade_time else None,
            'decisions_today': self.decisions_today,
            'trades_today': self.trades_today,
            'pnl_today': self.pnl_today
        }


class TradingOrchestrator:
    """Orchestrateur principal du système de trading"""

    def __init__(self):
        # État du système
        self.trading_state = TradingState()
        self.trading_performance = TradingPerformance()

        # Configuration
        self.config = {
            'auto_decision_interval_seconds': 60,  # Décision toutes les minutes
            'max_decisions_per_hour': 12,  # Limite de décisions par heure
            'risk_check_interval_seconds': 30,  # Vérification risque toutes les 30s
            'performance_update_interval': 300,  # Mise à jour performance toutes les 5min
            'emergency_drawdown_threshold': 0.05,  # 5% de drawdown pour arrêt d'urgence
            'default_trading_symbols': ['R_10'],  # Symboles par défaut
            'enable_ai_insights': True,  # Utiliser les insights IA
            'min_confidence_for_execution': 0.65  # Confiance minimum pour exécuter
        }

        # Threading
        self.auto_trading_thread = None
        self.monitoring_thread = None
        self.trading_lock = threading.Lock()
        self.running = False

        # Historique et cache
        self.decision_history = []
        self.performance_history = []
        self.last_performance_update = None

        logger.info("Orchestrateur de trading initialisé")

    def initialize_trading_system(self) -> bool:
        """Initialise complètement le système de trading"""
        try:
            logger.info("🚀 Initialisation du système de trading...")

            with self.trading_lock:
                self.trading_state.status = TradingStatus.INITIALIZING

                # Vérifier les composants
                components_status = {
                    'decision_engine': decision_engine is not None,
                    'risk_manager': risk_manager is not None,
                    'order_executor': order_executor is not None
                }

                if not all(components_status.values()):
                    missing = [k for k, v in components_status.items() if not v]
                    logger.error(f"Composants manquants: {missing}")
                    self.trading_state.status = TradingStatus.ERROR
                    return False

                # Initialiser l'exécuteur d'ordres
                if not order_executor.running:
                    if not order_executor.start():
                        logger.error("Échec démarrage de l'exécuteur d'ordres")
                        self.trading_state.status = TradingStatus.ERROR
                        return False

                # Mettre à jour l'état des composants
                self.trading_state.decision_engine_active = True
                self.trading_state.risk_manager_active = True
                self.trading_state.order_executor_active = order_executor.running

                # Calcul initial de la performance
                self._update_trading_performance()

                # Marquer comme prêt
                self.trading_state.status = TradingStatus.READY
                self.trading_state.mode = TradingMode.MANUAL

                logger.info("✅ Système de trading initialisé avec succès")
                return True

        except Exception as e:
            logger.error(f"Erreur initialisation système de trading: {e}")
            self.trading_state.status = TradingStatus.ERROR
            return False

    def start_automated_trading(self, symbols: List[str] = None,
                                mode: TradingMode = TradingMode.FULL_AUTO) -> bool:
        """
        Démarre le trading automatisé

        Args:
            symbols: Liste des symboles à trader (défaut: R_10)
            mode: Mode de trading automatisé

        Returns:
            True si démarrage réussi
        """
        try:
            with self.trading_lock:
                if self.trading_state.status != TradingStatus.READY:
                    if not self.initialize_trading_system():
                        return False

                if self.running:
                    logger.warning("Trading automatisé déjà en cours")
                    return True

                # Configuration
                self.trading_symbols = symbols or self.config['default_trading_symbols']
                self.trading_state.mode = mode
                self.trading_state.auto_trading_enabled = True
                self.trading_state.status = TradingStatus.RUNNING
                self.running = True

                # Démarrer les threads
                self.auto_trading_thread = threading.Thread(
                    target=self._auto_trading_worker,
                    daemon=True
                )
                self.auto_trading_thread.start()

                self.monitoring_thread = threading.Thread(
                    target=self._monitoring_worker,
                    daemon=True
                )
                self.monitoring_thread.start()

                logger.info(f"🚀 Trading automatisé démarré ({mode.value}) sur {self.trading_symbols}")
                return True

        except Exception as e:
            logger.error(f"Erreur démarrage trading automatisé: {e}")
            self.trading_state.status = TradingStatus.ERROR
            return False

    def stop_automated_trading(self) -> bool:
        """Arrête le trading automatisé"""
        try:
            with self.trading_lock:
                if not self.running:
                    logger.info("Trading automatisé déjà arrêté")
                    return True

                logger.info("🛑 Arrêt du trading automatisé...")

                self.running = False
                self.trading_state.auto_trading_enabled = False
                self.trading_state.status = TradingStatus.PAUSED
                self.trading_state.mode = TradingMode.MANUAL

                # Attendre l'arrêt des threads
                if self.auto_trading_thread and self.auto_trading_thread.is_alive():
                    self.auto_trading_thread.join(timeout=5)

                if self.monitoring_thread and self.monitoring_thread.is_alive():
                    self.monitoring_thread.join(timeout=5)

                logger.info("✅ Trading automatisé arrêté")
                return True

        except Exception as e:
            logger.error(f"Erreur arrêt trading automatisé: {e}")
            return False

    def execute_complete_trading_cycle(self, symbol: str = "R_10",
                                       timeframe: str = "1m") -> Optional[Dict[str, Any]]:
        """
        Exécute un cycle complet de trading (décision -> validation -> exécution)

        Args:
            symbol: Symbole à trader
            timeframe: Timeframe d'analyse

        Returns:
            Résultats du cycle complet ou None si échec
        """
        try:
            cycle_start = time.time()

            logger.debug(f"🔄 Cycle de trading complet pour {symbol}")

            # Étape 1: Prise de décision
            decision = decision_engine.make_decision(symbol, timeframe)

            if not decision or decision.decision_type == DecisionType.HOLD:
                return {
                    'success': False,
                    'reason': 'Aucune décision de trading',
                    'decision': decision.to_dict() if decision else None,
                    'cycle_time_ms': (time.time() - cycle_start) * 1000
                }

            # Étape 2: Validation des risques
            position_type_str = "BUY" if decision.decision_type == DecisionType.BUY else "SELL"
            is_valid, risk_issues = validate_trade(
                symbol, position_type_str, decision.position_size,
                decision.entry_price or 100.0
            )

            if not is_valid:
                return {
                    'success': False,
                    'reason': 'Décision rejetée par la gestion des risques',
                    'decision': decision.to_dict(),
                    'risk_issues': risk_issues,
                    'cycle_time_ms': (time.time() - cycle_start) * 1000
                }

            # Étape 3: Vérification de la confiance
            if decision.confidence.value in ['very_low', 'low']:
                if decision.overall_score < self.config['min_confidence_for_execution']:
                    return {
                        'success': False,
                        'reason': 'Confiance insuffisante pour l\'exécution',
                        'decision': decision.to_dict(),
                        'confidence': decision.confidence.value,
                        'score': decision.overall_score,
                        'cycle_time_ms': (time.time() - cycle_start) * 1000
                    }

            # Étape 4: Exécution
            order_id = execute_trading_decision(decision)

            if not order_id:
                return {
                    'success': False,
                    'reason': 'Échec de l\'exécution de l\'ordre',
                    'decision': decision.to_dict(),
                    'cycle_time_ms': (time.time() - cycle_start) * 1000
                }

            # Étape 5: Mise à jour des statistiques
            self._update_daily_stats(decision, order_id)

            cycle_time = (time.time() - cycle_start) * 1000

            result = {
                'success': True,
                'decision': decision.to_dict(),
                'order_id': order_id,
                'risk_validation': 'passed',
                'execution_status': 'submitted',
                'cycle_time_ms': cycle_time
            }

            logger.info(f"✅ Cycle complet exécuté: {decision.decision_type.value} "
                        f"({cycle_time:.1f}ms)")

            return result

        except Exception as e:
            logger.error(f"Erreur cycle de trading: {e}")
            return {
                'success': False,
                'reason': f'Erreur technique: {str(e)}',
                'cycle_time_ms': (time.time() - cycle_start) * 1000
            }

    def _auto_trading_worker(self):
        """Worker pour le trading automatisé"""
        logger.info("🤖 Worker de trading automatisé démarré")

        while self.running:
            try:
                if not self.trading_state.auto_trading_enabled:
                    time.sleep(1)
                    continue

                # Vérifier les conditions de trading
                if not self._can_trade():
                    time.sleep(5)
                    continue

                # Exécuter un cycle pour chaque symbole
                for symbol in self.trading_symbols:
                    if not self.running:
                        break

                    try:
                        result = self.execute_complete_trading_cycle(symbol)

                        if result and result.get('success'):
                            logger.info(f"🎯 Trading automatisé réussi pour {symbol}")
                        else:
                            reason = result.get('reason', 'Raison inconnue') if result else 'Résultat null'
                            logger.debug(f"⏭️ Pas de trade pour {symbol}: {reason}")

                    except Exception as e:
                        logger.error(f"Erreur trading automatisé pour {symbol}: {e}")

                # Attendre avant le prochain cycle
                time.sleep(self.config['auto_decision_interval_seconds'])

            except Exception as e:
                logger.error(f"Erreur worker trading automatisé: {e}")
                time.sleep(10)

        logger.info("🛑 Worker de trading automatisé arrêté")

    def _monitoring_worker(self):
        """Worker de monitoring et maintenance"""
        logger.info("👁️ Worker de monitoring démarré")

        while self.running:
            try:
                # Vérification des risques
                self._check_emergency_conditions()

                # Mise à jour de la performance
                if (not self.last_performance_update or
                        (datetime.now(timezone.utc) - self.last_performance_update).total_seconds() >
                        self.config['performance_update_interval']):
                    self._update_trading_performance()

                # Nettoyage de l'historique
                self._cleanup_history()

                # Logger les statistiques périodiquement
                if time.time() % 300 < 30:  # Toutes les 5 minutes
                    self._log_trading_stats()

                time.sleep(self.config['risk_check_interval_seconds'])

            except Exception as e:
                logger.error(f"Erreur worker monitoring: {e}")
                time.sleep(30)

        logger.info("🛑 Worker de monitoring arrêté")

    def _can_trade(self) -> bool:
        """Vérifie si les conditions permettent de trader"""
        try:
            # Vérifier l'arrêt d'urgence
            if self.trading_state.emergency_stop_triggered:
                return False

            # Vérifier les composants
            if not all([
                self.trading_state.decision_engine_active,
                self.trading_state.risk_manager_active,
                self.trading_state.order_executor_active
            ]):
                return False

            # Vérifier les limites de décisions
            if self.trading_state.decisions_today >= self.config['max_decisions_per_hour']:
                return False

            # Vérifier le drawdown
            risk_summary = get_risk_summary()
            current_drawdown = risk_summary.get('risk_metrics', {}).get('current_drawdown', 0)

            if current_drawdown > self.config['emergency_drawdown_threshold']:
                logger.warning(f"Drawdown trop élevé: {current_drawdown:.1%}")
                return False

            return True

        except Exception as e:
            logger.error(f"Erreur vérification conditions de trading: {e}")
            return False

    def _check_emergency_conditions(self):
        """Vérifie les conditions d'arrêt d'urgence"""
        try:
            # Vérifier le gestionnaire de risques
            risk_summary = get_risk_summary()

            if risk_summary.get('emergency_stop', False):
                self.emergency_stop_all_trading()
                return

            # Vérifier les alertes critiques
            active_alerts = risk_summary.get('active_alerts', [])
            critical_alerts = [
                alert for alert in active_alerts
                if alert.get('severity') == 'critical'
            ]

            if critical_alerts:
                logger.critical(f"🚨 {len(critical_alerts)} alerte(s) critique(s) détectée(s)")
                for alert in critical_alerts:
                    logger.critical(f"   - {alert.get('message', 'Alerte inconnue')}")

                if len(critical_alerts) >= 3:  # Plus de 3 alertes critiques
                    self.emergency_stop_all_trading()

        except Exception as e:
            logger.error(f"Erreur vérification conditions d'urgence: {e}")

    def _update_trading_performance(self):
        """Met à jour les métriques de performance"""
        try:
            # Récupérer les données du gestionnaire de risques
            risk_summary = get_risk_summary()
            risk_metrics = risk_summary.get('risk_metrics', {})

            # Récupérer les statistiques d'exécution
            execution_summary = get_execution_summary()
            execution_stats = execution_summary.get('execution_stats', {})

            # Mettre à jour la performance
            self.trading_performance.total_pnl = risk_metrics.get('total_pnl', 0.0)
            self.trading_performance.max_drawdown = risk_metrics.get('max_drawdown', 0.0)
            self.trading_performance.current_exposure = risk_metrics.get('total_exposure', 0.0)
            self.trading_performance.risk_score = risk_metrics.get('risk_score', 0.0)

            # Métriques d'exécution
            if execution_stats.get('orders_sent', 0) > 0:
                self.trading_performance.total_trades = execution_stats['orders_sent']
                self.trading_performance.total_pnl = execution_stats.get('total_pnl', 0.0)

            # Calculer les ratios
            if self.trading_performance.total_trades > 0:
                filled_orders = execution_stats.get('orders_filled', 0)
                self.trading_performance.win_rate = filled_orders / self.trading_performance.total_trades

            self.trading_performance.last_update = datetime.now(timezone.utc)
            self.last_performance_update = self.trading_performance.last_update

            # Ajouter à l'historique
            self.performance_history.append({
                'timestamp': self.trading_performance.last_update,
                'performance': self.trading_performance.to_dict()
            })

        except Exception as e:
            logger.error(f"Erreur mise à jour performance: {e}")

    def _update_daily_stats(self, decision: TradingDecision, order_id: str):
        """Met à jour les statistiques journalières"""
        try:
            self.trading_state.decisions_today += 1
            self.trading_state.last_decision_time = decision.timestamp

            if order_id:
                self.trading_state.trades_today += 1
                self.trading_state.last_trade_time = datetime.now(timezone.utc)

            # Ajouter à l'historique
            self.decision_history.append({
                'timestamp': decision.timestamp,
                'decision': decision.to_dict(),
                'order_id': order_id
            })

        except Exception as e:
            logger.error(f"Erreur mise à jour stats journalières: {e}")

    def _cleanup_history(self):
        """Nettoie l'historique ancien"""
        try:
            # Limiter la taille des historiques
            max_history_size = 1000

            if len(self.decision_history) > max_history_size:
                self.decision_history = self.decision_history[-max_history_size // 2:]

            if len(self.performance_history) > max_history_size:
                self.performance_history = self.performance_history[-max_history_size // 2:]

        except Exception as e:
            logger.error(f"Erreur nettoyage historique: {e}")

    def _log_trading_stats(self):
        """Log les statistiques de trading"""
        try:
            logger.info(f"📊 Stats Trading - "
                        f"Décisions: {self.trading_state.decisions_today}, "
                        f"Trades: {self.trading_state.trades_today}, "
                        f"PnL: {self.trading_performance.total_pnl:.2f}, "
                        f"Exposition: {self.trading_performance.current_exposure:.2f}")

        except Exception as e:
            logger.error(f"Erreur log des stats: {e}")

    def emergency_stop_all_trading(self) -> bool:
        """Arrêt d'urgence complet du trading"""
        try:
            logger.critical("🚨 ARRÊT D'URGENCE DU TRADING ACTIVÉ")

            with self.trading_lock:
                # Marquer l'arrêt d'urgence
                self.trading_state.emergency_stop_triggered = True
                self.trading_state.status = TradingStatus.EMERGENCY_STOP
                self.trading_state.auto_trading_enabled = False

                # Arrêter le trading automatisé
                self.running = False

                # Activer l'arrêt d'urgence dans le gestionnaire de risques
                risk_manager.emergency_stop_triggered = True
                risk_manager.trading_enabled = False

                logger.critical("🛑 Tous les systèmes de trading arrêtés")
                return True

        except Exception as e:
            logger.error(f"Erreur arrêt d'urgence: {e}")
            return False

    def reset_emergency_stop(self) -> bool:
        """Remet à zéro l'arrêt d'urgence (utilisation manuelle)"""
        try:
            with self.trading_lock:
                if not self.trading_state.emergency_stop_triggered:
                    return True

                logger.warning("⚠️ Remise à zéro de l'arrêt d'urgence...")

                # Reset des états
                self.trading_state.emergency_stop_triggered = False
                self.trading_state.status = TradingStatus.READY

                # Reset du gestionnaire de risques
                risk_manager.reset_emergency_stop()

                logger.info("✅ Arrêt d'urgence remis à zéro")
                return True

        except Exception as e:
            logger.error(f"Erreur reset arrêt d'urgence: {e}")
            return False

    def get_trading_pipeline_status(self) -> Dict[str, Any]:
        """Retourne le statut complet du pipeline de trading"""
        try:
            # Statuts des composants
            decision_stats = get_decision_stats()
            risk_summary = get_risk_summary()
            execution_summary = get_execution_summary()

            status = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'trading_state': self.trading_state.to_dict(),
                'trading_performance': self.trading_performance.to_dict(),
                'components': {
                    'decision_engine': {
                        'active': self.trading_state.decision_engine_active,
                        'stats': decision_stats
                    },
                    'risk_manager': {
                        'active': self.trading_state.risk_manager_active,
                        'summary': risk_summary
                    },
                    'order_executor': {
                        'active': self.trading_state.order_executor_active,
                        'summary': execution_summary
                    }
                },
                'recent_activity': {
                    'decisions_count': len(self.decision_history),
                    'last_decision': self.decision_history[-1] if self.decision_history else None,
                    'performance_updates': len(self.performance_history)
                }
            }

            # Santé globale
            if self.trading_state.status == TradingStatus.EMERGENCY_STOP:
                status['overall_health'] = 'critical'
            elif self.trading_state.status in [TradingStatus.ERROR, TradingStatus.PAUSED]:
                status['overall_health'] = 'warning'
            elif self.trading_state.status == TradingStatus.RUNNING:
                status['overall_health'] = 'excellent'
            else:
                status['overall_health'] = 'good'

            return status

        except Exception as e:
            logger.error(f"Erreur récupération statut pipeline: {e}")
            return {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'error': str(e),
                'overall_health': 'error'
            }

    def get_portfolio_overview(self) -> Dict[str, Any]:
        """Retourne un aperçu complet du portfolio"""
        try:
            risk_summary = get_risk_summary()
            execution_summary = get_execution_summary()

            overview = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'account_info': {
                    'balance': risk_summary.get('risk_metrics', {}).get('account_balance', 0),
                    'equity': risk_summary.get('risk_metrics', {}).get('total_equity', 0),
                    'free_margin': risk_summary.get('risk_metrics', {}).get('free_margin', 0),
                    'margin_level': risk_summary.get('risk_metrics', {}).get('margin_level', 0)
                },
                'positions': {
                    'count': risk_summary.get('risk_metrics', {}).get('open_positions', 0),
                    'total_exposure': risk_summary.get('risk_metrics', {}).get('total_exposure', 0),
                    'unrealized_pnl': risk_summary.get('positions_summary', {}).get('unrealized_pnl', 0),
                    'symbols': risk_summary.get('positions_summary', {}).get('symbols', [])
                },
                'performance': {
                    'total_pnl': self.trading_performance.total_pnl,
                    'win_rate': self.trading_performance.win_rate,
                    'max_drawdown': self.trading_performance.max_drawdown,
                    'profit_factor': self.trading_performance.profit_factor,
                    'total_trades': self.trading_performance.total_trades
                },
                'today': {
                    'decisions': self.trading_state.decisions_today,
                    'trades': self.trading_state.trades_today,
                    'pnl': self.trading_state.pnl_today
                },
                'risk': {
                    'score': self.trading_performance.risk_score,
                    'current_drawdown': risk_summary.get('risk_metrics', {}).get('current_drawdown', 0),
                    'active_alerts': len(risk_summary.get('active_alerts', [])),
                    'emergency_stop': self.trading_state.emergency_stop_triggered
                },
                'execution': {
                    'orders_filled': execution_summary.get('execution_stats', {}).get('orders_filled', 0),
                    'orders_rejected': execution_summary.get('execution_stats', {}).get('orders_rejected', 0),
                    'avg_execution_time': execution_summary.get('execution_stats', {}).get('avg_execution_time_ms', 0),
                    'paper_trading': execution_summary.get('paper_trading', True)
                }
            }

            return overview

        except Exception as e:
            logger.error(f"Erreur génération aperçu portfolio: {e}")
            return {'error': str(e)}

    def optimize_trading_parameters(self, optimization_period_days: int = 30) -> Dict[str, Any]:
        """
        Optimise les paramètres de trading basés sur la performance historique

        Args:
            optimization_period_days: Période d'analyse en jours

        Returns:
            Paramètres optimisés et recommandations
        """
        try:
            logger.info(f"🔧 Optimisation des paramètres de trading sur {optimization_period_days} jours")

            optimization = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'analysis_period_days': optimization_period_days,
                'current_parameters': {},
                'optimized_parameters': {},
                'recommendations': [],
                'performance_impact': {}
            }

            # Analyser la performance actuelle
            current_performance = self.trading_performance.to_dict()

            # Paramètres actuels
            optimization['current_parameters'] = {
                'max_position_size': risk_manager.risk_limits.max_position_size_pct,
                'max_drawdown': risk_manager.risk_limits.max_drawdown_pct,
                'decision_interval': self.config['auto_decision_interval_seconds'],
                'min_confidence': self.config['min_confidence_for_execution'],
                'max_decisions_per_hour': self.config['max_decisions_per_hour']
            }

            # Analyser les patterns de performance
            recommendations = []

            # Optimisation basée sur le win rate
            if self.trading_performance.win_rate < 0.5:
                recommendations.append({
                    'parameter': 'min_confidence_for_execution',
                    'current_value': self.config['min_confidence_for_execution'],
                    'suggested_value': min(0.8, self.config['min_confidence_for_execution'] + 0.1),
                    'reason': 'Win rate faible - augmenter le seuil de confiance'
                })

            # Optimisation basée sur le drawdown
            if self.trading_performance.max_drawdown > 0.05:  # Plus de 5%
                recommendations.append({
                    'parameter': 'max_position_size_pct',
                    'current_value': risk_manager.risk_limits.max_position_size_pct,
                    'suggested_value': max(0.01, risk_manager.risk_limits.max_position_size_pct * 0.8),
                    'reason': 'Drawdown élevé - réduire la taille des positions'
                })

            # Optimisation basée sur le nombre de trades
            if self.trading_state.decisions_today < 5:  # Peu de trades
                recommendations.append({
                    'parameter': 'auto_decision_interval_seconds',
                    'current_value': self.config['auto_decision_interval_seconds'],
                    'suggested_value': max(30, self.config['auto_decision_interval_seconds'] - 15),
                    'reason': 'Peu d\'activité - réduire l\'intervalle de décision'
                })

            # Optimisation basée sur les erreurs
            execution_summary = get_execution_summary()
            rejection_rate = execution_summary.get('execution_stats', {}).get('orders_rejected', 0)

            if rejection_rate > 0.2:  # Plus de 20% de rejets
                recommendations.append({
                    'parameter': 'risk_check_enhancement',
                    'current_value': 'standard',
                    'suggested_value': 'enhanced',
                    'reason': 'Taux de rejet élevé - améliorer la validation des risques'
                })

            optimization['recommendations'] = recommendations

            # Estimation de l'impact
            if recommendations:
                optimization['performance_impact'] = {
                    'estimated_win_rate_improvement': 0.05,
                    'estimated_drawdown_reduction': 0.02,
                    'estimated_sharpe_improvement': 0.1
                }

            logger.info(f"✅ Optimisation terminée: {len(recommendations)} recommandation(s)")
            return optimization

        except Exception as e:
            logger.error(f"Erreur optimisation paramètres: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }


# Instance globale de l'orchestrateur
trading_orchestrator = TradingOrchestrator()


# Fonctions principales du module
def get_trading_pipeline_status() -> Dict[str, Any]:
    """Retourne le statut complet du pipeline de trading"""
    return trading_orchestrator.get_trading_pipeline_status()


def execute_complete_trading_cycle(symbol: str = "R_10", timeframe: str = "1m") -> Optional[Dict[str, Any]]:
    """Exécute un cycle complet de trading"""
    return trading_orchestrator.execute_complete_trading_cycle(symbol, timeframe)


def start_automated_trading(symbols: List[str] = None, mode: str = "full_auto") -> bool:
    """Démarre le trading automatisé"""
    trading_mode = TradingMode.FULL_AUTO if mode == "full_auto" else TradingMode.SEMI_AUTO
    return trading_orchestrator.start_automated_trading(symbols, trading_mode)


def stop_automated_trading() -> bool:
    """Arrête le trading automatisé"""
    return trading_orchestrator.stop_automated_trading()


def get_trading_performance() -> TradingPerformance:
    """Retourne les métriques de performance"""
    return trading_orchestrator.trading_performance


def emergency_stop_all_trading() -> bool:
    """Arrêt d'urgence complet"""
    return trading_orchestrator.emergency_stop_all_trading()


def get_portfolio_overview() -> Dict[str, Any]:
    """Retourne un aperçu du portfolio"""
    return trading_orchestrator.get_portfolio_overview()


def optimize_trading_parameters(days: int = 30) -> Dict[str, Any]:
    """Optimise les paramètres de trading"""
    return trading_orchestrator.optimize_trading_parameters(days)


def initialize_trading_system() -> bool:
    """Initialise le système de trading"""
    return trading_orchestrator.initialize_trading_system()


def print_trading_banner():
    """Affiche la bannière du module de trading"""
    print("=" * 80)
    print("💼 TRADING BOT VOLATILITY 10 - MODULE DE TRADING")
    print("=" * 80)
    print(f"📦 Version: {__version__}")
    print(f"👥 Auteur: {__author__}")
    print(f"📝 Description: {__description__}")

    # Statut du pipeline
    status = get_trading_pipeline_status()
    health_emoji = {
        "excellent": "✅",
        "good": "🟢",
        "warning": "⚠️",
        "critical": "❌",
        "error": "💥"
    }

    overall_health = status.get('overall_health', 'unknown')
    print(f"\n🏥 Santé du pipeline: {health_emoji.get(overall_health, '❓')} {overall_health.upper()}")

    # État du trading
    trading_state = status.get('trading_state', {})
    print(f"🤖 Mode: {trading_state.get('mode', 'unknown').upper()}")
    print(f"🔄 Statut: {trading_state.get('status', 'unknown').upper()}")
    print(f"⚡ Auto-trading: {'ON' if trading_state.get('auto_trading_enabled') else 'OFF'}")

    # Composants
    components = status.get('components', {})
    for component, info in components.items():
        active = info.get('active', False)
        print(f"   {'✅' if active else '❌'} {component.replace('_', ' ').title()}")

    # Performance aujourd'hui
    print(f"\n📊 Aujourd'hui:")
    print(f"   Décisions: {trading_state.get('decisions_today', 0)}")
    print(f"   Trades: {trading_state.get('trades_today', 0)}")
    print(f"   P&L: {trading_state.get('pnl_today', 0):.2f}")

    # Arrêt d'urgence
    if trading_state.get('emergency_stop_triggered'):
        print(f"\n🚨 ARRÊT D'URGENCE ACTIVÉ")

    print("=" * 80)


# Initialisation automatique au chargement du module
try:
    logger.info(f"Module de trading chargé (version {__version__})")

    # Vérifier que tous les sous-modules sont chargés
    components_loaded = [
        decision_engine is not None,
        risk_manager is not None,
        order_executor is not None
    ]

    if all(components_loaded):
        logger.info("✅ Tous les modules de trading sont opérationnels")

        # Initialisation silencieuse du système
        try:
            trading_orchestrator.initialize_trading_system()
        except Exception as e:
            logger.warning(f"Initialisation partielle du système de trading: {e}")
    else:
        logger.warning("⚠️ Certains modules de trading ne sont pas chargés")

except Exception as e:
    logger.error(f"⚠️ Erreur lors de l'initialisation du module de trading: {e}")
    # Ne pas faire planter l'import, permettre l'utilisation partielle