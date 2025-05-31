"""
Gestionnaire de risques pour le Trading Bot Volatility 10
Gestion avancée des risques, position sizing et protection du capital
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
import threading
from threading import Lock

from config import config
from data import db_manager, Trades

# Configuration du logger
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class RiskLevel(Enum):
    """Niveaux de risque"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CRITICAL = "critical"


class PositionType(Enum):
    """Types de position"""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class RiskEvent(Enum):
    """Types d'événements de risque"""
    DRAWDOWN_LIMIT = "drawdown_limit"
    POSITION_LIMIT = "position_limit"
    CORRELATION_LIMIT = "correlation_limit"
    VOLATILITY_SPIKE = "volatility_spike"
    LOSS_STREAK = "loss_streak"
    MARGIN_CALL = "margin_call"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class PositionInfo:
    """Informations sur une position"""
    position_id: str
    symbol: str
    position_type: PositionType
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_time: datetime = None
    duration_hours: float = 0.0
    max_adverse_excursion: float = 0.0  # MAE
    max_favorable_excursion: float = 0.0  # MFE

    @property
    def pnl_percentage(self) -> float:
        """PnL en pourcentage"""
        if self.entry_price > 0:
            return (self.unrealized_pnl / (self.entry_price * self.size)) * 100
        return 0.0

    @property
    def is_profitable(self) -> bool:
        """Position est-elle profitable"""
        return self.unrealized_pnl > 0


@dataclass
class RiskMetrics:
    """Métriques de risque du portfolio"""
    # Capital et exposition
    account_balance: float = 0.0
    total_equity: float = 0.0
    used_margin: float = 0.0
    free_margin: float = 0.0
    margin_level: float = 0.0

    # Exposition
    total_exposure: float = 0.0
    long_exposure: float = 0.0
    short_exposure: float = 0.0
    net_exposure: float = 0.0

    # Drawdown
    peak_equity: float = 0.0
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    drawdown_duration_days: int = 0

    # Performance
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0

    # Risque
    var_daily: float = 0.0  # Value at Risk
    expected_shortfall: float = 0.0
    risk_score: float = 0.0

    # Positions
    open_positions: int = 0
    winning_positions: int = 0
    losing_positions: int = 0

    def to_dict(self) -> Dict[str, float]:
        return {
            'account_balance': self.account_balance,
            'total_equity': self.total_equity,
            'used_margin': self.used_margin,
            'free_margin': self.free_margin,
            'margin_level': self.margin_level,
            'total_exposure': self.total_exposure,
            'net_exposure': self.net_exposure,
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'total_pnl': self.total_pnl,
            'daily_pnl': self.daily_pnl,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'sharpe_ratio': self.sharpe_ratio,
            'var_daily': self.var_daily,
            'risk_score': self.risk_score,
            'open_positions': self.open_positions
        }


@dataclass
class RiskLimits:
    """Limites de risque configurables"""
    # Limites de position
    max_position_size_pct: float = 0.02  # 2% du capital par position
    max_total_exposure_pct: float = 0.10  # 10% d'exposition totale
    max_symbol_exposure_pct: float = 0.05  # 5% par symbole
    max_positions_count: int = 5  # Nombre max de positions

    # Limites de drawdown
    max_daily_loss_pct: float = 0.02  # 2% de perte journalière max
    max_drawdown_pct: float = 0.10  # 10% de drawdown max
    emergency_stop_pct: float = 0.05  # 5% pour arrêt d'urgence

    # Limites de trading
    max_trades_per_day: int = 20
    max_losing_streak: int = 5
    min_time_between_trades_minutes: int = 5

    # Limites de margin
    min_margin_level_pct: float = 200.0  # 200% minimum
    margin_call_level_pct: float = 150.0  # 150% pour margin call

    # Volatilité
    max_volatility_threshold: float = 0.05  # 5% de volatilité max


@dataclass
class RiskAlert:
    """Alerte de risque"""
    alert_id: str
    risk_event: RiskEvent
    severity: RiskLevel
    message: str
    timestamp: datetime
    symbol: Optional[str] = None
    position_id: Optional[str] = None
    metric_value: Optional[float] = None
    threshold_value: Optional[float] = None
    recommended_action: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'alert_id': self.alert_id,
            'risk_event': self.risk_event.value,
            'severity': self.severity.value,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'position_id': self.position_id,
            'metric_value': self.metric_value,
            'threshold_value': self.threshold_value,
            'recommended_action': self.recommended_action
        }


class RiskManager:
    """Gestionnaire de risques avancé"""

    def __init__(self, initial_balance: float = 1000.0):
        # Configuration
        self.risk_limits = RiskLimits()
        self.initial_balance = initial_balance
        self.current_balance = initial_balance

        # État du portfolio
        self.positions = {}  # Dict[str, PositionInfo]
        self.equity_history = []
        self.pnl_history = []
        self.risk_metrics = RiskMetrics(account_balance=initial_balance)

        # Alertes et événements
        self.active_alerts = []
        self.alert_history = []
        self.risk_events = []

        # Protection
        self.trading_enabled = True
        self.emergency_stop_triggered = False
        self.last_trade_time = None

        # Statistiques
        self.risk_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'consecutive_losses': 0,
            'max_consecutive_losses': 0,
            'alerts_generated': 0,
            'risk_events_triggered': 0,
            'emergency_stops': 0,
            'last_risk_check': None
        }

        # Threading
        self.risk_lock = Lock()

        logger.info(f"Gestionnaire de risques initialisé avec {initial_balance} de capital")

    def calculate_position_size(self, symbol: str, entry_price: float,
                                stop_loss: Optional[float] = None,
                                risk_percentage: Optional[float] = None) -> float:
        """
        Calcule la taille de position optimale basée sur le risque

        Args:
            symbol: Symbole à trader
            entry_price: Prix d'entrée
            stop_loss: Niveau de stop loss
            risk_percentage: Pourcentage de risque souhaité

        Returns:
            Taille de position recommandée
        """
        try:
            with self.risk_lock:
                # Vérifier si le trading est autorisé
                if not self.trading_enabled or self.emergency_stop_triggered:
                    logger.warning("Trading désactivé - taille position = 0")
                    return 0.0

                # Pourcentage de risque par défaut
                if risk_percentage is None:
                    risk_percentage = self.risk_limits.max_position_size_pct

                # Capital disponible pour le risque
                available_capital = self.current_balance
                risk_capital = available_capital * risk_percentage

                # Méthode 1: Position sizing basé sur le stop loss
                if stop_loss and stop_loss != entry_price:
                    risk_per_unit = abs(entry_price - stop_loss)
                    position_size_risk_based = risk_capital / risk_per_unit
                else:
                    # Méthode 2: Position sizing basé sur la volatilité
                    volatility = self._estimate_symbol_volatility(symbol)
                    if volatility > 0:
                        # Utiliser 2x volatilité comme risque estimé
                        estimated_risk_per_unit = entry_price * volatility * 2
                        position_size_risk_based = risk_capital / estimated_risk_per_unit
                    else:
                        # Fallback: position size fixe
                        position_size_risk_based = available_capital * 0.01  # 1%

                # Vérifier les limites
                position_size = self._apply_position_limits(
                    symbol, position_size_risk_based, entry_price
                )

                logger.debug(f"Position size calculée pour {symbol}: {position_size:.6f}")
                return position_size

        except Exception as e:
            logger.error(f"Erreur calcul position size: {e}")
            return 0.0

    def _estimate_symbol_volatility(self, symbol: str, periods: int = 20) -> float:
        """Estime la volatilité d'un symbole"""
        try:
            # Récupérer les données récentes
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=2)

            price_data = db_manager.get_price_data(
                symbol=symbol,
                timeframe="1m",
                start_time=start_time,
                end_time=end_time,
                limit=periods + 10
            )

            if not price_data or len(price_data) < periods:
                return 0.01  # 1% par défaut

            # Calculer la volatilité
            df = pd.DataFrame([data.to_dict() for data in price_data])
            closes = df['close']
            returns = closes.pct_change().dropna()

            if len(returns) > 5:
                volatility = returns.std()
                return min(0.1, max(0.001, volatility))  # Entre 0.1% et 10%

            return 0.01

        except Exception as e:
            logger.error(f"Erreur estimation volatilité: {e}")
            return 0.01

    def _apply_position_limits(self, symbol: str, requested_size: float,
                               entry_price: float) -> float:
        """Applique les limites de position"""
        try:
            # Limite par position
            max_position_value = self.current_balance * self.risk_limits.max_position_size_pct
            max_size_by_position = max_position_value / entry_price

            # Limite par symbole (exposition existante)
            current_symbol_exposure = self._get_symbol_exposure(symbol)
            max_symbol_value = self.current_balance * self.risk_limits.max_symbol_exposure_pct
            max_additional_size = (max_symbol_value - current_symbol_exposure) / entry_price

            # Limite d'exposition totale
            current_total_exposure = self.risk_metrics.total_exposure
            max_total_value = self.current_balance * self.risk_limits.max_total_exposure_pct
            max_size_by_total = (max_total_value - current_total_exposure) / entry_price

            # Limite par nombre de positions
            if len(self.positions) >= self.risk_limits.max_positions_count:
                logger.warning("Nombre maximum de positions atteint")
                return 0.0

            # Prendre le minimum de toutes les limites
            final_size = min(
                requested_size,
                max_size_by_position,
                max_additional_size,
                max_size_by_total
            )

            # Assurer que la taille est positive
            return max(0.0, final_size)

        except Exception as e:
            logger.error(f"Erreur application des limites: {e}")
            return 0.0

    def _get_symbol_exposure(self, symbol: str) -> float:
        """Calcule l'exposition actuelle pour un symbole"""
        try:
            exposure = 0.0
            for position in self.positions.values():
                if position.symbol == symbol:
                    exposure += abs(position.size * position.current_price)
            return exposure
        except Exception as e:
            logger.error(f"Erreur calcul exposition symbole: {e}")
            return 0.0

    def validate_trade(self, symbol: str, position_type: PositionType,
                       size: float, entry_price: float,
                       stop_loss: Optional[float] = None,
                       take_profit: Optional[float] = None) -> Tuple[bool, List[str]]:
        """
        Valide si un trade peut être exécuté selon les règles de risque

        Args:
            symbol: Symbole à trader
            position_type: Type de position (LONG/SHORT)
            size: Taille de la position
            entry_price: Prix d'entrée
            stop_loss: Stop loss (optionnel)
            take_profit: Take profit (optionnel)

        Returns:
            Tuple (validation_passed, list_of_issues)
        """
        try:
            with self.risk_lock:
                issues = []

                # Vérifier si le trading est autorisé
                if not self.trading_enabled:
                    issues.append("Trading désactivé")
                    return False, issues

                if self.emergency_stop_triggered:
                    issues.append("Arrêt d'urgence activé")
                    return False, issues

                # Vérifier les limites temporelles
                if not self._check_time_limits():
                    issues.append("Limites temporelles non respectées")

                # Vérifier les limites de position
                position_value = size * entry_price
                max_position_value = self.current_balance * self.risk_limits.max_position_size_pct

                if position_value > max_position_value:
                    issues.append(f"Position trop grande: {position_value:.2f} > {max_position_value:.2f}")

                # Vérifier l'exposition totale
                new_total_exposure = self.risk_metrics.total_exposure + position_value
                max_total_exposure = self.current_balance * self.risk_limits.max_total_exposure_pct

                if new_total_exposure > max_total_exposure:
                    issues.append(f"Exposition totale dépassée: {new_total_exposure:.2f} > {max_total_exposure:.2f}")

                # Vérifier l'exposition par symbole
                current_symbol_exposure = self._get_symbol_exposure(symbol)
                new_symbol_exposure = current_symbol_exposure + position_value
                max_symbol_exposure = self.current_balance * self.risk_limits.max_symbol_exposure_pct

                if new_symbol_exposure > max_symbol_exposure:
                    issues.append(
                        f"Exposition {symbol} dépassée: {new_symbol_exposure:.2f} > {max_symbol_exposure:.2f}")

                # Vérifier le nombre de positions
                if len(self.positions) >= self.risk_limits.max_positions_count:
                    issues.append(f"Nombre max de positions atteint: {len(self.positions)}")

                # Vérifier les niveaux de stop loss/take profit
                if stop_loss:
                    if position_type == PositionType.LONG and stop_loss >= entry_price:
                        issues.append("Stop loss LONG invalide")
                    elif position_type == PositionType.SHORT and stop_loss <= entry_price:
                        issues.append("Stop loss SHORT invalide")

                if take_profit:
                    if position_type == PositionType.LONG and take_profit <= entry_price:
                        issues.append("Take profit LONG invalide")
                    elif position_type == PositionType.SHORT and take_profit >= entry_price:
                        issues.append("Take profit SHORT invalide")

                # Vérifier le drawdown actuel
                if self.risk_metrics.current_drawdown > self.risk_limits.max_drawdown_pct:
                    issues.append(f"Drawdown trop élevé: {self.risk_metrics.current_drawdown:.1%}")

                # Vérifier les pertes consécutives
                if self.risk_stats['consecutive_losses'] >= self.risk_limits.max_losing_streak:
                    issues.append(f"Trop de pertes consécutives: {self.risk_stats['consecutive_losses']}")

                # Vérifier la marge disponible
                if self.risk_metrics.margin_level < self.risk_limits.min_margin_level_pct:
                    issues.append(f"Niveau de marge insuffisant: {self.risk_metrics.margin_level:.1f}%")

                return len(issues) == 0, issues

        except Exception as e:
            logger.error(f"Erreur validation trade: {e}")
            return False, [f"Erreur de validation: {str(e)}"]

    def _check_time_limits(self) -> bool:
        """Vérifie les limites temporelles"""
        try:
            current_time = datetime.now(timezone.utc)

            # Vérifier le temps minimum entre trades
            if self.last_trade_time:
                time_since_last = (current_time - self.last_trade_time).total_seconds()
                min_interval = self.risk_limits.min_time_between_trades_minutes * 60

                if time_since_last < min_interval:
                    return False

            # Vérifier le nombre de trades aujourd'hui
            today_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
            today_trades = len([
                t for t in self.risk_events
                if t.get('timestamp', datetime.min.replace(tzinfo=timezone.utc)) > today_start
                   and t.get('event_type') == 'trade_opened'
            ])

            if today_trades >= self.risk_limits.max_trades_per_day:
                return False

            return True

        except Exception as e:
            logger.error(f"Erreur vérification limites temporelles: {e}")
            return False

    def add_position(self, position_id: str, symbol: str, position_type: PositionType,
                     size: float, entry_price: float, stop_loss: Optional[float] = None,
                     take_profit: Optional[float] = None) -> bool:
        """
        Ajoute une nouvelle position au portfolio

        Args:
            position_id: ID unique de la position
            symbol: Symbole tradé
            position_type: Type de position
            size: Taille de la position
            entry_price: Prix d'entrée
            stop_loss: Stop loss (optionnel)
            take_profit: Take profit (optionnel)

        Returns:
            True si ajout réussi
        """
        try:
            with self.risk_lock:
                if position_id in self.positions:
                    logger.warning(f"Position {position_id} existe déjà")
                    return False

                position = PositionInfo(
                    position_id=position_id,
                    symbol=symbol,
                    position_type=position_type,
                    size=size,
                    entry_price=entry_price,
                    current_price=entry_price,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    entry_time=datetime.now(timezone.utc)
                )

                self.positions[position_id] = position
                self.last_trade_time = datetime.now(timezone.utc)

                # Enregistrer l'événement
                self.risk_events.append({
                    'timestamp': datetime.now(timezone.utc),
                    'event_type': 'trade_opened',
                    'position_id': position_id,
                    'symbol': symbol,
                    'size': size,
                    'entry_price': entry_price
                })

                # Mettre à jour les métriques
                self._update_risk_metrics()

                logger.info(f"Position ajoutée: {position_id} ({symbol}, {size:.6f})")
                return True

        except Exception as e:
            logger.error(f"Erreur ajout position: {e}")
            return False

    def update_position_price(self, position_id: str, current_price: float) -> bool:
        """
        Met à jour le prix actuel d'une position

        Args:
            position_id: ID de la position
            current_price: Prix actuel

        Returns:
            True si mise à jour réussie
        """
        try:
            with self.risk_lock:
                if position_id not in self.positions:
                    logger.warning(f"Position {position_id} non trouvée")
                    return False

                position = self.positions[position_id]
                old_price = position.current_price
                position.current_price = current_price

                # Calculer le PnL non réalisé
                if position.position_type == PositionType.LONG:
                    position.unrealized_pnl = (current_price - position.entry_price) * position.size
                else:  # SHORT
                    position.unrealized_pnl = (position.entry_price - current_price) * position.size

                # Mettre à jour MAE et MFE
                pnl_pct = position.pnl_percentage
                if pnl_pct < 0:  # Perte
                    position.max_adverse_excursion = min(position.max_adverse_excursion, pnl_pct)
                else:  # Profit
                    position.max_favorable_excursion = max(position.max_favorable_excursion, pnl_pct)

                # Vérifier les niveaux de stop loss et take profit
                self._check_exit_levels(position)

                # Mettre à jour les métriques globales
                self._update_risk_metrics()

                return True

        except Exception as e:
            logger.error(f"Erreur mise à jour prix position: {e}")
            return False

    def _check_exit_levels(self, position: PositionInfo):
        """Vérifie si les niveaux d'exit sont atteints"""
        try:
            alerts_to_add = []

            # Vérifier stop loss
            if position.stop_loss:
                if position.position_type == PositionType.LONG and position.current_price <= position.stop_loss:
                    alerts_to_add.append(
                        RiskAlert(
                            alert_id=f"sl_{position.position_id}_{int(datetime.now().timestamp())}",
                            risk_event=RiskEvent.DRAWDOWN_LIMIT,
                            severity=RiskLevel.HIGH,
                            message=f"Stop Loss atteint pour {position.symbol}",
                            timestamp=datetime.now(timezone.utc),
                            symbol=position.symbol,
                            position_id=position.position_id,
                            metric_value=position.current_price,
                            threshold_value=position.stop_loss,
                            recommended_action="Fermer la position immédiatement"
                        )
                    )
                elif position.position_type == PositionType.SHORT and position.current_price >= position.stop_loss:
                    alerts_to_add.append(
                        RiskAlert(
                            alert_id=f"sl_{position.position_id}_{int(datetime.now().timestamp())}",
                            risk_event=RiskEvent.DRAWDOWN_LIMIT,
                            severity=RiskLevel.HIGH,
                            message=f"Stop Loss atteint pour {position.symbol}",
                            timestamp=datetime.now(timezone.utc),
                            symbol=position.symbol,
                            position_id=position.position_id,
                            metric_value=position.current_price,
                            threshold_value=position.stop_loss,
                            recommended_action="Fermer la position immédiatement"
                        )
                    )

            # Vérifier take profit
            if position.take_profit:
                if position.position_type == PositionType.LONG and position.current_price >= position.take_profit:
                    alerts_to_add.append(
                        RiskAlert(
                            alert_id=f"tp_{position.position_id}_{int(datetime.now().timestamp())}",
                            risk_event=RiskEvent.POSITION_LIMIT,
                            severity=RiskLevel.LOW,
                            message=f"Take Profit atteint pour {position.symbol}",
                            timestamp=datetime.now(timezone.utc),
                            symbol=position.symbol,
                            position_id=position.position_id,
                            metric_value=position.current_price,
                            threshold_value=position.take_profit,
                            recommended_action="Fermer la position pour sécuriser le profit"
                        )
                    )
                elif position.position_type == PositionType.SHORT and position.current_price <= position.take_profit:
                    alerts_to_add.append(
                        RiskAlert(
                            alert_id=f"tp_{position.position_id}_{int(datetime.now().timestamp())}",
                            risk_event=RiskEvent.POSITION_LIMIT,
                            severity=RiskLevel.LOW,
                            message=f"Take Profit atteint pour {position.symbol}",
                            timestamp=datetime.now(timezone.utc),
                            symbol=position.symbol,
                            position_id=position.position_id,
                            metric_value=position.current_price,
                            threshold_value=position.take_profit,
                            recommended_action="Fermer la position pour sécuriser le profit"
                        )
                    )

            # Ajouter les alertes
            for alert in alerts_to_add:
                self._add_risk_alert(alert)

        except Exception as e:
            logger.error(f"Erreur vérification niveaux exit: {e}")

    def close_position(self, position_id: str, exit_price: float, reason: str = "") -> bool:
        """
        Ferme une position

        Args:
            position_id: ID de la position
            exit_price: Prix de sortie
            reason: Raison de la fermeture

        Returns:
            True si fermeture réussie
        """
        try:
            with self.risk_lock:
                if position_id not in self.positions:
                    logger.warning(f"Position {position_id} non trouvée")
                    return False

                position = self.positions[position_id]

                # Calculer le PnL réalisé
                if position.position_type == PositionType.LONG:
                    realized_pnl = (exit_price - position.entry_price) * position.size
                else:  # SHORT
                    realized_pnl = (position.entry_price - exit_price) * position.size

                # Mettre à jour les statistiques
                self.risk_stats['total_trades'] += 1

                if realized_pnl > 0:
                    self.risk_stats['winning_trades'] += 1
                    self.risk_stats['consecutive_losses'] = 0
                else:
                    self.risk_stats['losing_trades'] += 1
                    self.risk_stats['consecutive_losses'] += 1
                    self.risk_stats['max_consecutive_losses'] = max(
                        self.risk_stats['max_consecutive_losses'],
                        self.risk_stats['consecutive_losses']
                    )

                # Enregistrer dans l'historique
                self.pnl_history.append({
                    'timestamp': datetime.now(timezone.utc),
                    'position_id': position_id,
                    'symbol': position.symbol,
                    'pnl': realized_pnl,
                    'entry_price': position.entry_price,
                    'exit_price': exit_price,
                    'duration_hours': position.duration_hours,
                    'reason': reason
                })

                # Mettre à jour le solde
                self.current_balance += realized_pnl

                # Supprimer la position
                del self.positions[position_id]

                # Enregistrer l'événement
                self.risk_events.append({
                    'timestamp': datetime.now(timezone.utc),
                    'event_type': 'trade_closed',
                    'position_id': position_id,
                    'symbol': position.symbol,
                    'exit_price': exit_price,
                    'realized_pnl': realized_pnl,
                    'reason': reason
                })

                # Mettre à jour les métriques
                self._update_risk_metrics()

                logger.info(f"Position fermée: {position_id} (PnL: {realized_pnl:.2f}, Raison: {reason})")
                return True

        except Exception as e:
            logger.error(f"Erreur fermeture position: {e}")
            return False

    def _update_risk_metrics(self):
        """Met à jour toutes les métriques de risque"""
        try:
            # Calculer l'équité totale
            unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            self.risk_metrics.total_equity = self.current_balance + unrealized_pnl

            # Calculer les expositions
            long_exposure = sum(
                pos.size * pos.current_price
                for pos in self.positions.values()
                if pos.position_type == PositionType.LONG
            )

            short_exposure = sum(
                pos.size * pos.current_price
                for pos in self.positions.values()
                if pos.position_type == PositionType.SHORT
            )

            self.risk_metrics.long_exposure = long_exposure
            self.risk_metrics.short_exposure = short_exposure
            self.risk_metrics.total_exposure = long_exposure + short_exposure
            self.risk_metrics.net_exposure = long_exposure - short_exposure

            # Calculer le drawdown
            if self.risk_metrics.total_equity > self.risk_metrics.peak_equity:
                self.risk_metrics.peak_equity = self.risk_metrics.total_equity

            self.risk_metrics.current_drawdown = (
                    (self.risk_metrics.peak_equity - self.risk_metrics.total_equity) /
                    self.risk_metrics.peak_equity
            ) if self.risk_metrics.peak_equity > 0 else 0

            self.risk_metrics.max_drawdown = max(
                self.risk_metrics.max_drawdown,
                self.risk_metrics.current_drawdown
            )

            # Calculer le PnL
            total_realized_pnl = sum(trade['pnl'] for trade in self.pnl_history)
            self.risk_metrics.total_pnl = total_realized_pnl + unrealized_pnl

            # Calculer le win rate
            if self.risk_stats['total_trades'] > 0:
                self.risk_metrics.win_rate = (
                        self.risk_stats['winning_trades'] / self.risk_stats['total_trades']
                )

            # Calculer le profit factor
            winning_pnl = sum(
                trade['pnl'] for trade in self.pnl_history if trade['pnl'] > 0
            )
            losing_pnl = abs(sum(
                trade['pnl'] for trade in self.pnl_history if trade['pnl'] < 0
            ))

            if losing_pnl > 0:
                self.risk_metrics.profit_factor = winning_pnl / losing_pnl

            # Positions actuelles
            self.risk_metrics.open_positions = len(self.positions)
            self.risk_metrics.winning_positions = sum(
                1 for pos in self.positions.values() if pos.is_profitable
            )
            self.risk_metrics.losing_positions = (
                    self.risk_metrics.open_positions - self.risk_metrics.winning_positions
            )

            # Calculer le score de risque global
            self._calculate_risk_score()

            # Vérifier les alertes
            self._check_risk_alerts()

        except Exception as e:
            logger.error(f"Erreur mise à jour métriques: {e}")

    def _calculate_risk_score(self):
        """Calcule un score de risque global (0-100)"""
        try:
            risk_score = 0.0

            # Facteur drawdown (0-30 points)
            drawdown_score = min(30, self.risk_metrics.current_drawdown * 300)
            risk_score += drawdown_score

            # Facteur exposition (0-25 points)
            exposure_ratio = self.risk_metrics.total_exposure / self.current_balance
            exposure_score = min(25, exposure_ratio * 250)
            risk_score += exposure_score

            # Facteur pertes consécutives (0-20 points)
            consecutive_losses = self.risk_stats['consecutive_losses']
            loss_score = min(20, consecutive_losses * 4)
            risk_score += loss_score

            # Facteur nombre de positions (0-15 points)
            position_ratio = len(self.positions) / self.risk_limits.max_positions_count
            position_score = min(15, position_ratio * 15)
            risk_score += position_score

            # Facteur volatilité (0-10 points)
            # TODO: Intégrer la volatilité du marché
            volatility_score = 5  # Score neutre
            risk_score += volatility_score

            self.risk_metrics.risk_score = min(100, risk_score)

        except Exception as e:
            logger.error(f"Erreur calcul score de risque: {e}")
            self.risk_metrics.risk_score = 50  # Score neutre en cas d'erreur

    def _check_risk_alerts(self):
        """Vérifie et génère les alertes de risque"""
        try:
            current_time = datetime.now(timezone.utc)
            alerts_to_add = []

            # Alerte drawdown
            if self.risk_metrics.current_drawdown > self.risk_limits.max_drawdown_pct:
                alerts_to_add.append(
                    RiskAlert(
                        alert_id=f"drawdown_{int(current_time.timestamp())}",
                        risk_event=RiskEvent.DRAWDOWN_LIMIT,
                        severity=RiskLevel.CRITICAL,
                        message=f"Drawdown critique: {self.risk_metrics.current_drawdown:.1%}",
                        timestamp=current_time,
                        metric_value=self.risk_metrics.current_drawdown,
                        threshold_value=self.risk_limits.max_drawdown_pct,
                        recommended_action="Fermer toutes les positions et suspendre le trading"
                    )
                )

            # Alerte pertes consécutives
            if self.risk_stats['consecutive_losses'] >= self.risk_limits.max_losing_streak:
                alerts_to_add.append(
                    RiskAlert(
                        alert_id=f"losses_{int(current_time.timestamp())}",
                        risk_event=RiskEvent.LOSS_STREAK,
                        severity=RiskLevel.HIGH,
                        message=f"Série de pertes: {self.risk_stats['consecutive_losses']} trades",
                        timestamp=current_time,
                        metric_value=self.risk_stats['consecutive_losses'],
                        threshold_value=self.risk_limits.max_losing_streak,
                        recommended_action="Réduire la taille des positions ou suspendre temporairement"
                    )
                )

            # Alerte exposition
            exposure_ratio = self.risk_metrics.total_exposure / self.current_balance
            if exposure_ratio > self.risk_limits.max_total_exposure_pct:
                alerts_to_add.append(
                    RiskAlert(
                        alert_id=f"exposure_{int(current_time.timestamp())}",
                        risk_event=RiskEvent.POSITION_LIMIT,
                        severity=RiskLevel.MEDIUM,
                        message=f"Exposition excessive: {exposure_ratio:.1%}",
                        timestamp=current_time,
                        metric_value=exposure_ratio,
                        threshold_value=self.risk_limits.max_total_exposure_pct,
                        recommended_action="Réduire la taille des positions"
                    )
                )

            # Alerte arrêt d'urgence
            if self.risk_metrics.current_drawdown > self.risk_limits.emergency_stop_pct:
                self.emergency_stop_triggered = True
                self.trading_enabled = False

                alerts_to_add.append(
                    RiskAlert(
                        alert_id=f"emergency_{int(current_time.timestamp())}",
                        risk_event=RiskEvent.EMERGENCY_STOP,
                        severity=RiskLevel.CRITICAL,
                        message=f"ARRÊT D'URGENCE ACTIVÉ - Drawdown: {self.risk_metrics.current_drawdown:.1%}",
                        timestamp=current_time,
                        metric_value=self.risk_metrics.current_drawdown,
                        threshold_value=self.risk_limits.emergency_stop_pct,
                        recommended_action="Trading suspendu - Intervention manuelle requise"
                    )
                )

                self.risk_stats['emergency_stops'] += 1

            # Ajouter toutes les nouvelles alertes
            for alert in alerts_to_add:
                self._add_risk_alert(alert)

        except Exception as e:
            logger.error(f"Erreur vérification alertes: {e}")

    def _add_risk_alert(self, alert: RiskAlert):
        """Ajoute une alerte de risque"""
        try:
            # Vérifier si une alerte similaire existe déjà
            similar_exists = any(
                existing.risk_event == alert.risk_event and
                existing.symbol == alert.symbol and
                (datetime.now(timezone.utc) - existing.timestamp).total_seconds() < 300  # 5 minutes
                for existing in self.active_alerts
            )

            if not similar_exists:
                self.active_alerts.append(alert)
                self.alert_history.append(alert)
                self.risk_stats['alerts_generated'] += 1

                logger.warning(f"🚨 ALERTE RISQUE: {alert.message}")

                # Limiter le nombre d'alertes actives
                if len(self.active_alerts) > 10:
                    self.active_alerts = self.active_alerts[-10:]

        except Exception as e:
            logger.error(f"Erreur ajout alerte: {e}")

    def get_risk_summary(self) -> Dict[str, Any]:
        """Retourne un résumé complet des risques"""
        try:
            with self.risk_lock:
                summary = {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'trading_enabled': self.trading_enabled,
                    'emergency_stop': self.emergency_stop_triggered,
                    'risk_metrics': self.risk_metrics.to_dict(),
                    'positions_summary': {
                        'count': len(self.positions),
                        'total_exposure': self.risk_metrics.total_exposure,
                        'unrealized_pnl': sum(pos.unrealized_pnl for pos in self.positions.values()),
                        'symbols': list(set(pos.symbol for pos in self.positions.values()))
                    },
                    'active_alerts': [alert.to_dict() for alert in self.active_alerts],
                    'risk_stats': self.risk_stats.copy(),
                    'limits': {
                        'max_position_size_pct': self.risk_limits.max_position_size_pct,
                        'max_drawdown_pct': self.risk_limits.max_drawdown_pct,
                        'max_positions': self.risk_limits.max_positions_count,
                        'emergency_stop_pct': self.risk_limits.emergency_stop_pct
                    }
                }

                return summary

        except Exception as e:
            logger.error(f"Erreur génération résumé risques: {e}")
            return {'error': str(e)}

    def reset_emergency_stop(self) -> bool:
        """Remet à zéro l'arrêt d'urgence (utilisation manuelle uniquement)"""
        try:
            with self.risk_lock:
                if self.emergency_stop_triggered:
                    self.emergency_stop_triggered = False
                    self.trading_enabled = True

                    logger.warning("⚠️ Arrêt d'urgence réinitialisé manuellement")

                    # Enregistrer l'événement
                    self.risk_events.append({
                        'timestamp': datetime.now(timezone.utc),
                        'event_type': 'emergency_stop_reset',
                        'manual_override': True
                    })

                    return True

                return False

        except Exception as e:
            logger.error(f"Erreur reset arrêt d'urgence: {e}")
            return False


# Instance globale
risk_manager = RiskManager(initial_balance=config.trading.initial_capital)


# Fonctions utilitaires
def calculate_position_size(symbol: str, entry_price: float,
                            stop_loss: Optional[float] = None) -> float:
    """Fonction utilitaire pour calculer la taille de position"""
    return risk_manager.calculate_position_size(symbol, entry_price, stop_loss)


def validate_trade(symbol: str, position_type: str, size: float,
                   entry_price: float) -> Tuple[bool, List[str]]:
    """Fonction utilitaire pour valider un trade"""
    pos_type = PositionType.LONG if position_type.upper() == "BUY" else PositionType.SHORT
    return risk_manager.validate_trade(symbol, pos_type, size, entry_price)


def get_risk_summary() -> Dict[str, Any]:
    """Fonction utilitaire pour récupérer le résumé des risques"""
    return risk_manager.get_risk_summary()


if __name__ == "__main__":
    # Test du gestionnaire de risques
    print("🛡️ Test du gestionnaire de risques...")

    try:
        rm = RiskManager(initial_balance=1000.0)

        print(f"✅ Gestionnaire initialisé avec {rm.initial_balance} de capital")
        print(f"   Max position: {rm.risk_limits.max_position_size_pct:.1%}")
        print(f"   Max drawdown: {rm.risk_limits.max_drawdown_pct:.1%}")
        print(f"   Max positions: {rm.risk_limits.max_positions_count}")

        # Test calcul position size
        entry_price = 100.0
        stop_loss = 99.0
        position_size = rm.calculate_position_size("R_10", entry_price, stop_loss)
        print(f"\n💰 Position size calculée: {position_size:.6f}")

        # Test validation trade
        is_valid, issues = rm.validate_trade("R_10", PositionType.LONG, position_size, entry_price)
        print(f"\n✅ Validation trade: {'VALIDE' if is_valid else 'REJETÉ'}")
        if issues:
            print(f"   Issues: {', '.join(issues)}")

        # Test ajout position
        if is_valid:
            success = rm.add_position("test_001", "R_10", PositionType.LONG,
                                      position_size, entry_price, stop_loss)
            print(f"\n📈 Position ajoutée: {'Succès' if success else 'Échec'}")

            # Test mise à jour prix
            new_price = 101.0
            rm.update_position_price("test_001", new_price)

            # Résumé des risques
            summary = rm.get_risk_summary()
            print(f"\n📊 Résumé des risques:")
            print(f"   Score de risque: {summary['risk_metrics']['risk_score']:.1f}/100")
            print(f"   Exposition totale: {summary['risk_metrics']['total_exposure']:.2f}")
            print(f"   PnL non réalisé: {summary['positions_summary']['unrealized_pnl']:.2f}")
            print(f"   Alertes actives: {len(summary['active_alerts'])}")

        print("✅ Test du gestionnaire de risques réussi !")

    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        logger.error(f"Test du gestionnaire de risques échoué: {e}")