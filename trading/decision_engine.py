"""
Moteur de décision de trading pour le Trading Bot Volatility 10
Combine analyse technique, IA et gestion des risques pour prendre des décisions
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
from threading import Lock
import warnings

from config import config
from data import db_manager, TradingSignals
from analysis import generate_trading_signal, get_analysis_summary, perform_complete_analysis
from ai_model import predict_price, get_ai_insights, get_model_performance

# Configuration du logger
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class DecisionType(Enum):
    """Types de décisions de trading"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"
    REDUCE_POSITION = "REDUCE_POSITION"
    INCREASE_POSITION = "INCREASE_POSITION"


class DecisionConfidence(Enum):
    """Niveaux de confiance de décision"""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


class MarketRegime(Enum):
    """Régimes de marché détectés"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNCERTAIN = "uncertain"


@dataclass
class DecisionContext:
    """Contexte pour la prise de décision"""
    # Prix et marché
    current_price: float = 0.0
    symbol: str = "R_10"
    timeframe: str = "1m"
    timestamp: datetime = None

    # Analyse technique
    technical_signals: Dict[str, Any] = field(default_factory=dict)
    technical_score: float = 0.0

    # Intelligence artificielle
    ai_prediction: Optional[Any] = None
    ai_score: float = 0.0
    ai_confidence: float = 0.0

    # Conditions de marché
    market_regime: MarketRegime = MarketRegime.UNCERTAIN
    volatility_level: str = "NORMAL"
    trend_strength: float = 0.0

    # Portfolio et risque
    account_balance: float = 0.0
    open_positions: List[Dict] = field(default_factory=list)
    portfolio_exposure: float = 0.0
    current_drawdown: float = 0.0

    # Contraintes
    max_position_size: float = 0.02
    risk_budget_available: float = 1.0
    trading_session_active: bool = True

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


@dataclass
class TradingDecision:
    """Décision de trading complète"""
    # Décision principale
    decision_type: DecisionType
    confidence: DecisionConfidence
    overall_score: float

    # Paramètres d'exécution
    symbol: str
    position_size: float
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    # Justification
    reasoning: str = ""
    contributing_factors: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)

    # Métadonnées
    timestamp: datetime = None
    decision_id: str = ""
    expiry_time: Optional[datetime] = None

    # Scores détaillés
    technical_contribution: float = 0.0
    ai_contribution: float = 0.0
    risk_adjustment: float = 0.0
    market_condition_adjustment: float = 0.0

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        if not self.decision_id:
            self.decision_id = f"decision_{int(self.timestamp.timestamp())}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'decision_type': self.decision_type.value,
            'confidence': self.confidence.value,
            'overall_score': self.overall_score,
            'symbol': self.symbol,
            'position_size': self.position_size,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'reasoning': self.reasoning,
            'contributing_factors': self.contributing_factors,
            'risk_factors': self.risk_factors,
            'timestamp': self.timestamp.isoformat(),
            'decision_id': self.decision_id,
            'expiry_time': self.expiry_time.isoformat() if self.expiry_time else None,
            'technical_contribution': self.technical_contribution,
            'ai_contribution': self.ai_contribution,
            'risk_adjustment': self.risk_adjustment,
            'market_condition_adjustment': self.market_condition_adjustment
        }


class DecisionEngine:
    """Moteur de décision de trading utilisant analyse technique et IA"""

    def __init__(self):
        # Configuration
        self.config = {
            'technical_weight': 0.35,  # Poids analyse technique
            'ai_weight': 0.40,  # Poids IA
            'risk_weight': 0.25,  # Poids gestion des risques
            'min_decision_score': 0.65,  # Score minimum pour agir
            'confluence_bonus': 0.1,  # Bonus si technical + IA concordent
            'decision_expiry_minutes': 10,  # Durée de validité d'une décision
            'max_decisions_per_hour': 12,  # Limite de décisions par heure
            'cooldown_after_decision_minutes': 5  # Temps d'attente entre décisions
        }

        # État du moteur
        self.last_decision = None
        self.decision_history = []
        self.market_regime_cache = {}
        self.decision_lock = Lock()

        # Statistiques
        self.decision_stats = {
            'total_decisions': 0,
            'buy_decisions': 0,
            'sell_decisions': 0,
            'hold_decisions': 0,
            'successful_decisions': 0,
            'avg_decision_score': 0.0,
            'avg_processing_time_ms': 0.0,
            'last_decision_time': None,
            'confluence_decisions': 0
        }

        # Cache des analyses
        self.analysis_cache = {}
        self.cache_timeout_seconds = 30

        logger.info("Moteur de décision de trading initialisé")

    def make_decision(self, symbol: str = "R_10", timeframe: str = "1m",
                      context_override: Optional[DecisionContext] = None) -> Optional[TradingDecision]:
        """
        Prend une décision de trading basée sur tous les signaux disponibles

        Args:
            symbol: Symbole à trader
            timeframe: Timeframe d'analyse
            context_override: Contexte spécifique (optionnel)

        Returns:
            Décision de trading ou None si aucune décision
        """
        start_time = time.time()

        try:
            with self.decision_lock:
                logger.debug(f"🤔 Analyse de décision pour {symbol}")

                # Vérifier les contraintes temporelles
                if not self._can_make_decision():
                    logger.debug("Décision bloquée par les contraintes temporelles")
                    return None

                # Construire le contexte de décision
                context = context_override or self._build_decision_context(symbol, timeframe)

                if context.current_price <= 0:
                    logger.warning("Prix actuel invalide - décision impossible")
                    return None

                # Collecter tous les signaux
                signals = self._collect_all_signals(context)

                # Calculer les scores pondérés
                decision_scores = self._calculate_decision_scores(signals, context)

                # Déterminer la décision finale
                decision = self._determine_final_decision(decision_scores, context, signals)

                if decision is None:
                    logger.debug("Aucune décision suffisamment forte")
                    return None

                # Valider la décision
                validated_decision = self._validate_decision(decision, context)

                if validated_decision is None:
                    logger.debug("Décision rejetée par la validation")
                    return None

                # Enregistrer et mettre à jour les statistiques
                self._record_decision(validated_decision)

                # Temps de traitement
                processing_time = (time.time() - start_time) * 1000
                self._update_decision_stats(validated_decision, processing_time)

                logger.info(f"💡 Décision: {validated_decision.decision_type.value} "
                            f"(score: {validated_decision.overall_score:.3f}, "
                            f"confiance: {validated_decision.confidence.value})")

                return validated_decision

        except Exception as e:
            logger.error(f"Erreur lors de la prise de décision: {e}")
            return None

    def _can_make_decision(self) -> bool:
        """Vérifie si une décision peut être prise (contraintes temporelles)"""
        try:
            current_time = datetime.now(timezone.utc)

            # Vérifier le cooldown après la dernière décision
            if self.last_decision:
                time_since_last = (current_time - self.last_decision.timestamp).total_seconds()
                cooldown_seconds = self.config['cooldown_after_decision_minutes'] * 60

                if time_since_last < cooldown_seconds:
                    return False

            # Vérifier la limite de décisions par heure
            one_hour_ago = current_time - timedelta(hours=1)
            recent_decisions = [
                d for d in self.decision_history
                if d.timestamp > one_hour_ago
            ]

            if len(recent_decisions) >= self.config['max_decisions_per_hour']:
                return False

            return True

        except Exception as e:
            logger.error(f"Erreur vérification contraintes: {e}")
            return False

    def _build_decision_context(self, symbol: str, timeframe: str) -> DecisionContext:
        """Construit le contexte de décision"""
        try:
            context = DecisionContext(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime.now(timezone.utc)
            )

            # Prix actuel
            latest_price = db_manager.get_latest_price(symbol, timeframe)
            if latest_price:
                context.current_price = latest_price.close_price

            # Régime de marché
            context.market_regime = self._detect_market_regime(symbol, timeframe)

            # TODO: Intégrer avec le gestionnaire de portfolio
            context.account_balance = 1000.0  # Valeur par défaut
            context.max_position_size = config.trading.max_capital_per_trade_pct

            return context

        except Exception as e:
            logger.error(f"Erreur construction du contexte: {e}")
            return DecisionContext(symbol=symbol, timeframe=timeframe)

    def _detect_market_regime(self, symbol: str, timeframe: str) -> MarketRegime:
        """Détecte le régime de marché actuel"""
        try:
            # Vérifier le cache
            cache_key = f"{symbol}_{timeframe}"
            current_time = datetime.now(timezone.utc)

            if cache_key in self.market_regime_cache:
                cached_regime, cache_time = self.market_regime_cache[cache_key]
                if (current_time - cache_time).total_seconds() < 300:  # Cache 5 minutes
                    return cached_regime

            # Récupérer les données récentes
            end_time = current_time
            start_time = end_time - timedelta(hours=2)

            price_data = db_manager.get_price_data(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                limit=120
            )

            if not price_data or len(price_data) < 50:
                regime = MarketRegime.UNCERTAIN
            else:
                # Convertir en DataFrame
                df = pd.DataFrame([data.to_dict() for data in price_data])
                closes = df['close']

                # Analyser la tendance
                sma_20 = closes.rolling(20).mean()
                sma_50 = closes.rolling(50).mean()

                current_price = closes.iloc[-1]
                sma_20_current = sma_20.iloc[-1]
                sma_50_current = sma_50.iloc[-1]

                # Analyser la volatilité
                returns = closes.pct_change().dropna()
                volatility = returns.std()

                # Déterminer le régime
                if volatility > 0.015:  # Haute volatilité (1.5%)
                    regime = MarketRegime.HIGH_VOLATILITY
                elif volatility < 0.005:  # Faible volatilité (0.5%)
                    regime = MarketRegime.LOW_VOLATILITY
                elif current_price > sma_20_current > sma_50_current:
                    # Calcul de la force de la tendance
                    trend_strength = (sma_20_current - sma_50_current) / sma_50_current
                    if trend_strength > 0.005:  # Tendance haussière forte
                        regime = MarketRegime.TRENDING_UP
                    else:
                        regime = MarketRegime.SIDEWAYS
                elif current_price < sma_20_current < sma_50_current:
                    # Tendance baissière
                    trend_strength = (sma_50_current - sma_20_current) / sma_50_current
                    if trend_strength > 0.005:  # Tendance baissière forte
                        regime = MarketRegime.TRENDING_DOWN
                    else:
                        regime = MarketRegime.SIDEWAYS
                else:
                    regime = MarketRegime.SIDEWAYS

            # Mettre en cache
            self.market_regime_cache[cache_key] = (regime, current_time)

            return regime

        except Exception as e:
            logger.error(f"Erreur détection régime de marché: {e}")
            return MarketRegime.UNCERTAIN

    def _collect_all_signals(self, context: DecisionContext) -> Dict[str, Any]:
        """Collecte tous les signaux disponibles"""
        try:
            signals = {
                'technical': None,
                'ai_prediction': None,
                'ai_insights': None,
                'analysis_summary': None
            }

            # Signaux d'analyse technique
            try:
                technical_signal = generate_trading_signal(context.symbol, context.timeframe)
                signals['technical'] = technical_signal

                if technical_signal:
                    context.technical_score = technical_signal.combined_score
                    context.technical_signals = {
                        'signal_type': technical_signal.signal_type.value,
                        'confidence': technical_signal.confidence,
                        'strength': technical_signal.strength.value
                    }
            except Exception as e:
                logger.warning(f"Erreur récupération signal technique: {e}")

            # Prédictions IA
            try:
                ai_prediction = predict_price(context.symbol, context.timeframe)
                signals['ai_prediction'] = ai_prediction

                if ai_prediction:
                    context.ai_confidence = ai_prediction.confidence
                    context.ai_score = ai_prediction.model_confidence
            except Exception as e:
                logger.warning(f"Erreur récupération prédiction IA: {e}")

            # Insights IA
            try:
                ai_insights = get_ai_insights(context.symbol, context.timeframe)
                signals['ai_insights'] = ai_insights

                if ai_insights:
                    context.volatility_level = ai_insights.volatility_forecast
                    context.trend_strength = ai_insights.trend_strength
            except Exception as e:
                logger.warning(f"Erreur récupération insights IA: {e}")

            # Résumé d'analyse
            try:
                analysis_summary = get_analysis_summary(context.symbol, context.timeframe)
                signals['analysis_summary'] = analysis_summary
            except Exception as e:
                logger.warning(f"Erreur récupération résumé d'analyse: {e}")

            return signals

        except Exception as e:
            logger.error(f"Erreur collecte des signaux: {e}")
            return {}

    def _calculate_decision_scores(self, signals: Dict[str, Any], context: DecisionContext) -> Dict[str, float]:
        """Calcule les scores pondérés pour chaque type de décision"""
        try:
            scores = {
                'buy_score': 0.0,
                'sell_score': 0.0,
                'hold_score': 0.0
            }

            # Score technique
            technical_signal = signals.get('technical')
            technical_buy = 0.0
            technical_sell = 0.0

            if technical_signal:
                if technical_signal.signal_type.value == 'BUY':
                    technical_buy = technical_signal.confidence * technical_signal.combined_score
                elif technical_signal.signal_type.value == 'SELL':
                    technical_sell = technical_signal.confidence * technical_signal.combined_score

            # Score IA
            ai_prediction = signals.get('ai_prediction')
            ai_buy = 0.0
            ai_sell = 0.0

            if ai_prediction:
                if ai_prediction.predicted_direction == 'UP':
                    ai_buy = ai_prediction.confidence * ai_prediction.model_confidence
                elif ai_prediction.predicted_direction == 'DOWN':
                    ai_sell = ai_prediction.confidence * ai_prediction.model_confidence

            # Ajustements selon le régime de marché
            market_adjustment = self._get_market_regime_adjustment(context.market_regime)

            # Scores pondérés
            scores['buy_score'] = (
                                          technical_buy * self.config['technical_weight'] +
                                          ai_buy * self.config['ai_weight']
                                  ) * market_adjustment['buy_multiplier']

            scores['sell_score'] = (
                                           technical_sell * self.config['technical_weight'] +
                                           ai_sell * self.config['ai_weight']
                                   ) * market_adjustment['sell_multiplier']

            # Score HOLD basé sur l'incertitude
            uncertainty = abs(scores['buy_score'] - scores['sell_score'])
            scores['hold_score'] = 1.0 - uncertainty

            # Bonus de confluence si technique et IA sont d'accord
            if ((technical_buy > 0.5 and ai_buy > 0.5) or
                    (technical_sell > 0.5 and ai_sell > 0.5)):
                confluence_bonus = self.config['confluence_bonus']
                scores['buy_score'] += confluence_bonus if technical_buy > 0.5 else 0
                scores['sell_score'] += confluence_bonus if technical_sell > 0.5 else 0

                # Compter les décisions de confluence
                self.decision_stats['confluence_decisions'] += 1

            return scores

        except Exception as e:
            logger.error(f"Erreur calcul des scores: {e}")
            return {'buy_score': 0.0, 'sell_score': 0.0, 'hold_score': 1.0}

    def _get_market_regime_adjustment(self, regime: MarketRegime) -> Dict[str, float]:
        """Retourne les ajustements selon le régime de marché"""
        adjustments = {
            MarketRegime.TRENDING_UP: {
                'buy_multiplier': 1.2,
                'sell_multiplier': 0.8,
                'hold_multiplier': 0.9
            },
            MarketRegime.TRENDING_DOWN: {
                'buy_multiplier': 0.8,
                'sell_multiplier': 1.2,
                'hold_multiplier': 0.9
            },
            MarketRegime.SIDEWAYS: {
                'buy_multiplier': 0.9,
                'sell_multiplier': 0.9,
                'hold_multiplier': 1.1
            },
            MarketRegime.HIGH_VOLATILITY: {
                'buy_multiplier': 0.7,
                'sell_multiplier': 0.7,
                'hold_multiplier': 1.3
            },
            MarketRegime.LOW_VOLATILITY: {
                'buy_multiplier': 1.1,
                'sell_multiplier': 1.1,
                'hold_multiplier': 0.9
            },
            MarketRegime.UNCERTAIN: {
                'buy_multiplier': 0.5,
                'sell_multiplier': 0.5,
                'hold_multiplier': 1.5
            }
        }

        return adjustments.get(regime, {
            'buy_multiplier': 1.0,
            'sell_multiplier': 1.0,
            'hold_multiplier': 1.0
        })

    def _determine_final_decision(self, scores: Dict[str, float],
                                  context: DecisionContext,
                                  signals: Dict[str, Any]) -> Optional[TradingDecision]:
        """Détermine la décision finale basée sur les scores"""
        try:
            # Trouver le score maximum
            max_score = max(scores.values())
            decision_type = None

            if scores['buy_score'] == max_score and max_score > self.config['min_decision_score']:
                decision_type = DecisionType.BUY
            elif scores['sell_score'] == max_score and max_score > self.config['min_decision_score']:
                decision_type = DecisionType.SELL
            else:
                decision_type = DecisionType.HOLD

            # Si le score est trop faible, maintenir HOLD
            if max_score < self.config['min_decision_score'] and decision_type != DecisionType.HOLD:
                decision_type = DecisionType.HOLD
                max_score = scores['hold_score']

            # Déterminer la confiance
            confidence = self._calculate_confidence(max_score, scores)

            # Calculer la taille de position
            position_size = self._calculate_position_size(decision_type, max_score, context)

            # Construire la décision
            decision = TradingDecision(
                decision_type=decision_type,
                confidence=confidence,
                overall_score=max_score,
                symbol=context.symbol,
                position_size=position_size,
                entry_price=context.current_price,
                timestamp=context.timestamp
            )

            # Calculer stop loss et take profit
            if decision_type in [DecisionType.BUY, DecisionType.SELL]:
                stop_loss, take_profit = self._calculate_exit_levels(
                    decision_type, context.current_price, signals
                )
                decision.stop_loss = stop_loss
                decision.take_profit = take_profit

            # Ajouter la justification
            decision.reasoning = self._generate_decision_reasoning(
                decision_type, scores, signals, context
            )

            # Scores détaillés
            decision.technical_contribution = scores.get('technical_contribution', 0.0)
            decision.ai_contribution = scores.get('ai_contribution', 0.0)
            decision.market_condition_adjustment = self._get_market_regime_adjustment(context.market_regime)[
                'buy_multiplier']

            # Expiration
            decision.expiry_time = context.timestamp + timedelta(
                minutes=self.config['decision_expiry_minutes']
            )

            return decision

        except Exception as e:
            logger.error(f"Erreur détermination décision finale: {e}")
            return None

    def _calculate_confidence(self, max_score: float, scores: Dict[str, float]) -> DecisionConfidence:
        """Calcule le niveau de confiance de la décision"""
        try:
            # Normaliser le score (0-1)
            normalized_score = min(1.0, max_score)

            # Calculer l'écart entre le meilleur et le deuxième score
            sorted_scores = sorted(scores.values(), reverse=True)
            score_gap = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0

            # Score de confiance combiné
            confidence_score = (normalized_score * 0.7) + (score_gap * 0.3)

            if confidence_score >= 0.9:
                return DecisionConfidence.VERY_HIGH
            elif confidence_score >= 0.8:
                return DecisionConfidence.HIGH
            elif confidence_score >= 0.6:
                return DecisionConfidence.MEDIUM
            elif confidence_score >= 0.4:
                return DecisionConfidence.LOW
            else:
                return DecisionConfidence.VERY_LOW

        except Exception as e:
            logger.error(f"Erreur calcul confiance: {e}")
            return DecisionConfidence.LOW

    def _calculate_position_size(self, decision_type: DecisionType,
                                 score: float, context: DecisionContext) -> float:
        """Calcule la taille de position optimale"""
        try:
            if decision_type == DecisionType.HOLD:
                return 0.0

            # Taille de base
            base_size = context.max_position_size

            # Ajustement selon le score
            score_multiplier = min(1.0, score)

            # Ajustement selon la volatilité
            volatility_multiplier = 1.0
            if context.volatility_level == "HIGH":
                volatility_multiplier = 0.5
            elif context.volatility_level == "LOW":
                volatility_multiplier = 1.2

            # Ajustement selon le régime de marché
            regime_adjustment = self._get_market_regime_adjustment(context.market_regime)
            if decision_type == DecisionType.BUY:
                regime_multiplier = regime_adjustment['buy_multiplier']
            else:
                regime_multiplier = regime_adjustment['sell_multiplier']

            # Taille finale
            position_size = (base_size * score_multiplier *
                             volatility_multiplier * regime_multiplier)

            # Limites
            min_size = base_size * 0.25  # Au minimum 25% de la taille max
            max_size = base_size  # Au maximum la taille max configurée

            return max(min_size, min(max_size, position_size))

        except Exception as e:
            logger.error(f"Erreur calcul taille position: {e}")
            return context.max_position_size * 0.5

    def _calculate_exit_levels(self, decision_type: DecisionType, entry_price: float,
                               signals: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
        """Calcule les niveaux de stop loss et take profit"""
        try:
            if decision_type == DecisionType.HOLD:
                return None, None

            # Utiliser les niveaux du signal technique si disponibles
            technical_signal = signals.get('technical')
            if technical_signal and technical_signal.stop_loss and technical_signal.take_profit:
                return technical_signal.stop_loss, technical_signal.take_profit

            # Calcul basé sur l'ATR ou volatilité
            volatility_pct = 0.001  # 0.1% par défaut pour Volatility 10

            # Ajuster selon les prédictions IA
            ai_prediction = signals.get('ai_prediction')
            if ai_prediction and ai_prediction.price_change_pct:
                predicted_change = abs(ai_prediction.price_change_pct) / 100
                volatility_pct = max(volatility_pct, predicted_change * 0.5)

            # Calculer stop loss et take profit
            if decision_type == DecisionType.BUY:
                stop_loss = entry_price * (1 - volatility_pct * 15)  # 15x volatilité pour SL
                take_profit = entry_price * (1 + volatility_pct * 20)  # 20x volatilité pour TP
            else:  # SELL
                stop_loss = entry_price * (1 + volatility_pct * 15)
                take_profit = entry_price * (1 - volatility_pct * 20)

            return stop_loss, take_profit

        except Exception as e:
            logger.error(f"Erreur calcul niveaux de sortie: {e}")
            return None, None

    def _generate_decision_reasoning(self, decision_type: DecisionType,
                                     scores: Dict[str, float],
                                     signals: Dict[str, Any],
                                     context: DecisionContext) -> str:
        """Génère l'explication de la décision"""
        try:
            reasoning_parts = []

            # Décision principale
            reasoning_parts.append(f"Décision: {decision_type.value}")

            # Scores
            max_score = max(scores.values())
            reasoning_parts.append(f"Score: {max_score:.3f}")

            # Signaux techniques
            technical_signal = signals.get('technical')
            if technical_signal:
                reasoning_parts.append(
                    f"Technique: {technical_signal.signal_type.value} "
                    f"(conf: {technical_signal.confidence:.2f})"
                )

            # IA
            ai_prediction = signals.get('ai_prediction')
            if ai_prediction:
                reasoning_parts.append(
                    f"IA: {ai_prediction.predicted_direction} "
                    f"(conf: {ai_prediction.confidence:.2f})"
                )

            # Régime de marché
            reasoning_parts.append(f"Marché: {context.market_regime.value}")

            # Volatilité
            reasoning_parts.append(f"Volatilité: {context.volatility_level}")

            return " | ".join(reasoning_parts)

        except Exception as e:
            logger.error(f"Erreur génération reasoning: {e}")
            return f"Décision {decision_type.value}"

    def _validate_decision(self, decision: TradingDecision,
                           context: DecisionContext) -> Optional[TradingDecision]:
        """Valide la décision avant exécution"""
        try:
            validation_issues = []

            # Vérifier la taille de position
            if decision.position_size <= 0 and decision.decision_type != DecisionType.HOLD:
                validation_issues.append("Taille de position invalide")

            # Vérifier les niveaux de prix
            if decision.decision_type == DecisionType.BUY:
                if decision.stop_loss and decision.stop_loss >= decision.entry_price:
                    validation_issues.append("Stop loss BUY invalide")
                if decision.take_profit and decision.take_profit <= decision.entry_price:
                    validation_issues.append("Take profit BUY invalide")
            elif decision.decision_type == DecisionType.SELL:
                if decision.stop_loss and decision.stop_loss <= decision.entry_price:
                    validation_issues.append("Stop loss SELL invalide")
                if decision.take_profit and decision.take_profit >= decision.entry_price:
                    validation_issues.append("Take profit SELL invalide")

            # Vérifier la confiance minimale
            if (decision.confidence in [DecisionConfidence.VERY_LOW, DecisionConfidence.LOW] and
                    decision.decision_type != DecisionType.HOLD):
                validation_issues.append("Confiance trop faible")

            # Vérifier les conditions de marché
            if context.market_regime == MarketRegime.HIGH_VOLATILITY:
                if decision.confidence not in [DecisionConfidence.VERY_HIGH, DecisionConfidence.HIGH]:
                    validation_issues.append("Confiance insuffisante en haute volatilité")

            if validation_issues:
                logger.warning(f"Décision rejetée: {', '.join(validation_issues)}")
                # Convertir en HOLD si problèmes
                decision.decision_type = DecisionType.HOLD
                decision.position_size = 0.0
                decision.risk_factors = validation_issues

            return decision

        except Exception as e:
            logger.error(f"Erreur validation décision: {e}")
            return None

    def _record_decision(self, decision: TradingDecision):
        """Enregistre la décision dans l'historique"""
        try:
            self.last_decision = decision
            self.decision_history.append(decision)

            # Limiter la taille de l'historique
            if len(self.decision_history) > 1000:
                self.decision_history = self.decision_history[-500:]

            logger.debug(f"Décision enregistrée: {decision.decision_id}")

        except Exception as e:
            logger.error(f"Erreur enregistrement décision: {e}")

    def _update_decision_stats(self, decision: TradingDecision, processing_time: float):
        """Met à jour les statistiques de décision"""
        try:
            self.decision_stats['total_decisions'] += 1
            self.decision_stats['last_decision_time'] = decision.timestamp

            # Compter par type
            if decision.decision_type == DecisionType.BUY:
                self.decision_stats['buy_decisions'] += 1
            elif decision.decision_type == DecisionType.SELL:
                self.decision_stats['sell_decisions'] += 1
            else:
                self.decision_stats['hold_decisions'] += 1

            # Score moyen
            total = self.decision_stats['total_decisions']
            old_avg = self.decision_stats['avg_decision_score']
            self.decision_stats['avg_decision_score'] = (
                    (old_avg * (total - 1) + decision.overall_score) / total
            )

            # Temps de traitement moyen
            old_avg_time = self.decision_stats['avg_processing_time_ms']
            self.decision_stats['avg_processing_time_ms'] = (
                    (old_avg_time * (total - 1) + processing_time) / total
            )

        except Exception as e:
            logger.error(f"Erreur mise à jour statistiques: {e}")

    def get_decision_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de décision"""
        stats = self.decision_stats.copy()

        # Calculer des métriques supplémentaires
        total = stats['total_decisions']
        if total > 0:
            stats['buy_percentage'] = (stats['buy_decisions'] / total) * 100
            stats['sell_percentage'] = (stats['sell_decisions'] / total) * 100
            stats['hold_percentage'] = (stats['hold_decisions'] / total) * 100
            stats['confluence_percentage'] = (stats['confluence_decisions'] / total) * 100

        # Informations sur la dernière décision
        if self.last_decision:
            stats['last_decision'] = self.last_decision.to_dict()

        # Historique récent
        stats['decisions_last_hour'] = len([
            d for d in self.decision_history
            if (datetime.now(timezone.utc) - d.timestamp).total_seconds() < 3600
        ])

        return stats

    def get_recent_decisions(self, hours: int = 24) -> List[TradingDecision]:
        """Retourne les décisions récentes"""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            return [
                d for d in self.decision_history
                if d.timestamp > cutoff_time
            ]
        except Exception as e:
            logger.error(f"Erreur récupération décisions récentes: {e}")
            return []


# Instance globale
decision_engine = DecisionEngine()


# Fonctions utilitaires
def make_trading_decision(symbol: str = "R_10", timeframe: str = "1m") -> Optional[TradingDecision]:
    """Fonction utilitaire pour prendre une décision de trading"""
    return decision_engine.make_decision(symbol, timeframe)


def get_decision_stats() -> Dict[str, Any]:
    """Fonction utilitaire pour récupérer les statistiques de décision"""
    return decision_engine.get_decision_stats()


if __name__ == "__main__":
    # Test du moteur de décision
    print("🧠 Test du moteur de décision...")

    try:
        engine = DecisionEngine()

        print(f"✅ Moteur de décision configuré")
        print(f"   Poids technique: {engine.config['technical_weight']}")
        print(f"   Poids IA: {engine.config['ai_weight']}")
        print(f"   Score minimum: {engine.config['min_decision_score']}")

        # Test de détection de régime
        regime = engine._detect_market_regime("R_10", "1m")
        print(f"\n📊 Régime de marché détecté: {regime.value}")

        # Test de décision (si données disponibles)
        try:
            decision = engine.make_decision("R_10", "1m")
            if decision:
                print(f"\n💡 Décision de test:")
                print(f"   Type: {decision.decision_type.value}")
                print(f"   Confiance: {decision.confidence.value}")
                print(f"   Score: {decision.overall_score:.3f}")
                print(f"   Taille position: {decision.position_size:.4f}")
                print(f"   Reasoning: {decision.reasoning}")
            else:
                print(f"\n❌ Aucune décision prise")
        except Exception as e:
            print(f"⚠️ Test de décision échoué (normal si pas de données): {e}")

        # Statistiques
        stats = engine.get_decision_stats()
        print(f"\n📈 Statistiques:")
        for key, value in list(stats.items())[:8]:
            print(f"   {key}: {value}")

        print("✅ Test du moteur de décision réussi !")

    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        logger.error(f"Test du moteur de décision échoué: {e}")