"""
Générateur de signaux de trading pour le Trading Bot Volatility 10
Combine indicateurs techniques et patterns pour créer des signaux cohérents
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
import uuid

from config import config
from data import db_manager, TradingSignals
from .technical_indicators import technical_indicators, IndicatorResult, get_indicator_summary
from .pattern_recognition import pattern_recognizer, PatternResult, PatternSignal, get_pattern_summary

# Configuration du logger
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class SignalType(Enum):
    """Types de signaux de trading"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class SignalStrength(Enum):
    """Force du signal"""
    VERY_STRONG = "very_strong"
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NEUTRAL = "neutral"


class SignalSource(Enum):
    """Source du signal"""
    TECHNICAL_INDICATORS = "technical_indicators"
    PATTERN_RECOGNITION = "pattern_recognition"
    COMBINED_ANALYSIS = "combined_analysis"
    AI_PREDICTION = "ai_prediction"
    RISK_MANAGEMENT = "risk_management"


@dataclass
class MarketConditions:
    """Conditions actuelles du marché"""
    volatility_regime: str  # 'low', 'normal', 'high'
    trend_direction: str  # 'up', 'down', 'sideways'
    trend_strength: float  # 0.0 à 1.0
    market_session: str  # 'asian', 'european', 'american', 'overlap'
    volume_profile: str  # 'low', 'normal', 'high'
    support_resistance_proximity: float  # Distance aux niveaux S/R

    def to_dict(self) -> Dict[str, Any]:
        return {
            'volatility_regime': self.volatility_regime,
            'trend_direction': self.trend_direction,
            'trend_strength': self.trend_strength,
            'market_session': self.market_session,
            'volume_profile': self.volume_profile,
            'support_resistance_proximity': self.support_resistance_proximity
        }


@dataclass
class TradingSignal:
    """Signal de trading complet"""
    signal_id: str
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    strength: SignalStrength
    confidence: float

    # Prix et niveaux
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward_ratio: Optional[float] = None

    # Sources et scores
    technical_score: float = 0.0
    pattern_score: float = 0.0
    ai_score: float = 0.0
    combined_score: float = 0.0

    # Contexte
    market_conditions: Optional[MarketConditions] = None
    indicators_used: List[str] = field(default_factory=list)
    patterns_detected: List[str] = field(default_factory=list)
    reasoning: str = ""

    # Métadonnées
    expires_at: Optional[datetime] = None
    position_size_suggestion: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'signal_id': self.signal_id,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'strength': self.strength.value,
            'confidence': self.confidence,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'risk_reward_ratio': self.risk_reward_ratio,
            'technical_score': self.technical_score,
            'pattern_score': self.pattern_score,
            'ai_score': self.ai_score,
            'combined_score': self.combined_score,
            'market_conditions': self.market_conditions.to_dict() if self.market_conditions else None,
            'indicators_used': self.indicators_used,
            'patterns_detected': self.patterns_detected,
            'reasoning': self.reasoning,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'position_size_suggestion': self.position_size_suggestion
        }


class SignalGenerator:
    """Générateur de signaux de trading combinant analyse technique et patterns"""

    def __init__(self):
        # Configuration des seuils
        self.config = {
            'min_confidence_threshold': config.ai_model.min_confidence_threshold,
            'technical_weight': 0.4,  # Poids des indicateurs techniques
            'pattern_weight': 0.3,  # Poids des patterns
            'ai_weight': 0.3,  # Poids de l'IA (quand disponible)
            'confluence_threshold': 0.65,  # Seuil pour considérer une confluence
            'signal_expiry_minutes': 15,  # Durée de validité d'un signal
            'max_signals_per_hour': 10,  # Limite de signaux par heure
            'risk_reward_min': 1.5,  # Ratio R/R minimum requis
        }

        # Cache et historique
        self.signal_cache = {}
        self.signal_history = []
        self.recent_signals = []  # Signaux de la dernière heure

        # Statistiques
        self.generation_stats = {
            'signals_generated': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'hold_signals': 0,
            'avg_confidence': 0.0,
            'confluence_signals': 0,
            'last_signal_time': None
        }

        logger.info("Générateur de signaux initialisé")

    def generate_signal(self, symbol: str = "R_10", timeframe: str = "1m") -> Optional[TradingSignal]:
        """
        Génère un signal de trading complet pour un symbole

        Args:
            symbol: Symbole à analyser
            timeframe: Timeframe des données

        Returns:
            Signal de trading ou None si aucun signal valide
        """
        try:
            current_time = datetime.now(timezone.utc)

            # Vérifier la limite de signaux par heure
            if self._is_signal_limit_reached():
                logger.debug("Limite de signaux par heure atteinte")
                return None

            # Récupérer les prix actuels
            latest_market_data = self._get_latest_market_data(symbol, timeframe)
            if not latest_market_data:
                logger.warning(f"Aucune donnée de marché disponible pour {symbol}")
                return None

            current_price = latest_market_data.get('close', latest_market_data.get('mid_price', 0))
            if current_price <= 0:
                logger.warning("Prix actuel invalide")
                return None

            # Analyser les conditions de marché
            market_conditions = self._analyze_market_conditions(symbol, timeframe)

            # Récupérer les indicateurs techniques
            indicators = technical_indicators.get_indicators_for_symbol(symbol, timeframe)
            technical_summary = get_indicator_summary(indicators)

            # Récupérer les patterns
            patterns = pattern_recognizer.get_patterns_for_symbol(symbol, timeframe)
            pattern_summary = get_pattern_summary(patterns)

            # Calculer les scores individuels
            technical_score = self._calculate_technical_score(technical_summary, indicators)
            pattern_score = self._calculate_pattern_score(pattern_summary, patterns)
            ai_score = 0.0  # TODO: Intégrer les prédictions IA

            # Score combiné pondéré
            combined_score = (
                    technical_score * self.config['technical_weight'] +
                    pattern_score * self.config['pattern_weight'] +
                    ai_score * self.config['ai_weight']
            )

            # Déterminer le type et la force du signal
            signal_type, strength, confidence = self._determine_signal(
                combined_score, technical_summary, pattern_summary, market_conditions
            )

            # Vérifier si le signal est assez fort
            if confidence < self.config['min_confidence_threshold']:
                logger.debug(
                    f"Signal trop faible: confiance {confidence:.2f} < {self.config['min_confidence_threshold']:.2f}")
                return None

            # Calculer les niveaux de prix
            entry_price = current_price
            stop_loss, take_profit = self._calculate_price_levels(
                signal_type, entry_price, market_conditions, indicators, patterns
            )

            # Calculer le ratio R/R
            risk_reward_ratio = None
            if stop_loss and take_profit:
                risk = abs(entry_price - stop_loss)
                reward = abs(take_profit - entry_price)
                risk_reward_ratio = reward / risk if risk > 0 else 0

                # Rejeter si R/R insuffisant
                if risk_reward_ratio < self.config['risk_reward_min']:
                    logger.debug(f"Ratio R/R insuffisant: {risk_reward_ratio:.2f} < {self.config['risk_reward_min']}")
                    return None

            # Créer le signal
            signal = TradingSignal(
                signal_id=str(uuid.uuid4()),
                timestamp=current_time,
                symbol=symbol,
                signal_type=signal_type,
                strength=strength,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=risk_reward_ratio,
                technical_score=technical_score,
                pattern_score=pattern_score,
                ai_score=ai_score,
                combined_score=combined_score,
                market_conditions=market_conditions,
                indicators_used=list(indicators.keys()),
                patterns_detected=[p.pattern_type.value for p in patterns],
                reasoning=self._generate_reasoning(
                    signal_type, technical_summary, pattern_summary, market_conditions
                ),
                expires_at=current_time + timedelta(minutes=self.config['signal_expiry_minutes']),
                position_size_suggestion=self._suggest_position_size(confidence, market_conditions)
            )

            # Sauvegarder et mettre en cache
            self._save_signal(signal)
            self._update_statistics(signal)

            logger.info(f"Signal généré: {signal_type.value} pour {symbol} "
                        f"(confiance: {confidence:.2f}, R/R: {risk_reward_ratio:.2f if risk_reward_ratio else 'N/A'})")

            return signal

        except Exception as e:
            logger.error(f"Erreur lors de la génération du signal: {e}")
            return None

    def _get_latest_market_data(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """Récupère les dernières données de marché"""
        try:
            # Récupérer le dernier prix depuis la base de données
            latest_price = db_manager.get_latest_price(symbol, timeframe)
            if latest_price:
                return {
                    'open': latest_price.open_price,
                    'high': latest_price.high_price,
                    'low': latest_price.low_price,
                    'close': latest_price.close_price,
                    'volume': latest_price.volume,
                    'timestamp': latest_price.timestamp
                }
            return None
        except Exception as e:
            logger.error(f"Erreur récupération données marché: {e}")
            return None

    def _analyze_market_conditions(self, symbol: str, timeframe: str) -> MarketConditions:
        """Analyse les conditions actuelles du marché"""
        try:
            # Récupérer les données historiques récentes
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=4)

            price_data = db_manager.get_price_data(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                limit=240  # 4 heures de données 1min
            )

            if not price_data or len(price_data) < 50:
                # Conditions par défaut si pas assez de données
                return MarketConditions(
                    volatility_regime='normal',
                    trend_direction='sideways',
                    trend_strength=0.3,
                    market_session=self._get_market_session(),
                    volume_profile='normal',
                    support_resistance_proximity=0.5
                )

            # Convertir en DataFrame
            df = pd.DataFrame([data.to_dict() for data in price_data])
            closes = df['close']
            volumes = df['volume']

            # Analyser la volatilité
            returns = closes.pct_change().dropna()
            volatility = returns.std() * np.sqrt(1440)  # Annualisée
            vol_percentile = pd.Series([volatility] * 100).rolling(100).rank(pct=True).iloc[-1]

            if vol_percentile > 0.8:
                volatility_regime = 'high'
            elif vol_percentile < 0.3:
                volatility_regime = 'low'
            else:
                volatility_regime = 'normal'

            # Analyser la tendance
            sma_20 = closes.rolling(20).mean()
            sma_50 = closes.rolling(50).mean()

            if len(sma_20) > 0 and len(sma_50) > 0:
                current_price = closes.iloc[-1]
                sma_20_val = sma_20.iloc[-1]
                sma_50_val = sma_50.iloc[-1]

                if current_price > sma_20_val > sma_50_val:
                    trend_direction = 'up'
                    trend_strength = min(1.0, (current_price - sma_50_val) / sma_50_val * 100)
                elif current_price < sma_20_val < sma_50_val:
                    trend_direction = 'down'
                    trend_strength = min(1.0, (sma_50_val - current_price) / sma_50_val * 100)
                else:
                    trend_direction = 'sideways'
                    trend_strength = 0.3
            else:
                trend_direction = 'sideways'
                trend_strength = 0.3

            # Analyser le volume
            if volumes.notna().any():
                volume_sma = volumes.rolling(20).mean()
                current_volume = volumes.iloc[-1]
                avg_volume = volume_sma.iloc[-1]

                if current_volume > avg_volume * 1.5:
                    volume_profile = 'high'
                elif current_volume < avg_volume * 0.7:
                    volume_profile = 'low'
                else:
                    volume_profile = 'normal'
            else:
                volume_profile = 'normal'

            # Proximité aux supports/résistances (simplifié)
            high_20 = df['high'].rolling(20).max().iloc[-1]
            low_20 = df['low'].rolling(20).min().iloc[-1]
            current_price = closes.iloc[-1]

            distance_to_resistance = (high_20 - current_price) / current_price
            distance_to_support = (current_price - low_20) / current_price
            support_resistance_proximity = min(distance_to_resistance, distance_to_support)

            return MarketConditions(
                volatility_regime=volatility_regime,
                trend_direction=trend_direction,
                trend_strength=abs(trend_strength),
                market_session=self._get_market_session(),
                volume_profile=volume_profile,
                support_resistance_proximity=support_resistance_proximity
            )

        except Exception as e:
            logger.error(f"Erreur analyse conditions marché: {e}")
            return MarketConditions(
                volatility_regime='normal',
                trend_direction='sideways',
                trend_strength=0.3,
                market_session='unknown',
                volume_profile='normal',
                support_resistance_proximity=0.5
            )

    def _get_market_session(self) -> str:
        """Détermine la session de marché actuelle"""
        current_hour = datetime.now(timezone.utc).hour

        if 0 <= current_hour < 8:
            return 'asian'
        elif 8 <= current_hour < 16:
            return 'european'
        elif 16 <= current_hour < 24:
            return 'american'
        else:
            return 'overlap'

    def _calculate_technical_score(self, summary: Dict[str, Any], indicators: Dict[str, IndicatorResult]) -> float:
        """Calcule le score basé sur les indicateurs techniques"""
        try:
            if not indicators:
                return 0.0

            # Score basé sur la confluence des signaux
            buy_weight = summary.get('weighted_buy_score', 0)
            sell_weight = summary.get('weighted_sell_score', 0)
            total_indicators = summary.get('total_indicators', 1)

            # Normaliser les scores
            max_possible_score = total_indicators * 1.0  # Confiance max de 1.0 par indicateur

            if buy_weight > sell_weight:
                score = buy_weight / max_possible_score
            elif sell_weight > buy_weight:
                score = -sell_weight / max_possible_score
            else:
                score = 0.0

            # Bonus pour la confluence (plus d'indicateurs dans le même sens)
            confluence_bonus = 0
            if summary.get('overall_signal') in ['BUY', 'SELL']:
                dominant_signals = max(summary.get('buy_signals', 0), summary.get('sell_signals', 0))
                confluence_bonus = (dominant_signals / total_indicators) * 0.2

            final_score = np.clip(score + confluence_bonus, -1.0, 1.0)
            return final_score

        except Exception as e:
            logger.error(f"Erreur calcul score technique: {e}")
            return 0.0

    def _calculate_pattern_score(self, summary: Dict[str, Any], patterns: List[PatternResult]) -> float:
        """Calcule le score basé sur les patterns"""
        try:
            if not patterns:
                return 0.0

            # Pondérer par la confiance des patterns
            bullish_score = 0
            bearish_score = 0

            for pattern in patterns:
                weight = pattern.confidence * pattern.strength

                if pattern.signal in [PatternSignal.BUY, PatternSignal.STRONG_BUY, PatternSignal.WEAK_BUY]:
                    bullish_score += weight
                elif pattern.signal in [PatternSignal.SELL, PatternSignal.STRONG_SELL, PatternSignal.WEAK_SELL]:
                    bearish_score += weight

            # Score net
            net_score = bullish_score - bearish_score

            # Normaliser par le nombre de patterns
            max_possible_score = len(patterns) * 1.0  # Confiance max * force max
            normalized_score = net_score / max_possible_score if max_possible_score > 0 else 0

            return np.clip(normalized_score, -1.0, 1.0)

        except Exception as e:
            logger.error(f"Erreur calcul score patterns: {e}")
            return 0.0

    def _determine_signal(self, combined_score: float, technical_summary: Dict[str, Any],
                          pattern_summary: Dict[str, Any], market_conditions: MarketConditions) -> Tuple[
        SignalType, SignalStrength, float]:
        """Détermine le type de signal, sa force et sa confiance"""

        # Ajustements selon les conditions de marché
        volatility_adjustment = 1.0
        if market_conditions.volatility_regime == 'high':
            volatility_adjustment = 0.8  # Réduire la confiance en haute volatilité
        elif market_conditions.volatility_regime == 'low':
            volatility_adjustment = 0.9  # Légère réduction en faible volatilité

        # Ajustement selon la tendance
        trend_adjustment = 1.0
        if market_conditions.trend_direction == 'up' and combined_score > 0:
            trend_adjustment = 1.1  # Bonus pour signal dans le sens de la tendance
        elif market_conditions.trend_direction == 'down' and combined_score < 0:
            trend_adjustment = 1.1
        elif market_conditions.trend_direction != 'sideways' and np.sign(combined_score) != np.sign(
                1 if market_conditions.trend_direction == 'up' else -1):
            trend_adjustment = 0.8  # Pénalité pour signal contre-tendance

        # Confiance ajustée
        base_confidence = abs(combined_score)
        adjusted_confidence = base_confidence * volatility_adjustment * trend_adjustment
        adjusted_confidence = np.clip(adjusted_confidence, 0.0, 1.0)

        # Déterminer le type de signal
        if combined_score > 0.1:
            signal_type = SignalType.BUY
        elif combined_score < -0.1:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD

        # Déterminer la force
        if adjusted_confidence >= 0.8:
            strength = SignalStrength.VERY_STRONG
        elif adjusted_confidence >= 0.7:
            strength = SignalStrength.STRONG
        elif adjusted_confidence >= 0.5:
            strength = SignalStrength.MODERATE
        elif adjusted_confidence >= 0.3:
            strength = SignalStrength.WEAK
        else:
            strength = SignalStrength.NEUTRAL

        return signal_type, strength, adjusted_confidence

    def _calculate_price_levels(self, signal_type: SignalType, entry_price: float,
                                market_conditions: MarketConditions,
                                indicators: Dict[str, IndicatorResult],
                                patterns: List[PatternResult]) -> Tuple[Optional[float], Optional[float]]:
        """Calcule les niveaux de stop loss et take profit"""
        try:
            if signal_type == SignalType.HOLD:
                return None, None

            # ATR pour dimensionner les niveaux
            atr_value = None
            if 'atr' in indicators:
                atr_value = indicators['atr'].value

            # Volatilité de base (% du prix)
            base_volatility = 0.001  # 0.1% pour Volatility 10
            if market_conditions.volatility_regime == 'high':
                base_volatility *= 2
            elif market_conditions.volatility_regime == 'low':
                base_volatility *= 0.5

            # Support/Résistance depuis les indicateurs
            support_level = None
            resistance_level = None

            if 'support' in indicators:
                support_level = indicators['support'].value
            if 'resistance' in indicators:
                resistance_level = indicators['resistance'].value

            # Bollinger Bands
            bb_upper = None
            bb_lower = None
            if 'bollinger' in indicators:
                bb_params = indicators['bollinger'].parameters
                bb_upper = bb_params.get('upper_band')
                bb_lower = bb_params.get('lower_band')

            # Calculer stop loss et take profit
            if signal_type == SignalType.BUY:
                # Stop loss: plus bas récent, support, ou ATR
                stop_options = []

                if support_level and support_level < entry_price:
                    stop_options.append(support_level * 0.999)  # Légèrement en dessous

                if bb_lower and bb_lower < entry_price:
                    stop_options.append(bb_lower)

                if atr_value:
                    stop_options.append(entry_price - (atr_value * 2))

                # Stop basé sur la volatilité
                stop_options.append(entry_price * (1 - base_volatility * 15))

                stop_loss = max(stop_options) if stop_options else entry_price * 0.985

                # Take profit: résistance ou multiple du stop
                risk = entry_price - stop_loss
                take_profit_options = []

                if resistance_level and resistance_level > entry_price:
                    take_profit_options.append(resistance_level * 0.999)  # Légèrement en dessous

                if bb_upper and bb_upper > entry_price:
                    take_profit_options.append(bb_upper)

                # TP basé sur ratio R/R
                take_profit_options.append(entry_price + (risk * 2))  # R/R 1:2

                take_profit = min(take_profit_options) if take_profit_options else entry_price * 1.02

            else:  # SELL
                # Stop loss: plus haut récent, résistance, ou ATR
                stop_options = []

                if resistance_level and resistance_level > entry_price:
                    stop_options.append(resistance_level * 1.001)  # Légèrement au dessus

                if bb_upper and bb_upper > entry_price:
                    stop_options.append(bb_upper)

                if atr_value:
                    stop_options.append(entry_price + (atr_value * 2))

                # Stop basé sur la volatilité
                stop_options.append(entry_price * (1 + base_volatility * 15))

                stop_loss = min(stop_options) if stop_options else entry_price * 1.015

                # Take profit: support ou multiple du stop
                risk = stop_loss - entry_price
                take_profit_options = []

                if support_level and support_level < entry_price:
                    take_profit_options.append(support_level * 1.001)  # Légèrement au dessus

                if bb_lower and bb_lower < entry_price:
                    take_profit_options.append(bb_lower)

                # TP basé sur ratio R/R
                take_profit_options.append(entry_price - (risk * 2))  # R/R 1:2

                take_profit = max(take_profit_options) if take_profit_options else entry_price * 0.98

            # Validation des niveaux
            if signal_type == SignalType.BUY:
                if stop_loss >= entry_price or take_profit <= entry_price:
                    return None, None
            else:
                if stop_loss <= entry_price or take_profit >= entry_price:
                    return None, None

            return stop_loss, take_profit

        except Exception as e:
            logger.error(f"Erreur calcul niveaux de prix: {e}")
            return None, None

    def _generate_reasoning(self, signal_type: SignalType, technical_summary: Dict[str, Any],
                            pattern_summary: Dict[str, Any], market_conditions: MarketConditions) -> str:
        """Génère l'explication du signal"""
        try:
            reasoning_parts = []

            # Signal principal
            reasoning_parts.append(f"Signal {signal_type.value}")

            # Indicateurs techniques
            if technical_summary.get('total_indicators', 0) > 0:
                overall_signal = technical_summary.get('overall_signal', 'HOLD')
                confidence = technical_summary.get('confidence', 0)
                reasoning_parts.append(f"Indicateurs techniques: {overall_signal} (conf: {confidence:.2f})")

            # Patterns
            if pattern_summary.get('total_patterns', 0) > 0:
                strongest_signal = pattern_summary.get('strongest_signal', 'NEUTRAL')
                strongest_pattern = pattern_summary.get('strongest_pattern', 'none')
                reasoning_parts.append(f"Pattern: {strongest_pattern} → {strongest_signal}")

            # Conditions de marché
            reasoning_parts.append(f"Marché: tendance {market_conditions.trend_direction}, "
                                   f"volatilité {market_conditions.volatility_regime}")

            # Session
            reasoning_parts.append(f"Session: {market_conditions.market_session}")

            return " | ".join(reasoning_parts)

        except Exception as e:
            logger.error(f"Erreur génération reasoning: {e}")
            return f"Signal {signal_type.value}"

    def _suggest_position_size(self, confidence: float, market_conditions: MarketConditions) -> float:
        """Suggère une taille de position basée sur la confiance et les conditions"""
        try:
            # Taille de base (% du capital)
            base_size = config.trading.max_capital_per_trade_pct

            # Ajustement selon la confiance
            confidence_multiplier = confidence  # 0.0 à 1.0

            # Ajustement selon la volatilité
            volatility_multiplier = 1.0
            if market_conditions.volatility_regime == 'high':
                volatility_multiplier = 0.5  # Réduire en haute volatilité
            elif market_conditions.volatility_regime == 'low':
                volatility_multiplier = 1.2  # Augmenter légèrement

            # Ajustement selon la force de tendance
            trend_multiplier = 1.0
            if market_conditions.trend_strength > 0.7:
                trend_multiplier = 1.1  # Augmenter si tendance forte
            elif market_conditions.trend_strength < 0.3:
                trend_multiplier = 0.8  # Réduire si pas de tendance

            suggested_size = base_size * confidence_multiplier * volatility_multiplier * trend_multiplier

            # Limites
            min_size = config.trading.max_capital_per_trade_pct * 0.25  # 25% du max
            max_size = config.trading.max_capital_per_trade_pct  # 100% du max

            return np.clip(suggested_size, min_size, max_size)

        except Exception as e:
            logger.error(f"Erreur suggestion taille position: {e}")
            return config.trading.max_capital_per_trade_pct * 0.5

    def _save_signal(self, signal: TradingSignal) -> bool:
        """Sauvegarde le signal en base de données"""
        try:
            signal_data = {
                'signal_id': signal.signal_id,
                'timestamp': signal.timestamp,
                'symbol': signal.symbol,
                'signal_type': signal.signal_type.value,
                'confidence': signal.confidence,
                'strength': signal.strength.value,
                'technical_score': signal.technical_score,
                'pattern_score': signal.pattern_score,
                'ai_prediction': signal.ai_score,
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'position_size': signal.position_size_suggestion,
                'risk_reward_ratio': signal.risk_reward_ratio,
                'indicators_used': signal.indicators_used,
                'reasoning': signal.reasoning,
                'market_conditions': signal.market_conditions.to_dict() if signal.market_conditions else {},
                'expires_at': signal.expires_at
            }

            saved_signal = db_manager.save_trading_signal(signal_data)

            if saved_signal:
                # Ajouter aux caches
                self.signal_cache[signal.signal_id] = signal
                self.recent_signals.append(signal)
                self.signal_history.append(signal)

                # Nettoyer les anciens signaux (garder seulement la dernière heure)
                current_time = datetime.now(timezone.utc)
                self.recent_signals = [
                    s for s in self.recent_signals
                    if (current_time - s.timestamp).total_seconds() < 3600
                ]

                logger.debug(f"Signal sauvegardé: {signal.signal_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Erreur sauvegarde signal: {e}")
            return False

    def _update_statistics(self, signal: TradingSignal):
        """Met à jour les statistiques de génération"""
        try:
            self.generation_stats['signals_generated'] += 1
            self.generation_stats['last_signal_time'] = signal.timestamp

            if signal.signal_type == SignalType.BUY:
                self.generation_stats['buy_signals'] += 1
            elif signal.signal_type == SignalType.SELL:
                self.generation_stats['sell_signals'] += 1
            else:
                self.generation_stats['hold_signals'] += 1

            # Confiance moyenne
            total_signals = self.generation_stats['signals_generated']
            old_avg = self.generation_stats['avg_confidence']
            self.generation_stats['avg_confidence'] = (
                    (old_avg * (total_signals - 1) + signal.confidence) / total_signals
            )

            # Confluence (score élevé)
            if signal.combined_score > self.config['confluence_threshold']:
                self.generation_stats['confluence_signals'] += 1

        except Exception as e:
            logger.error(f"Erreur mise à jour statistiques: {e}")

    def _is_signal_limit_reached(self) -> bool:
        """Vérifie si la limite de signaux par heure est atteinte"""
        return len(self.recent_signals) >= self.config['max_signals_per_hour']

    def get_active_signals(self, symbol: str) -> List[TradingSignal]:
        """Récupère les signaux actifs pour un symbole"""
        try:
            current_time = datetime.now(timezone.utc)

            # Filtrer les signaux actifs (non expirés)
            active_signals = [
                signal for signal in self.recent_signals
                if (signal.symbol == symbol and
                    signal.expires_at and
                    signal.expires_at > current_time)
            ]

            return active_signals

        except Exception as e:
            logger.error(f"Erreur récupération signaux actifs: {e}")
            return []

    def get_generation_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de génération"""
        return self.generation_stats.copy()


# Instance globale
signal_generator = SignalGenerator()


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def generate_trading_signal(symbol: str = "R_10", timeframe: str = "1m") -> Optional[TradingSignal]:
    """Génère un signal de trading pour un symbole"""
    return signal_generator.generate_signal(symbol, timeframe)


def get_active_signals(symbol: str = "R_10") -> List[TradingSignal]:
    """Récupère les signaux actifs pour un symbole"""
    return signal_generator.get_active_signals(symbol)


def get_signal_strength_description(strength: SignalStrength) -> str:
    """Retourne une description textuelle de la force du signal"""
    descriptions = {
        SignalStrength.VERY_STRONG: "Très Fort - Confluence de tous les indicateurs",
        SignalStrength.STRONG: "Fort - Plusieurs indicateurs convergents",
        SignalStrength.MODERATE: "Modéré - Quelques indicateurs alignés",
        SignalStrength.WEAK: "Faible - Peu d'indicateurs supportent",
        SignalStrength.NEUTRAL: "Neutre - Signaux contradictoires"
    }
    return descriptions.get(strength, "Inconnu")


if __name__ == "__main__":
    # Test du générateur de signaux
    print("📈 Test du générateur de signaux...")

    try:
        # Générer un signal de test
        signal = generate_trading_signal("R_10", "1m")

        if signal:
            print(f"✅ Signal généré:")
            print(f"   Type: {signal.signal_type.value}")
            print(f"   Force: {signal.strength.value}")
            print(f"   Confiance: {signal.confidence:.2f}")
            print(f"   Prix d'entrée: {signal.entry_price:.5f}")
            print(f"   Stop Loss: {signal.stop_loss:.5f if signal.stop_loss else 'N/A'}")
            print(f"   Take Profit: {signal.take_profit:.5f if signal.take_profit else 'N/A'}")
            print(f"   Ratio R/R: {signal.risk_reward_ratio:.2f if signal.risk_reward_ratio else 'N/A'}")
            print(f"   Reasoning: {signal.reasoning}")
        else:
            print("❌ Aucun signal généré")

        # Statistiques
        stats = signal_generator.get_generation_stats()
        print(f"\n📊 Statistiques:")
        for key, value in stats.items():
            print(f"   {key}: {value}")

        print("✅ Test du générateur de signaux terminé !")

    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        logger.error(f"Test du générateur de signaux échoué: {e}")