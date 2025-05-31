"""
Reconnaissance des patterns de chandeliers japonais et formations graphiques
Trading Bot Volatility 10 - Optimisé pour les indices synthétiques
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import warnings

from config import config
from data import db_manager

# Configuration du logger
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class PatternType(Enum):
    """Types de patterns de chandeliers"""
    # Patterns de continuation
    DOJI = "doji"
    SPINNING_TOP = "spinning_top"
    MARUBOZU_BULLISH = "marubozu_bullish"
    MARUBOZU_BEARISH = "marubozu_bearish"

    # Patterns de retournement haussiers
    HAMMER = "hammer"
    INVERTED_HAMMER = "inverted_hammer"
    DRAGONFLY_DOJI = "dragonfly_doji"
    BULLISH_ENGULFING = "bullish_engulfing"
    PIERCING_PATTERN = "piercing_pattern"
    MORNING_STAR = "morning_star"
    THREE_WHITE_SOLDIERS = "three_white_soldiers"

    # Patterns de retournement baissiers
    HANGING_MAN = "hanging_man"
    SHOOTING_STAR = "shooting_star"
    GRAVESTONE_DOJI = "gravestone_doji"
    BEARISH_ENGULFING = "bearish_engulfing"
    DARK_CLOUD_COVER = "dark_cloud_cover"
    EVENING_STAR = "evening_star"
    THREE_BLACK_CROWS = "three_black_crows"

    # Patterns multi-chandeliers
    INSIDE_BAR = "inside_bar"
    OUTSIDE_BAR = "outside_bar"
    HARAMI_BULLISH = "harami_bullish"
    HARAMI_BEARISH = "harami_bearish"


class PatternSignal(Enum):
    """Signal généré par le pattern"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    WEAK_BUY = "weak_buy"
    NEUTRAL = "neutral"
    WEAK_SELL = "weak_sell"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class CandleProperties:
    """Propriétés d'un chandelier"""
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: datetime

    @property
    def body_size(self) -> float:
        """Taille du corps du chandelier"""
        return abs(self.close - self.open)

    @property
    def total_range(self) -> float:
        """Range total du chandelier"""
        return self.high - self.low

    @property
    def upper_shadow(self) -> float:
        """Taille de l'ombre haute"""
        return self.high - max(self.open, self.close)

    @property
    def lower_shadow(self) -> float:
        """Taille de l'ombre basse"""
        return min(self.open, self.close) - self.low

    @property
    def is_bullish(self) -> bool:
        """Le chandelier est-il haussier"""
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        """Le chandelier est-il baissier"""
        return self.close < self.open

    @property
    def is_doji(self) -> bool:
        """Le chandelier est-il un doji"""
        return self.body_size <= self.total_range * 0.1

    @property
    def body_ratio(self) -> float:
        """Ratio corps/range total"""
        return self.body_size / (self.total_range + 1e-8)

    @property
    def upper_shadow_ratio(self) -> float:
        """Ratio ombre haute/range total"""
        return self.upper_shadow / (self.total_range + 1e-8)

    @property
    def lower_shadow_ratio(self) -> float:
        """Ratio ombre basse/range total"""
        return self.lower_shadow / (self.total_range + 1e-8)


@dataclass
class PatternResult:
    """Résultat de reconnaissance de pattern"""
    pattern_type: PatternType
    signal: PatternSignal
    confidence: float
    strength: float
    timestamp: datetime
    candles_involved: List[CandleProperties]
    description: str
    entry_level: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'pattern_type': self.pattern_type.value,
            'signal': self.signal.value,
            'confidence': self.confidence,
            'strength': self.strength,
            'timestamp': self.timestamp.isoformat(),
            'description': self.description,
            'entry_level': self.entry_level,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'candles_count': len(self.candles_involved)
        }


class PatternRecognizer:
    """Reconnaisseur de patterns de chandeliers japonais"""

    def __init__(self):
        # Seuils de configuration pour Volatility 10
        self.config = {
            'doji_threshold': 0.1,  # 10% du range pour être considéré comme doji
            'long_body_threshold': 0.6,  # 60% du range pour un corps long
            'short_body_threshold': 0.3,  # 30% du range pour un corps court
            'long_shadow_threshold': 0.6,  # 60% du range pour une ombre longue
            'min_volume_ratio': 0.8,  # Ratio de volume minimum
            'trend_lookback': 10,  # Périodes pour déterminer la tendance
            'volatility_adjustment': True  # Ajuster selon la volatilité
        }

        # Statistiques
        self.recognition_stats = {
            'patterns_detected': 0,
            'single_candle_patterns': 0,
            'multi_candle_patterns': 0,
            'bullish_patterns': 0,
            'bearish_patterns': 0,
            'last_recognition_time': None
        }

        logger.info("Reconnaisseur de patterns initialisé")

    def recognize_patterns(self, data: pd.DataFrame) -> List[PatternResult]:
        """
        Reconnaît tous les patterns dans un dataset

        Args:
            data: DataFrame avec données OHLCV

        Returns:
            Liste des patterns détectés
        """
        try:
            if len(data) < 3:
                logger.warning("Données insuffisantes pour la reconnaissance de patterns")
                return []

            logger.debug(f"Reconnaissance de patterns sur {len(data)} chandeliers")

            # Convertir en CandleProperties
            candles = self._convert_to_candles(data)

            patterns = []
            current_time = datetime.now(timezone.utc)

            # Reconnaissance des patterns single-candle
            patterns.extend(self._recognize_single_candle_patterns(candles))

            # Reconnaissance des patterns multi-candles
            if len(candles) >= 2:
                patterns.extend(self._recognize_two_candle_patterns(candles))

            if len(candles) >= 3:
                patterns.extend(self._recognize_three_candle_patterns(candles))

            # Mettre à jour les statistiques
            self.recognition_stats['patterns_detected'] += len(patterns)
            self.recognition_stats['last_recognition_time'] = current_time

            for pattern in patterns:
                if pattern.signal in [PatternSignal.BUY, PatternSignal.STRONG_BUY, PatternSignal.WEAK_BUY]:
                    self.recognition_stats['bullish_patterns'] += 1
                elif pattern.signal in [PatternSignal.SELL, PatternSignal.STRONG_SELL, PatternSignal.WEAK_SELL]:
                    self.recognition_stats['bearish_patterns'] += 1

            # Trier par confiance décroissante
            patterns.sort(key=lambda x: x.confidence, reverse=True)

            logger.debug(f"Détecté {len(patterns)} patterns")
            return patterns

        except Exception as e:
            logger.error(f"Erreur lors de la reconnaissance de patterns: {e}")
            return []

    def _convert_to_candles(self, data: pd.DataFrame) -> List[CandleProperties]:
        """Convertit le DataFrame en liste de CandleProperties"""
        candles = []

        for _, row in data.iterrows():
            candle = CandleProperties(
                open=row.get('open_price', row.get('open', 0)),
                high=row.get('high_price', row.get('high', 0)),
                low=row.get('low_price', row.get('low', 0)),
                close=row.get('close_price', row.get('close', 0)),
                volume=row.get('volume', 1000),
                timestamp=pd.to_datetime(row.get('timestamp', datetime.now()))
            )
            candles.append(candle)

        return candles

    def _recognize_single_candle_patterns(self, candles: List[CandleProperties]) -> List[PatternResult]:
        """Reconnaît les patterns à un seul chandelier"""
        patterns = []

        for i, candle in enumerate(candles[-20:]):  # Analyser les 20 derniers
            actual_index = len(candles) - 20 + i

            # Doji
            if self._is_doji(candle):
                pattern = self._analyze_doji(candle, candles[:actual_index + 1])
                if pattern:
                    patterns.append(pattern)

            # Hammer / Hanging Man
            if self._is_hammer_hanging_man(candle):
                pattern = self._analyze_hammer_hanging_man(candle, candles[:actual_index + 1])
                if pattern:
                    patterns.append(pattern)

            # Shooting Star / Inverted Hammer
            if self._is_shooting_star_inverted_hammer(candle):
                pattern = self._analyze_shooting_star_inverted_hammer(candle, candles[:actual_index + 1])
                if pattern:
                    patterns.append(pattern)

            # Marubozu
            if self._is_marubozu(candle):
                pattern = self._analyze_marubozu(candle)
                if pattern:
                    patterns.append(pattern)

            # Spinning Top
            if self._is_spinning_top(candle):
                pattern = self._analyze_spinning_top(candle, candles[:actual_index + 1])
                if pattern:
                    patterns.append(pattern)

        self.recognition_stats['single_candle_patterns'] += len(patterns)
        return patterns

    def _recognize_two_candle_patterns(self, candles: List[CandleProperties]) -> List[PatternResult]:
        """Reconnaît les patterns à deux chandeliers"""
        patterns = []

        for i in range(len(candles) - 1):
            if i < len(candles) - 20:  # Analyser les 20 dernières paires
                continue

            candle1, candle2 = candles[i], candles[i + 1]

            # Engulfing patterns
            if self._is_engulfing(candle1, candle2):
                pattern = self._analyze_engulfing(candle1, candle2, candles[:i + 2])
                if pattern:
                    patterns.append(pattern)

            # Harami patterns
            if self._is_harami(candle1, candle2):
                pattern = self._analyze_harami(candle1, candle2, candles[:i + 2])
                if pattern:
                    patterns.append(pattern)

            # Piercing Pattern / Dark Cloud Cover
            if self._is_piercing_or_dark_cloud(candle1, candle2):
                pattern = self._analyze_piercing_dark_cloud(candle1, candle2, candles[:i + 2])
                if pattern:
                    patterns.append(pattern)

            # Inside/Outside bars
            if self._is_inside_outside_bar(candle1, candle2):
                pattern = self._analyze_inside_outside_bar(candle1, candle2)
                if pattern:
                    patterns.append(pattern)

        self.recognition_stats['multi_candle_patterns'] += len(patterns)
        return patterns

    def _recognize_three_candle_patterns(self, candles: List[CandleProperties]) -> List[PatternResult]:
        """Reconnaît les patterns à trois chandeliers"""
        patterns = []

        for i in range(len(candles) - 2):
            if i < len(candles) - 20:  # Analyser les 20 derniers triplets
                continue

            candle1, candle2, candle3 = candles[i], candles[i + 1], candles[i + 2]

            # Morning Star / Evening Star
            if self._is_star_pattern(candle1, candle2, candle3):
                pattern = self._analyze_star_pattern(candle1, candle2, candle3, candles[:i + 3])
                if pattern:
                    patterns.append(pattern)

            # Three White Soldiers / Three Black Crows
            if self._is_three_soldiers_crows(candle1, candle2, candle3):
                pattern = self._analyze_three_soldiers_crows(candle1, candle2, candle3)
                if pattern:
                    patterns.append(pattern)

        self.recognition_stats['multi_candle_patterns'] += len(patterns)
        return patterns

    # =========================================================================
    # MÉTHODES DE DÉTECTION DES PATTERNS
    # =========================================================================

    def _is_doji(self, candle: CandleProperties) -> bool:
        """Vérifie si le chandelier est un doji"""
        return candle.body_ratio <= self.config['doji_threshold']

    def _is_hammer_hanging_man(self, candle: CandleProperties) -> bool:
        """Vérifie si c'est un hammer ou hanging man"""
        return (candle.lower_shadow_ratio >= self.config['long_shadow_threshold'] and
                candle.upper_shadow_ratio <= 0.1 and
                candle.body_ratio >= 0.1)

    def _is_shooting_star_inverted_hammer(self, candle: CandleProperties) -> bool:
        """Vérifie si c'est un shooting star ou inverted hammer"""
        return (candle.upper_shadow_ratio >= self.config['long_shadow_threshold'] and
                candle.lower_shadow_ratio <= 0.1 and
                candle.body_ratio >= 0.1)

    def _is_marubozu(self, candle: CandleProperties) -> bool:
        """Vérifie si c'est un marubozu"""
        return (candle.body_ratio >= self.config['long_body_threshold'] and
                candle.upper_shadow_ratio <= 0.05 and
                candle.lower_shadow_ratio <= 0.05)

    def _is_spinning_top(self, candle: CandleProperties) -> bool:
        """Vérifie si c'est un spinning top"""
        return (candle.body_ratio <= self.config['short_body_threshold'] and
                candle.upper_shadow_ratio >= 0.2 and
                candle.lower_shadow_ratio >= 0.2)

    def _is_engulfing(self, candle1: CandleProperties, candle2: CandleProperties) -> bool:
        """Vérifie si c'est un pattern engulfing"""
        return (candle1.is_bearish != candle2.is_bullish and
                candle2.body_size > candle1.body_size and
                ((candle2.is_bullish and candle2.close > candle1.open and candle2.open < candle1.close) or
                 (candle2.is_bearish and candle2.close < candle1.open and candle2.open > candle1.close)))

    def _is_harami(self, candle1: CandleProperties, candle2: CandleProperties) -> bool:
        """Vérifie si c'est un pattern harami"""
        return (candle1.body_size > candle2.body_size and
                max(candle2.open, candle2.close) < max(candle1.open, candle1.close) and
                min(candle2.open, candle2.close) > min(candle1.open, candle1.close))

    def _is_piercing_or_dark_cloud(self, candle1: CandleProperties, candle2: CandleProperties) -> bool:
        """Vérifie si c'est un piercing pattern ou dark cloud cover"""
        if candle1.is_bearish and candle2.is_bullish:
            # Piercing pattern
            return (candle2.close > (candle1.open + candle1.close) / 2 and
                    candle2.open < candle1.low)
        elif candle1.is_bullish and candle2.is_bearish:
            # Dark cloud cover
            return (candle2.close < (candle1.open + candle1.close) / 2 and
                    candle2.open > candle1.high)
        return False

    def _is_inside_outside_bar(self, candle1: CandleProperties, candle2: CandleProperties) -> bool:
        """Vérifie si c'est un inside ou outside bar"""
        # Inside bar
        inside = (candle2.high <= candle1.high and candle2.low >= candle1.low)
        # Outside bar
        outside = (candle2.high >= candle1.high and candle2.low <= candle1.low)
        return inside or outside

    def _is_star_pattern(self, candle1: CandleProperties, candle2: CandleProperties, candle3: CandleProperties) -> bool:
        """Vérifie si c'est un pattern étoile (morning/evening star)"""
        # Candle du milieu doit être petite et gapper
        small_middle = candle2.body_ratio <= 0.3
        gap_up = candle2.low > max(candle1.open, candle1.close)
        gap_down = candle2.high < min(candle1.open, candle1.close)

        return small_middle and (gap_up or gap_down)

    def _is_three_soldiers_crows(self, candle1: CandleProperties, candle2: CandleProperties,
                                 candle3: CandleProperties) -> bool:
        """Vérifie si c'est three white soldiers ou three black crows"""
        # Tous de la même couleur avec corps longs
        same_direction = ((candle1.is_bullish and candle2.is_bullish and candle3.is_bullish) or
                          (candle1.is_bearish and candle2.is_bearish and candle3.is_bearish))

        long_bodies = (candle1.body_ratio >= 0.5 and
                       candle2.body_ratio >= 0.5 and
                       candle3.body_ratio >= 0.5)

        return same_direction and long_bodies

    # =========================================================================
    # MÉTHODES D'ANALYSE DES PATTERNS
    # =========================================================================

    def _analyze_doji(self, candle: CandleProperties, history: List[CandleProperties]) -> Optional[PatternResult]:
        """Analyse un pattern doji"""
        trend = self._determine_trend(history)

        # Doji types
        if candle.lower_shadow_ratio >= 0.6:
            pattern_type = PatternType.DRAGONFLY_DOJI
            signal = PatternSignal.BUY if trend == 'down' else PatternSignal.NEUTRAL
            confidence = 0.7 if trend == 'down' else 0.4
        elif candle.upper_shadow_ratio >= 0.6:
            pattern_type = PatternType.GRAVESTONE_DOJI
            signal = PatternSignal.SELL if trend == 'up' else PatternSignal.NEUTRAL
            confidence = 0.7 if trend == 'up' else 0.4
        else:
            pattern_type = PatternType.DOJI
            signal = PatternSignal.NEUTRAL
            confidence = 0.5

        return PatternResult(
            pattern_type=pattern_type,
            signal=signal,
            confidence=confidence,
            strength=confidence,
            timestamp=candle.timestamp,
            candles_involved=[candle],
            description=f"{pattern_type.value} détecté dans tendance {trend}",
            entry_level=candle.close,
            stop_loss=candle.low if signal == PatternSignal.BUY else candle.high,
            take_profit=candle.close * 1.01 if signal == PatternSignal.BUY else candle.close * 0.99
        )

    def _analyze_hammer_hanging_man(self, candle: CandleProperties, history: List[CandleProperties]) -> Optional[
        PatternResult]:
        """Analyse un pattern hammer/hanging man"""
        trend = self._determine_trend(history)

        if trend == 'down':
            pattern_type = PatternType.HAMMER
            signal = PatternSignal.BUY
            confidence = 0.8
            description = "Hammer - Signal de retournement haussier"
        elif trend == 'up':
            pattern_type = PatternType.HANGING_MAN
            signal = PatternSignal.SELL
            confidence = 0.7
            description = "Hanging Man - Signal de retournement baissier"
        else:
            return None

        return PatternResult(
            pattern_type=pattern_type,
            signal=signal,
            confidence=confidence,
            strength=confidence,
            timestamp=candle.timestamp,
            candles_involved=[candle],
            description=description,
            entry_level=candle.close,
            stop_loss=candle.low if signal == PatternSignal.BUY else candle.high,
            take_profit=candle.close * 1.015 if signal == PatternSignal.BUY else candle.close * 0.985
        )

    def _analyze_shooting_star_inverted_hammer(self, candle: CandleProperties, history: List[CandleProperties]) -> \
    Optional[PatternResult]:
        """Analyse un pattern shooting star/inverted hammer"""
        trend = self._determine_trend(history)

        if trend == 'up':
            pattern_type = PatternType.SHOOTING_STAR
            signal = PatternSignal.SELL
            confidence = 0.8
            description = "Shooting Star - Signal de retournement baissier"
        elif trend == 'down':
            pattern_type = PatternType.INVERTED_HAMMER
            signal = PatternSignal.BUY
            confidence = 0.7
            description = "Inverted Hammer - Signal de retournement haussier"
        else:
            return None

        return PatternResult(
            pattern_type=pattern_type,
            signal=signal,
            confidence=confidence,
            strength=confidence,
            timestamp=candle.timestamp,
            candles_involved=[candle],
            description=description,
            entry_level=candle.close,
            stop_loss=candle.low if signal == PatternSignal.BUY else candle.high,
            take_profit=candle.close * 1.015 if signal == PatternSignal.BUY else candle.close * 0.985
        )

    def _analyze_marubozu(self, candle: CandleProperties) -> PatternResult:
        """Analyse un pattern marubozu"""
        if candle.is_bullish:
            pattern_type = PatternType.MARUBOZU_BULLISH
            signal = PatternSignal.BUY
            description = "Marubozu Bullish - Force haussière"
        else:
            pattern_type = PatternType.MARUBOZU_BEARISH
            signal = PatternSignal.SELL
            description = "Marubozu Bearish - Force baissière"

        confidence = min(0.9, candle.body_ratio)

        return PatternResult(
            pattern_type=pattern_type,
            signal=signal,
            confidence=confidence,
            strength=confidence,
            timestamp=candle.timestamp,
            candles_involved=[candle],
            description=description,
            entry_level=candle.close,
            stop_loss=candle.open,
            take_profit=candle.close * 1.01 if signal == PatternSignal.BUY else candle.close * 0.99
        )

    def _analyze_spinning_top(self, candle: CandleProperties, history: List[CandleProperties]) -> PatternResult:
        """Analyse un pattern spinning top"""
        return PatternResult(
            pattern_type=PatternType.SPINNING_TOP,
            signal=PatternSignal.NEUTRAL,
            confidence=0.6,
            strength=0.3,
            timestamp=candle.timestamp,
            candles_involved=[candle],
            description="Spinning Top - Indécision du marché",
            entry_level=candle.close
        )

    def _analyze_engulfing(self, candle1: CandleProperties, candle2: CandleProperties,
                           history: List[CandleProperties]) -> PatternResult:
        """Analyse un pattern engulfing"""
        if candle2.is_bullish:
            pattern_type = PatternType.BULLISH_ENGULFING
            signal = PatternSignal.STRONG_BUY
            description = "Bullish Engulfing - Signal haussier fort"
            entry_level = candle2.close
            stop_loss = candle1.low
            take_profit = candle2.close * 1.02
        else:
            pattern_type = PatternType.BEARISH_ENGULFING
            signal = PatternSignal.STRONG_SELL
            description = "Bearish Engulfing - Signal baissier fort"
            entry_level = candle2.close
            stop_loss = candle1.high
            take_profit = candle2.close * 0.98

        # Ajuster la confiance selon la taille relative
        size_ratio = candle2.body_size / candle1.body_size
        confidence = min(0.95, 0.7 + (size_ratio - 1) * 0.2)

        return PatternResult(
            pattern_type=pattern_type,
            signal=signal,
            confidence=confidence,
            strength=confidence,
            timestamp=candle2.timestamp,
            candles_involved=[candle1, candle2],
            description=description,
            entry_level=entry_level,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

    def _analyze_harami(self, candle1: CandleProperties, candle2: CandleProperties,
                        history: List[CandleProperties]) -> PatternResult:
        """Analyse un pattern harami"""
        trend = self._determine_trend(history[:-2])

        if trend == 'up' and candle1.is_bullish:
            pattern_type = PatternType.HARAMI_BEARISH
            signal = PatternSignal.WEAK_SELL
            description = "Bearish Harami - Signal de retournement faible"
        elif trend == 'down' and candle1.is_bearish:
            pattern_type = PatternType.HARAMI_BULLISH
            signal = PatternSignal.WEAK_BUY
            description = "Bullish Harami - Signal de retournement faible"
        else:
            pattern_type = PatternType.HARAMI_BULLISH if candle2.is_bullish else PatternType.HARAMI_BEARISH
            signal = PatternSignal.NEUTRAL
            description = "Harami - Indécision"

        confidence = 0.6

        return PatternResult(
            pattern_type=pattern_type,
            signal=signal,
            confidence=confidence,
            strength=0.5,
            timestamp=candle2.timestamp,
            candles_involved=[candle1, candle2],
            description=description,
            entry_level=candle2.close,
            stop_loss=candle1.low if signal == PatternSignal.WEAK_BUY else candle1.high,
            take_profit=candle2.close * 1.01 if signal == PatternSignal.WEAK_BUY else candle2.close * 0.99
        )

    def _analyze_piercing_dark_cloud(self, candle1: CandleProperties, candle2: CandleProperties,
                                     history: List[CandleProperties]) -> PatternResult:
        """Analyse piercing pattern ou dark cloud cover"""
        if candle1.is_bearish and candle2.is_bullish:
            pattern_type = PatternType.PIERCING_PATTERN
            signal = PatternSignal.BUY
            description = "Piercing Pattern - Signal haussier"
            stop_loss = candle2.low
            take_profit = candle2.close * 1.015
        else:
            pattern_type = PatternType.DARK_CLOUD_COVER
            signal = PatternSignal.SELL
            description = "Dark Cloud Cover - Signal baissier"
            stop_loss = candle2.high
            take_profit = candle2.close * 0.985

        # Confiance basée sur la pénétration
        penetration = abs(candle2.close - (candle1.open + candle1.close) / 2) / candle1.body_size
        confidence = min(0.85, 0.6 + penetration * 0.3)

        return PatternResult(
            pattern_type=pattern_type,
            signal=signal,
            confidence=confidence,
            strength=confidence,
            timestamp=candle2.timestamp,
            candles_involved=[candle1, candle2],
            description=description,
            entry_level=candle2.close,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

    def _analyze_inside_outside_bar(self, candle1: CandleProperties, candle2: CandleProperties) -> PatternResult:
        """Analyse inside/outside bar"""
        if candle2.high <= candle1.high and candle2.low >= candle1.low:
            pattern_type = PatternType.INSIDE_BAR
            signal = PatternSignal.NEUTRAL
            description = "Inside Bar - Consolidation"
            confidence = 0.4
        else:
            pattern_type = PatternType.OUTSIDE_BAR
            signal = PatternSignal.BUY if candle2.is_bullish else PatternSignal.SELL
            description = "Outside Bar - Breakout"
            confidence = 0.7

        return PatternResult(
            pattern_type=pattern_type,
            signal=signal,
            confidence=confidence,
            strength=confidence,
            timestamp=candle2.timestamp,
            candles_involved=[candle1, candle2],
            description=description,
            entry_level=candle2.close
        )

    def _analyze_star_pattern(self, candle1: CandleProperties, candle2: CandleProperties, candle3: CandleProperties,
                              history: List[CandleProperties]) -> PatternResult:
        """Analyse morning/evening star pattern"""
        if candle1.is_bearish and candle3.is_bullish:
            pattern_type = PatternType.MORNING_STAR
            signal = PatternSignal.STRONG_BUY
            description = "Morning Star - Retournement haussier fort"
            stop_loss = min(candle1.low, candle2.low, candle3.low)
            take_profit = candle3.close * 1.025
        else:
            pattern_type = PatternType.EVENING_STAR
            signal = PatternSignal.STRONG_SELL
            description = "Evening Star - Retournement baissier fort"
            stop_loss = max(candle1.high, candle2.high, candle3.high)
            take_profit = candle3.close * 0.975

        # Confiance basée sur la taille des gaps et des corps
        gap_quality = (abs(candle2.low - max(candle1.open, candle1.close)) / candle1.body_size +
                       abs(min(candle3.open, candle3.close) - candle2.high) / candle3.body_size) / 2
        confidence = min(0.9, 0.7 + gap_quality * 0.2)

        return PatternResult(
            pattern_type=pattern_type,
            signal=signal,
            confidence=confidence,
            strength=confidence,
            timestamp=candle3.timestamp,
            candles_involved=[candle1, candle2, candle3],
            description=description,
            entry_level=candle3.close,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

    def _analyze_three_soldiers_crows(self, candle1: CandleProperties, candle2: CandleProperties,
                                      candle3: CandleProperties) -> PatternResult:
        """Analyse three white soldiers/three black crows"""
        if candle1.is_bullish:
            pattern_type = PatternType.THREE_WHITE_SOLDIERS
            signal = PatternSignal.STRONG_BUY
            description = "Three White Soldiers - Tendance haussière forte"
            stop_loss = candle1.low
            take_profit = candle3.close * 1.03
        else:
            pattern_type = PatternType.THREE_BLACK_CROWS
            signal = PatternSignal.STRONG_SELL
            description = "Three Black Crows - Tendance baissière forte"
            stop_loss = candle1.high
            take_profit = candle3.close * 0.97

        # Confiance basée sur la progression des clôtures
        progression = abs(candle3.close - candle1.close) / candle1.close
        confidence = min(0.9, 0.8 + progression * 10)

        return PatternResult(
            pattern_type=pattern_type,
            signal=signal,
            confidence=confidence,
            strength=confidence,
            timestamp=candle3.timestamp,
            candles_involved=[candle1, candle2, candle3],
            description=description,
            entry_level=candle3.close,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

    # =========================================================================
    # MÉTHODES UTILITAIRES
    # =========================================================================

    def _determine_trend(self, history: List[CandleProperties], lookback: Optional[int] = None) -> str:
        """Détermine la tendance récente"""
        if not history:
            return 'sideways'

        lookback = lookback or self.config['trend_lookback']
        recent_candles = history[-lookback:] if len(history) >= lookback else history

        if len(recent_candles) < 3:
            return 'sideways'

        # Calculer la pente de régression linéaire des clôtures
        closes = [c.close for c in recent_candles]
        x = np.arange(len(closes))

        try:
            slope = np.polyfit(x, closes, 1)[0]
            avg_price = np.mean(closes)
            slope_pct = (slope * len(closes)) / avg_price

            if slope_pct > 0.002:  # 0.2%
                return 'up'
            elif slope_pct < -0.002:  # -0.2%
                return 'down'
            else:
                return 'sideways'
        except:
            return 'sideways'

    def get_patterns_for_symbol(self, symbol: str = "R_10", timeframe: str = "1m",
                                lookback_hours: int = 6) -> List[PatternResult]:
        """
        Récupère et analyse les patterns pour un symbole

        Args:
            symbol: Symbole à analyser
            timeframe: Timeframe des données
            lookback_hours: Nombre d'heures de données à analyser

        Returns:
            Liste des patterns détectés
        """
        try:
            # Récupérer les données depuis la base
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=lookback_hours)

            price_data = db_manager.get_price_data(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                limit=500
            )

            if not price_data or len(price_data) < 10:
                logger.warning(
                    f"Données insuffisantes pour l'analyse de patterns: {len(price_data) if price_data else 0}")
                return []

            # Convertir en DataFrame
            df = pd.DataFrame([data.to_dict() for data in price_data])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)

            # Reconnaître les patterns
            patterns = self.recognize_patterns(df)

            return patterns

        except Exception as e:
            logger.error(f"Erreur lors de l'analyse des patterns pour {symbol}: {e}")
            return []

    def get_recognition_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de reconnaissance"""
        return self.recognition_stats.copy()


# Instance globale
pattern_recognizer = PatternRecognizer()


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def recognize_patterns(symbol: str = "R_10", timeframe: str = "1m") -> List[PatternResult]:
    """Reconnaît les patterns pour un symbole"""
    return pattern_recognizer.get_patterns_for_symbol(symbol, timeframe)


def get_pattern_summary(patterns: List[PatternResult]) -> Dict[str, Any]:
    """Génère un résumé des patterns détectés"""
    if not patterns:
        return {
            'total_patterns': 0,
            'bullish_patterns': 0,
            'bearish_patterns': 0,
            'neutral_patterns': 0,
            'strongest_signal': 'NEUTRAL',
            'avg_confidence': 0.0
        }

    bullish_count = sum(
        1 for p in patterns if p.signal in [PatternSignal.BUY, PatternSignal.STRONG_BUY, PatternSignal.WEAK_BUY])
    bearish_count = sum(
        1 for p in patterns if p.signal in [PatternSignal.SELL, PatternSignal.STRONG_SELL, PatternSignal.WEAK_SELL])
    neutral_count = sum(1 for p in patterns if p.signal == PatternSignal.NEUTRAL)

    # Signal le plus fort basé sur la confiance
    strongest_pattern = max(patterns, key=lambda x: x.confidence)
    avg_confidence = sum(p.confidence for p in patterns) / len(patterns)

    return {
        'total_patterns': len(patterns),
        'bullish_patterns': bullish_count,
        'bearish_patterns': bearish_count,
        'neutral_patterns': neutral_count,
        'strongest_signal': strongest_pattern.signal.value,
        'strongest_pattern': strongest_pattern.pattern_type.value,
        'avg_confidence': avg_confidence,
        'max_confidence': strongest_pattern.confidence
    }


if __name__ == "__main__":
    # Test du reconnaisseur de patterns
    print("🕯️ Test du reconnaisseur de patterns...")

    try:
        # Créer des données de test avec patterns spécifiques
        np.random.seed(42)

        # Simuler des chandeliers avec patterns
        test_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=50, freq='1min'),
            'open_price': [100, 101, 100.5, 99, 98.5, 99.2, 100.1, 101.5, 102, 101.8] + [100 + np.random.normal(0, 0.5)
                                                                                         for _ in range(40)],
            'high_price': [100.2, 101.3, 100.8, 99.1, 98.7, 99.8, 100.6, 102.1, 102.5, 102.2] + [
                100.5 + np.random.normal(0, 0.5) for _ in range(40)],
            'low_price': [99.8, 100.8, 100.2, 98.5, 98.2, 98.9, 99.8, 101.2, 101.5, 101.3] + [
                99.5 + np.random.normal(0, 0.5) for _ in range(40)],
            'close_price': [101, 100.9, 99.2, 98.7, 99.1, 100, 101.2, 102, 101.9, 101.5] + [
                100 + np.random.normal(0, 0.5) for _ in range(40)],
            'volume': [1000 + np.random.normal(0, 100) for _ in range(50)]
        })

        print(f"📊 Données de test créées: {len(test_data)} chandeliers")

        # Reconnaître les patterns
        patterns = pattern_recognizer.recognize_patterns(test_data)

        print(f"🔍 Patterns détectés: {len(patterns)}")

        # Afficher les patterns trouvés
        for i, pattern in enumerate(patterns[:5]):  # Premiers 5
            print(f"   {i + 1}. {pattern.pattern_type.value}: {pattern.signal.value} "
                  f"(confiance: {pattern.confidence:.2f})")

        # Résumé
        summary = get_pattern_summary(patterns)
        print(f"\n📈 Résumé des patterns:")
        print(f"   Total: {summary['total_patterns']}")
        print(f"   Haussiers: {summary['bullish_patterns']}")
        print(f"   Baissiers: {summary['bearish_patterns']}")
        print(f"   Neutres: {summary['neutral_patterns']}")
        print(f"   Signal le plus fort: {summary['strongest_signal']}")
        print(f"   Confiance moyenne: {summary['avg_confidence']:.2f}")

        # Statistiques
        stats = pattern_recognizer.get_recognition_stats()
        print(f"\n📊 Statistiques:")
        for key, value in stats.items():
            print(f"   {key}: {value}")

        print("✅ Test de reconnaissance de patterns réussi !")

    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        logger.error(f"Test de reconnaissance de patterns échoué: {e}")