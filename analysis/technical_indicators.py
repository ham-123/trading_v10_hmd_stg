"""
Calcul des indicateurs techniques pour le Trading Bot Volatility 10
Optimisé pour les indices synthétiques et trading haute fréquence
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import warnings
from scipy import stats
from sklearn.linear_model import LinearRegression

from config import config
from data import db_manager, get_latest_market_data

# Configuration du logger
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class TrendDirection(Enum):
    """Direction de la tendance"""
    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    SIDEWAYS = "sideways"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"


class SignalStrength(Enum):
    """Force du signal"""
    VERY_STRONG = "very_strong"
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NEUTRAL = "neutral"


@dataclass
class IndicatorResult:
    """Résultat d'un indicateur technique"""
    name: str
    value: float
    signal: str  # 'BUY', 'SELL', 'HOLD'
    strength: SignalStrength
    confidence: float
    timestamp: datetime
    parameters: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'value': self.value,
            'signal': self.signal,
            'strength': self.strength.value,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'parameters': self.parameters
        }


@dataclass
class SupportResistanceLevel:
    """Niveau de support ou résistance"""
    level: float
    strength: float
    type: str  # 'support' ou 'resistance'
    touches: int
    last_touch: datetime
    validity_score: float


class TechnicalIndicators:
    """Calculateur d'indicateurs techniques optimisé pour Volatility 10"""

    def __init__(self):
        # Configuration depuis le fichier config
        self.config = config.indicators

        # Cache des calculs récents
        self.indicator_cache = {}
        self.cache_timeout = 60  # Cache valide pendant 1 minute

        # Statistiques de performance
        self.calculation_stats = {
            'indicators_calculated': 0,
            'cache_hits': 0,
            'errors': 0,
            'last_calculation_time': None
        }

        logger.info("Calculateur d'indicateurs techniques initialisé")

    def calculate_all_indicators(self, data: pd.DataFrame, symbol: str = "R_10") -> Dict[str, IndicatorResult]:
        """
        Calcule tous les indicateurs techniques pour un dataset

        Args:
            data: DataFrame avec colonnes OHLCV
            symbol: Symbole tradé

        Returns:
            Dictionnaire avec tous les indicateurs calculés
        """
        try:
            if len(data) < 50:
                logger.warning(f"Données insuffisantes pour le calcul des indicateurs: {len(data)} périodes")
                return {}

            logger.debug(f"Calcul des indicateurs pour {symbol} sur {len(data)} périodes")

            results = {}
            current_time = datetime.now(timezone.utc)

            # Préparer les données
            close = data['close_price'] if 'close_price' in data.columns else data['close']
            high = data['high_price'] if 'high_price' in data.columns else data['high']
            low = data['low_price'] if 'low_price' in data.columns else data['low']
            open_price = data['open_price'] if 'open_price' in data.columns else data['open']
            volume = data.get('volume', pd.Series(index=data.index, data=1000))

            # 1. Moyennes mobiles
            results.update(self._calculate_moving_averages(close, current_time))

            # 2. RSI
            results['rsi'] = self._calculate_rsi(close, current_time)

            # 3. MACD
            results.update(self._calculate_macd(close, current_time))

            # 4. Bollinger Bands
            results.update(self._calculate_bollinger_bands(close, current_time))

            # 5. Stochastic
            results['stochastic'] = self._calculate_stochastic(high, low, close, current_time)

            # 6. Williams %R
            results['williams_r'] = self._calculate_williams_r(high, low, close, current_time)

            # 7. CCI (Commodity Channel Index)
            results['cci'] = self._calculate_cci(high, low, close, current_time)

            # 8. ATR (Average True Range)
            results['atr'] = self._calculate_atr(high, low, close, current_time)

            # 9. Momentum
            results.update(self._calculate_momentum_indicators(close, current_time))

            # 10. Volume (si disponible)
            if not volume.empty and volume.notna().any():
                results.update(self._calculate_volume_indicators(close, volume, current_time))

            # 11. Support/Résistance
            sr_levels = self._calculate_support_resistance(high, low, close)
            if sr_levels:
                results.update(self._analyze_support_resistance(close.iloc[-1], sr_levels, current_time))

            # 12. Trend Analysis
            results['trend'] = self._analyze_trend(close, current_time)

            # 13. Volatilité
            results['volatility'] = self._calculate_volatility_indicators(close, current_time)

            self.calculation_stats['indicators_calculated'] += len(results)
            self.calculation_stats['last_calculation_time'] = current_time

            logger.debug(f"Calculé {len(results)} indicateurs pour {symbol}")
            return results

        except Exception as e:
            logger.error(f"Erreur lors du calcul des indicateurs: {e}")
            self.calculation_stats['errors'] += 1
            return {}

    def _calculate_moving_averages(self, close: pd.Series, timestamp: datetime) -> Dict[str, IndicatorResult]:
        """Calcule les moyennes mobiles SMA et EMA"""
        results = {}

        try:
            current_price = close.iloc[-1]

            # SMA
            for period in self.config.sma_periods:
                if len(close) >= period:
                    sma = close.rolling(window=period).mean().iloc[-1]

                    # Déterminer le signal
                    if current_price > sma * 1.001:  # 0.1% au-dessus
                        signal = 'BUY'
                        strength = SignalStrength.MODERATE
                    elif current_price < sma * 0.999:  # 0.1% en-dessous
                        signal = 'SELL'
                        strength = SignalStrength.MODERATE
                    else:
                        signal = 'HOLD'
                        strength = SignalStrength.WEAK

                    # Calculer la confiance basée sur la distance
                    distance_pct = abs((current_price - sma) / sma) * 100
                    confidence = min(0.9, distance_pct * 10)  # Max 90%

                    results[f'sma_{period}'] = IndicatorResult(
                        name=f'SMA_{period}',
                        value=sma,
                        signal=signal,
                        strength=strength,
                        confidence=confidence,
                        timestamp=timestamp,
                        parameters={'period': period, 'current_price': current_price}
                    )

            # EMA
            for period in self.config.ema_periods:
                if len(close) >= period:
                    ema = close.ewm(span=period).mean().iloc[-1]

                    # Signal plus sensible pour EMA
                    if current_price > ema * 1.0005:  # 0.05% au-dessus
                        signal = 'BUY'
                        strength = SignalStrength.MODERATE
                    elif current_price < ema * 0.9995:  # 0.05% en-dessous
                        signal = 'SELL'
                        strength = SignalStrength.MODERATE
                    else:
                        signal = 'HOLD'
                        strength = SignalStrength.WEAK

                    distance_pct = abs((current_price - ema) / ema) * 100
                    confidence = min(0.9, distance_pct * 15)  # Plus sensible

                    results[f'ema_{period}'] = IndicatorResult(
                        name=f'EMA_{period}',
                        value=ema,
                        signal=signal,
                        strength=strength,
                        confidence=confidence,
                        timestamp=timestamp,
                        parameters={'period': period, 'current_price': current_price}
                    )

        except Exception as e:
            logger.error(f"Erreur calcul moyennes mobiles: {e}")

        return results

    def _calculate_rsi(self, close: pd.Series, timestamp: datetime) -> IndicatorResult:
        """Calcule le RSI (Relative Strength Index)"""
        try:
            period = self.config.rsi_period

            # Calcul du RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi_value = rsi.iloc[-1]

            # Déterminer le signal
            if rsi_value > self.config.rsi_overbought:
                signal = 'SELL'
                strength = SignalStrength.STRONG if rsi_value > 80 else SignalStrength.MODERATE
                confidence = min(0.95, (rsi_value - 70) / 30)
            elif rsi_value < self.config.rsi_oversold:
                signal = 'BUY'
                strength = SignalStrength.STRONG if rsi_value < 20 else SignalStrength.MODERATE
                confidence = min(0.95, (30 - rsi_value) / 30)
            else:
                signal = 'HOLD'
                strength = SignalStrength.WEAK
                confidence = 0.1

            return IndicatorResult(
                name='RSI',
                value=rsi_value,
                signal=signal,
                strength=strength,
                confidence=confidence,
                timestamp=timestamp,
                parameters={
                    'period': period,
                    'overbought': self.config.rsi_overbought,
                    'oversold': self.config.rsi_oversold
                }
            )

        except Exception as e:
            logger.error(f"Erreur calcul RSI: {e}")
            return IndicatorResult('RSI', 50, 'HOLD', SignalStrength.NEUTRAL, 0, timestamp, {})

    def _calculate_macd(self, close: pd.Series, timestamp: datetime) -> Dict[str, IndicatorResult]:
        """Calcule le MACD (Moving Average Convergence Divergence)"""
        results = {}

        try:
            fast_period = self.config.macd_fast
            slow_period = self.config.macd_slow
            signal_period = self.config.macd_signal

            # Calcul MACD
            ema_fast = close.ewm(span=fast_period).mean()
            ema_slow = close.ewm(span=slow_period).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal_period).mean()
            histogram = macd_line - signal_line

            macd_value = macd_line.iloc[-1]
            signal_value = signal_line.iloc[-1]
            histogram_value = histogram.iloc[-1]

            # Signal basé sur l'histogramme et le croisement
            if histogram_value > 0 and histogram.iloc[-2] <= 0:
                signal = 'BUY'
                strength = SignalStrength.STRONG
                confidence = 0.8
            elif histogram_value < 0 and histogram.iloc[-2] >= 0:
                signal = 'SELL'
                strength = SignalStrength.STRONG
                confidence = 0.8
            elif histogram_value > 0:
                signal = 'BUY'
                strength = SignalStrength.MODERATE
                confidence = 0.6
            elif histogram_value < 0:
                signal = 'SELL'
                strength = SignalStrength.MODERATE
                confidence = 0.6
            else:
                signal = 'HOLD'
                strength = SignalStrength.WEAK
                confidence = 0.1

            results['macd'] = IndicatorResult(
                name='MACD',
                value=macd_value,
                signal=signal,
                strength=strength,
                confidence=confidence,
                timestamp=timestamp,
                parameters={
                    'fast_period': fast_period,
                    'slow_period': slow_period,
                    'signal_period': signal_period,
                    'signal_line': signal_value,
                    'histogram': histogram_value
                }
            )

        except Exception as e:
            logger.error(f"Erreur calcul MACD: {e}")

        return results

    def _calculate_bollinger_bands(self, close: pd.Series, timestamp: datetime) -> Dict[str, IndicatorResult]:
        """Calcule les Bollinger Bands"""
        results = {}

        try:
            period = self.config.bb_period
            std_dev = self.config.bb_std

            # Calcul des bandes
            sma = close.rolling(window=period).mean()
            std = close.rolling(window=period).std()
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)

            current_price = close.iloc[-1]
            upper_value = upper_band.iloc[-1]
            lower_value = lower_band.iloc[-1]
            middle_value = sma.iloc[-1]

            # Position dans les bandes (0 = bande basse, 1 = bande haute)
            band_position = (current_price - lower_value) / (upper_value - lower_value)

            # Largeur des bandes (volatilité)
            band_width = (upper_value - lower_value) / middle_value

            # Signal basé sur la position
            if band_position > 0.8:  # Proche de la bande haute
                signal = 'SELL'
                strength = SignalStrength.MODERATE
                confidence = min(0.9, (band_position - 0.8) * 5)
            elif band_position < 0.2:  # Proche de la bande basse
                signal = 'BUY'
                strength = SignalStrength.MODERATE
                confidence = min(0.9, (0.2 - band_position) * 5)
            else:
                signal = 'HOLD'
                strength = SignalStrength.WEAK
                confidence = 0.1

            results['bollinger'] = IndicatorResult(
                name='Bollinger_Bands',
                value=band_position,
                signal=signal,
                strength=strength,
                confidence=confidence,
                timestamp=timestamp,
                parameters={
                    'period': period,
                    'std_dev': std_dev,
                    'upper_band': upper_value,
                    'lower_band': lower_value,
                    'middle_band': middle_value,
                    'band_width': band_width
                }
            )

        except Exception as e:
            logger.error(f"Erreur calcul Bollinger Bands: {e}")

        return results

    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series,
                              timestamp: datetime) -> IndicatorResult:
        """Calcule l'oscillateur Stochastique"""
        try:
            k_period = 14
            d_period = 3

            # Calcul %K
            lowest_low = low.rolling(window=k_period).min()
            highest_high = high.rolling(window=k_period).max()
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))

            # Calcul %D (moyenne mobile de %K)
            d_percent = k_percent.rolling(window=d_period).mean()

            k_value = k_percent.iloc[-1]
            d_value = d_percent.iloc[-1]

            # Signal basé sur les niveaux et croisements
            if k_value > 80 and d_value > 80:
                signal = 'SELL'
                strength = SignalStrength.MODERATE
                confidence = min(0.9, (k_value - 80) / 20)
            elif k_value < 20 and d_value < 20:
                signal = 'BUY'
                strength = SignalStrength.MODERATE
                confidence = min(0.9, (20 - k_value) / 20)
            elif k_value > d_value and k_percent.iloc[-2] <= d_percent.iloc[-2]:
                signal = 'BUY'
                strength = SignalStrength.WEAK
                confidence = 0.5
            elif k_value < d_value and k_percent.iloc[-2] >= d_percent.iloc[-2]:
                signal = 'SELL'
                strength = SignalStrength.WEAK
                confidence = 0.5
            else:
                signal = 'HOLD'
                strength = SignalStrength.WEAK
                confidence = 0.1

            return IndicatorResult(
                name='Stochastic',
                value=k_value,
                signal=signal,
                strength=strength,
                confidence=confidence,
                timestamp=timestamp,
                parameters={
                    'k_period': k_period,
                    'd_period': d_period,
                    'k_value': k_value,
                    'd_value': d_value
                }
            )

        except Exception as e:
            logger.error(f"Erreur calcul Stochastic: {e}")
            return IndicatorResult('Stochastic', 50, 'HOLD', SignalStrength.NEUTRAL, 0, timestamp, {})

    def _calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series,
                              timestamp: datetime) -> IndicatorResult:
        """Calcule Williams %R"""
        try:
            period = 14

            # Calcul Williams %R
            highest_high = high.rolling(window=period).max()
            lowest_low = low.rolling(window=period).min()
            williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))

            wr_value = williams_r.iloc[-1]

            # Signal (Williams %R est inversé par rapport au Stochastic)
            if wr_value > -20:  # Suracheté
                signal = 'SELL'
                strength = SignalStrength.MODERATE
                confidence = min(0.9, abs(wr_value + 20) / 80)
            elif wr_value < -80:  # Survendu
                signal = 'BUY'
                strength = SignalStrength.MODERATE
                confidence = min(0.9, abs(wr_value + 80) / 20)
            else:
                signal = 'HOLD'
                strength = SignalStrength.WEAK
                confidence = 0.1

            return IndicatorResult(
                name='Williams_R',
                value=wr_value,
                signal=signal,
                strength=strength,
                confidence=confidence,
                timestamp=timestamp,
                parameters={'period': period}
            )

        except Exception as e:
            logger.error(f"Erreur calcul Williams %R: {e}")
            return IndicatorResult('Williams_R', -50, 'HOLD', SignalStrength.NEUTRAL, 0, timestamp, {})

    def _calculate_cci(self, high: pd.Series, low: pd.Series, close: pd.Series, timestamp: datetime) -> IndicatorResult:
        """Calcule le Commodity Channel Index (CCI)"""
        try:
            period = 20

            # Prix typique
            typical_price = (high + low + close) / 3

            # Moyenne mobile du prix typique
            sma_tp = typical_price.rolling(window=period).mean()

            # Déviation moyenne
            mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))

            # CCI
            cci = (typical_price - sma_tp) / (0.015 * mad)
            cci_value = cci.iloc[-1]

            # Signal
            if cci_value > 100:
                signal = 'SELL'
                strength = SignalStrength.MODERATE if cci_value > 200 else SignalStrength.WEAK
                confidence = min(0.9, (cci_value - 100) / 200)
            elif cci_value < -100:
                signal = 'BUY'
                strength = SignalStrength.MODERATE if cci_value < -200 else SignalStrength.WEAK
                confidence = min(0.9, abs(cci_value + 100) / 200)
            else:
                signal = 'HOLD'
                strength = SignalStrength.WEAK
                confidence = 0.1

            return IndicatorResult(
                name='CCI',
                value=cci_value,
                signal=signal,
                strength=strength,
                confidence=confidence,
                timestamp=timestamp,
                parameters={'period': period}
            )

        except Exception as e:
            logger.error(f"Erreur calcul CCI: {e}")
            return IndicatorResult('CCI', 0, 'HOLD', SignalStrength.NEUTRAL, 0, timestamp, {})

    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, timestamp: datetime) -> IndicatorResult:
        """Calcule l'Average True Range (ATR) - mesure de volatilité"""
        try:
            period = 14

            # True Range
            high_low = high - low
            high_close_prev = np.abs(high - close.shift(1))
            low_close_prev = np.abs(low - close.shift(1))

            true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)

            # ATR
            atr = true_range.rolling(window=period).mean()
            atr_value = atr.iloc[-1]

            # Calculer le percentile de volatilité
            atr_percentile = atr.rolling(window=100).rank(pct=True).iloc[-1]

            # Signal basé sur le niveau de volatilité
            if atr_percentile > 0.8:  # Volatilité élevée
                signal = 'HOLD'  # Éviter de trader en haute volatilité
                strength = SignalStrength.WEAK
                confidence = 0.8
            elif atr_percentile < 0.2:  # Volatilité faible
                signal = 'HOLD'  # Attendre plus de mouvement
                strength = SignalStrength.WEAK
                confidence = 0.6
            else:
                signal = 'HOLD'
                strength = SignalStrength.NEUTRAL
                confidence = 0.3

            return IndicatorResult(
                name='ATR',
                value=atr_value,
                signal=signal,
                strength=strength,
                confidence=confidence,
                timestamp=timestamp,
                parameters={
                    'period': period,
                    'volatility_percentile': atr_percentile
                }
            )

        except Exception as e:
            logger.error(f"Erreur calcul ATR: {e}")
            return IndicatorResult('ATR', 0, 'HOLD', SignalStrength.NEUTRAL, 0, timestamp, {})

    def _calculate_momentum_indicators(self, close: pd.Series, timestamp: datetime) -> Dict[str, IndicatorResult]:
        """Calcule les indicateurs de momentum"""
        results = {}

        try:
            # Rate of Change (ROC)
            for period in [5, 10, 20]:
                if len(close) > period:
                    roc = ((close - close.shift(period)) / close.shift(period)) * 100
                    roc_value = roc.iloc[-1]

                    # Signal basé sur la magnitude du ROC
                    if roc_value > 1:  # Plus de 1% de hausse
                        signal = 'BUY'
                        strength = SignalStrength.MODERATE if roc_value > 2 else SignalStrength.WEAK
                        confidence = min(0.8, abs(roc_value) / 5)
                    elif roc_value < -1:  # Plus de 1% de baisse
                        signal = 'SELL'
                        strength = SignalStrength.MODERATE if roc_value < -2 else SignalStrength.WEAK
                        confidence = min(0.8, abs(roc_value) / 5)
                    else:
                        signal = 'HOLD'
                        strength = SignalStrength.WEAK
                        confidence = 0.1

                    results[f'roc_{period}'] = IndicatorResult(
                        name=f'ROC_{period}',
                        value=roc_value,
                        signal=signal,
                        strength=strength,
                        confidence=confidence,
                        timestamp=timestamp,
                        parameters={'period': period}
                    )

        except Exception as e:
            logger.error(f"Erreur calcul momentum: {e}")

        return results

    def _calculate_volume_indicators(self, close: pd.Series, volume: pd.Series, timestamp: datetime) -> Dict[
        str, IndicatorResult]:
        """Calcule les indicateurs basés sur le volume"""
        results = {}

        try:
            # Volume SMA
            vol_sma = volume.rolling(window=self.config.volume_sma).mean()
            vol_ratio = volume.iloc[-1] / vol_sma.iloc[-1] if vol_sma.iloc[-1] > 0 else 1

            # On-Balance Volume (OBV)
            obv = pd.Series(index=close.index, dtype=float)
            obv.iloc[0] = volume.iloc[0]

            for i in range(1, len(close)):
                if close.iloc[i] > close.iloc[i - 1]:
                    obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i]
                elif close.iloc[i] < close.iloc[i - 1]:
                    obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i - 1]

            # Signal OBV basé sur la tendance
            obv_trend = obv.rolling(window=10).apply(lambda x: stats.linregress(range(len(x)), x)[0])
            obv_slope = obv_trend.iloc[-1]

            if vol_ratio > 1.5 and obv_slope > 0:  # Volume élevé et OBV montant
                signal = 'BUY'
                strength = SignalStrength.MODERATE
                confidence = min(0.8, vol_ratio / 3)
            elif vol_ratio > 1.5 and obv_slope < 0:  # Volume élevé et OBV descendant
                signal = 'SELL'
                strength = SignalStrength.MODERATE
                confidence = min(0.8, vol_ratio / 3)
            else:
                signal = 'HOLD'
                strength = SignalStrength.WEAK
                confidence = 0.2

            results['volume'] = IndicatorResult(
                name='Volume_Analysis',
                value=vol_ratio,
                signal=signal,
                strength=strength,
                confidence=confidence,
                timestamp=timestamp,
                parameters={
                    'volume_ratio': vol_ratio,
                    'obv_slope': obv_slope,
                    'current_volume': volume.iloc[-1]
                }
            )

        except Exception as e:
            logger.error(f"Erreur calcul indicateurs volume: {e}")

        return results

    def _calculate_support_resistance(self, high: pd.Series, low: pd.Series, close: pd.Series) -> List[
        SupportResistanceLevel]:
        """Identifie les niveaux de support et résistance dynamiques"""
        try:
            levels = []
            lookback = self.config.sr_lookback
            strength_threshold = self.config.sr_strength

            if len(close) < lookback:
                return levels

            # Identifier les pivots hauts et bas
            highs = high.rolling(window=5, center=True).max() == high
            lows = low.rolling(window=5, center=True).min() == low

            # Collecter les niveaux de résistance (pivots hauts)
            resistance_levels = high[highs].dropna()
            for level in resistance_levels.tail(20):  # 20 derniers pivots
                touches = ((high >= level * 0.999) & (high <= level * 1.001)).sum()
                if touches >= strength_threshold:
                    levels.append(SupportResistanceLevel(
                        level=level,
                        strength=touches,
                        type='resistance',
                        touches=touches,
                        last_touch=high[high >= level * 0.999].index[-1],
                        validity_score=min(1.0, touches / 10)
                    ))

            # Collecter les niveaux de support (pivots bas)
            support_levels = low[lows].dropna()
            for level in support_levels.tail(20):  # 20 derniers pivots
                touches = ((low >= level * 0.999) & (low <= level * 1.001)).sum()
                if touches >= strength_threshold:
                    levels.append(SupportResistanceLevel(
                        level=level,
                        strength=touches,
                        type='support',
                        touches=touches,
                        last_touch=low[low <= level * 1.001].index[-1],
                        validity_score=min(1.0, touches / 10)
                    ))

            # Trier par force
            levels.sort(key=lambda x: x.strength, reverse=True)
            return levels[:10]  # Retourner les 10 plus forts

        except Exception as e:
            logger.error(f"Erreur calcul support/résistance: {e}")
            return []

    def _analyze_support_resistance(self, current_price: float, levels: List[SupportResistanceLevel],
                                    timestamp: datetime) -> Dict[str, IndicatorResult]:
        """Analyse la position du prix par rapport aux supports/résistances"""
        results = {}

        try:
            if not levels:
                return results

            # Trouver le support et la résistance les plus proches
            supports = [l for l in levels if l.type == 'support' and l.level < current_price]
            resistances = [l for l in levels if l.type == 'resistance' and l.level > current_price]

            nearest_support = max(supports, key=lambda x: x.level) if supports else None
            nearest_resistance = min(resistances, key=lambda x: x.level) if resistances else None

            # Analyser la proximité
            if nearest_resistance:
                distance_to_resistance = (nearest_resistance.level - current_price) / current_price
                if distance_to_resistance < 0.001:  # Moins de 0.1%
                    signal = 'SELL'
                    strength = SignalStrength.STRONG
                    confidence = nearest_resistance.validity_score
                else:
                    signal = 'HOLD'
                    strength = SignalStrength.WEAK
                    confidence = 0.3

                results['resistance'] = IndicatorResult(
                    name='Resistance_Level',
                    value=nearest_resistance.level,
                    signal=signal,
                    strength=strength,
                    confidence=confidence,
                    timestamp=timestamp,
                    parameters={
                        'level': nearest_resistance.level,
                        'distance_pct': distance_to_resistance * 100,
                        'touches': nearest_resistance.touches,
                        'strength': nearest_resistance.strength
                    }
                )

            if nearest_support:
                distance_to_support = (current_price - nearest_support.level) / current_price
                if distance_to_support < 0.001:  # Moins de 0.1%
                    signal = 'BUY'
                    strength = SignalStrength.STRONG
                    confidence = nearest_support.validity_score
                else:
                    signal = 'HOLD'
                    strength = SignalStrength.WEAK
                    confidence = 0.3

                results['support'] = IndicatorResult(
                    name='Support_Level',
                    value=nearest_support.level,
                    signal=signal,
                    strength=strength,
                    confidence=confidence,
                    timestamp=timestamp,
                    parameters={
                        'level': nearest_support.level,
                        'distance_pct': distance_to_support * 100,
                        'touches': nearest_support.touches,
                        'strength': nearest_support.strength
                    }
                )

        except Exception as e:
            logger.error(f"Erreur analyse support/résistance: {e}")

        return results

    def _analyze_trend(self, close: pd.Series, timestamp: datetime) -> IndicatorResult:
        """Analyse la tendance générale"""
        try:
            # Utiliser plusieurs timeframes pour la tendance
            short_ma = close.rolling(window=20).mean()
            medium_ma = close.rolling(window=50).mean()
            long_ma = close.rolling(window=200).mean() if len(close) >= 200 else medium_ma

            current_price = close.iloc[-1]
            short_ma_val = short_ma.iloc[-1]
            medium_ma_val = medium_ma.iloc[-1]
            long_ma_val = long_ma.iloc[-1]

            # Calculer la pente de la tendance
            slope_short = stats.linregress(range(20), close.tail(20))[0]
            slope_medium = stats.linregress(range(min(50, len(close))), close.tail(min(50, len(close))))[0]

            # Déterminer la direction de tendance
            if (current_price > short_ma_val > medium_ma_val > long_ma_val and
                    slope_short > 0 and slope_medium > 0):
                trend_direction = TrendDirection.STRONG_UPTREND
                signal = 'BUY'
                strength = SignalStrength.STRONG
                confidence = 0.9
            elif (current_price > short_ma_val > medium_ma_val and slope_short > 0):
                trend_direction = TrendDirection.UPTREND
                signal = 'BUY'
                strength = SignalStrength.MODERATE
                confidence = 0.7
            elif (current_price < short_ma_val < medium_ma_val < long_ma_val and
                  slope_short < 0 and slope_medium < 0):
                trend_direction = TrendDirection.STRONG_DOWNTREND
                signal = 'SELL'
                strength = SignalStrength.STRONG
                confidence = 0.9
            elif (current_price < short_ma_val < medium_ma_val and slope_short < 0):
                trend_direction = TrendDirection.DOWNTREND
                signal = 'SELL'
                strength = SignalStrength.MODERATE
                confidence = 0.7
            else:
                trend_direction = TrendDirection.SIDEWAYS
                signal = 'HOLD'
                strength = SignalStrength.WEAK
                confidence = 0.3

            return IndicatorResult(
                name='Trend_Analysis',
                value=slope_short,
                signal=signal,
                strength=strength,
                confidence=confidence,
                timestamp=timestamp,
                parameters={
                    'direction': trend_direction.value,
                    'slope_short': slope_short,
                    'slope_medium': slope_medium,
                    'short_ma': short_ma_val,
                    'medium_ma': medium_ma_val,
                    'long_ma': long_ma_val
                }
            )

        except Exception as e:
            logger.error(f"Erreur analyse tendance: {e}")
            return IndicatorResult('Trend_Analysis', 0, 'HOLD', SignalStrength.NEUTRAL, 0, timestamp, {})

    def _calculate_volatility_indicators(self, close: pd.Series, timestamp: datetime) -> IndicatorResult:
        """Calcule les indicateurs de volatilité"""
        try:
            # Volatilité réalisée (écart-type des rendements)
            returns = close.pct_change().dropna()
            volatility_20 = returns.rolling(window=20).std() * np.sqrt(1440)  # Annualisée pour 1min

            current_vol = volatility_20.iloc[-1]
            vol_percentile = volatility_20.rolling(window=100).rank(pct=True).iloc[-1]

            # Signal basé sur le régime de volatilité
            if vol_percentile > 0.8:  # Volatilité très élevée
                signal = 'HOLD'  # Éviter de trader
                strength = SignalStrength.MODERATE
                confidence = 0.8
            elif vol_percentile < 0.2:  # Volatilité très faible
                signal = 'HOLD'  # Attendre plus de mouvement
                strength = SignalStrength.WEAK
                confidence = 0.6
            else:  # Volatilité normale
                signal = 'HOLD'
                strength = SignalStrength.NEUTRAL
                confidence = 0.4

            return IndicatorResult(
                name='Volatility',
                value=current_vol,
                signal=signal,
                strength=strength,
                confidence=confidence,
                timestamp=timestamp,
                parameters={
                    'volatility_percentile': vol_percentile,
                    'volatility_20d': current_vol
                }
            )

        except Exception as e:
            logger.error(f"Erreur calcul volatilité: {e}")
            return IndicatorResult('Volatility', 0, 'HOLD', SignalStrength.NEUTRAL, 0, timestamp, {})

    def get_indicators_for_symbol(self, symbol: str = "R_10", timeframe: str = "1m",
                                  lookback_hours: int = 24) -> Dict[str, IndicatorResult]:
        """
        Récupère et calcule tous les indicateurs pour un symbole

        Args:
            symbol: Symbole à analyser
            timeframe: Timeframe des données
            lookback_hours: Nombre d'heures de données à récupérer

        Returns:
            Dictionnaire avec tous les indicateurs
        """
        try:
            # Vérifier le cache
            cache_key = f"{symbol}_{timeframe}_{lookback_hours}"
            if cache_key in self.indicator_cache:
                cache_time, cached_results = self.indicator_cache[cache_key]
                if (datetime.now(timezone.utc) - cache_time).seconds < self.cache_timeout:
                    self.calculation_stats['cache_hits'] += 1
                    return cached_results

            # Récupérer les données depuis la base
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=lookback_hours)

            price_data = db_manager.get_price_data(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                limit=1000
            )

            if not price_data or len(price_data) < 50:
                logger.warning(
                    f"Données insuffisantes pour {symbol}: {len(price_data) if price_data else 0} enregistrements")
                return {}

            # Convertir en DataFrame
            df = pd.DataFrame([data.to_dict() for data in price_data])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)

            # Calculer tous les indicateurs
            indicators = self.calculate_all_indicators(df, symbol)

            # Mettre en cache
            self.indicator_cache[cache_key] = (datetime.now(timezone.utc), indicators)

            return indicators

        except Exception as e:
            logger.error(f"Erreur lors de la récupération des indicateurs pour {symbol}: {e}")
            return {}

    def get_calculation_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de calcul"""
        return self.calculation_stats.copy()


# Instance globale
technical_indicators = TechnicalIndicators()


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def calculate_indicators(symbol: str = "R_10", timeframe: str = "1m") -> Dict[str, IndicatorResult]:
    """Calcule tous les indicateurs pour un symbole"""
    return technical_indicators.get_indicators_for_symbol(symbol, timeframe)


def get_indicator_summary(indicators: Dict[str, IndicatorResult]) -> Dict[str, Any]:
    """Génère un résumé des signaux des indicateurs"""
    if not indicators:
        return {
            'total_indicators': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'hold_signals': 0,
            'overall_signal': 'HOLD',
            'confidence': 0.0
        }

    buy_count = sum(1 for ind in indicators.values() if ind.signal == 'BUY')
    sell_count = sum(1 for ind in indicators.values() if ind.signal == 'SELL')
    hold_count = sum(1 for ind in indicators.values() if ind.signal == 'HOLD')

    # Signal global basé sur la majorité pondérée par la confiance
    weighted_buy = sum(ind.confidence for ind in indicators.values() if ind.signal == 'BUY')
    weighted_sell = sum(ind.confidence for ind in indicators.values() if ind.signal == 'SELL')

    if weighted_buy > weighted_sell * 1.2:  # 20% de marge
        overall_signal = 'BUY'
        overall_confidence = weighted_buy / len(indicators)
    elif weighted_sell > weighted_buy * 1.2:
        overall_signal = 'SELL'
        overall_confidence = weighted_sell / len(indicators)
    else:
        overall_signal = 'HOLD'
        overall_confidence = 0.3

    return {
        'total_indicators': len(indicators),
        'buy_signals': buy_count,
        'sell_signals': sell_count,
        'hold_signals': hold_count,
        'overall_signal': overall_signal,
        'confidence': min(0.95, overall_confidence),
        'weighted_buy_score': weighted_buy,
        'weighted_sell_score': weighted_sell
    }


if __name__ == "__main__":
    # Test des indicateurs techniques
    print("📊 Test des indicateurs techniques...")

    try:
        # Créer des données de test
        np.random.seed(42)
        n_periods = 200

        # Simulation d'un prix avec tendance
        base_price = 100
        trend = np.linspace(0, 10, n_periods)
        noise = np.random.normal(0, 1, n_periods)
        prices = base_price + trend + noise

        test_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_periods, freq='1min'),
            'open_price': prices + np.random.normal(0, 0.1, n_periods),
            'high_price': prices + np.abs(np.random.normal(0, 0.5, n_periods)),
            'low_price': prices - np.abs(np.random.normal(0, 0.5, n_periods)),
            'close_price': prices,
            'volume': np.random.normal(1000, 200, n_periods)
        })

        print(f"📈 Données de test créées: {len(test_data)} périodes")

        # Calculer les indicateurs
        indicators = technical_indicators.calculate_all_indicators(test_data)

        print(f"🔢 Indicateurs calculés: {len(indicators)}")

        # Afficher un résumé
        for name, indicator in list(indicators.items())[:5]:  # Premiers 5
            print(f"   {name}: {indicator.signal} (confiance: {indicator.confidence:.2f})")

        # Résumé global
        summary = get_indicator_summary(indicators)
        print(f"\n📊 Résumé global:")
        print(f"   Signal global: {summary['overall_signal']}")
        print(f"   Confiance: {summary['confidence']:.2f}")
        print(f"   BUY: {summary['buy_signals']}, SELL: {summary['sell_signals']}, HOLD: {summary['hold_signals']}")

        # Statistiques
        stats = technical_indicators.get_calculation_stats()
        print(f"\n📈 Statistiques:")
        for key, value in stats.items():
            print(f"   {key}: {value}")

        print("✅ Test des indicateurs réussi !")

    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        logger.error(f"Test des indicateurs échoué: {e}")