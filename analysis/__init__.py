"""
Module d'analyse technique pour le Trading Bot Volatility 10
Centralise les indicateurs techniques, reconnaissance de patterns et génération de signaux

Usage:
    from analysis import generate_trading_signal, calculate_indicators, recognize_patterns
    from analysis import signal_generator, technical_indicators, pattern_recognizer
    from analysis import TradingSignal, PatternResult, IndicatorResult
"""

# Imports des classes principales
from .technical_indicators import (
    TechnicalIndicators,
    IndicatorResult,
    TrendDirection,
    SignalStrength,
    technical_indicators,
    calculate_indicators,
    get_indicator_summary
)

from .pattern_recognition import (
    PatternRecognizer,
    PatternResult,
    PatternType,
    PatternSignal,
    CandleProperties,
    pattern_recognizer,
    recognize_patterns,
    get_pattern_summary
)

from .signal_generator import (
    SignalGenerator,
    TradingSignal,
    SignalType,
    SignalSource,
    MarketConditions,
    signal_generator,
    generate_trading_signal,
    get_active_signals,
    get_signal_strength_description
)

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

# Configuration du logger
logger = logging.getLogger(__name__)

# Version du module
__version__ = "1.0.0"

# Informations sur le module
__author__ = "Trading Bot Team"
__description__ = "Module d'analyse technique pour le Trading Bot Volatility 10"

# Exports principaux
__all__ = [
    # Classes d'indicateurs techniques
    "TechnicalIndicators",
    "IndicatorResult",
    "TrendDirection",

    # Classes de reconnaissance de patterns
    "PatternRecognizer",
    "PatternResult",
    "PatternType",
    "PatternSignal",
    "CandleProperties",

    # Classes de génération de signaux
    "SignalGenerator",
    "TradingSignal",
    "SignalType",
    "SignalStrength",
    "SignalSource",
    "MarketConditions",

    # Instances globales
    "technical_indicators",
    "pattern_recognizer",
    "signal_generator",

    # Fonctions utilitaires - Indicateurs
    "calculate_indicators",
    "get_indicator_summary",

    # Fonctions utilitaires - Patterns
    "recognize_patterns",
    "get_pattern_summary",

    # Fonctions utilitaires - Signaux
    "generate_trading_signal",
    "get_active_signals",
    "get_signal_strength_description",

    # Fonctions du module
    "perform_complete_analysis",
    "get_analysis_summary",
    "get_market_overview",
    "validate_trading_conditions",
    "get_analysis_stats",

    # Métadonnées
    "__version__",
    "__author__",
    "__description__"
]


@dataclass
class CompleteAnalysis:
    """Résultat d'une analyse technique complète"""
    timestamp: datetime
    symbol: str
    timeframe: str

    # Résultats des analyses
    indicators: Dict[str, IndicatorResult]
    patterns: List[PatternResult]
    signal: Optional[TradingSignal]
    market_conditions: Optional[MarketConditions]

    # Résumés
    indicator_summary: Dict[str, Any]
    pattern_summary: Dict[str, Any]

    # Recommandations
    overall_recommendation: str  # 'BUY', 'SELL', 'HOLD'
    confidence_level: float
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'indicators_count': len(self.indicators),
            'patterns_count': len(self.patterns),
            'signal': self.signal.to_dict() if self.signal else None,
            'market_conditions': self.market_conditions.to_dict() if self.market_conditions else None,
            'indicator_summary': self.indicator_summary,
            'pattern_summary': self.pattern_summary,
            'overall_recommendation': self.overall_recommendation,
            'confidence_level': self.confidence_level,
            'risk_level': self.risk_level
        }


def perform_complete_analysis(symbol: str = "R_10", timeframe: str = "1m") -> CompleteAnalysis:
    """
    Effectue une analyse technique complète

    Args:
        symbol: Symbole à analyser
        timeframe: Timeframe des données

    Returns:
        Analyse complète avec tous les composants
    """
    try:
        current_time = datetime.now(timezone.utc)
        logger.info(f"🔍 Analyse complète pour {symbol} {timeframe}")

        # 1. Calcul des indicateurs techniques
        indicators = calculate_indicators(symbol, timeframe)
        indicator_summary = get_indicator_summary(indicators)

        # 2. Reconnaissance des patterns
        patterns = recognize_patterns(symbol, timeframe)
        pattern_summary = get_pattern_summary(patterns)

        # 3. Génération du signal de trading
        signal = generate_trading_signal(symbol, timeframe)

        # 4. Conditions de marché (depuis le signal ou recalculées)
        market_conditions = signal.market_conditions if signal else None

        # 5. Recommandation globale
        overall_recommendation, confidence_level, risk_level = _determine_overall_recommendation(
            indicator_summary, pattern_summary, signal
        )

        analysis = CompleteAnalysis(
            timestamp=current_time,
            symbol=symbol,
            timeframe=timeframe,
            indicators=indicators,
            patterns=patterns,
            signal=signal,
            market_conditions=market_conditions,
            indicator_summary=indicator_summary,
            pattern_summary=pattern_summary,
            overall_recommendation=overall_recommendation,
            confidence_level=confidence_level,
            risk_level=risk_level
        )

        logger.info(f"✅ Analyse complète terminée: {overall_recommendation} "
                    f"(confiance: {confidence_level:.2f}, risque: {risk_level})")

        return analysis

    except Exception as e:
        logger.error(f"Erreur lors de l'analyse complète: {e}")

        # Retourner une analyse vide en cas d'erreur
        return CompleteAnalysis(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            timeframe=timeframe,
            indicators={},
            patterns=[],
            signal=None,
            market_conditions=None,
            indicator_summary={'overall_signal': 'HOLD', 'confidence': 0.0},
            pattern_summary={'strongest_signal': 'NEUTRAL'},
            overall_recommendation='HOLD',
            confidence_level=0.0,
            risk_level='HIGH'
        )


def _determine_overall_recommendation(indicator_summary: Dict[str, Any],
                                      pattern_summary: Dict[str, Any],
                                      signal: Optional[TradingSignal]) -> Tuple[str, float, str]:
    """Détermine la recommandation globale"""
    try:
        # Si un signal a été généré, l'utiliser comme base
        if signal and signal.signal_type != SignalType.HOLD:
            recommendation = signal.signal_type.value
            confidence = signal.confidence

            # Déterminer le niveau de risque
            if signal.strength in [SignalStrength.VERY_STRONG, SignalStrength.STRONG]:
                risk_level = 'LOW'
            elif signal.strength == SignalStrength.MODERATE:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'HIGH'

            return recommendation, confidence, risk_level

        # Sinon, analyser les composants individuels
        technical_signal = indicator_summary.get('overall_signal', 'HOLD')
        technical_confidence = indicator_summary.get('confidence', 0.0)

        pattern_signal = pattern_summary.get('strongest_signal', 'NEUTRAL')
        pattern_confidence = pattern_summary.get('max_confidence', 0.0)

        # Combiner les signaux
        if technical_signal == 'BUY' and pattern_signal in ['BUY', 'STRONG_BUY', 'WEAK_BUY']:
            recommendation = 'BUY'
            confidence = (technical_confidence + pattern_confidence) / 2
        elif technical_signal == 'SELL' and pattern_signal in ['SELL', 'STRONG_SELL', 'WEAK_SELL']:
            recommendation = 'SELL'
            confidence = (technical_confidence + pattern_confidence) / 2
        elif technical_signal in ['BUY', 'SELL'] and pattern_signal == 'NEUTRAL':
            recommendation = technical_signal
            confidence = technical_confidence * 0.8  # Réduction sans support patterns
        else:
            recommendation = 'HOLD'
            confidence = 0.3

        # Niveau de risque basé sur la confiance
        if confidence >= 0.7:
            risk_level = 'LOW'
        elif confidence >= 0.5:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'HIGH'

        return recommendation, confidence, risk_level

    except Exception as e:
        logger.error(f"Erreur détermination recommandation: {e}")
        return 'HOLD', 0.0, 'HIGH'


def get_analysis_summary(symbol: str = "R_10", timeframe: str = "1m") -> Dict[str, Any]:
    """
    Génère un résumé rapide de l'analyse

    Args:
        symbol: Symbole à analyser
        timeframe: Timeframe des données

    Returns:
        Résumé de l'analyse
    """
    try:
        analysis = perform_complete_analysis(symbol, timeframe)

        return {
            'timestamp': analysis.timestamp.isoformat(),
            'symbol': symbol,
            'recommendation': analysis.overall_recommendation,
            'confidence': analysis.confidence_level,
            'risk_level': analysis.risk_level,
            'indicators_analyzed': len(analysis.indicators),
            'patterns_detected': len(analysis.patterns),
            'signal_generated': analysis.signal is not None,
            'market_trend': analysis.market_conditions.trend_direction if analysis.market_conditions else 'unknown',
            'volatility_regime': analysis.market_conditions.volatility_regime if analysis.market_conditions else 'unknown',
            'entry_price': analysis.signal.entry_price if analysis.signal else None,
            'stop_loss': analysis.signal.stop_loss if analysis.signal else None,
            'take_profit': analysis.signal.take_profit if analysis.signal else None,
            'risk_reward_ratio': analysis.signal.risk_reward_ratio if analysis.signal else None
        }

    except Exception as e:
        logger.error(f"Erreur génération résumé: {e}")
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'symbol': symbol,
            'recommendation': 'HOLD',
            'confidence': 0.0,
            'risk_level': 'HIGH',
            'error': str(e)
        }


def get_market_overview(symbols: List[str] = None, timeframe: str = "1m") -> Dict[str, Any]:
    """
    Génère un aperçu du marché pour plusieurs symboles

    Args:
        symbols: Liste des symboles à analyser
        timeframe: Timeframe des données

    Returns:
        Aperçu du marché
    """
    try:
        if symbols is None:
            symbols = ["R_10"]  # Par défaut, seulement Volatility 10

        overview = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'timeframe': timeframe,
            'symbols_analyzed': len(symbols),
            'market_sentiment': {'BUY': 0, 'SELL': 0, 'HOLD': 0},
            'avg_confidence': 0.0,
            'high_confidence_signals': 0,
            'symbols_detail': {}
        }

        total_confidence = 0

        for symbol in symbols:
            try:
                summary = get_analysis_summary(symbol, timeframe)

                # Compter les sentiments
                recommendation = summary.get('recommendation', 'HOLD')
                overview['market_sentiment'][recommendation] += 1

                # Confiance
                confidence = summary.get('confidence', 0.0)
                total_confidence += confidence

                if confidence >= 0.7:
                    overview['high_confidence_signals'] += 1

                # Détails par symbole
                overview['symbols_detail'][symbol] = {
                    'recommendation': recommendation,
                    'confidence': confidence,
                    'risk_level': summary.get('risk_level', 'HIGH'),
                    'signal_generated': summary.get('signal_generated', False)
                }

            except Exception as e:
                logger.error(f"Erreur analyse {symbol}: {e}")
                overview['symbols_detail'][symbol] = {
                    'recommendation': 'HOLD',
                    'confidence': 0.0,
                    'risk_level': 'HIGH',
                    'error': str(e)
                }

        # Confiance moyenne
        overview['avg_confidence'] = total_confidence / len(symbols) if symbols else 0.0

        # Sentiment global
        sentiment_counts = overview['market_sentiment']
        if sentiment_counts['BUY'] > sentiment_counts['SELL']:
            overview['overall_sentiment'] = 'BULLISH'
        elif sentiment_counts['SELL'] > sentiment_counts['BUY']:
            overview['overall_sentiment'] = 'BEARISH'
        else:
            overview['overall_sentiment'] = 'NEUTRAL'

        return overview

    except Exception as e:
        logger.error(f"Erreur aperçu marché: {e}")
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'error': str(e),
            'overall_sentiment': 'NEUTRAL'
        }


def validate_trading_conditions(symbol: str = "R_10", timeframe: str = "1m") -> Dict[str, Any]:
    """
    Valide les conditions de trading actuelles

    Args:
        symbol: Symbole à analyser
        timeframe: Timeframe des données

    Returns:
        Validation des conditions de trading
    """
    try:
        analysis = perform_complete_analysis(symbol, timeframe)

        validation = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'symbol': symbol,
            'trading_recommended': False,
            'reasons': [],
            'warnings': [],
            'green_flags': [],
            'overall_score': 0.0
        }

        score = 0.0
        max_score = 10.0

        # 1. Qualité du signal (3 points)
        if analysis.signal and analysis.confidence_level >= 0.7:
            validation['green_flags'].append(f"Signal de haute qualité (confiance: {analysis.confidence_level:.2f})")
            score += 3
        elif analysis.signal and analysis.confidence_level >= 0.5:
            validation['green_flags'].append(f"Signal modéré (confiance: {analysis.confidence_level:.2f})")
            score += 2
        elif analysis.signal:
            validation['warnings'].append(f"Signal de faible confiance ({analysis.confidence_level:.2f})")
            score += 1
        else:
            validation['reasons'].append("Aucun signal de trading généré")

        # 2. Confluence des indicateurs (2 points)
        indicator_signal = analysis.indicator_summary.get('overall_signal', 'HOLD')
        if indicator_signal in ['BUY', 'SELL']:
            buy_signals = analysis.indicator_summary.get('buy_signals', 0)
            sell_signals = analysis.indicator_summary.get('sell_signals', 0)
            total_signals = analysis.indicator_summary.get('total_indicators', 1)

            dominant_ratio = max(buy_signals, sell_signals) / total_signals
            if dominant_ratio >= 0.6:
                validation['green_flags'].append(f"Confluence des indicateurs ({dominant_ratio:.0%})")
                score += 2
            elif dominant_ratio >= 0.4:
                validation['green_flags'].append(f"Indicateurs modérément alignés ({dominant_ratio:.0%})")
                score += 1
            else:
                validation['warnings'].append(f"Indicateurs peu alignés ({dominant_ratio:.0%})")
        else:
            validation['warnings'].append("Indicateurs neutres")

        # 3. Support des patterns (2 points)
        if analysis.patterns:
            strong_patterns = [p for p in analysis.patterns if p.confidence >= 0.7]
            if strong_patterns:
                validation['green_flags'].append(f"{len(strong_patterns)} pattern(s) fort(s) détecté(s)")
                score += 2
            elif analysis.patterns:
                validation['green_flags'].append(f"{len(analysis.patterns)} pattern(s) détecté(s)")
                score += 1
        else:
            validation['warnings'].append("Aucun pattern significatif détecté")

        # 4. Conditions de marché (2 points)
        if analysis.market_conditions:
            # Volatilité
            if analysis.market_conditions.volatility_regime == 'normal':
                validation['green_flags'].append("Volatilité normale")
                score += 1
            elif analysis.market_conditions.volatility_regime == 'high':
                validation['warnings'].append("Volatilité élevée - risque accru")

            # Tendance
            if analysis.market_conditions.trend_strength > 0.6:
                validation['green_flags'].append(f"Tendance forte ({analysis.market_conditions.trend_direction})")
                score += 1
            elif analysis.market_conditions.trend_direction == 'sideways':
                validation['warnings'].append("Marché sans tendance claire")

        # 5. Ratio R/R (1 point)
        if analysis.signal and analysis.signal.risk_reward_ratio:
            if analysis.signal.risk_reward_ratio >= 2.0:
                validation['green_flags'].append(f"Excellent ratio R/R ({analysis.signal.risk_reward_ratio:.1f})")
                score += 1
            elif analysis.signal.risk_reward_ratio >= 1.5:
                validation['green_flags'].append(f"Bon ratio R/R ({analysis.signal.risk_reward_ratio:.1f})")
                score += 0.5
            else:
                validation['warnings'].append(f"Ratio R/R faible ({analysis.signal.risk_reward_ratio:.1f})")
        else:
            validation['warnings'].append("Pas de calcul de ratio R/R")

        # Score final et recommandation
        validation['overall_score'] = score / max_score

        if score >= 7:
            validation['trading_recommended'] = True
            validation['reasons'].append("Conditions excellentes pour le trading")
        elif score >= 5:
            validation['trading_recommended'] = True
            validation['reasons'].append("Conditions acceptables pour le trading")
        else:
            validation['trading_recommended'] = False
            validation['reasons'].append("Conditions insuffisantes pour le trading")

        return validation

    except Exception as e:
        logger.error(f"Erreur validation conditions: {e}")
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'symbol': symbol,
            'trading_recommended': False,
            'reasons': [f"Erreur lors de la validation: {str(e)}"],
            'overall_score': 0.0
        }


def get_analysis_stats() -> Dict[str, Any]:
    """
    Retourne les statistiques globales d'analyse

    Returns:
        Statistiques des modules d'analyse
    """
    try:
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'module_version': __version__,
            'technical_indicators': technical_indicators.get_calculation_stats(),
            'pattern_recognition': pattern_recognizer.get_recognition_stats(),
            'signal_generation': signal_generator.get_generation_stats(),
            'modules_loaded': {
                'technical_indicators': technical_indicators is not None,
                'pattern_recognizer': pattern_recognizer is not None,
                'signal_generator': signal_generator is not None
            }
        }

    except Exception as e:
        logger.error(f"Erreur récupération statistiques: {e}")
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'error': str(e)
        }


def print_analysis_banner():
    """Affiche la bannière du module d'analyse"""
    print("=" * 80)
    print("📊 TRADING BOT VOLATILITY 10 - MODULE D'ANALYSE TECHNIQUE")
    print("=" * 80)
    print(f"📦 Version: {__version__}")
    print(f"👥 Auteur: {__author__}")
    print(f"📝 Description: {__description__}")

    # Vérifier les modules chargés
    modules_status = {
        'Indicateurs techniques': technical_indicators is not None,
        'Reconnaissance patterns': pattern_recognizer is not None,
        'Génération signaux': signal_generator is not None
    }

    print(f"\n🔧 Modules chargés:")
    for module, loaded in modules_status.items():
        status_icon = "✅" if loaded else "❌"
        print(f"   {status_icon} {module}")

    # Statistiques rapides
    try:
        stats = get_analysis_stats()
        tech_stats = stats.get('technical_indicators', {})
        pattern_stats = stats.get('pattern_recognition', {})
        signal_stats = stats.get('signal_generation', {})

        print(f"\n📈 Statistiques:")
        print(f"   Indicateurs calculés: {tech_stats.get('indicators_calculated', 0)}")
        print(f"   Patterns détectés: {pattern_stats.get('patterns_detected', 0)}")
        print(f"   Signaux générés: {signal_stats.get('signals_generated', 0)}")

    except Exception as e:
        print(f"   ⚠️ Erreur récupération statistiques: {e}")

    print("=" * 80)


# Initialisation automatique au chargement du module
try:
    logger.info(f"Module d'analyse technique chargé (version {__version__})")

    # Vérifier que tous les sous-modules sont bien chargés
    if technical_indicators and pattern_recognizer and signal_generator:
        logger.info("✅ Tous les modules d'analyse sont opérationnels")
    else:
        logger.warning("⚠️ Certains modules d'analyse ne sont pas chargés")

except Exception as e:
    logger.error(f"⚠️ Erreur lors de l'initialisation du module d'analyse: {e}")
    # Ne pas faire planter l'import, permettre l'utilisation partielle