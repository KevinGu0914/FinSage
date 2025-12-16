#!/usr/bin/env python
"""
Deep Coverage Tests for StockExpert
Target: 100% coverage of stock_expert.py

This test suite covers:
- All public methods including analyze()
- All private helper methods
- Edge cases and error handling
- Different market conditions (bullish, bearish, neutral)
- Signal detection (RSI, MACD, Bollinger Bands, trends)
- Parse failures and fallback behavior
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import json
from typing import Dict, List, Any
from unittest.mock import Mock, MagicMock, patch

from finsage.agents.experts.stock_expert import StockExpert
from finsage.agents.base_expert import (
    BaseExpert,
    ExpertRecommendation,
    ExpertReport,
    Action,
)
from finsage.config import AssetConfig


# ============================================================
# Test 1: Initialization and Class Methods
# ============================================================

class TestStockExpertInit:
    """Test initialization and class setup"""

    def test_init_with_default_symbols(self):
        """Test initialization with default symbols from config"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm)

        assert expert.asset_class == "stocks"
        assert len(expert.symbols) > 0
        assert "AAPL" in expert.symbols or "SPY" in expert.symbols

    def test_init_with_custom_symbols(self):
        """Test initialization with custom symbol list"""
        mock_llm = Mock()
        custom_symbols = ["TSLA", "NVDA", "AMD"]
        expert = StockExpert(llm_provider=mock_llm, symbols=custom_symbols)

        assert expert.symbols == custom_symbols
        assert "TSLA" in expert.symbols
        assert len(expert.symbols) == 3

    def test_init_with_config(self):
        """Test initialization with custom config"""
        mock_llm = Mock()
        custom_config = {
            "max_single_weight": 0.20,
            "min_confidence": 0.6
        }
        expert = StockExpert(
            llm_provider=mock_llm,
            symbols=["SPY"],
            config=custom_config
        )

        assert expert.max_single_weight == 0.20
        assert expert.min_confidence == 0.6

    def test_get_default_symbols_classmethod(self):
        """Test _get_default_symbols classmethod"""
        symbols = StockExpert._get_default_symbols()

        assert isinstance(symbols, list)
        assert len(symbols) > 0
        # Should contain either config defaults or hardcoded defaults
        assert any(s in symbols for s in ["AAPL", "SPY", "MSFT"])


# ============================================================
# Test 2: Property Methods
# ============================================================

class TestStockExpertProperties:
    """Test property methods"""

    def test_name_property(self):
        """Test name property returns correct string"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["SPY"])

        assert expert.name == "Stock Expert"
        assert isinstance(expert.name, str)

    def test_description_property(self):
        """Test description property"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["SPY"])

        description = expert.description
        assert isinstance(description, str)
        assert len(description) > 0
        assert "股票" in description or "美股" in description

    def test_expertise_property(self):
        """Test expertise property returns list of capabilities"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["SPY"])

        expertise = expert.expertise
        assert isinstance(expertise, list)
        assert len(expertise) > 0
        # Should contain key expertise areas
        assert any("基本面" in item for item in expertise)
        assert any("技术面" in item for item in expertise)


# ============================================================
# Test 3: Format Price Data
# ============================================================

class TestFormatPriceData:
    """Test _format_price_data method with various inputs"""

    def test_format_price_data_normal(self):
        """Test formatting with normal price data"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["AAPL", "MSFT"])

        market_data = {
            "AAPL": {
                "close": 150.25,
                "change_pct": 2.5,
                "volume": 50000000
            },
            "MSFT": {
                "close": 300.50,
                "change_pct": -1.2,
                "volume": 30000000
            }
        }

        result = expert._format_price_data(market_data)

        assert "AAPL" in result
        assert "MSFT" in result
        assert "$150.25" in result
        assert "$300.50" in result
        assert "+2.50%" in result or "2.50%" in result
        assert "-1.20%" in result or "1.20%" in result

    def test_format_price_data_empty(self):
        """Test formatting with empty data"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["SPY"])

        result = expert._format_price_data({})
        assert "暂无价格数据" in result

    def test_format_price_data_none(self):
        """Test formatting with None data"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["SPY"])

        result = expert._format_price_data(None)
        assert "暂无价格数据" in result

    def test_format_price_data_missing_fields(self):
        """Test formatting with missing fields (uses 'price' fallback)"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["TSLA"])

        market_data = {
            "TSLA": {
                "price": 200.0,  # Using 'price' instead of 'close'
                "change_pct": 0,
                "volume": "N/A"
            }
        }

        result = expert._format_price_data(market_data)
        assert "TSLA" in result
        assert "$200.00" in result

    def test_format_price_data_filters_symbols(self):
        """Test that only symbols in expert's list are included"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["AAPL"])

        market_data = {
            "AAPL": {"close": 150.0, "change_pct": 1.0, "volume": 1000000},
            "GOOGL": {"close": 2500.0, "change_pct": 0.5, "volume": 500000}  # Not in symbols
        }

        result = expert._format_price_data(market_data)
        assert "AAPL" in result
        assert "GOOGL" not in result


# ============================================================
# Test 4: Format Technical Indicators
# ============================================================

class TestFormatTechnicalIndicators:
    """Test _format_technical_indicators with signal detection"""

    def test_format_technical_indicators_normal(self):
        """Test formatting with complete indicator data"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["SPY"])

        indicators = {
            "SPY": {
                "rsi": 65.5,
                "macd": 0.523,
                "macd_signal": 0.450,
                "macd_hist": 0.073,
                "macd_cross": "bullish",
                "ma_20": 450.0,
                "ma_50": 445.0,
                "price": 455.0,
                "trend": "uptrend",
                "bb_position": "neutral"
            }
        }

        result = expert._format_technical_indicators(indicators)

        assert "SPY" in result
        assert "65.5" in result  # RSI
        assert "0.523" in result  # MACD
        assert "MACD金叉" in result
        assert "上涨趋势" in result

    def test_format_technical_indicators_empty(self):
        """Test formatting with empty indicators"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["SPY"])

        result = expert._format_technical_indicators({})
        assert "暂无技术指标" in result

    def test_format_technical_indicators_none(self):
        """Test formatting with None indicators"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["SPY"])

        result = expert._format_technical_indicators(None)
        assert "暂无技术指标" in result

    def test_format_technical_indicators_rsi_overbought(self):
        """Test RSI overbought signal detection (>70)"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["TSLA"])

        indicators = {
            "TSLA": {
                "rsi": 75.0,
                "macd": 0.1,
                "macd_signal": 0.1,
                "macd_hist": 0.0,
                "macd_cross": "neutral",
                "ma_20": 200.0,
                "ma_50": 195.0,
                "price": 205.0,
                "trend": "sideways",
                "bb_position": "neutral"
            }
        }

        result = expert._format_technical_indicators(indicators)

        assert "RSI超买" in result
        assert "做空信号汇总" in result or "⚠️" in result

    def test_format_technical_indicators_rsi_oversold(self):
        """Test RSI oversold signal detection (<30)"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["AAPL"])

        indicators = {
            "AAPL": {
                "rsi": 25.0,
                "macd": -0.1,
                "macd_signal": -0.1,
                "macd_hist": 0.0,
                "macd_cross": "neutral",
                "ma_20": 150.0,
                "ma_50": 155.0,
                "price": 145.0,
                "trend": "sideways",
                "bb_position": "neutral"
            }
        }

        result = expert._format_technical_indicators(indicators)

        assert "RSI超卖" in result
        assert "做多信号汇总" in result or "📈" in result

    def test_format_technical_indicators_macd_bearish_cross(self):
        """Test MACD bearish cross (death cross) detection"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["SPY"])

        indicators = {
            "SPY": {
                "rsi": 50.0,
                "macd": 0.1,
                "macd_signal": 0.2,
                "macd_hist": -0.1,
                "macd_cross": "bearish",
                "ma_20": 450.0,
                "ma_50": 455.0,
                "price": 445.0,
                "trend": "sideways",
                "bb_position": "neutral"
            }
        }

        result = expert._format_technical_indicators(indicators)

        assert "MACD死叉" in result
        assert "做空信号" in result or "⚠️" in result

    def test_format_technical_indicators_macd_bullish_cross(self):
        """Test MACD bullish cross (golden cross) detection"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["QQQ"])

        indicators = {
            "QQQ": {
                "rsi": 50.0,
                "macd": 0.2,
                "macd_signal": 0.1,
                "macd_hist": 0.1,
                "macd_cross": "bullish",
                "ma_20": 350.0,
                "ma_50": 345.0,
                "price": 355.0,
                "trend": "sideways",
                "bb_position": "neutral"
            }
        }

        result = expert._format_technical_indicators(indicators)

        assert "MACD金叉" in result
        assert "做多信号" in result or "📈" in result

    def test_format_technical_indicators_downtrend(self):
        """Test downtrend signal detection"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["NVDA"])

        indicators = {
            "NVDA": {
                "rsi": 45.0,
                "macd": -0.1,
                "macd_signal": -0.1,
                "macd_hist": 0.0,
                "macd_cross": "neutral",
                "ma_20": 500.0,
                "ma_50": 505.0,
                "price": 490.0,
                "trend": "downtrend",
                "bb_position": "neutral"
            }
        }

        result = expert._format_technical_indicators(indicators)

        assert "下跌趋势" in result
        assert "做空信号" in result or "⚠️" in result

    def test_format_technical_indicators_uptrend(self):
        """Test uptrend signal detection"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["MSFT"])

        indicators = {
            "MSFT": {
                "rsi": 55.0,
                "macd": 0.1,
                "macd_signal": 0.1,
                "macd_hist": 0.0,
                "macd_cross": "neutral",
                "ma_20": 300.0,
                "ma_50": 295.0,
                "price": 305.0,
                "trend": "uptrend",
                "bb_position": "neutral"
            }
        }

        result = expert._format_technical_indicators(indicators)

        assert "上涨趋势" in result

    def test_format_technical_indicators_bb_overbought(self):
        """Test Bollinger Bands overbought signal"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["AMZN"])

        indicators = {
            "AMZN": {
                "rsi": 60.0,
                "macd": 0.1,
                "macd_signal": 0.1,
                "macd_hist": 0.0,
                "macd_cross": "neutral",
                "ma_20": 150.0,
                "ma_50": 148.0,
                "price": 155.0,
                "trend": "sideways",
                "bb_position": "overbought"
            }
        }

        result = expert._format_technical_indicators(indicators)

        assert "BB超买" in result
        assert "布林带上轨" in result
        assert "做空信号" in result or "⚠️" in result

    def test_format_technical_indicators_bb_oversold(self):
        """Test Bollinger Bands oversold signal"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["META"])

        indicators = {
            "META": {
                "rsi": 40.0,
                "macd": -0.1,
                "macd_signal": -0.1,
                "macd_hist": 0.0,
                "macd_cross": "neutral",
                "ma_20": 250.0,
                "ma_50": 252.0,
                "price": 245.0,
                "trend": "sideways",
                "bb_position": "oversold"
            }
        }

        result = expert._format_technical_indicators(indicators)

        assert "BB超卖" in result
        assert "布林带下轨" in result
        assert "做多信号" in result or "📈" in result

    def test_format_technical_indicators_multiple_signals(self):
        """Test multiple bearish signals together"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["TSLA"])

        indicators = {
            "TSLA": {
                "rsi": 72.0,  # Overbought
                "macd": 0.1,
                "macd_signal": 0.2,
                "macd_hist": -0.1,
                "macd_cross": "bearish",  # Death cross
                "ma_20": 200.0,
                "ma_50": 205.0,
                "price": 195.0,
                "trend": "downtrend",  # Downtrend
                "bb_position": "overbought"  # BB overbought
            }
        }

        result = expert._format_technical_indicators(indicators)

        # Should detect multiple bearish signals
        assert "RSI超买" in result
        assert "MACD死叉" in result
        assert "下跌趋势" in result
        assert "BB超买" in result
        assert "做空信号汇总" in result

    def test_format_technical_indicators_alternative_field_names(self):
        """Test with alternative field names (rsi_14, sma_20, sma_50)"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["SPY"])

        indicators = {
            "SPY": {
                "rsi_14": 55.0,  # Alternative field name
                "macd": 0.1,
                "macd_signal": 0.1,
                "macd_hist": 0.0,
                "macd_cross": "neutral",
                "sma_20": 450.0,  # Alternative field name
                "sma_50": 445.0,  # Alternative field name
                "price": 455.0,
                "trend": "sideways",
                "bb_position": "neutral"
            }
        }

        result = expert._format_technical_indicators(indicators)

        assert "55.0" in result
        assert "450.00" in result
        assert "445.00" in result


# ============================================================
# Test 5: Format News
# ============================================================

class TestFormatNews:
    """Test _format_news method"""

    def test_format_news_normal(self):
        """Test formatting with normal news data"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["AAPL", "MSFT"])

        news_data = [
            {
                "title": "Apple releases new iPhone",
                "sentiment": "positive",
                "symbols": ["AAPL"]
            },
            {
                "title": "Microsoft cloud revenue up",
                "sentiment": "positive",
                "symbols": ["MSFT"]
            },
            {
                "title": "Market volatility increases",
                "sentiment": "neutral",
                "symbols": []  # General market news
            }
        ]

        result = expert._format_news(news_data)

        assert "Apple releases new iPhone" in result
        assert "Microsoft cloud revenue" in result
        assert "Market volatility" in result
        assert "positive" in result

    def test_format_news_empty(self):
        """Test formatting with empty news list"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["SPY"])

        result = expert._format_news([])
        assert "暂无相关新闻" in result

    def test_format_news_none(self):
        """Test formatting with None news"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["SPY"])

        result = expert._format_news(None)
        assert "暂无相关新闻" in result

    def test_format_news_filters_relevant(self):
        """Test that only relevant news is included"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["AAPL"])

        news_data = [
            {
                "title": "Apple news",
                "sentiment": "positive",
                "symbols": ["AAPL"]
            },
            {
                "title": "Tesla news",
                "sentiment": "neutral",
                "symbols": ["TSLA"]  # Not in expert symbols
            }
        ]

        result = expert._format_news(news_data)

        assert "Apple news" in result
        assert "Tesla news" not in result

    def test_format_news_limit_10(self):
        """Test that only first 10 news items are processed"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["SPY"])

        # Create 15 news items
        news_data = [
            {
                "title": f"News item {i}",
                "sentiment": "neutral",
                "symbols": []
            }
            for i in range(15)
        ]

        result = expert._format_news(news_data)

        # Should only include first 10
        lines = [line for line in result.split('\n') if line.strip()]
        assert len(lines) <= 10

    def test_format_news_no_symbols_field(self):
        """Test news items without symbols field"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["SPY"])

        news_data = [
            {
                "title": "Market update",
                "sentiment": "neutral"
                # No symbols field
            }
        ]

        result = expert._format_news(news_data)
        # Should still be included as general market news
        assert "Market update" in result


# ============================================================
# Test 6: Parse Action
# ============================================================

class TestParseAction:
    """Test _parse_action method"""

    def test_parse_action_short_actions(self):
        """Test parsing SHORT actions"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["SPY"])

        assert expert._parse_action("SHORT_100%") == Action.SHORT_100
        assert expert._parse_action("SHORT_75%") == Action.SHORT_75
        assert expert._parse_action("SHORT_50%") == Action.SHORT_50
        assert expert._parse_action("SHORT_25%") == Action.SHORT_25

    def test_parse_action_sell_actions(self):
        """Test parsing SELL actions"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["SPY"])

        assert expert._parse_action("SELL_100%") == Action.SELL_100
        assert expert._parse_action("SELL_75%") == Action.SELL_75
        assert expert._parse_action("SELL_50%") == Action.SELL_50
        assert expert._parse_action("SELL_25%") == Action.SELL_25

    def test_parse_action_hold(self):
        """Test parsing HOLD action"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["SPY"])

        assert expert._parse_action("HOLD") == Action.HOLD

    def test_parse_action_buy_actions(self):
        """Test parsing BUY actions"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["SPY"])

        assert expert._parse_action("BUY_25%") == Action.BUY_25
        assert expert._parse_action("BUY_50%") == Action.BUY_50
        assert expert._parse_action("BUY_75%") == Action.BUY_75
        assert expert._parse_action("BUY_100%") == Action.BUY_100

    def test_parse_action_invalid_defaults_to_hold(self):
        """Test that invalid actions default to HOLD"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["SPY"])

        assert expert._parse_action("INVALID") == Action.HOLD
        assert expert._parse_action("") == Action.HOLD
        assert expert._parse_action("BUY") == Action.HOLD  # Missing percentage


# ============================================================
# Test 7: Parse LLM Response
# ============================================================

class TestParseLLMResponse:
    """Test _parse_llm_response with various formats and errors"""

    def test_parse_llm_response_valid_json(self):
        """Test parsing valid JSON response"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["AAPL", "MSFT"])

        response = json.dumps({
            "recommendations": [
                {
                    "symbol": "AAPL",
                    "action": "BUY_50%",
                    "confidence": 0.8,
                    "target_weight": 0.15,
                    "reasoning": "Strong fundamentals",
                    "trend": "bullish",
                    "risk_level": "medium",
                    "volatility": 0.2,
                    "downside_risk": 0.1
                },
                {
                    "symbol": "MSFT",
                    "action": "HOLD",
                    "confidence": 0.6,
                    "target_weight": 0.10,
                    "reasoning": "Consolidation phase",
                    "trend": "neutral",
                    "risk_level": "low",
                    "volatility": 0.15,
                    "downside_risk": 0.05
                }
            ]
        })

        recommendations = expert._parse_llm_response(response)

        assert len(recommendations) == 2
        assert recommendations[0].symbol == "AAPL"
        assert recommendations[0].action == Action.BUY_50
        assert recommendations[0].confidence == 0.8
        assert recommendations[1].symbol == "MSFT"
        assert recommendations[1].action == Action.HOLD

    def test_parse_llm_response_json_code_block(self):
        """Test parsing JSON within ```json code blocks"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["SPY"])

        response = """Here is my analysis:

```json
{
    "recommendations": [
        {
            "symbol": "SPY",
            "action": "BUY_75%",
            "confidence": 0.85,
            "target_weight": 0.20,
            "reasoning": "Bullish trend",
            "trend": "bullish",
            "risk_level": "low"
        }
    ]
}
```

This is my recommendation."""

        recommendations = expert._parse_llm_response(response)

        assert len(recommendations) == 1
        assert recommendations[0].symbol == "SPY"
        assert recommendations[0].action == Action.BUY_75

    def test_parse_llm_response_generic_code_block(self):
        """Test parsing JSON within generic ``` code blocks"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["QQQ"])

        response = """```
{
    "recommendations": [
        {
            "symbol": "QQQ",
            "action": "SHORT_25%",
            "confidence": 0.7,
            "target_weight": 0.05,
            "reasoning": "Overvalued"
        }
    ]
}
```"""

        recommendations = expert._parse_llm_response(response)

        assert len(recommendations) == 1
        assert recommendations[0].symbol == "QQQ"
        assert recommendations[0].action == Action.SHORT_25

    def test_parse_llm_response_invalid_json(self):
        """Test handling of invalid JSON with fallback"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["AAPL", "MSFT", "GOOGL"])

        response = "This is not valid JSON at all"

        recommendations = expert._parse_llm_response(response)

        # Should return fallback HOLD recommendations
        assert len(recommendations) == 3
        assert all(rec.action == Action.HOLD for rec in recommendations)
        assert all(rec.confidence == 0.3 for rec in recommendations)
        assert all("[PARSE_FAILED]" in rec.reasoning for rec in recommendations)

    def test_parse_llm_response_missing_recommendations_key(self):
        """Test handling when 'recommendations' key is missing"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["SPY", "QQQ", "IWM"])

        response = json.dumps({
            "analysis": "Market is neutral",
            "other_field": "value"
        })

        recommendations = expert._parse_llm_response(response)

        # When "recommendations" key is missing, .get() returns [] default
        # and the loop doesn't execute, returning empty list (no fallback)
        assert len(recommendations) == 0

    def test_parse_llm_response_empty_recommendations(self):
        """Test handling of empty recommendations list"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["SPY"])

        response = json.dumps({
            "recommendations": []
        })

        recommendations = expert._parse_llm_response(response)

        # Should return empty list (no fallback for explicitly empty)
        assert len(recommendations) == 0

    def test_parse_llm_response_missing_fields(self):
        """Test handling of recommendations with missing fields"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["NVDA"])

        response = json.dumps({
            "recommendations": [
                {
                    "symbol": "NVDA",
                    "action": "BUY_50%"
                    # Missing other fields
                }
            ]
        })

        recommendations = expert._parse_llm_response(response)

        # Should use default values for missing fields
        assert len(recommendations) == 1
        assert recommendations[0].symbol == "NVDA"
        assert recommendations[0].confidence == 0.5  # Default
        assert recommendations[0].target_weight == 0.0  # Default

    def test_parse_llm_response_all_action_types(self):
        """Test parsing all 13 action types"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=list(range(13)))

        actions = [
            "SHORT_100%", "SHORT_75%", "SHORT_50%", "SHORT_25%",
            "SELL_100%", "SELL_75%", "SELL_50%", "SELL_25%",
            "HOLD",
            "BUY_25%", "BUY_50%", "BUY_75%", "BUY_100%"
        ]

        recommendations_data = [
            {
                "symbol": str(i),
                "action": action,
                "confidence": 0.7,
                "target_weight": 0.1,
                "reasoning": f"Test {action}"
            }
            for i, action in enumerate(actions)
        ]

        response = json.dumps({"recommendations": recommendations_data})

        recommendations = expert._parse_llm_response(response)

        assert len(recommendations) == 13
        assert recommendations[0].action == Action.SHORT_100
        assert recommendations[4].action == Action.SELL_100
        assert recommendations[8].action == Action.HOLD
        assert recommendations[12].action == Action.BUY_100


# ============================================================
# Test 8: Build Analysis Prompt
# ============================================================

class TestBuildAnalysisPrompt:
    """Test _build_analysis_prompt method"""

    def test_build_analysis_prompt_complete(self):
        """Test building prompt with complete data"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["AAPL", "MSFT"])

        market_data = {
            "AAPL": {"close": 150.0, "change_pct": 2.0, "volume": 50000000},
            "MSFT": {"close": 300.0, "change_pct": -1.0, "volume": 30000000}
        }

        news_data = [
            {
                "title": "Tech stocks rally",
                "sentiment": "positive",
                "symbols": ["AAPL", "MSFT"]
            }
        ]

        technical_indicators = {
            "AAPL": {
                "rsi": 65.0,
                "macd": 0.5,
                "macd_signal": 0.4,
                "macd_hist": 0.1,
                "macd_cross": "bullish",
                "ma_20": 145.0,
                "ma_50": 140.0,
                "price": 150.0,
                "trend": "uptrend",
                "bb_position": "neutral"
            }
        }

        prompt = expert._build_analysis_prompt(
            market_data=market_data,
            news_data=news_data,
            technical_indicators=technical_indicators
        )

        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "AAPL" in prompt
        assert "MSFT" in prompt
        assert "价格数据" in prompt or "价格" in prompt
        assert "技术指标" in prompt
        assert "新闻" in prompt

    def test_build_analysis_prompt_empty_data(self):
        """Test building prompt with empty data"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["SPY"])

        prompt = expert._build_analysis_prompt(
            market_data={},
            news_data=[],
            technical_indicators={}
        )

        assert isinstance(prompt, str)
        assert "暂无" in prompt  # Should indicate no data

    def test_build_analysis_prompt_symbols_in_header(self):
        """Test that symbols are listed in the prompt"""
        mock_llm = Mock()
        symbols = ["AAPL", "MSFT", "GOOGL"]
        expert = StockExpert(llm_provider=mock_llm, symbols=symbols)

        prompt = expert._build_analysis_prompt(
            market_data={},
            news_data=[],
            technical_indicators={}
        )

        # All symbols should be mentioned
        for symbol in symbols:
            assert symbol in prompt


# ============================================================
# Test 9: Analyze Method (Integration)
# ============================================================

class TestAnalyzeMethod:
    """Test the main analyze() method"""

    def test_analyze_success(self):
        """Test successful analysis with mocked LLM"""
        mock_llm = Mock()

        # Mock LLM response
        llm_response = json.dumps({
            "recommendations": [
                {
                    "symbol": "AAPL",
                    "action": "BUY_50%",
                    "confidence": 0.8,
                    "target_weight": 0.15,
                    "reasoning": "Strong momentum",
                    "trend": "bullish",
                    "risk_level": "medium"
                }
            ],
            "key_factors": ["momentum", "earnings"],
            "market_analysis": "Bullish outlook"
        })

        mock_llm.create_completion = Mock(return_value=llm_response)

        expert = StockExpert(llm_provider=mock_llm, symbols=["AAPL"])

        market_data = {"AAPL": {"close": 150.0, "change_pct": 2.0, "volume": 50000000}}

        report = expert.analyze(market_data=market_data)

        assert isinstance(report, ExpertReport)
        assert report.expert_name == "Stock Expert"
        assert report.asset_class == "stocks"
        assert len(report.recommendations) == 1
        assert report.recommendations[0].symbol == "AAPL"

    def test_analyze_with_all_data(self):
        """Test analyze with all optional parameters"""
        mock_llm = Mock()

        llm_response = json.dumps({
            "recommendations": [
                {
                    "symbol": "SPY",
                    "action": "HOLD",
                    "confidence": 0.7,
                    "target_weight": 0.10,
                    "reasoning": "Consolidation"
                }
            ]
        })

        mock_llm.create_completion = Mock(return_value=llm_response)

        expert = StockExpert(llm_provider=mock_llm, symbols=["SPY"])

        market_data = {"SPY": {"close": 450.0, "change_pct": 0.5, "volume": 80000000}}
        news_data = [{"title": "Market update", "sentiment": "neutral", "symbols": []}]
        technical_indicators = {
            "SPY": {
                "rsi": 50.0,
                "macd": 0.0,
                "macd_signal": 0.0,
                "macd_hist": 0.0,
                "macd_cross": "neutral",
                "ma_20": 445.0,
                "ma_50": 440.0,
                "price": 450.0,
                "trend": "sideways",
                "bb_position": "neutral"
            }
        }

        report = expert.analyze(
            market_data=market_data,
            news_data=news_data,
            technical_indicators=technical_indicators,
            macro_data={"gdp_growth": 2.5}  # Optional parameter
        )

        assert isinstance(report, ExpertReport)
        assert len(report.recommendations) == 1

    def test_analyze_llm_exception(self):
        """Test analyze when LLM raises exception"""
        mock_llm = Mock()
        mock_llm.create_completion = Mock(side_effect=Exception("LLM API Error"))

        expert = StockExpert(llm_provider=mock_llm, symbols=["SPY"])

        market_data = {"SPY": {"close": 450.0, "change_pct": 0.0, "volume": 1000000}}

        with pytest.raises(Exception) as exc_info:
            expert.analyze(market_data=market_data)

        assert "LLM API Error" in str(exc_info.value)

    def test_analyze_system_prompt_called(self):
        """Test that system prompt is properly constructed"""
        mock_llm = Mock()

        llm_response = json.dumps({
            "recommendations": [
                {"symbol": "AAPL", "action": "HOLD", "confidence": 0.5, "target_weight": 0.1}
            ]
        })

        mock_llm.create_completion = Mock(return_value=llm_response)

        expert = StockExpert(llm_provider=mock_llm, symbols=["AAPL"])

        market_data = {"AAPL": {"close": 150.0, "change_pct": 0.0, "volume": 1000000}}

        expert.analyze(market_data=market_data)

        # Verify LLM was called with proper structure
        mock_llm.create_completion.assert_called_once()
        call_args = mock_llm.create_completion.call_args

        messages = call_args[1]["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "stocks" in messages[0]["content"] or "股票" in messages[0]["content"]

    def test_analyze_overall_view_calculation(self):
        """Test that overall view is correctly determined"""
        mock_llm = Mock()

        # Create mostly bullish recommendations
        llm_response = json.dumps({
            "recommendations": [
                {"symbol": "AAPL", "action": "BUY_75%", "confidence": 0.9, "target_weight": 0.2},
                {"symbol": "MSFT", "action": "BUY_50%", "confidence": 0.8, "target_weight": 0.15},
                {"symbol": "GOOGL", "action": "HOLD", "confidence": 0.5, "target_weight": 0.1}
            ]
        })

        mock_llm.create_completion = Mock(return_value=llm_response)

        expert = StockExpert(llm_provider=mock_llm, symbols=["AAPL", "MSFT", "GOOGL"])

        market_data = {
            "AAPL": {"close": 150.0, "change_pct": 2.0, "volume": 50000000},
            "MSFT": {"close": 300.0, "change_pct": 1.5, "volume": 30000000},
            "GOOGL": {"close": 2500.0, "change_pct": 0.5, "volume": 10000000}
        }

        report = expert.analyze(market_data=market_data)

        assert report.overall_view == "bullish"

    def test_analyze_sector_allocation(self):
        """Test that sector allocation is properly calculated"""
        mock_llm = Mock()

        llm_response = json.dumps({
            "recommendations": [
                {"symbol": "AAPL", "action": "BUY_50%", "confidence": 0.8, "target_weight": 0.3},
                {"symbol": "MSFT", "action": "BUY_25%", "confidence": 0.7, "target_weight": 0.2}
            ]
        })

        mock_llm.create_completion = Mock(return_value=llm_response)

        expert = StockExpert(llm_provider=mock_llm, symbols=["AAPL", "MSFT"])

        market_data = {
            "AAPL": {"close": 150.0, "change_pct": 1.0, "volume": 50000000},
            "MSFT": {"close": 300.0, "change_pct": 0.5, "volume": 30000000}
        }

        report = expert.analyze(market_data=market_data)

        # Sector allocation should sum to 1.0
        total = sum(report.sector_allocation.values())
        assert abs(total - 1.0) < 0.001
        assert "AAPL" in report.sector_allocation
        assert "MSFT" in report.sector_allocation


# ============================================================
# Test 10: Edge Cases and Error Handling
# ============================================================

class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_symbols_list(self):
        """Test initialization with empty symbols list - falls back to defaults"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=[])

        # Empty list is falsy, so defaults are used (this is expected behavior)
        assert len(expert.symbols) > 0  # Falls back to default symbols
        assert expert.asset_class == "stocks"

    def test_none_config(self):
        """Test initialization with None config"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["SPY"], config=None)

        # Should use defaults from BaseExpert
        assert expert.max_single_weight == 0.15
        assert expert.min_confidence == 0.5

    def test_format_price_data_with_na_values(self):
        """Test price formatting when volume is N/A"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["SPY"])

        market_data = {
            "SPY": {
                "close": 450.0,
                "change_pct": 1.0,
                "volume": "N/A"
            }
        }

        result = expert._format_price_data(market_data)
        assert "SPY" in result
        assert "N/A" in result

    def test_format_technical_indicators_no_signals(self):
        """Test formatting when no signals are detected"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["SPY"])

        indicators = {
            "SPY": {
                "rsi": 50.0,  # Neutral
                "macd": 0.0,
                "macd_signal": 0.0,
                "macd_hist": 0.0,
                "macd_cross": "neutral",
                "ma_20": 450.0,
                "ma_50": 450.0,
                "price": 450.0,
                "trend": "sideways",
                "bb_position": "neutral"
            }
        }

        result = expert._format_technical_indicators(indicators)

        assert "无明显信号" in result
        assert "做空信号汇总" not in result
        assert "做多信号汇总" not in result

    def test_parse_llm_response_malformed_json(self):
        """Test parsing with malformed JSON (missing brackets)"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["AAPL", "MSFT", "GOOGL"])

        response = '{"recommendations": [{"symbol": "AAPL"'  # Incomplete JSON

        recommendations = expert._parse_llm_response(response)

        # Should fall back to default HOLD recommendations
        assert len(recommendations) == 3
        assert all(rec.action == Action.HOLD for rec in recommendations)

    def test_parse_llm_response_type_error(self):
        """Test handling of TypeError in parsing - raises AttributeError not caught"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["SPY", "QQQ", "IWM"])

        # When recommendations is a string, iterating and calling .get() raises AttributeError
        # which is NOT in the except clause, so it propagates
        response = json.dumps({"recommendations": "not a list"})

        # This will raise AttributeError (str has no .get method)
        with pytest.raises(AttributeError):
            expert._parse_llm_response(response)

    def test_update_symbols_method(self):
        """Test updating symbols dynamically"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["SPY"])

        assert len(expert.symbols) == 1

        new_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
        expert.update_symbols(new_symbols)

        assert len(expert.symbols) == 4
        assert expert.symbols == new_symbols

    def test_analyze_with_none_optional_params(self):
        """Test analyze with None for optional parameters"""
        mock_llm = Mock()

        llm_response = json.dumps({
            "recommendations": [
                {"symbol": "SPY", "action": "HOLD", "confidence": 0.6, "target_weight": 0.1}
            ]
        })

        mock_llm.create_completion = Mock(return_value=llm_response)

        expert = StockExpert(llm_provider=mock_llm, symbols=["SPY"])

        market_data = {"SPY": {"close": 450.0, "change_pct": 0.0, "volume": 1000000}}

        # Pass None explicitly
        report = expert.analyze(
            market_data=market_data,
            news_data=None,
            technical_indicators=None,
            macro_data=None
        )

        assert isinstance(report, ExpertReport)

    def test_large_symbol_list(self):
        """Test with large number of symbols"""
        mock_llm = Mock()

        # Create 50 symbols
        large_symbol_list = [f"STOCK{i}" for i in range(50)]
        expert = StockExpert(llm_provider=mock_llm, symbols=large_symbol_list)

        assert len(expert.symbols) == 50

    def test_special_characters_in_symbol(self):
        """Test handling symbols with special characters"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["BRK.B", "BTC-USD"])

        assert "BRK.B" in expert.symbols
        assert "BTC-USD" in expert.symbols


# ============================================================
# Test 11: Integration with BaseExpert Methods
# ============================================================

class TestBaseExpertIntegration:
    """Test inherited methods from BaseExpert"""

    def test_get_system_prompt(self):
        """Test system prompt generation"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["SPY"])

        system_prompt = expert._get_system_prompt()

        assert isinstance(system_prompt, str)
        assert len(system_prompt) > 0
        assert "stocks" in system_prompt or "股票" in system_prompt
        # Should include expertise areas
        for expertise_item in expert.expertise:
            assert expertise_item in system_prompt

    def test_extract_key_factors(self):
        """Test key factors extraction"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["SPY"])

        response = json.dumps({
            "key_factors": ["momentum", "valuation", "sentiment"]
        })

        factors = expert._extract_key_factors(response)

        assert len(factors) == 3
        assert "momentum" in factors

    def test_extract_key_factors_invalid_json(self):
        """Test key factors extraction with invalid JSON"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["SPY"])

        response = "This is not JSON"

        factors = expert._extract_key_factors(response)

        assert factors == []

    def test_extract_key_factors_missing_key(self):
        """Test key factors extraction when key is missing"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["SPY"])

        response = json.dumps({"other_field": "value"})

        factors = expert._extract_key_factors(response)

        assert factors == []


# ============================================================
# Test 12: Market Condition Scenarios
# ============================================================

class TestMarketScenarios:
    """Test different market condition scenarios"""

    def test_bull_market_scenario(self):
        """Test analysis in strong bull market"""
        mock_llm = Mock()

        # Strong bullish signals
        llm_response = json.dumps({
            "recommendations": [
                {"symbol": "SPY", "action": "BUY_100%", "confidence": 0.95, "target_weight": 0.3},
                {"symbol": "QQQ", "action": "BUY_75%", "confidence": 0.9, "target_weight": 0.25},
                {"symbol": "IWM", "action": "BUY_50%", "confidence": 0.85, "target_weight": 0.2}
            ]
        })

        mock_llm.create_completion = Mock(return_value=llm_response)

        expert = StockExpert(llm_provider=mock_llm, symbols=["SPY", "QQQ", "IWM"])

        market_data = {
            "SPY": {"close": 500.0, "change_pct": 5.0, "volume": 100000000},
            "QQQ": {"close": 400.0, "change_pct": 6.0, "volume": 80000000},
            "IWM": {"close": 200.0, "change_pct": 4.0, "volume": 60000000}
        }

        report = expert.analyze(market_data=market_data)

        assert report.overall_view == "bullish"
        assert all(rec.action.value.startswith("BUY") for rec in report.recommendations)

    def test_bear_market_scenario(self):
        """Test analysis in strong bear market"""
        mock_llm = Mock()

        # Strong bearish signals
        llm_response = json.dumps({
            "recommendations": [
                {"symbol": "SPY", "action": "SHORT_100%", "confidence": 0.9, "target_weight": 0.2},
                {"symbol": "QQQ", "action": "SHORT_75%", "confidence": 0.85, "target_weight": 0.15},
                {"symbol": "IWM", "action": "SELL_100%", "confidence": 0.8, "target_weight": 0.0}
            ]
        })

        mock_llm.create_completion = Mock(return_value=llm_response)

        expert = StockExpert(llm_provider=mock_llm, symbols=["SPY", "QQQ", "IWM"])

        market_data = {
            "SPY": {"close": 400.0, "change_pct": -5.0, "volume": 150000000},
            "QQQ": {"close": 300.0, "change_pct": -6.0, "volume": 120000000},
            "IWM": {"close": 150.0, "change_pct": -7.0, "volume": 100000000}
        }

        report = expert.analyze(market_data=market_data)

        assert report.overall_view == "bearish"

    def test_sideways_market_scenario(self):
        """Test analysis in sideways/neutral market"""
        mock_llm = Mock()

        # Mixed signals
        llm_response = json.dumps({
            "recommendations": [
                {"symbol": "SPY", "action": "HOLD", "confidence": 0.6, "target_weight": 0.15},
                {"symbol": "QQQ", "action": "BUY_25%", "confidence": 0.55, "target_weight": 0.12},
                {"symbol": "IWM", "action": "SELL_25%", "confidence": 0.55, "target_weight": 0.08}
            ]
        })

        mock_llm.create_completion = Mock(return_value=llm_response)

        expert = StockExpert(llm_provider=mock_llm, symbols=["SPY", "QQQ", "IWM"])

        market_data = {
            "SPY": {"close": 450.0, "change_pct": 0.2, "volume": 70000000},
            "QQQ": {"close": 350.0, "change_pct": -0.1, "volume": 60000000},
            "IWM": {"close": 180.0, "change_pct": 0.1, "volume": 50000000}
        }

        report = expert.analyze(market_data=market_data)

        assert report.overall_view == "neutral"

    def test_volatile_market_scenario(self):
        """Test analysis in high volatility market"""
        mock_llm = Mock()
        expert = StockExpert(llm_provider=mock_llm, symbols=["SPY"])

        indicators = {
            "SPY": {
                "rsi": 72.0,  # Overbought
                "macd": 0.5,
                "macd_signal": 0.3,
                "macd_hist": 0.2,
                "macd_cross": "bullish",  # But still bullish cross
                "ma_20": 450.0,
                "ma_50": 440.0,
                "price": 465.0,
                "trend": "uptrend",
                "bb_position": "overbought"  # At upper band
            }
        }

        result = expert._format_technical_indicators(indicators)

        # Should detect conflicting signals
        assert "RSI超买" in result
        assert "MACD金叉" in result
        assert "BB超买" in result


# ============================================================
# Run Tests
# ============================================================

def run_tests():
    """Run all tests"""
    print("\n" + "=" * 70)
    print(" Deep Coverage Tests for StockExpert")
    print(" Target: 100% line coverage of stock_expert.py")
    print("=" * 70)

    pytest.main([__file__, "-v", "--tb=short", "--cov=finsage.agents.experts.stock_expert",
                 "--cov-report=term-missing", "--cov-report=html"])


if __name__ == "__main__":
    run_tests()
