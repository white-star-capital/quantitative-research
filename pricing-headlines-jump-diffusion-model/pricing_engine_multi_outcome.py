#!/usr/bin/env python3
"""
Polymarket Integrated Pricing Engine
Combined Pipeline: Market Monitoring + AI Analysis + Quantitative Pricing
Author: White Star Capital Quantative Strategies
Model: Event-Volatility Diffusion with Bayesian News Integration
"""

import asyncio
import json
import os
import sys
import time
import logging
from datetime import datetime, timezone, date, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import httpx
import numpy as np
from scipy.stats import norm
from dotenv import load_dotenv

# Allow imports of repo-root packages (agent_tools) when run from this subfolder.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent_tools.brave_browser_tool import BraveSearchClient, extract_news_summaries
from ollama_server import OllamaClient
from market_discovery import discover_top_geopolitical_slugs, slugs_only
from json_store import save_results_to_json

# Load environment variables
load_dotenv()

# Configuration
PRICE_INTERVALS = ["1h", "6h", "1d", "1w", "max"]
DEFAULT_INTERVAL = "1d"
OLLAMA_MODEL = "qwen3:8b"
NEWS_FRESHNESS_DAYS = 30

# ==================== MARKET TYPE DETECTION ====================

def is_binary_market(market_data: Dict) -> bool:
    """Detect if market is binary (Yes/No) or multi-outcome."""
    outcomes = market_data.get('outcomes', [])
    if isinstance(outcomes, str):
        try:
            outcomes = json.loads(outcomes)
        except json.JSONDecodeError:
            return False
    
    # Binary if 2 outcomes and contains "Yes"
    if len(outcomes) == 2 and ('Yes' in outcomes or 'No' in outcomes):
        return True
    return False

def get_outcome_list(market_data: Dict) -> List[str]:
    """Extract list of outcome names from market data."""
    outcomes = market_data.get('outcomes', [])
    if isinstance(outcomes, str):
        try:
            outcomes = json.loads(outcomes)
        except json.JSONDecodeError:
            return []
    return outcomes

# ==================== STEP 1: POLYMARKET API CLIENT ====================

class PolymarketMonitorClient:
    """Client for fetching comprehensive market data from Polymarket APIs."""

    GAMMA_API_BASE = "https://gamma-api.polymarket.com"
    CLOB_API_BASE = "https://clob.polymarket.com"

    async def get_market_by_slug(self, slug: str) -> Dict[str, any]:
        """Fetch market data by slug from Gamma API."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.GAMMA_API_BASE}/markets/slug/{slug}")
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                raise httpx.HTTPError(f"Failed to fetch market by slug '{slug}': {e}")

    async def get_event_by_slug(self, slug: str) -> Dict[str, any]:
        """Fetch event data by slug from Gamma API."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.GAMMA_API_BASE}/events/slug/{slug}")
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                raise httpx.HTTPError(f"Failed to fetch event by slug '{slug}': {e}")

    async def get_price_history(
        self, token_id: str, interval: str = DEFAULT_INTERVAL
    ) -> Dict[str, any]:
        """Fetch price history for a token from CLOB API."""
        if interval not in PRICE_INTERVALS:
            raise ValueError(f"Invalid interval: {interval}. Must be one of: {PRICE_INTERVALS}")

        params = {"market": token_id, "interval": interval}

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.CLOB_API_BASE}/prices-history", params=params)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                raise httpx.HTTPError(f"Failed to fetch price history for token '{token_id}': {e}")

    async def get_orderbook(self, token_id: str) -> Dict[str, any]:
        """Fetch current orderbook for a token from CLOB API."""
        params = {"token_id": token_id}

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.CLOB_API_BASE}/book", params=params)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                raise httpx.HTTPError(f"Failed to fetch orderbook for token '{token_id}': {e}")

    async def fetch_market_data(
        self, slug: str, price_interval: str = DEFAULT_INTERVAL
    ) -> Dict[str, any]:
        """Fetch comprehensive market data by combining market info, price history, and orderbook."""
        try:
            market_data = await self.get_market_by_slug(slug)
        except httpx.HTTPError as market_error:
            if "404" in str(market_error):
                print(f"   ℹ️  Not a market slug, trying event endpoint...")
                try:
                    event_data = await self.get_event_by_slug(slug)
                    markets = event_data.get("markets", [])
                    if not markets:
                        raise httpx.HTTPError(f"Event '{slug}' contains no markets")
                    
                    # Check if this is a multi-outcome event (multiple markets, each binary)
                    # Aggregate them into a single multi-outcome market
                    if len(markets) > 2:
                        print(f"   ✓ Found multi-outcome event with {len(markets)} outcomes")
                        
                        # Create aggregated multi-outcome market
                        outcomes = []
                        outcome_prices = []
                        total_volume = 0
                        total_liquidity = 0
                        all_tokens = []
                        
                        for market in markets:
                            # Extract the outcome name from the question
                            # Most questions are like "Will Trump nominate X as the next Fed chair?"
                            question = market.get('question', '')
                            
                            # Try to extract outcome name - look for patterns
                            # This is heuristic and may need adjustment per event type
                            if 'nominate' in question.lower() or 'elect' in question.lower() or 'win' in question.lower():
                                # Extract the name/option being asked about
                                # Try to get it from the question text
                                parts = question.split(' as ')
                                if len(parts) >= 2:
                                    outcome_name = parts[0].split()[-1] if parts[0].split() else market.get('id')
                                else:
                                    # Fallback: use full question minus common words
                                    outcome_name = question.replace('Will Trump nominate ', '').replace(' as the next Fed chair?', '').strip()
                            else:
                                outcome_name = question
                            
                            # Get the "Yes" price for this outcome
                            try:
                                market_outcomes = market.get('outcomes', [])
                                market_prices = market.get('outcomePrices', [])
                                
                                if isinstance(market_outcomes, str):
                                    market_outcomes = json.loads(market_outcomes)
                                if isinstance(market_prices, str):
                                    market_prices = json.loads(market_prices)
                                
                                if 'Yes' in market_outcomes:
                                    yes_idx = market_outcomes.index('Yes')
                                    price = float(market_prices[yes_idx]) if yes_idx < len(market_prices) else 0.0
                                else:
                                    price = 0.0
                            except:
                                price = 0.0
                            
                            outcomes.append(outcome_name)
                            outcome_prices.append(price)
                            total_volume += float(market.get('volume', 0))
                            total_liquidity += float(market.get('liquidity', 0))
                            
                            # Collect tokens from all markets
                            all_tokens.extend(market.get('tokens', []))
                        
                        # Use event-level data where available
                        market_data = {
                            'id': event_data.get('id'),
                            'question': event_data.get('title', markets[0].get('question', 'Unknown')),
                            'description': event_data.get('description', ''),
                            'outcomes': outcomes,
                            'outcomePrices': outcome_prices,
                            'conditionId': markets[0].get('conditionId'),
                            'endDate': event_data.get('endDate', markets[0].get('endDate')),
                            'volume': str(total_volume),
                            'liquidity': str(total_liquidity),
                            'active': event_data.get('active', True),
                            'closed': event_data.get('closed', False),
                            'tokens': all_tokens
                        }
                    else:
                        # Single or dual market event - use first market
                        market_data = markets[0]
                        if len(markets) > 1:
                            print(f"   ℹ️  Event contains {len(markets)} markets, using first: {market_data.get('question', 'Unknown')[:60]}...")
                        else:
                            print(f"   ✓ Found event with 1 market: {market_data.get('question', 'Unknown')[:60]}...")
                except httpx.HTTPError as event_error:
                    raise httpx.HTTPError(f"Failed to fetch '{slug}' as market or event: {event_error}")
            else:
                raise market_error

        tokens = market_data.get("tokens", [])
        
        # Extract current prices from outcomePrices field
        current_prices = {}
        try:
            outcomes_str = market_data.get("outcomes", "[]")
            prices_str = market_data.get("outcomePrices", "[]")
            
            if isinstance(outcomes_str, str):
                outcomes_list = json.loads(outcomes_str)
            else:
                outcomes_list = outcomes_str
                
            if isinstance(prices_str, str):
                prices_list = json.loads(prices_str)
            else:
                prices_list = prices_str
            
            for outcome, price in zip(outcomes_list, prices_list):
                current_prices[outcome] = float(price)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        
        market_data_enriched = {
            "slug": slug,
            "market_id": market_data.get("id"),
            "question": market_data.get("question"),
            "description": market_data.get("description"),
            "outcomes": market_data.get("outcomes", []),
            "current_prices": current_prices,
            "condition_id": market_data.get("conditionId"),
            "end_date": market_data.get("endDate"),
            # Normalize to float; discovered markets sometimes return null here.
            "volume": float(market_data.get("volume") or 0),
            "liquidity": float(market_data.get("liquidity") or 0),
            "active": market_data.get("active"),
            "closed": market_data.get("closed"),
            "tokens": [],
            "fetched_at": datetime.now().isoformat(),
        }

        for token in tokens:
            token_id = token.get("id")
            if not token_id:
                continue

            token_data = {
                "token_id": token_id,
                "price": token.get("price"),
                "outcome": token.get("outcome"),
                "price_history": None,
                "orderbook": None,
            }

            try:
                price_hist = await self.get_price_history(token_id, price_interval)
                token_data["price_history"] = price_hist
            except httpx.HTTPError as e:
                pass

            try:
                orderbook = await self.get_orderbook(token_id)
                token_data["orderbook"] = orderbook
            except httpx.HTTPError as e:
                pass

            market_data_enriched["tokens"].append(token_data)

        return market_data_enriched

# ==================== STEP 2: NEWS & AI ANALYSIS ====================

def extract_key_terms_from_question(question: str) -> List[str]:
    """Extract key terms from a question for search queries."""
    stop_words = {
        'will', 'the', 'be', 'in', 'on', 'at', 'to', 'a', 'an', 'by', 'for', 
        'of', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'has', 'have',
        'do', 'does', 'did', 'can', 'could', 'should', 'would', 'may', 'might',
        'if', 'when', 'how', 'what', 'which', 'who', 'where', 'why', 'any',
        'there', 'than', 'before', 'after', 'between', 'many', 'much', 'more',
        'most', 'some', 'such', 'no', 'nor', 'not', 'only', 'same', 'so',
        'then', 'too', 'very', 'said', 'says', 'get', 'got', 'make', 'made',
        'go', 'goes', 'going', 'different', 'end', 'day', 'year', 'time'
    }
    
    cleaned = question.replace('?', '').replace('"', '').replace('x', '').strip()
    words = cleaned.split()
    
    key_terms = []
    for word in words:
        word_clean = word.strip('.,!:;')
        if not word_clean:
            continue
            
        word_lower = word_clean.lower()
        
        if word_lower not in stop_words:
            if word_clean[0].isupper() or len(word_clean) > 4:
                key_terms.append(word_clean)
    
    return key_terms


def generate_search_queries_for_question(question: str, model: Optional[str] = None) -> List[str]:
    """Generate targeted search queries for a prediction market question."""
    selected_model = model or OLLAMA_MODEL

    prompt = f"""
Given this prediction market question: "{question}"

Generate exactly 2 highly targeted, news-oriented search queries that would help gather relevant information to inform a prediction about this question. The queries should:

1. Focus on recent developments (past 30 days)
2. Include specific names, events, or topics mentioned in the question
3. Use neutral, factual search terms
4. Optimize for quality news sources
5. Avoid overly broad or generic terms
6. Do NOT wrap the entire question in quotes - extract key terms instead

Output only JSON in this format: {{"queries": ["query1", "query2"]}}
"""

    messages = [{"role": "user", "content": prompt}]

    try:
        with OllamaClient() as client:
            resp = client.create_chat_completion(
                model=selected_model,
                messages=messages,
                temperature=0.3,
            )
        content = (
            resp.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        try:
            parsed = json.loads(content)
            queries = parsed.get("queries", [])
            if queries:
                return queries[:2]
        except json.JSONDecodeError:
            pass

        lines = [line.strip() for line in content.split('\n') if line.strip()]
        candidates = []
        for line in lines:
            if len(line) > 10 and not any(skip in line.lower() for skip in ['output', 'json', 'queries']):
                candidates.append(line)
        if candidates:
            return candidates[:2]
        
        key_terms = extract_key_terms_from_question(question)
        return [' '.join(key_terms[:5]) + ' news']

    except Exception as e:
        print(f"⚠️  OpenRouter API error: {e}")
        print("📝 Using fallback queries with extracted key terms...")
        
        key_terms = extract_key_terms_from_question(question)
        if len(key_terms) >= 3:
            return [
                ' '.join(key_terms[:4]) + ' news',
                ' '.join(key_terms[:4]) + ' latest updates'
            ]
        else:
            return [' '.join(key_terms) + ' news']


def search_news_for_question(
    question: str,
    brave_api_key: Optional[str] = None,
    days_freshness: int = NEWS_FRESHNESS_DAYS
) -> Dict[str, any]:
    """Search for news related to a specific question."""
    queries = generate_search_queries_for_question(question)
    print(f"🔍 Generated queries: {queries}")

    brave = BraveSearchClient(api_key=brave_api_key)
    aggregated_results = []

    try:
        end_date = date.today()
        start_date = max(date(1970, 1, 1), end_date - timedelta(days=max(1, days_freshness)))
        freshness = f"{start_date.isoformat()}to{end_date.isoformat()}"
    except Exception:
        freshness = ""

    for i, query in enumerate(queries):
        try:
            raw = brave.search_news(
                query,
                count=5,
                country="us",
                search_lang="en",
                spellcheck=1,
                freshness=freshness if freshness else None,
            )
            items = extract_news_summaries(raw, limit=5)
            print(f"✓ Found {len(items)} news items for query: '{query[:50]}...'")
        except Exception as e:
            print(f"⚠️  Error during Brave search for query '{query}': {e}")
            items = []
        aggregated_results.append({"query": query, "items": items})

        if i < len(queries) - 1:
            time.sleep(1)

    return {
        "question": question,
        "search_queries": queries,
        "news_results": aggregated_results,
        "total_news_items": sum(len(result["items"]) for result in aggregated_results)
    }


def analyze_question_with_news(
    question_data: Dict[str, any],
    outcomes: Optional[List[str]] = None,
    model: Optional[str] = None
) -> Dict[str, any]:
    """Use OpenRouter to analyze news results and provide an informed answer.
    
    Args:
        question_data: Dict containing question and news_results
        outcomes: Optional list of outcome names for multi-outcome markets
        model: Optional AI model to use
    
    Returns:
        Analysis dict with prediction(s) and confidence
    """
    selected_model = model or OLLAMA_MODEL

    question = question_data["question"]
    news_results = question_data["news_results"]

    news_summaries = []
    for result in news_results:
        query = result["query"]
        items = result["items"]
        for item in items:
            if item.get("title") and item.get("description"):
                news_summaries.append({
                    "query": query,
                    "title": item["title"],
                    "description": item["description"],
                    "source": item.get("source", "Unknown"),
                    "published": item.get("published", "Unknown")
                })

    news_summaries = news_summaries[:10]

    if not news_summaries:
        base_response = {
            "question": question,
            "analysis": "No relevant news found to analyze.",
            "prediction": "Unable to make prediction due to lack of information.",
            "confidence": "Low",
            "key_factors": [],
            "news_articles_analyzed": 0
        }
        if outcomes:
            # For multi-outcome, return uniform probabilities
            base_response["outcome_probabilities"] = {
                outcome: 1.0 / len(outcomes) for outcome in outcomes
            }
        return base_response

    news_text = "\n\n".join([
        f"Source: {item['source']} ({item['published']})\nTitle: {item['title']}\nSummary: {item['description']}"
        for item in news_summaries
    ])

    # Different prompts for binary vs multi-outcome markets
    if outcomes:
        # Multi-outcome market prompt
        outcomes_display = ', '.join(outcomes[:15])  # Limit to top 15 for context
        if len(outcomes) > 15:
            outcomes_display += f" ... and {len(outcomes) - 15} more"
        
        prompt = f"""
You are an expert analyst evaluating prediction market questions based on recent news and developments.

Question: "{question}"

Possible Outcomes: {outcomes_display}

Recent News Articles ({len(news_summaries)} articles):
{news_text}

Based on this news information, please provide:

1. A brief analysis of the current situation and key developments
2. Probability estimates for each outcome (values between 0.0 and 1.0, they don't need to sum to 1.0 as these are independent binary evaluations)
3. Your confidence level (High/Medium/Low) in your analysis
4. Key factors that could influence the outcome

For outcomes with very low probability (<0.01), you may omit them from outcome_probabilities.

Be objective, factual, and cite specific news sources where relevant. If the news is inconclusive, state this clearly.

Output your response in JSON format:
{{
    "analysis": "brief analysis text",
    "outcome_probabilities": {{"outcome1": 0.35, "outcome2": 0.15, "outcome3": 0.05}},
    "confidence": "High/Medium/Low",
    "key_factors": ["factor1", "factor2", ...]
}}
"""
    else:
        # Binary market prompt (original)
        prompt = f"""
You are an expert analyst evaluating prediction market questions based on recent news and developments.

Question: "{question}"

Recent News Articles ({len(news_summaries)} articles):
{news_text}

Based on this news information, please provide:

1. A brief analysis of the current situation and key developments
2. Your prediction for how this question will resolve (Yes/No format where applicable)
3. Your confidence level (High/Medium/Low) and reasoning
4. Key factors that could influence the outcome

Be objective, factual, and cite specific news sources where relevant. If the news is inconclusive, state this clearly.

Output your response in JSON format:
{{
    "analysis": "brief analysis text",
    "prediction": "Yes/No or other prediction",
    "confidence": "High/Medium/Low",
    "key_factors": ["factor1", "factor2", ...]
}}
"""

    messages = [{"role": "user", "content": prompt}]

    try:
        with OllamaClient() as client:
            resp = client.create_chat_completion(
                model=selected_model,
                messages=messages,
                temperature=0.2,
            )

        content = (
            resp.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        try:
            analysis = json.loads(content)
            analysis["question"] = question
            analysis["news_articles_analyzed"] = len(news_summaries)
            return analysis
        except json.JSONDecodeError:
            import re
            json_pattern = r'```(?:json)?\s*(\{[\s\S]*?\})\s*```'
            json_match = re.search(json_pattern, content, re.IGNORECASE)

            if json_match:
                try:
                    analysis = json.loads(json_match.group(1))
                    analysis["question"] = question
                    analysis["news_articles_analyzed"] = len(news_summaries)
                    return analysis
                except json.JSONDecodeError:
                    pass

            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                try:
                    potential_json = content[json_start:json_end]
                    analysis = json.loads(potential_json)
                    analysis["question"] = question
                    analysis["news_articles_analyzed"] = len(news_summaries)
                    return analysis
                except json.JSONDecodeError:
                    pass

            return {
                "question": question,
                "analysis": f"Error parsing analysis response.",
                "prediction": "Unable to determine",
                "confidence": "Unknown",
                "key_factors": [],
                "news_articles_analyzed": len(news_summaries)
            }

    except Exception as e:
        print(f"⚠️  OpenRouter API error during analysis: {e}")
        return {
            "question": question,
            "analysis": f"Unable to analyze due to API error: {e}",
            "prediction": "Unable to determine",
            "confidence": "Unknown",
            "key_factors": [],
            "news_articles_analyzed": len(news_summaries)
        }

# ==================== STEP 3: MARKET ASSESSMENT ====================

class MarketRegime(Enum):
    """Regime classification for prediction markets"""
    LOW_VOL_RANGE = "low_volatility_range_bound"
    HIGH_VOL_TRANSITIONAL = "high_volatility_transitional"
    PRE_CATALYST = "pre_catalyst_event"
    POST_CATALYST = "post_catalyst_resolution"
    ILLIQUID_TAIL = "illiquid_tail_risk"
    DEEP_LIQUIDITY = "deep_liquidity_institutional"

@dataclass
class MarketMetrics:
    """Core market metrics for regime classification"""
    market_id: str
    slug: str
    time_to_expiry_days: float
    liquidity_usd: float
    volume_24h: float
    bid_ask_spread: float
    implied_volatility: float = field(init=False)
    liquidity_score: float = field(init=False)
    volatility_score: float = field(init=False)
    
    def __post_init__(self):
        if self.time_to_expiry_days > 0:
            self.implied_volatility = self.bid_ask_spread / (2 * np.sqrt(self.time_to_expiry_days / 365)) * 100
        else:
            self.implied_volatility = 999.0
            
        self.liquidity_score = min(100, (self.liquidity_usd / 10000) * np.log1p(self.volume_24h / 100000))
        self.volatility_score = min(100, self.implied_volatility * 10)


def estimate_spread_from_liquidity(liquidity: float, volume: float, tte_days: float) -> float:
    """Empirical spread model based on Polymarket liquidity data."""
    k = 75000
    alpha = 0.55
    beta = 0.65
    gamma = 150
    delta = 0.25
    
    spread_bps = alpha * (k / max(liquidity, 1000)) ** beta + gamma * (1 / max(volume, 10000)) ** delta
    
    if tte_days < 7:
        spread_bps *= 2.5
    elif tte_days < 30:
        spread_bps *= 1.5
    
    return spread_bps / 100


class MarketAssessor:
    """Regime classification engine"""
    
    def __init__(self):
        self.regime_thresholds = {
            'liquidity_deep': 75,
            'liquidity_illiquid': 25,
            'vol_low': 30,
            'vol_high': 70,
            'days_short': 30,
            'days_medium': 90
        }
    
    def classify_regime(self, metrics: MarketMetrics) -> MarketRegime:
        """Classify market into regime"""
        if metrics.liquidity_score > self.regime_thresholds['liquidity_deep']:
            if metrics.volatility_score < self.regime_thresholds['vol_low']:
                return MarketRegime.DEEP_LIQUIDITY
            else:
                return MarketRegime.HIGH_VOL_TRANSITIONAL
        
        if metrics.liquidity_score < self.regime_thresholds['liquidity_illiquid']:
            return MarketRegime.ILLIQUID_TAIL
        
        if (metrics.time_to_expiry_days < self.regime_thresholds['days_medium'] and 
            metrics.volatility_score > self.regime_thresholds['vol_high']):
            return MarketRegime.PRE_CATALYST
        
        return MarketRegime.HIGH_VOL_TRANSITIONAL
    
    def assess_market(self, market_data: Dict) -> Tuple[MarketMetrics, MarketRegime]:
        """Full assessment pipeline"""
        expiry = datetime.fromisoformat(market_data['end_date'].replace('Z', '+00:00'))
        now = datetime.fromisoformat(market_data['fetched_at']).replace(tzinfo=timezone.utc)
        tte = max(1, (expiry - now).days)
        
        liquidity = float(market_data['liquidity'])
        volume = float(market_data['volume'])
        
        spread = estimate_spread_from_liquidity(liquidity, volume, tte)
        
        metrics = MarketMetrics(
            market_id=market_data['market_id'],
            slug=market_data['slug'],
            time_to_expiry_days=tte,
            liquidity_usd=liquidity,
            volume_24h=volume,
            bid_ask_spread=spread
        )
        
        regime = self.classify_regime(metrics)
        
        logging.info(f"[ASSESSMENT] {metrics.slug}")
        logging.info(f"  Regime: {regime.value}")
        logging.info(f"  TTE: {tte:.1f} days | Liquidity: ${metrics.liquidity_usd:,.0f} | Score: {metrics.liquidity_score:.1f}")
        logging.info(f"  Spread: {spread:.2f}¢ | IV: {metrics.implied_volatility:.2f}%")
        
        return metrics, regime

# ==================== STEP 4: PRICING FRAMEWORK ====================

@dataclass
class ProbabilityDiffusionState:
    """Probability diffusion process state"""
    prior_probability: float
    posterior_probability: float
    news_intensity_lambda: float
    jump_mean: float
    jump_std: float
    theta_decay: float
    drift_rate: float
    
    def simulate_path(self, days: int, n_paths: int = 1000) -> np.ndarray:
        """Monte Carlo simulation of probability paths"""
        dt = 1/365
        paths = np.zeros((n_paths, days + 1))
        paths[:, 0] = self.posterior_probability
        
        for t in range(1, days + 1):
            dW = np.random.normal(0, np.sqrt(dt), n_paths)
            diffusion = self.drift_rate * dt + self.jump_std * dW
            
            jumps = np.random.poisson(self.news_intensity_lambda * dt, n_paths)
            jump_sizes = np.random.normal(self.jump_mean, self.jump_std, n_paths)
            
            decay = -self.theta_decay * dt
            
            paths[:, t] = paths[:, t-1] + diffusion + jumps * jump_sizes + decay
            paths[:, t] = np.clip(paths[:, t], 0.001, 0.999)
        
        return paths

class StrategyEngine:
    """Probability-diffusion modeling"""
    
    def __init__(self):
        self.base_rates = {
            'geopolitical_invasion': 0.03,
            'war_ceasefire': 0.35,
        }
        
    def extract_ai_signal(self, analysis_data: Dict) -> Tuple[float, float]:
        """Convert AI analysis into quantitative prior"""
        confidence_map = {'Low': 0.1, 'Medium': 0.25, 'High': 0.5}
        confidence = confidence_map.get(analysis_data['confidence'], 0.2)
        
        prediction = 1.0 if analysis_data['prediction'] == 'Yes' else 0.0
        
        prior = (1 - confidence) * 0.5 + confidence * prediction
        
        return prior, confidence
    
    def build_diffusion_model(self, market_data: Dict, regime: MarketRegime) -> ProbabilityDiffusionState:
        """Construct diffusion model"""
        ai_prior, confidence = self.extract_ai_signal(market_data['analysis'])
        
        tte = (datetime.fromisoformat(market_data['end_date'].replace('Z', '+00:00')) - 
               datetime.fromisoformat(market_data['fetched_at']).replace(tzinfo=timezone.utc)).days
        
        if regime == MarketRegime.ILLIQUID_TAIL:
            lambda_news = 0.08
            jump_vol = 0.35
            theta = 0.005
        elif regime == MarketRegime.DEEP_LIQUIDITY:
            lambda_news = 0.25
            jump_vol = 0.15
            theta = 0.002
        else:
            lambda_news = 0.15
            jump_vol = 0.25
            theta = 0.003
        
        market_price = market_data['current_prices']['Yes']
        
        liquidity_score = min(100, float(market_data['liquidity']) / 10000)
        weight_ai = 0.7 * (1 - liquidity_score / 100)
        weight_market = 1 - weight_ai
        
        posterior = weight_ai * ai_prior + weight_market * market_price
        
        return ProbabilityDiffusionState(
            prior_probability=ai_prior,
            posterior_probability=posterior,
            news_intensity_lambda=lambda_news,
            jump_mean=0.01,
            jump_std=jump_vol,
            theta_decay=theta,
            drift_rate=0.0 if tte > 180 else -0.001
        )
    
    def build_diffusion_model_for_outcome(
        self, 
        market_data: Dict, 
        regime: MarketRegime,
        outcome_analysis: Dict,
        market_price: float
    ) -> ProbabilityDiffusionState:
        """Build diffusion model for single outcome in multi-outcome market.
        
        Args:
            market_data: Full market data dict
            regime: Market regime classification
            outcome_analysis: Dict with 'probability' and 'confidence' for this outcome
            market_price: Current market price for this specific outcome
        
        Returns:
            ProbabilityDiffusionState for this outcome
        """
        ai_prior = outcome_analysis['probability']
        confidence_map = {'Low': 0.1, 'Medium': 0.25, 'High': 0.5}
        confidence = confidence_map.get(outcome_analysis.get('confidence', 'Medium'), 0.2)
        
        tte = (datetime.fromisoformat(market_data['end_date'].replace('Z', '+00:00')) - 
               datetime.fromisoformat(market_data['fetched_at']).replace(tzinfo=timezone.utc)).days
        
        # Regime-specific parameters
        if regime == MarketRegime.ILLIQUID_TAIL:
            lambda_news = 0.08
            jump_vol = 0.35
            theta = 0.005
        elif regime == MarketRegime.DEEP_LIQUIDITY:
            lambda_news = 0.25
            jump_vol = 0.15
            theta = 0.002
        else:
            lambda_news = 0.15
            jump_vol = 0.25
            theta = 0.003
        
        # Blend AI prior with market price based on liquidity
        liquidity_score = min(100, float(market_data['liquidity']) / 10000)
        weight_ai = 0.7 * (1 - liquidity_score / 100)
        weight_market = 1 - weight_ai
        
        posterior = weight_ai * ai_prior + weight_market * market_price
        
        return ProbabilityDiffusionState(
            prior_probability=ai_prior,
            posterior_probability=posterior,
            news_intensity_lambda=lambda_news,
            jump_mean=0.01,
            jump_std=jump_vol,
            theta_decay=theta,
            drift_rate=0.0 if tte > 180 else -0.001
        )

@dataclass
class PricingResult:
    """Container for pricing model output"""
    fair_probability: float
    edge_percentage: float
    kelly_fraction: float
    expected_value: float
    risk_adjusted_return: float
    model_parameters: Dict[str, float]
    
    @property
    def recommendation(self) -> str:
        if self.edge_percentage > 3.0:
            return "STRONG_SIGNAL"
        elif self.edge_percentage > 1.5:
            return "MODERATE_SIGNAL"
        elif self.edge_percentage > 0.5:
            return "WEAK_SIGNAL"
        else:
            return "NO_TRADE"

class JumpDiffusionPricingModel:
    """Institutional-grade pricing model"""
    
    def __init__(self, risk_free_rate: float = 0.04, slippage_per_k: float = 0.001):
        self.risk_free_rate = risk_free_rate
        self.slippage_per_k = slippage_per_k
        
    def calculate_fair_probability(
        self, 
        diffusion_state: ProbabilityDiffusionState,
        market_metrics: MarketMetrics,
        regime: MarketRegime
    ) -> float:
        """Core pricing algorithm"""
        tte = market_metrics.time_to_expiry_days / 365
        if tte > 0:
            time_component = diffusion_state.posterior_probability * np.exp(
                -diffusion_state.theta_decay * np.sqrt(tte)
            )
        else:
            time_component = diffusion_state.posterior_probability
        
        jump_premium = diffusion_state.news_intensity_lambda * diffusion_state.jump_std ** 2
        
        liquidity_discount = max(0.01, 100 - market_metrics.liquidity_score) / 1000
        
        regime_multiplier = {
            MarketRegime.ILLIQUID_TAIL: 0.85,
            MarketRegime.DEEP_LIQUIDITY: 1.02,
            MarketRegime.HIGH_VOL_TRANSITIONAL: 0.95,
            MarketRegime.PRE_CATALYST: 0.90
        }.get(regime, 1.0)
        
        fair_prob = (time_component + jump_premium - liquidity_discount) * regime_multiplier
        
        return max(0.001, min(0.999, fair_prob))
    
    def calculate_kelly_fraction(
        self, 
        fair_prob: float, 
        market_prob: float,
        confidence: float = 0.25
    ) -> float:
        """Kelly criterion with fractional adjustment"""
        # Guard against degenerate prices (0 or 1) that break the odds division.
        market_prob = min(max(market_prob, 1e-6), 1 - 1e-6)
        edge = fair_prob - market_prob

        if market_prob < fair_prob:
            odds = (1 / market_prob) - 1
            kelly = (odds * fair_prob - (1 - fair_prob)) / odds if odds > 0 else 0
        else:
            market_prob_no = 1 - market_prob
            fair_prob_no = 1 - fair_prob
            odds = (1 / market_prob_no) - 1
            kelly = (odds * fair_prob_no - (1 - fair_prob_no)) / odds if odds > 0 else 0
        
        kelly *= confidence
        kelly *= 0.25
        
        return max(0.0, kelly)
    
    def price_market(
        self, 
        market_data: Dict,
        metrics: MarketMetrics,
        regime: MarketRegime,
        diffusion: ProbabilityDiffusionState
    ) -> PricingResult:
        """Full pricing pipeline"""
        market_price = market_data['current_prices']['Yes']
        
        fair_prob = self.calculate_fair_probability(diffusion, metrics, regime)
        
        edge = fair_prob - market_price
        
        _, confidence = StrategyEngine().extract_ai_signal(market_data['analysis'])
        kelly_frac = self.calculate_kelly_fraction(fair_prob, market_price, confidence)
        
        expected_value = edge * 100
        risk_adj_return = expected_value / (diffusion.jump_std * 100) if diffusion.jump_std > 0 else 0
        
        return PricingResult(
            fair_probability=fair_prob,
            edge_percentage=edge * 100,
            kelly_fraction=kelly_frac,
            expected_value=expected_value,
            risk_adjusted_return=risk_adj_return,
            model_parameters={
                'tte_days': metrics.time_to_expiry_days,
                'liquidity_score': metrics.liquidity_score,
                'volatility_score': metrics.volatility_score,
                'news_lambda': diffusion.news_intensity_lambda,
                'jump_std': diffusion.jump_std,
                'regime': regime.value
            }
        )

# ==================== MULTI-OUTCOME PRICING ====================

def price_multi_outcome_market(
    market_data: Dict,
    metrics: MarketMetrics,
    regime: MarketRegime,
    outcomes: List[str],
    strategist: StrategyEngine,
    pricer: JumpDiffusionPricingModel
) -> List[Dict]:
    """Price each outcome as independent binary event.
    
    Args:
        market_data: Full market data including analysis
        metrics: Market metrics for regime classification
        regime: Classified market regime
        outcomes: List of outcome names
        strategist: StrategyEngine instance
        pricer: JumpDiffusionPricingModel instance
    
    Returns:
        List of dicts with pricing info for each outcome, sorted by absolute edge
    """
    outcome_results = []
    ai_probs = market_data['analysis'].get('outcome_probabilities', {})
    
    for outcome in outcomes:
        # Get market price for this outcome
        market_price = market_data['current_prices'].get(outcome, 0.0)
        
        # Skip outcomes with zero market price (no liquidity)
        if market_price <= 0.0:
            continue
        
        # Get AI probability estimate (or use uniform if not provided)
        ai_prob = ai_probs.get(outcome, 1.0 / len(outcomes))
        
        # Create temporary analysis dict for this outcome
        outcome_analysis = {
            'prediction': outcome,
            'confidence': market_data['analysis'].get('confidence', 'Medium'),
            'probability': ai_prob
        }
        
        # Build diffusion model treating this outcome as binary event
        diffusion = strategist.build_diffusion_model_for_outcome(
            market_data, regime, outcome_analysis, market_price
        )
        
        # Calculate fair price and edge
        fair_prob = pricer.calculate_fair_probability(diffusion, metrics, regime)
        edge = fair_prob - market_price
        
        # Calculate Kelly fraction
        confidence_map = {'Low': 0.1, 'Medium': 0.25, 'High': 0.5}
        confidence = confidence_map.get(outcome_analysis['confidence'], 0.25)
        kelly_frac = pricer.calculate_kelly_fraction(fair_prob, market_price, confidence)
        
        # Expected value and risk-adjusted return
        expected_value = edge * 100
        risk_adj_return = expected_value / (diffusion.jump_std * 100) if diffusion.jump_std > 0 else 0
        
        outcome_results.append({
            'outcome': outcome,
            'market_price': float(f"{market_price:.3f}"),
            'ai_probability': float(f"{ai_prob:.3f}"),
            'fair_price': float(f"{fair_prob:.3f}"),
            'edge_percentage': float(f"{edge * 100:.2f}"),
            'edge_bps': float(f"{edge * 10000:.1f}"),
            'kelly_fraction': float(f"{kelly_frac:.3f}"),
            'expected_value': float(f"{expected_value:.2f}"),
            'risk_adjusted_return': float(f"{risk_adj_return:.2f}"),
            'recommendation': 'BUY' if edge > 0.01 else 'SELL' if edge < -0.01 else 'PASS'
        })
    
    # Sort by absolute edge (highest opportunities first)
    outcome_results.sort(key=lambda x: abs(x['edge_percentage']), reverse=True)
    
    return outcome_results

# ==================== UNIFIED PIPELINE ====================

async def run_pricing_engine(market_slugs: List[str]) -> List[Dict]:
    """Complete pipeline: Fetch -> Analyze -> Price -> Results"""
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    client = PolymarketMonitorClient()
    assessor = MarketAssessor()
    strategist = StrategyEngine()
    pricer = JumpDiffusionPricingModel()
    
    results = []
    
    for i, slug in enumerate(market_slugs, 1):
        try:
            print(f"\n{'=' * 80}")
            print(f"⏳ [{i}/{len(market_slugs)}] Processing market: {slug}")
            print(f"{'=' * 80}")
            
            # Step 1: Fetch market data
            print("\n📊 Step 1: Fetching market data...")
            market_data = await client.fetch_market_data(slug, DEFAULT_INTERVAL)
            print(f"✓ Market data fetched")
            
            # Detect market type
            is_binary = is_binary_market(market_data)
            outcomes = get_outcome_list(market_data)
            
            if is_binary:
                print(f"  Market Type: BINARY (Yes/No)")
            else:
                print(f"  Market Type: MULTI-OUTCOME ({len(outcomes)} outcomes)")
            
            # Step 2: News search + AI analysis
            print("\n📰 Step 2: Searching for relevant news...")
            question = market_data['question']
            news_data = search_news_for_question(question)
            print(f"✓ Found {news_data['total_news_items']} news items")
            
            if news_data['total_news_items'] > 0:
                print("\n🤖 Step 3: Analyzing with AI...")
                if is_binary:
                    market_data['analysis'] = analyze_question_with_news(news_data)
                else:
                    market_data['analysis'] = analyze_question_with_news(news_data, outcomes=outcomes)
                print(f"✓ AI analysis complete")
            else:
                print("\n⚠️  No news found for analysis")
                market_data['analysis'] = {
                    "prediction": "Insufficient data",
                    "confidence": "Low",
                    "analysis": "No recent news found",
                    "key_factors": [],
                    "news_articles_analyzed": 0,
                    "question": question
                }
                if not is_binary:
                    market_data['analysis']['outcome_probabilities'] = {
                        outcome: 1.0 / len(outcomes) for outcome in outcomes
                    }
            
            # Step 3: Quantitative pricing
            print("\n💰 Step 4: Running pricing model...")
            metrics, regime = assessor.assess_market(market_data)
            
            if is_binary:
                # Binary market pricing (original path)
                diffusion = strategist.build_diffusion_model(market_data, regime)
                pricing = pricer.price_market(market_data, metrics, regime, diffusion)
                
                result = {
                    **market_data,
                    'market_type': 'binary',
                    'pricing': {
                        'regime': regime.value,
                        'fair_price_yes': float(f"{pricing.fair_probability:.3f}"),
                        'edge_bps': float(f"{pricing.edge_percentage * 100:.1f}"),
                        'kelly_fraction': float(f"{pricing.kelly_fraction:.3f}"),
                        'recommendation': pricing.recommendation,
                        'risk_adjusted_return': float(f"{pricing.risk_adjusted_return:.2f}"),
                        'model_params': pricing.model_parameters,
                        'monte_carlo_10d': float(f"{np.mean(diffusion.simulate_path(10)[:, -1]):.3f}") if metrics.time_to_expiry_days >= 10 else None
                    }
                }
                
                print(f"✓ Pricing complete")
                print(f"  Recommendation: {pricing.recommendation} | Fair: ${pricing.fair_probability:.3f} | Edge: {pricing.edge_percentage*100:.1f}bps")
            else:
                # Multi-outcome market pricing (new path)
                outcome_pricings = price_multi_outcome_market(
                    market_data, metrics, regime, outcomes, strategist, pricer
                )
                
                # Get top opportunities (positive edge > 0.5%)
                top_opportunities = [
                    o for o in outcome_pricings 
                    if o['edge_percentage'] > 0.5
                ][:5]
                
                result = {
                    **market_data,
                    'market_type': 'multi_outcome',
                    'outcome_count': len(outcomes),
                    'regime': regime.value,
                    'outcome_pricings': outcome_pricings,
                    'top_opportunities': top_opportunities
                }
                
                print(f"✓ Pricing complete")
                print(f"  Analyzed {len(outcome_pricings)} outcomes with liquidity")
                if top_opportunities:
                    print(f"  Top Opportunity: {top_opportunities[0]['outcome'][:40]}")
                    print(f"    Market: {top_opportunities[0]['market_price']:.1%} | "
                          f"Fair: {top_opportunities[0]['fair_price']:.1%} | "
                          f"Edge: {top_opportunities[0]['edge_percentage']:.1f}%")
                else:
                    print(f"  No significant opportunities found (edge < 0.5%)")
            
            results.append(result)
            
            if i < len(market_slugs):
                print(f"\n⏱️  Waiting 3 seconds before next market...")
                time.sleep(3)
                
        except Exception as e:
            print(f"❌ Failed to process market '{slug}': {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    print("🎯 Polymarket Integrated Pricing Engine")
    print("=" * 80)
    print(f"🤖 AI Model: {OLLAMA_MODEL} (local Ollama)")
    print(f"📰 News Freshness: {NEWS_FRESHNESS_DAYS} days")
    print("=" * 80)

    async def _main():
        print("\n🔎 Discovering top geopolitical markets by 24h volume...")
        discovered = await discover_top_geopolitical_slugs(top_n=10)
        slugs = slugs_only(discovered)
        for i, m in enumerate(discovered, 1):
            print(f"  {i:>2}. v24h=${m['volume24hr']:>12,.0f} | {m['slug']}")
        if not slugs:
            print("❌ No markets discovered; aborting.")
            return []
        return await run_pricing_engine(slugs)

    results = asyncio.run(_main())

    # Save to local JSON
    if results:
        print("\n" + "=" * 80)
        print("💾 Saving results to JSON...")
        print("=" * 80)

        try:
            filepath = save_results_to_json(results)
            print(f"✅ Successfully saved {len(results)} results to {filepath}")
        except Exception as e:
            print(f"❌ Error saving to JSON: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    print("\n" + "=" * 80)
    print("📊 PRICING ENGINE SUMMARY")
    print("=" * 80)
    for result in results:
        print(f"\n{result['slug']}")
        print(f"  Question: {result['question'][:60]}...")
        
        if result.get('market_type') == 'binary':
            # Binary market summary
            print(f"  Market Type: Binary (Yes/No)")
            print(f"  Current Price (Yes): ${result['current_prices']['Yes']:.3f}")
            pricing = result.get('pricing', {})
            print(f"  Regime: {pricing.get('regime', 'Unknown')}")
            print(f"  Fair Price: ${pricing.get('fair_price_yes', 'N/A'):.3f}")
            print(f"  Edge: {pricing.get('edge_bps', 'N/A')}bps")
            print(f"  Signal: {pricing.get('recommendation', 'N/A')}")
        else:
            # Multi-outcome market summary
            print(f"  Market Type: Multi-Outcome ({result.get('outcome_count', 0)} outcomes)")
            print(f"  Regime: {result.get('regime', 'Unknown')}")
            
            top_opps = result.get('top_opportunities', [])
            if top_opps:
                print(f"  Top Opportunities ({len(top_opps)} with edge > 0.5%):")
                for i, opp in enumerate(top_opps[:5], 1):
                    outcome_name = opp['outcome'][:35]
                    if len(opp['outcome']) > 35:
                        outcome_name += "..."
                    print(f"    {i}. {outcome_name}")
                    print(f"       Market: {opp['market_price']:.1%} | "
                          f"Fair: {opp['fair_price']:.1%} | "
                          f"Edge: {opp['edge_percentage']:+.1f}% | "
                          f"Signal: {opp['recommendation']}")
            else:
                print(f"  No significant opportunities found (all edges < 0.5%)")

