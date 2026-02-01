"""
LangSearch Worker - Real Implementation

Researches terms and gathers contextual information using the LangSearch API.

Supports two modes:
- Script mode: Extracts key terms from a script and researches each one
- Direct query mode: Searches a specific term/query directly

Auto-detected based on input length (>100 chars = script mode).
"""

import re
import time
import logging
import requests
from typing import Dict, List, Any
from datetime import datetime
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LangSearch:
    """
    LangSearch researches terms using the LangSearch API.

    Two modes:
    1. Script mode - takes a script, extracts key terms, researches each
    2. Direct query mode - takes a search term/query and researches it directly

    Auto-detects mode based on input length (>100 chars = script mode).
    """

    STOP_WORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'shall', 'can', 'need', 'dare',
        'this', 'that', 'these', 'those', 'it', 'its', 'not', 'no', 'so',
        'if', 'then', 'than', 'too', 'very', 'just', 'about', 'up', 'out',
        'we', 'they', 'he', 'she', 'you', 'me', 'him', 'her', 'us', 'them',
        'my', 'your', 'his', 'our', 'their', 'what', 'which', 'who', 'whom',
        'how', 'when', 'where', 'why', 'all', 'each', 'every', 'both',
        'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own',
        'same', 'as', 'also', 'into', 'through', 'during', 'before', 'after',
        'above', 'below', 'between', 'under', 'again', 'further', 'once',
        'here', 'there', 'now', 'let', 'get', 'got', 'like', 'know',
        'hook', 'body', 'call', 'action', 'watch', 'video', 'subscribe',
        'comment', 'thanks', 'watching', 'forget', "don't", "doesn't",
        'enjoy', 'enjoyed', 'today', 'really', 'actually', 'basically',
        'something', 'everything', 'anything', 'nothing', 'someone',
        'everyone', 'anyone', 'much', 'many', 'been', 'going', 'come',
        'coming', 'take', 'make', 'made', 'one', 'two', 'new', 'way',
        'time', 'day', 'year', 'people', 'back', 'thing', 'things'
    }

    SCRIPT_MARKERS = ['[HOOK]', '[BODY]', '[CALL TO ACTION]', '[END OF MOCK SCRIPT]']

    SCRIPT_THRESHOLD = 100
    MAX_TERMS = 5

    def __init__(self, api_key: str = "sk-e0235f795d7f4237aaf5048e6816af6b"):
        self.api_key = api_key
        self.endpoint = "https://api.langsearch.com/v1/web-search"
        self.call_count = 0

        logger.info("LangSearch initialized:")
        logger.info(f"  - Endpoint: {self.endpoint}")

    def research_terms(self, input_text: str) -> Dict[str, Any]:
        """
        Main entry point. Researches terms from a script or a direct query.

        Auto-detects mode:
        - Input > 100 chars → Script mode (extract terms, then research each)
        - Input <= 100 chars → Direct query mode (search the input directly)

        Args:
            input_text: Either a full script or a direct search query

        Returns:
            Standard result dict with success, outputs, metadata
        """
        self.call_count += 1
        start_time = time.time()

        logger.info("=" * 60)
        logger.info(f"LANGSEARCH CALL #{self.call_count}")
        logger.info("=" * 60)

        try:
            if len(input_text) > self.SCRIPT_THRESHOLD:
                mode = "script"
                logger.info("Mode: SCRIPT (extracting terms from script)")
                terms = self._extract_terms(input_text)
            else:
                mode = "direct"
                logger.info(f"Mode: DIRECT QUERY → '{input_text}'")
                terms = [input_text.strip()]

            logger.info(f"Terms to research: {terms}")

            # Research each term
            research_results = []
            for term in terms[:self.MAX_TERMS]:
                logger.info(f"\nSearching: '{term}'...")
                result = self._search_term(term)
                if result:
                    result['relevance_score'] = self._score_relevance(term, input_text)
                    research_results.append(result)

            # Sort by relevance
            research_results.sort(key=lambda x: x['relevance_score'], reverse=True)

            elapsed = round(time.time() - start_time, 2)

            research_data = {
                "terms": research_results,
                "total_terms_found": len(research_results),
                "search_time": elapsed,
                "mode": mode
            }

            logger.info(f"\n✅ Research complete: {len(research_results)} terms in {elapsed}s")

            return {
                "success": True,
                "outputs": {
                    "research_data": research_data,
                    "research_file": "research_data.json"
                },
                "metadata": {
                    "worker": "langsearch",
                    "timestamp": datetime.now().isoformat(),
                    "call_count": self.call_count,
                    "mode": mode,
                    "terms_searched": len(research_results)
                }
            }

        except Exception as e:
            logger.error(f"❌ LangSearch failed: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": f"LangSearch error: {str(e)}",
                "outputs": {},
                "metadata": {
                    "worker": "langsearch",
                    "timestamp": datetime.now().isoformat(),
                    "call_count": self.call_count
                }
            }

    def _extract_terms(self, script: str) -> List[str]:
        """
        Extract key terms from a script.

        Strips script markers, filters stop words, and ranks
        terms by frequency to find the most important ones.
        """
        text = script
        for marker in self.SCRIPT_MARKERS:
            text = text.replace(marker, '')

        # Extract words (alphabetic, 3+ chars)
        words = re.findall(r"[a-zA-Z']+", text.lower())
        words = [w.strip("'") for w in words if len(w) > 2 and w.strip("'") not in self.STOP_WORDS]

        word_counts = Counter(words)

        # Prioritize terms that appear more than once
        top_terms = [term for term, count in word_counts.most_common(self.MAX_TERMS * 2) if count > 1]

        # Fill remaining slots with most common single-occurrence terms
        if len(top_terms) < self.MAX_TERMS:
            for term, _ in word_counts.most_common():
                if term not in top_terms:
                    top_terms.append(term)
                if len(top_terms) >= self.MAX_TERMS:
                    break

        logger.info(f"Extracted {len(top_terms)} key terms: {top_terms[:self.MAX_TERMS]}")
        return top_terms[:self.MAX_TERMS]

    def _search_term(self, term: str) -> Dict[str, Any]:
        """
        Search a single term via the LangSearch API.

        Returns structured result with term, context, and sources.
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "query": term,
                "num_results": 3
            }

            response = requests.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=15
            )

            if response.status_code == 200:
                data = response.json()
                results = data.get('results', []) or data.get('data', [])

                sources = []
                context_parts = []

                for r in results[:3]:
                    if isinstance(r, dict):
                        url = r.get('url', '')
                        snippet = r.get('snippet', '') or r.get('content', '')

                        if url:
                            sources.append(url)
                        if snippet:
                            context_parts.append(snippet[:200])

                context = ' '.join(context_parts) if context_parts else f"Information about {term}"

                logger.info(f"  ✓ '{term}' → {len(sources)} sources")
                return {
                    "term": term,
                    "context": context,
                    "sources": sources,
                    "relevance_score": 0.0
                }
            else:
                logger.warning(f"  ⚠️ API returned {response.status_code} for '{term}'")
                return {
                    "term": term,
                    "context": f"Information about {term}",
                    "sources": [],
                    "relevance_score": 0.0
                }

        except Exception as e:
            logger.warning(f"  ⚠️ Search failed for '{term}': {str(e)}")
            return {
                "term": term,
                "context": f"Information about {term}",
                "sources": [],
                "relevance_score": 0.0
            }

    def _score_relevance(self, term: str, input_text: str) -> float:
        """
        Score how relevant a term is to the input text.

        Based on:
        - Frequency of term in the text (higher = more relevant)
        - Position (terms appearing earlier get a slight boost)

        Returns score between 0.0 and 1.0.
        """
        text_lower = input_text.lower()
        term_lower = term.lower()

        count = text_lower.count(term_lower)
        if count == 0:
            return 0.5  # Base score for direct queries where term IS the input

        text_len = max(len(text_lower), 1)

        # Frequency score: normalize by text length
        frequency_score = min(count / (text_len / 1000), 1.0)

        # Position score: terms appearing earlier score higher
        first_pos = text_lower.find(term_lower)
        position_score = 1.0 - (first_pos / text_len)

        score = (frequency_score * 0.7) + (position_score * 0.3)
        return round(min(score, 1.0), 2)
