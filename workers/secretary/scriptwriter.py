"""
ScriptWriter Worker - Real Implementation

Generates video scripts using LangSearch for research and Ollama/Mistral for generation.
"""

import requests
import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ScriptWriter:
    """
    ScriptWriter generates video scripts through:
    1. Research phase (LangSearch API)
    2. Talking points generation (Ollama/Mistral)
    3. Body expansion (Ollama/Mistral)
    4. Script assembly (Hook/Body/CTA)
    """

    # Constants
    WORDS_PER_MINUTE = 150  # Average speaking speed
    HOOK_DURATION_SECONDS = 12  # Hook length
    CTA_DURATION_SECONDS = 12   # CTA length

    def __init__(
        self,
        langsearch_api_key: str = "sk-e0235f795d7f4237aaf5048e6816af6b",
        ollama_endpoint: str = "http://localhost:11434",
        model: str = "mistral"
    ):
        """
        Initialize ScriptWriter

        Args:
            langsearch_api_key: LangSearch API key
            ollama_endpoint: Ollama API endpoint
            model: Ollama model to use (default: mistral)
        """
        self.langsearch_api_key = langsearch_api_key
        self.langsearch_endpoint = "https://api.langsearch.com/v1/web-search"
        self.ollama_endpoint = ollama_endpoint
        self.model = model
        self.call_count = 0

        logger.info(f"ScriptWriter initialized:")
        logger.info(f"  - LangSearch endpoint: {self.langsearch_endpoint}")
        logger.info(f"  - Ollama endpoint: {self.ollama_endpoint}")
        logger.info(f"  - Model: {self.model}")

    def generate_script(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main method to generate complete script

        Args:
            requirements: Dictionary with topic, style, duration_range, comments

        Returns:
            Dictionary with success, outputs, and metadata
        """
        self.call_count += 1
        logger.info("="*60)
        logger.info(f"SCRIPTWRITER CALL #{self.call_count} - Starting script generation")
        logger.info("="*60)

        try:
            # Extract requirements
            topic = requirements.get('topic', '')
            style = requirements.get('style', '')
            duration_range = requirements.get('duration_range', '')
            comments = requirements.get('comments', '')

            logger.info(f"Requirements:")
            logger.info(f"  - Topic: {topic}")
            logger.info(f"  - Style: {style}")
            logger.info(f"  - Duration: {duration_range}")
            logger.info(f"  - Comments: {comments if comments else 'None'}")

            # Step 1: Calculate target metrics
            logger.info("\n" + "-"*60)
            logger.info("STEP 1: Calculating target metrics")
            logger.info("-"*60)

            target_words = self._calculate_target_words(duration_range)
            logger.info(f"Target word count: {target_words} words")

            # Step 2: Research phase
            logger.info("\n" + "-"*60)
            logger.info("STEP 2: Research phase (LangSearch API)")
            logger.info("-"*60)

            research_data = self._research_topic(topic)
            logger.info(f"Research complete:")
            logger.info(f"  - Sources found: {len(research_data.get('sources', []))}")
            logger.info(f"  - Key insights: {len(research_data.get('insights', []))}")

            # Step 3: Generate talking points
            logger.info("\n" + "-"*60)
            logger.info("STEP 3: Generating talking points")
            logger.info("-"*60)

            talking_points = self._generate_talking_points(
                topic, style, target_words, comments, research_data
            )
            logger.info(f"Generated {len(talking_points)} talking points:")
            for i, point in enumerate(talking_points, 1):
                logger.info(f"  {i}. {point[:80]}...")

            # Step 4: Expand points to body
            logger.info("\n" + "-"*60)
            logger.info("STEP 4: Expanding talking points to body")
            logger.info("-"*60)

            body_paragraphs = self._expand_body(talking_points, style, target_words)
            logger.info(f"Expanded into {len(body_paragraphs)} paragraphs")
            total_body_words = sum(len(p.split()) for p in body_paragraphs)
            logger.info(f"Body word count: {total_body_words} words")

            # Step 5: Generate hook
            logger.info("\n" + "-"*60)
            logger.info("STEP 5: Generating hook")
            logger.info("-"*60)

            hook = self._generate_hook(topic, style)
            hook_words = len(hook.split())
            logger.info(f"Hook generated: {hook_words} words")
            logger.info(f"Hook preview: {hook[:100]}...")

            # Step 6: Generate CTA
            logger.info("\n" + "-"*60)
            logger.info("STEP 6: Generating call-to-action")
            logger.info("-"*60)

            cta = self._generate_cta(topic, style)
            cta_words = len(cta.split())
            logger.info(f"CTA generated: {cta_words} words")
            logger.info(f"CTA preview: {cta[:100]}...")

            # Step 7: Assemble final script
            logger.info("\n" + "-"*60)
            logger.info("STEP 7: Assembling final script")
            logger.info("-"*60)

            script_parts = {
                "hook": hook,
                "body": body_paragraphs,
                "cta": cta
            }

            full_script = self._assemble_script(script_parts)
            total_words = len(full_script.split())
            estimated_duration = total_words / self.WORDS_PER_MINUTE

            logger.info(f"Script assembled:")
            logger.info(f"  - Total words: {total_words}")
            logger.info(f"  - Estimated duration: {estimated_duration:.1f} minutes")
            logger.info(f"  - Target was: {duration_range}")

            # Prepare output
            logger.info("\n" + "-"*60)
            logger.info("STEP 8: Preparing output")
            logger.info("-"*60)

            result = {
                "success": True,
                "outputs": {
                    "script": full_script,
                    "script_file": "script.txt",
                    "structure": script_parts,
                    "word_count": total_words,
                    "estimated_duration_minutes": round(estimated_duration, 1),
                    "talking_points_count": len(talking_points),
                    "research_sources": research_data.get('sources', [])
                },
                "metadata": {
                    "worker": "scriptwriter",
                    "timestamp": datetime.now().isoformat(),
                    "call_count": self.call_count,
                    "model_used": self.model,
                    "target_words": target_words,
                    "actual_words": total_words
                }
            }

            logger.info("✅ Script generation successful!")
            logger.info("="*60)

            return result

        except Exception as e:
            logger.error(f"❌ Script generation failed: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": f"ScriptWriter error: {str(e)}",
                "outputs": {},
                "metadata": {
                    "worker": "scriptwriter",
                    "timestamp": datetime.now().isoformat(),
                    "call_count": self.call_count
                }
            }

    def _calculate_target_words(self, duration_range: str) -> int:
        """Calculate target word count from duration"""
        logger.info(f"Parsing duration: {duration_range}")

        # Extract numbers from duration (e.g., "5 minutes" or "2-3 minutes")
        numbers = re.findall(r'\d+', duration_range)

        if not numbers:
            logger.warning("No numbers found in duration, defaulting to 2 minutes")
            return 2 * 60 * self.WORDS_PER_MINUTE / 60

        # If range (e.g., "2-3"), use the middle
        if len(numbers) >= 2:
            target_minutes = (int(numbers[0]) + int(numbers[1])) / 2
        else:
            target_minutes = int(numbers[0])

        # Check if seconds or minutes
        if 'second' in duration_range.lower() or 'sec' in duration_range.lower():
            target_minutes = target_minutes / 60

        target_words = int(target_minutes * self.WORDS_PER_MINUTE)

        # Reserve words for hook and CTA
        hook_words = int((self.HOOK_DURATION_SECONDS / 60) * self.WORDS_PER_MINUTE)
        cta_words = int((self.CTA_DURATION_SECONDS / 60) * self.WORDS_PER_MINUTE)
        body_words = target_words - hook_words - cta_words

        logger.info(f"Duration breakdown:")
        logger.info(f"  - Total duration: {target_minutes:.1f} minutes")
        logger.info(f"  - Hook: ~{hook_words} words")
        logger.info(f"  - Body: ~{body_words} words")
        logger.info(f"  - CTA: ~{cta_words} words")
        logger.info(f"  - Total: ~{target_words} words")

        return body_words  # Return body word count

    def _research_topic(self, topic: str) -> Dict[str, Any]:
        """Research topic using LangSearch API"""
        logger.info(f"Calling LangSearch API for topic: {topic}")

        try:
            headers = {
                "Authorization": f"Bearer {self.langsearch_api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "query": topic,
                "num_results": 5
            }

            logger.info(f"Request URL: {self.langsearch_endpoint}")
            logger.info(f"Request payload: {json.dumps(payload, indent=2)}")

            response = requests.post(
                self.langsearch_endpoint,
                headers=headers,
                json=payload,
                timeout=30
            )

            logger.info(f"Response status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                logger.info(f"Research data received: {len(str(data))} characters")

                # Extract sources and insights
                sources = []
                insights = []

                if isinstance(data, dict):
                    results = data.get('results', []) or data.get('data', [])

                    for result in results[:5]:
                        if isinstance(result, dict):
                            title = result.get('title', '')
                            url = result.get('url', '')
                            snippet = result.get('snippet', '') or result.get('content', '')

                            if title or url:
                                sources.append({
                                    "title": title,
                                    "url": url,
                                    "snippet": snippet[:200]
                                })

                            if snippet:
                                insights.append(snippet[:300])

                logger.info(f"Extracted {len(sources)} sources and {len(insights)} insights")

                return {
                    "sources": sources,
                    "insights": insights,
                    "raw_data": data
                }
            else:
                logger.warning(f"LangSearch API returned status {response.status_code}")
                logger.warning(f"Response: {response.text[:500]}")
                return {
                    "sources": [],
                    "insights": [f"General information about {topic}"],
                    "raw_data": {}
                }

        except Exception as e:
            logger.error(f"LangSearch API error: {str(e)}", exc_info=True)
            return {
                "sources": [],
                "insights": [f"General information about {topic}"],
                "raw_data": {},
                "error": str(e)
            }

    def _generate_talking_points(
        self,
        topic: str,
        style: str,
        target_words: int,
        comments: str,
        research_data: Dict
    ) -> List[str]:
        """Generate main talking points using Ollama"""
        logger.info("Generating talking points with Ollama/Mistral")

        # Calculate number of points based on target words
        words_per_point = 100  # Average words per expanded point
        num_points = max(3, min(7, target_words // words_per_point))

        logger.info(f"Target: {num_points} talking points")

        # Build research context
        research_context = "\n".join(research_data.get('insights', [])[:3])

        prompt = f"""You are creating a {style} video script about "{topic}".

Research context:
{research_context}

Additional requirements: {comments if comments else 'None'}

Generate exactly {num_points} main talking points for this video. Each point should be:
- Clear and concise (one sentence)
- Appropriate for a {style} style
- Based on the research provided
- Interesting and engaging

Output ONLY the talking points, numbered 1-{num_points}, nothing else."""

        logger.info("Sending prompt to Ollama:")
        logger.info(f"Prompt length: {len(prompt)} characters")

        try:
            response = self._call_ollama(prompt)
            logger.info(f"Received response: {len(response)} characters")

            # Parse talking points
            points = []
            lines = response.split('\n')

            for line in lines:
                line = line.strip()
                # Match numbered points like "1. " or "1) "
                if re.match(r'^\d+[\.\)]\s+', line):
                    point = re.sub(r'^\d+[\.\)]\s+', '', line)
                    if point:
                        points.append(point)
                        logger.info(f"Extracted point: {point[:60]}...")

            if not points:
                logger.warning("No points extracted, using fallback")
                points = [
                    f"Introduction to {topic}",
                    f"Key aspects of {topic}",
                    f"Important details about {topic}"
                ]

            logger.info(f"Successfully extracted {len(points)} talking points")
            return points[:num_points]

        except Exception as e:
            logger.error(f"Error generating talking points: {str(e)}", exc_info=True)
            return [
                f"Introduction to {topic}",
                f"Main concepts of {topic}",
                f"Conclusion about {topic}"
            ]

    def _expand_body(self, talking_points: List[str], style: str, target_words: int) -> List[str]:
        """Expand talking points into full body paragraphs"""
        logger.info(f"Expanding {len(talking_points)} points into body paragraphs")

        words_per_point = target_words // len(talking_points)
        logger.info(f"Target: ~{words_per_point} words per point")

        paragraphs = []

        for i, point in enumerate(talking_points, 1):
            logger.info(f"Expanding point {i}/{len(talking_points)}: {point[:50]}...")

            prompt = f"""Expand this talking point into a detailed paragraph for a {style} video script:

Talking point: {point}

Write a paragraph of approximately {words_per_point} words that:
- Maintains a {style} tone
- Is engaging and informative
- Flows naturally when spoken
- Includes specific details and examples

Output ONLY the paragraph, no preamble or labels."""

            try:
                paragraph = self._call_ollama(prompt)
                word_count = len(paragraph.split())
                logger.info(f"Generated paragraph: {word_count} words")
                paragraphs.append(paragraph.strip())

            except Exception as e:
                logger.error(f"Error expanding point {i}: {str(e)}")
                paragraphs.append(f"{point}. [Detailed explanation would go here.]")

        logger.info(f"Successfully expanded all {len(paragraphs)} points")
        return paragraphs

    def _generate_hook(self, topic: str, style: str) -> str:
        """Generate opening hook"""
        logger.info(f"Generating hook for {style} style")

        target_words = int((self.HOOK_DURATION_SECONDS / 60) * self.WORDS_PER_MINUTE)

        prompt = f"""Write an engaging opening hook for a {style} video about "{topic}".

The hook should:
- Be approximately {target_words} words ({self.HOOK_DURATION_SECONDS} seconds when spoken)
- Grab attention immediately
- Match a {style} tone
- Make viewers want to keep watching

Output ONLY the hook text, no labels or explanations."""

        try:
            hook = self._call_ollama(prompt)
            logger.info(f"Hook generated: {len(hook.split())} words")
            return hook.strip()

        except Exception as e:
            logger.error(f"Error generating hook: {str(e)}")
            return f"Welcome! Today we're exploring {topic}. Let's dive in!"

    def _generate_cta(self, topic: str, style: str) -> str:
        """Generate call-to-action"""
        logger.info(f"Generating CTA for {style} style")

        target_words = int((self.CTA_DURATION_SECONDS / 60) * self.WORDS_PER_MINUTE)

        prompt = f"""Write a closing call-to-action for a {style} video about "{topic}".

The CTA should:
- Be approximately {target_words} words ({self.CTA_DURATION_SECONDS} seconds when spoken)
- Encourage engagement (like, subscribe, comment)
- Match a {style} tone
- Leave a positive impression

Output ONLY the CTA text, no labels or explanations."""

        try:
            cta = self._call_ollama(prompt)
            logger.info(f"CTA generated: {len(cta.split())} words")
            return cta.strip()

        except Exception as e:
            logger.error(f"Error generating CTA: {str(e)}")
            return f"Thanks for watching! If you enjoyed this video about {topic}, please like and subscribe for more content!"

    def _assemble_script(self, parts: Dict[str, Any]) -> str:
        """Assemble final script from parts"""
        logger.info("Assembling final script from parts")

        script = f"""[HOOK]
{parts['hook']}

[BODY]
"""

        for i, paragraph in enumerate(parts['body'], 1):
            script += f"\n{paragraph}\n"

        script += f"""
[CALL TO ACTION]
{parts['cta']}
"""

        logger.info("Script assembly complete")
        return script.strip()

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API"""
        logger.info(f"Calling Ollama API (model: {self.model})")

        try:
            url = f"{self.ollama_endpoint}/api/generate"

            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }

            logger.info(f"Request URL: {url}")
            logger.info(f"Prompt length: {len(prompt)} characters")

            response = requests.post(url, json=payload, timeout=120)

            logger.info(f"Response status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                result = data.get('response', '')
                logger.info(f"Received response: {len(result)} characters")
                return result
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                logger.error(f"Response: {response.text[:500]}")
                raise Exception(f"Ollama API returned {response.status_code}")

        except Exception as e:
            logger.error(f"Ollama API call failed: {str(e)}", exc_info=True)
            raise
