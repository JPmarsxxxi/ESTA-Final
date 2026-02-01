"""
BrainBox Worker - Style-Driven Video Planning

Creates detailed shot-by-shot video plans by:
1. Studying real videos of the target style (via AssetCollector)
2. Using Ollama/Mistral to reason about shot planning based on
   the style analysis data, script segments, and audio timestamps

Every decision is printed with full reasoning so users can inspect
and understand why each shot was planned the way it was.

Planning Pipeline:
  Phase 1: Style Analysis  -> AssetCollector.analyze_style() downloads and
                               analyzes real videos, produces pacing data
  Phase 2: LLM Planning    -> Style data + script + timestamps sent to Ollama;
                               LLM produces shot plan with per-shot reasoning
  Phase 3: Plan Output     -> Parsed plan printed with reasoning per shot;
                               fallback to rule-based plan if Ollama unavailable
"""

import re
import time
import random
import logging
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import AssetCollector for style analysis
try:
    from asset_collector import AssetCollector
    ASSET_COLLECTOR_AVAILABLE = True
    logger.info("✅ AssetCollector imported for style analysis")
except ImportError as e:
    ASSET_COLLECTOR_AVAILABLE = False
    logger.warning(f"⚠️  AssetCollector not available: {e}")


class BrainBox:
    """
    Style-driven video planner.

    Receives script, audio timestamps, and research data from Secretary.
    Internally triggers AssetCollector to study real videos of the target
    style, then uses Ollama/Mistral to produce a shot-by-shot plan where
    every decision references specific numbers from the style analysis.
    """

    OLLAMA_URL = "http://localhost:11434/api/generate"
    MODEL = "mistral"
    OLLAMA_TIMEOUT = 120  # Seconds

    def __init__(self):
        self.call_count = 0
        self.asset_collector = None
        if ASSET_COLLECTOR_AVAILABLE:
            self.asset_collector = AssetCollector()
        logger.info(f"BrainBox initialized (AssetCollector: {ASSET_COLLECTOR_AVAILABLE})")

    def create_video_plan(self, script: str, transcript_timestamps: Dict,
                          research_data: Dict, requirements: Dict) -> Dict[str, Any]:
        """
        Main entry point. Orchestrates the three-phase planning process.

        Args:
            script: Generated script text
            transcript_timestamps: Audio segments with timing from AudioAgent
            research_data: Research results from LangSearch
            requirements: Original user requirements (topic, style, duration)

        Returns:
            Standard result dict with video_plan in outputs
        """
        self.call_count += 1
        start_time = time.time()

        style = requirements.get('style', 'default')
        topic = requirements.get('topic', 'unknown')

        print(f"\n{'='*60}")
        print("BRAINBOX - VIDEO PLANNING")
        print(f"Style: {style} | Topic: {topic}")
        print(f"{'='*60}")

        try:
            # ─── Phase 1: Style Analysis ──────────────────────────
            print(f"\n{'─'*60}")
            print("PHASE 1: Style Analysis")
            print(f"{'─'*60}")

            style_analysis = self._run_style_analysis(style, topic)

            # ─── Phase 2: LLM Planning ────────────────────────────
            print(f"\n{'─'*60}")
            print("PHASE 2: LLM-Driven Planning")
            print(f"{'─'*60}")

            segments = self._parse_segments(transcript_timestamps)
            video_plan = self._generate_plan_with_llm(
                style_analysis, script, segments, research_data, requirements
            )

            if not video_plan:
                print("\n⚠️  LLM planning failed - using rule-based fallback")
                video_plan = self._generate_plan_fallback(
                    style_analysis, segments, requirements
                )

            # ─── Phase 3: Output ──────────────────────────────────
            print(f"\n{'─'*60}")
            print("PHASE 3: Final Plan")
            print(f"{'─'*60}")

            self._print_plan(video_plan)

            elapsed = round(time.time() - start_time, 2)

            return {
                "success": True,
                "outputs": {
                    "video_plan": video_plan,
                    "plan_file": "video_plan.json"
                },
                "metadata": {
                    "worker": "brainbox",
                    "timestamp": datetime.now().isoformat(),
                    "call_count": self.call_count,
                    "planning_time": elapsed,
                    "style_analysis_used": not style_analysis.get('fallback', False),
                    "llm_used": video_plan.get('_planning_method') != 'rule-based-fallback'
                }
            }

        except Exception as e:
            logger.error(f"BrainBox failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"BrainBox error: {str(e)}",
                "outputs": {},
                "metadata": {
                    "worker": "brainbox",
                    "timestamp": datetime.now().isoformat(),
                    "call_count": self.call_count
                }
            }

    # ─── Phase 1: Style Analysis ─────────────────────────────────────

    def _run_style_analysis(self, style: str, topic: str) -> Dict:
        """Run style analysis via AssetCollector. Prints full report."""
        if not self.asset_collector:
            print("  ⚠️  AssetCollector not available - using default analysis")
            return self._default_style_analysis(style)

        print(f"  Analyzing style via AssetCollector...")
        result = self.asset_collector.analyze_style(style, topic)

        if result.get('success'):
            return result['outputs']['style_analysis']

        print(f"  ⚠️  Style analysis failed: {result.get('error', 'unknown')}")
        return self._default_style_analysis(style)

    def _default_style_analysis(self, style: str) -> Dict:
        """Default analysis when AssetCollector is unavailable."""
        return {
            'style': style,
            'videos_analyzed': 0,
            'total_shots_analyzed': 0,
            'pacing': {
                'avg_cuts_per_minute': 6.0,
                'avg_shot_duration': 5.0,
                'duration_buckets': {
                    '0-2s': 2, '2-5s': 4, '5-10s': 3, '10-20s': 1, '20s+': 0
                }
            },
            'shot_durations': {
                'mean': 5.0, 'median': 4.5, 'p25': 2.5, 'p75': 7.0,
                'min': 1.0, 'max': 15.0, 'std': 3.5
            },
            'shot_type_distribution': {
                'b-roll': 0.35, 'talking-head': 0.30, 'action': 0.15,
                'text-overlay': 0.10, 'transition': 0.05, 'static': 0.05
            },
            'motion_distribution': {'medium': 0.5, 'high': 0.3, 'low': 0.2},
            'classification_method': 'default-fallback',
            'fallback': True
        }

    # ─── Phase 2: LLM Planning ───────────────────────────────────────

    def _parse_segments(self, transcript_timestamps: Dict) -> List[Dict]:
        """Extract audio segments from transcript_timestamps."""
        segments = transcript_timestamps.get('segments', [])
        if not segments:
            return [{'start': 0.0, 'end': 120.0, 'text': 'Full video', 'type': 'narration'}]
        return segments

    def _generate_plan_with_llm(self, style_analysis: Dict, script: str,
                                segments: List[Dict], research_data: Dict,
                                requirements: Dict) -> Optional[Dict]:
        """
        Generate video plan using Ollama/Mistral.

        The prompt is constructed to:
        1. Present style analysis as concrete numbers the LLM must reference
        2. Require per-shot reasoning that cites specific analysis figures
        3. Specify a strict output format for reliable parsing
        4. Include audio segments so shots align with narration

        Args:
            style_analysis: Output from AssetCollector.analyze_style()
            script: Full script text
            segments: Audio segments with timestamps
            research_data: LangSearch research results
            requirements: User requirements

        Returns:
            Parsed video_plan dict, or None if LLM call fails
        """
        if not self._check_ollama():
            print("  ⚠️  Ollama not available")
            return None

        prompt = self._build_planning_prompt(
            style_analysis, script, segments, research_data, requirements
        )

        print(f"  Sending planning request to Ollama ({self.MODEL})...")
        print(f"  Prompt length: {len(prompt)} chars")

        try:
            response = requests.post(
                self.OLLAMA_URL,
                json={
                    "model": self.MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 4096,
                    }
                },
                timeout=self.OLLAMA_TIMEOUT
            )

            if response.status_code != 200:
                logger.error(f"Ollama returned {response.status_code}")
                return None

            llm_response = response.json().get('response', '')

            # Print raw LLM output for inspection
            print(f"\n  📝 RAW LLM RESPONSE:")
            print(f"  {'─'*56}")
            for line in llm_response.split('\n'):
                print(f"  {line}")
            print(f"  {'─'*56}")

            # Parse into structured plan
            video_plan = self._parse_llm_plan(llm_response, segments, requirements)

            if video_plan:
                video_plan['_planning_method'] = 'llm'
                video_plan['_style_analysis'] = style_analysis
                return video_plan

            return None

        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out")
            return None
        except Exception as e:
            logger.error(f"LLM planning failed: {e}")
            return None

    def _build_planning_prompt(self, style_analysis: Dict, script: str,
                               segments: List[Dict], research_data: Dict,
                               requirements: Dict) -> str:
        """
        Build the planning prompt with style analysis data embedded as
        concrete numbers. The prompt explicitly instructs the LLM to
        reference these numbers in its per-shot reasoning.
        """
        pacing = style_analysis.get('pacing', {})
        durations = style_analysis.get('shot_durations', {})
        type_dist = style_analysis.get('shot_type_distribution', {})
        motion_dist = style_analysis.get('motion_distribution', {})
        buckets = pacing.get('duration_buckets', {})

        analysis_text = (
            f"STYLE ANALYSIS DATA (from {style_analysis.get('videos_analyzed', 0)} real "
            f"{style_analysis.get('style', 'unknown')} videos, "
            f"{style_analysis.get('total_shots_analyzed', 0)} shots total):\n\n"
            f"Pacing:\n"
            f"- Average cuts per minute: {pacing.get('avg_cuts_per_minute', 6.0)}\n"
            f"- Average shot duration: {pacing.get('avg_shot_duration', 5.0)}s\n\n"
            f"Shot Duration Statistics:\n"
            f"- Mean: {durations.get('mean', 5.0)}s | Median: {durations.get('median', 4.5)}s\n"
            f"- 25th percentile: {durations.get('p25', 2.5)}s | 75th percentile: {durations.get('p75', 7.0)}s\n"
            f"- Range: {durations.get('min', 1.0)}s to {durations.get('max', 15.0)}s | "
            f"Std dev: {durations.get('std', 3.5)}s\n\n"
            f"Duration Buckets:\n"
            + "\n".join(f"- {k}: {v} shots" for k, v in buckets.items()) + "\n\n"
            f"Shot Type Distribution:\n"
            + "\n".join(
                f"- {k}: {v * 100:.0f}%" for k, v in sorted(type_dist.items(), key=lambda x: -x[1])
            ) + "\n\n"
            f"Motion Level Distribution:\n"
            + "\n".join(
                f"- {k}: {v * 100:.0f}%" for k, v in sorted(motion_dist.items(), key=lambda x: -x[1])
            )
        )

        # Format audio segments (cap at 20 to keep prompt size reasonable)
        segments_text = "\n".join(
            f"- Segment {i + 1} [{seg['start']:.1f}s-{seg['end']:.1f}s]: {seg.get('text', '')[:80]}"
            for i, seg in enumerate(segments[:20])
        )

        # Research highlights
        research_terms = research_data.get('terms', [])
        research_text = "\n".join(
            f"- {t['term']}: {t.get('context', '')[:100]}"
            for t in research_terms[:5]
        ) if research_terms else "No research data available."

        # Script (truncated for prompt size)
        script_text = script[:2000] + ("...[truncated]" if len(script) > 2000 else "")

        total_duration = segments[-1]['end'] if segments else 120.0

        prompt = f"""You are a video editor planning a shot-by-shot video. You must create a plan that matches the STYLE of real videos in this genre.

IMPORTANT: You MUST reference specific numbers from the Style Analysis Data in your reasoning for each shot. Your shot durations should match the patterns observed in real videos of this style.

{analysis_text}

USER REQUIREMENTS:
- Topic: {requirements.get('topic', 'unknown')}
- Style: {requirements.get('style', 'unknown')}
- Duration: {requirements.get('duration_range', 'unknown')}
- Comments: {requirements.get('comments', 'None')}

SCRIPT:
{script_text}

AUDIO SEGMENTS (with timestamps):
{segments_text}

RESEARCH CONTEXT:
{research_text}

Total video duration: {total_duration:.1f} seconds

TASK: Create a shot-by-shot video plan. For each shot, specify:
1. Start and end time (must cover the full {total_duration:.1f}s)
2. Shot type (talking-head, b-roll, action, text-overlay, aerial, product, transition, static)
3. Visual description (what the viewer sees)
4. REASONING: Explain WHY you chose this duration and type, referencing specific numbers from the style analysis above

OUTPUT FORMAT - Use exactly this format for each shot (no markdown, no extra text):

SHOT 1:
TIME: [start]s-[end]s
TYPE: [shot_type]
VISUAL: [description]
REASONING: [your reasoning referencing style analysis numbers]

SHOT 2:
TIME: [start]s-[end]s
TYPE: [shot_type]
VISUAL: [description]
REASONING: [your reasoning referencing style analysis numbers]

Begin planning:"""

        return prompt

    # ─── LLM Response Parsing ────────────────────────────────────────

    def _parse_llm_plan(self, llm_response: str, segments: List[Dict],
                        requirements: Dict) -> Optional[Dict]:
        """
        Parse the LLM's shot plan from its text response.

        Tries strict regex first. If that fails, falls back to a more
        lenient line-by-line parser that handles varied LLM formatting.

        Returns:
            Structured video_plan dict, or None if no shots could be parsed
        """
        # Strict regex: expects the exact format we requested
        shot_pattern = re.compile(
            r'SHOT\s+(\d+)\s*:\s*\n'
            r'TIME:\s*\[?(\d+(?:\.\d+)?)\]?s?\s*-\s*\[?(\d+(?:\.\d+)?)\]?s?\s*\n'
            r'TYPE:\s*(.+?)\s*\n'
            r'VISUAL:\s*(.+?)\s*\n'
            r'REASONING:\s*(.+?)(?=\nSHOT\s+\d+|\Z)',
            re.DOTALL | re.IGNORECASE
        )

        matches = shot_pattern.findall(llm_response)

        if not matches:
            logger.info("Strict parsing failed, trying lenient parse...")
            matches = self._lenient_parse(llm_response)

        if not matches:
            logger.warning("Could not parse any shots from LLM response")
            return None

        timeline = []
        for match in matches:
            shot_num, start, end, shot_type, visual, reasoning = match

            timeline.append({
                'shot_number': int(shot_num),
                'timestamp': f"{float(start):.1f}s-{float(end):.1f}s",
                'start_time': float(start),
                'end_time': float(end),
                'duration': round(float(end) - float(start), 2),
                'asset_type': self._normalize_shot_type(shot_type.strip().lower()),
                'description': visual.strip(),
                'reasoning': reasoning.strip(),
                'script_line': self._find_script_line(float(start), float(end), segments)
            })

        if not timeline:
            return None

        timeline.sort(key=lambda x: x['start_time'])
        total_duration = timeline[-1]['end_time']

        return {
            'timeline': timeline,
            'total_duration': total_duration,
            'total_shots': len(timeline),
            'theme': requirements.get('style', 'default'),
            'topic': requirements.get('topic', ''),
        }

    def _lenient_parse(self, response: str) -> List[tuple]:
        """
        Line-by-line parser for when LLM output doesn't match strict format.
        Accumulates fields as it encounters them, emits a shot tuple when
        it hits the next SHOT marker or end of text.
        """
        results = []
        current_shot = {}

        for line in response.split('\n'):
            line = line.strip()

            # SHOT N:
            m = re.match(r'SHOT\s+(\d+)', line, re.IGNORECASE)
            if m:
                if current_shot.get('num'):
                    results.append(self._shot_dict_to_tuple(current_shot))
                current_shot = {'num': m.group(1)}
                continue

            # TIME: Xs-Ys
            m = re.match(
                r'TIME:\s*\[?(\d+(?:\.\d+)?)\]?s?\s*-\s*\[?(\d+(?:\.\d+)?)\]?s?',
                line, re.IGNORECASE
            )
            if m:
                current_shot['start'] = m.group(1)
                current_shot['end'] = m.group(2)
                continue

            # TYPE:
            m = re.match(r'TYPE:\s*(.+)', line, re.IGNORECASE)
            if m:
                current_shot['type'] = m.group(1).strip()
                continue

            # VISUAL:
            m = re.match(r'VISUAL:\s*(.+)', line, re.IGNORECASE)
            if m:
                current_shot['visual'] = m.group(1).strip()
                continue

            # REASONING:
            m = re.match(r'REASONING:\s*(.+)', line, re.IGNORECASE)
            if m:
                current_shot['reasoning'] = m.group(1).strip()
                continue

            # Multi-line reasoning continuation
            if 'reasoning' in current_shot and line and not any(
                line.upper().startswith(k)
                for k in ['SHOT', 'TIME:', 'TYPE:', 'VISUAL:', 'REASONING:']
            ):
                current_shot['reasoning'] += ' ' + line

        # Emit last shot
        if current_shot.get('num'):
            results.append(self._shot_dict_to_tuple(current_shot))

        return results

    def _shot_dict_to_tuple(self, d: Dict) -> tuple:
        """Convert a parsed shot dict to the 6-tuple format expected by _parse_llm_plan."""
        return (
            d.get('num', '0'),
            d.get('start', '0'),
            d.get('end', '0'),
            d.get('type', 'b-roll'),
            d.get('visual', 'Visual content'),
            d.get('reasoning', 'No reasoning provided')
        )

    def _normalize_shot_type(self, shot_type: str) -> str:
        """Map varied LLM shot type strings to canonical types."""
        type_map = {
            'talking': 'talking-head', 'face': 'talking-head', 'head': 'talking-head',
            'broll': 'b-roll', 'b roll': 'b-roll', 'establishing': 'b-roll', 'cutaway': 'b-roll',
            'action': 'action', 'dynamic': 'action', 'energetic': 'action',
            'text': 'text-overlay', 'title': 'text-overlay', 'graphic': 'text-overlay', 'overlay': 'text-overlay',
            'aerial': 'aerial', 'drone': 'aerial',
            'product': 'product', 'showcase': 'product',
            'transition': 'transition', 'fade': 'transition', 'dissolve': 'transition',
            'static': 'static', 'still': 'static', 'frozen': 'static'
        }

        for keyword, canonical in type_map.items():
            if keyword in shot_type:
                return canonical

        canonical_types = [
            'talking-head', 'b-roll', 'action', 'text-overlay',
            'aerial', 'product', 'transition', 'static'
        ]
        return shot_type if shot_type in canonical_types else 'b-roll'

    def _find_script_line(self, start: float, end: float, segments: List[Dict]) -> str:
        """Find the audio segment that overlaps with a shot's time range."""
        mid = (start + end) / 2
        for seg in segments:
            if seg.get('start', 0) <= mid <= seg.get('end', 0):
                return seg.get('text', '')[:80]
        return ''

    # ─── Phase 2 (Fallback): Rule-Based Planning ────────────────────

    def _generate_plan_fallback(self, style_analysis: Dict, segments: List[Dict],
                                requirements: Dict) -> Dict:
        """
        Rule-based fallback plan when Ollama is unavailable.

        Still uses style analysis data to drive shot durations (mean +
        gaussian noise scaled by std dev) and cycles shot types according
        to the observed type distribution. Prints reasoning for each shot
        so the user can see what data drove each decision.
        """
        print("  Building rule-based plan from style analysis...")

        total_duration = segments[-1]['end'] if segments else 120.0

        avg_shot_dur = style_analysis.get('pacing', {}).get('avg_shot_duration', 5.0)
        std_dev = style_analysis.get('shot_durations', {}).get('std', 2.0)
        type_dist = style_analysis.get('shot_type_distribution', {})

        # Build type cycle from distribution (exclude rare transition/static)
        sorted_types = sorted(type_dist.items(), key=lambda x: -x[1])
        type_cycle = [t for t, _ in sorted_types if t not in ('transition', 'static')]
        if not type_cycle:
            type_cycle = ['b-roll', 'talking-head']

        timeline = []
        current_time = 0.0
        shot_num = 1
        type_idx = 0

        while current_time < total_duration:
            # Duration: mean + gaussian noise (scaled to 30% of std dev)
            variation = random.gauss(0, std_dev * 0.3)
            shot_duration = max(1.0, avg_shot_dur + variation)
            shot_duration = min(shot_duration, total_duration - current_time)
            end_time = round(current_time + shot_duration, 2)

            # Shot type assignment
            if shot_num == 1:
                shot_type = 'b-roll'  # Opening B-roll is common across styles
            elif end_time >= total_duration - 2:
                shot_type = 'text-overlay'  # CTA at end
            else:
                shot_type = type_cycle[type_idx % len(type_cycle)]
                type_idx += 1

            script_line = self._find_script_line(current_time, end_time, segments)

            reasoning = (
                f"Style analysis shows avg shot duration of {avg_shot_dur}s "
                f"(std: {std_dev}s). Applied gaussian variation ({variation:+.1f}s) "
                f"to get {shot_duration:.1f}s. Assigned {shot_type} based on "
                f"observed type distribution ({type_dist.get(shot_type, 0) * 100:.0f}% "
                f"of shots in this style)."
            )

            timeline.append({
                'shot_number': shot_num,
                'timestamp': f"{current_time:.1f}s-{end_time:.1f}s",
                'start_time': current_time,
                'end_time': end_time,
                'duration': round(shot_duration, 2),
                'asset_type': shot_type,
                'description': f"[Fallback] Shot {shot_num}: {shot_type}",
                'reasoning': reasoning,
                'script_line': script_line
            })

            current_time = end_time
            shot_num += 1

        return {
            'timeline': timeline,
            'total_duration': total_duration,
            'total_shots': len(timeline),
            'theme': requirements.get('style', 'default'),
            'topic': requirements.get('topic', ''),
            '_planning_method': 'rule-based-fallback',
            '_style_analysis': style_analysis
        }

    # ─── Phase 3: Plan Output ────────────────────────────────────────

    def _print_plan(self, video_plan: Dict):
        """Print the complete plan with per-shot reasoning."""
        timeline = video_plan.get('timeline', [])
        method = video_plan.get('_planning_method', 'llm')

        print(f"\n  Planning method: {'LLM (Ollama/Mistral)' if method == 'llm' else 'Rule-based fallback'}")
        print(f"  Total shots: {len(timeline)} | Duration: {video_plan.get('total_duration', 0):.1f}s")
        print(f"  Theme: {video_plan.get('theme', 'unknown')} | Topic: {video_plan.get('topic', 'unknown')}")

        for shot in timeline:
            print(f"\n  {'─'*56}")
            print(f"  SHOT {shot['shot_number']}: [{shot['timestamp']}] ({shot['duration']}s)")
            print(f"  Type:   {shot['asset_type']}")
            print(f"  Visual: {shot['description']}")
            if shot.get('script_line'):
                print(f"  Audio:  \"{shot['script_line']}\"")
            print(f"  💡 Reasoning: {shot['reasoning']}")

        print(f"\n  {'─'*56}")

        # Shot type summary
        type_counts = {}
        for shot in timeline:
            t = shot['asset_type']
            type_counts[t] = type_counts.get(t, 0) + 1

        print(f"\n  📊 Shot type summary:")
        for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"     {t}: {count} shots")

    # ─── Utilities ───────────────────────────────────────────────────

    def _check_ollama(self) -> bool:
        """Check if Ollama is running and has the required model."""
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=2)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return any(self.MODEL in m.get('name', '') for m in models)
            return False
        except Exception:
            return False
