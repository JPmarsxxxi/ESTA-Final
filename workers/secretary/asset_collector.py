"""
AssetCollector Worker - Video Asset Research and Collection

Two capabilities:
1. analyze_style(style, topic) - Called by BrainBox to study real videos
   of a target style, detecting shot patterns and visual characteristics.
2. collect_assets(video_plan) - Called by Secretary to gather actual assets
   for the final video based on BrainBox's plan.

Style Analysis Pipeline:
  YouTube search (yt-dlp) -> Download clips -> Shot detection (PySceneDetect)
  -> Visual classification (VL-JEPA or OpenCV fallback) -> Compile report
"""

import os
import re
import json
import time
import logging
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Optional dependency imports with graceful fallback ---

VLJEPA_AVAILABLE = False
try:
    # VL-JEPA requires custom install from Meta's repo:
    # https://github.com/facebookresearch/vl-jepa
    import torch
    from vl_jepa.model import VLJepaModel
    VLJEPA_AVAILABLE = True
    logger.info("✅ VL-JEPA loaded")
except ImportError:
    logger.info("ℹ️  VL-JEPA not available - will use OpenCV heuristic fallback")

CV2_AVAILABLE = False
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
    logger.info("✅ OpenCV loaded")
except ImportError:
    logger.warning("⚠️  OpenCV not available - visual classification disabled")

SCENEDETECT_AVAILABLE = False
try:
    from scenedetect import detect, ContentDetector
    SCENEDETECT_AVAILABLE = True
    logger.info("✅ PySceneDetect loaded")
except ImportError:
    logger.warning("⚠️  PySceneDetect not available - will use uniform-split fallback")


class AssetCollector:
    """
    Handles YouTube research for style analysis and asset collection
    for video production.

    Style analysis is the primary function for BrainBox integration:
    it downloads real videos, detects shots, classifies visual content,
    and produces a data-driven style report that BrainBox uses for planning.
    """

    NUM_SEARCH_RESULTS = 5   # Videos to search
    CLIP_DURATION = 45       # Seconds to download per video
    MAX_DOWNLOAD_RETRIES = 2
    DOWNLOAD_TIMEOUT = 120   # Seconds before giving up on a download
    MIN_SHOT_DURATION = 0.3  # Ignore sub-frame detection artifacts

    SHOT_TYPES = [
        'talking-head', 'b-roll', 'action', 'text-overlay',
        'aerial', 'product', 'transition', 'static'
    ]

    def __init__(self):
        self.call_count = 0
        self.temp_dir = None
        logger.info(
            f"AssetCollector initialized "
            f"(VL-JEPA: {VLJEPA_AVAILABLE}, OpenCV: {CV2_AVAILABLE}, "
            f"PySceneDetect: {SCENEDETECT_AVAILABLE})"
        )

    def _setup_temp_dir(self):
        """Create a fresh temp directory for this analysis run."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.temp_dir = Path(tempfile.mkdtemp(prefix='esta_assets_'))
        logger.info(f"Temp dir: {self.temp_dir}")

    def _cleanup_temp(self):
        """Remove temp files after analysis."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.temp_dir = None
            logger.info("Temp files cleaned up")

    # ─── Primary Entry Points ────────────────────────────────────────

    def analyze_style(self, style: str, topic: str) -> Dict[str, Any]:
        """
        Analyze real videos to learn style patterns.

        Called by BrainBox before planning. Downloads actual videos from
        YouTube, detects shot boundaries, classifies visual content, and
        compiles a style analysis report with concrete pacing data.

        Args:
            style: Target style (e.g., "funny", "documentary")
            topic: Video topic for search context

        Returns:
            Standard result dict. outputs["style_analysis"] contains the
            full report with shot duration stats, type distribution, etc.
        """
        start_time = time.time()
        self._setup_temp_dir()

        print(f"\n{'='*60}")
        print(f"STYLE ANALYSIS: {style} / {topic}")
        print(f"{'='*60}")

        try:
            # Phase 1: Search and download
            print(f"\n📹 Phase 1: Searching YouTube for '{style} {topic}' videos...")
            videos = self._youtube_search(
                f"{style} {topic}", max_results=self.NUM_SEARCH_RESULTS
            )

            if not videos:
                print("⚠️  No videos found - returning default analysis")
                return self._wrap_result(self._default_analysis(style), 0, 0)

            print(f"   Found {len(videos)} videos")

            # Download clips
            downloaded = []
            for i, video in enumerate(videos):
                title = video.get('title', f'clip_{i}')
                print(f"\n   ⬇️  Downloading clip {i+1}/{len(videos)}: {title[:50]}...")
                path = self._download_clip(video['url'], title)
                if path:
                    downloaded.append({'path': path, 'info': video})
                    size_mb = path.stat().st_size / (1024 * 1024)
                    print(f"      ✓ Downloaded ({size_mb:.1f} MB)")
                else:
                    print("      ✗ Failed")

            if not downloaded:
                print("⚠️  No clips downloaded - returning default analysis")
                return self._wrap_result(self._default_analysis(style), 0, 0)

            print(f"\n   ✓ Downloaded {len(downloaded)}/{len(videos)} clips")

            # Phase 2: Analyze each clip
            print(f"\n🔬 Phase 2: Analyzing shot patterns...")
            all_shots = []
            per_video_stats = []

            for i, clip in enumerate(downloaded):
                print(f"\n   📊 Analyzing clip {i+1}/{len(downloaded)}...")
                clip_analysis = self._analyze_clip(clip['path'])
                if clip_analysis:
                    all_shots.extend(clip_analysis['shots'])
                    per_video_stats.append(clip_analysis['stats'])
                    stats = clip_analysis['stats']
                    print(
                        f"      Shots: {stats['shot_count']}, "
                        f"Avg duration: {stats['avg_shot_duration']:.1f}s, "
                        f"Cuts/min: {stats['cuts_per_minute']:.1f}"
                    )

            # Phase 3: Compile report
            print(f"\n📋 Phase 3: Compiling style report...")
            report = self._compile_style_report(
                style, topic, all_shots, per_video_stats
            )

            elapsed = round(time.time() - start_time, 2)
            report['analysis_time'] = elapsed

            print(f"\n{'='*60}")
            print(f"STYLE ANALYSIS COMPLETE ({elapsed}s)")
            print(f"{'='*60}")
            self._print_report(report)

            return self._wrap_result(report, len(downloaded), len(all_shots))

        except Exception as e:
            logger.error(f"Style analysis failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Style analysis error: {str(e)}",
                "outputs": {"style_analysis": self._default_analysis(style)},
                "metadata": {
                    "worker": "asset_collector",
                    "timestamp": datetime.now().isoformat()
                }
            }
        finally:
            self._cleanup_temp()

    def collect_assets(self, video_plan: Dict) -> Dict[str, Any]:
        """
        Collect assets based on BrainBox's video plan.

        Called by Secretary in the main pipeline after BrainBox produces
        the shot-by-shot plan.

        Args:
            video_plan: Video plan from BrainBox with shot specifications

        Returns:
            Asset manifest with collected file paths
        """
        self.call_count += 1
        timeline = video_plan.get('timeline', [])
        print(f"\n📦 Collecting assets for {len(timeline)} shots...")

        assets = []
        for i, shot in enumerate(timeline):
            assets.append({
                "id": f"asset_{i+1:03d}",
                "shot_index": i,
                "type": shot.get('asset_type', 'video'),
                "description": shot.get('description', ''),
                "timestamp": shot.get('timestamp', ''),
                "status": "pending"
            })

        asset_manifest = {
            "assets": assets,
            "total_assets": len(assets),
            "assets_folder": "assets/"
        }

        return {
            "success": True,
            "outputs": {
                "asset_manifest": asset_manifest,
                "manifest_file": "asset_manifest.json",
                "assets_folder": "assets/"
            },
            "metadata": {
                "worker": "asset_collector",
                "mode": "collect",
                "timestamp": datetime.now().isoformat(),
                "call_count": self.call_count
            }
        }

    def _wrap_result(self, report: Dict, videos_analyzed: int, total_shots: int) -> Dict:
        """Wrap a style analysis report in the standard result envelope."""
        return {
            "success": True,
            "outputs": {"style_analysis": report},
            "metadata": {
                "worker": "asset_collector",
                "mode": "style_analysis",
                "timestamp": datetime.now().isoformat(),
                "videos_analyzed": videos_analyzed,
                "total_shots": total_shots
            }
        }

    # ─── YouTube Search & Download ───────────────────────────────────

    def _youtube_search(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search YouTube via yt-dlp and return video metadata.

        yt-dlp's ytsearch extractor returns results as a playlist wrapper.
        We parse the JSON output and filter for videos in a useful duration
        range (30s-10min) for style analysis.

        Args:
            query: Search query string
            max_results: Maximum number of results

        Returns:
            List of video dicts with url, title, duration
        """
        try:
            result = subprocess.run(
                [
                    'yt-dlp',
                    f'ytsearch{max_results}:{query}',
                    '--dump-json',
                    '--no-playlist',
                    '--no-warnings',
                ],
                capture_output=True, text=True, timeout=30
            )

            if result.returncode != 0:
                logger.warning(f"yt-dlp search error: {result.stderr[:200]}")
                return []

            videos = []
            for line in result.stdout.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get('_type') == 'playlist':
                        # ytsearch returns a playlist wrapper; entries are inside
                        for entry in data.get('entries', []):
                            if entry and (entry.get('url') or entry.get('webpage_url')):
                                videos.append({
                                    'url': entry.get('webpage_url') or entry['url'],
                                    'title': entry.get('title', 'Unknown'),
                                    'duration': entry.get('duration', 0)
                                })
                    elif data.get('url') or data.get('webpage_url'):
                        videos.append({
                            'url': data.get('webpage_url') or data['url'],
                            'title': data.get('title', 'Unknown'),
                            'duration': data.get('duration', 0)
                        })
                except json.JSONDecodeError:
                    continue

            # Prefer videos 30s-10min (good for style analysis)
            filtered = [v for v in videos if 30 <= (v.get('duration') or 0) <= 600]
            return filtered[:max_results] if filtered else videos[:max_results]

        except subprocess.TimeoutExpired:
            logger.warning("YouTube search timed out")
            return []
        except FileNotFoundError:
            logger.error("yt-dlp not installed. Run: pip install yt-dlp")
            return []
        except Exception as e:
            logger.error(f"YouTube search failed: {e}")
            return []

    def _download_clip(self, url: str, title: str) -> Optional[Path]:
        """
        Download the first CLIP_DURATION seconds of a video.

        Uses 360p resolution since we only need visual analysis, not
        playback quality. Tries --trim first; if that fails (older yt-dlp
        versions), falls back to full download + ffmpeg trim.

        Args:
            url: YouTube URL
            title: Video title (for filename)

        Returns:
            Path to downloaded file, or None if all attempts failed
        """
        safe_title = re.sub(r'[^\w\s-]', '', title)[:40].strip()
        output_path = self.temp_dir / f"{safe_title}.mp4"

        # Attempt 1: yt-dlp --trim (clips server-side or during download)
        try:
            result = subprocess.run(
                [
                    'yt-dlp', url,
                    '-f', 'best[height<=360][ext=mp4] / worst[ext=mp4] / best',
                    '--trim', f'0-{self.CLIP_DURATION}',
                    '-o', str(output_path),
                    '--no-playlist', '--no-warnings', '-q',
                ],
                capture_output=True, text=True, timeout=self.DOWNLOAD_TIMEOUT
            )
            if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000:
                return output_path
        except (subprocess.TimeoutExpired, Exception) as e:
            logger.info(f"--trim attempt failed: {e}")

        # Attempt 2: Download then trim with ffmpeg
        output_path_raw = self.temp_dir / f"{safe_title}_raw.mp4"
        try:
            result = subprocess.run(
                [
                    'yt-dlp', url,
                    '-f', 'best[height<=360][ext=mp4] / worst[ext=mp4] / best',
                    '-o', str(output_path_raw),
                    '--no-playlist', '--no-warnings', '-q',
                ],
                capture_output=True, text=True, timeout=self.DOWNLOAD_TIMEOUT
            )

            if result.returncode == 0 and output_path_raw.exists():
                subprocess.run(
                    [
                        'ffmpeg', '-i', str(output_path_raw),
                        '-t', str(self.CLIP_DURATION),
                        '-c', 'copy', str(output_path),
                        '-y', '-hide_banner', '-loglevel', 'error'
                    ],
                    capture_output=True, timeout=30
                )
                output_path_raw.unlink(missing_ok=True)

                if output_path.exists() and output_path.stat().st_size > 1000:
                    return output_path
        except (subprocess.TimeoutExpired, Exception) as e:
            logger.warning(f"Download+trim failed: {e}")

        return None

    # ─── Shot Detection ──────────────────────────────────────────────

    def _analyze_clip(self, video_path: Path) -> Optional[Dict]:
        """
        Full analysis pipeline for a single clip:
        detect shots -> classify each shot -> compute per-clip stats.

        Args:
            video_path: Path to downloaded video clip

        Returns:
            Dict with 'shots' list and 'stats' summary, or None on failure
        """
        shots = self._detect_shots(video_path)
        if not shots:
            return None

        classified_shots = self._classify_shots(video_path, shots)

        durations = [s['duration'] for s in classified_shots]
        total_duration = sum(durations)
        sorted_durs = sorted(durations)
        n = len(sorted_durs)

        stats = {
            'shot_count': len(classified_shots),
            'total_duration': round(total_duration, 2),
            'avg_shot_duration': round(total_duration / n, 2) if n else 0,
            'median_shot_duration': round(sorted_durs[n // 2], 2) if n else 0,
            'min_shot_duration': round(min(durations), 2) if durations else 0,
            'max_shot_duration': round(max(durations), 2) if durations else 0,
            'cuts_per_minute': round((n - 1) / (total_duration / 60), 2) if total_duration > 0 and n > 1 else 0,
            'shot_type_distribution': dict(Counter(s['type'] for s in classified_shots)),
        }

        return {'shots': classified_shots, 'stats': stats}

    def _detect_shots(self, video_path: Path) -> List[Dict]:
        """
        Detect shot boundaries using PySceneDetect's ContentDetector.

        ContentDetector looks for frame-to-frame content changes (color
        histogram + edge differences). threshold=30 balances sensitivity
        vs noise for typical YouTube content.

        Falls back to uniform splitting if PySceneDetect is unavailable
        or returns no scenes.

        Returns:
            List of shot dicts with start_time, end_time, duration
        """
        if not SCENEDETECT_AVAILABLE:
            logger.warning("PySceneDetect not available - using uniform split")
            return self._detect_shots_fallback(video_path)

        try:
            scenes = detect(str(video_path), ContentDetector(threshold=30))

            shots = []
            for start_tc, end_tc in scenes:
                start_sec = start_tc.get_seconds()
                end_sec = end_tc.get_seconds()
                duration = end_sec - start_sec

                if duration < self.MIN_SHOT_DURATION:
                    continue

                shots.append({
                    'start_time': round(start_sec, 2),
                    'end_time': round(end_sec, 2),
                    'duration': round(duration, 2)
                })

            if not shots:
                return self._detect_shots_fallback(video_path)

            logger.info(f"  Detected {len(shots)} shots via PySceneDetect")
            return shots

        except Exception as e:
            logger.warning(f"PySceneDetect failed: {e} - using fallback")
            return self._detect_shots_fallback(video_path)

    def _detect_shots_fallback(self, video_path: Path) -> List[Dict]:
        """
        Uniform-segment fallback when PySceneDetect is unavailable.
        Uses OpenCV to read actual video duration; splits into ~6 segments.
        """
        if not CV2_AVAILABLE:
            # Absolute fallback: assume 45s, 5 equal shots
            return [
                {'start_time': i * 9.0, 'end_time': (i + 1) * 9.0, 'duration': 9.0}
                for i in range(5)
            ]

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        duration = frame_count / fps if fps > 0 else 45.0
        cap.release()

        segment_duration = max(3.0, min(10.0, duration / 6))
        shots = []
        t = 0.0
        while t < duration:
            end = min(t + segment_duration, duration)
            shots.append({
                'start_time': round(t, 2),
                'end_time': round(end, 2),
                'duration': round(end - t, 2)
            })
            t = end

        return shots

    # ─── Visual Classification ──────────────────────────────────────

    def _classify_shots(self, video_path: Path, shots: List[Dict]) -> List[Dict]:
        """
        Classify visual content of each shot.
        Routes to VL-JEPA if available, otherwise OpenCV heuristics.
        """
        if VLJEPA_AVAILABLE:
            return self._classify_shots_vljepa(video_path, shots)
        logger.info("  Using OpenCV heuristic classification")
        return self._classify_shots_opencv(video_path, shots)

    def _classify_shots_vljepa(self, video_path: Path, shots: List[Dict]) -> List[Dict]:
        """
        VL-JEPA classification: extracts video embeddings per shot and
        matches against text-described category prototypes via cosine
        similarity (same paradigm as CLIP but using VL-JEPA's encoders).

        Falls back to OpenCV on any error.
        """
        try:
            model = VLJepaModel.load_pretrained()
            model.eval()
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = model.to(device)

            classified = []
            for shot in shots:
                frames = self._sample_frames(
                    video_path, shot['start_time'], shot['end_time'], num_frames=8
                )

                if not frames:
                    shot['type'] = 'unknown'
                    shot['motion_level'] = 'medium'
                    shot['complexity'] = 'medium'
                    classified.append(shot)
                    continue

                frame_tensor = torch.tensor(frames).unsqueeze(0).to(device)
                with torch.no_grad():
                    embeddings = model.encode_video(frame_tensor)

                shot['type'] = self._match_vljepa_prototype(embeddings, model, device)
                motion_level, complexity = self._compute_frame_stats(frames)
                shot['motion_level'] = motion_level
                shot['complexity'] = complexity
                classified.append(shot)

            return classified

        except Exception as e:
            logger.warning(f"VL-JEPA classification failed: {e} - falling back to OpenCV")
            return self._classify_shots_opencv(video_path, shots)

    def _match_vljepa_prototype(self, embeddings, model, device) -> str:
        """
        Match a video embedding against category prototypes using
        VL-JEPA's text encoder + cosine similarity.
        """
        categories = {
            'talking-head': 'a person talking to camera, face visible, close up',
            'b-roll': 'scenic footage, no person talking, establishing wide shot',
            'action': 'fast moving, energetic, dynamic physical action',
            'text-overlay': 'text displayed on screen, graphics, title card, infographic',
            'aerial': 'aerial view, drone shot, bird eye view from above',
            'product': 'product display, showcase single item, close up of object',
            'transition': 'transition effect, fade to black, dissolve between scenes',
            'static': 'still image, frozen frame, no movement at all'
        }

        best_type = 'b-roll'
        best_score = -1.0

        for cat_name, description in categories.items():
            try:
                text_emb = model.encode_text(description, device=device)
                score = torch.nn.functional.cosine_similarity(
                    embeddings, text_emb
                ).item()
                if score > best_score:
                    best_score = score
                    best_type = cat_name
            except Exception:
                continue

        return best_type

    def _classify_shots_opencv(self, video_path: Path, shots: List[Dict]) -> List[Dict]:
        """
        OpenCV heuristic classification when VL-JEPA is unavailable.

        Per shot, samples up to 10 frames and computes:
        - Motion: mean frame-to-frame pixel difference
        - Face presence: Haar cascade detection (proxy for talking-head)
        - Edge density: Canny edge ratio (proxy for text-overlay)
        - Brightness variance: detects hard cuts / transitions
        - Color variety: number of distinct hue bins (scene complexity)

        Classification thresholds were tuned against typical YouTube content.
        """
        if not CV2_AVAILABLE:
            return [
                {**s, 'type': 'unknown', 'motion_level': 'medium', 'complexity': 'medium'}
                for s in shots
            ]

        # Load face cascade detector
        face_cascade = None
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if os.path.exists(cascade_path):
            face_cascade = cv2.CascadeClassifier(cascade_path)
            if face_cascade.empty():
                face_cascade = None

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        classified = []

        for shot in shots:
            start_frame = int(shot['start_time'] * fps)
            end_frame = int(shot['end_time'] * fps)
            num_frames_in_shot = max(1, end_frame - start_frame)

            # Sample up to 10 evenly spaced frames
            sample_count = min(10, num_frames_in_shot)
            step = max(1, num_frames_in_shot // sample_count)

            frames = []
            for i in range(sample_count):
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + (i * step))
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)

            if len(frames) < 2:
                shot['type'] = 'unknown'
                shot['motion_level'] = 'medium'
                shot['complexity'] = 'medium'
                classified.append(shot)
                continue

            # --- Motion: mean absolute difference between consecutive gray frames ---
            motion_scores = []
            for i in range(1, len(frames)):
                prev_gray = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(prev_gray, curr_gray)
                motion_scores.append(np.mean(diff))
            avg_motion = np.mean(motion_scores) if motion_scores else 0.0

            # --- Face detection on first 3 frames ---
            face_count = 0
            if face_cascade is not None:
                for frame in frames[:3]:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                    )
                    face_count += len(faces)
            has_face = face_count > 0

            # --- Edge density: ratio of Canny edge pixels ---
            edge_densities = []
            for frame in frames[:3]:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                edge_densities.append(np.mean(edges) / 255.0)
            avg_edge_density = np.mean(edge_densities) if edge_densities else 0.0

            # --- Brightness variance across frames (hard cuts cause spikes) ---
            brightnesses = [
                np.mean(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)) for f in frames
            ]
            brightness_var = np.var(brightnesses) if len(brightnesses) > 1 else 0.0

            # --- Classification ---
            MOTION_HIGH = 30.0
            MOTION_LOW = 8.0
            EDGE_TEXT_THRESHOLD = 0.15
            BRIGHTNESS_VAR_TRANSITION = 500.0

            if brightness_var > BRIGHTNESS_VAR_TRANSITION and avg_motion > MOTION_HIGH:
                shot_type = 'transition'
            elif avg_edge_density > EDGE_TEXT_THRESHOLD and avg_motion < MOTION_LOW:
                shot_type = 'text-overlay'
            elif has_face and avg_motion < MOTION_HIGH:
                shot_type = 'talking-head'
            elif avg_motion > MOTION_HIGH:
                shot_type = 'action'
            elif avg_motion < MOTION_LOW:
                shot_type = 'static'
            else:
                shot_type = 'b-roll'

            # Motion level
            if avg_motion > MOTION_HIGH:
                motion_level = 'high'
            elif avg_motion > MOTION_LOW:
                motion_level = 'medium'
            else:
                motion_level = 'low'

            # Complexity: edge density + color variety
            color_counts = []
            for frame in frames[:3]:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hue_hist, _ = np.histogram(hsv[:, :, 0], bins=8, range=(0, 180))
                color_counts.append(int(np.sum(hue_hist > 0)))
            avg_colors = np.mean(color_counts)

            if avg_edge_density > 0.12 and avg_colors > 5:
                complexity = 'high'
            elif avg_edge_density > 0.06 or avg_colors > 3:
                complexity = 'medium'
            else:
                complexity = 'low'

            shot['type'] = shot_type
            shot['motion_level'] = motion_level
            shot['complexity'] = complexity
            shot['_metrics'] = {
                'avg_motion': round(float(avg_motion), 2),
                'has_face': has_face,
                'edge_density': round(float(avg_edge_density), 3),
                'brightness_var': round(float(brightness_var), 1),
                'avg_colors': round(float(avg_colors), 1)
            }
            classified.append(shot)

        cap.release()
        return classified

    def _sample_frames(self, video_path: Path, start_time: float,
                       end_time: float, num_frames: int = 8) -> Optional[List]:
        """
        Sample frames from a video segment, preprocessed for VL-JEPA input.
        Resizes to 224x224, normalizes with ImageNet stats, converts to CHW.
        """
        if not CV2_AVAILABLE:
            return None

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        total_frames = end_frame - start_frame

        if total_frames <= 0:
            cap.release()
            return None

        step = max(1, total_frames // num_frames)
        frames = []

        for i in range(num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + (i * step))
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype('float32') / 255.0
                # ImageNet normalization
                frame = (frame - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                frame = frame.transpose(2, 0, 1)  # HWC -> CHW
                frames.append(frame)

        cap.release()
        return frames if frames else None

    def _compute_frame_stats(self, frames: List) -> tuple:
        """
        Compute motion level and complexity from preprocessed frame list.
        Used as a supplement to VL-JEPA classification.
        """
        if not CV2_AVAILABLE or len(frames) < 2:
            return 'medium', 'medium'

        # Motion from frame differences (frames are already CHW float tensors)
        motion_scores = []
        for i in range(1, len(frames)):
            diff = np.mean(np.abs(frames[i] - frames[i - 1]))
            motion_scores.append(diff)
        avg_motion = np.mean(motion_scores) if motion_scores else 0.0

        if avg_motion > 0.15:
            motion_level = 'high'
        elif avg_motion > 0.05:
            motion_level = 'medium'
        else:
            motion_level = 'low'

        # Complexity from channel variance
        avg_var = np.mean([np.var(f) for f in frames])
        if avg_var > 0.05:
            complexity = 'high'
        elif avg_var > 0.02:
            complexity = 'medium'
        else:
            complexity = 'low'

        return motion_level, complexity

    # ─── Report Compilation ──────────────────────────────────────────

    def _compile_style_report(self, style: str, topic: str,
                              all_shots: List[Dict],
                              per_video_stats: List[Dict]) -> Dict:
        """
        Aggregate shot data from all analyzed videos into a single
        style report. This is what BrainBox receives for planning.
        """
        if not per_video_stats or not CV2_AVAILABLE:
            return self._default_analysis(style)

        all_durations = [s['duration'] for s in all_shots]
        all_shot_types = [s['type'] for s in all_shots]
        all_motion_levels = [s.get('motion_level', 'medium') for s in all_shots]

        sorted_durations = sorted(all_durations) if all_durations else [3.0]
        n = len(sorted_durations)

        duration_stats = {
            'mean': round(float(np.mean(all_durations)), 2),
            'median': round(sorted_durations[n // 2], 2),
            'p25': round(sorted_durations[n // 4], 2),
            'p75': round(sorted_durations[3 * n // 4], 2),
            'min': round(min(all_durations), 2),
            'max': round(max(all_durations), 2),
            'std': round(float(np.std(all_durations)), 2) if n > 1 else 0.0
        }

        total_shots = len(all_shot_types) or 1
        type_counts = Counter(all_shot_types)
        type_distribution = {
            t: round(c / total_shots, 3) for t, c in type_counts.most_common()
        }

        motion_counts = Counter(all_motion_levels)
        motion_distribution = {
            m: round(c / total_shots, 3) for m, c in motion_counts.most_common()
        }

        avg_cuts_per_minute = round(
            float(np.mean([s['cuts_per_minute'] for s in per_video_stats])), 2
        )
        avg_shot_dur = round(
            float(np.mean([s['avg_shot_duration'] for s in per_video_stats])), 2
        )

        duration_buckets = {
            '0-2s': sum(1 for d in all_durations if d < 2),
            '2-5s': sum(1 for d in all_durations if 2 <= d < 5),
            '5-10s': sum(1 for d in all_durations if 5 <= d < 10),
            '10-20s': sum(1 for d in all_durations if 10 <= d < 20),
            '20s+': sum(1 for d in all_durations if d >= 20),
        }

        return {
            'style': style,
            'topic': topic,
            'videos_analyzed': len(per_video_stats),
            'total_shots_analyzed': len(all_shots),
            'pacing': {
                'avg_cuts_per_minute': avg_cuts_per_minute,
                'avg_shot_duration': avg_shot_dur,
                'duration_buckets': duration_buckets,
            },
            'shot_durations': duration_stats,
            'shot_type_distribution': type_distribution,
            'motion_distribution': motion_distribution,
            'classification_method': 'vl-jepa' if VLJEPA_AVAILABLE else 'opencv-heuristic',
            'per_video_stats': per_video_stats,
        }

    def _print_report(self, report: Dict):
        """Pretty-print the style analysis report."""
        print(f"\n{'─'*60}")
        print(f"  Style: {report['style']} | Topic: {report['topic']}")
        print(f"  Videos analyzed: {report['videos_analyzed']} | "
              f"Total shots: {report['total_shots_analyzed']}")
        print(f"  Classification: {report['classification_method']}")
        print(f"{'─'*60}")

        pacing = report.get('pacing', {})
        print(f"\n  📊 PACING:")
        print(f"     Cuts/min:        {pacing.get('avg_cuts_per_minute', '?')}")
        print(f"     Avg shot dur:    {pacing.get('avg_shot_duration', '?')}s")

        buckets = pacing.get('duration_buckets', {})
        print(f"\n     Duration spread:")
        for bucket, count in buckets.items():
            bar = '█' * count
            print(f"       {bucket:>6}: {bar} ({count})")

        durations = report.get('shot_durations', {})
        print(f"\n  ⏱️  SHOT DURATIONS:")
        print(f"     Mean: {durations.get('mean', '?')}s | "
              f"Median: {durations.get('median', '?')}s | "
              f"Std: {durations.get('std', '?')}s")
        print(f"     P25: {durations.get('p25', '?')}s | "
              f"P75: {durations.get('p75', '?')}s | "
              f"Range: [{durations.get('min', '?')}-{durations.get('max', '?')}s]")

        print(f"\n  🎬 SHOT TYPES:")
        for stype, ratio in report.get('shot_type_distribution', {}).items():
            bar = '█' * int(ratio * 30)
            print(f"     {stype:>14}: {bar} {ratio * 100:.0f}%")

        print(f"\n  🏃 MOTION LEVELS:")
        for mlevel, ratio in report.get('motion_distribution', {}).items():
            bar = '█' * int(ratio * 30)
            print(f"     {mlevel:>8}: {bar} {ratio * 100:.0f}%")

        print(f"{'─'*60}")

    def _default_analysis(self, style: str) -> Dict:
        """
        Neutral default analysis when no videos could be analyzed.
        Values are intentionally generic. BrainBox should flag that
        this is a fallback and apply extra creative judgment.
        """
        return {
            'style': style,
            'topic': 'unknown',
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
            'per_video_stats': [],
            'fallback': True
        }
