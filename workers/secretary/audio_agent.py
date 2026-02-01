"""
Audio Agent Worker - Real Implementation

Generates timed audio from script using:
- Coqui TTS XTTS v2 for voice generation
- Whisper for precise timestamp extraction
"""

import os
import json
import logging
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioAgent:
    """
    AudioAgent generates timed audio from scripts using:
    1. Voice selection (style-based, pre-recorded, or user-uploaded)
    2. TTS generation with Coqui XTTS v2
    3. Timestamp extraction with Whisper
    4. Metadata generation
    """

    # Voice mapping for different styles
    STYLE_VOICE_MAP = {
        "documentary": "male_professional",
        "tutorial": "female_clear",
        "funny": "male_energetic",
        "serious": "male_authoritative",
        "graphic-heavy": "neutral_narrator"
    }

    def __init__(
        self,
        output_dir: str = None,
        use_gpu: bool = True
    ):
        """
        Initialize AudioAgent

        Args:
            output_dir: Directory for audio output files (defaults to audio_outputs/ next to this script)
            use_gpu: Use GPU for TTS if available
        """
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_outputs")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_gpu = use_gpu
        self.call_count = 0

        # Lazy load models (loaded on first use)
        self.tts_model = None
        self.whisper_model = None

        logger.info(f"AudioAgent initialized:")
        logger.info(f"  - Output directory: {self.output_dir}")
        logger.info(f"  - GPU enabled: {self.use_gpu}")

    def generate_audio(
        self,
        script: str,
        style: Optional[str] = None,
        voice_sample: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main method to generate audio from script

        Args:
            script: Script text to convert to audio
            style: Video style (for voice selection)
            voice_sample: Optional path to custom voice sample

        Returns:
            Dictionary with success, outputs, and metadata
        """
        self.call_count += 1
        logger.info("="*60)
        logger.info(f"AUDIO AGENT CALL #{self.call_count} - Starting audio generation")
        logger.info("="*60)

        try:
            # Step 1: Voice Selection
            logger.info("\n" + "-"*60)
            logger.info("STEP 1: Voice Selection")
            logger.info("-"*60)

            voice_path = self._select_voice(style, voice_sample)
            logger.info(f"Selected voice: {voice_path}")

            # Step 2: Parse Script
            logger.info("\n" + "-"*60)
            logger.info("STEP 2: Parsing Script")
            logger.info("-"*60)

            script_length = len(script)
            word_count = len(script.split())
            logger.info(f"Script length: {script_length} characters")
            logger.info(f"Word count: {word_count} words")
            logger.info(f"Estimated duration: ~{word_count / 150:.1f} minutes")

            # Step 3: Generate Audio with TTS
            logger.info("\n" + "-"*60)
            logger.info("STEP 3: Generating Audio (XTTS v2)")
            logger.info("-"*60)

            audio_path = self._generate_tts(script, voice_path)
            logger.info(f"Audio generated: {audio_path}")

            # Step 4: Extract Timestamps with Whisper
            logger.info("\n" + "-"*60)
            logger.info("STEP 4: Extracting Timestamps (Whisper)")
            logger.info("-"*60)

            transcript_data = self._extract_timestamps(audio_path, script)
            logger.info(f"Timestamp extraction complete")
            logger.info(f"Total segments: {len(transcript_data['segments'])}")
            logger.info(f"Total duration: {transcript_data['total_duration']:.2f}s")

            # Step 5: Generate Metadata
            logger.info("\n" + "-"*60)
            logger.info("STEP 5: Generating Metadata")
            logger.info("-"*60)

            metadata = self._generate_metadata(audio_path)
            logger.info(f"Sample rate: {metadata['sample_rate']} Hz")
            logger.info(f"File size: {metadata['file_size_mb']:.2f} MB")

            # Prepare output
            logger.info("\n" + "-"*60)
            logger.info("STEP 6: Preparing Output")
            logger.info("-"*60)

            result = {
                "success": True,
                "outputs": {
                    "audio_file": audio_path.name,
                    "audio_path": str(audio_path),
                    "transcript_timestamps": transcript_data,
                    "duration": transcript_data['total_duration'],
                    "sample_rate": metadata['sample_rate'],
                    "file_size_mb": metadata['file_size_mb']
                },
                "metadata": {
                    "worker": "audio_agent",
                    "timestamp": datetime.now().isoformat(),
                    "call_count": self.call_count,
                    "voice_used": voice_path,
                    "word_count": word_count
                }
            }

            logger.info("✅ Audio generation successful!")
            logger.info("="*60)

            return result

        except Exception as e:
            logger.error(f"❌ Audio generation failed: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": f"AudioAgent error: {str(e)}",
                "outputs": {},
                "metadata": {
                    "worker": "audio_agent",
                    "timestamp": datetime.now().isoformat(),
                    "call_count": self.call_count
                }
            }

    def _select_voice(
        self,
        style: Optional[str],
        voice_sample: Optional[str]
    ) -> str:
        """
        Select voice based on strategy (A/B/C)

        Strategy priority:
        1. User-uploaded voice sample (Option B)
        2. Pre-recorded style voice (Option A)
        3. XTTS default voice (Option C)

        Args:
            style: Video style
            voice_sample: Custom voice sample path

        Returns:
            Path to voice sample file
        """
        # Option B: User-uploaded voice
        if voice_sample and os.path.exists(voice_sample):
            logger.info(f"Using user-uploaded voice: {voice_sample}")
            return voice_sample

        # Option A: Pre-recorded voice for style
        voice_dir = self.output_dir.parent / "voice_samples"
        voice_dir.mkdir(parents=True, exist_ok=True)

        if style and style in self.STYLE_VOICE_MAP:
            voice_name = self.STYLE_VOICE_MAP[style]
            voice_path = voice_dir / f"{voice_name}.wav"

            if voice_path.exists():
                logger.info(f"Using pre-recorded voice for style '{style}': {voice_name}")
                return str(voice_path)

        # Option C: XTTS default voice (fallback)
        logger.info("Using XTTS default voice (no custom sample provided)")
        return None  # Will use XTTS default

    def _generate_tts(self, script: str, voice_path: Optional[str]) -> Path:
        """
        Generate audio using Coqui TTS XTTS v2

        Args:
            script: Text to synthesize
            voice_path: Path to voice sample or None for default

        Returns:
            Path to generated audio file
        """
        logger.info("Loading TTS model...")

        try:
            # Lazy load TTS model
            if self.tts_model is None:
                from TTS.api import TTS
                logger.info("Initializing XTTS v2 model (first time may take a while)...")
                self.tts_model = TTS(
                    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                    gpu=self.use_gpu
                )
                logger.info("✅ TTS model loaded")

            # Generate output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"audio_{timestamp}.wav"

            logger.info(f"Generating audio to: {output_path}")
            logger.info(f"Script length: {len(script)} chars")

            # Generate audio
            if voice_path:
                logger.info(f"Using voice cloning with: {voice_path}")
                self.tts_model.tts_to_file(
                    text=script,
                    speaker_wav=voice_path,
                    language="en",
                    file_path=str(output_path)
                )
            else:
                logger.info("Using default XTTS voice")
                # Use default speaker
                self.tts_model.tts_to_file(
                    text=script,
                    language="en",
                    file_path=str(output_path),
                    speaker="Claribel Dervla"  # XTTS default speaker
                )

            logger.info(f"✅ Audio file generated: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"TTS generation failed: {str(e)}", exc_info=True)
            raise

    def _extract_timestamps(self, audio_path: Path, script: str) -> Dict[str, Any]:
        """
        Extract precise timestamps using Whisper

        Args:
            audio_path: Path to audio file
            script: Original script text

        Returns:
            Dictionary with segments and timing data
        """
        logger.info("Loading Whisper model...")

        try:
            # Lazy load Whisper
            if self.whisper_model is None:
                import whisper
                logger.info("Loading Whisper base model...")
                self.whisper_model = whisper.load_model("base")
                logger.info("✅ Whisper model loaded")

            logger.info(f"Transcribing audio: {audio_path}")

            # Transcribe with word-level timestamps
            result = self.whisper_model.transcribe(
                str(audio_path),
                word_timestamps=True,
                language="en"
            )

            logger.info(f"Transcription complete")
            logger.info(f"Detected text: {result['text'][:100]}...")

            # Process segments
            segments = []
            silence_points = []

            for segment in result['segments']:
                seg_data = {
                    "start": segment['start'],
                    "end": segment['end'],
                    "text": segment['text'].strip(),
                    "type": self._classify_segment(segment['start'], result['segments'])
                }
                segments.append(seg_data)

                # Detect silence (gap > 1 second)
                if len(segments) > 1:
                    gap = segment['start'] - segments[-2]['end']
                    if gap > 1.0:
                        silence_points.append(segments[-2]['end'])

            total_duration = result['segments'][-1]['end'] if result['segments'] else 0.0

            logger.info(f"Processed {len(segments)} segments")
            logger.info(f"Detected {len(silence_points)} silence points")

            return {
                "segments": segments,
                "total_duration": total_duration,
                "silence_points": silence_points,
                "full_text": result['text']
            }

        except Exception as e:
            logger.error(f"Timestamp extraction failed: {str(e)}", exc_info=True)
            raise

    def _classify_segment(self, start_time: float, all_segments: list) -> str:
        """
        Classify segment as hook/body/cta based on timing

        Args:
            start_time: Segment start time
            all_segments: All segments for context

        Returns:
            Segment type (hook/body/cta)
        """
        total_duration = all_segments[-1]['end'] if all_segments else 0

        if start_time < 15:  # First 15 seconds
            return "hook"
        elif start_time > total_duration - 15:  # Last 15 seconds
            return "cta"
        else:
            return "body"

    def _generate_metadata(self, audio_path: Path) -> Dict[str, Any]:
        """
        Generate audio metadata

        Args:
            audio_path: Path to audio file

        Returns:
            Metadata dictionary
        """
        import wave

        try:
            with wave.open(str(audio_path), 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                frames = wav_file.getnframes()

            file_size_mb = audio_path.stat().st_size / (1024 * 1024)

            return {
                "sample_rate": sample_rate,
                "channels": channels,
                "sample_width": sample_width,
                "frames": frames,
                "file_size_mb": file_size_mb
            }

        except Exception as e:
            logger.warning(f"Could not extract metadata: {e}")
            return {
                "sample_rate": 22050,
                "channels": 1,
                "file_size_mb": 0.0
            }
