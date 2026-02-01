"""
Tool Registry and Mock Workers for Secretary Module

This module defines the available tools (workers) and provides mock implementations
for testing the tool calling system.
"""

import json
import logging
from typing import Dict, List, Any
from datetime import datetime

# Configure logging FIRST
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import real workers
try:
    from scriptwriter import ScriptWriter as RealScriptWriter
    REAL_SCRIPTWRITER_AVAILABLE = True
    logger.info("✅ Real ScriptWriter imported successfully")
except ImportError as e:
    REAL_SCRIPTWRITER_AVAILABLE = False
    logger.warning(f"⚠️  Real ScriptWriter not available: {e}")

try:
    from audio_agent import AudioAgent as RealAudioAgent
    REAL_AUDIOAGENT_AVAILABLE = True
    logger.info("✅ Real AudioAgent imported successfully")
except ImportError as e:
    REAL_AUDIOAGENT_AVAILABLE = False
    logger.warning(f"⚠️  Real AudioAgent not available: {e}")

try:
    from langsearch import LangSearch as RealLangSearch
    REAL_LANGSEARCH_AVAILABLE = True
    logger.info("✅ Real LangSearch imported successfully")
except ImportError as e:
    REAL_LANGSEARCH_AVAILABLE = False
    logger.warning(f"⚠️  Real LangSearch not available: {e}")

try:
    from asset_collector import AssetCollector as RealAssetCollector
    REAL_ASSETCOLLECTOR_AVAILABLE = True
    logger.info("✅ Real AssetCollector imported successfully")
except ImportError as e:
    REAL_ASSETCOLLECTOR_AVAILABLE = False
    logger.warning(f"⚠️  Real AssetCollector not available: {e}")

try:
    from brainbox import BrainBox as RealBrainBox
    REAL_BRAINBOX_AVAILABLE = True
    logger.info("✅ Real BrainBox imported successfully")
except ImportError as e:
    REAL_BRAINBOX_AVAILABLE = False
    logger.warning(f"⚠️  Real BrainBox not available: {e}")


# Tool Registry - Defines all available workers and their specifications
AVAILABLE_TOOLS = {
    "scriptwriter": {
        "description": "Generates video script based on user requirements",
        "required_inputs": ["topic", "style", "duration_range"],
        "optional_inputs": ["comments"],
        "outputs": ["script.txt"],
        "execution_time": "30-60 seconds"
    },
    "audio_agent": {
        "description": "Generates timed audio transcript from script",
        "required_inputs": ["script"],
        "optional_inputs": [],
        "outputs": ["audio.wav", "transcript_timestamps.json"],
        "execution_time": "20-40 seconds"
    },
    "langsearch": {
        "description": "Researches terms via web search. Script mode extracts and researches key terms; direct mode searches a specific query",
        "required_inputs": [],
        "optional_inputs": ["script", "query"],
        "outputs": ["research_data.json"],
        "execution_time": "10-30 seconds"
    },
    "brainbox": {
        "description": "Creates detailed video plan with timeline and asset specifications",
        "required_inputs": ["script", "transcript_timestamps", "research_data", "requirements"],
        "optional_inputs": [],
        "outputs": ["video_plan.json"],
        "execution_time": "40-90 seconds"
    },
    "asset_collector": {
        "description": "Collects and generates all required assets for video",
        "required_inputs": ["video_plan"],
        "optional_inputs": [],
        "outputs": ["assets/", "asset_manifest.json"],
        "execution_time": "60-300 seconds"
    },
    "executor": {
        "description": "Assembles and renders final video from assets and plan",
        "required_inputs": ["video_plan", "assets", "audio"],
        "optional_inputs": [],
        "outputs": ["final_video.mp4"],
        "execution_time": "120-600 seconds"
    }
}


# Mock Worker Implementations
class MockScriptWriter:
    """Mock implementation of ScriptWriter worker"""

    def __init__(self):
        self.call_count = 0
        logger.info("MockScriptWriter initialized")

    def generate_script(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mock script generation

        Args:
            requirements: User requirements dictionary

        Returns:
            Dictionary with script and metadata
        """
        self.call_count += 1
        topic = requirements.get('topic', 'Unknown Topic')
        style = requirements.get('style', 'Unknown Style')
        duration = requirements.get('duration_range', 'Unknown Duration')

        logger.info(f"MockScriptWriter called (count: {self.call_count}) for topic: {topic}")

        # Generate mock script
        script = f"""[MOCK SCRIPT - Generated on {datetime.now().isoformat()}]

Topic: {topic}
Style: {style}
Target Duration: {duration}

HOOK (0:00-0:15):
Welcome to this {style} video about {topic}!

BODY (0:15-1:45):
Let me tell you all about {topic}. This is incredibly important because...
[Main content would go here in a real script]

CALL TO ACTION (1:45-2:00):
Thanks for watching! Don't forget to like and subscribe!

[END OF MOCK SCRIPT]
"""

        return {
            "success": True,
            "outputs": {
                "script": script,
                "script_file": "script.txt",
                "word_count": len(script.split()),
                "estimated_duration": duration
            },
            "metadata": {
                "worker": "scriptwriter",
                "timestamp": datetime.now().isoformat(),
                "call_count": self.call_count
            }
        }


class MockAudioAgent:
    """Mock implementation of Audio Agent worker"""

    def __init__(self):
        self.call_count = 0
        logger.info("MockAudioAgent initialized")

    def generate_audio(self, script: str) -> Dict[str, Any]:
        """
        Mock audio generation

        Args:
            script: Script text to convert to audio

        Returns:
            Dictionary with audio file and transcript timestamps
        """
        self.call_count += 1
        logger.info(f"MockAudioAgent called (count: {self.call_count})")

        # Generate mock transcript with timestamps
        transcript_timestamps = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 15.0,
                    "text": "Welcome to this video...",
                    "type": "narration"
                },
                {
                    "start": 15.0,
                    "end": 105.0,
                    "text": "Main content...",
                    "type": "narration"
                },
                {
                    "start": 105.0,
                    "end": 120.0,
                    "text": "Thanks for watching...",
                    "type": "narration"
                }
            ],
            "total_duration": 120.0,
            "silence_points": [14.5, 104.5]
        }

        return {
            "success": True,
            "outputs": {
                "audio_file": "mock_audio.wav",
                "transcript_timestamps": transcript_timestamps,
                "duration": 120.0,
                "sample_rate": 44100
            },
            "metadata": {
                "worker": "audio_agent",
                "timestamp": datetime.now().isoformat(),
                "call_count": self.call_count
            }
        }


class MockLangSearch:
    """Mock implementation of LangSearch worker"""

    def __init__(self):
        self.call_count = 0
        logger.info("MockLangSearch initialized")

    def research_terms(self, script: str) -> Dict[str, Any]:
        """
        Mock research and term extraction

        Args:
            script: Script text to research

        Returns:
            Dictionary with research data
        """
        self.call_count += 1
        logger.info(f"MockLangSearch called (count: {self.call_count})")

        # Generate mock research data
        research_data = {
            "terms": [
                {
                    "term": "example term 1",
                    "context": "Mock context information...",
                    "relevance_score": 0.85,
                    "sources": ["https://example.com/1"]
                },
                {
                    "term": "example term 2",
                    "context": "Mock context information...",
                    "relevance_score": 0.72,
                    "sources": ["https://example.com/2"]
                }
            ],
            "total_terms_found": 2,
            "search_time": 5.2
        }

        return {
            "success": True,
            "outputs": {
                "research_data": research_data,
                "research_file": "research_data.json"
            },
            "metadata": {
                "worker": "langsearch",
                "timestamp": datetime.now().isoformat(),
                "call_count": self.call_count
            }
        }


class MockBrainBox:
    """Mock implementation of BrainBox worker"""

    def __init__(self):
        self.call_count = 0
        logger.info("MockBrainBox initialized")

    def create_video_plan(self, script: str, transcript_timestamps: Dict,
                         research_data: Dict, requirements: Dict) -> Dict[str, Any]:
        """
        Mock video plan creation

        Args:
            script: Video script
            transcript_timestamps: Audio timing data
            research_data: Research information
            requirements: User requirements

        Returns:
            Dictionary with video plan
        """
        self.call_count += 1
        topic = requirements.get('topic', 'Unknown')
        logger.info(f"MockBrainBox called (count: {self.call_count}) for topic: {topic}")

        # Generate mock video plan
        video_plan = {
            "timeline": [
                {
                    "timestamp": "0:00-0:05",
                    "asset_type": "video",
                    "description": f"Opening shot related to {topic}",
                    "script_line": "Welcome to this video..."
                },
                {
                    "timestamp": "0:05-0:15",
                    "asset_type": "image",
                    "description": f"Infographic about {topic}",
                    "script_line": "Let me show you..."
                },
                {
                    "timestamp": "0:15-1:45",
                    "asset_type": "video",
                    "description": "Main content visuals",
                    "script_line": "The key points are..."
                },
                {
                    "timestamp": "1:45-2:00",
                    "asset_type": "image",
                    "description": "Call to action graphic",
                    "script_line": "Thanks for watching..."
                }
            ],
            "total_duration": 120.0,
            "total_assets": 4,
            "theme": requirements.get('style', 'default')
        }

        return {
            "success": True,
            "outputs": {
                "video_plan": video_plan,
                "plan_file": "video_plan.json"
            },
            "metadata": {
                "worker": "brainbox",
                "timestamp": datetime.now().isoformat(),
                "call_count": self.call_count
            }
        }


class MockAssetCollector:
    """Mock implementation of Asset Collector worker"""

    def __init__(self):
        self.call_count = 0
        logger.info("MockAssetCollector initialized")

    def collect_assets(self, video_plan: Dict) -> Dict[str, Any]:
        """
        Mock asset collection

        Args:
            video_plan: Video plan with asset specifications

        Returns:
            Dictionary with collected assets info
        """
        self.call_count += 1
        logger.info(f"MockAssetCollector called (count: {self.call_count})")

        # Generate mock asset manifest
        asset_manifest = {
            "assets": [
                {
                    "id": "asset_001",
                    "type": "video",
                    "file": "assets/opening_shot.mp4",
                    "timestamp": "0:00-0:05",
                    "status": "collected"
                },
                {
                    "id": "asset_002",
                    "type": "image",
                    "file": "assets/infographic_01.png",
                    "timestamp": "0:05-0:15",
                    "status": "collected"
                },
                {
                    "id": "asset_003",
                    "type": "video",
                    "file": "assets/main_content.mp4",
                    "timestamp": "0:15-1:45",
                    "status": "collected"
                },
                {
                    "id": "asset_004",
                    "type": "image",
                    "file": "assets/cta_graphic.png",
                    "timestamp": "1:45-2:00",
                    "status": "collected"
                }
            ],
            "total_collected": 4,
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
                "timestamp": datetime.now().isoformat(),
                "call_count": self.call_count
            }
        }


class MockExecutor:
    """Mock implementation of Executor worker"""

    def __init__(self):
        self.call_count = 0
        logger.info("MockExecutor initialized")

    def render_video(self, video_plan: Dict, assets: Dict, audio: str) -> Dict[str, Any]:
        """
        Mock video rendering

        Args:
            video_plan: Video plan with timeline
            assets: Asset manifest and files
            audio: Audio file path

        Returns:
            Dictionary with rendered video info
        """
        self.call_count += 1
        logger.info(f"MockExecutor called (count: {self.call_count})")

        return {
            "success": True,
            "outputs": {
                "video_file": "final_video.mp4",
                "duration": 120.0,
                "resolution": "1920x1080",
                "fps": 30,
                "file_size_mb": 45.2
            },
            "metadata": {
                "worker": "executor",
                "timestamp": datetime.now().isoformat(),
                "call_count": self.call_count,
                "render_time": 125.5
            }
        }


# Worker factory
class WorkerFactory:
    """Factory for creating mock worker instances"""

    _instances = {}

    @classmethod
    def get_worker(cls, tool_name: str):
        """
        Get or create a worker instance

        Args:
            tool_name: Name of the tool/worker

        Returns:
            Worker instance

        Raises:
            ValueError: If tool name is not recognized
        """
        if tool_name not in cls._instances:
            # Use real implementations when available, otherwise fall back to mocks
            worker_map = {
                "scriptwriter": RealScriptWriter if REAL_SCRIPTWRITER_AVAILABLE else MockScriptWriter,
                "audio_agent": RealAudioAgent if REAL_AUDIOAGENT_AVAILABLE else MockAudioAgent,
                "langsearch": RealLangSearch if REAL_LANGSEARCH_AVAILABLE else MockLangSearch,
                "brainbox": RealBrainBox if REAL_BRAINBOX_AVAILABLE else MockBrainBox,
                "asset_collector": RealAssetCollector if REAL_ASSETCOLLECTOR_AVAILABLE else MockAssetCollector,
                "executor": MockExecutor
            }

            if tool_name not in worker_map:
                raise ValueError(f"Unknown tool: {tool_name}. Available: {list(worker_map.keys())}")

            cls._instances[tool_name] = worker_map[tool_name]()

            # Determine worker type
            is_real = (
                (tool_name == "scriptwriter" and REAL_SCRIPTWRITER_AVAILABLE) or
                (tool_name == "audio_agent" and REAL_AUDIOAGENT_AVAILABLE) or
                (tool_name == "langsearch" and REAL_LANGSEARCH_AVAILABLE) or
                (tool_name == "asset_collector" and REAL_ASSETCOLLECTOR_AVAILABLE) or
                (tool_name == "brainbox" and REAL_BRAINBOX_AVAILABLE)
            )
            worker_type = "REAL" if is_real else "MOCK"
            logger.info(f"Created new worker instance: {tool_name} ({worker_type})")

        return cls._instances[tool_name]

    @classmethod
    def reset_workers(cls):
        """Reset all worker instances (useful for testing)"""
        cls._instances = {}
        logger.info("All worker instances reset")


def validate_tool_inputs(tool_name: str, inputs: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    Validate that required inputs are provided for a tool

    Args:
        tool_name: Name of the tool
        inputs: Input dictionary

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    if tool_name not in AVAILABLE_TOOLS:
        return False, [f"Unknown tool: {tool_name}"]

    tool_spec = AVAILABLE_TOOLS[tool_name]
    required_inputs = tool_spec["required_inputs"]
    errors = []

    for required_input in required_inputs:
        if required_input not in inputs or inputs[required_input] is None:
            errors.append(f"Missing required input: {required_input}")

    return len(errors) == 0, errors


def get_tool_info(tool_name: str = None) -> Dict:
    """
    Get information about a tool or all tools

    Args:
        tool_name: Specific tool name, or None for all tools

    Returns:
        Tool information dictionary
    """
    if tool_name:
        if tool_name not in AVAILABLE_TOOLS:
            return {"error": f"Unknown tool: {tool_name}"}
        return {tool_name: AVAILABLE_TOOLS[tool_name]}

    return AVAILABLE_TOOLS
