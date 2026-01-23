"""
Secretary Module - Orchestration Hub for Video Generation Pipeline

This module collects user requirements, validates inputs, and coordinates
all worker modules through tool calling.
"""

import json
import re
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

try:
    from ipywidgets import widgets, Layout, VBox, HBox, Button, HTML
    from IPython.display import display, clear_output
    NOTEBOOK_MODE = True
except ImportError:
    NOTEBOOK_MODE = False
    logging.warning("ipywidgets not available - notebook features disabled")

from tool_registry import (
    AVAILABLE_TOOLS,
    WorkerFactory,
    validate_tool_inputs,
    get_tool_info
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Secretary:
    """
    Secretary class - Orchestration hub for video generation pipeline

    Responsibilities:
    - Collect user requirements via interactive form
    - Validate inputs using pure Python logic
    - Output structured JSON
    - Call worker tools with proper inputs
    - Execute complete workflows
    """

    # Style presets
    STYLE_PRESETS = ["funny", "documentary", "serious", "graphic-heavy", "tutorial"]

    # Duration format regex pattern (requires space between number and unit)
    DURATION_PATTERN = re.compile(
        r'^\d+(-\d+)?\s+(minute|minutes|second|seconds|min|mins|sec|secs)$',
        re.IGNORECASE
    )

    def __init__(self, output_dir: str = "./outputs"):
        """
        Initialize Secretary

        Args:
            output_dir: Directory for saving output files
        """
        self.requirements = {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.workflow_history = []
        logger.info(f"Secretary initialized with output directory: {self.output_dir}")

    def display_form(self) -> Dict[str, Any]:
        """
        Display interactive form to collect user requirements

        Returns:
            Dictionary with collected inputs

        Note:
            In notebook mode, uses ipywidgets for interactive form.
            Otherwise, falls back to console input.
        """
        if NOTEBOOK_MODE:
            return self._display_notebook_form()
        else:
            return self._display_console_form()

    def _display_notebook_form(self) -> Dict[str, Any]:
        """Display form using ipywidgets (for Jupyter/Colab)"""
        logger.info("Displaying notebook form")

        # Create form widgets
        topic_widget = widgets.Text(
            placeholder='Enter video topic (min 5 characters)',
            description='Topic:',
            layout=Layout(width='95%'),
            style={'description_width': '150px'}
        )

        # Style selection - radio for presets or text for custom
        style_radio = widgets.RadioButtons(
            options=self.STYLE_PRESETS,
            description='Style Preset:',
            style={'description_width': '150px'}
        )

        style_custom = widgets.Text(
            placeholder='Or enter custom style (min 3 characters)',
            description='Custom Style:',
            layout=Layout(width='95%'),
            style={'description_width': '150px'}
        )

        duration_widget = widgets.Text(
            placeholder='e.g., "2-3 minutes", "30-60 seconds", "1 minute"',
            description='Duration:',
            layout=Layout(width='95%'),
            style={'description_width': '150px'}
        )

        audio_widget = widgets.Dropdown(
            options=['generate', 'upload'],
            description='Audio Mode:',
            style={'description_width': '150px'}
        )

        script_widget = widgets.Dropdown(
            options=['generate', 'upload'],
            description='Script Mode:',
            style={'description_width': '150px'}
        )

        comments_widget = widgets.Textarea(
            placeholder='Additional comments (optional)',
            description='Comments:',
            layout=Layout(width='95%', height='80px'),
            style={'description_width': '150px'}
        )

        # Output area for validation messages
        output = widgets.Output()

        # Submit button
        submit_btn = Button(
            description='Submit Requirements',
            button_style='success',
            layout=Layout(width='200px')
        )

        # Storage for collected data
        collected_data = {}

        def on_submit(btn):
            """Handle form submission"""
            with output:
                clear_output()

                # Collect data
                style_value = style_custom.value.strip() if style_custom.value.strip() else style_radio.value

                data = {
                    'topic': topic_widget.value.strip(),
                    'style': style_value,
                    'duration_range': duration_widget.value.strip(),
                    'audio_mode': audio_widget.value,
                    'script_mode': script_widget.value,
                    'comments': comments_widget.value.strip() or None
                }

                # Validate
                is_valid, errors = self.validate_inputs(data)

                if is_valid:
                    collected_data.update(data)
                    print("✓ Requirements validated successfully!")
                    print(f"\nCollected requirements:")
                    for key, value in data.items():
                        print(f"  {key}: {value}")
                else:
                    print("✗ Validation failed:")
                    for error in errors:
                        print(f"  - {error}")

        submit_btn.on_click(on_submit)

        # Layout form
        form = VBox([
            HTML("<h3>Video Generation Requirements</h3>"),
            HTML("<p>Please fill out all required fields:</p>"),
            topic_widget,
            HTML("<hr>"),
            style_radio,
            style_custom,
            HTML("<hr>"),
            duration_widget,
            audio_widget,
            script_widget,
            comments_widget,
            HTML("<hr>"),
            submit_btn,
            output
        ])

        display(form)
        return collected_data

    def _display_console_form(self) -> Dict[str, Any]:
        """Display form using console input (fallback)"""
        logger.info("Displaying console form")
        print("=" * 60)
        print("VIDEO GENERATION REQUIREMENTS")
        print("=" * 60)

        # Collect inputs
        topic = input("\n1. Topic (min 5 chars): ").strip()

        print("\n2. Style - Choose preset or enter custom:")
        print(f"   Presets: {', '.join(self.STYLE_PRESETS)}")
        style = input("   Enter style: ").strip()

        duration = input("\n3. Duration (e.g., '2-3 minutes', '30 seconds'): ").strip()

        print("\n4. Audio mode:")
        audio_mode = input("   Enter 'generate' or 'upload': ").strip().lower()

        print("\n5. Script mode:")
        script_mode = input("   Enter 'generate' or 'upload': ").strip().lower()

        comments = input("\n6. Comments (optional, press Enter to skip): ").strip()

        data = {
            'topic': topic,
            'style': style,
            'duration_range': duration,
            'audio_mode': audio_mode,
            'script_mode': script_mode,
            'comments': comments or None
        }

        print("\n" + "=" * 60)
        return data

    def validate_inputs(self, inputs: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate all input fields

        Args:
            inputs: Dictionary of user inputs

        Returns:
            Tuple of (is_valid: bool, errors: List[str])
        """
        errors = []

        # Validate topic
        topic = inputs.get('topic', '').strip()
        if not topic:
            errors.append("Topic is required")
        elif len(topic) < 5:
            errors.append("Topic must be at least 5 characters")

        # Validate style
        style = inputs.get('style', '').strip()
        if not style:
            errors.append("Style is required")
        elif style not in self.STYLE_PRESETS:
            # Custom style - must be at least 3 chars
            if len(style) < 3:
                errors.append("Custom style must be at least 3 characters")

        # Validate duration (with smart parsing)
        duration = inputs.get('duration_range', '').strip()
        if not duration:
            errors.append("Duration is required")
        else:
            # Try to parse and normalize duration
            normalized_duration = self._normalize_duration(duration)
            if normalized_duration:
                # Update the input with normalized version
                inputs['duration_range'] = normalized_duration
                logger.info(f"Duration normalized: '{duration}' → '{normalized_duration}'")
            else:
                errors.append(
                    f"Duration format invalid. Could not parse '{duration}'. "
                    "Examples: '5 minutes', '2-3 minutes', 'like 30 seconds', 'around 1-2 mins'"
                )

        # Validate audio mode (case-insensitive but no whitespace allowed)
        audio_mode_raw = inputs.get('audio_mode', '')
        audio_mode = audio_mode_raw.lower() if isinstance(audio_mode_raw, str) else ''
        if audio_mode not in ['generate', 'upload']:
            errors.append("Audio mode must be exactly 'generate' or 'upload'")
        elif audio_mode_raw != audio_mode_raw.strip():
            errors.append("Audio mode must be exactly 'generate' or 'upload'")

        # Validate script mode (case-insensitive but no whitespace allowed)
        script_mode_raw = inputs.get('script_mode', '')
        script_mode = script_mode_raw.lower() if isinstance(script_mode_raw, str) else ''
        if script_mode not in ['generate', 'upload']:
            errors.append("Script mode must be exactly 'generate' or 'upload'")
        elif script_mode_raw != script_mode_raw.strip():
            errors.append("Script mode must be exactly 'generate' or 'upload'")

        # Comments are optional - no validation needed

        is_valid = len(errors) == 0
        logger.info(f"Validation result: {'PASS' if is_valid else 'FAIL'} ({len(errors)} errors)")

        return is_valid, errors

    def _normalize_duration(self, duration: str) -> Optional[str]:
        """
        Parse and normalize duration from natural language

        Handles inputs like:
        - "like 15 seconds" → "15 seconds"
        - "around 2-3 minutes" → "2-3 minutes"
        - "about 5 mins" → "5 minutes"
        - "roughly 30-60 secs" → "30-60 seconds"
        - "maybe 2 minutes" → "2 minutes"

        Args:
            duration: User input duration string

        Returns:
            Normalized duration string or None if invalid
        """
        if not duration:
            return None

        # Remove filler words
        filler_words = [
            'like', 'around', 'about', 'roughly', 'maybe', 'approximately',
            'approx', 'nearly', 'almost', 'close to', 'just', 'only'
        ]

        cleaned = duration.lower()
        for filler in filler_words:
            cleaned = cleaned.replace(filler, '')

        cleaned = cleaned.strip()

        # Extract number(s) and unit
        # Pattern: captures "5" or "2-3" followed by "minutes/mins/min/seconds/secs/sec"
        pattern = r'(\d+(?:-\d+)?)\s*(minute|minutes|min|mins|second|seconds|sec|secs)'
        match = re.search(pattern, cleaned, re.IGNORECASE)

        if match:
            number_part = match.group(1)  # "5" or "2-3"
            unit_part = match.group(2).lower()  # "minutes", "mins", etc.

            # Normalize unit to full form
            if unit_part in ['minute', 'minutes', 'min', 'mins']:
                unit = 'minutes' if '-' in number_part or int(number_part.split('-')[0]) != 1 else 'minute'
            elif unit_part in ['second', 'seconds', 'sec', 'secs']:
                unit = 'seconds' if '-' in number_part or int(number_part.split('-')[0]) != 1 else 'second'
            else:
                return None

            normalized = f"{number_part} {unit}"

            # Final validation against strict pattern
            if self.DURATION_PATTERN.match(normalized):
                return normalized

        return None

    def collect_requirements(self, interactive: bool = True) -> Dict[str, Any]:
        """
        Main method to collect and validate requirements

        Args:
            interactive: If True, display form. If False, use pre-set requirements.

        Returns:
            Dictionary with validated requirements
        """
        logger.info("Starting requirements collection")

        if interactive:
            # Display form and collect inputs
            inputs = self.display_form()

            # If notebook mode and form wasn't submitted, return empty
            if NOTEBOOK_MODE and not inputs:
                logger.warning("Form displayed but not submitted")
                return {}

            # Validate inputs
            is_valid, errors = self.validate_inputs(inputs)

            if not is_valid:
                logger.error(f"Validation failed: {errors}")
                if not NOTEBOOK_MODE:
                    print("\n✗ Validation failed:")
                    for error in errors:
                        print(f"  - {error}")
                return {}

            # Build requirements object
            self.requirements = {
                "topic": inputs['topic'],
                "style": inputs['style'],
                "duration_range": inputs['duration_range'],
                "audio_mode": inputs['audio_mode'],
                "script_mode": inputs['script_mode'],
                "comments": inputs['comments'],
                "timestamp": datetime.now().isoformat(),
                "status": "validated"
            }

            logger.info("Requirements collected and validated successfully")
            return self.requirements

        else:
            # Return pre-set requirements (for testing)
            if not self.requirements:
                logger.warning("No requirements set - call set_requirements first")
            return self.requirements

    def set_requirements(self, requirements: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Set requirements programmatically (useful for testing)

        Args:
            requirements: Requirements dictionary

        Returns:
            Tuple of (is_valid, errors)
        """
        is_valid, errors = self.validate_inputs(requirements)

        if is_valid:
            self.requirements = {
                **requirements,
                "timestamp": datetime.now().isoformat(),
                "status": "validated"
            }
            logger.info("Requirements set programmatically")

        return is_valid, errors

    def save_requirements(self, filepath: Optional[str] = None) -> str:
        """
        Save requirements to JSON file

        Args:
            filepath: Custom filepath, or None to use default

        Returns:
            Path to saved file
        """
        if not self.requirements:
            raise ValueError("No requirements to save. Call collect_requirements first.")

        if filepath is None:
            filepath = self.output_dir / "requirements.json"
        else:
            filepath = Path(filepath)

        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.requirements, f, indent=2)

        logger.info(f"Requirements saved to: {filepath}")
        return str(filepath)

    def load_requirements(self, filepath: str) -> Dict[str, Any]:
        """
        Load requirements from JSON file

        Args:
            filepath: Path to requirements file

        Returns:
            Requirements dictionary
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Requirements file not found: {filepath}")

        with open(filepath, 'r') as f:
            self.requirements = json.load(f)

        logger.info(f"Requirements loaded from: {filepath}")
        return self.requirements

    def call_tool(self, tool_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a worker tool with given inputs

        Args:
            tool_name: Name of tool from AVAILABLE_TOOLS
            inputs: Input parameters for the tool

        Returns:
            Dictionary with:
                - success: bool
                - outputs: dict of output files/data
                - error: str if failed
                - metadata: execution metadata
        """
        logger.info(f"Calling tool: {tool_name}")

        # Validate tool exists
        if tool_name not in AVAILABLE_TOOLS:
            error_msg = f"Unknown tool: {tool_name}. Available: {list(AVAILABLE_TOOLS.keys())}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "outputs": {},
                "metadata": {
                    "tool": tool_name,
                    "timestamp": datetime.now().isoformat()
                }
            }

        # Validate inputs
        is_valid, errors = validate_tool_inputs(tool_name, inputs)
        if not is_valid:
            error_msg = f"Invalid inputs for {tool_name}: {', '.join(errors)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "outputs": {},
                "metadata": {
                    "tool": tool_name,
                    "timestamp": datetime.now().isoformat()
                }
            }

        # Get worker instance
        try:
            worker = WorkerFactory.get_worker(tool_name)
        except Exception as e:
            error_msg = f"Failed to get worker {tool_name}: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "outputs": {},
                "metadata": {
                    "tool": tool_name,
                    "timestamp": datetime.now().isoformat()
                }
            }

        # Execute tool
        try:
            # Map tool names to worker methods
            if tool_name == "scriptwriter":
                result = worker.generate_script(inputs)
            elif tool_name == "audio_agent":
                result = worker.generate_audio(inputs.get('script', ''))
            elif tool_name == "langsearch":
                result = worker.research_terms(inputs.get('script', ''))
            elif tool_name == "brainbox":
                result = worker.create_video_plan(
                    inputs.get('script', ''),
                    inputs.get('transcript_timestamps', {}),
                    inputs.get('research_data', {}),
                    inputs.get('requirements', {})
                )
            elif tool_name == "asset_collector":
                result = worker.collect_assets(inputs.get('video_plan', {}))
            elif tool_name == "executor":
                result = worker.render_video(
                    inputs.get('video_plan', {}),
                    inputs.get('assets', {}),
                    inputs.get('audio', '')
                )
            else:
                raise ValueError(f"No execution method for tool: {tool_name}")

            logger.info(f"Tool {tool_name} executed successfully")
            return result

        except Exception as e:
            error_msg = f"Tool execution failed for {tool_name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "error": error_msg,
                "outputs": {},
                "metadata": {
                    "tool": tool_name,
                    "timestamp": datetime.now().isoformat()
                }
            }

    def execute_workflow(self, workflow: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Execute a sequence of tool calls

        Args:
            workflow: List of (tool_name, inputs) tuples

        Returns:
            Dictionary with results from each step:
                - results: Dict mapping step index to result
                - success: bool (True if all steps succeeded)
                - failed_step: index of first failed step (if any)
        """
        logger.info(f"Executing workflow with {len(workflow)} steps")

        results = {}
        workflow_outputs = {}  # Store outputs for passing between steps

        for step_idx, (tool_name, inputs) in enumerate(workflow):
            logger.info(f"Step {step_idx + 1}/{len(workflow)}: {tool_name}")

            # Replace placeholder references with actual outputs from previous steps
            resolved_inputs = self._resolve_workflow_inputs(inputs, workflow_outputs)

            # Execute tool
            result = self.call_tool(tool_name, resolved_inputs)
            results[step_idx] = {
                "tool": tool_name,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }

            # Check if step failed
            if not result.get('success', False):
                logger.error(f"Workflow failed at step {step_idx + 1}: {tool_name}")
                return {
                    "results": results,
                    "success": False,
                    "failed_step": step_idx,
                    "total_steps": len(workflow),
                    "completed_steps": step_idx
                }

            # Store outputs for next steps
            workflow_outputs[tool_name] = result.get('outputs', {})

        logger.info("Workflow completed successfully")
        self.workflow_history.append({
            "workflow": workflow,
            "results": results,
            "timestamp": datetime.now().isoformat()
        })

        return {
            "results": results,
            "success": True,
            "failed_step": None,
            "total_steps": len(workflow),
            "completed_steps": len(workflow)
        }

    def _resolve_workflow_inputs(self, inputs: Dict[str, Any],
                                 workflow_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve input placeholders with actual outputs from previous steps

        Args:
            inputs: Input dictionary that may contain references
            workflow_outputs: Outputs from previous workflow steps

        Returns:
            Resolved inputs dictionary
        """
        resolved = {}

        for key, value in inputs.items():
            # If value is a string reference like "$scriptwriter.script"
            if isinstance(value, str) and value.startswith('$'):
                parts = value[1:].split('.')
                if len(parts) == 2:
                    tool_ref, output_key = parts
                    if tool_ref in workflow_outputs:
                        resolved[key] = workflow_outputs[tool_ref].get(output_key, value)
                    else:
                        resolved[key] = value
                else:
                    resolved[key] = value
            else:
                resolved[key] = value

        return resolved

    def get_tool_info(self, tool_name: Optional[str] = None) -> Dict:
        """
        Get information about available tools

        Args:
            tool_name: Specific tool name, or None for all tools

        Returns:
            Tool information dictionary
        """
        return get_tool_info(tool_name)

    def reset(self):
        """Reset secretary state (useful for testing)"""
        self.requirements = {}
        self.workflow_history = []
        WorkerFactory.reset_workers()
        logger.info("Secretary state reset")


# Convenience function
def create_full_workflow(requirements: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Create a standard full workflow from requirements to final video

    Args:
        requirements: User requirements dictionary

    Returns:
        Workflow list ready for execute_workflow
    """
    workflow = [
        ("scriptwriter", requirements),
        ("audio_agent", {"script": "$scriptwriter.script"}),
        ("langsearch", {"script": "$scriptwriter.script"}),
        ("brainbox", {
            "script": "$scriptwriter.script",
            "transcript_timestamps": "$audio_agent.transcript_timestamps",
            "research_data": "$langsearch.research_data",
            "requirements": requirements
        }),
        ("asset_collector", {"video_plan": "$brainbox.video_plan"}),
        ("executor", {
            "video_plan": "$brainbox.video_plan",
            "assets": "$asset_collector.asset_manifest",
            "audio": "$audio_agent.audio_file"
        })
    ]

    return workflow
