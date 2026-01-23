# Secretary Module

The **Secretary** is the orchestration hub for the automated video generation pipeline. It collects user requirements, validates inputs, and coordinates all worker modules through a sophisticated tool calling system.

## Overview

The Secretary module serves as the central coordinator that:
- 📋 Collects 6 user requirements via interactive forms
- ✅ Validates inputs using pure Python logic (no LLM)
- 💾 Outputs structured JSON
- 🔧 Has tool calling capability to invoke other workers
- 🔄 Executes complete workflows from requirements to final video

## Features

### Input Collection
- **Interactive Forms**: Works in Jupyter/Colab notebooks with ipywidgets
- **Console Fallback**: Supports command-line input when notebooks aren't available
- **Simultaneous Display**: All 6 questions displayed at once for better UX

### Validation
- **Topic**: Minimum 5 characters, not empty
- **Style**: Either preset (funny, documentary, serious, graphic-heavy, tutorial) or custom (min 3 chars)
- **Duration**: Valid format like "2-3 minutes", "30 seconds", "1 minute"
- **Audio/Script**: Exactly "generate" or "upload" (case-insensitive)
- **Comments**: Optional, any text

### Tool Calling
- **6 Worker Tools**: scriptwriter, audio_agent, langsearch, brainbox, asset_collector, executor
- **Mock Workers**: Complete mock implementations for testing
- **Workflow Execution**: Chain multiple tools with automatic output passing
- **Error Handling**: Graceful failure handling with detailed error messages

## Installation

```bash
pip install -r requirements.txt
```

For Google Colab:
```python
!pip install ipywidgets ipython
```

## Quick Start

### Basic Usage

```python
from secretary import Secretary

# Initialize
sec = Secretary()

# Collect requirements (interactive form)
requirements = sec.collect_requirements()

# Or set programmatically
requirements = {
    'topic': 'Introduction to Machine Learning',
    'style': 'tutorial',
    'duration_range': '5-7 minutes',
    'audio_mode': 'generate',
    'script_mode': 'generate',
    'comments': 'Focus on beginners'
}
is_valid, errors = sec.set_requirements(requirements)

# Save requirements
sec.save_requirements('./outputs/requirements.json')
```

### Tool Calling

```python
# Call a single tool
result = sec.call_tool('scriptwriter', requirements)
if result['success']:
    script = result['outputs']['script']
    print(f"Generated script: {script}")
else:
    print(f"Error: {result['error']}")

# Call multiple tools
audio_result = sec.call_tool('audio_agent', {
    'script': script
})

research_result = sec.call_tool('langsearch', {
    'script': script
})
```

### Workflow Execution

```python
from secretary import Secretary, create_full_workflow

# Initialize
sec = Secretary()

# Set requirements
requirements = {
    'topic': 'Climate Change Solutions',
    'style': 'documentary',
    'duration_range': '3-5 minutes',
    'audio_mode': 'generate',
    'script_mode': 'generate',
    'comments': 'Include expert opinions'
}
sec.set_requirements(requirements)

# Create full workflow
workflow = create_full_workflow(sec.requirements)

# Execute
result = sec.execute_workflow(workflow)

if result['success']:
    print(f"✅ Workflow completed! {result['completed_steps']}/{result['total_steps']} steps")

    # Access results from each step
    for step_idx, step_data in result['results'].items():
        tool_name = step_data['tool']
        print(f"  Step {step_idx + 1}: {tool_name} - Success!")
else:
    print(f"❌ Workflow failed at step {result['failed_step'] + 1}")
```

### Custom Workflows

```python
# Build custom workflow
workflow = [
    # Step 1: Generate script
    ('scriptwriter', requirements),

    # Step 2: Generate audio (uses script from step 1)
    ('audio_agent', {'script': '$scriptwriter.script'}),

    # Step 3: Research terms (parallel with audio)
    ('langsearch', {'script': '$scriptwriter.script'}),

    # Step 4: Create video plan (uses outputs from all previous steps)
    ('brainbox', {
        'script': '$scriptwriter.script',
        'transcript_timestamps': '$audio_agent.transcript_timestamps',
        'research_data': '$langsearch.research_data',
        'requirements': requirements
    })
]

# Execute custom workflow
result = sec.execute_workflow(workflow)
```

## API Reference

### Secretary Class

#### `__init__(output_dir='./outputs')`
Initialize Secretary with optional output directory.

#### `display_form() -> Dict`
Display interactive form to collect requirements. Returns dictionary of inputs.

#### `validate_inputs(inputs: Dict) -> Tuple[bool, List[str]]`
Validate input dictionary. Returns (is_valid, error_list).

#### `collect_requirements(interactive=True) -> Dict`
Main method to collect and validate requirements.

#### `set_requirements(requirements: Dict) -> Tuple[bool, List[str]]`
Set requirements programmatically. Returns (is_valid, errors).

#### `save_requirements(filepath=None) -> str`
Save requirements to JSON file. Returns path to saved file.

#### `load_requirements(filepath: str) -> Dict`
Load requirements from JSON file. Returns requirements dictionary.

#### `call_tool(tool_name: str, inputs: Dict) -> Dict`
Call a worker tool with given inputs. Returns result dictionary with:
- `success`: bool
- `outputs`: dict of output files/data
- `error`: str (if failed)
- `metadata`: execution metadata

#### `execute_workflow(workflow: List[Tuple[str, Dict]]) -> Dict`
Execute a sequence of tool calls. Returns result dictionary with:
- `results`: dict mapping step index to result
- `success`: bool
- `failed_step`: index of first failed step (if any)
- `total_steps`: total number of steps
- `completed_steps`: number of completed steps

#### `get_tool_info(tool_name=None) -> Dict`
Get information about available tools.

#### `reset()`
Reset secretary state (useful for testing).

### Available Tools

| Tool | Description | Required Inputs | Outputs |
|------|-------------|-----------------|---------|
| `scriptwriter` | Generates video script | topic, style, duration_range, comments | script.txt |
| `audio_agent` | Generates timed audio transcript | script | audio.wav, transcript_timestamps.json |
| `langsearch` | Researches script terms | script | research_data.json |
| `brainbox` | Creates video plan | script, transcript_timestamps, research_data, requirements | video_plan.json |
| `asset_collector` | Collects/generates assets | video_plan | assets/, asset_manifest.json |
| `executor` | Assembles and renders video | video_plan, assets, audio | final_video.mp4 |

## Requirements Format

### Input Requirements
```json
{
  "topic": "Introduction to Python",
  "style": "tutorial",
  "duration_range": "5-7 minutes",
  "audio_mode": "generate",
  "script_mode": "generate",
  "comments": "Focus on beginners"
}
```

### Output Format (requirements.json)
```json
{
  "topic": "Introduction to Python",
  "style": "tutorial",
  "duration_range": "5-7 minutes",
  "audio_mode": "generate",
  "script_mode": "generate",
  "comments": "Focus on beginners",
  "timestamp": "2025-01-13T10:30:00.000000",
  "status": "validated"
}
```

## Testing

Run all tests:
```bash
# Using unittest
python -m unittest discover -s . -p "test_*.py"

# Or using pytest
pytest test_secretary.py test_tool_calling.py -v

# With coverage
pytest --cov=. --cov-report=html
```

Run specific test suites:
```bash
# Validation tests only
python -m unittest test_secretary.py

# Tool calling tests only
python -m unittest test_tool_calling.py
```

## Examples

### Example 1: Quick Script Generation
```python
from secretary import Secretary

sec = Secretary()

# Set requirements
sec.set_requirements({
    'topic': 'Python Tips for Beginners',
    'style': 'tutorial',
    'duration_range': '3 minutes',
    'audio_mode': 'generate',
    'script_mode': 'generate',
    'comments': 'Keep it simple'
})

# Generate script
result = sec.call_tool('scriptwriter', sec.requirements)
print(result['outputs']['script'])
```

### Example 2: Full Video Pipeline
```python
from secretary import Secretary, create_full_workflow

# Setup
sec = Secretary()
sec.set_requirements({
    'topic': 'Climate Change Impact',
    'style': 'documentary',
    'duration_range': '5 minutes',
    'audio_mode': 'generate',
    'script_mode': 'generate',
    'comments': 'Use scientific data'
})

# Execute full pipeline
workflow = create_full_workflow(sec.requirements)
result = sec.execute_workflow(workflow)

# Check results
if result['success']:
    final_video = result['results'][5]['result']['outputs']['video_file']
    print(f"✅ Video created: {final_video}")
```

### Example 3: Custom Workflow with Error Handling
```python
from secretary import Secretary

sec = Secretary()
requirements = {...}  # Your requirements

workflow = [
    ('scriptwriter', requirements),
    ('audio_agent', {'script': '$scriptwriter.script'}),
]

result = sec.execute_workflow(workflow)

if not result['success']:
    failed_step = result['failed_step']
    failed_tool = result['results'][failed_step]['tool']
    error = result['results'][failed_step]['result']['error']
    print(f"❌ Failed at step {failed_step + 1} ({failed_tool}): {error}")
else:
    print("✅ All steps completed successfully!")
```

### Example 4: Notebook Integration (Google Colab)
```python
# In a Colab notebook
from secretary import Secretary

# Initialize
sec = Secretary()

# Display interactive form (will show ipywidgets)
requirements = sec.collect_requirements()

# Form submission automatically validates
# Continue with workflow execution...
```

## Architecture

```
Secretary Module
├── secretary.py          # Main Secretary class
├── tool_registry.py      # Tool definitions + mock workers
├── test_secretary.py     # Validation tests
├── test_tool_calling.py  # Tool calling tests
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## Design Principles

1. **Modularity**: Secretary doesn't implement worker logic, only coordinates
2. **Testability**: Mock workers allow testing without real implementations
3. **Extensibility**: Easy to add new workers to the registry
4. **Robustness**: Comprehensive validation and error handling
5. **Usability**: Works in notebooks and console

## Validation Rules

### Topic
- ✅ Not empty
- ✅ Minimum 5 characters
- ❌ Whitespace-only strings

### Style
- ✅ One of: funny, documentary, serious, graphic-heavy, tutorial
- ✅ OR custom style with min 3 characters
- ❌ Empty string
- ❌ Custom style < 3 characters

### Duration
- ✅ Format: "X-Y minutes/seconds" or "X minutes/seconds"
- ✅ Examples: "2-3 minutes", "30 seconds", "1 minute"
- ❌ Invalid formats: "five minutes", "2-3", "1.5 minutes"

### Audio/Script Mode
- ✅ Exactly "generate" or "upload" (case-insensitive)
- ❌ Any other value including: "create", "make", "auto"

### Comments
- ✅ Any text or empty/null
- ✅ No validation required

## Error Handling

The Secretary provides detailed error messages for all failure scenarios:

```python
# Validation errors
is_valid, errors = sec.validate_inputs(bad_inputs)
# errors = [
#     "Topic must be at least 5 characters",
#     "Duration format invalid. Expected format: 'X-Y minutes/seconds'",
#     "Audio mode must be exactly 'generate' or 'upload'"
# ]

# Tool calling errors
result = sec.call_tool('nonexistent_tool', {})
# result = {
#     'success': False,
#     'error': "Unknown tool: nonexistent_tool. Available: ['scriptwriter', ...]",
#     'outputs': {},
#     'metadata': {...}
# }

# Workflow errors
result = sec.execute_workflow(workflow)
# result = {
#     'success': False,
#     'failed_step': 2,
#     'results': {...},
#     'total_steps': 5,
#     'completed_steps': 2
# }
```

## Performance

- ⚡ Validation: < 1 second
- ⚡ Single tool call: < 1 second (mock)
- ⚡ Full workflow: < 5 seconds (mock)
- ⚡ Form display: Instant

## Limitations

- Mock workers don't perform actual work (by design)
- Notebook form requires ipywidgets (falls back to console)
- No async/parallel execution (sequential workflow only)
- No persistence of workflow state between sessions

## Future Enhancements

- [ ] Add async/parallel workflow execution
- [ ] Implement workflow state persistence
- [ ] Add workflow visualization
- [ ] Support for conditional workflows
- [ ] Add workflow templates library
- [ ] Implement real-time progress tracking
- [ ] Add workflow rollback capability

## Contributing

To extend the Secretary with new workers:

1. Add tool definition to `AVAILABLE_TOOLS` in `tool_registry.py`
2. Create mock worker class implementing the worker interface
3. Add worker to `WorkerFactory.get_worker()` method
4. Update tests to include new worker
5. Update this README with new tool documentation

## License

Part of the ESTA Final Project - Automated Video Generation Pipeline

## Support

For issues, questions, or contributions, please refer to the main project repository.
