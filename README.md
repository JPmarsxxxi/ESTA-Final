PROJECT OVERVIEW - AUTOMATED VIDEO GENERATION PIPELINE
PROJECT GOAL
Build a modular, AI-powered video generation system where users input requirements and receive a complete video with minimal intervention.
SYSTEM ARCHITECTURE
USER INPUT (Topic, Style, Duration, etc.)
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 1. SECRETARY (Orchestration & Tool Calling)                 │
│    - Collects 6 requirements via form                       │
│    - Validates inputs                                        │
│    - Coordinates all other workers via tool calling         │
│    - Routes between workers based on needs                  │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. SCRIPTWRITER                                             │
│    - Takes: topic, style, duration, comments                │
│    - Generates: talking points → bodies → Hook/Body/CTA     │
│    - Output: Complete script text                           │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. AUDIO AGENT                                              │
│    - Takes: script                                          │
│    - Detects: narration needs, multi-character dialogue     │
│    - Generates: Timed transcript with silences              │
│    - Output: Audio file + timestamp data                    │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. LANGSEARCH (Research)                                    │
│    - Takes: script                                          │
│    - Identifies: terms needing research/context             │
│    - Searches: web for current info                         │
│    - Output: Contextual information for script terms        │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. BRAINBOX (Creative Planning)                             │
│    - Takes: script, audio transcript, style, search results │
│    - Studies: training examples of similar style            │
│    - Creates: Detailed video plan with timeline             │
│    - Defines: Pacing, asset types, sound design, theme      │
│    - Output: Structured video plan (atomic tasks)           │
│              Format: Asset type → Description → Script line  │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. ASSET COLLECTOR                                          │
│    - Takes: video plan (line by line)                       │
│    - For each asset:                                        │
│      - If AI generation needed → generates                  │
│      - If web search needed → searches and downloads        │
│    - Output: All video/image/audio assets collected         │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. EXECUTOR (Assembly & Rendering)                          │
│    - Takes: assets + video plan                             │
│    - Builds: timeline matching plan                         │
│    - Presents: Adobe-like timeline visualization            │
│    - Allows: User edits and manual tweaks                   │
│    - Refines: alignment with BrainBox vision                │
│    - Renders: Final video                                   │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 8. ANALYTICS (Learning & Logging)                           │
│    - Logs: all actions and decisions                        │
│    - Stores: training data                                  │
│    - Tracks: performance metrics                            │
└─────────────────────────────────────────────────────────────┘
DEVELOPMENT APPROACH
Phase 1: Build Independent Workers (CURRENT)

Build each worker as standalone, testable module
Each has clear input/output contracts
No inter-worker dependencies yet
Test thoroughly in isolation

Phase 2: Tool Calling Integration

Secretary gets tool calling ability
Connect workers through Secretary orchestration
Test end-to-end flow

Phase 3: Refinement & Optimization

Fine-tune each worker
Add training data collection
Optimize performance

TECHNICAL STACK

Language: Python
Environment: Jupyter Notebooks (Google Colab)
LLM: Local Ollama (llama3.1:8b) for development
Search: DuckDuckGo + LangSearch fallback
State Management: JSON files between workers
Future: LangGraph for orchestration

MODULE SPECIFICATIONS
1. SECRETARY
Inputs: User via form (6 questions)
Outputs: requirements.json
Tool Calling: Can call any worker
Key Features:

Structured input collection
Validation (no LLM)
Tool routing logic
Progress tracking

2. SCRIPTWRITER
Inputs: requirements.json
Outputs: script.txt
Process: Talking points → Bodies → Structure (Hook/Body/CTA)
Key Features:

Style-aware generation
Duration estimation
Modular generation (3 phases)

3. AUDIO AGENT
Inputs: script.txt
Outputs: audio.wav + transcript_timestamps.json
Key Features:

Multi-character detection
Silence timing
Narration vs dialogue classification

4. LANGSEARCH
Inputs: script.txt
Outputs: research_data.json
Key Features:

Term extraction from script
Contextual search
Relevance filtering

5. BRAINBOX
Inputs: script.txt, transcript_timestamps.json, research_data.json, requirements.json
Outputs: video_plan.json
Key Features:

Style training/loading
Timeline calculation
Asset type decisions
Format:

json{
  "timeline": [
    {
      "timestamp": "0:00-0:05",
      "asset_type": "video",
      "description": "Trump speaking at podium",
      "script_line": "So Trump decided..."
    }
  ]
}
```

### 6. ASSET COLLECTOR
**Inputs**: `video_plan.json`
**Outputs**: `assets/` folder + `asset_manifest.json`
**Key Features**:
- AI generation (images, videos)
- Web scraping
- Asset validation

### 7. EXECUTOR
**Inputs**: `video_plan.json`, `assets/`, `audio.wav`
**Outputs**: `final_video.mp4` + timeline visualization
**Key Features**:
- Timeline assembly
- Interactive editing UI
- Render pipeline

### 8. ANALYTICS
**Inputs**: All worker outputs + logs
**Outputs**: `analytics.json`, training datasets
**Key Features**:
- Performance tracking
- Training data collection
- Success metrics

## FILE STRUCTURE
```
project/
├── overview.md (this file)
├── workers/
│   ├── secretary/
│   │   ├── secretary.py
│   │   ├── test_secretary.py
│   │   └── requirements.txt
│   ├── scriptwriter/
│   │   ├── scriptwriter.py
│   │   ├── test_scriptwriter.py
│   │   └── requirements.txt
│   ├── audio_agent/
│   ├── langsearch/
│   ├── brainbox/
│   ├── asset_collector/
│   ├── executor/
│   └── analytics/
├── shared/
│   ├── search_tools.py (DuckDuckGo + LangSearch)
│   └── utils.py
├── outputs/
│   ├── requirements.json
│   ├── script.txt
│   ├── audio.wav
│   ├── video_plan.json
│   ├── assets/
│   └── final_video.mp4
└── tests/
    └── integration_tests.py
CURRENT STATUS
Phase: Building independent workers
Next Worker: Secretary
Testing Priority: Tool calling mechanism
DESIGN PRINCIPLES

Modularity: Each worker is independent
Testability: Comprehensive tests for each module
Simplicity: Clear contracts, minimal dependencies
Flexibility: Easy to swap/upgrade workers
Observability: Detailed logging at each step

SUCCESS CRITERIA

Each worker passes isolated tests
Tool calling mechanism works reliably
End-to-end pipeline produces coherent video
User can intervene at any stage
System learns from each run
