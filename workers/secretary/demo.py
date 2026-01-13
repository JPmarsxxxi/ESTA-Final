#!/usr/bin/env python3
"""
Secretary Module Demo Script

This script demonstrates the Secretary module capabilities.
Run it to see interactive examples of validation and tool calling.
"""

from secretary import Secretary, create_full_workflow
from tool_registry import AVAILABLE_TOOLS
import json


def demo_1_manual_validation():
    """Demo 1: Manual validation with different inputs"""
    print("\n" + "="*60)
    print("DEMO 1: Input Validation")
    print("="*60)

    sec = Secretary()

    # Test cases
    test_cases = [
        {
            "name": "✅ Valid Complete Input",
            "data": {
                'topic': 'Introduction to Python Programming',
                'style': 'tutorial',
                'duration_range': '5-7 minutes',
                'audio_mode': 'generate',
                'script_mode': 'generate',
                'comments': 'Make it beginner-friendly'
            }
        },
        {
            "name": "❌ Invalid Topic (too short)",
            "data": {
                'topic': 'AI',  # Only 2 chars
                'style': 'tutorial',
                'duration_range': '5 minutes',
                'audio_mode': 'generate',
                'script_mode': 'generate',
                'comments': None
            }
        },
        {
            "name": "❌ Invalid Duration Format",
            "data": {
                'topic': 'Valid Topic Here',
                'style': 'funny',
                'duration_range': 'five minutes',  # Wrong format
                'audio_mode': 'generate',
                'script_mode': 'upload',
                'comments': None
            }
        },
        {
            "name": "✅ Valid Custom Style",
            "data": {
                'topic': 'Climate Change Solutions',
                'style': 'Educational and engaging',  # Custom style
                'duration_range': '3-5 minutes',
                'audio_mode': 'upload',
                'script_mode': 'generate',
                'comments': 'Include statistics'
            }
        }
    ]

    for test_case in test_cases:
        print(f"\nTest: {test_case['name']}")
        print(f"Input: {json.dumps(test_case['data'], indent=2)}")

        is_valid, errors = sec.validate_inputs(test_case['data'])

        if is_valid:
            print("Result: ✅ VALID")
        else:
            print("Result: ❌ INVALID")
            for error in errors:
                print(f"  - {error}")


def demo_2_tool_calling():
    """Demo 2: Calling individual tools"""
    print("\n" + "="*60)
    print("DEMO 2: Tool Calling")
    print("="*60)

    sec = Secretary()

    # Set requirements
    requirements = {
        'topic': 'Machine Learning Basics',
        'style': 'tutorial',
        'duration_range': '5 minutes',
        'audio_mode': 'generate',
        'script_mode': 'generate',
        'comments': 'Focus on practical examples'
    }

    print("\nRequirements:")
    print(json.dumps(requirements, indent=2))

    # Call scriptwriter
    print("\n--- Calling Scriptwriter ---")
    result = sec.call_tool('scriptwriter', requirements)

    if result['success']:
        print("✅ Success!")
        print(f"Generated script preview:")
        script = result['outputs']['script']
        print(script[:300] + "...\n")

        # Call audio agent with the script
        print("\n--- Calling Audio Agent ---")
        audio_result = sec.call_tool('audio_agent', {'script': script})

        if audio_result['success']:
            print("✅ Success!")
            print(f"Audio file: {audio_result['outputs']['audio_file']}")
            print(f"Duration: {audio_result['outputs']['duration']} seconds")
            print(f"Segments: {len(audio_result['outputs']['transcript_timestamps']['segments'])}")
    else:
        print(f"❌ Failed: {result['error']}")


def demo_3_full_workflow():
    """Demo 3: Execute complete workflow"""
    print("\n" + "="*60)
    print("DEMO 3: Full Workflow Execution")
    print("="*60)

    sec = Secretary()

    # Set requirements
    requirements = {
        'topic': 'Climate Change Impact on Oceans',
        'style': 'documentary',
        'duration_range': '3-5 minutes',
        'audio_mode': 'generate',
        'script_mode': 'generate',
        'comments': 'Include scientific data and expert opinions'
    }

    print("\nRequirements:")
    print(json.dumps(requirements, indent=2))

    is_valid, errors = sec.set_requirements(requirements)

    if not is_valid:
        print(f"❌ Invalid requirements: {errors}")
        return

    print("\n✅ Requirements validated!")

    # Create and execute full workflow
    print("\n--- Creating Full Workflow ---")
    workflow = create_full_workflow(sec.requirements)

    print(f"Workflow has {len(workflow)} steps:")
    for idx, (tool_name, _) in enumerate(workflow):
        print(f"  {idx + 1}. {tool_name}")

    print("\n--- Executing Workflow ---")
    result = sec.execute_workflow(workflow)

    if result['success']:
        print(f"\n✅ Workflow completed successfully!")
        print(f"Total steps: {result['total_steps']}")
        print(f"Completed: {result['completed_steps']}")

        print("\n--- Step Results ---")
        for step_idx, step_data in result['results'].items():
            tool_name = step_data['tool']
            step_result = step_data['result']
            print(f"\nStep {step_idx + 1}: {tool_name}")
            print(f"  Status: {'✅ Success' if step_result['success'] else '❌ Failed'}")
            print(f"  Outputs: {list(step_result['outputs'].keys())}")

        # Show final video info
        final_result = result['results'][5]['result']
        video_info = final_result['outputs']
        print("\n--- Final Video ---")
        print(f"File: {video_info['video_file']}")
        print(f"Duration: {video_info['duration']} seconds")
        print(f"Resolution: {video_info['resolution']}")
        print(f"Size: {video_info['file_size_mb']} MB")
    else:
        print(f"\n❌ Workflow failed at step {result['failed_step'] + 1}")
        failed_step = result['results'][result['failed_step']]
        print(f"Failed tool: {failed_step['tool']}")
        print(f"Error: {failed_step['result']['error']}")


def demo_4_available_tools():
    """Demo 4: Show available tools"""
    print("\n" + "="*60)
    print("DEMO 4: Available Tools")
    print("="*60)

    sec = Secretary()

    for tool_name, tool_spec in AVAILABLE_TOOLS.items():
        print(f"\n{tool_name.upper()}")
        print(f"  Description: {tool_spec['description']}")
        print(f"  Required inputs: {', '.join(tool_spec['required_inputs'])}")
        print(f"  Outputs: {', '.join(tool_spec['outputs'])}")
        print(f"  Execution time: {tool_spec['execution_time']}")


def demo_5_save_load():
    """Demo 5: Save and load requirements"""
    print("\n" + "="*60)
    print("DEMO 5: Save and Load Requirements")
    print("="*60)

    sec = Secretary()

    requirements = {
        'topic': 'Python Best Practices',
        'style': 'tutorial',
        'duration_range': '10 minutes',
        'audio_mode': 'generate',
        'script_mode': 'generate',
        'comments': 'Include code examples'
    }

    print("\nSetting requirements...")
    sec.set_requirements(requirements)

    print("Saving to file...")
    filepath = sec.save_requirements('./test_requirements.json')
    print(f"✅ Saved to: {filepath}")

    print("\nLoading from file...")
    new_sec = Secretary()
    loaded_reqs = new_sec.load_requirements(filepath)
    print(f"✅ Loaded successfully!")
    print(json.dumps(loaded_reqs, indent=2))


def main():
    """Run all demos"""
    print("\n" + "="*60)
    print("SECRETARY MODULE DEMO")
    print("="*60)
    print("\nThis demo showcases the Secretary module capabilities:")
    print("1. Input validation")
    print("2. Single tool calling")
    print("3. Full workflow execution")
    print("4. Available tools info")
    print("5. Save/load requirements")

    try:
        demo_1_manual_validation()
        demo_2_tool_calling()
        demo_3_full_workflow()
        demo_4_available_tools()
        demo_5_save_load()

        print("\n" + "="*60)
        print("✅ All demos completed successfully!")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
