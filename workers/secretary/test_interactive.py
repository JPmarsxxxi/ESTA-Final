#!/usr/bin/env python3
"""
Interactive Secretary Testing Script

Use this to test the Secretary with your own requirements.
"""

from secretary import Secretary, create_full_workflow
import json


def test_custom_requirements():
    """Test with your own custom requirements"""
    print("\n" + "="*60)
    print("CUSTOM REQUIREMENTS TEST")
    print("="*60)

    # ===== CUSTOMIZE YOUR REQUIREMENTS HERE =====
    my_requirements = {
        'topic': 'Your Video Topic Here (min 5 chars)',
        'style': 'tutorial',  # Options: funny, documentary, serious, graphic-heavy, tutorial, or custom
        'duration_range': '3-5 minutes',  # Format: "X-Y minutes/seconds" or "X minutes/seconds"
        'audio_mode': 'generate',  # Options: generate, upload
        'script_mode': 'generate',  # Options: generate, upload
        'comments': 'Any additional comments here (optional)'
    }
    # ============================================

    print("\nYour Requirements:")
    print(json.dumps(my_requirements, indent=2))

    # Initialize Secretary
    sec = Secretary()

    # Validate
    print("\n--- Validating ---")
    is_valid, errors = sec.validate_inputs(my_requirements)

    if is_valid:
        print("✅ Validation PASSED!")

        # Set requirements
        sec.set_requirements(my_requirements)

        # Test single tool
        print("\n--- Testing ScriptWriter ---")
        result = sec.call_tool('scriptwriter', my_requirements)

        if result['success']:
            print("✅ ScriptWriter Success!")
            print("\nGenerated Script:")
            print(result['outputs']['script'])

            # Test full workflow
            print("\n--- Testing Full Workflow ---")
            workflow = create_full_workflow(sec.requirements)
            workflow_result = sec.execute_workflow(workflow)

            if workflow_result['success']:
                print(f"\n✅ Full Workflow Complete!")
                print(f"Completed all {workflow_result['completed_steps']} steps")

                # Show outputs from each step
                print("\n--- Workflow Outputs ---")
                for step_idx, step_data in workflow_result['results'].items():
                    tool = step_data['tool']
                    outputs = list(step_data['result']['outputs'].keys())
                    print(f"  Step {step_idx + 1} ({tool}): {', '.join(outputs)}")
            else:
                print(f"\n❌ Workflow failed at step {workflow_result['failed_step'] + 1}")

        else:
            print(f"❌ ScriptWriter Failed: {result['error']}")

    else:
        print("❌ Validation FAILED!")
        print("\nErrors:")
        for error in errors:
            print(f"  - {error}")

        print("\n💡 Fix the errors above and try again!")


if __name__ == '__main__':
    test_custom_requirements()
