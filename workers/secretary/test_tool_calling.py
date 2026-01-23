"""
Unit Tests for Secretary Module - Tool Calling

Tests the tool calling system and workflow execution
"""

import unittest
from secretary import Secretary, create_full_workflow
from tool_registry import WorkerFactory, AVAILABLE_TOOLS


class TestToolCalling(unittest.TestCase):
    """Test tool calling functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.secretary = Secretary()
        WorkerFactory.reset_workers()

    def tearDown(self):
        """Clean up after tests"""
        self.secretary.reset()

    # ===== SINGLE TOOL CALL TESTS =====

    def test_call_scriptwriter_success(self):
        """Test calling scriptwriter tool successfully"""
        requirements = {
            'topic': 'Python Programming',
            'style': 'tutorial',
            'duration_range': '5 minutes',
            'comments': 'Beginner level'
        }

        result = self.secretary.call_tool('scriptwriter', requirements)

        self.assertTrue(result['success'])
        self.assertIn('script', result['outputs'])
        self.assertIn('metadata', result)
        self.assertEqual(result['metadata']['worker'], 'scriptwriter')

    def test_call_audio_agent_success(self):
        """Test calling audio agent successfully"""
        script = "This is a test script for audio generation."

        result = self.secretary.call_tool('audio_agent', {'script': script})

        self.assertTrue(result['success'])
        self.assertIn('audio_file', result['outputs'])
        self.assertIn('transcript_timestamps', result['outputs'])

    def test_call_langsearch_success(self):
        """Test calling langsearch successfully"""
        script = "Machine learning and neural networks are fascinating topics."

        result = self.secretary.call_tool('langsearch', {'script': script})

        self.assertTrue(result['success'])
        self.assertIn('research_data', result['outputs'])

    def test_call_brainbox_success(self):
        """Test calling brainbox successfully"""
        inputs = {
            'script': 'Test script',
            'transcript_timestamps': {'segments': []},
            'research_data': {'terms': []},
            'requirements': {'topic': 'Test', 'style': 'tutorial'}
        }

        result = self.secretary.call_tool('brainbox', inputs)

        self.assertTrue(result['success'])
        self.assertIn('video_plan', result['outputs'])

    def test_call_asset_collector_success(self):
        """Test calling asset collector successfully"""
        video_plan = {
            'timeline': [
                {'timestamp': '0:00-0:05', 'asset_type': 'video'}
            ]
        }

        result = self.secretary.call_tool('asset_collector', {'video_plan': video_plan})

        self.assertTrue(result['success'])
        self.assertIn('asset_manifest', result['outputs'])

    def test_call_executor_success(self):
        """Test calling executor successfully"""
        inputs = {
            'video_plan': {'timeline': []},
            'assets': {'assets': []},
            'audio': 'audio.wav'
        }

        result = self.secretary.call_tool('executor', inputs)

        self.assertTrue(result['success'])
        self.assertIn('video_file', result['outputs'])

    # ===== INVALID TOOL CALL TESTS =====

    def test_call_nonexistent_tool(self):
        """Test calling a non-existent tool"""
        result = self.secretary.call_tool('nonexistent_tool', {})

        self.assertFalse(result['success'])
        self.assertIn('error', result)
        self.assertIn('Unknown tool', result['error'])

    def test_call_tool_missing_required_inputs(self):
        """Test calling tool with missing required inputs"""
        result = self.secretary.call_tool('scriptwriter', {})

        self.assertFalse(result['success'])
        self.assertIn('error', result)
        self.assertIn('Invalid inputs', result['error'])

    def test_call_tool_partial_inputs(self):
        """Test calling tool with only some required inputs"""
        result = self.secretary.call_tool('scriptwriter', {
            'topic': 'Test Topic'
            # Missing: style, duration_range, comments
        })

        self.assertFalse(result['success'])
        self.assertIn('error', result)

    def test_call_brainbox_missing_inputs(self):
        """Test calling brainbox with missing inputs"""
        result = self.secretary.call_tool('brainbox', {
            'script': 'Test script'
            # Missing: transcript_timestamps, research_data, requirements
        })

        self.assertFalse(result['success'])

    # ===== TOOL OUTPUT VERIFICATION TESTS =====

    def test_scriptwriter_output_format(self):
        """Test scriptwriter returns properly formatted output"""
        requirements = {
            'topic': 'Test Topic',
            'style': 'tutorial',
            'duration_range': '2 minutes',
            'comments': 'Test'
        }

        result = self.secretary.call_tool('scriptwriter', requirements)

        self.assertTrue(result['success'])
        self.assertIn('outputs', result)
        self.assertIn('script', result['outputs'])
        self.assertIn('metadata', result)
        self.assertIn('timestamp', result['metadata'])

        # Verify script content includes topic
        script = result['outputs']['script']
        self.assertIn('Test Topic', script)

    def test_audio_agent_output_format(self):
        """Test audio agent returns properly formatted output"""
        result = self.secretary.call_tool('audio_agent', {
            'script': 'Test script'
        })

        self.assertTrue(result['success'])
        outputs = result['outputs']

        self.assertIn('audio_file', outputs)
        self.assertIn('transcript_timestamps', outputs)
        self.assertIn('segments', outputs['transcript_timestamps'])

        # Verify segments structure
        segments = outputs['transcript_timestamps']['segments']
        self.assertIsInstance(segments, list)
        if len(segments) > 0:
            self.assertIn('start', segments[0])
            self.assertIn('end', segments[0])
            self.assertIn('text', segments[0])

    def test_brainbox_output_format(self):
        """Test brainbox returns properly formatted video plan"""
        inputs = {
            'script': 'Test script',
            'transcript_timestamps': {'segments': []},
            'research_data': {'terms': []},
            'requirements': {'topic': 'AI', 'style': 'tutorial'}
        }

        result = self.secretary.call_tool('brainbox', inputs)

        self.assertTrue(result['success'])
        video_plan = result['outputs']['video_plan']

        self.assertIn('timeline', video_plan)
        self.assertIsInstance(video_plan['timeline'], list)

        # Verify timeline item structure
        if len(video_plan['timeline']) > 0:
            item = video_plan['timeline'][0]
            self.assertIn('timestamp', item)
            self.assertIn('asset_type', item)
            self.assertIn('description', item)

    # ===== MULTIPLE CALLS TESTS =====

    def test_multiple_calls_same_tool(self):
        """Test calling the same tool multiple times"""
        requirements = {
            'topic': 'Test Topic',
            'style': 'tutorial',
            'duration_range': '2 minutes',
            'comments': None
        }

        result1 = self.secretary.call_tool('scriptwriter', requirements)
        result2 = self.secretary.call_tool('scriptwriter', requirements)
        result3 = self.secretary.call_tool('scriptwriter', requirements)

        self.assertTrue(result1['success'])
        self.assertTrue(result2['success'])
        self.assertTrue(result3['success'])

        # Verify call counts increment
        worker = WorkerFactory.get_worker('scriptwriter')
        self.assertEqual(worker.call_count, 3)

    def test_multiple_different_tools(self):
        """Test calling different tools sequentially"""
        # Call scriptwriter
        result1 = self.secretary.call_tool('scriptwriter', {
            'topic': 'Test',
            'style': 'tutorial',
            'duration_range': '2 minutes',
            'comments': None
        })
        self.assertTrue(result1['success'])

        # Call audio agent
        result2 = self.secretary.call_tool('audio_agent', {
            'script': 'Test script'
        })
        self.assertTrue(result2['success'])

        # Call langsearch
        result3 = self.secretary.call_tool('langsearch', {
            'script': 'Test script'
        })
        self.assertTrue(result3['success'])

        # Verify all succeeded
        self.assertTrue(all([result1['success'], result2['success'], result3['success']]))


class TestWorkflowExecution(unittest.TestCase):
    """Test workflow execution functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.secretary = Secretary()
        WorkerFactory.reset_workers()

    def tearDown(self):
        """Clean up after tests"""
        self.secretary.reset()

    # ===== SIMPLE WORKFLOW TESTS =====

    def test_simple_workflow_two_steps(self):
        """Test workflow with two sequential steps"""
        workflow = [
            ('scriptwriter', {
                'topic': 'Test Topic',
                'style': 'tutorial',
                'duration_range': '2 minutes',
                'comments': None
            }),
            ('audio_agent', {'script': 'Test script'})
        ]

        result = self.secretary.execute_workflow(workflow)

        self.assertTrue(result['success'])
        self.assertEqual(result['total_steps'], 2)
        self.assertEqual(result['completed_steps'], 2)
        self.assertIsNone(result['failed_step'])

    def test_simple_workflow_three_steps(self):
        """Test workflow with three sequential steps"""
        workflow = [
            ('scriptwriter', {
                'topic': 'ML Basics',
                'style': 'tutorial',
                'duration_range': '3 minutes',
                'comments': 'Beginner level'
            }),
            ('audio_agent', {'script': 'Test script'}),
            ('langsearch', {'script': 'Test script'})
        ]

        result = self.secretary.execute_workflow(workflow)

        self.assertTrue(result['success'])
        self.assertEqual(result['total_steps'], 3)
        self.assertEqual(result['completed_steps'], 3)

        # Verify all steps have results
        self.assertEqual(len(result['results']), 3)
        for step_idx in range(3):
            self.assertIn(step_idx, result['results'])
            self.assertTrue(result['results'][step_idx]['result']['success'])

    # ===== FULL PIPELINE WORKFLOW TEST =====

    def test_full_workflow_all_workers(self):
        """Test complete workflow through all workers"""
        requirements = {
            'topic': 'Introduction to AI',
            'style': 'tutorial',
            'duration_range': '5 minutes',
            'comments': 'Make it engaging',
            'audio_mode': 'generate',
            'script_mode': 'generate'
        }

        # Create full workflow
        workflow = create_full_workflow(requirements)

        # Execute
        result = self.secretary.execute_workflow(workflow)

        self.assertTrue(result['success'], f"Workflow failed: {result}")
        self.assertEqual(result['total_steps'], 6)
        self.assertEqual(result['completed_steps'], 6)

        # Verify each worker was called
        expected_tools = ['scriptwriter', 'audio_agent', 'langsearch',
                         'brainbox', 'asset_collector', 'executor']

        for idx, tool_name in enumerate(expected_tools):
            self.assertEqual(result['results'][idx]['tool'], tool_name)
            self.assertTrue(result['results'][idx]['result']['success'])

    # ===== WORKFLOW FAILURE TESTS =====

    def test_workflow_fails_at_first_step(self):
        """Test workflow fails gracefully at first step"""
        workflow = [
            ('scriptwriter', {}),  # Missing required inputs
            ('audio_agent', {'script': 'Test'})
        ]

        result = self.secretary.execute_workflow(workflow)

        self.assertFalse(result['success'])
        self.assertEqual(result['failed_step'], 0)
        self.assertEqual(result['completed_steps'], 0)

    def test_workflow_fails_at_middle_step(self):
        """Test workflow fails gracefully in middle"""
        workflow = [
            ('scriptwriter', {
                'topic': 'Test',
                'style': 'tutorial',
                'duration_range': '2 minutes',
                'comments': None
            }),
            ('audio_agent', {}),  # Missing required input
            ('langsearch', {'script': 'Test'})
        ]

        result = self.secretary.execute_workflow(workflow)

        self.assertFalse(result['success'])
        self.assertEqual(result['failed_step'], 1)
        self.assertEqual(result['completed_steps'], 1)

        # Verify first step succeeded
        self.assertTrue(result['results'][0]['result']['success'])

    def test_workflow_with_nonexistent_tool(self):
        """Test workflow fails with non-existent tool"""
        workflow = [
            ('scriptwriter', {
                'topic': 'Test',
                'style': 'tutorial',
                'duration_range': '2 minutes',
                'comments': None
            }),
            ('nonexistent_tool', {})
        ]

        result = self.secretary.execute_workflow(workflow)

        self.assertFalse(result['success'])
        self.assertEqual(result['failed_step'], 1)

    # ===== WORKFLOW OUTPUT PASSING TESTS =====

    def test_workflow_passes_outputs_between_steps(self):
        """Test that outputs from one step can be used in next step"""
        workflow = [
            ('scriptwriter', {
                'topic': 'Test Topic',
                'style': 'tutorial',
                'duration_range': '2 minutes',
                'comments': None
            }),
            ('audio_agent', {'script': '$scriptwriter.script'})
        ]

        result = self.secretary.execute_workflow(workflow)

        self.assertTrue(result['success'])

        # Verify scriptwriter output was passed to audio agent
        scriptwriter_output = result['results'][0]['result']['outputs']
        self.assertIn('script', scriptwriter_output)

    def test_workflow_with_complex_output_passing(self):
        """Test workflow with multiple output dependencies"""
        requirements = {
            'topic': 'AI Ethics',
            'style': 'documentary',
            'duration_range': '3 minutes',
            'comments': 'Focus on real-world examples'
        }

        workflow = [
            ('scriptwriter', requirements),
            ('audio_agent', {'script': '$scriptwriter.script'}),
            ('langsearch', {'script': '$scriptwriter.script'}),
            ('brainbox', {
                'script': '$scriptwriter.script',
                'transcript_timestamps': '$audio_agent.transcript_timestamps',
                'research_data': '$langsearch.research_data',
                'requirements': requirements
            })
        ]

        result = self.secretary.execute_workflow(workflow)

        self.assertTrue(result['success'])
        self.assertEqual(result['completed_steps'], 4)

    # ===== WORKFLOW HISTORY TESTS =====

    def test_workflow_history_recorded(self):
        """Test that workflow execution is recorded in history"""
        workflow = [
            ('scriptwriter', {
                'topic': 'Test',
                'style': 'tutorial',
                'duration_range': '2 minutes',
                'comments': None
            })
        ]

        self.assertEqual(len(self.secretary.workflow_history), 0)

        self.secretary.execute_workflow(workflow)

        self.assertEqual(len(self.secretary.workflow_history), 1)
        history_entry = self.secretary.workflow_history[0]
        self.assertIn('workflow', history_entry)
        self.assertIn('results', history_entry)
        self.assertIn('timestamp', history_entry)

    def test_multiple_workflows_recorded(self):
        """Test multiple workflows are recorded separately"""
        workflow1 = [('scriptwriter', {
            'topic': 'Test 1',
            'style': 'tutorial',
            'duration_range': '1 minute',
            'comments': None
        })]

        workflow2 = [('audio_agent', {'script': 'Test script'})]

        self.secretary.execute_workflow(workflow1)
        self.secretary.execute_workflow(workflow2)

        self.assertEqual(len(self.secretary.workflow_history), 2)


class TestToolRegistry(unittest.TestCase):
    """Test tool registry functionality"""

    def test_all_tools_registered(self):
        """Test that all expected tools are registered"""
        expected_tools = [
            'scriptwriter',
            'audio_agent',
            'langsearch',
            'brainbox',
            'asset_collector',
            'executor'
        ]

        for tool in expected_tools:
            self.assertIn(tool, AVAILABLE_TOOLS)

    def test_tool_specifications_complete(self):
        """Test that each tool has complete specifications"""
        required_keys = ['description', 'required_inputs', 'outputs']

        for tool_name, tool_spec in AVAILABLE_TOOLS.items():
            for key in required_keys:
                self.assertIn(key, tool_spec,
                            f"Tool {tool_name} missing key: {key}")

    def test_worker_factory_creates_workers(self):
        """Test that worker factory can create all workers"""
        WorkerFactory.reset_workers()

        for tool_name in AVAILABLE_TOOLS.keys():
            worker = WorkerFactory.get_worker(tool_name)
            self.assertIsNotNone(worker)

    def test_worker_factory_reuses_instances(self):
        """Test that worker factory reuses instances"""
        WorkerFactory.reset_workers()

        worker1 = WorkerFactory.get_worker('scriptwriter')
        worker2 = WorkerFactory.get_worker('scriptwriter')

        self.assertIs(worker1, worker2)

    def test_worker_factory_reset(self):
        """Test that worker factory reset clears instances"""
        WorkerFactory.get_worker('scriptwriter')
        WorkerFactory.reset_workers()

        # After reset, should create new instance
        worker = WorkerFactory.get_worker('scriptwriter')
        self.assertEqual(worker.call_count, 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows"""

    def setUp(self):
        """Set up test fixtures"""
        self.secretary = Secretary()
        WorkerFactory.reset_workers()

    def tearDown(self):
        """Clean up"""
        self.secretary.reset()

    def test_complete_video_generation_flow(self):
        """Test complete flow from requirements to final video"""
        # Set requirements
        requirements = {
            'topic': 'Climate Change Solutions',
            'style': 'documentary',
            'duration_range': '3-5 minutes',
            'audio_mode': 'generate',
            'script_mode': 'generate',
            'comments': 'Include statistics and expert opinions'
        }

        is_valid, errors = self.secretary.set_requirements(requirements)
        self.assertTrue(is_valid, f"Invalid requirements: {errors}")

        # Create and execute full workflow
        workflow = create_full_workflow(self.secretary.requirements)
        result = self.secretary.execute_workflow(workflow)

        # Verify success
        self.assertTrue(result['success'], "Workflow execution failed")
        self.assertEqual(result['total_steps'], 6)
        self.assertEqual(result['completed_steps'], 6)

        # Verify each step
        step_names = ['scriptwriter', 'audio_agent', 'langsearch',
                     'brainbox', 'asset_collector', 'executor']

        for idx, expected_name in enumerate(step_names):
            step_result = result['results'][idx]
            self.assertEqual(step_result['tool'], expected_name)
            self.assertTrue(step_result['result']['success'])

        # Verify final output exists
        final_result = result['results'][5]['result']
        self.assertIn('video_file', final_result['outputs'])

    def test_workflow_with_saved_requirements(self):
        """Test workflow using saved and loaded requirements"""
        import tempfile
        import os

        # Create requirements and save
        requirements = {
            'topic': 'Python Best Practices',
            'style': 'tutorial',
            'duration_range': '5 minutes',
            'audio_mode': 'generate',
            'script_mode': 'generate',
            'comments': 'Include code examples'
        }

        self.secretary.set_requirements(requirements)

        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        temp_file.close()

        try:
            self.secretary.save_requirements(temp_file.name)

            # Create new secretary and load requirements
            new_secretary = Secretary()
            loaded_reqs = new_secretary.load_requirements(temp_file.name)

            # Execute workflow with loaded requirements
            workflow = [
                ('scriptwriter', loaded_reqs)
            ]

            result = new_secretary.execute_workflow(workflow)
            self.assertTrue(result['success'])

        finally:
            os.unlink(temp_file.name)


if __name__ == '__main__':
    unittest.main()
