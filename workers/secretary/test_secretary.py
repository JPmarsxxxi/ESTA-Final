"""
Unit Tests for Secretary Module - Input Validation

Tests all validation logic for user requirements collection
"""

import unittest
import json
import tempfile
from pathlib import Path
from secretary import Secretary


class TestSecretaryValidation(unittest.TestCase):
    """Test input validation functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.secretary = Secretary()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up after tests"""
        # Clean up temp files
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    # ===== VALID INPUT TESTS =====

    def test_valid_complete_inputs_preset_style(self):
        """Test validation with complete valid inputs using preset style"""
        inputs = {
            'topic': 'Introduction to Python',
            'style': 'tutorial',
            'duration_range': '5-10 minutes',
            'audio_mode': 'generate',
            'script_mode': 'generate',
            'comments': 'Make it beginner-friendly'
        }

        is_valid, errors = self.secretary.validate_inputs(inputs)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    def test_valid_complete_inputs_custom_style(self):
        """Test validation with complete valid inputs using custom style"""
        inputs = {
            'topic': 'Machine Learning Basics',
            'style': 'Educational and engaging',
            'duration_range': '2-3 minutes',
            'audio_mode': 'upload',
            'script_mode': 'generate',
            'comments': None
        }

        is_valid, errors = self.secretary.validate_inputs(inputs)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    def test_valid_minimal_inputs(self):
        """Test validation with minimal valid inputs (no comments)"""
        inputs = {
            'topic': 'Quick Tech Tips',
            'style': 'funny',
            'duration_range': '30 seconds',
            'audio_mode': 'generate',
            'script_mode': 'upload',
            'comments': None
        }

        is_valid, errors = self.secretary.validate_inputs(inputs)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    def test_valid_all_style_presets(self):
        """Test validation with all style presets"""
        presets = ["funny", "documentary", "serious", "graphic-heavy", "tutorial"]

        for preset in presets:
            inputs = {
                'topic': f'Test video for {preset} style',
                'style': preset,
                'duration_range': '1 minute',
                'audio_mode': 'generate',
                'script_mode': 'generate',
                'comments': None
            }

            is_valid, errors = self.secretary.validate_inputs(inputs)
            self.assertTrue(is_valid, f"Failed for preset: {preset}")
            self.assertEqual(len(errors), 0)

    # ===== DURATION FORMAT TESTS =====

    def test_valid_duration_formats(self):
        """Test various valid duration formats"""
        valid_formats = [
            '1 minute',
            '2 minutes',
            '30 seconds',
            '45 secs',
            '1-2 minutes',
            '30-60 seconds',
            '2-5 mins',
            '10-15 secs'
        ]

        for duration in valid_formats:
            inputs = {
                'topic': 'Duration test',
                'style': 'tutorial',
                'duration_range': duration,
                'audio_mode': 'generate',
                'script_mode': 'generate',
                'comments': None
            }

            is_valid, errors = self.secretary.validate_inputs(inputs)
            self.assertTrue(is_valid, f"Failed for duration: {duration}")
            self.assertEqual(len(errors), 0)

    def test_invalid_duration_formats(self):
        """Test invalid duration formats"""
        invalid_formats = [
            '',
            'five minutes',
            '2-3',
            'minutes',
            '1.5 minutes',
            '2 hours',
            'really long',
            '1 - 2 minutes',  # spaces around dash
            '1min',  # no space
        ]

        for duration in invalid_formats:
            inputs = {
                'topic': 'Duration test',
                'style': 'tutorial',
                'duration_range': duration,
                'audio_mode': 'generate',
                'script_mode': 'generate',
                'comments': None
            }

            is_valid, errors = self.secretary.validate_inputs(inputs)
            self.assertFalse(is_valid, f"Should fail for duration: {duration}")
            self.assertGreater(len(errors), 0)

    # ===== TOPIC VALIDATION TESTS =====

    def test_invalid_topic_empty(self):
        """Test validation fails with empty topic"""
        inputs = {
            'topic': '',
            'style': 'tutorial',
            'duration_range': '2 minutes',
            'audio_mode': 'generate',
            'script_mode': 'generate',
            'comments': None
        }

        is_valid, errors = self.secretary.validate_inputs(inputs)
        self.assertFalse(is_valid)
        self.assertIn("Topic is required", errors)

    def test_invalid_topic_too_short(self):
        """Test validation fails with topic < 5 chars"""
        inputs = {
            'topic': 'AI',  # Only 2 chars
            'style': 'tutorial',
            'duration_range': '2 minutes',
            'audio_mode': 'generate',
            'script_mode': 'generate',
            'comments': None
        }

        is_valid, errors = self.secretary.validate_inputs(inputs)
        self.assertFalse(is_valid)
        self.assertIn("at least 5 characters", errors[0])

    def test_valid_topic_exactly_5_chars(self):
        """Test validation passes with topic exactly 5 chars"""
        inputs = {
            'topic': 'Tests',  # Exactly 5 chars
            'style': 'tutorial',
            'duration_range': '2 minutes',
            'audio_mode': 'generate',
            'script_mode': 'generate',
            'comments': None
        }

        is_valid, errors = self.secretary.validate_inputs(inputs)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    # ===== STYLE VALIDATION TESTS =====

    def test_invalid_style_empty(self):
        """Test validation fails with empty style"""
        inputs = {
            'topic': 'Valid Topic',
            'style': '',
            'duration_range': '2 minutes',
            'audio_mode': 'generate',
            'script_mode': 'generate',
            'comments': None
        }

        is_valid, errors = self.secretary.validate_inputs(inputs)
        self.assertFalse(is_valid)
        self.assertIn("Style is required", errors)

    def test_invalid_custom_style_too_short(self):
        """Test validation fails with custom style < 3 chars"""
        inputs = {
            'topic': 'Valid Topic',
            'style': 'ab',  # Only 2 chars
            'duration_range': '2 minutes',
            'audio_mode': 'generate',
            'script_mode': 'generate',
            'comments': None
        }

        is_valid, errors = self.secretary.validate_inputs(inputs)
        self.assertFalse(is_valid)
        self.assertIn("at least 3 characters", errors[0])

    def test_valid_custom_style_exactly_3_chars(self):
        """Test validation passes with custom style exactly 3 chars"""
        inputs = {
            'topic': 'Valid Topic',
            'style': 'Fun',  # Exactly 3 chars
            'duration_range': '2 minutes',
            'audio_mode': 'generate',
            'script_mode': 'generate',
            'comments': None
        }

        is_valid, errors = self.secretary.validate_inputs(inputs)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    # ===== AUDIO/SCRIPT MODE TESTS =====

    def test_invalid_audio_mode(self):
        """Test validation fails with invalid audio mode"""
        invalid_modes = ['created', 'make', 'auto', '', 'GENERATE ', 'generate ']

        for mode in invalid_modes:
            inputs = {
                'topic': 'Valid Topic',
                'style': 'tutorial',
                'duration_range': '2 minutes',
                'audio_mode': mode,
                'script_mode': 'generate',
                'comments': None
            }

            is_valid, errors = self.secretary.validate_inputs(inputs)
            self.assertFalse(is_valid, f"Should fail for audio_mode: '{mode}'")

    def test_invalid_script_mode(self):
        """Test validation fails with invalid script mode"""
        invalid_modes = ['created', 'make', 'auto', '', 'UPLOAD ', 'upload ']

        for mode in invalid_modes:
            inputs = {
                'topic': 'Valid Topic',
                'style': 'tutorial',
                'duration_range': '2 minutes',
                'audio_mode': 'generate',
                'script_mode': mode,
                'comments': None
            }

            is_valid, errors = self.secretary.validate_inputs(inputs)
            self.assertFalse(is_valid, f"Should fail for script_mode: '{mode}'")

    def test_valid_audio_script_combinations(self):
        """Test all valid combinations of audio and script modes"""
        combinations = [
            ('generate', 'generate'),
            ('generate', 'upload'),
            ('upload', 'generate'),
            ('upload', 'upload')
        ]

        for audio, script in combinations:
            inputs = {
                'topic': 'Valid Topic',
                'style': 'tutorial',
                'duration_range': '2 minutes',
                'audio_mode': audio,
                'script_mode': script,
                'comments': None
            }

            is_valid, errors = self.secretary.validate_inputs(inputs)
            self.assertTrue(is_valid, f"Failed for combo: {audio}/{script}")

    # ===== EDGE CASE TESTS =====

    def test_special_characters_in_topic(self):
        """Test validation with special characters in topic"""
        inputs = {
            'topic': 'Python & AI: The Future!',
            'style': 'tutorial',
            'duration_range': '2 minutes',
            'audio_mode': 'generate',
            'script_mode': 'generate',
            'comments': None
        }

        is_valid, errors = self.secretary.validate_inputs(inputs)
        self.assertTrue(is_valid)

    def test_unicode_characters(self):
        """Test validation with unicode characters"""
        inputs = {
            'topic': 'Café Culture in Paris 🇫🇷',
            'style': 'documentary',
            'duration_range': '3 minutes',
            'audio_mode': 'generate',
            'script_mode': 'generate',
            'comments': 'Include French accents: café, résumé'
        }

        is_valid, errors = self.secretary.validate_inputs(inputs)
        self.assertTrue(is_valid)

    def test_very_long_inputs(self):
        """Test validation with very long inputs"""
        inputs = {
            'topic': 'A' * 500,  # 500 chars
            'style': 'Very detailed custom style description ' * 10,  # Long style
            'duration_range': '10-15 minutes',
            'audio_mode': 'generate',
            'script_mode': 'generate',
            'comments': 'Lorem ipsum ' * 100  # Long comments
        }

        is_valid, errors = self.secretary.validate_inputs(inputs)
        self.assertTrue(is_valid)

    def test_whitespace_handling(self):
        """Test validation handles leading/trailing whitespace"""
        inputs = {
            'topic': '  Valid Topic  ',
            'style': '  tutorial  ',
            'duration_range': '  2 minutes  ',
            'audio_mode': '  generate  ',
            'script_mode': '  upload  ',
            'comments': '  Some comments  '
        }

        # Should fail because modes have spaces
        is_valid, errors = self.secretary.validate_inputs(inputs)
        self.assertFalse(is_valid)

    # ===== MULTIPLE ERRORS TEST =====

    def test_multiple_validation_errors(self):
        """Test validation catches multiple errors at once"""
        inputs = {
            'topic': 'AI',  # Too short
            'style': 'ab',  # Too short
            'duration_range': 'invalid',  # Invalid format
            'audio_mode': 'wrong',  # Invalid mode
            'script_mode': 'bad',  # Invalid mode
            'comments': None
        }

        is_valid, errors = self.secretary.validate_inputs(inputs)
        self.assertFalse(is_valid)
        self.assertGreaterEqual(len(errors), 5)  # Should have at least 5 errors


class TestSecretaryRequirements(unittest.TestCase):
    """Test requirements management functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.secretary = Secretary()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up after tests"""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_set_requirements_valid(self):
        """Test setting requirements programmatically"""
        requirements = {
            'topic': 'Test Video',
            'style': 'tutorial',
            'duration_range': '2 minutes',
            'audio_mode': 'generate',
            'script_mode': 'generate',
            'comments': 'Test comments'
        }

        is_valid, errors = self.secretary.set_requirements(requirements)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        self.assertIsNotNone(self.secretary.requirements)
        self.assertEqual(self.secretary.requirements['topic'], 'Test Video')

    def test_set_requirements_invalid(self):
        """Test setting invalid requirements"""
        requirements = {
            'topic': 'AI',  # Too short
            'style': 'tutorial',
            'duration_range': 'invalid',
            'audio_mode': 'wrong',
            'script_mode': 'bad',
            'comments': None
        }

        is_valid, errors = self.secretary.set_requirements(requirements)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)

    def test_save_requirements(self):
        """Test saving requirements to file"""
        requirements = {
            'topic': 'Test Video',
            'style': 'tutorial',
            'duration_range': '2 minutes',
            'audio_mode': 'generate',
            'script_mode': 'generate',
            'comments': 'Test'
        }

        self.secretary.set_requirements(requirements)
        filepath = Path(self.temp_dir) / "test_requirements.json"
        saved_path = self.secretary.save_requirements(str(filepath))

        self.assertTrue(Path(saved_path).exists())

        # Verify content
        with open(saved_path, 'r') as f:
            loaded = json.load(f)

        self.assertEqual(loaded['topic'], 'Test Video')
        self.assertEqual(loaded['status'], 'validated')

    def test_save_requirements_without_setting(self):
        """Test saving requirements without setting them first"""
        with self.assertRaises(ValueError):
            self.secretary.save_requirements()

    def test_load_requirements(self):
        """Test loading requirements from file"""
        requirements = {
            'topic': 'Test Video',
            'style': 'tutorial',
            'duration_range': '2 minutes',
            'audio_mode': 'generate',
            'script_mode': 'generate',
            'comments': 'Test'
        }

        self.secretary.set_requirements(requirements)
        filepath = Path(self.temp_dir) / "test_requirements.json"
        self.secretary.save_requirements(str(filepath))

        # Create new secretary and load
        new_secretary = Secretary()
        loaded = new_secretary.load_requirements(str(filepath))

        self.assertEqual(loaded['topic'], 'Test Video')
        self.assertEqual(loaded['style'], 'tutorial')

    def test_load_nonexistent_file(self):
        """Test loading from non-existent file"""
        with self.assertRaises(FileNotFoundError):
            self.secretary.load_requirements("/nonexistent/file.json")

    def test_requirements_include_timestamp(self):
        """Test that saved requirements include timestamp"""
        requirements = {
            'topic': 'Test Video',
            'style': 'tutorial',
            'duration_range': '2 minutes',
            'audio_mode': 'generate',
            'script_mode': 'generate',
            'comments': None
        }

        self.secretary.set_requirements(requirements)
        self.assertIn('timestamp', self.secretary.requirements)
        self.assertIn('status', self.secretary.requirements)
        self.assertEqual(self.secretary.requirements['status'], 'validated')


class TestSecretaryUtilities(unittest.TestCase):
    """Test utility methods"""

    def setUp(self):
        """Set up test fixtures"""
        self.secretary = Secretary()

    def test_get_tool_info_all(self):
        """Test getting info for all tools"""
        info = self.secretary.get_tool_info()
        self.assertIsInstance(info, dict)
        self.assertIn('scriptwriter', info)
        self.assertIn('audio_agent', info)
        self.assertIn('executor', info)

    def test_get_tool_info_specific(self):
        """Test getting info for specific tool"""
        info = self.secretary.get_tool_info('scriptwriter')
        self.assertIn('scriptwriter', info)
        self.assertEqual(len(info), 1)

    def test_get_tool_info_invalid(self):
        """Test getting info for invalid tool"""
        info = self.secretary.get_tool_info('nonexistent')
        self.assertIn('error', info)

    def test_reset(self):
        """Test resetting secretary state"""
        requirements = {
            'topic': 'Test Video',
            'style': 'tutorial',
            'duration_range': '2 minutes',
            'audio_mode': 'generate',
            'script_mode': 'generate',
            'comments': None
        }

        self.secretary.set_requirements(requirements)
        self.assertIsNotNone(self.secretary.requirements)

        self.secretary.reset()
        self.assertEqual(self.secretary.requirements, {})
        self.assertEqual(self.secretary.workflow_history, [])


if __name__ == '__main__':
    unittest.main()
