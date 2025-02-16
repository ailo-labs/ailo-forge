# test_generators.py

import unittest
from src.toggles import apply_toggle_settings

class TestToggles(unittest.TestCase):
    def test_apply_toggle_settings(self):
        prompt = "Base prompt."
        toggles = {"unrestricted": True, "domain_focus": "Medical"}
        modified_prompt = apply_toggle_settings(prompt, toggles)
        self.assertIn("UNRESTRICTED MODE", modified_prompt)
        self.assertIn("DOMAIN FOCUS: Medical", modified_prompt)

if __name__ == "__main__":
    unittest.main()

