# Copyright 2026 Aoyang Fang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/bin/env python3
"""Run recovery tests without full verl dependencies."""

import os
import sys
import unittest

# Add the fault_injection directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import test modules
from test_recovery_hierarchical import TestHierarchicalRecoveryManager, TestIntelligentRecoveryDecisionEngine
from test_recovery_integration import TestRecoveryIntegration
from test_recovery_strategies import TestRecoveryStrategies

if __name__ == "__main__":
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add tests
    suite.addTests(loader.loadTestsFromTestCase(TestRecoveryStrategies))
    suite.addTests(loader.loadTestsFromTestCase(TestHierarchicalRecoveryManager))
    suite.addTests(loader.loadTestsFromTestCase(TestIntelligentRecoveryDecisionEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestRecoveryIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
