#!/usr/bin/env python3
"""Run recovery tests without full verl dependencies."""

import sys
import os
import unittest

# Add the fault_injection directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import test modules
from test_recovery_strategies import TestRecoveryStrategies
from test_recovery_hierarchical import TestHierarchicalRecoveryManager, TestIntelligentRecoveryDecisionEngine
from test_recovery_integration import TestRecoveryIntegration

if __name__ == '__main__':
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