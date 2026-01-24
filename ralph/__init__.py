"""
Ralph - Fault Injection Framework for verl.

A systematic fault injection and chaos engineering framework for testing
resilience in distributed reinforcement learning systems.

Dependencies:
    - pyyaml: Required for YAML configuration parsing
    - torch: Required for tensor operations (provided by verl)
    - ray: Required for distributed computing (provided by verl)
"""

__version__ = "0.1.0"

# Dependency checking


def check_dependencies():
    """Check that required dependencies are available.

    Returns:
        dict: Status of each dependency with version info.

    Raises:
        ImportError: If a critical dependency is missing.
    """
    status = {}

    # Check pyyaml (required)
    try:
        import yaml

        status["pyyaml"] = {"available": True, "version": yaml.__version__}
    except ImportError as e:
        status["pyyaml"] = {"available": False, "error": str(e)}
        raise ImportError(
            "pyyaml is required for Ralph. Install it with: pip install pyyaml"
        ) from e

    # Check torch (optional - provided by verl runtime)
    try:
        import torch

        status["torch"] = {"available": True, "version": torch.__version__}
    except ImportError:
        status["torch"] = {
            "available": False,
            "note": "torch is expected in verl runtime",
        }

    # Check ray (optional - provided by verl runtime)
    try:
        import ray

        status["ray"] = {"available": True, "version": ray.__version__}
    except ImportError:
        status["ray"] = {
            "available": False,
            "note": "ray is expected in verl runtime",
        }

    return status


def get_dependency_status():
    """Get the status of all dependencies without raising errors.

    Returns:
        dict: Status of each dependency.
    """
    status = {}

    try:
        import yaml

        status["pyyaml"] = {"available": True, "version": yaml.__version__}
    except ImportError as e:
        status["pyyaml"] = {"available": False, "error": str(e)}

    try:
        import torch

        status["torch"] = {"available": True, "version": torch.__version__}
    except ImportError:
        status["torch"] = {"available": False}

    try:
        import ray

        status["ray"] = {"available": True, "version": ray.__version__}
    except ImportError:
        status["ray"] = {"available": False}

    return status
