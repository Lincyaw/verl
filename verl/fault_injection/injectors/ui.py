# Copyright 2026 Individual Contributor: Aoyang Fang
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


"""UI layer fault injectors for verl fault injection system."""

import os
import time

from verl.fault_injection.base import BaseFaultInjector, FaultContext, FaultInjectorRegistry, FaultResult, FaultStatus
from verl.fault_injection.config import FaultType, UIFaultConfig


@FaultInjectorRegistry.register(FaultType.HYDRA_CONFIG_ERROR)
class HydraConfigErrorInjector(BaseFaultInjector):
    """Injector for Hydra configuration errors."""

    def __init__(self, config: UIFaultConfig):
        super().__init__(config)
        self._original_config = None

    def inject(self, context: FaultContext) -> FaultResult:
        """Inject Hydra configuration error."""
        error_type = self.config.hydra_error_type or "parse"

        if error_type == "parse":
            # Simulate parsing error by corrupting config structure
            from omegaconf import OmegaConf

            # Create invalid YAML content
            invalid_yaml = """
            invalid_config: {
                missing_closing_brace: "value"
                invalid_syntax: [1, 2, 3,]
            """

            try:
                # This will raise a parsing error
                OmegaConf.create(invalid_yaml)
            except Exception as e:
                return FaultResult(
                    fault_id=self.fault_id,
                    status=FaultStatus.FAILED,
                    start_time=time.time(),
                    error=Exception(f"Hydra config parse error: {str(e)}"),
                    metadata={"error_type": error_type},
                )

        elif error_type == "validate":
            # Simulate validation error
            return FaultResult(
                fault_id=self.fault_id,
                status=FaultStatus.FAILED,
                start_time=time.time(),
                error=Exception("Hydra config validation failed: missing required field 'model.path'"),
                metadata={"error_type": error_type},
            )

        elif error_type == "merge":
            # Simulate merge conflict
            return FaultResult(
                fault_id=self.fault_id,
                status=FaultStatus.FAILED,
                start_time=time.time(),
                error=Exception("Hydra config merge error: conflicting values for key 'learning_rate'"),
                metadata={"error_type": error_type},
            )

        return FaultResult(
            fault_id=self.fault_id,
            status=FaultStatus.SUCCESS,
            start_time=time.time(),
            metadata={"error_type": error_type},
        )

    def recover(self, context: FaultContext) -> None:
        """No recovery needed for config errors."""
        pass


@FaultInjectorRegistry.register(FaultType.RAY_INIT_FAILURE)
class RayInitFailureInjector(BaseFaultInjector):
    """Injector for Ray initialization failures."""

    def __init__(self, config: UIFaultConfig):
        self.config = config
        self._original_ray_init = None

    def inject(self, context: FaultContext) -> FaultResult:
        """Inject Ray initialization failure."""
        import ray

        error_msg = self.config.ray_init_error or "Failed to initialize Ray cluster"

        # Monkey patch ray.init to simulate failure
        self._original_ray_init = ray.init

        def failing_ray_init(*args, **kwargs):
            raise RuntimeError(error_msg)

        ray.init = failing_ray_init

        return FaultResult(status=FaultStatus.SUCCESS, metadata={"error_message": error_msg})

    def recover(self, context: FaultContext) -> None:
        """Restore original ray.init function."""
        if self._original_ray_init:
            import ray

            ray.init = self._original_ray_init


@FaultInjectorRegistry.register(FaultType.CLI_ARG_ERROR)
class CLIArgErrorInjector(BaseFaultInjector):
    """Injector for CLI argument errors."""

    def __init__(self, config: UIFaultConfig):
        self.config = config
        self._original_sys_argv = None

    def inject(self, context: FaultContext) -> FaultResult:
        """Inject CLI argument error."""
        import sys

        self._original_sys_argv = sys.argv.copy()

        if self.config.cli_arg_missing:
            # Remove a required argument
            arg_to_remove = self.config.cli_arg_missing
            if arg_to_remove in sys.argv:
                sys.argv.remove(arg_to_remove)

            return FaultResult(status=FaultStatus.SUCCESS, metadata={"action": "removed_arg", "arg": arg_to_remove})

        elif self.config.cli_arg_invalid:
            # Add invalid argument
            invalid_arg = self.config.cli_arg_invalid
            sys.argv.append(invalid_arg)

            return FaultResult(status=FaultStatus.SUCCESS, metadata={"action": "added_invalid_arg", "arg": invalid_arg})

        return FaultResult(status=FaultStatus.FAILED, error_message="No CLI argument error configured")

    def recover(self, context: FaultContext) -> None:
        """Restore original sys.argv."""
        if self._original_sys_argv:
            import sys

            sys.argv = self._original_sys_argv


@FaultInjectorRegistry.register(FaultType.ENV_VAR_ERROR)
class EnvVarErrorInjector(BaseFaultInjector):
    """Injector for environment variable errors."""

    def __init__(self, config: UIFaultConfig):
        self.config = config
        self._original_env = {}

    def inject(self, context: FaultContext) -> FaultResult:
        """Inject environment variable error."""
        metadata = {}

        if self.config.env_var_unset:
            # Unset a required environment variable
            var_name = self.config.env_var_unset
            if var_name in os.environ:
                self._original_env[var_name] = os.environ[var_name]
                del os.environ[var_name]
                metadata["unset_var"] = var_name

        elif self.config.env_var_invalid:
            # Set invalid value for environment variable
            var_name = self.config.env_var_invalid
            if var_name in os.environ:
                self._original_env[var_name] = os.environ[var_name]

            invalid_value = self.config.env_var_value or "INVALID_VALUE"
            os.environ[var_name] = invalid_value
            metadata["set_invalid_var"] = f"{var_name}={invalid_value}"

        if metadata:
            return FaultResult(status=FaultStatus.SUCCESS, metadata=metadata)

        return FaultResult(status=FaultStatus.FAILED, error_message="No environment variable error configured")

    def recover(self, context: FaultContext) -> None:
        """Restore original environment variables."""
        for var_name, original_value in self._original_env.items():
            os.environ[var_name] = original_value

        self._original_env.clear()


@FaultInjectorRegistry.register(FaultType.UI_FREEZE)
class UIFreezeInjector(BaseFaultInjector):
    """Injector for UI freeze simulation."""

    def __init__(self, config: UIFaultConfig):
        self.config = config

    def inject(self, context: FaultContext) -> FaultResult:
        """Freeze the UI for specified duration."""
        duration = self.config.freeze_duration_seconds or 30.0

        # Log the freeze
        print(f"[UI Freeze] Freezing UI for {duration} seconds...")

        # Sleep to simulate freeze
        time.sleep(duration)

        print(f"[UI Freeze] UI unfrozen after {duration} seconds")

        return FaultResult(status=FaultStatus.SUCCESS, metadata={"freeze_duration": duration})

    def recover(self, context: FaultContext) -> None:
        """No recovery needed for freeze."""
        pass


@FaultInjectorRegistry.register(FaultType.UI_CRASH)
class UICrashInjector(BaseFaultInjector):
    """Injector for UI crash simulation."""

    def __init__(self, config: UIFaultConfig):
        self.config = config

    def inject(self, context: FaultContext) -> FaultResult:
        """Crash the UI process."""
        crash_msg = self.config.crash_message or "Simulated UI crash"

        # Log the crash
        print(f"[UI Crash] {crash_msg}")

        # Exit the process
        import sys

        sys.exit(1)

    def recover(self, context: FaultContext) -> None:
        """Cannot recover from crash."""
        pass
