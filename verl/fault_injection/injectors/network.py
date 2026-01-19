"""Network fault injectors."""

import time
import subprocess
import socket
from typing import Optional

from ..base import BaseFaultInjector, FaultContext, FaultResult, FaultStatus
from ..config import NetworkFaultConfig, FaultType
from ..base import FaultInjectorRegistry


@FaultInjectorRegistry.register(FaultType.NETWORK_DELAY)
class NetworkDelayInjector(BaseFaultInjector):
    """Injector for network delays."""

    def __init__(self, config: NetworkFaultConfig):
        super().__init__(config)
        self.config = config
        self._original_tc_rules = []

    def inject(self, context: FaultContext) -> FaultResult:
        """Inject network delay using tc (traffic control)."""
        logger = self._get_logger()
        delay_ms = self.config.delay_ms

        logger.info(f"Injecting network delay: {delay_ms}ms")

        try:
            # Check if tc is available
            result = subprocess.run(['which', 'tc'], capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception("tc command not found. Network faults require tc (traffic control)")

            # Get network interface
            interface = self._get_network_interface()
            if not interface:
                raise Exception("Could not determine network interface")

            # Save current rules
            self._save_tc_rules(interface)

            # Add delay rule
            cmd = [
                'tc', 'qdisc', 'add', 'dev', interface, 'root', 'netem',
                'delay', f'{delay_ms}ms'
            ]

            if self.config.loss_rate > 0:
                cmd.extend(['loss', f'{self.config.loss_rate * 100}%'])

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"Failed to add tc rule: {result.stderr}")

            logger.info(f"Network delay injected on {interface}: {delay_ms}ms")

            return FaultResult(
                fault_id=self.fault_id,
                status=FaultStatus.COMPLETED,
                start_time=time.time(),
                end_time=time.time(),
                metadata={
                    "interface": interface,
                    "delay_ms": delay_ms,
                    "loss_rate": self.config.loss_rate
                }
            )

        except Exception as e:
            logger.error(f"Failed to inject network delay: {e}")
            return FaultResult(
                fault_id=self.fault_id,
                status=FaultStatus.FAILED,
                start_time=time.time(),
                end_time=time.time(),
                error=e,
                metadata={"delay_ms": delay_ms}
            )

    def recover(self, context: FaultContext) -> None:
        """Remove network delay."""
        logger = self._get_logger()

        try:
            interface = self._get_network_interface()
            if interface and self._original_tc_rules:
                # Remove our added rules
                subprocess.run(['tc', 'qdisc', 'del', 'dev', interface, 'root'],
                               capture_output=True)

                # Restore original rules
                self._restore_tc_rules(interface)

                logger.info(f"Network delay removed from {interface}")

        except Exception as e:
            logger.error(f"Failed to recover network delay: {e}")

    def _get_network_interface(self) -> Optional[str]:
        """Get the primary network interface."""
        try:
            # Try to get the default route interface
            result = subprocess.run(['ip', 'route', 'show', 'default'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                parts = result.stdout.split()
                if 'dev' in parts:
                    idx = parts.index('dev')
                    if idx + 1 < len(parts):
                        return parts[idx + 1]

            # Fallback to common interface names
            for iface in ['eth0', 'ens3', 'enp0s3']:
                if self._interface_exists(iface):
                    return iface

        except Exception:
            pass

        return None

    def _interface_exists(self, interface: str) -> bool:
        """Check if network interface exists."""
        try:
            with open(f'/sys/class/net/{interface}/operstate', 'r') as f:
                return True
        except:
            return False

    def _save_tc_rules(self, interface: str) -> None:
        """Save current tc rules."""
        try:
            result = subprocess.run(['tc', 'qdisc', 'show', 'dev', interface],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                self._original_tc_rules = result.stdout.strip().split('\n')
        except:
            pass

    def _restore_tc_rules(self, interface: str) -> None:
        """Restore original tc rules."""
        # For simplicity, we just clear and let system use defaults
        # In a real implementation, you'd restore the exact original rules
        pass

    def _get_logger(self):
        """Get logger instance."""
        import logging
        return logging.getLogger(f"{__name__}.{self.__class__.__name__}")


@FaultInjectorRegistry.register(FaultType.NETWORK_PARTITION)
class NetworkPartitionInjector(BaseFaultInjector):
    """Injector for network partitions."""

    def __init__(self, config: NetworkFaultConfig):
        super().__init__(config)
        self.config = config
        self._blocked_ports = set()

    def inject(self, context: FaultContext) -> FaultResult:
        """Inject network partition by blocking ports."""
        logger = self._get_logger()
        target_ports = self.config.target_ports or []

        logger.info(f"Injecting network partition for ports: {target_ports}")

        try:
            # Use iptables to block ports
            for port in target_ports:
                self._block_port(port)
                self._blocked_ports.add(port)

            logger.info(f"Network partition injected for {len(target_ports)} ports")

            return FaultResult(
                fault_id=self.fault_id,
                status=FaultStatus.COMPLETED,
                start_time=time.time(),
                end_time=time.time(),
                metadata={"blocked_ports": list(self._blocked_ports)}
            )

        except Exception as e:
            logger.error(f"Failed to inject network partition: {e}")
            return FaultResult(
                fault_id=self.fault_id,
                status=FaultStatus.FAILED,
                start_time=time.time(),
                end_time=time.time(),
                error=e,
                metadata={"target_ports": target_ports}
            )

    def recover(self, context: FaultContext) -> None:
        """Remove network partition."""
        logger = self._get_logger()

        try:
            # Unblock all ports
            for port in self._blocked_ports:
                self._unblock_port(port)

            logger.info(f"Network partition removed for {len(self._blocked_ports)} ports")
            self._blocked_ports.clear()

        except Exception as e:
            logger.error(f"Failed to recover network partition: {e}")

    def _block_port(self, port: int) -> None:
        """Block a port using iptables."""
        # Block incoming
        subprocess.run([
            'iptables', '-A', 'INPUT', '-p', 'tcp', '--dport', str(port),
            '-j', 'DROP'
        ], check=True)

        # Block outgoing
        subprocess.run([
            'iptables', '-A', 'OUTPUT', '-p', 'tcp', '--sport', str(port),
            '-j', 'DROP'
        ], check=True)

    def _unblock_port(self, port: int) -> None:
        """Unblock a port."""
        # Remove incoming block
        subprocess.run([
            'iptables', '-D', 'INPUT', '-p', 'tcp', '--dport', str(port),
            '-j', 'DROP'
        ], capture_output=True)

        # Remove outgoing block
        subprocess.run([
            'iptables', '-D', 'OUTPUT', '-p', 'tcp', '--sport', str(port),
            '-j', 'DROP'
        ], capture_output=True)

    def _get_logger(self):
        """Get logger instance."""
        import logging
        return logging.getLogger(f"{__name__}.{self.__class__.__name__}")