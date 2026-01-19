"""Network fault executor using tc and iptables."""

import logging
import subprocess

from ..config import FaultSpec, FaultType
from .base import BaseExecutor

logger = logging.getLogger(__name__)


class NetworkExecutor(BaseExecutor):
    """Execute network faults using tc netem and iptables."""

    def __init__(self):
        self._cleanup_commands = []

    def execute(self, spec: FaultSpec) -> bool:
        params = spec.params
        try:
            if spec.type == FaultType.NETWORK_DELAY:
                return self._add_delay(params.get("delay_ms", 100), params.get("interface", "eth0"))
            elif spec.type == FaultType.NETWORK_LOSS:
                return self._add_loss(params.get("loss_percent", 10), params.get("interface", "eth0"))
            elif spec.type == FaultType.PORT_BLOCK:
                return self._block_port(params.get("port", 29500))
            elif spec.type == FaultType.NETWORK_PARTITION:
                return self._partition(params.get("target_ip"))
        except Exception as e:
            logger.error(f"Network fault failed: {e}")
            return False
        return False

    def _add_delay(self, delay_ms: int, interface: str) -> bool:
        cmd = f"tc qdisc add dev {interface} root netem delay {delay_ms}ms"
        self._cleanup_commands.append(f"tc qdisc del dev {interface} root")
        return self._run(cmd)

    def _add_loss(self, loss_percent: int, interface: str) -> bool:
        cmd = f"tc qdisc add dev {interface} root netem loss {loss_percent}%"
        self._cleanup_commands.append(f"tc qdisc del dev {interface} root")
        return self._run(cmd)

    def _block_port(self, port: int) -> bool:
        cmd = f"iptables -A INPUT -p tcp --dport {port} -j DROP"
        self._cleanup_commands.append(f"iptables -D INPUT -p tcp --dport {port} -j DROP")
        return self._run(cmd)

    def _partition(self, target_ip: str) -> bool:
        if not target_ip:
            return False
        cmd = f"iptables -A INPUT -s {target_ip} -j DROP && iptables -A OUTPUT -d {target_ip} -j DROP"
        self._cleanup_commands.extend(
            [
                f"iptables -D INPUT -s {target_ip} -j DROP",
                f"iptables -D OUTPUT -d {target_ip} -j DROP",
            ]
        )
        return self._run(cmd)

    def _run(self, cmd: str) -> bool:
        logger.info(f"Executing: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True)
        if result.returncode != 0:
            logger.warning(f"Command failed: {result.stderr.decode()}")
        return result.returncode == 0

    def cleanup(self) -> None:
        for cmd in reversed(self._cleanup_commands):
            subprocess.run(cmd, shell=True, capture_output=True)
        self._cleanup_commands.clear()
