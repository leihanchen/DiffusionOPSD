"""Select NCCL's IPv4 socket network from the OS route to the rendezvous host."""

from __future__ import annotations

import fcntl
import os
import socket
import struct
from collections.abc import MutableMapping


def route_source(master: str) -> tuple[str, str]:
    """Resolve the master and ask the kernel for a source address; send no packets."""
    address = socket.gethostbyname(master)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect((address, 9))
        return address, probe.getsockname()[0]


def ipv4_interfaces() -> dict[str, str]:
    """Read Linux interface addresses without depending on an ip executable."""
    result = {}
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        for _, name in socket.if_nameindex():
            try:
                request = struct.pack("256s", name.encode()[:15])
                response = fcntl.ioctl(probe.fileno(), 0x8915, request)  # SIOCGIFADDR
                result[name] = socket.inet_ntoa(response[20:24])
            except OSError:
                continue  # Interface has no IPv4 address.
    return result


def configure_socket_interface(master: str, environment: MutableMapping[str, str] | None = None) -> dict:
    """Preserve an explicit override, otherwise bind sockets to the rendezvous network.

    Route selection does not prove peer connectivity; validate with a GPU collective
    probe. This selects socket/bootstrap transport, not an RDMA HCA or its libraries.
    """
    environment = os.environ if environment is None else environment
    if environment.get("NCCL_SOCKET_IFNAME"):
        return {"source": "explicit", "NCCL_SOCKET_IFNAME": environment["NCCL_SOCKET_IFNAME"]}
    master_ip, local_ip = route_source(master)
    matches = [name for name, address in ipv4_interfaces().items() if address == local_ip]
    if not matches:
        raise RuntimeError(f"No interface matches route source {local_ip} for master {master} ({master_ip})")
    if len(matches) != 1 or local_ip.startswith("127."):
        raise RuntimeError(
            f"Ambiguous or loopback master route: {matches}, {local_ip}; set NCCL_SOCKET_IFNAME explicitly"
        )
    environment["NCCL_SOCKET_IFNAME"] = "=" + matches[0]
    return {
        "source": "route",
        "master_ip": master_ip,
        "local_ip": local_ip,
        "NCCL_SOCKET_IFNAME": environment["NCCL_SOCKET_IFNAME"],
    }
