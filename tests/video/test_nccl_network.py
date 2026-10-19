import pytest

from diffusionopsd.video import nccl_network


def test_route_selects_interface_without_cluster_names(monkeypatch):
    monkeypatch.setattr(nccl_network, "route_source", lambda master: ("10.1.1.172", "10.1.1.173"))
    monkeypatch.setattr(nccl_network, "ipv4_interfaces", lambda: {"ib0": "10.0.1.173", "eno7": "10.1.1.173"})
    environment = {}
    result = nccl_network.configure_socket_interface("kn172", environment)
    assert environment["NCCL_SOCKET_IFNAME"] == "=eno7"
    assert result["source"] == "route"


def test_override_is_preserved(monkeypatch):
    monkeypatch.setattr(nccl_network, "route_source", lambda master: pytest.fail("override must avoid discovery"))
    environment = {"NCCL_SOCKET_IFNAME": "=ib2"}
    result = nccl_network.configure_socket_interface("master", environment)
    assert environment["NCCL_SOCKET_IFNAME"] == "=ib2"
    assert result["source"] == "explicit"


def test_missing_interface_fails_before_nccl(monkeypatch):
    monkeypatch.setattr(nccl_network, "route_source", lambda master: ("10.1.1.172", "10.1.1.173"))
    monkeypatch.setattr(nccl_network, "ipv4_interfaces", lambda: {"ib0": "10.0.1.173"})
    with pytest.raises(RuntimeError, match="No interface"):
        nccl_network.configure_socket_interface("master", {})
