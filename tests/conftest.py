"""The entire test suite is offline, including integration tests."""
import socket
import pytest

@pytest.fixture(autouse=True)
def prohibit_real_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Tests must not access the network; inject an HTTP mock.")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
