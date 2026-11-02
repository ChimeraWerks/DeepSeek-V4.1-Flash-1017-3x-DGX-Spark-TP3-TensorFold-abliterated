"""Keep deployment regression tests offline even if a transport mock is missed."""
from pathlib import Path
import socket
import subprocess
import urllib.request

import pytest


@pytest.fixture(autouse=True)
def offline_transports(monkeypatch):
    run = subprocess.run

    def local_only(argv, *args, **kwargs):
        if isinstance(argv, (list, tuple)) and Path(str(argv[0])).name in {"ssh", "rsync", "curl", "systemctl"}:
            pytest.fail("unmocked remote/service command in an offline deployment test")
        return run(argv, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("unmocked network operation in an offline deployment test")

    monkeypatch.setattr(subprocess, "run", local_only)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
