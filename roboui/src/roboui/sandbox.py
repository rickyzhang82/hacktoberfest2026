"""Network isolation: a process-level socket guard plus a locked-down preview iframe."""
from __future__ import annotations

import contextlib
import html
import logging
import re
import socket
from collections.abc import Iterator

from .errors import NetworkBlockedError

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def network_guard() -> Iterator[None]:
    """Deny all outbound network access from this process while active.

    Patches connect / DNS entry points so any attempt to reach the network (from
    the model runtime or any code it triggers) raises NetworkBlockedError. Unix
    domain sockets are still allowed. This is a best-effort in-process guard, not
    a kernel boundary; for hard isolation also run under an OS sandbox
    (e.g. `unshare -n`, a no-network container, or macOS `sandbox-exec`).
    """
    orig_connect = socket.socket.connect
    orig_connect_ex = socket.socket.connect_ex
    orig_create = socket.create_connection
    orig_getaddrinfo = socket.getaddrinfo

    def _deny(*_a, **_k):
        raise NetworkBlockedError("Outbound network access is disabled during local inference.")

    def connect(self, address):
        if self.family == getattr(socket, "AF_UNIX", None):
            return orig_connect(self, address)
        _deny()

    def connect_ex(self, address):
        if self.family == getattr(socket, "AF_UNIX", None):
            return orig_connect_ex(self, address)
        _deny()

    socket.socket.connect = connect  # type: ignore[method-assign]
    socket.socket.connect_ex = connect_ex  # type: ignore[method-assign]
    socket.create_connection = _deny  # type: ignore[assignment]
    socket.getaddrinfo = _deny  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket.connect = orig_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = orig_connect_ex  # type: ignore[method-assign]
        socket.create_connection = orig_create  # type: ignore[assignment]
        socket.getaddrinfo = orig_getaddrinfo  # type: ignore[assignment]


# Generated pages may load ONLY the Tailwind CDN script; every runtime request
# (fetch/XHR/WebSocket/forms/images/frames) is denied by policy.
CSP = (
    "default-src 'none'; "
    "script-src 'unsafe-inline' https://cdn.tailwindcss.com; "
    "style-src 'unsafe-inline'; "
    "img-src data: blob:; font-src data:; media-src data: blob:; "
    "connect-src 'none'; form-action 'none'; base-uri 'none'; frame-src 'none'"
)
_CSP_META = f'<meta http-equiv="Content-Security-Policy" content="{CSP}">'
_HEAD_OPEN = re.compile(r"<head[^>]*>", re.IGNORECASE)
_HTML_OPEN = re.compile(r"<html[^>]*>", re.IGNORECASE)


def apply_csp(page: str) -> str:
    """Inject the CSP meta tag as early as possible in the document."""
    if m := _HEAD_OPEN.search(page):
        return page[: m.end()] + _CSP_META + page[m.end():]
    if m := _HTML_OPEN.search(page):
        return page[: m.end()] + f"<head>{_CSP_META}</head>" + page[m.end():]
    return f"<!DOCTYPE html><html><head>{_CSP_META}</head><body>{page}</body></html>"


def sandboxed_iframe(page: str, height: int = 700) -> str:
    """Return an <iframe srcdoc> with scripts allowed but no same-origin, forms,
    popups, or top navigation, and with a network-denying CSP inside."""
    srcdoc = html.escape(apply_csp(page), quote=True)
    return (
        f'<iframe sandbox="allow-scripts" referrerpolicy="no-referrer" srcdoc="{srcdoc}" '
        f'style="width:100%;height:{height}px;border:1px solid #ddd;border-radius:8px;background:#fff"></iframe>'
    )
