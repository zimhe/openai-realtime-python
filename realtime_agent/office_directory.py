"""Short-lived registry for discoverable Unity office lobby servers.

The Unity dedicated server is authoritative for its own heartbeat.  This
module never proxies gameplay traffic; clients connect to the advertised UDP
endpoint directly.
"""

import hmac
import ipaddress
import os
import re
import time
from urllib.parse import urlparse

from aiohttp import web


_LOBBY_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
_HOSTNAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9.-]{0,252}$")
_TTL_SECONDS = 75
_MAX_BODY_BYTES = 8192
REGISTRY_KEY = web.AppKey("office_registry", dict)


def _string(value, field, limit=128, required=True):
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    value = value.strip()
    if required and not value:
        raise ValueError(f"{field} is required")
    if len(value) > limit:
        raise ValueError(f"{field} is too long")
    return value


def _validate_host(value):
    host = _string(value, "address", 253)
    if host in {"0.0.0.0", "::", "localhost"}:
        raise ValueError("address must be a reachable advertised endpoint")
    try:
        parsed = ipaddress.ip_address(host)
        if parsed.is_unspecified or parsed.is_loopback or parsed.is_multicast:
            raise ValueError("address must be a reachable advertised endpoint")
    except ValueError as exc:
        if "reachable advertised" in str(exc):
            raise
        if not _HOSTNAME.fullmatch(host) or ".." in host:
            raise ValueError("address must be an IP address or hostname") from exc
    return host


def _validate_office(data):
    if not isinstance(data, dict):
        raise ValueError("JSON object required")
    lobby_id = _string(data.get("lobbyId"), "lobbyId", 64)
    if not _LOBBY_ID.fullmatch(lobby_id):
        raise ValueError("invalid lobbyId")
    instance_id = _string(data.get("instanceId"), "instanceId", 80)
    port = data.get("port")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    max_players = data.get("maxPlayers", 40)
    current_players = data.get("currentPlayers", 0)
    if (isinstance(max_players, bool) or not isinstance(max_players, int)
            or not 1 <= max_players <= 512):
        raise ValueError("invalid maxPlayers")
    if (isinstance(current_players, bool) or not isinstance(current_players, int)
            or not 0 <= current_players <= max_players):
        raise ValueError("invalid currentPlayers")
    scope = _string(data.get("scope", "public"), "scope", 16).lower()
    if scope not in {"local", "public"}:
        raise ValueError("scope must be local or public")
    preview = _string(data.get("previewImageUrl", ""), "previewImageUrl", 512, False)
    if preview:
        parsed = urlparse(preview)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("previewImageUrl must be a public HTTPS URL")
    return {
        "lobbyId": lobby_id,
        "instanceId": instance_id,
        "displayName": _string(data.get("displayName"), "displayName", 100),
        "address": _validate_host(data.get("address")),
        "port": port,
        "scope": scope,
        "sceneName": _string(data.get("sceneName", ""), "sceneName", 100, False),
        "buildVersion": _string(data.get("buildVersion", ""), "buildVersion", 64, False),
        "previewImageUrl": preview,
        "maxPlayers": max_players,
        "currentPlayers": current_players,
    }


def _authorize(request):
    expected = os.environ.get("OFFICE_DIRECTORY_KEY", "")
    if not expected:
        raise web.HTTPServiceUnavailable(text="Office directory registration is disabled")
    provided = request.headers.get("Authorization", "")
    if not hmac.compare_digest(provided, "Bearer " + expected):
        raise web.HTTPUnauthorized(text="Invalid directory credential")


def _live_entries(registry):
    now = time.monotonic()
    for instance_id, record in list(registry.items()):
        if record["expiresAt"] <= now:
            del registry[instance_id]
    return registry


async def list_offices(request):
    registry = _live_entries(request.app[REGISTRY_KEY])
    offices = [record["office"] for record in registry.values()]
    offices.sort(key=lambda office: (office["displayName"].lower(), office["lobbyId"]))
    return web.json_response({"offices": offices}, headers={"Cache-Control": "no-store"})


async def heartbeat_office(request):
    _authorize(request)
    if request.content_length is not None and request.content_length > _MAX_BODY_BYTES:
        raise web.HTTPRequestEntityTooLarge(max_size=_MAX_BODY_BYTES, actual_size=request.content_length)
    try:
        office = _validate_office(await request.json())
    except (ValueError, TypeError) as exc:
        return web.json_response({"error": str(exc)}, status=400)
    registry = _live_entries(request.app[REGISTRY_KEY])
    registry[office["instanceId"]] = {
        "office": office,
        "expiresAt": time.monotonic() + _TTL_SECONDS,
    }
    return web.json_response({"status": "registered", "ttlSeconds": _TTL_SECONDS})


async def remove_office(request):
    _authorize(request)
    instance_id = request.match_info["instance_id"]
    registry = _live_entries(request.app[REGISTRY_KEY])
    registry.pop(instance_id, None)
    return web.json_response({"status": "removed"})


def install_office_directory(app):
    app[REGISTRY_KEY] = {}
    app.router.add_get("/offices", list_offices)
    app.router.add_post("/offices/heartbeat", heartbeat_office)
    app.router.add_delete("/offices/{instance_id}", remove_office)
