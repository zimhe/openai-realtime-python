"""Exchange a verified Unity player's identity for a Streamoji auth token.

The Streamoji Firebase endpoint and client secret are contacted only by this
server. The Unity client receives the token but never receives the secret.
"""

import asyncio
import logging
import os
import re
from urllib.parse import quote

from aiohttp import ClientError, ClientSession, ClientTimeout, web


_LOG = logging.getLogger(__name__)
_SESSION_KEY = web.AppKey("streamoji_auth_session", ClientSession)
_MAX_BODY_BYTES = 4096
_UNITY_AUTH_URL = "https://player-auth.services.api.unity.com"
_STREAMOJI_AUTH_URL = "https://us-central1-streamoji-265f4.cloudfunctions.net/getAuthToken"
_STREAMOJI_FUNCTION_BASE = "https://us-central1-streamoji-265f4.cloudfunctions.net"
_AVATAR_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


async def _open_session(app: web.Application) -> None:
    app[_SESSION_KEY] = ClientSession(timeout=ClientTimeout(total=25))


async def _close_session(app: web.Application) -> None:
    await app[_SESSION_KEY].close()


def _bearer(request: web.Request) -> str:
    authorization = request.headers.get("Authorization", "")
    if not authorization.startswith("Bearer "):
        raise web.HTTPUnauthorized(text="Unity Authentication is required")
    token = authorization[len("Bearer "):].strip()
    if not token or any(char.isspace() for char in token):
        raise web.HTTPUnauthorized(text="Invalid Unity Authentication token")
    return token


async def _verify_unity_player(session: ClientSession, project_id: str,
                               player_id: str, access_token: str) -> None:
    url = f"{_UNITY_AUTH_URL}/v1/users/{quote(player_id, safe='')}"
    try:
        async with session.get(
            url,
            headers={"Authorization": f"Bearer {access_token}", "ProjectId": project_id},
            allow_redirects=False,
        ) as response:
            if response.status in (401, 403, 404):
                raise web.HTTPUnauthorized(text="Unity player could not be verified")
            if response.status != 200:
                raise web.HTTPBadGateway(text="Unity Authentication is unavailable")
            payload = await response.json(content_type=None)
    except (ClientError, asyncio.TimeoutError, ValueError) as exc:
        _LOG.warning("Unity Authentication verification request failed: %s", exc)
        raise web.HTTPBadGateway(text="Unity Authentication is unavailable") from exc
    if not isinstance(payload, dict) or payload.get("id") != player_id or payload.get("disabled"):
        raise web.HTTPUnauthorized(text="Unity player does not match requested user")


async def get_streamoji_auth_token(request: web.Request) -> web.Response:
    project_id = os.getenv("UNITY_PROJECT_ID", "").strip()
    client_id = os.getenv("STREAMOJI_CLIENT_ID", "").strip()
    client_secret = os.getenv("STREAMOJI_CLIENT_SECRET", "").strip()
    if not project_id or not client_id or not client_secret:
        raise web.HTTPServiceUnavailable(text="Streamoji authentication is not configured")
    access_token = _bearer(request)
    if request.content_length is not None and request.content_length > _MAX_BODY_BYTES:
        raise web.HTTPRequestEntityTooLarge(max_size=_MAX_BODY_BYTES, actual_size=request.content_length)
    body = await request.read()
    if len(body) > _MAX_BODY_BYTES:
        raise web.HTTPRequestEntityTooLarge(max_size=_MAX_BODY_BYTES, actual_size=len(body))
    try:
        payload = await request.json()
    except (ValueError, UnicodeDecodeError):
        raise web.HTTPBadRequest(text="Expected JSON body")
    if not isinstance(payload, dict):
        raise web.HTTPBadRequest(text="Expected JSON object")
    player_id = payload.get("userId")
    user_name = payload.get("userName")
    if not isinstance(player_id, str) or not player_id or len(player_id) > 128 or \
       not isinstance(user_name, str) or not user_name.strip() or len(user_name) > 128:
        raise web.HTTPBadRequest(text="Invalid Streamoji user identity")

    session = request.app[_SESSION_KEY]
    await _verify_unity_player(session, project_id, player_id, access_token)
    try:
        async with session.post(
            _STREAMOJI_AUTH_URL,
            json={"userId": player_id, "userName": user_name.strip()},
            headers={
                "Client-Id": client_id,
                "Client-Secret": client_secret,
            },
            allow_redirects=False,
        ) as response:
            if response.status != 200:
                _LOG.warning("Streamoji auth request returned HTTP %s", response.status)
                raise web.HTTPBadGateway(text=f"Streamoji authentication failed ({response.status})")
            result = await response.json(content_type=None)
    except (ClientError, asyncio.TimeoutError, ValueError) as exc:
        _LOG.warning("Streamoji auth request failed: %s", exc)
        raise web.HTTPBadGateway(text="Streamoji authentication is unavailable") from exc
    if not isinstance(result, dict) or result.get("success") is not True or \
       not isinstance(result.get("authToken"), str) or not result["authToken"]:
        raise web.HTTPBadGateway(text="Streamoji returned an invalid auth response")
    return web.json_response(
        {"success": True, "authToken": result["authToken"]},
        headers={"Cache-Control": "no-store"},
    )


async def proxy_streamoji_avatar(request: web.Request) -> web.Response:
    """Relay only the three fixed avatar Cloud Functions, after Unity identity verification."""
    project_id = os.getenv("UNITY_PROJECT_ID", "").strip()
    if not project_id:
        raise web.HTTPServiceUnavailable(text="Unity Authentication is not configured")
    unity_token = _bearer(request)
    player_id = request.headers.get("X-Unity-Player-Id", "").strip()
    streamoji_token = request.headers.get("X-Streamoji-Token", "").strip()
    if not player_id or len(player_id) > 128 or not streamoji_token or len(streamoji_token) > 4096:
        raise web.HTTPBadRequest(text="Missing avatar relay identity or token")
    session = request.app[_SESSION_KEY]
    await _verify_unity_player(session, project_id, player_id, unity_token)

    action = request.path.rsplit("/", 1)[-1]
    routes = {
        "config": ("GET", "getAvatarConfig"),
        "saved": ("GET", "getSavedAvatarsForUser"),
        "save": ("POST", "saveAvatarConfig"),
    }
    if action not in routes:
        raise web.HTTPNotFound()
    method, function = routes[action]
    if request.method != method:
        raise web.HTTPMethodNotAllowed(request.method, [method])
    avatar_id = None
    if action == "config":
        avatar_id = request.query.get("avatarId", "")
        if not _AVATAR_ID.fullmatch(avatar_id):
            raise web.HTTPBadRequest(text="Invalid avatar ID")
    body = None
    if action == "save":
        if request.content_length is not None and request.content_length > 1_000_000:
            raise web.HTTPRequestEntityTooLarge(max_size=1_000_000, actual_size=request.content_length)
        body = await request.read()
        if len(body) > 1_000_000:
            raise web.HTTPRequestEntityTooLarge(max_size=1_000_000, actual_size=len(body))
        try:
            import json
            if not isinstance(json.loads(body), dict):
                raise ValueError("Expected JSON object")
        except (ValueError, UnicodeDecodeError):
            raise web.HTTPBadRequest(text="Invalid avatar JSON")

    url = f"{_STREAMOJI_FUNCTION_BASE}/{function}"
    headers = {"Authorization": f"Bearer {streamoji_token}",
               "Origin": "https://avatars.streamoji.com", "Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    try:
        async with session.request(method, url, params={"avatarId": avatar_id} if avatar_id else None,
                                   data=body, headers=headers, allow_redirects=False) as response:
            result = await response.read()
            if response.status >= 400:
                _LOG.warning("Streamoji avatar upstream returned HTTP %s for %s (avatar_id=%s)",
                             response.status, action, avatar_id or "n/a")
            content_type = response.headers.get("Content-Type", "application/json")
            return web.Response(body=result, status=response.status,
                                headers={"Cache-Control": "no-store", "Content-Type": content_type})
    except (ClientError, asyncio.TimeoutError) as exc:
        _LOG.warning("Streamoji avatar relay %s failed: %s", action, exc)
        raise web.HTTPBadGateway(text="Streamoji avatar service is unavailable") from exc


def install_streamoji_auth(app: web.Application) -> None:
    app.on_startup.append(_open_session)
    app.on_cleanup.append(_close_session)
    app.router.add_post("/streamoji/auth-token", get_streamoji_auth_token)
    app.router.add_get("/streamoji/avatar/config", proxy_streamoji_avatar)
    app.router.add_get("/streamoji/avatar/saved", proxy_streamoji_avatar)
    app.router.add_post("/streamoji/avatar/save", proxy_streamoji_avatar)
