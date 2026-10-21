"""Expose the bundled Agora token service through the existing HTTP server.

The Go service keeps the established /getToken behavior, including Unity
Authentication checks and Agora Chat user registration. It listens only on
loopback; aiohttp is the public entry point.
"""

import asyncio
import logging
import os
from pathlib import Path

from aiohttp import ClientError, ClientSession, ClientTimeout, web


_LOG = logging.getLogger(__name__)
_PROCESS_KEY = web.AppKey("agora_token_process", object)
_SESSION_KEY = web.AppKey("agora_token_session", ClientSession)
_PORT_KEY = web.AppKey("agora_token_port", int)
_MAX_REQUEST_BYTES = 8192


async def _start_token_service(app):
    # First Chat login may validate Unity identity and create an Agora user.
    app[_SESSION_KEY] = ClientSession(timeout=ClientTimeout(total=45))
    app[_PROCESS_KEY] = None

    app_id = os.environ.get("AGORA_APP_ID", "")
    app_cert = os.environ.get("AGORA_APP_CERT", "")
    if not app_id or not app_cert:
        _LOG.error("Token service disabled: AGORA_APP_ID and AGORA_APP_CERT are required")
        return

    binary = Path(os.environ.get("AGORA_TOKEN_BINARY", "/app/bin/agora-token-service"))
    if not binary.is_file():
        _LOG.error("Token service binary is missing: %s", binary)
        return

    try:
        port = int(os.environ.get("AGORA_TOKEN_INTERNAL_PORT", "18081"))
        if not 1 <= port <= 65535:
            raise ValueError("port out of range")
    except ValueError as exc:
        _LOG.error("Invalid AGORA_TOKEN_INTERNAL_PORT: %s", exc)
        return

    child_env = os.environ.copy()
    child_env.update({
        "APP_ID": app_id,
        "APP_CERTIFICATE": app_cert,
        "SERVER_PORT": str(port),
    })
    try:
        process = await asyncio.create_subprocess_exec(str(binary), env=child_env)
    except OSError:
        _LOG.exception("Could not start the Agora token service")
        return

    app[_PROCESS_KEY] = process
    app[_PORT_KEY] = port
    ping_url = f"http://127.0.0.1:{port}/ping"
    for _ in range(30):
        if process.returncode is not None:
            _LOG.error("Agora token service exited during startup: %s", process.returncode)
            return
        try:
            async with app[_SESSION_KEY].get(ping_url, timeout=ClientTimeout(total=0.5)) as response:
                if response.status == 200:
                    _LOG.info("Agora token service ready on loopback port %s", port)
                    return
        except (ClientError, asyncio.TimeoutError):
            pass
        await asyncio.sleep(0.1)
    _LOG.error("Agora token service did not become ready")


async def _stop_token_service(app):
    process = app.get(_PROCESS_KEY)
    if process is not None and process.returncode is None:
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=3)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
    session = app.get(_SESSION_KEY)
    if session is not None:
        await session.close()


async def get_token(request):
    process = request.app.get(_PROCESS_KEY)
    if process is None or process.returncode is not None:
        raise web.HTTPServiceUnavailable(text="Agora token service unavailable")
    if request.content_length is not None and request.content_length > _MAX_REQUEST_BYTES:
        raise web.HTTPRequestEntityTooLarge(
            max_size=_MAX_REQUEST_BYTES, actual_size=request.content_length)
    body = await request.read()
    if len(body) > _MAX_REQUEST_BYTES:
        raise web.HTTPRequestEntityTooLarge(
            max_size=_MAX_REQUEST_BYTES, actual_size=len(body))

    headers = {"Content-Type": request.headers.get("Content-Type", "application/json")}
    if "Authorization" in request.headers:
        headers["Authorization"] = request.headers["Authorization"]
    url = f"http://127.0.0.1:{request.app[_PORT_KEY]}/getToken"
    try:
        async with request.app[_SESSION_KEY].post(url, data=body, headers=headers) as response:
            response_body = await response.read()
            return web.Response(
                body=response_body,
                status=response.status,
                headers={
                    "Content-Type": response.headers.get("Content-Type", "text/plain"),
                    "Cache-Control": "no-store",
                },
            )
    except (ClientError, asyncio.TimeoutError):
        _LOG.exception("Agora token service request failed")
        raise web.HTTPServiceUnavailable(text="Agora token service unavailable")


def install_token_service(app):
    app.on_startup.append(_start_token_service)
    app.on_cleanup.append(_stop_token_service)
    app.router.add_post("/getToken", get_token)
