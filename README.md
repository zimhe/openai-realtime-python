# Realtime Agent

This project demonstrates how to deliver ultra-low latency access to OpenAI with exceptional audio quality using Agora's SD-RTN and OpenAI's Realtime API. By integrating Agora's SDK with OpenAI's Realtime API, it ensures seamless performance and minimal delay across the globe.

## Prerequisites

Before running the demo, ensure you have the following installed and configured:

- Python 3.11 or above

- Agora account:

  - [Login to Agora](https://console.agora.io/en/)
  - Create a [New Project](https://console.agora.io/projects), using `Secured mode: APP ID + Token` to obtain an App ID and App Certificate.

- OpenAI account:

  - [Login to OpenAI](https://platform.openai.com/signup)
  - Go to Dashboard and [obtain your API key](https://platform.openai.com/api-keys).

- Additional Packages:

  - On macOS:
    ```bash
    brew install ffmpeg portaudio
    ```
  - On Ubuntu (verified on versions 22.04 & 24.04):
    ```bash
    sudo apt install portaudio19-dev python3-dev build-essential
    sudo apt install ffmpeg
    ```

## Network Architecture

<!-- <img src="./architecture.png" alt="architecture" width="700" height="400" /> -->
<picture>
  <source srcset="architecture-dark-theme.png" media="(prefers-color-scheme: dark)">
  <img src="architecture-light-theme.png" alt="Architecture diagram of Conversational Ai by Agora and OpenAi">
</picture>

## Organization of this Repo

- `realtimeAgent/realtime` contains the Python implementation for interacting with the Realtime API.
- `realtimeAgent/agent.py` includes a demo agent that leverages the `realtime` module and the [agora-realtime-ai-api](https://pypi.org/project/agora-realtime-ai-api/) package to build a simple application.
- `realtimeAgent/main.py` provides a web server that allows clients to start and stop AI-driven agents.

## Run the Demo

### Setup and run the backend

1. Create a `.env` file for the backend. Copy `.env.example` to `.env` in the root of the repo and fill in the required values:
   ```bash
   cp .env.example .env
   ```
1. Create a virtual environment:
   ```bash
   python3 -m venv venv && source venv/bin/activate
   ```
1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
1. If you are using the Volcengine provider, configure the context-related environment variables in `.env`:
  ```env
  VOLCENGINE_DIALOG_ID=your-stable-dialog-id
  ARK_API_KEY=your-ark-api-key
  VOLCENGINE_FUNCTION_CALL_MODEL=doubao-seed-2-0-lite-260215
  ```
  Reuse the same `VOLCENGINE_DIALOG_ID` for the same user or conversation when you want Doubao to load recent history across sessions.
1. Run the demo agent:
   ```bash
   python -m realtime_agent.main agent --channel_name=<channel_name> --uid=<agent_uid>
   ```

### Start HTTP Server

1. Run the http server to start demo agent via restful service
   ```bash
   python -m realtime_agent.main server
   ```
   The server provides a simple layer for managing agent processes.

### API Resources

- [POST /start](#post-start)
- [POST /stop](#post-stop)

### POST /start

This api starts an agent with given graph and override properties. The started agent will join into the specified channel, and subscribe to the uid which your browser/device's rtc use to join.

| Param        | Description                                                                                                                                                            |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| channel_name | (string) channel name, it needs to be the same with the one your browser/device joins, agent needs to stay with your browser/device in the same channel to communicate |
| uid          | (int)the uid which ai agent use to join                                                                                                                                |
| system_instruction    | The system instruction for the agent                                                                                                                          |
| voice        | The voice of the agent                                                                                                                                                 |

Example:

```bash
curl 'http://localhost:8080/start_agent' \
  -H 'Content-Type: application/json' \
  --data-raw '{
    "channel_name": "test",
    "uid": 123
  }'
```

### POST /stop

This api stops the agent you started

| Param        | Description                                                |
| ------------ | ---------------------------------------------------------- |
| channel_name | (string) channel name, the one you used to start the agent |

Example:

```bash
curl 'http://localhost:8080/stop_agent' \
  -H 'Content-Type: application/json' \
  --data-raw '{
    "channel_name": "test"
  }'
```

### Front-End for Testing

To test agents, use Agora's [Voice Call Demo](https://webdemo.agora.io/basicVoiceCall/index.html).
# Office lobby directory

This always-on HTTP service can also list live Unity Office dedicated servers.
It only stores routing metadata; Unity gameplay still runs on the dedicated
server and clients connect directly to its advertised UDP address and port.

Set `OFFICE_DIRECTORY_KEY` to a long random secret in this service's environment
and in each dedicated server's environment. Do not put it in a Unity client
build or commit it to a batch file. The directory endpoints are:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/offices` | Public list of live offices, returned as `{"offices":[...]}`. |
| `POST` | `/offices/heartbeat` | Register or refresh an office. Requires `Authorization: Bearer <OFFICE_DIRECTORY_KEY>`. |
| `DELETE` | `/offices/{instanceId}` | Remove an office on orderly shutdown. Requires the same credential. |

Heartbeat entries expire after 75 seconds. Unity dedicated servers refresh
every 25 seconds and register only after their environment scene is ready. A
service restart temporarily empties the in-memory list until the next heartbeat.
Run one HTTP instance for this initial setup; a future multi-instance deployment
needs a shared TTL store such as Redis. The `instanceId` uniquely identifies a
server process, while `lobbyId` identifies its whitelisted Unity scene. Multiple
processes may advertise the same scene.

Unity also sends a stable `slotId` for each hosted lobby and port. When a
restarted process registers the same slot, its new heartbeat replaces the old
process immediately, even if shutdown could not send DELETE. A late DELETE
from the old process does not remove the replacement. By default the slot is
derived from the server machine name, lobby ID, and port; hosted environments
with changing machine names should pass a stable `-directorySlotId=<name>` to
the Unity dedicated server. Use distinct names for independently hosted servers.

The server sends `lobbyId`, `instanceId`, `slotId`, `displayName`, `address`, `port`,
`scope` (`local` or `public`), `sceneName`, `buildVersion`, `maxPlayers`,
`currentPlayers`, and an optional `previewImageUrl`. Preview images must be
public HTTPS URLs. The Unity client can also use its existing local catalog
image when the directory entry has the same lobby ID. An advertised `local`
address is only reachable by clients on that LAN; public deployment must use a
publicly reachable host and open the Unity Transport UDP port.

## Agora Token endpoint

The Docker image builds the existing Agora Go token service from `token_service/`
and runs it on a loopback port inside this container. This HTTP server exposes
only `POST /getToken` and forwards requests to that local process. The request
and response format is the same as the former standalone Agora token service,
including RTC, RTM, and Chat token types. Chat requests still validate the
Unity Authentication access token, then look up or register the Agora Chat
account before returning a user token.

Configure these environment variables on this service:

| Variable | Purpose |
| --- | --- |
| `AGORA_TOKEN_APP_ID` | Copy `APP_ID` from the former token service; it must match the Unity `AgoraAppData` App ID. |
| `AGORA_TOKEN_APP_CERT` | Copy `APP_CERTIFICATE` from the former token service; keep it server-side. |
| `UNITY_PROJECT_ID` | Unity Authentication project ID, required for Chat token requests. |
| `CHAT_REST_HOST` | HTTPS Agora Chat REST host for user lookup and registration. |

`AGORA_TOKEN_INTERNAL_PORT` is optional and defaults to `18081`. Do not expose
that port. The Go process binds to `127.0.0.1`; only the aiohttp server's
normal HTTP port is public. `AGORA_TOKEN_BINARY` can point to a locally built
Go binary for development outside Docker. If the binary or required Agora
credentials are missing, the other HTTP features remain available and
`/getToken` responds with 503.

The existing `AGORA_APP_ID` and `AGORA_APP_CERT` still belong to the voice
agent. They are deliberately separate because the voice agent and Unity client
may use different Agora projects.

The local Unity `AgoraAppData` token server URL has been changed to this
service's base URL. Deploy this image and configure the variables before
rebuilding client applications. The existing Unity client sends `POST /getToken` and expects a
JSON `token` field; no protocol change is needed. The bundled Go source is
based on the AgoraIO-Community token service and retains its MIT license in
`token_service/LICENSE`.
