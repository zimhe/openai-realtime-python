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

### Workspace agent with Seeduplex 3.0

The workspace voice path can use `REALTIME_PROVIDER=volcengine_duplex` without
changing the Agora channel or the existing `/start` and `/stop` endpoints. It
receives each participant's Agora audio separately, uses a short-lived SAUC ASR
stream to detect a wake phrase and collect meeting context, and forwards only
the addressed speaker's audio to Seeduplex. Until a wake phrase is recognized,
the Seeduplex input is explicitly muted. The default wake phrases are
`hey agent`, `hi agent`, `hello agent`, `你好助手`, `嘿助手`, and `嗨助手`.

Enable **Seeduplex 3.0 full duplex realtime voice** and **Streaming ASR 2.0
(SAUC, hourly)** in the same Volcengine project. Create an API key for that
project. In the Render service environment, set:

```env
REALTIME_PROVIDER=volcengine_duplex
VOLCENGINE_API_KEY=<secret API key>
VOLCENGINE_ASR_RESOURCE_ID=volc.seedasr.sauc.duration
WORKSPACE_AGENT_WAKE_PHRASES=hey agent,hi agent,hello agent,你好助手,嘿助手,嗨助手
```

`VOLCENGINE_ASR_RESOURCE_ID` and `WORKSPACE_AGENT_WAKE_PHRASES` are optional;
the values shown are the defaults. Keep the existing Agora credentials. If
the ASR service uses legacy app credentials, set `VOLCENGINE_APP_ID` and
`VOLCENGINE_ACCESS_KEY` for ASR instead; Seeduplex still requires
`VOLCENGINE_API_KEY`. The current mixed-audio provider remains available with
`REALTIME_PROVIDER=volcengine_split`.

### Streamoji authentication relay

The Unity client now calls this service's `POST /streamoji/auth-token` instead
of the Streamoji Firebase function. The request body contains `userId` and
`userName` and must include `Authorization: Bearer <Unity Authentication access
token>`. The backend verifies that the bearer belongs to `userId`, then sends
the request to Streamoji with server-side credentials. Configure these Render
environment variables:

```env
UNITY_PROJECT_ID=<existing Unity project ID>
STREAMOJI_CLIENT_ID=<Streamoji client ID>
STREAMOJI_CLIENT_SECRET=<Streamoji client secret>
```

Do not put the Streamoji secret in Unity assets or client builds. The legacy
Unity Editor-only avatar config fetch tool reads the same two Streamoji
variables from the Editor process environment; that tool still reaches
Streamoji directly. Rotate any credentials previously stored in a tracked
`Resources/SMJApiConfig.asset`, because deleting them from the current asset
does not erase Git history.

Rebuild and deploy this backend and rebuild the Unity client containing the
Agora identity handshake. For a smoke test, join the workspace with two
clients, start the agent, speak without the wake phrase, then say “hey agent”
and a question from one client. Confirm that only that client's voice is sent
to Seeduplex and that the transcript carries its Agora UID. Meeting transcript
context is currently kept in process memory only. The identity message is
client-reported and must not be used for authorization until validated against
a server-signed roster.

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
and runs it on a loopback port inside this container. This HTTP server forwards
`POST /getToken` and the legacy `GET /rtc/...`, `/rtm/...`, and `/rte/...`
routes to that local process. The request and response format is the same as
the former standalone Agora token service,
including RTC, RTM, and Chat token types. Chat requests still validate the
Unity Authentication access token, then look up or register the Agora Chat
account before returning a user token.

Configure these environment variables on this service:

| Variable | Purpose |
| --- | --- |
| `AGORA_APP_ID` | Shared Agora App ID; it must match the Unity `AgoraAppData` App ID. |
| `AGORA_APP_CERT` | Certificate for that same Agora project; keep it server-side. |
| `UNITY_PROJECT_ID` | Unity Authentication project ID, required for Chat token requests. |
| `CHAT_REST_HOST` | HTTPS Agora Chat REST host for user lookup and registration. |

`AGORA_TOKEN_INTERNAL_PORT` is optional and defaults to `18081`. Do not expose
that port. The Go process binds to `127.0.0.1`; only the aiohttp server's
normal HTTP port is public. `AGORA_TOKEN_BINARY` can point to a locally built
Go binary for development outside Docker. If the binary or required Agora
credentials are missing, the other HTTP features remain available and
the token routes respond with 503.

The voice agent and token endpoint use the same Agora project. If the current
voice-service App ID has not yet been updated to the Unity project's App ID,
update both `AGORA_APP_ID` and `AGORA_APP_CERT` together before cutover.

The local Unity `AgoraAppData` token server URL has been changed to this
service's base URL. Deploy this image and configure the variables before
rebuilding client applications. Most Unity token requests use `POST /getToken`
and expect a JSON `token` field; the Workspace screen-share path still uses the
legacy `GET /rtc/...` format. No client protocol change is needed. The bundled Go source is
based on the AgoraIO-Community token service and retains its MIT license in
`token_service/LICENSE`.
