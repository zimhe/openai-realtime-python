FROM golang:1.23-bookworm AS token-builder

WORKDIR /src
COPY token_service/go.mod token_service/go.sum ./
RUN go mod download
COPY token_service/ ./
RUN CGO_ENABLED=0 go build -o /out/agora-token-service ./cmd

FROM python:3.12-bookworm

ARG DEBIAN_FRONTEND=noninteractive

# Install native dependencies used by Agora media and audio packages.
RUN apt-get update && \
	apt-get install -y --no-install-recommends ffmpeg gcc libc++-dev libc++abi-dev portaudio19-dev && \
	rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=token-builder /out/agora-token-service /app/bin/agora-token-service

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY realtime_agent realtime_agent

EXPOSE 8080

# Default command to run the app
CMD ["python3", "-m", "realtime_agent.main", "server"]

