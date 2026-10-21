FROM golang:1.23-bookworm AS token-builder

WORKDIR /src
COPY token_service/go.mod token_service/go.sum ./
RUN go mod download
COPY token_service/ ./
RUN CGO_ENABLED=0 go build -o /out/agora-token-service ./cmd

FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies, including gcc and other libraries
RUN apt-get update && \
	apt-get install -y software-properties-common ffmpeg gcc libc++-dev libc++abi-dev portaudio19-dev

# Install Python 3.12 and pip
RUN add-apt-repository ppa:deadsnakes/ppa && \
	apt update && \
	apt install -y python3.12 python3-pip python3.12-dev python3.12-venv && \
	update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 100 && \
	python3 -m ensurepip --upgrade && \
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
