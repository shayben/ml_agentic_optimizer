# Docker topology

The current design uses **two runtime images** plus a local Copilot CLI session:

1. **Broker image** (`docker/Dockerfile.broker`) — runs the FastAPI control plane (`agentic-optimizer-broker`).
   This is the only network-reachable component.
2. **Node image** (`docker/Dockerfile.node`) — runs the PyTorch training entrypoint with `TrainingBridge` and
   connects to the broker.
3. **Local machine** — runs GitHub Copilot CLI with the MCP stdio server from `agent/mcp-config.json`.

The old single image that installed Copilot CLI on the training node is deprecated. The Copilot CLI is **not**
installed in either new image.

## Broker image

```powershell
$Registry = "<registry-name>.azurecr.io"
$BrokerImage = "$Registry/agentic-optimizer-broker:0.1.0"

docker build -t $BrokerImage -f .\docker\Dockerfile.broker .
docker push $BrokerImage
```

Run with a strong bearer token and TLS/ingress or an SSH tunnel for non-loopback exposure:

```powershell
docker run --rm -p 8765:8765 `
  -e CONTROL_PLANE_HOST=0.0.0.0 `
  -e CONTROL_PLANE_TOKEN="<strong-token>" `
  -e CONTROL_PLANE_PERSIST=/data/control-plane.db `
  -v broker-data:/data `
  $BrokerImage
```

Useful broker env vars: `CONTROL_PLANE_HOST`, `CONTROL_PLANE_PORT`, `CONTROL_PLANE_TOKEN`,
`CONTROL_PLANE_PERSIST`, `CONTROL_PLANE_MAX_BODY_BYTES`, and `CONTROL_PLANE_INSECURE=1` for intentional local
unsafe testing only.

## Training-node image

```powershell
$NodeImage = "$Registry/agentic-optimizer-node:0.1.0"

docker build -t $NodeImage -f .\docker\Dockerfile.node .
docker push $NodeImage
```

The node container needs the broker address, token, and run namespace:

```powershell
docker run --rm --gpus all `
  -e CONTROL_PLANE_URL="https://<broker-host>" `
  -e CONTROL_PLANE_TOKEN="<strong-token>" `
  -e CONTROL_PLANE_RUN_ID="run-001" `
  $NodeImage
```

## Local agent

Install local MCP/HPO extras and point Copilot CLI at the stdio server:

```powershell
pip install -e ".[mcp,hpo]"
$env:CONTROL_PLANE_URL = "https://<broker-host>"
$env:CONTROL_PLANE_TOKEN = "<strong-token>"
$env:CONTROL_PLANE_RUN_ID = "run-001"
copilot --additional-mcp-config @agent/mcp-config.json
```
