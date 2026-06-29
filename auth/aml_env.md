# Broker bearer token and AML environment

The primary production shape is:

- GitHub Copilot CLI authenticates **locally** with the user's GitHub/EMU token.
- The local MCP stdio server, the remote training node, and the broker share a separate
  `CONTROL_PLANE_TOKEN` bearer credential.
- The broker is the only network-reachable component.

The older “run Copilot on AML” model is legacy and should only be used for the secondary file-contract mode.

## Broker token setup

Generate a strong random token and store it in a secret manager such as Azure Key Vault or AML secrets:

```powershell
$bytes = New-Object byte[] 32
[Security.Cryptography.RandomNumberGenerator]::Fill($bytes)
$token = [Convert]::ToBase64String($bytes)
az keyvault secret set --vault-name <key-vault-name> --name control-plane-token --value $token
```

Set the same token in all three places:

| Process | Variables |
| --- | --- |
| Broker | `CONTROL_PLANE_TOKEN`, optional `CONTROL_PLANE_PERSIST`, `CONTROL_PLANE_MAX_BODY_BYTES` |
| Training node | `CONTROL_PLANE_URL`, `CONTROL_PLANE_TOKEN`, `CONTROL_PLANE_RUN_ID` |
| Local CLI/MCP | `CONTROL_PLANE_URL`, `CONTROL_PLANE_TOKEN`, `CONTROL_PLANE_RUN_ID` |

Use TLS or an SSH tunnel for any non-loopback broker. The token is privileged: anyone holding it can pause,
resume, tune, or otherwise mutate a live training process.

## Broker safeguards

- Bearer-token comparison is constant-time.
- Request body size is limited by `CONTROL_PLANE_MAX_BODY_BYTES`.
- Non-loopback binding without `CONTROL_PLANE_TOKEN` is refused unless `CONTROL_PLANE_INSECURE=1` is explicitly set
  for local unsafe testing.
- Optional SQLite persistence is enabled with `CONTROL_PLANE_PERSIST=<path.db>`.

## AML job secret injection

Reference the broker token from your AML job rather than writing it to files:

```yaml
environment_variables:
  CONTROL_PLANE_URL: https://<broker-host>
  CONTROL_PLANE_TOKEN: ${{secrets.control-plane-token}}
  CONTROL_PLANE_RUN_ID: run-001
```

If your workspace references Key Vault explicitly, keep the final env var name the same:

```yaml
environment_variables:
  CONTROL_PLANE_TOKEN: "{{keyvault:<key-vault-name>:control-plane-token}}"
```

## Local Copilot authentication

Copilot CLI now runs on your local machine, so use the normal local Copilot/GitHub sign-in flow. You do not need
to forward `COPILOT_GITHUB_TOKEN` to AML for the primary MCP broker topology.

`auth/read_local_token.ps1` is retained for legacy experiments that intentionally run Copilot on remote compute.
Forwarding a user OAuth token to AML is credential movement; prefer keeping Copilot local.
