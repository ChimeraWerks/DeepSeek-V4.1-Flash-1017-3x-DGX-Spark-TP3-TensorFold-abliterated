# Security

Run on a trusted private network. Keep the serving API and native engine off the public internet; disable router port forwarding. Both bind to head loopback by default: `API_PORT` is 18002 and `ENGINE_PORT` is 8003. For remote API access, use SSH forwarding or an authenticated private reverse proxy. Keep the native engine private.

## Report a vulnerability

Use the repository's private vulnerability-reporting option when enabled. Otherwise, open a public issue requesting a private maintainer contact, without sensitive details.

Include recipe and engine revisions, API version, affected component, impact and minimal reproduction steps. Redact credentials, private prompts, host addresses and unrelated user data. Test systems you own or have permission to assess.

## Protect credentials

`SPARK_API_KEY` authenticates serving clients; `SPARK_ADMIN_KEY` controls operator settings. Leave either empty in `cluster.env` to generate it on the head or reuse its existing value in the mode-0600 credential file. Use distinct keys. Serving clients have no administrator privileges.

Never commit `cluster.env`. Keep it private. API client configuration uses environment-variable names, not literal secrets. Restrict access to the head's credential file and API state directory. The state directory requires mode 0700 and cannot be a symlink.

Administrator access controls sampling, reasoning and admission settings. Give it to trusted operators only. Replace exposed keys in the installation's credential configuration and stop using the old values. See the [serving API guide](serving/README.md).

## Service boundaries

`/health`, `/ready` and `/metrics` are unauthenticated. Private listeners protect operational information and generation traffic.

The API returns function-tool calls to clients without executing them. Clients that execute them need their own authorization and isolation.

A disconnected response can leave accepted work running. Check request state before replaying a generation or partial stream, and check an accepted image job's identity before retrying. Keep request state, diagnostics and logs private.

This release supports text; vision is untested with these weights. Secure access independently of refusal removal, and review outputs for safety and correctness. See [weight provenance](release/weight-provenance.md) and [troubleshooting](docs/troubleshooting.md).
