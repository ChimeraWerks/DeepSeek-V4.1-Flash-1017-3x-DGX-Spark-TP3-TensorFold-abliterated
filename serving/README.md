# DGX Spark serving API

`chimera-dgx-sparks` serves DeepSeek V4.1 Flash 1017 on three DGX Spark systems with TensorFold TP3. The FastAPI package runs on the head Spark. It provides OpenAI-compatible Chat Completions and Responses routes and native Anthropic Messages routes, client authentication, sampling presets, request admission controls and operator settings. TensorFold runs native generation in a separate process.

## Install and configure

From the repository root:

```sh
pip install ./serving
```

Or install the built wheel with `pip install` and its local filename. Deploy the model and engine separately using `cluster.env.example`, the deployment scripts and `release/manifest.json`. See the [repository README](../README.md) for deployment and [tuning guide](../docs/tuning.md) for engine settings.

The API reads JSON from a top-level configuration object or an `api` key. This example sets the deployment context size, local state directory and one client:

```json
{
  "api": {
    "upstream": "http://localhost:8003",
    "model": "deepseek-v4.1-flash",
    "host": "localhost",
    "port": 18002,
    "state_dir": "./state",
    "clients": {"default": "SPARK_API_KEY"},
    "admin_clients": [],
    "context_tokens": 1048576
  }
}
```

| Configuration key | Package default | Meaning or constraint |
|---|---|---|
| `upstream` | HTTP on loopback, port `8003` | Native engine origin. Must use HTTP or HTTPS, with no credentials, query or fragment and no path other than `/`. |
| `model` | `deepseek-v4.1-flash` | Base model identifier accepted by the API. |
| `host` | Loopback only | API listener address. |
| `port` | `18002` | Positive integer, at most `65535`. |
| `state_dir` | User-local state directory | Persistent API state. A relative path is resolved against the JSON file's directory. The directory must be private (`0700`) and must not be a symlink. |
| `clients` | `{"default":"SPARK_API_KEY"}` | Client IDs mapped to environment-variable names, not literal secrets. Each referenced value must be distinct and at least 16 characters. |
| `admin_clients` | `[]` | Configured client IDs permitted to read and change operator settings. No client has this access by default. |
| `upstream_key_env` | `null` | Optional environment-variable reference for native authentication; adapter behavior must be checked separately. |

Set `SPARK_API_KEY` in the service environment before startup. Keep the engine listener private. For remote API access, use SSH forwarding or an authenticated private reverse proxy. Examples use the local API on port `18002`. Run one configured API worker per process, with a separate job ledger for each worker.

The sections below cover sampling, reasoning, admission, capacity and vision settings. Change fields in the live-settings table through the operator API. For other fields, edit the base JSON and restart.

## Endpoints

All `/v1/*`, `/tokenize` and `/admin/*` routes listed here require `Authorization: Bearer $SPARK_API_KEY` or `x-api-key: $SPARK_API_KEY`. Operator routes also require the client ID in `admin_clients`. `/health`, `/ready` and `/metrics` are unauthenticated.

| Method and path | Purpose |
|---|---|
| `POST /v1/chat/completions` | Native Chat Completions, JSON or server-sent events (SSE). |
| `POST /v1/responses` | Native Responses, JSON or SSE. |
| `POST /v1/messages` | Native Anthropic Messages, JSON or SSE. The request goes to the engine unconverted; the API fills only the sampling, thinking and output fields the client left out. |
| `POST /v1/messages/count_tokens` | Forward to the engine's native token count. |
| `GET /v1/models` | Base model and one `model@preset` entry per configured preset. |
| `GET /v1/profiles` | Current sampling defaults, reasoning default and labelled presets. |
| `GET /health`, `GET /ready` | Readiness and current API/engine state. |
| `GET /metrics` | Plain-text request, stream-error, active-request, waiting-request, admitted-token and vision-job counts. |
| `POST /tokenize`, `POST /v1/tokenize` | Forward a JSON object to native `/tokenize`, with a 30-second call timeout. |
| `GET /admin/settings` | Read live settings, base settings and engine capability. |
| `PUT /admin/settings` | Validate and apply a partial live-settings update. |
| `POST /admin/settings/reset` | Restore all live fields to their base JSON values. |
| `POST /v1/vision/jobs` | Submit a supported standalone image question. See [Vision jobs](#vision-jobs). |
| `GET /v1/vision/jobs/{identity}` | Read a job with the credential of the client that owns it. |

### Chat Completions and Responses

| Request field | Behavior |
|---|---|
| `model` | Defaults to the configured base model. Accepts that model or `model@preset`; another base model is rejected. |
| `messages` | Chat Completions requires a nonempty array. Native rendering handles message contents. |
| `input` | Required for Responses. Native rendering handles its contents. |
| `stream` | Boolean `true` selects SSE. Otherwise the API returns a JSON response. |
| `n` | Only `1` is supported. |
| `max_tokens` | Chat output cap. Defaults to `output_tokens`, bounded by `max_output_tokens`. |
| `max_output_tokens` | Responses output cap, with the same configured default and ceiling. |
| `previous_response_id` | Accepted only when one client is configured, because stored responses are not tied to a client. The stored history it continues is not counted by admission; send the full `input` when that matters. |
| `max_completion_tokens` | Fallback output-cap field when the endpoint's primary cap field is absent; removed before forwarding. |

Output caps must be positive integers; a null primary cap uses the configured default. The API sends generation to TensorFold and leaves returned function-tool calls for the client to execute. Sampling and reasoning fields are described below.

```sh
curl http://localhost:18002/v1/chat/completions \
  -H "Authorization: Bearer $SPARK_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash@standard","messages":[{"role":"user","content":"Explain tensor parallelism in three sentences."}],"max_tokens":512}'
```

```sh
curl -N http://localhost:18002/v1/responses \
  -H "Authorization: Bearer $SPARK_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","input":"Explain tensor parallelism in three sentences.","max_output_tokens":512,"stream":true}'
```

While the native stream opens, ordinary SSE requests send `: recipe waiting` heartbeat comments, separate from model output. A relay failure sends `stream_incomplete` without restarting generation. A client disconnect releases its ordinary native stream. Accepted vision jobs run independently of the connection.

### Models and presets

```sh
curl http://localhost:18002/v1/models \
  -H "Authorization: Bearer $SPARK_API_KEY"

curl http://localhost:18002/v1/profiles \
  -H "Authorization: Bearer $SPARK_API_KEY"
```

| Response field | Meaning |
|---|---|
| Models: `object`, `data` | `object` is `list`; `data` contains the base model followed by presets sorted by name. |
| Model entry: `id`, `object`, `owned_by` | Model identifier, `model`, and `recipe`. Preset IDs are `deepseek-v4.1-flash@<name>` with the default base model. |
| Preset model entry: `description` | Preset label, or its name if the label is empty or absent. |
| Profiles: `defaults` | Current `sampling` plus `reasoning_effort` from `reasoning_default`. |
| Profiles: `default_profile`, `profiles` | Default preset name or null, and the full preset definitions. |
| Profiles: `select.header`, `select.model` | Examples of the header and model-suffix selectors. |

### Health

Readiness returns HTTP `200` when native health succeeds, native `warming` is false or absent, and the vision worker is running. Otherwise it returns HTTP `503`.

| Response field | Meaning |
|---|---|
| `ok` | Combined readiness result. |
| `version` | Serving package version. |
| `native` | Native health JSON; `{"available":false}` on the handled connection or JSON errors. |
| `active`, `waiting` | Current admission counts. |
| `max_concurrency`, `default_profile`, `sampling`, `reasoning_output` | Current configured values. |
| `vision_jobs.states` | Job counts returned by the ledger. |
| `vision_jobs.automatic_replay` | `false`. |
| `vision_jobs.worker_alive` | Whether the job runner has not finished. |
| `vision_jobs.output_budget` | Output-budget information returned by the job implementation. |

### Operator settings

Add the operator's client ID to `admin_clients` in the base configuration. For the example client, use `"admin_clients":["default"]` to grant read and write access.

```sh
curl http://localhost:18002/admin/settings \
  -H "Authorization: Bearer $SPARK_API_KEY"

curl -X PUT http://localhost:18002/admin/settings \
  -H "Authorization: Bearer $SPARK_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"max_concurrency":8,"max_queue":64}'
```

Only these fields can change live:

| Live field | Update behavior |
|---|---|
| `sampling` | Merge individual sampling values. Null defaults are not allowed. |
| `profiles` | Merge by preset name. Each supplied preset object replaces that preset's definition; a null entry removes it. |
| `client_profiles` | Merge by client ID. A null entry removes an assignment. |
| `default_profile` | Replace with an existing preset name or null. |
| `reasoning_default` | Replace with a supported named effort. |
| `max_concurrency` | Replace with a positive integer no greater than `engine_streams`. |
| `max_queue` | Replace with a positive integer. |
| `effort_map` | Merge named effort numbers; the resulting map must remain complete and valid. |

The API validates the full configuration before applying changes and resizing admission limits. It saves overrides in `live-settings.json` under `state_dir` and loads them over base values on restart. An invalid saved file blocks startup. Reset restores all live fields to base values, leaving non-live fields unchanged. It removes the override file when no differences remain.

```sh
curl -X POST http://localhost:18002/admin/settings/reset \
  -H "Authorization: Bearer $SPARK_API_KEY"
```

| Settings response field | Meaning |
|---|---|
| `settings` | Current values of the eight live fields. |
| `base` | Their base configuration values, captured before saved overrides load. |
| `engine_streams` | Configured engine-stream ceiling. |
| `engine_efforts` | Native named-effort number table. |
| `numeric_effort` | True only when a successful native health response reports this capability. Unavailable or unreadable capability checks return false. |
| `active`, `waiting` | Current admission counts. |
| `saved` | Whether the saved live-settings file exists. |

Invalid settings return HTTP `400` with `invalid_settings`. Changing `effort_map` to a non-native mapping requires engine `numeric_effort` support; otherwise PUT returns HTTP `409` with `engine_capability`. PUT skips this check when restoring the native mapping or leaving the map unchanged. Reset skips it too, so remapped base configurations need a compatible engine.

## Sampling defaults and presets

Each field resolves in order: installation default, selected preset, then an explicit non-null request value. Zero counts as a value within the field's range; null inherits the default.

| Sampling field | Package default | Accepted API range |
|---|---|---|
| `temperature` | `1.0` | Finite number from `0` to `2`. |
| `top_p` | `1.0` | Finite number greater than `0`, at most `1`. |
| `top_k` | `20` | Integer from `0` to `129408`. |
| `min_p` | Not set | Finite number from `0` to `1`. |
| `presence_penalty` | `0.0` | Finite number from `-2` to `2`; no native penalty support. |
| `frequency_penalty` | Not set | Finite number from `-2` to `2`; no native penalty support. |

Sampling rejects booleans. The `sampling` object requires `temperature`, `top_p`, `top_k` and `presence_penalty`. The penalty fields above have no effect in this TensorFold recipe. `top_k: 20` samples GPU candidates; `top_k: 0` uses whole-vocabulary CPU sampling.

| Preset | Label | `temperature` | `top_p` | `top_k` | `min_p` |
|---|---|---|---|---|---|
| `standard` | DeepSeek 1017 card: temperature 1, top_p 1 | 1.0 | 1.0 | 20 | Not set |
| `focused` | temperature 1, top_p 0.95, min_p 0.05 | 1.0 | 0.95 | 20 | 0.05 |

The first nonempty selector wins: `X-Recipe-Profile` header, model suffix after `@`, the client's `client_profiles` entry, then `default_profile`. Unknown presets are rejected. With no preset, installation defaults apply. `client_profiles` defaults to `{}`; `default_profile` defaults to null.

| Preset field or rule | Constraint |
|---|---|
| Preset name | 1 to 48 characters. Starts with a lowercase letter or digit; remaining characters may also contain `.`, `_` or `-`. |
| `label` | Optional display text, at most 120 characters. |
| Sampling fields | Any of the six sampling fields above, with the same validation. Unspecified fields inherit installation defaults. |
| `reasoning_effort` | Optional named effort used when a request supplies none. Numeric preset efforts are not accepted. |

Add a preset live with an operator credential:

```sh
curl -X PUT http://localhost:18002/admin/settings \
  -H "Authorization: Bearer $SPARK_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"profiles":{"concise":{"label":"Focused sampling with low thinking","temperature":1,"top_p":0.95,"top_k":20,"min_p":0.05,"reasoning_effort":"low"}}}'
```

Select it with `deepseek-v4.1-flash@concise` or `X-Recipe-Profile: concise`. Set `default_profile` for the default, or assign clients through `client_profiles`. When deleting a preset, remove or change its default and client assignments in the same update.

## Thinking and reasoning effort

Thinking defaults to `high`. Named efforts use this native mapping:

| Effort | Native number or behavior |
|---|---|
| `none` | Thinking disabled; not a numeric map entry. |
| `minimal` | `25` |
| `low` | `50` |
| `medium` | `62` |
| `high` | `75` |
| `xhigh` | `88` |
| `max` | `100` |

| Field | Behavior |
|---|---|
| `reasoning_default` | Configuration default for requests without a preset effort or explicit effort. One of the seven names above; default `high`. Live-editable. |
| Request `reasoning_effort` | A named effort, integer `1` to `100`, or digit string representing that range. Booleans and floating-point values are not accepted as numeric effort. |
| Request `reasoning.effort` | Alternate input when the top-level `reasoning_effort` key is absent. |
| Request `chat_template_kwargs.reasoning_effort` | Fallback when the preceding effort input is null or absent. |
| Request `thinking`, `chat_template_kwargs.thinking`, `chat_template_kwargs.enable_thinking` | Boolean thinking controls. False disables thinking; true enables it. A conflicting explicit effort is rejected. |
| Request `thinking_budget` | Optional positive integer no greater than the request's total output cap. It is separate from the effort level. |
| `effort_map` | Configuration map with exactly the six numeric names above, each mapped to an integer from `1` to `100`. Live-editable. |
| `reasoning_output` | `preserve` by default, or `hide`. Controls returned reasoning, not whether the model thinks. Requires restart. |

Request aliases `off`, `disabled` and `false` mean `none`; `standard` means `medium`. Without explicit effort, the preset overrides `reasoning_default`. Enabling thinking with a resulting default of `none` selects `low`. Use one consistent set of thinking controls.

Unchanged named efforts reach the engine by name; remapped levels and explicit numbers reach it as integers. Both numeric forms require an engine whose `/health` reports `numeric_effort`. The operator endpoint checks mapping changes as described above. The request normalizer skips engine-health checks.

To remap `low` while retaining the other levels:

```sh
curl -X PUT http://localhost:18002/admin/settings \
  -H "Authorization: Bearer $SPARK_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"effort_map":{"low":45}}'
```

An explicit request number bypasses named-level remapping:

```sh
curl http://localhost:18002/v1/chat/completions \
  -H "Authorization: Bearer $SPARK_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"Explain why this deployment uses three ranks."}],"reasoning_effort":45,"max_tokens":512}'
```

For ordinary Chat and Responses, `reasoning_output: hide` strips reasoning fields from JSON and reasoning data from SSE, leaving effort and output caps unchanged. Vision-job results use a separate path; check its reasoning handling separately.

## Request admission

| Configuration key | Package default | Constraint or purpose |
|---|---|---|
| `engine_streams` | `16` | Positive integer, at most `16`; must match the engine's configured decode-stream capacity. |
| `max_concurrency` | `16` | Positive integer, no greater than `engine_streams`. Live-editable admission cap. |
| `max_queue` | `64` | Positive integer queue limit. Live-editable. |
| `admitted_tokens` | `5336064` | Positive integer admitted-token budget. |
| `request_timeout` | `3590` seconds | Ordinary request deadline; also supplied to the vision-job implementation. |
| `queue_timeout` | `1800` seconds | Configured queue deadline; also supplied to the vision-job implementation. |
| `heartbeat` | `15` seconds | SSE waiting-comment interval. |

All three timeout/interval values must be finite, greater than zero and at most `3600` seconds. The API concurrency cap controls admission, not engine decode streams. Excess requests wait within the queue limits; queue ordering and fairness are unspecified.

| HTTP status | Error code or condition |
|---|---|
| `400` | `invalid_request` for policy failures; `admission_oversized` for an admitted-token budget rejection. |
| `401` | `invalid_api_key` for a missing or invalid bearer credential. |
| `403` | `forbidden` for an authenticated client without operator access. |
| `413` | Request body exceeds the configured byte limit. |
| `422` | `reasoning_repetitive` for Chat input rejected by the incoming-reasoning guard. |
| `429` | `queue_full`. |
| `502` | `upstream_transport` for native HTTP transport failure; `empty_final` for a successful nonstream Chat result with neither final text nor tool calls. |
| `504` | `deadline`. |

API errors contain an `error` object with `message`, `type: "recipe_error"` and `code`, plus `Cache-Control: no-store` and `X-Should-Retry: false` headers. After SSE starts, errors can arrive as stream data instead of a new HTTP status. For accepted image jobs, check the job identity before retrying after an error or lost connection.

## Capacity, spill and context

The bundled native profile `v1.2` and API defaults match the release manifest: `V4-MRF-1017` (EXL3 3.84 bpw), engine base `30a9647` plus the documented comment/test-name-only release delta, 16 streams and 1,048,576-token context. Each configured root contains portable `bin/`, `weights/`, `ranks/`, `kit/` and `spill/` paths. Weights: [Hugging Face](https://huggingface.co/ChimeraWerks/DeepSeek-V4.1-Flash-1017-EXL3-3.84bpw-abliterated). See the [model card](../docs/model-card.md). The earlier `v1.1` / `14803a3` and `v1` / `7d5b4d8` / 10M configurations are historical.

| Deployment setting | Value |
|---|---|
| Engine decode streams | `16` |
| CUDA graph cap | `128` |
| KV pool | `8000000` tokens |
| Context | `1048576` tokens |
| Head-system NVMe spill | `256` GiB |
| Startup memory reserve | `4` GiB |
| Native sampler default | `top_k: 20` |

The engine uses `TF_DS_SPILL_DIR` and `TF_DS_SPILL_GIB=256`, with rank 0 options `--spill-dir`, `--spill-gib 256` and `--top-k 20`. Keep the remaining engine flags in `release/manifest.json`, separate from API JSON. Match API context to native capacity; changing the API value alone allocates no KV storage.

| API configuration key | Package default | Constraint or purpose |
|---|---|---|
| `context_tokens` | `1048576` | Positive integer context setting, matching the v1.2 native profile and release manifest. |
| `output_tokens` | `32768` | Default request output cap; positive integer, at most `max_output_tokens`. |
| `max_output_tokens` | `131072` | Positive integer maximum request output cap. |
| `safety_tokens` | `1024` | Positive integer context reserve. `max_output_tokens + safety_tokens` must be strictly less than `context_tokens`. |
| `max_body_bytes` | `67108864` (64 MiB) | Positive integer request-body byte limit. |
| `max_response_bytes` | `67108864` (64 MiB) | Positive integer limit for native responses read into memory by the API. |
| `image_token_reserve` | `16384` | Positive integer image reserve configured for the engine adapter. |

Fit input, output and safety reservations within native context. The engine adapter calculates input accounting and admitted tokens. API validators check configuration values. The output ceiling and admitted-token budget stay independent of KV pool size.

### Optional capacity controls

Spill and optional KV sharding are separate features. The production spill setting enables neither optional sharding, disk chat continuation nor automatic shortening.

The managed lifecycle preflights v1.2 spill on every node using a matching `capacity_build_receipt`, executable checksum and `rank_cache_sha256`. Prepare these artifacts for preflight. For a new installation, run the managed supervisor with `--qualification-window` for acceptance testing and keep its results. Direct native `--run` stays blocked until v1.2 passes acceptance; the direct launcher has no qualification-window option.

Artifact validation requires `checksums` for `bin/BUILD.json`, both binaries and `weights/chat_template.jinja`. The `ds41f.native-build.v1` file records engine base `30a9647`, `adaptive_learning: true` and actual binary hashes. Record the documented source delta alongside it; the launcher checks artifacts, not source equivalence. Rebuilt files can differ byte for byte; see [build verification](../release/build-verification.json). An explicit `cluster.capacity` block replaces profile spill controls without duplicate flags. Set `spill_gib: 0` to disable spill. Free-form spill environment overrides are rejected.

Optional capacity features need a matching capacity build receipt and preflight on all three systems. Spill needs private local-disk paths, sufficient free-space reserves and matching model, engine, Engram and rank-cache identities. Optional TP3 KV sharding supports fetch decode only. Each owner part must fit the full window plus round reserve, and the total pool must be divisible by `6144`. The production `8000000` pool fails that divisibility requirement. Spill with `cache_salt` is unsupported. Check these prerequisites against the selected engine; the API does not enforce them.

The API does not shorten prompts. A prompt plus its output reserve must fit `context_tokens`, or the request is refused; clients manage their own context.

## Vision jobs

**Vision is not tested with these weights.**

`POST /v1/vision/jobs` accepts one standalone user image question without assistant history or tools. It rejects repetitive incoming reasoning. After validation and job acceptance, the API returns HTTP `202`. Standalone image questions sent to Chat Completions use the same job path; other Chat requests use ordinary native generation.

| Header or response field | Meaning |
|---|---|
| `Idempotency-Key` | Preferred caller-provided job key, at most 256 characters. |
| `X-Client-Request-Id` | Fallback job key when the preferred header is empty or absent; same length limit. |
| `job_id` | Identity returned by the job implementation. |
| `status_url` | Relative path `/v1/vision/jobs/{identity}` to poll. |
| `created` | Creation flag returned by the job implementation. |

Poll `status_url` with the same client's credential. Missing or inaccessible jobs return `404` with `not_found`; admission conflicts return `409` with `vision_not_accepted`. The job implementation defines status details, idempotency and recovery. Disconnecting a Chat waiter after acceptance leaves the job running without cancellation or replay.

Completed image bodies expire one hour after the later of the job deadline and last update. Results, job identities and dispatch receipts remain available for seven days after that same baseline, including for reconnects and idempotency checks. After seven days the old job ID returns `404` and its idempotency key may identify new work. Queued, running and unknown jobs, and any job with an unresolved physical attempt, are never removed by storage maintenance.

A bounded sweep runs at startup and every 60 seconds on a worker thread. Under capacity pressure, completed payloads outside the 15-minute deduplication window are transactionally compressed before expiry. SQLite reuses freed pages; maintenance does not run `VACUUM` or replace the database. The migration only adds a payload-archive table and preserves existing rows and columns.

`/health` reports `vision_jobs.storage` with hot, reserved and compressed byte counts, retention windows and free-disk headroom. Vision capacity exhaustion sets `vision_jobs.ready=false` and `blocked=true` without marking a healthy text API unavailable. Admission reserves UTF-8 payload, result and metadata capacity, and dispatch checks capacity again before contacting the engine.

| Configuration key | Package default | Constraint |
|---|---|---|
| `media_concurrency` | `2` | `1` or `2`; supplied to the job implementation. |
| `vision_effort` | `medium` | `medium` or `max`; supplied as the vision reasoning default. |
| `vision_thinking_tokens` | `3072` | Positive integer thinking-budget setting. |
| `vision_final_tokens` | `32768` | Positive integer final-answer budget setting. |

Jobs use a SQLite ledger, `vision.sqlite3`, under `state_dir`. Keep it on private persistent storage with one dispatcher per ledger.

## Credits

Thanks to LakyFx for the imported files listed in `UPSTREAM.json`, licensed under `LICENSES/LakyFx-MIT.txt`, and to Mia's AI Lab for the engine work. See the repository's [credits](../CREDITS.md).
