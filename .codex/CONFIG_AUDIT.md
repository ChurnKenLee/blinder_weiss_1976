# Codex configuration audit

Audited with Codex CLI 0.153.4 on 2026-09-07.

Fetched with curl from https://developers.openai.com/codex/config-schema.json.
SHA-256: `692da7699367f6f4fbbd46c0021278c1311440bcebf0bcb9b836690c05e56196`.
Read all 96 root settings and 171 definitions, including
nested properties and enums. Lossless read chunks reconstruct the exact schema.
The full downloaded schema is retained as config-schema.json for validation.

The refreshed installed model catalog advertises Astra effort levels through
`ultra`, a `priority` (Fast) tier, and a 272,000-token context window. Config uses
ultra for the primary, planning, and delegated work; priority requests the faster
service tier. This prioritizes capability, not lowest latency or cost. Ultra may
reason longer; priority uses more allowance. Actual entitlement and host settings
still control service availability.

Concurrency is set to the current host's three subordinate slots, with nesting
depth two. Increasing local configuration cannot create extra managed slots.
Compression, unified execution, and shell snapshots are stable and already on
by default; they are pinned explicitly rather than presented as new speed gains.
Low response verbosity, disabled TUI animations, and disabled automatic recaps
reduce avoidable output and UI/background work. Tool output retention is 16,000
tokens to reduce lost diagnostics. Live search provides current documentation.

Automatic approval review and network access remain enabled. Existing managed
permissions are retained. No review thresholds, risk-classifier instructions,
trust acknowledgements, or guardrail-disabling flags are changed. Built-in model
instructions, context-window sizing, compaction, retries, and optional MCP startup
budgets retain their supported defaults. No experimental feature is newly enabled.

Both the project config and this environment's user config validate against the
schema. Existing user trust and TUI metadata are preserved. New sessions load the
settings; a running managed session may retain its original model and policy.
These are supported tuning choices, not a benchmark-proven universal optimum.
