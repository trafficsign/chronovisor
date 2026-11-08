# Capture hooks

`chronovisor-hook --host codex --event Stop --hook` enqueues `codex-save`.
For Claude Code use `--host claude-code`; this enqueues `claude-code-save`.
Both save raw transcripts without model calls, page generation, or follow-ups.
Use `--dry-run --format json` to inspect the task without enqueuing it.

`UserPromptSubmit` is a no-op: Codex receives `{}`, Claude Code receives no
output. Existing prompt-hook registrations can remain installed.

`chronovisor-background-jobs retry --limit 8` runs only Codex/Claude Code save
modules. Queued retired modules become terminal with reason `retired_module`.
`chronovisor-session-sweeper --limit 8` recovers idle user-session tails. The
capture wrapper invokes these commands in that order every 120 seconds when
its launchd template is installed by the operator.
