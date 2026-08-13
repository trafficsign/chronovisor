# OSS v1 Security Review

Reviewed: 2026-08-11

## Decision

The reviewed OSS v1 source and test boundaries have no identified unresolved
high-severity finding, secret leak, authentication bypass, arbitrary file
access, SSRF, or command-execution path. Release remains fail-closed: the exact
release commit must pass the tests and security workflow below, and any such
finding blocks publication.

This is a source-and-test review, not a claim that local compromise is contained.
The security boundary excludes arbitrary code running as the same OS user,
root/administrator or host compromise, and inspection of live Python/model
process memory.

## Boundary evidence

| Boundary | Reviewed control and evidence |
| --- | --- |
| Dashboard access | The default label remains loopback-only with exact Host and same-origin browser checks. The separate opt-in LAN label requires an explicit private IPv4 bind, TLS, scrypt credentials in an owner-owned `0600` regular file, failed-login rate limiting, bounded in-memory Secure sessions, and exact HTTPS Host/Origin/WebSocket checks. URL tokens and wildcard binds are rejected. `tests/test_dashboard.py::test_dashboard_private_client_scope_rejects_public_addresses`, `::test_dashboard_host_allowlist_uses_loopback_names_and_actual_port`, `::test_dashboard_http_origin_is_same_origin_or_absent`, `::test_dashboard_websocket_requires_same_origin`, `::test_dashboard_loopback_default_stays_unauthenticated_and_lan_share_disabled`, `::test_dashboard_lan_requires_basic_then_uses_secure_bounded_session`, and `::test_dashboard_lan_websocket_accepts_basic_only_on_exact_https_origin`. |
| Traversal and arbitrary file access | Page reads/writes and static assets reject traversal and symlink escapes; write paths recheck containment at apply time. `tests/test_store_security.py`, `tests/test_page_write.py::test_writer_rejects_canonical_parent_symlink_escape`, `::test_writer_rechecks_parent_symlink_under_apply_lock`, and `tests/test_dashboard.py::test_dashboard_static_path_resolver_rejects_directory_and_symlink_escapes`. |
| SSRF and credential forwarding | Research fetch rejects local/mixed DNS targets and pins the validated public address. Provider endpoints reject insecure/special targets, bind credentials to one origin, reject authority/auth headers, and do not follow redirects. `tests/test_research_security.py`, `tests/test_llm_security.py::test_cloud_endpoint_policy_rejects_local_and_special_ip_literals`, `::test_transport_injects_auth_only_at_wire_and_never_follows_redirects`, and `::test_transport_rejects_cross_origin_and_caller_auth_before_sender_call`. |
| Subprocess and command execution | Child environments are allowlisted, frontier repair requires a validated incident/reproduction command, and disabled or malformed repair work stops before process creation. The static command below enumerates every process call and makes shell-string use visible during review. `tests/test_llm_security.py::test_child_process_env_is_allowlisted_and_drops_arbitrary_canaries`, `tests/test_frontier_review.py::test_run_codex_missing_auth_stops_before_subprocess`, `::test_missing_reproduction_command_is_rejected_before_baseline_or_process`, and `::test_disabled_repair_lane_starts_no_process`. |
| Markdown and DOM XSS | Cortex excludes page bodies instead of rendering Markdown. A headless-browser test serves the production `cortex.html` and scripts, supplies hostile graph API `title`, `tag`, and `packageName` values, then clicks the real package/module DOM to drive `initializeGraph` and `renderPanel` through `panel.innerHTML`. It asserts exact text/attribute values, no injected `script`/`img`/`svg` or inline-handler nodes, and an untriggered execution canary. `tests/test_cortex.py::test_build_cortex_graph_uses_local_wiki_without_exposing_bodies` and `::test_cortex_hostile_graph_strings_are_escaped_before_html_sinks`. A new HTML sink or Markdown renderer reopens review. |
| Raw/system confidentiality and permissions | Reads are confined to `pages/` and `system/`; raw segments, activity state, locks, and migration artifacts reject symlinks and use private modes. `tests/test_store_security.py`, `tests/test_raw_segment.py::test_append_creates_and_corrects_private_segment_files`, `tests/test_activity_log.py::test_activity_rejects_unsafe_message_and_symlink_leaf`, `::test_activity_delta_reader_rejects_symlink_parent`, and `tests/test_okf_writer_lock.py::test_writer_lock_rejects_symlink_leaf_and_absent_exclusive_root`. |
| Cloud egress | Remote execution denies restricted `raw`/`system`/high-sensitivity data before backend calls and does not fall back to another provider. `tests/test_llm_runtime.py::test_remote_default_denies_restricted_data_before_backend_call`, `::test_remote_denial_covers_all_capabilities_without_fallback`, `tests/test_ingest.py::test_remote_high_egress_denial_has_no_backend_or_provider_fallback`, and `tests/test_recall_answer_adapters.py::test_remote_raw_high_egress_denial_runs_no_backend_or_ollama_control`. |
| Dependencies and supply chain | `uv.lock` freezes Python resolution and workflow actions are commit-pinned. `.github/workflows/security.yml` checks full Git history, the current tree, built wheel/sdist archives, and the hash-locked dependency export. Public repositories additionally run CodeQL; its private-repository skip is not represented as a successful CodeQL scan. GitHub secret-scanning/push-protection settings remain a repository-administration gate, separate from this source review. |

## Reproducible gates

The reviewed boundary command passed 319 tests on 2026-08-11. It emitted one
dependency forward-reference warning and one macOS `fork()` deprecation warning;
no test failed. The Cortex DOM regression requires headless Chrome/Chromium,
and its absence fails rather than skips the gate. Reproduce it with an isolated
data root:

```sh
security_test_root="$(mktemp -d -t chronovisor-security)"
CHRONOVISOR_ROOT="$security_test_root" uv run pytest -q \
  tests/test_cortex.py::test_cortex_hostile_graph_strings_are_escaped_before_html_sinks \
  tests/test_dashboard.py \
  tests/test_store_security.py \
  tests/test_page_write.py \
  tests/test_research_security.py \
  tests/test_llm_security.py \
  tests/test_llm_runtime.py \
  tests/test_raw_segment.py \
  tests/test_activity_log.py \
  tests/test_okf_writer_lock.py
uv run ruff check --no-cache src scripts tests
uv run mypy src
rg -n 'subprocess\.(run|Popen)|os\.system|create_subprocess_shell|shell\s*=\s*True' src scripts
```

The release commit must also pass `.github/workflows/security.yml`, whose
executed gates are equivalent to:

```sh
DIST_DIR="$(mktemp -d -t chronovisor-dist)"
REQUIREMENTS_FILE="$(mktemp -t chronovisor-requirements)"
gitleaks git . --log-opts="--all" --config=.gitleaks.toml --redact --no-banner --no-color
for target in src scripts tests docs examples README.md SECURITY.md pyproject.toml config.toml.example; do
  test ! -e "$target" || gitleaks dir "$target" --config=.gitleaks.toml --redact --no-banner --no-color
done
uv build --no-sources --out-dir "$DIST_DIR"
gitleaks dir "$DIST_DIR" --max-archive-depth=2 --max-decode-depth=2 --config=.gitleaks.toml --redact --no-banner --no-color
uv export --frozen --all-groups --all-extras --no-emit-project --format requirements-txt --quiet --output-file "$REQUIREMENTS_FILE"
uvx --from pip-audit==2.10.1 pip-audit --require-hashes --no-deps --disable-pip --progress-spinner=off --desc=off --requirement "$REQUIREMENTS_FILE"
```

Release is blocked by any unresolved high-severity result, known secret,
Dashboard access/authentication bypass, arbitrary file access or traversal,
SSRF or credential-forwarding path, cloud-egress bypass, subprocess/command
injection, or stored/reflected XSS.
