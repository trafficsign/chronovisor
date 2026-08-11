# Threat Model

## Scope and status

This model covers the Chronovisor process, local data root, host hooks, MCP
server, Dashboard, local model services, research network access, and the
provider-neutral runtime foundation. The OSS v1 review is recorded in
[security-review-v1.md](security-review-v1.md); this model remains the boundary
contract for future changes.

Local-only is the only production-supported topology. Cloud-only and hybrid
topologies are experimental and require explicit egress configuration.

## Assets

- private transcripts in `raw/` and long-term knowledge in `pages/` and
  `system/`;
- credentials and credential references;
- configuration, authority/adoption artifacts, audit evidence, and mutation
  receipts;
- integrity of indexes, decisions, recalls, and stored pages;
- availability of local services and bounded compute budgets.

## Trust boundaries

| Boundary | Trusted side | Untrusted or less-trusted input |
| --- | --- | --- |
| Host/MCP | Chronovisor tool contracts | prompts, transcripts, tool arguments, host metadata |
| Filesystem | contained data-root paths and validated artifacts | filenames, links, imported Markdown, permissions, symlinks |
| Dashboard | loopback server and same-origin browser requests | request paths/headers and stored content rendered in the UI |
| Model runtime | normalized contracts and local policy | model output and remote provider responses |
| Network egress | explicit policy and allowlisted action | URLs, DNS answers, redirects, fetched content, generic endpoints |
| Subprocess | fixed executable/argument contracts | incident evidence, paths, environment, generated repair input |
| Supply chain | reviewed source and lockfile | packages, build actions, release artifacts, model files |

External content and model output are data, never instructions or authority.
Mutations require their normal deterministic and semantic authorization; a
model response alone is not permission to write.

## Threats, current controls, and open work

### Credentials

The runtime foundation parses explicit `CredentialRef` values, keeps resolved
values opaque, validates mounted-file ownership/mode/type/size, binds outbound
credentials to profile ID, HTTPS origin, and authentication scheme, rejects
caller-supplied authority/authentication headers, and disables redirects at the
authenticated transport boundary. Safe errors expose categories, not secret
values.

OS keyring resolution, the credential CLI, provider configuration validation,
and remote adapter wiring are covered by the OSS v1 review. Remote profiles
remain opt-in and fail closed. Plaintext secrets must not be stored in
repository files, `config.toml`, command arguments, logs, Dashboard payloads,
artifacts, or child-process environments.

### Dashboard and XSS

The Dashboard binds only to `localhost` or `127.0.0.1`, rejects non-loopback
clients, and validates Host and Origin. LAN mode is disabled. It has no separate
user authentication boundary, so loopback is not permission to expose it
through a proxy, tunnel, container port, or browser-sharing service.

Stored page names, snippets, errors, and model-derived values are untrusted.
They must reach the DOM through text-safe rendering or explicit escaping. The
current Cortex HTML sinks use explicit escaping; the browser-DOM regression
covers hostile graph API titles, tags, and package names through the production
render path. Any new HTML sink or Markdown renderer reopens the XSS release gate.

### Filesystem

Page lookup and static asset serving perform path containment checks, and
mounted credentials reject symlinks and unsafe locations. The OSS v1 review
covers read, write, import, archive, migration, recovery, and private-file mode
controls. `raw/`, `system/`, configuration, locks, and receipts must remain
protected from other OS users; any new creation or write path reopens review.

### Generic endpoints, SSRF, and cloud egress

Research Web fetch has a separate SSRF-resistant URL guard, bounded fetches,
and an explicit egress switch. The common model runtime denies cloud egress for
`raw`, `system`, and high-sensitivity data by default and performs the decision
before calling a backend.

Generic model-provider endpoints are locally owned configuration. Before any
request is built, a credentialed endpoint must be HTTPS and rejects URL
userinfo, `localhost` names, and unspecified, loopback, private, link-local,
multicast, or reserved IP literals. The authenticated transport binds the
credential to the configured canonical origin, never follows redirects, and
rejects caller-supplied authentication, `Host`, and HTTP framing headers.

This is not a DNS-rebinding claim. The runtime deliberately performs no DNS
preflight: a lookup before connection would not bind the eventual socket to
that answer. Operators must therefore treat generic endpoint configuration and
their DNS/proxy path as trusted local administration. A hostile DNS or proxy
after configuration remains a release-review threat, not a condition silently
made safe by endpoint validation. Provider failure must not silently reroute
data. Cloud eligibility is not consent to send credentials or an entire private
record; only the minimum policy-approved payload may leave the host.

### Subprocess and command execution

Chronovisor launches local workers and contains a guarded exceptional
system-code repair path. Every subprocess boundary must use fixed executables,
argument arrays, bounded input/output/time, a minimal environment, and
contained paths. Shell interpolation, untrusted working directories, inherited
credentials, and repair input becoming executable instructions are in scope for
the pre-release review.

### Supply chain

Python dependencies are locked and CI runs static, architecture, test,
full-history/tree/release-artifact secret scans, and dependency audits. Public
repositories also run CodeQL; private releases require the recorded manual
static review because GitHub code scanning is unavailable to this workflow.
Model-file integrity remains an operator responsibility. A lockfile limits
drift; it does not establish that a dependency or model is trustworthy.

## Honest exclusions

Chronovisor does not claim to protect data or credentials after any of the
following:

- arbitrary code execution as the same OS user;
- root/administrator, kernel, hypervisor, or physical-host compromise;
- reading or instrumenting live Python or model-process memory;
- compromise of the OS credential store, provider account, browser profile, or
  host agent before data reaches Chronovisor;
- deliberate operator configuration that exports the data root, loopback
  Dashboard, or sensitive content outside the documented boundary.

Python secret memory is not guaranteed to be zeroized. Local-only means no
intended provider egress for model work; it does not turn a compromised host
into a trusted one.

## OSS v1 release blockers

Do not publish OSS v1 while any of these remain:

- a known secret in source, tests, docs, examples, Git history, CI logs, or
  release artifacts;
- an unresolved high-severity finding;
- a Dashboard authentication/access-control bypass or remote exposure;
- arbitrary file read/write or path traversal outside the intended data root;
- exploitable SSRF, credential forwarding, cross-origin redirect, or cloud
  egress policy bypass;
- command execution or subprocess injection from untrusted input;
- an exploitable stored/reflected XSS path;
- a failed or missing full-history secret scan, dependency audit, static review,
  or targeted test for the boundaries above.

Credential cleanup follows the
[credential leak response runbook](runbooks/credential-leak-response.md).
