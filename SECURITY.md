# Security Policy

## Supported versions

Chronovisor has not published a stable release. Security fixes currently target
the latest `main` branch only; older commits and untagged snapshots are not
supported.

| Version | Security support |
| --- | --- |
| Latest `main` | Yes, pre-release |
| Tagged stable releases | None yet |

## Report a vulnerability privately

Use [GitHub private vulnerability reporting](https://github.com/trafficsign/chronovisor/security/advisories/new).
Do not disclose a suspected vulnerability in a public issue, discussion, pull
request, or commit.

Include the affected revision, impact, prerequisites, minimal reproduction,
and any evidence that can be shared safely. Remove credentials, private memory
content, local paths, and other personal data. If GitHub private reporting is
unavailable, do not post details publicly; ask the maintainer for a private
channel without including the vulnerability details.

Response targets, not an SLA:

- acknowledgement within three business days;
- an initial severity and scope assessment within seven business days;
- a status update at least weekly until remediation or closure.

## Credential exposure

If a credential may have been exposed, revoke or rotate it at the provider
immediately. Do this before editing Git history, deleting artifacts, or waiting
for confirmation. Then follow the
[credential leak response runbook](docs/runbooks/credential-leak-response.md).

The current security scope and known exclusions are documented in the
[threat model](docs/threat-model.md).
