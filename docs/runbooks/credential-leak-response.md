# Credential Leak Response

Use this runbook whenever a credential may have appeared in source, Git
history, logs, CI output, a release, a package, an image, or a runtime artifact.
Treat uncertain exposure as real until the provider proves otherwise.

## 1. Revoke or rotate first

1. Revoke the exposed credential at its provider immediately. If revocation is
   unavailable, rotate it and invalidate the old value.
2. Invalidate derived sessions, temporary credentials, or signing material that
   the exposed credential could create.
3. Confirm the old value can no longer authenticate. Do not paste it into a
   shell history, ticket, chat, scanner argument, or test.
4. Record the provider, credential identity, revocation time, owner, and safe
   fingerprint or provider-side ID. Never record the secret itself.

History cleanup is not revocation. Anyone who fetched an old commit or artifact
may retain the value.

## 2. Contain

1. Disable the affected provider profile and outbound workflow. Stop only the
   relevant Chronovisor processes if they may still hold the value in memory.
2. Remove the exposed value from active configuration, environment injection,
   secret mounts, OS credential stores, CI variables, and deployment settings.
3. Replace it with a new credential reference only after scope and binding are
   correct. Do not commit the replacement.
4. Restrict access to affected logs and artifacts while preserving a minimal,
   access-controlled evidence record.
5. Identify every place the value could have propagated: forks, caches,
   mirrors, build outputs, package registries, containers, backups, and
   downstream systems.

## 3. Clean Git history and artifacts

1. Locate the first introduction and every reachable branch, tag, pull request,
   stash/export, and generated file containing the value.
2. Remove it from the current tree and rewrite affected Git history with a
   reviewed history-rewrite tool. Coordinate any force-push and tell
   collaborators to discard contaminated clones and caches.
3. Delete or replace affected GitHub Actions logs/artifacts, releases, packages,
   images, generated documentation, mirrors, and downloadable archives.
4. Ask hosting/provider support to purge server-side caches when repository
   controls cannot remove an exposed object.
5. Preserve only non-secret hashes, timestamps, object IDs, and remediation
   receipts needed for audit.

## 4. Run a full rescan

Scan all of the following, not only the working tree:

- current source, tests, fixtures, docs, examples, and local configuration;
- the full Git object database, every branch/tag/ref, and rewritten history;
- CI logs and artifacts, releases, built distributions, packages, containers,
  mirrors, and deployment bundles;
- runtime logs, research/review artifacts, Dashboard traces, and process launch
  configuration;
- related credentials, using provider audit logs to detect use after exposure.

Run the repository's configured secret, dependency, static, and targeted
security checks. Those scanners are still part of the OSS pre-release work; if
they are not available, the incident remains open rather than being declared
clean from a working-tree search alone.

## 5. Postmortem and closure

Document the timeline, root cause, affected systems/data, access evidence,
revocation proof, cleanup scope, scan evidence, and preventive action. Add the
smallest regression check that would have caught the leak without embedding the
credential. Notify affected users privately when required.

Close the incident only when revocation is verified, propagation paths are
cleaned, the full rescan is clean, replacements are operating through approved
references, and every release blocker is resolved.
