# Generic provider endpoint boundary

Generic provider endpoints are locally owned configuration. Before any request
is built, a credentialed endpoint must be HTTPS and rejects URL userinfo,
`localhost` names, and unspecified, loopback, private, link-local, multicast,
or reserved IP literals. The authenticated transport binds the credential to
the configured canonical origin and never follows redirects; callers cannot
supply authentication, `Host`, or HTTP framing headers.

This is not a DNS-rebinding claim. The runtime deliberately performs no DNS
preflight: a lookup before connection would not bind the eventual socket to
that answer. Operators must therefore treat generic endpoint configuration and
their DNS/proxy path as trusted local administration. A hostile DNS or proxy
after configuration is outside this v1 guard and remains a release-review
threat, not a condition silently made safe by endpoint validation.
