# Security Policy

## Scope

QwenForge is an early-stage research / educational inference engine. No production-hardening or security support level is claimed. The current v0.1.x series is experimental; no long-term support period has been established.

## Reporting vulnerabilities

Please **do not** disclose exploit details, credentials, or sensitive logs in a public issue or pull request.

Prefer GitHub's private vulnerability reporting for this repository if the **Report a vulnerability** option is available under the Security tab. If that option is unavailable, open a public issue with only the title "Private security contact requested" and a non-sensitive summary, asking the maintainer for a secure communication channel. Do not include a proof of concept or exploitable technical details in that issue.

Include the affected version/commit, impact, and minimal reproduction details in your **private** report. The maintainer will assess reports and coordinate fixes and disclosure as capacity allows; no response or remediation deadline is promised.

## Security considerations for users

- Treat model weights and third-party artifacts as untrusted unless their source and integrity have been verified.
- Run benchmarks in isolated environments without production credentials.
- Avoid exposing inference or benchmark services to untrusted networks without independent review.
