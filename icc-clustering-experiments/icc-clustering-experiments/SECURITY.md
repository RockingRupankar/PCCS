# Security Policy

## Supported versions

Security fixes are applied to the latest release on the `main` branch.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability. Use GitHub's private vulnerability reporting
(Security tab, "Report a vulnerability") or e-mail `<CONTACT EMAIL>`. Include a description, steps to reproduce, and the
affected version. You can expect an acknowledgement within 7 days.

## Scope notes

This repository is research code. It reads local CSV files and writes result files; it does not open network connections
or execute untrusted input. Please load only data files you trust, and pin dependency versions when reproducing published results.
