# Security policy

## Reporting a vulnerability or data incident

Do not open a public issue containing participant data, credentials, file paths with identifiers, or protected logs. Contact the repository security maintainers and institutional data steward through the private channel registered for the project. Include only the minimum non-sensitive reproduction information initially.

For suspected participant-data transmission, follow the incident procedure in [docs/GOVERNANCE.md](docs/GOVERNANCE.md) immediately. Stop the transfer, preserve audit evidence, revoke the integration, and notify the steward/security contact. Do not paste affected data into an AI assistant or external support ticket.

## Supported versions

Security fixes target the current minor release. Research snapshots are not production services and receive no clinical-availability guarantee.

## Data boundary

CBR participant-level data and derivatives are prohibited from this repository and from public AI/external services. Protected workflows run only in the approved `secure_cbr` environment with network disabled. A passing automated audit does not replace DUA and human export review.
