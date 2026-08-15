# ADR 0003: Separate MONAI Bundle, Label, and Deploy responsibilities

- Status: accepted
- Date: 2026-08-15

## Decision

Use versioned MONAI Bundles to reconstruct model training/inference, MONAI Label with 3D Slicer/ITK-SNAP for expert correction and active learning, and MONAI Deploy for later inference-application interoperability. Pin released versions/tags, not the moving MONAI `dev` branch or `latest` container tag.

## Consequences

Annotation, reproducible model packaging, and deployment are independently testable. A Deploy application is not presented as a medical device, and a Bundle does not replace a data/model card or validation report.
