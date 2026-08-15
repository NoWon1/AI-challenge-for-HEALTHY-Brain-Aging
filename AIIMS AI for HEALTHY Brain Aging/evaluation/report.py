"""Validation report assembly."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ValidationReport:
    title: str
    metrics: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", "", "## Metrics"]
        for name, value in self.metrics.items():
            lines.append(f"- {name}: {value:.4f}")
        if self.notes:
            lines.extend(["", "## Notes"])
            lines.extend(f"- {note}" for note in self.notes)
        return "\n".join(lines) + "\n"

