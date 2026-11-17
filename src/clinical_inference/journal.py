"""Typed run journal: every agent step is logged (inputs, decisions, outputs).

The journal is the audit trail the paper's Section VI calls for ("clear
governance structures and audit trails") and the accountability layer for
the deterministic state machine: the LLM never decides a transition, and
every transition is recorded here with its evidence.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class JournalEntry:
    step: int
    timestamp: str
    agent: str
    state: str
    decision: str
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    details: str = ""


@dataclass
class RunJournal:
    run_id: str
    started_at: str
    entries: list[JournalEntry] = field(default_factory=list)

    @classmethod
    def new(cls) -> "RunJournal":
        return cls(run_id=uuid.uuid4().hex[:12], started_at=_now())

    def log(
        self,
        agent: str,
        state: str,
        decision: str,
        inputs: dict | None = None,
        outputs: dict | None = None,
        details: str = "",
    ) -> JournalEntry:
        entry = JournalEntry(
            step=len(self.entries) + 1,
            timestamp=_now(),
            agent=agent,
            state=state,
            decision=decision,
            inputs=inputs or {},
            outputs=outputs or {},
            details=details,
        )
        self.entries.append(entry)
        return entry

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "entries": [asdict(e) for e in self.entries],
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    def report_md(self) -> str:
        lines = [f"# Run report — {self.run_id}", "", f"Started: {self.started_at}", ""]
        for e in self.entries:
            lines.append(f"## Step {e.step}: {e.agent} ({e.state})")
            lines.append(f"- **Decision:** {e.decision}")
            if e.details:
                lines.append(f"- {e.details}")
            if e.inputs:
                lines.append(f"- Inputs: `{json.dumps(e.inputs)}`")
            if e.outputs:
                lines.append(f"- Outputs: `{json.dumps(e.outputs)}`")
            lines.append("")
        return "\n".join(lines)
