"""Daily note rendered by this code, not by the model.

Section headers are part of the contract. A test looks for them. It does
not compare the page to a saved copy.
"""

from __future__ import annotations


def render_daily_report(
    *,
    as_of: str,
    context: str,
    proposals: list[dict[str, object]],
    approved_ids: list[str],
    results: list[dict[str, object]],
    notes: list[str],
) -> str:
    approved = [item for item in approved_ids if item]
    lines = [
        "# Daily research note",
        "",
        f"As of: {as_of}",
        "",
        "## Data",
        "",
        _quote(context),
        "",
        "## Proposals",
        "",
    ]
    if proposals:
        for proposal in proposals:
            action = proposal.get("action", "")
            reason = proposal.get("reason", "")
            lines.append(f"- {action}: {reason}".rstrip())
    else:
        lines.append("- None.")
    lines.extend(["", "## Decision", ""])
    if approved:
        lines.append("Approved: " + ", ".join(approved) + ".")
        lines.append("The model did not set these ids. A person did.")
    else:
        lines.append("Approved: none. Nothing was run.")
    lines.extend(["", "## Actions", ""])
    if results:
        for result in results:
            lines.append(
                f"- {result.get('id')}: {result.get('status')}. {result.get('detail', '')}".rstrip()
            )
    else:
        lines.append("- None.")
    if notes:
        lines.extend(["", "Notes:"])
        for note in notes:
            lines.append(f"- {' '.join(str(note).split())}")
    lines.extend(
        [
            "",
            "## Limits",
            "",
            "The model cannot approve its own proposals. Actions run only after a person",
            "sets the approved ids. submit_orders can only use the Alpaca paper host.",
            "This note is not a live trading result.",
            "",
        ]
    )
    return "\n".join(lines)


def _quote(text: str) -> str:
    body = text.strip() or "(empty)"
    return "\n".join(f"> {line}" if line else ">" for line in body.splitlines())
