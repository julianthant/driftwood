"""Split one day's target-book return across the signals that were on."""

from driftwood.attribution.allocate import attribute_return
from driftwood.attribution.report import write_attribution_report
from driftwood.attribution.study import Attribution, attribute_store

__all__ = [
    "Attribution",
    "attribute_return",
    "attribute_store",
    "write_attribution_report",
]
