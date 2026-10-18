## Summary

What changed, and which milestone it belongs to.

## Design notes

The decision, the alternative that was set aside, and where that is written up in `docs/DESIGN_DECISIONS.md`.

## How to test

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m pytest -q
```

## Test results

Paste the pytest summary (and the C++ extension import, once the simulator exists). Note anything that was not run, such as a live vendor call.
