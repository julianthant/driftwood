"""The exploratory notebook is source, so its code cells have to at least compile."""

import json
from pathlib import Path


def test_market_data_notebook_cells_compile() -> None:
    path = Path("notebooks/01_explore_market_data.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) >= 5
    for cell in code_cells:
        source = "".join(cell["source"])
        compile(source, str(path), "exec")
    text = "".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "adj_close" in text
    assert "center=True" in text
    assert "pct_change(21)" in text
