"""Read-only study page.

Run from the repository root:

    streamlit run dashboard/app.py

This file does not construct a broker client and does not send orders.
"""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    import sys

    import streamlit as st

    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)

    from dashboard.views import attribution_table, coverage_table
    from driftwood.attribution import attribute_store
    from driftwood.config import database_path, load_env
    from driftwood.data.store import MarketStore
    from driftwood.models.persist import read_predictions

    load_env()
    st.set_page_config(page_title="Driftwood", layout="wide")
    st.title("Driftwood")
    st.caption("Read only. This page does not send orders.")
    path = Path(st.text_input("Database", value=str(database_path(None))))
    if not path.exists():
        st.info("No database at that path. Run driftwood ingest first.")
        return
    with MarketStore(path) as store:
        bars = store.read_bars()
        st.subheader("Coverage")
        st.dataframe(coverage_table(bars), hide_index=True)
        try:
            predictions = read_predictions(store)
        except RuntimeError:
            st.info("No predictions yet. Run driftwood train.")
            return
        models = sorted(predictions["model"].astype(str).unique())
        model = st.selectbox("Model", models)
        try:
            result = attribute_store(store, model=model)
        except RuntimeError as exc:
            st.warning(str(exc))
            return
    st.subheader(
        f"Attribution {result.signal_date.date()} to {result.next_date.date()}"
    )
    st.write(result.note)
    st.dataframe(attribution_table(result.by_signal), hide_index=True)
    st.caption(
        "The total is the target book's close-to-close return. "
        "It is not the filled equity curve."
    )


if __name__ == "__main__":
    main()
