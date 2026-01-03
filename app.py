import streamlit as st
import pandas as pd
from engine import compute_edges

st.set_page_config(page_title="NCAAB Edge Finder", layout="wide")

st.title("NCAAB Spread Edge Finder")
st.caption("SRS model plus consensus Vegas spread, strict matching, exportable results")

with st.sidebar:
    st.header("Settings")

    api_key = st.secrets.get("ODDS_API_KEY", "")
    if not api_key:
        api_key = st.text_input("Odds API key", type="password")

    home_court_adv = st.number_input("Home court advantage", value=3.25, min_value=0.0, max_value=7.0, step=0.25)
    edge_threshold = st.number_input("Edge threshold", value=3.0, min_value=0.5, max_value=15.0, step=0.5)
    fuzzy_min_score = st.slider("Name match strictness", min_value=85, max_value=99, value=93)
    min_books = st.slider("Minimum books required", min_value=1, max_value=10, value=3)
    max_rows = st.slider("Max rows shown", min_value=10, max_value=200, value=40)

    run_btn = st.button("Run")

if run_btn:
    if not api_key:
        st.error("Add your API key in sidebar, or set ODDS_API_KEY in Streamlit secrets.")
        st.stop()

    with st.spinner("Pulling ratings and odds, computing edges"):
        try:
            df = compute_edges(
                api_key=api_key,
                home_court_adv=float(home_court_adv),
                edge_threshold=float(edge_threshold),
                fuzzy_min_score=int(fuzzy_min_score),
                min_books=int(min_books),
            )
        except Exception as e:
            st.error(f"Failed: {e}")
            st.stop()

    if df.empty:
        st.warning("No results. Try lowering minimum books or lowering edge threshold.")
        st.stop()

    picks_only = st.toggle("Show picks only", value=True)
    if picks_only:
        df_view = df[df["Pick"] != ""].copy()
    else:
        df_view = df.copy()

    st.subheader("Edges")
    st.dataframe(df_view.head(int(max_rows)), use_container_width=True)

    csv = df_view.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="ncaab_edges.csv", mime="text/csv")

    st.subheader("Quick summary")
    st.write(f"Games analyzed: {len(df)}, picks: {(df['Pick'] != '').sum()}")

else:
    st.info("Set your settings, then click Run.")
