import streamlit as st
from engine import compute_edges, compute_top_bets

st.set_page_config(
    page_title="NCAAB Betting Engine",
    layout="wide",
)

st.title("🏀 NCAAB Betting Engine")
st.caption("SRS-based model with consensus Vegas lines")

# --- API KEY ---
import os

api_key = os.getenv("ODDS_API_KEY")

if not api_key:
    api_key = st.text_input(
        "Odds API Key",
        type="password",
        help="Enter your The Odds API key (not stored)"
    )

if not api_key:
    st.warning("Enter your Odds API key to run the app.")
    st.stop()
    help="Stored as env var ODDS_API_KEY or Streamlit secret",


if not api_key:
    st.warning("Enter your Odds API key to run the app.")
    st.stop()

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("Settings")

    season = st.number_input("Season", value=2026, step=1)
    home_court = st.slider("Home Court Advantage", 0.0, 7.0, 3.25, 0.25)
    min_books = st.slider("Minimum Books", 1, 8, 3)
    edge_threshold = st.slider("Spread Edge Threshold", 1.0, 8.0, 3.0, 0.5)
    moneyline_edge = st.slider("Moneyline Edge Threshold", 0.01, 0.15, 0.04, 0.01)
    top_n = st.slider("Top Bets Count", 5, 30, 10)

    run_btn = st.button("Run Engine 🚀")

# --- RUN ---
if run_btn:
    with st.spinner("Fetching odds and computing edges..."):
        edges = compute_edges(
            api_key=api_key,
            season_year=season,
            home_court_adv=home_court,
            edge_threshold=edge_threshold,
            min_books=min_books,
        )

        top_bets = compute_top_bets(
            api_key=api_key,
            season_year=season,
            home_court_adv=home_court,
            spread_edge_threshold=edge_threshold,
            moneyline_edge_threshold=moneyline_edge,
            min_books=min_books,
            top_n=top_n,
        )

    st.subheader("📈 Spread Edges")
    if edges.empty:
        st.info("No spread edges found with current settings.")
    else:
        st.dataframe(edges, use_container_width=True)

    st.subheader("🔥 Top Bets")
    if top_bets.empty:
        st.info("No top bets found with current settings.")
    else:
        st.dataframe(top_bets, use_container_width=True)
