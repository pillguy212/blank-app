import json
from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Polymarket Arbitrage Scanner", layout="wide")

GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_jsonish_list(value):
    """
    Converts a Polymarket field that may be:
    - a Python list
    - a JSON string like '["Yes","No"]'
    - a stringified numeric list like '["0.42","0.58"]'
    """
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []

        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

        # fallback if malformed but list-like
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            if not inner:
                return []
            return [x.strip().strip('"').strip("'") for x in inner.split(",")]

    return []


@st.cache_data(ttl=120)
def fetch_events(limit=200, active=True, closed=False, archived=False):
    params = {
        "limit": limit,
        "active": str(active).lower(),
        "closed": str(closed).lower(),
        "archived": str(archived).lower(),
    }
    response = requests.get(GAMMA_EVENTS_URL, params=params, timeout=25)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, list):
        return data
    return []


def normalize_market(event_obj, market_obj):
    question = (
        market_obj.get("question")
        or market_obj.get("title")
        or event_obj.get("title")
        or "N/A"
    )

    slug = market_obj.get("slug") or event_obj.get("slug") or ""
    outcomes_raw = market_obj.get("outcomes")
    prices_raw = market_obj.get("outcomePrices")

    outcomes = parse_jsonish_list(outcomes_raw)
    prices = parse_jsonish_list(prices_raw)
    prices = [safe_float(p) for p in prices]
    prices = [p for p in prices if p is not None]

    if not prices:
        return None

    sum_prob = sum(prices)
    edge = 1 - sum_prob

    volume = safe_float(market_obj.get("volume"))
    liquidity = safe_float(market_obj.get("liquidity"))

    num_outcomes = len(prices)

    if sum_prob < 0.97:
        status = "Potential underpricing"
    elif sum_prob > 1.03:
        status = "Potential overpricing"
    else:
        status = "Near fair"

    return {
        "Market": question,
        "Event": event_obj.get("title", ""),
        "Outcomes": outcomes if outcomes else [],
        "Prices": prices,
        "Num Outcomes": num_outcomes,
        "Sum Prob": round(sum_prob, 4),
        "Edge": round(edge, 4),
        "Status": status,
        "Volume": volume if volume is not None else 0.0,
        "Liquidity": liquidity if liquidity is not None else 0.0,
        "Slug": slug,
        "URL": f"https://polymarket.com/event/{slug}" if slug else "",
        "End Date": market_obj.get("endDate") or event_obj.get("endDate") or "",
        "Active": market_obj.get("active", event_obj.get("active", False)),
        "Closed": market_obj.get("closed", event_obj.get("closed", False)),
    }


st.title("Polymarket Arbitrage Scanner")
st.caption("Educational tool for scanning pricing inconsistencies. Start with paper analysis only.")

with st.sidebar:
    st.header("Filters")
    limit = st.slider("Events to fetch", min_value=50, max_value=1000, value=200, step=50)
    min_volume = st.number_input("Minimum volume", min_value=0.0, value=0.0, step=1000.0)
    min_liquidity = st.number_input("Minimum liquidity", min_value=0.0, value=0.0, step=1000.0)
    under_threshold = st.number_input("Underpricing threshold", min_value=0.0, max_value=2.0, value=0.97, step=0.01)
    over_threshold = st.number_input("Overpricing threshold", min_value=0.0, max_value=2.0, value=1.03, step=0.01)
    keyword = st.text_input("Keyword search", value="")
    only_flagged = st.checkbox("Show only flagged opportunities", value=False)
    refresh = st.button("Refresh data")

if refresh:
    st.cache_data.clear()

try:
    events = fetch_events(limit=limit, active=True, closed=False, archived=False)
except Exception as e:
    st.error(f"API error: {e}")
    st.stop()

rows = []

for event_obj in events:
    markets = event_obj.get("markets", [])
    if not isinstance(markets, list):
        continue

    for market_obj in markets:
        if not isinstance(market_obj, dict):
            continue

        row = normalize_market(event_obj, market_obj)
        if row is not None:
            rows.append(row)

df = pd.DataFrame(rows)

if df.empty:
    st.warning("No usable markets found from the current API response.")
    if events:
        st.subheader("Sample raw event")
        st.json(events[0])
    st.stop()

# Apply filters
filtered = df.copy()
filtered = filtered[filtered["Volume"] >= min_volume]
filtered = filtered[filtered["Liquidity"] >= min_liquidity]

if keyword.strip():
    filtered = filtered[filtered["Market"].str.contains(keyword, case=False, na=False)]

filtered["Flagged"] = (
    (filtered["Sum Prob"] < under_threshold) |
    (filtered["Sum Prob"] > over_threshold)
)

if only_flagged:
    filtered = filtered[filtered["Flagged"]]

flagged_count = int(filtered["Flagged"].sum()) if not filtered.empty else 0

filtered = filtered.sort_values(
    by=["Flagged", "Edge"],
    ascending=[False, False]
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Markets scanned", len(df))
c2.metric("After filters", len(filtered))
c3.metric("Flagged", flagged_count)
c4.metric("UTC time", datetime.now(timezone.utc).strftime("%H:%M:%S"))

st.subheader("Scanner results")

display_df = filtered[
    [
        "Market",
        "Outcomes",
        "Prices",
        "Sum Prob",
        "Edge",
        "Status",
        "Volume",
        "Liquidity",
        "URL",
    ]
].copy()

st.dataframe(display_df, use_container_width=True)

csv_data = display_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download CSV",
    data=csv_data,
    file_name="polymarket_scanner.csv",
    mime="text/csv",
)

with st.expander("Debug / first raw event"):
    if events:
        st.json(events[0])
