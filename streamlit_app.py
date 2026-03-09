import streamlit as st
import requests
import pandas as pd

st.title("Polymarket Arbitrage Scanner")

API_URL = "https://gamma-api.polymarket.com/markets"

@st.cache_data
def fetch_markets():
    r = requests.get(API_URL)
    return r.json()

markets = fetch_markets()

data = []

for m in markets:

    prices = m.get("outcomePrices")

    if not prices:
        continue

    try:
        prices = [float(p) for p in prices]
    except:
        continue

    sum_prob = sum(prices)

    data.append({
        "Market": m.get("question"),
        "Prices": prices,
        "Sum Prob": sum_prob,
        "Edge": round(1 - sum_prob,4),
        "URL": f"https://polymarket.com/event/{m.get('slug')}"
    })

df = pd.DataFrame(data)

st.write("Total markets scanned:", len(df))

st.dataframe(df.sort_values("Edge", ascending=False))
