import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Polymarket Scanner", layout="wide")
st.title("Polymarket Arbitrage Scanner")

API_URL = "https://gamma-api.polymarket.com/markets"

@st.cache_data
def fetch_markets():
    r = requests.get(API_URL, timeout=20)
    r.raise_for_status()
    return r.json()

try:
    markets = fetch_markets()
except Exception as e:
    st.error(f"Erreur API : {e}")
    st.stop()

data = []

for m in markets:
    prices = m.get("outcomePrices")

    if not prices:
        continue

    # Gère les cas où outcomePrices est une string ou une liste
    try:
        if isinstance(prices, str):
            prices = prices.strip("[]")
            prices = [float(p.strip().replace('"', "")) for p in prices.split(",") if p.strip()]
        elif isinstance(prices, list):
            prices = [float(p) for p in prices]
        else:
            continue
    except Exception:
        continue

    if len(prices) == 0:
        continue

    sum_prob = sum(prices)

    data.append({
        "Market": m.get("question", "N/A"),
        "Prices": prices,
        "Sum Prob": round(sum_prob, 4),
        "Edge": round(1 - sum_prob, 4),
        "Slug": m.get("slug", ""),
        "URL": f"https://polymarket.com/event/{m.get('slug', '')}"
    })

df = pd.DataFrame(data)

st.write("Total markets scanned:", len(df))

if df.empty:
    st.warning("Aucune donnée exploitable trouvée. L’API répond, mais aucun marché n’a pu être parsé avec le format actuel.")
    st.write("Exemple brut du premier marché reçu :")
    if isinstance(markets, list) and len(markets) > 0:
        st.json(markets[0])
    else:
        st.write(markets)
else:
    st.dataframe(df.sort_values("Edge", ascending=False), use_container_width=True)
