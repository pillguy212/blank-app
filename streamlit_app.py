import math
import time
from io import BytesIO
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="BTC 5m analyzer", layout="wide")

st.title("BTC 5-minute analyzer")
st.caption("Télécharge les données BTC automatiquement, puis analyse les probabilités de continuation et reversal.")


# =========================
# CONFIG
# =========================
COINBASE_CANDLES_URL = "https://api.exchange.coinbase.com/products/{product_id}/candles"


# =========================
# DATA FETCH
# =========================
@st.cache_data(show_spinner=False, ttl=300)
def fetch_coinbase_candles(product_id: str, days: int, granularity: int = 60) -> pd.DataFrame:
    """
    Télécharge les candles Coinbase en 1 minute sur X jours.
    Coinbase retourne: [time, low, high, open, close, volume]
    """
    end_dt = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=days)

    # Blocs de 300 minutes pour être prudent
    chunk_minutes = 300
    current_start = start_dt
    all_rows = []

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }

    while current_start < end_dt:
        current_end = min(current_start + timedelta(minutes=chunk_minutes), end_dt)

        params = {
            "start": current_start.isoformat(),
            "end": current_end.isoformat(),
            "granularity": granularity,
        }

        url = COINBASE_CANDLES_URL.format(product_id=product_id)
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()

        data = response.json()
        if isinstance(data, list) and data:
            all_rows.extend(data)

        current_start = current_end
        time.sleep(0.12)

    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "low", "high", "open", "close", "volume"])

    df = pd.DataFrame(all_rows, columns=["timestamp", "low", "high", "open", "close", "volume"])

    for col in ["low", "high", "open", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.dropna(subset=["timestamp", "close"]).drop_duplicates(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Reindex minute par minute pour continuité
    full_range = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="min", tz="UTC")
    df = df.set_index("timestamp").reindex(full_range).rename_axis("timestamp").reset_index()

    # Remplissage simple
    for col in ["low", "high", "open", "close", "volume"]:
        if col == "volume":
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].ffill()

    df["timestamp_utc"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df["timestamp_et"] = df["timestamp"].dt.tz_convert("America/Toronto").dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


# =========================
# ANALYSIS
# =========================
def build_windows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fenêtres glissantes de 5 minutes:
    t0 = référence
    t+1, t+2, t+3, t+4 = états intermédiaires
    t+5 = résultat final
    """
    x = df[["timestamp", "close"]].copy()
    x = x.rename(columns={"close": "price"})

    x["ref_time"] = x["timestamp"]
    x["ref_price"] = x["price"]

    for i in range(1, 6):
        x[f"price_t{i}"] = x["price"].shift(-i)
        x[f"time_t{i}"] = x["timestamp"].shift(-i)

    x = x.dropna(subset=[f"price_t{i}" for i in range(1, 6)]).copy()

    # deltas vs ref
    for i in range(1, 6):
        x[f"delta_t{i}"] = (x[f"price_t{i}"] - x["ref_price"]) / x["ref_price"]
        x[f"delta_bps_t{i}"] = x[f"delta_t{i}"] * 10000

    # états intermédiaires
    for i in range(1, 5):
        x[f"state_t{i}"] = np.where(
            x[f"price_t{i}"] > x["ref_price"],
            "above",
            np.where(x[f"price_t{i}"] < x["ref_price"], "below", "flat")
        )

    # résultat final façon Polymarket
    # Up si final >= ref, sinon Down
    x["final_state"] = np.where(x["price_t5"] >= x["ref_price"], "up", "down")
    x["final_up"] = (x["final_state"] == "up").astype(int)
    x["final_down"] = (x["final_state"] == "down").astype(int)

    # stats intra-fenêtre
    x["max_delta_bps_1to4"] = x[[f"delta_bps_t{i}" for i in range(1, 5)]].max(axis=1)
    x["min_delta_bps_1to4"] = x[[f"delta_bps_t{i}" for i in range(1, 5)]].min(axis=1)
    x["range_bps_1to4"] = x["max_delta_bps_1to4"] - x["min_delta_bps_1to4"]

    # momentum simple
    x["mom_1_to_4_bps"] = ((x["price_t4"] - x["price_t1"]) / x["ref_price"]) * 10000
    x["mom_3_to_4_bps"] = ((x["price_t4"] - x["price_t3"]) / x["ref_price"]) * 10000

    return x.reset_index(drop=True)


def summarize_by_checkpoint(windows: pd.DataFrame, checkpoint: int) -> pd.DataFrame:
    state_col = f"state_t{checkpoint}"
    delta_col = f"delta_bps_t{checkpoint}"

    rows = []

    for side in ["above", "below"]:
        subset = windows[windows[state_col] == side].copy()
        n = len(subset)

        if n == 0:
            rows.append({
                "checkpoint": f"t+{checkpoint}",
                "side_now": side,
                "n": 0,
                "avg_delta_bps": np.nan,
                "prob_final_up": np.nan,
                "prob_final_down": np.nan,
                "prob_continue_same_side": np.nan,
                "prob_reverse": np.nan,
            })
            continue

        if side == "above":
            prob_continue = (subset["final_state"] == "up").mean()
            prob_reverse = (subset["final_state"] == "down").mean()
        else:
            prob_continue = (subset["final_state"] == "down").mean()
            prob_reverse = (subset["final_state"] == "up").mean()

        rows.append({
            "checkpoint": f"t+{checkpoint}",
            "side_now": side,
            "n": n,
            "avg_delta_bps": subset[delta_col].mean(),
            "prob_final_up": subset["final_up"].mean(),
            "prob_final_down": subset["final_down"].mean(),
            "prob_continue_same_side": prob_continue,
            "prob_reverse": prob_reverse,
        })

    return pd.DataFrame(rows)


def add_buckets(windows: pd.DataFrame, checkpoint: int, bucket_edges_bps: list[float]) -> pd.DataFrame:
    state_col = f"state_t{checkpoint}"
    delta_col = f"delta_bps_t{checkpoint}"

    subset = windows[windows[state_col].isin(["above", "below"])].copy()
    if subset.empty:
        return pd.DataFrame()

    # bucket sur valeur absolue du delta
    subset["abs_delta_bps"] = subset[delta_col].abs()

    labels = []
    for i in range(len(bucket_edges_bps) - 1):
        labels.append(f"{bucket_edges_bps[i]:.1f} to {bucket_edges_bps[i+1]:.1f}")

    subset["bucket_abs_bps"] = pd.cut(
        subset["abs_delta_bps"],
        bins=bucket_edges_bps,
        labels=labels,
        include_lowest=True,
        right=False
    )

    grouped_rows = []

    for side in ["above", "below"]:
        side_df = subset[subset[state_col] == side].copy()

        if side_df.empty:
            continue

        grouped = side_df.groupby("bucket_abs_bps", observed=False)
        for bucket, g in grouped:
            if len(g) == 0:
                continue

            if side == "above":
                prob_continue = (g["final_state"] == "up").mean()
                prob_reverse = (g["final_state"] == "down").mean()
            else:
                prob_continue = (g["final_state"] == "down").mean()
                prob_reverse = (g["final_state"] == "up").mean()

            grouped_rows.append({
                "checkpoint": f"t+{checkpoint}",
                "side_now": side,
                "bucket_abs_bps": str(bucket),
                "n": len(g),
                "avg_abs_delta_bps": g["abs_delta_bps"].mean(),
                "avg_signed_delta_bps": g[delta_col].mean(),
                "prob_final_up": g["final_up"].mean(),
                "prob_final_down": g["final_down"].mean(),
                "prob_continue_same_side": prob_continue,
                "prob_reverse": prob_reverse,
            })

    out = pd.DataFrame(grouped_rows)
    if not out.empty:
        out = out.sort_values(["checkpoint", "side_now", "avg_abs_delta_bps"]).reset_index(drop=True)
    return out


def model_edge_table(bucket_df: pd.DataFrame, checkpoint: int, side: str, market_prob: float) -> pd.DataFrame:
    """
    Compare la proba implicite du bet avec la proba historique bucketisée.
    """
    if bucket_df.empty:
        return pd.DataFrame()

    x = bucket_df[
        (bucket_df["checkpoint"] == f"t+{checkpoint}") &
        (bucket_df["side_now"] == side)
    ].copy()

    if x.empty:
        return x

    x["market_prob_input"] = market_prob

    if side == "above":
        # Si on pense bet "up"
        x["historical_prob_target"] = x["prob_final_up"]
        x["target_side"] = "up"
    else:
        # Si on pense bet "down"
        x["historical_prob_target"] = x["prob_final_down"]
        x["target_side"] = "down"

    x["edge_vs_market"] = x["historical_prob_target"] - x["market_prob_input"]
    return x.sort_values("avg_abs_delta_bps").reset_index(drop=True)


def to_excel_bytes(dataframes: dict[str, pd.DataFrame]) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df in dataframes.items():
            safe_name = sheet_name[:31]
            df.to_excel(writer, index=False, sheet_name=safe_name)
    return output.getvalue()


# =========================
# UI
# =========================
with st.sidebar:
    st.header("Paramètres")
    product_id = st.text_input("Produit Coinbase", value="BTC-USD")
    days = st.slider("Nombre de jours", min_value=3, max_value=30, value=7, step=1)
    run_button = st.button("Télécharger + analyser", type="primary")

    st.markdown("---")
    st.subheader("Simulation edge vs marché")
    chosen_checkpoint = st.selectbox("Checkpoint à comparer", options=[1, 2, 3, 4], index=3)
    chosen_side = st.selectbox("Côté observé maintenant", options=["above", "below"], index=0)
    market_prob = st.slider("Probabilité implicite du marché", min_value=0.01, max_value=0.99, value=0.55, step=0.01)

    st.markdown("---")
    st.caption("Exemple: si à t+4 le prix est déjà above, mets ici l'odds live du côté Up.")

if run_button:
    try:
        with st.spinner("Téléchargement des données BTC..."):
            raw_df = fetch_coinbase_candles(product_id=product_id.strip().upper(), days=int(days), granularity=60)

        if raw_df.empty:
            st.error("Aucune donnée récupérée.")
            st.stop()

        with st.spinner("Construction des fenêtres 5 minutes..."):
            windows = build_windows(raw_df)

        # Résumés checkpoint
        checkpoint_tables = []
        for cp in [1, 2, 3, 4]:
            checkpoint_tables.append(summarize_by_checkpoint(windows, cp))
        summary_df = pd.concat(checkpoint_tables, ignore_index=True)

        # Buckets en bps
        bucket_edges_bps = [0, 1, 2, 3, 5, 7.5, 10, 15, 25, 50, 100, 1000]
        bucket_tables = []
        for cp in [1, 2, 3, 4]:
            bucket_tables.append(add_buckets(windows, cp, bucket_edges_bps))
        bucket_df = pd.concat(bucket_tables, ignore_index=True) if bucket_tables else pd.DataFrame()

        edge_df = model_edge_table(bucket_df, chosen_checkpoint, chosen_side, market_prob)

        # KPI
        st.subheader("Résumé")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Minutes téléchargées", f"{len(raw_df):,}")
        c2.metric("Fenêtres 5 min", f"{len(windows):,}")
        c3.metric("Début", raw_df["timestamp_et"].iloc[0])
        c4.metric("Fin", raw_df["timestamp_et"].iloc[-1])

        # Aperçu data
        st.subheader("Données BTC téléchargées")
        st.dataframe(
            raw_df[["timestamp_utc", "timestamp_et", "open", "high", "low", "close", "volume"]],
            use_container_width=True,
            height=260
        )

        # Aperçu fenêtres
        st.subheader("Fenêtres glissantes de 5 minutes")
        preview_cols = [
            "ref_time", "ref_price",
            "price_t1", "price_t2", "price_t3", "price_t4", "price_t5",
            "delta_bps_t1", "delta_bps_t2", "delta_bps_t3", "delta_bps_t4", "delta_bps_t5",
            "state_t1", "state_t2", "state_t3", "state_t4", "final_state"
        ]
        preview_df = windows[preview_cols].copy()
        st.dataframe(preview_df.head(500), use_container_width=True, height=320)

        # Stats globales
        st.subheader("Probabilités globales par checkpoint")
        pretty_summary = summary_df.copy()
        for col in ["avg_delta_bps", "prob_final_up", "prob_final_down", "prob_continue_same_side", "prob_reverse"]:
            pretty_summary[col] = pretty_summary[col].round(4)
        st.dataframe(pretty_summary, use_container_width=True)

        # Buckets
        st.subheader("Buckets par amplitude du move (en bps)")
        if bucket_df.empty:
            st.warning("Aucun bucket disponible.")
        else:
            pretty_bucket = bucket_df.copy()
            for col in [
                "avg_abs_delta_bps", "avg_signed_delta_bps",
                "prob_final_up", "prob_final_down",
                "prob_continue_same_side", "prob_reverse"
            ]:
                pretty_bucket[col] = pretty_bucket[col].round(4)
            st.dataframe(pretty_bucket, use_container_width=True, height=360)

        # Edge table
        st.subheader("Comparaison avec l'odds du marché")
        if edge_df.empty:
            st.info("Pas assez de données pour cette combinaison.")
        else:
            pretty_edge = edge_df.copy()
            for col in ["market_prob_input", "historical_prob_target", "edge_vs_market"]:
                pretty_edge[col] = pretty_edge[col].round(4)
            st.dataframe(pretty_edge, use_container_width=True)

        # Graphique
        st.subheader("Graphique de continuation par bucket")
        chart_source = bucket_df[
            (bucket_df["checkpoint"] == f"t+{chosen_checkpoint}") &
            (bucket_df["side_now"] == chosen_side)
        ].copy()

        if not chart_source.empty:
            fig = plt.figure(figsize=(10, 5))
            plt.plot(chart_source["bucket_abs_bps"], chart_source["prob_continue_same_side"], marker="o")
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("Probabilité de continuer")
            plt.xlabel("Bucket abs(delta) en bps")
            plt.title(f"Continuation historique | {chosen_side} à t+{chosen_checkpoint}")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Pas de graphique disponible pour ce filtre.")

        # Téléchargement Excel
        excel_bytes = to_excel_bytes({
            "btc_raw_data": raw_df,
            "windows_5m": windows,
            "summary_by_checkpoint": summary_df,
            "bucket_analysis": bucket_df,
            "edge_vs_market": edge_df if not edge_df.empty else pd.DataFrame(),
        })

        st.download_button(
            label="Télécharger le rapport Excel",
            data=excel_bytes,
            file_name=f"btc_5m_analysis_{product_id.replace('-', '_')}_{days}d.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except requests.HTTPError as e:
        st.error(f"Erreur API Coinbase : {e}")
    except Exception as e:
        st.error(f"Erreur : {e}")

else:
    st.info("Choisis tes paramètres à gauche puis clique sur « Télécharger + analyser ».")

    st.markdown(
        """
        ### Ce que l'app calcule
        - **above à t+1 / t+2 / t+3 / t+4** : le prix est déjà au-dessus de la référence
        - **below à t+1 / t+2 / t+3 / t+4** : le prix est déjà en-dessous de la référence
        - **prob_continue_same_side** :
          - si `above`, probabilité de finir **up**
          - si `below`, probabilité de finir **down**
        - **prob_reverse** :
          - si `above`, probabilité de finir **down**
          - si `below`, probabilité de finir **up**
        - **buckets en bps** : amplitude du move avant la résolution
        """
    )
