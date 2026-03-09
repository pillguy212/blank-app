import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime, timedelta, timezone
from io import BytesIO

st.set_page_config(page_title="BTC minute data downloader", layout="wide")

st.title("BTC - données minute par minute (7 derniers jours)")
st.write("Récupération depuis Coinbase Exchange API, puis export Excel ou CSV.")

BASE_URL = "https://api.exchange.coinbase.com/products/{}/candles"


def fetch_coinbase_candles(product_id: str, start_dt: datetime, end_dt: datetime, granularity: int = 60):
    """
    Coinbase retourne les chandelles OHLC.
    On découpe la période en blocs pour récupérer les 7 jours minute par minute.
    """
    all_rows = []

    total_minutes = int((end_dt - start_dt).total_seconds() // 60)
    if total_minutes <= 0:
        return []

    # 300 bougies par appel = très stable pour éviter les problèmes
    chunk_size_minutes = 300

    progress = st.progress(0)
    status = st.empty()

    current_start = start_dt
    fetched = 0

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    while current_start < end_dt:
        current_end = min(current_start + timedelta(minutes=chunk_size_minutes), end_dt)

        params = {
            "start": current_start.isoformat(),
            "end": current_end.isoformat(),
            "granularity": granularity
        }

        url = BASE_URL.format(product_id)

        r = requests.get(url, params=params, headers=headers, timeout=20)
        r.raise_for_status()

        data = r.json()

        if isinstance(data, list) and data:
            all_rows.extend(data)

        fetched += int((current_end - current_start).total_seconds() // 60)
        progress.progress(min(fetched / total_minutes, 1.0))
        status.text(f"Récupération en cours... {fetched} / {total_minutes} minutes")

        current_start = current_end
        time.sleep(0.15)

    progress.progress(1.0)
    status.text("Téléchargement terminé.")

    return all_rows


def candles_to_dataframe(rows):
    """
    Format Coinbase:
    [time, low, high, open, close, volume]
    """
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["timestamp", "low", "high", "open", "close", "volume"])

    # Types numériques
    for col in ["low", "high", "open", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

    # Retirer doublons, trier
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Colonnes pratiques
    df["timestamp_utc"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df["date_utc"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    df["time_utc"] = df["timestamp"].dt.strftime("%H:%M:%S")

    # Heure Québec
    try:
        df["timestamp_montreal"] = df["timestamp"].dt.tz_convert("America/Toronto").dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        df["timestamp_montreal"] = ""

    df = df[
        [
            "timestamp_utc",
            "timestamp_montreal",
            "date_utc",
            "time_utc",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "timestamp",
        ]
    ]

    return df


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="BTC_1m_data")
    return output.getvalue()


col1, col2 = st.columns(2)

with col1:
    product_id = st.text_input("Produit Coinbase", value="BTC-USD")

with col2:
    days = st.number_input("Nombre de jours", min_value=1, max_value=30, value=7, step=1)

if st.button("Récupérer les données"):
    try:
        end_dt = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        start_dt = end_dt - timedelta(days=int(days))

        st.info(
            f"Récupération de {product_id} du "
            f"{start_dt.strftime('%Y-%m-%d %H:%M:%S UTC')} au "
            f"{end_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )

        rows = fetch_coinbase_candles(
            product_id=product_id.strip().upper(),
            start_dt=start_dt,
            end_dt=end_dt,
            granularity=60
        )

        df = candles_to_dataframe(rows)

        if df.empty:
            st.warning("Aucune donnée récupérée.")
        else:
            st.success(f"{len(df)} lignes récupérées.")
            st.dataframe(df, use_container_width=True, height=500)

            excel_bytes = dataframe_to_excel_bytes(df)
            csv_bytes = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Télécharger en Excel",
                data=excel_bytes,
                file_name=f"{product_id.replace('-', '_')}_{days}days_1m.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            st.download_button(
                label="Télécharger en CSV",
                data=csv_bytes,
                file_name=f"{product_id.replace('-', '_')}_{days}days_1m.csv",
                mime="text/csv",
            )

    except requests.HTTPError as e:
        st.error(f"Erreur API Coinbase : {e}")
    except Exception as e:
        st.error(f"Erreur : {e}")
