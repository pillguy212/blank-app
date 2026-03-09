import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime, timedelta, timezone
from io import BytesIO

st.set_page_config(page_title="BTC minute data downloader", layout="wide")

st.title("BTC - données minute par minute (7 derniers jours)")
st.write("Télécharge les données BTCUSDT à chaque minute et exporte en Excel.")

BASE_URL = "https://api.binance.com/api/v3/klines"


def fetch_binance_klines(symbol: str, interval: str, start_ms: int, end_ms: int):
    """
    Récupère toutes les bougies Binance entre start_ms et end_ms.
    Binance limite le nombre de klines par appel, donc on boucle.
    """
    all_rows = []
    current_start = start_ms
    limit = 1000  # Binance supporte startTime, endTime et limit sur klines

    progress = st.progress(0)
    status = st.empty()

    estimated_total_minutes = max(1, int((end_ms - start_ms) / 60000))
    fetched_minutes = 0

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": limit,
        }

        r = requests.get(BASE_URL, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()

        if not data:
            break

        all_rows.extend(data)

        last_open_time = data[-1][0]
        next_start = last_open_time + 60_000  # +1 minute

        newly_fetched = len(data)
        fetched_minutes += newly_fetched

        progress_ratio = min(fetched_minutes / estimated_total_minutes, 1.0)
        progress.progress(progress_ratio)
        status.text(f"Récupération en cours... {fetched_minutes} lignes")

        if next_start <= current_start:
            break

        current_start = next_start
        time.sleep(0.15)  # petite pause pour être poli avec l'API

    progress.progress(1.0)
    status.text(f"Terminé. {len(all_rows)} lignes récupérées.")
    return all_rows


def klines_to_dataframe(rows):
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    ]

    df = pd.DataFrame(rows, columns=columns)

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["number_of_trades"] = pd.to_numeric(df["number_of_trades"], errors="coerce")

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    # Colonnes pratiques pour Excel / lecture humaine
    df["date_utc"] = df["open_time"].dt.strftime("%Y-%m-%d")
    df["time_utc"] = df["open_time"].dt.strftime("%H:%M:%S")
    df["timestamp_utc"] = df["open_time"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Réorganiser les colonnes
    df = df[
        [
            "timestamp_utc",
            "date_utc",
            "time_utc",
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "number_of_trades",
            "quote_asset_volume",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "close_time",
        ]
    ]

    return df


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="BTC_1m_data")
    return output.getvalue()


col1, col2, col3 = st.columns(3)

with col1:
    symbol = st.text_input("Symbole Binance", value="BTCUSDT")

with col2:
    days = st.number_input("Nombre de jours", min_value=1, max_value=30, value=7, step=1)

with col3:
    interval = st.selectbox("Intervalle", options=["1m"], index=0)

if st.button("Récupérer les données"):
    try:
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=int(days))

        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        st.info(
            f"Récupération de {symbol} de {start_dt.strftime('%Y-%m-%d %H:%M:%S UTC')} "
            f"à {end_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )

        rows = fetch_binance_klines(
            symbol=symbol.strip().upper(),
            interval=interval,
            start_ms=start_ms,
            end_ms=end_ms,
        )

        if not rows:
            st.warning("Aucune donnée récupérée.")
        else:
            df = klines_to_dataframe(rows)

            st.success(f"{len(df)} lignes récupérées.")
            st.dataframe(df, use_container_width=True, height=500)

            excel_bytes = dataframe_to_excel_bytes(df)
            csv_bytes = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Télécharger en Excel",
                data=excel_bytes,
                file_name=f"{symbol.upper()}_{days}days_1m.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            st.download_button(
                label="Télécharger en CSV",
                data=csv_bytes,
                file_name=f"{symbol.upper()}_{days}days_1m.csv",
                mime="text/csv",
            )

    except requests.HTTPError as e:
        st.error(f"Erreur API Binance : {e}")
    except Exception as e:
        st.error(f"Erreur : {e}")
