# app.py
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import numpy as np
import os

# 新增：蠟燭圖套件
import mplfinance as mpf

from stock import fetch_stock_data

# ========== ✅ 你的「載入中文字型」保持不動 ==========
font_path = os.path.join(".streamlit", "fonts", "NotoSansTC-Regular.otf")
if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    st.write(f"✅ 成功載入中文字型：{font_prop.get_name()}")
else:
    font_prop = None
    st.warning("⚠️ 找不到中文字型，圖表中文字可能無法正確顯示。")
# ===================================

st.set_page_config(layout="wide")
st.title("📈 台股技術指標視覺化平台")

# === Sidebar input ===
st.sidebar.header("輸入參數")
stock_no = st.sidebar.text_input("股票代號（如 1303）", value="1303")

col1, col2 = st.sidebar.columns(2)
start_year = col1.number_input("起始年份（西元）", value=2025, step=1)
start_month = col2.selectbox("起始月份", list(range(1, 13)), index=0)

col3, col4 = st.sidebar.columns(2)
end_year = col3.number_input("結束年份（西元）", value=2025, step=1)
end_month = col4.selectbox("結束月份", list(range(1, 13)), index=1)

indicator = st.sidebar.selectbox("技術指標", ["布林帶", "移動平均線", "布林帶 + 移動平均線"])
chart_type = st.sidebar.selectbox("圖表類型", ["蠟燭圖", "折線圖"])  # 可切換

# ✅ 新增：資料頻率
freq_label = st.sidebar.selectbox("資料頻率", ["每天", "每週", "每月", "每小時", "每15分鐘"], index=0)
FREQ_MAP = {"每天": "D", "每週": "W", "每月": "M", "每小時": "H", "每15分鐘": "15min"}
rule = FREQ_MAP[freq_label]

# === 工具函式：計算 RSI / MACD ===
def add_rsi_macd(df: pd.DataFrame) -> pd.DataFrame:
    # RSI(14) - Wilder
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    window = 14
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean().replace(0, 1e-10)
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD(12,26,9)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    return df

# ✅ 新增：OHLCV 重採樣
def resample_ohlcv(df: pd.DataFrame, rule: str) -> tuple[pd.DataFrame, bool]:
    """
    回傳 (resampled_df, used_original)。
    - 若原始資料為日頻，且選擇小時/15分，則回傳 (原 df, True) 並由呼叫端提示 fallback。
    """
    # 檢查是否有時間索引
    if not np.issubdtype(df.index.dtype, np.datetime64):
        if "Date" in df.columns:
            df = df.copy()
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date").sort_index()
        else:
            return df, True  # 沒日期可用，直接返回

    # 判斷原始頻率是否至少日內
    diffs = df.index.to_series().diff().dropna()
    min_step = diffs.min() if not diffs.empty else pd.Timedelta("0s")

    # 如果使用者選「H/15min」但資料最小間距 >= 1 天，就無法向下取樣
    if rule in ("H", "15min") and (min_step >= pd.Timedelta("1D")):
        return df, True  # 使用原始資料（fallback）

    agg = {}
    for col, fn in {"Open": "first", "High": "max", "Low": "min", "Close": "last"}.items():
        if col in df.columns:
            agg[col] = fn
    if "Volume" in df.columns:
        agg["Volume"] = "sum"

    res = df.resample(rule, label="right", closed="right").agg(agg)
    # 砍掉無法形成完整 OHLC 的列
    for c in ("Open", "High", "Low", "Close"):
        if c in res.columns:
            res = res[res[c].notna()]
    return res, False  # 成功重採樣

# === Main action ===
if st.sidebar.button("開始繪製"):
    st.write(f"👉 查詢參數：股票代號={stock_no}, 起={start_year}/{start_month}, 迄={end_year}/{end_month}，頻率={freq_label}")
    st.info("正在抓取資料，請稍候...")
    df = fetch_stock_data(stock_no, start_year, start_month, end_year, end_month)

    if df is None or df.empty:
        st.error("找不到資料，請確認股票代號與時間範圍。")
        st.stop()

    # 先重採樣到指定頻率
    df_resampled, used_original = resample_ohlcv(df, rule)
    if used_original and rule in ("H", "15min"):
        st.warning("⚠️ 原始資料僅有日頻，無法產生每小時/每15分鐘資料，已改用『每天』。")
        rule = "D"  # 更新 rule，僅用於顯示
    df = df_resampled

    # 必要欄位
    needed_cols = {"Open", "High", "Low", "Close"}
    if not needed_cols.issubset(df.columns):
        st.error(f"缺少必要欄位：{needed_cols - set(df.columns)}")
        st.stop()

    has_volume = "Volume" in df.columns

    # 計算指標（以重採樣後的資料為基礎）
    show_ma = ("移動平均線" in indicator) or ("布林帶 + 移動平均線" in indicator)
    show_bb = ("布林帶" in indicator) or ("布林帶 + 移動平均線" in indicator)

    if show_ma:
        df["SMA20"] = df["Close"].rolling(20).mean()
        df["SMA60"] = df["Close"].rolling(60).mean()

    if show_bb:
        df["BB_MA"] = df["Close"].rolling(20).mean()
        df["BB_STD"] = df["Close"].rolling(20).std()
        df["UpperB"] = df["BB_MA"] + 2 * df["BB_STD"]
        df["LowerB"] = df["BB_MA"] - 2 * df["BB_STD"]

    df = add_rsi_macd(df)

    st.success(f"✅ 採用頻率：{freq_label}（實際列數：{len(df)}）")
    with st.expander("查看資料（最後 30 筆）"):
        st.dataframe(df.tail(30), use_container_width=True)

    # =======================
    # 折線圖（深色）＋ Volume 子圖 ＋ RSI/MACD 第三子圖
    # =======================
    if chart_type == "折線圖":
        plt.style.use("dark_background")  # 深色背景

        # 面板數：有 Volume → 三個子圖；沒有 Volume → 兩個子圖（價格 + RSI/MACD）
        if has_volume:
            fig = plt.figure(figsize=(14, 9))
            gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.1)
            ax_price = fig.add_subplot(gs[0, 0])
            ax_vol   = fig.add_subplot(gs[1, 0], sharex=ax_price)
            ax_ind   = fig.add_subplot(gs[2, 0], sharex=ax_price)
        else:
            fig = plt.figure(figsize=(14, 8))
            gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.1)
            ax_price = fig.add_subplot(gs[0, 0])
            ax_ind   = fig.add_subplot(gs[1, 0], sharex=ax_price)
            ax_vol = None

        # 主圖：收盤價 + MA/BB
        ax_price.plot(df.index, df["Close"], label="收盤價", color="#FFFFFF", linewidth=1.2)
        if show_ma:
            ax_price.plot(df.index, df["SMA20"], label="20日均線", color="#4FC3F7", linestyle="--")
            ax_price.plot(df.index, df["SMA60"], label="60日均線", color="#81C784", linestyle="--")
        if show_bb:
            ax_price.plot(df.index, df["UpperB"], label="上布林帶", color="#EF5350", linestyle=":")
            ax_price.plot(df.index, df["LowerB"], label="下布林帶", color="#EF5350", linestyle=":")

        ax_price.set_title(f"{stock_no} 技術分析（折線圖 / {freq_label}）", fontproperties=font_prop)
        ax_price.set_ylabel("股價", fontproperties=font_prop)
        ax_price.grid(True, alpha=0.25, linestyle=":")
        ax_price.legend(facecolor="#111", edgecolor="#333", labelcolor="w", prop=font_prop)

        # 成交量子圖（紅漲綠跌）
        if has_volume:
            up = df["Close"] >= df["Open"]
            vol_colors = np.where(up, "#FF5252", "#00C853")
            ax_vol.bar(df.index, df["Volume"], color=vol_colors, width=0.8)
            ax_vol.set_ylabel("成交量", fontproperties=font_prop)
            ax_vol.grid(True, alpha=0.2, linestyle=":")

        # 第三子圖：RSI + MACD（同一面板；RSI 左軸、MACD 右軸）
        ax_ind.set_ylabel("RSI(14)", fontproperties=font_prop, color="#80CBC4")
        ax_ind.plot(df.index, df["RSI"], color="#80CBC4", label="RSI(14)", linewidth=1.0)
        ax_ind.axhline(70, color="#B0BEC5", linestyle="--", linewidth=0.8)
        ax_ind.axhline(30, color="#B0BEC5", linestyle="--", linewidth=0.8)
        ax_ind.set_ylim(0, 100)
        ax_ind.grid(True, alpha=0.2, linestyle=":")

        ax_macd = ax_ind.twinx()
        hist_colors = np.where(df["MACD_hist"] >= 0, "#FF7043", "#26A69A")
        ax_macd.bar(df.index, df["MACD_hist"], color=hist_colors, width=0.8, alpha=0.5, label="MACD Hist")
        ax_macd.plot(df.index, df["MACD"], color="#FFA726", linewidth=1.0, label="MACD")
        ax_macd.plot(df.index, df["MACD_signal"], color="#FFFFFF", linewidth=1.0, label="Signal")
        ax_macd.set_ylabel("MACD", fontproperties=font_prop)

        # 標籤與刻度色
        for ax in [ax_price, ax_ind] + ([ax_vol] if ax_vol is not None else []):
            ax.tick_params(colors="w")
        ax_macd.tick_params(colors="w")
        plt.xticks(rotation=30, color="w")
        plt.yticks(color="w")

        st.pyplot(fig)

    # =======================
    # 蠟燭圖（深色 nightclouds）＋ Volume 面板 ＋ RSI/MACD 第三面板
    # =======================
    else:
        mc = mpf.make_marketcolors(
            up="r", down="g",
            edge="inherit", wick="inherit", volume="inherit",
        )
        s = mpf.make_mpf_style(base_mpf_style="nightclouds", marketcolors=mc, gridstyle=":")
        addplots = []

        # 布林帶畫在主圖
        if show_bb:
            addplots += [
                mpf.make_addplot(df["UpperB"], color="#EF5350", panel=0),
                mpf.make_addplot(df["LowerB"], color="#EF5350", panel=0),
            ]

        # 第三面板：RSI + MACD（在同一 panel=2）
        addplots += [
            mpf.make_addplot(df["RSI"], panel=2, color="#80CBC4"),
            mpf.make_addplot(pd.Series(70, index=df.index), panel=2, color="#B0BEC5", linestyle="--"),
            mpf.make_addplot(pd.Series(30, index=df.index), panel=2, color="#B0BEC5", linestyle="--"),
        ]
        # MACD hist + MACD + Signal
        macd_colors = ["#FF7043" if v >= 0 else "#26A69A" for v in df["MACD_hist"]]
        addplots += [
            mpf.make_addplot(df["MACD_hist"], type="bar", panel=2, color=macd_colors, alpha=0.5),
            mpf.make_addplot(df["MACD"], panel=2, color="#FFA726"),
            mpf.make_addplot(df["MACD_signal"], panel=2, color="#FFFFFF"),
        ]

        mav = (20, 60) if show_ma else None

        fig, axes = mpf.plot(
            df,
            type="candle",
            mav=mav,
            addplot=addplots,
            volume=("Volume" in df.columns),
            style=s,
            figsize=(14, 9),
            panel_ratios=(3, 1, 1),
            returnfig=True
        )
        # 用你的字型物件設標題/標籤（不改你的載入方式）
        try:
            axes[0].set_title(f"{stock_no} 技術分析（蠟燭圖 / {freq_label}）", fontproperties=font_prop)
            axes[0].set_ylabel("股價", fontproperties=font_prop)
            if "Volume" in df.columns:
                axes[1].set_ylabel("成交量", fontproperties=font_prop)
            axes[-1].set_ylabel("RSI / MACD", fontproperties=font_prop)
        except Exception:
            pass

        st.pyplot(fig)
