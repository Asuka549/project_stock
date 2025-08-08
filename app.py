# app.py
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import numpy as np
import os

# Plotly（互動）
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# 蠟燭圖（靜態）
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

c1, c2 = st.sidebar.columns(2)
start_year = c1.number_input("起始年份（西元）", value=2025, step=1)
start_month = c2.selectbox("起始月份", list(range(1, 13)), index=0)

c3, c4 = st.sidebar.columns(2)
end_year = c3.number_input("結束年份（西元）", value=2025, step=1)
end_month = c4.selectbox("結束月份", list(range(1, 13)), index=1)

indicator = st.sidebar.selectbox("技術指標", ["布林帶", "移動平均線", "布林帶 + 移動平均線"])
chart_type = st.sidebar.selectbox(
    "圖表類型",
    ["蠟燭圖", "折線圖", "互動蠟燭圖 (Plotly)", "互動折線圖 (Plotly)"]
)

# ✅ 資料頻率
freq_label = st.sidebar.selectbox("資料頻率", ["每天", "每週", "每月", "每小時", "每15分鐘"], index=0)
FREQ_MAP = {"每天": "D", "每週": "W", "每月": "M", "每小時": "H", "每15分鐘": "15min"}
rule = FREQ_MAP[freq_label]

# ========== 小工具 ==========
def has_data(s: pd.Series | None) -> bool:
    try:
        return s is not None and s.dropna().size > 0
    except Exception:
        return False

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

def resample_ohlcv(df: pd.DataFrame, rule: str) -> tuple[pd.DataFrame, bool]:
    """
    回傳 (resampled_df, used_original)。
    - 若原始資料為日頻，且選擇小時/15分，則回傳 (原 df, True)。
    """
    # 確保時間索引
    if not np.issubdtype(df.index.dtype, np.datetime64):
        if "Date" in df.columns:
            df = df.copy()
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date").sort_index()
        else:
            return df, True

    # 判斷原始最小間隔
    diffs = df.index.to_series().diff().dropna()
    min_step = diffs.min() if not diffs.empty else pd.Timedelta("0s")

    # 日線來源 → 無法向下取樣成時/分
    if rule in ("H", "15min") and (min_step >= pd.Timedelta("1D")):
        return df, True

    agg = {k: v for k, v in {"Open": "first", "High": "max", "Low": "min", "Close": "last"}.items() if k in df.columns}
    if "Volume" in df.columns:
        agg["Volume"] = "sum"

    res = df.resample(rule, label="right", closed="right").agg(agg)
    # 清掉不完整列
    for c in ("Open", "High", "Low", "Close"):
        if c in res.columns:
            res = res[res[c].notna()]
    return res, False

@st.cache_data(ttl=3600)
def get_daily(stock_no, sy, sm, ey, em):
    return fetch_stock_data(stock_no, sy, sm, ey, em)

# === Main ===
if st.sidebar.button("開始繪製"):
    st.write(f"👉 查詢參數：股票代號={stock_no}, 起={start_year}/{start_month}, 迄={end_year}/{end_month}，頻率={freq_label}")
    st.info("正在抓取資料，請稍候...")
    df = get_daily(stock_no, start_year, start_month, end_year, end_month)

    if df is None or df.empty:
        st.error("找不到資料，請確認股票代號與時間範圍。")
        st.stop()

    # 重採樣
    df_resampled, used_original = resample_ohlcv(df, rule)
    if used_original and rule in ("H", "15min"):
        st.warning("⚠️ 原始資料僅有日頻，無法產生每小時/每15分鐘資料，已改用『每天』。")
        rule = "D"
    df = df_resampled

    # 必要欄位
    needed_cols = {"Open", "High", "Low", "Close"}
    if not needed_cols.issubset(df.columns):
        st.error(f"缺少必要欄位：{needed_cols - set(df.columns)}")
        st.stop()

    has_volume = "Volume" in df.columns

    # 指標（以重採樣後資料為基礎）
    show_ma = ("移動平均線" in indicator) or ("布林帶 + 移動平均線" in indicator)
    show_bb = ("布林帶" in indicator) or ("布林帶 + 移動平均線" in indicator)

    if show_ma and len(df) >= 20:
        df["SMA20"] = df["Close"].rolling(20).mean()
    if show_ma and len(df) >= 60:
        df["SMA60"] = df["Close"].rolling(60).mean()

    if show_bb and len(df) >= 20:
        df["BB_MA"] = df["Close"].rolling(20).mean()
        df["BB_STD"] = df["Close"].rolling(20).std()
        df["UpperB"] = df["BB_MA"] + 2 * df["BB_STD"]
        df["LowerB"] = df["BB_MA"] - 2 * df["BB_STD"]

    df = add_rsi_macd(df)

    st.success(f"✅ 採用頻率：{freq_label}（實際列數：{len(df)}）")
    with st.expander("查看資料（最後 30 筆）"):
        st.dataframe(df.tail(30), use_container_width=True)

    # =======================
    # 折線圖（Matplotlib）
    # =======================
    if chart_type == "折線圖":
        plt.style.use("dark_background")

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

        # 主圖
        ax_price.plot(df.index, df["Close"], label="收盤價", color="#FFFFFF", linewidth=1.2)
        if show_ma and has_data(df.get("SMA20")):
            ax_price.plot(df.index, df["SMA20"], label="20日均線", color="#4FC3F7", linestyle="--")
        if show_ma and has_data(df.get("SMA60")):
            ax_price.plot(df.index, df["SMA60"], label="60日均線", color="#81C784", linestyle="--")
        if show_bb and has_data(df.get("UpperB")):
            ax_price.plot(df.index, df["UpperB"], label="上布林帶", color="#EF5350", linestyle=":")
        if show_bb and has_data(df.get("LowerB")):
            ax_price.plot(df.index, df["LowerB"], label="下布林帶", color="#EF5350", linestyle=":")

        ax_price.set_title(f"{stock_no} 技術分析（折線圖 / {freq_label}）", fontproperties=font_prop)
        ax_price.set_ylabel("股價", fontproperties=font_prop)
        ax_price.grid(True, alpha=0.25, linestyle=":")
        ax_price.legend(facecolor="#111", edgecolor="#333", labelcolor="w", prop=font_prop)

        # 成交量
        if has_volume:
            up = df["Close"] >= df["Open"]
            vol_colors = np.where(up, "#FF5252", "#00C853")
            ax_vol.bar(df.index, df["Volume"], color=vol_colors, width=0.8)
            ax_vol.set_ylabel("成交量", fontproperties=font_prop)
            ax_vol.grid(True, alpha=0.2, linestyle=":")

        # RSI + MACD
        ax_ind.set_ylabel("RSI(14)", fontproperties=font_prop, color="#80CBC4")
        if has_data(df.get("RSI")):
            ax_ind.plot(df.index, df["RSI"], color="#80CBC4", linewidth=1.0, label="RSI(14)")
            ax_ind.axhline(70, color="#B0BEC5", linestyle="--", linewidth=0.8)
            ax_ind.axhline(30, color="#B0BEC5", linestyle="--", linewidth=0.8)
            ax_ind.set_ylim(0, 100)
        ax_ind.grid(True, alpha=0.2, linestyle=":")

        ax_macd = ax_ind.twinx()
        if has_data(df.get("MACD_hist")):
            hist = df["MACD_hist"].dropna()
            macd_colors = ["#FF7043" if v >= 0 else "#26A69A" for v in hist]
            ax_macd.bar(df.index, hist, color=macd_colors, width=0.8, alpha=0.5)
        if has_data(df.get("MACD")):
            ax_macd.plot(df.index, df["MACD"], color="#FFA726", linewidth=1.0)
        if has_data(df.get("MACD_signal")):
            ax_macd.plot(df.index, df["MACD_signal"], color="#FFFFFF", linewidth=1.0)
        ax_macd.set_ylabel("MACD", fontproperties=font_prop)

        for ax in [ax_price, ax_ind] + ([ax_vol] if ax_vol is not None else []):
            ax.tick_params(colors="w")
        ax_macd.tick_params(colors="w")
        plt.xticks(rotation=30, color="w")
        plt.yticks(color="w")

        st.pyplot(fig)

    # =======================
    # 蠟燭圖（mplfinance）
    # =======================
    elif chart_type == "蠟燭圖":
        mc = mpf.make_marketcolors(up="r", down="g", edge="inherit", wick="inherit", volume="inherit")
        s = mpf.make_mpf_style(base_mpf_style="nightclouds", marketcolors=mc, gridstyle=":")

        has_vol_panel = ("Volume" in df.columns) and df["Volume"].dropna().size > 0
        has_ind_panel = any([has_data(df.get("RSI")), has_data(df.get("MACD")),
                             has_data(df.get("MACD_signal")), has_data(df.get("MACD_hist"))])
        ind_panel = 2 if has_vol_panel and has_ind_panel else (1 if (not has_vol_panel and has_ind_panel) else None)

        addplots = []
        if show_bb and has_data(df.get("UpperB")) and has_data(df.get("LowerB")):
            addplots += [
                mpf.make_addplot(df["UpperB"], color="#EF5350", panel=0),
                mpf.make_addplot(df["LowerB"], color="#EF5350", panel=0),
            ]
        if ind_panel is not None and has_data(df.get("RSI")):
            addplots += [mpf.make_addplot(df["RSI"], panel=ind_panel, color="#80CBC4")]
            addplots += [
                mpf.make_addplot(pd.Series(70, index=df.index), panel=ind_panel, color="#B0BEC5", linestyle="--"),
                mpf.make_addplot(pd.Series(30, index=df.index), panel=ind_panel, color="#B0BEC5", linestyle="--"),
            ]
        if ind_panel is not None and has_data(df.get("MACD_hist")):
            hist = df["MACD_hist"].dropna()
            macd_colors = ["#FF7043" if v >= 0 else "#26A69A" for v in hist]
            addplots += [mpf.make_addplot(hist, type="bar", panel=ind_panel, color=macd_colors, alpha=0.5)]
        if ind_panel is not None and has_data(df.get("MACD")):
            addplots += [mpf.make_addplot(df["MACD"], panel=ind_panel, color="#FFA726")]
        if ind_panel is not None and has_data(df.get("MACD_signal")):
            addplots += [mpf.make_addplot(df["MACD_signal"], panel=ind_panel, color="#FFFFFF")]

        mav = None
        if show_ma and (has_data(df.get("SMA20")) or has_data(df.get("SMA60"))):
            mav = tuple([p for p, name in [(20, "SMA20"), (60, "SMA60")] if has_data(df.get(name))]) or None

        kwargs = {"type": "candle", "volume": has_vol_panel, "style": s, "figsize": (14, 9), "returnfig": True}
        if mav:
            kwargs["mav"] = mav
        if addplots:
            kwargs["addplot"] = addplots

        fig, axes = mpf.plot(df, **kwargs)

        try:
            axes[0].set_title(f"{stock_no} 技術分析（蠟燭圖 / {freq_label}）", fontproperties=font_prop)
            axes[0].set_ylabel("股價", fontproperties=font_prop)
            if has_vol_panel and len(axes) > 1:
                axes[1].set_ylabel("成交量", fontproperties=font_prop)
            if has_ind_panel and len(axes) > 2:
                axes[-1].set_ylabel("RSI / MACD", fontproperties=font_prop)
        except Exception:
            pass

        st.pyplot(fig)

    # =======================
    # 互動折線圖（Plotly）
    # =======================
    elif chart_type == "互動折線圖 (Plotly)":
        rows = 3 if has_volume else 2
        fig = make_subplots(
            rows=rows, cols=1, shared_xaxes=True,
            row_heights=[0.6, 0.2, 0.2][:rows],
            vertical_spacing=0.04
        )

        # 主圖：收盤、MA、BB
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="收盤價", mode="lines"), row=1, col=1)
        if show_ma and has_data(df.get("SMA20")):
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="20日均線", mode="lines"), row=1, col=1)
        if show_ma and has_data(df.get("SMA60")):
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA60"], name="60日均線", mode="lines"), row=1, col=1)
        if show_bb and has_data(df.get("UpperB")):
            fig.add_trace(go.Scatter(x=df.index, y=df["UpperB"], name="上布林", mode="lines"), row=1, col=1)
        if show_bb and has_data(df.get("LowerB")):
            fig.add_trace(go.Scatter(x=df.index, y=df["LowerB"], name="下布林", mode="lines"), row=1, col=1)

        # 量能
        if has_volume:
            up = (df["Close"] >= df["Open"]).reindex(df.index, fill_value=False)
            colors = np.where(up, "red", "green")
            fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="成交量", marker_color=colors, opacity=0.7),
                          row=2, col=1)

        # 第三面板：RSI + MACD（同面板）
        rsi_row = 3 if has_volume else 2
        if has_data(df.get("RSI")):
            fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI(14)", mode="lines"), row=rsi_row, col=1)
            fig.add_hline(y=70, line_dash="dash", row=rsi_row, col=1)
            fig.add_hline(y=30, line_dash="dash", row=rsi_row, col=1)
        if has_data(df.get("MACD_hist")):
            hist = df["MACD_hist"].dropna()
            colors_hist = ["#FF7043" if v >= 0 else "#26A69A" for v in hist]
            fig.add_trace(go.Bar(x=hist.index, y=hist, name="MACD Hist", marker_color=colors_hist, opacity=0.5),
                          row=rsi_row, col=1)
        if has_data(df.get("MACD")):
            fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", mode="lines"), row=rsi_row, col=1)
        if has_data(df.get("MACD_signal")):
            fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal", mode="lines"),
                          row=rsi_row, col=1)

        # 版面＆十字準線
        fig.update_layout(
            template="plotly_dark",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=20, t=50, b=30),
            title=f"{stock_no} 技術分析（折線 / {freq_label}）",
            font=dict(family=(font_prop.get_name() if font_prop else None))
        )
        for i in range(1, rows + 1):
            fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", row=i, col=1)
            fig.update_yaxes(showspikes=True, spikemode="across", row=i, col=1)

        fig.update_yaxes(title_text="股價", row=1, col=1)
        if has_volume:
            fig.update_yaxes(title_text="成交量", row=2, col=1)
        fig.update_yaxes(title_text="RSI / MACD", row=rsi_row, col=1)

        st.plotly_chart(fig, use_container_width=True)

    # =======================
    # 互動蠟燭圖（Plotly）
    # =======================
    else:  # "互動蠟燭圖 (Plotly)"
        rows = 3 if has_volume else 2
        fig = make_subplots(
            rows=rows, cols=1, shared_xaxes=True,
            row_heights=[0.6, 0.2, 0.2][:rows],
            vertical_spacing=0.04
        )

        # K 線（紅漲綠跌）
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                increasing_line_color="red", decreasing_line_color="green",
                increasing_fillcolor="red", decreasing_fillcolor="green",
                name="K線"
            ),
            row=1, col=1
        )

        # MA / BB
        if show_ma and has_data(df.get("SMA20")):
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="20日均線", mode="lines"), row=1, col=1)
        if show_ma and has_data(df.get("SMA60")):
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA60"], name="60日均線", mode="lines"), row=1, col=1)
        if show_bb and has_data(df.get("UpperB")):
            fig.add_trace(go.Scatter(x=df.index, y=df["UpperB"], name="上布林", mode="lines"), row=1, col=1)
        if show_bb and has_data(df.get("LowerB")):
            fig.add_trace(go.Scatter(x=df.index, y=df["LowerB"], name="下布林", mode="lines"), row=1, col=1)

        # 量能
        if has_volume:
            up = (df["Close"] >= df["Open"]).reindex(df.index, fill_value=False)
            colors = np.where(up, "red", "green")
            fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="成交量", marker_color=colors, opacity=0.7),
                          row=2, col=1)

        # RSI + MACD 同面板
        rsi_row = 3 if has_volume else 2
        if has_data(df.get("RSI")):
            fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI(14)", mode="lines"), row=rsi_row, col=1)
            fig.add_hline(y=70, line_dash="dash", row=rsi_row, col=1)
            fig.add_hline(y=30, line_dash="dash", row=rsi_row, col=1)
        if has_data(df.get("MACD_hist")):
            hist = df["MACD_hist"].dropna()
            colors_hist = ["#FF7043" if v >= 0 else "#26A69A" for v in hist]
            fig.add_trace(go.Bar(x=hist.index, y=hist, name="MACD Hist", marker_color=colors_hist, opacity=0.5),
                          row=rsi_row, col=1)
        if has_data(df.get("MACD")):
            fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", mode="lines"), row=rsi_row, col=1)
        if has_data(df.get("MACD_signal")):
            fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal", mode="lines"),
                          row=rsi_row, col=1)

        # 版面＆十字準線
        fig.update_layout(
            template="plotly_dark",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=20, t=50, b=30),
            title=f"{stock_no} 技術分析（蠟燭 / {freq_label}）",
            font=dict(family=(font_prop.get_name() if font_prop else None))
        )
        for i in range(1, rows + 1):
            fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", row=i, col=1)
            fig.update_yaxes(showspikes=True, spikemode="across", row=i, col=1)

        # 隱藏 rangeslider（要也可改 True）
        fig.update_xaxes(rangeslider_visible=False, row=1, col=1)

        fig.update_yaxes(title_text="股價", row=1, col=1)
        if has_volume:
            fig.update_yaxes(title_text="成交量", row=2, col=1)
        fig.update_yaxes(title_text="RSI / MACD", row=rsi_row, col=1)

        st.plotly_chart(fig, use_container_width=True)
