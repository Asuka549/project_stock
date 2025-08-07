# app.py
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
from stock import fetch_stock_data
import os
import urllib.request

# 使用本地字型（已隨專案一併部署）
font_path = os.path.join(".streamlit", "fonts", "NotoSansTC-Regular.otf")
if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    matplotlib.rcParams["font.family"] = font_prop.get_name()
    st.write(f"✅ 成功載入中文字型：{font_prop.get_name()}")
else:
    st.warning("⚠️ 找不到中文字型，圖表中文字可能無法正確顯示。")

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

# === Main action ===
if st.sidebar.button("開始繪製"):
    st.write(f"👉 查詢參數：股票代號={stock_no}, 起={start_year}/{start_month}, 迄={end_year}/{end_month}")
    st.info("正在抓取資料，請稍候...")
    df = fetch_stock_data(stock_no, start_year, start_month, end_year, end_month)

    st.write("📦 抓到原始資料筆數：", len(df))
    st.dataframe(df.head())

    if df.empty:
        st.error("找不到資料，請確認股票代號與時間範圍。")
    else:

        st.write("📦 抓到資料筆數：", len(df))
        st.dataframe(df.head())

        # 計算技術指標
        if "移動平均線" in indicator:
            df['SMA20'] = df['Close'].rolling(window=20).mean()
            df['SMA60'] = df['Close'].rolling(window=60).mean()
        if "布林帶" in indicator:
            df['20SMA'] = df['Close'].rolling(window=20).mean()
            df['20STD'] = df['Close'].rolling(window=20).std()
            df['UpperB'] = df['20SMA'] + 2 * df['20STD']
            df['LowerB'] = df['20SMA'] - 2 * df['20STD']

        # 畫圖
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(df.index, df['Close'], label='收盤價', color='black')

        if "移動平均線" in indicator:
            ax.plot(df.index, df['SMA20'], label='20日均線', color='blue', linestyle='--')
            ax.plot(df.index, df['SMA60'], label='60日均線', color='green', linestyle='--')

        if "布林帶" in indicator:
            ax.plot(df.index, df['UpperB'], label='上布林帶', color='red', linestyle=':')
            ax.plot(df.index, df['LowerB'], label='下布林帶', color='red', linestyle=':')

        ax.set_title(f"{stock_no} 技術分析圖")
        ax.set_xlabel("日期")
        ax.set_ylabel("股價")
        ax.grid(True)
        ax.legend()
        plt.xticks(rotation=30)
        st.pyplot(fig)

        # 顯示資料表（可選）
        with st.expander("查看原始資料表"):
            st.dataframe(df.tail(30))
