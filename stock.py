def fetch_stock_data(stock_no, start_year, start_month, end_year, end_month):
    import time
    import requests as r
    import pandas as pd

    sess = r.Session()
    sess.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; StockApp/1.0)"
    })

    frames = []

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # 篩選月份區間
            if (year == start_year and month < start_month) or (year == end_year and month > end_month):
                continue

            date_param = f"{year}{str(month).zfill(2)}01"
            url = "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY"
            params = {"date": date_param, "stockNo": stock_no, "response": "json"}

            try:
                res = sess.get(url, params=params, timeout=10)
                if res.status_code != 200:
                    print(f"⚠️ HTTP {res.status_code}: {res.text[:120]} ...")
                    time.sleep(0.2)
                    continue

                stock_json = res.json()
                data = stock_json.get("data", [])
                fields = stock_json.get("fields", [])

                print(f"🔎 URL: {res.url}")
                print(f"🧾 筆數: {len(data)}, 欄位: {fields}")

                if data and fields:
                    frames.append(pd.DataFrame(data=data, columns=fields))
                else:
                    print(f"⚠️ 無有效資料或欄位：{res.url}")

            except Exception as e:
                print(f"❌ 例外：{url} {params}，原因：{e}")

            time.sleep(0.2)  # 避免請求過密，可視情況調整/移除

    if not frames:
        print("🚨 最終結果：找不到任何資料")
        return pd.DataFrame()

    daily = pd.concat(frames, ignore_index=True)

    # 欄位對應
    column_map = {}
    for col in daily.columns:
        if '日期' in col:
            column_map[col] = 'Date'
        elif '成交股數' in col:
            column_map[col] = 'Volume'
        elif '開盤' in col:
            column_map[col] = 'Open'
        elif '最高' in col:
            column_map[col] = 'High'
        elif '最低' in col:
            column_map[col] = 'Low'
        elif '收盤' in col:
            column_map[col] = 'Close'
    daily = daily.rename(columns=column_map)

    required = ['Date', 'Volume', 'Open', 'High', 'Low', 'Close']
    if not all(c in daily.columns for c in required):
        print("🚨 缺少必要欄位，目前欄位為：", daily.columns.tolist())
        return pd.DataFrame()

    # 民國年 → 西元年
    def convert_tw_date(date_str):
        try:
            y, m, d = date_str.split('/')
            return f"{int(y) + 1911}-{m}-{d}"
        except:
            return None

    daily['Date'] = pd.to_datetime(daily['Date'].apply(convert_tw_date), errors='coerce')
    daily = daily.dropna(subset=['Date'])

    # 數值清理
    for col in ['Volume', 'Open', 'High', 'Low', 'Close']:
        daily[col] = pd.to_numeric(daily[col].replace(',', '', regex=True), errors='coerce')

    daily = daily.dropna(subset=['Volume', 'Open', 'High', 'Low', 'Close']).set_index('Date').sort_index()

    print("🧪 前幾筆資料（轉換後）:")
    print(daily.head())
    print("✅ 整理後資料筆數：", len(daily))
    return daily
