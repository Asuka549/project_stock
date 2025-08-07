def fetch_stock_data(stock_no, start_year, start_month, end_year, end_month):
    import requests as r
    import pandas as pd

    list1 = []

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if (year == start_year and month < start_month) or (year == end_year and month > end_month):
                continue

            query_year = year
            url = f'https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?date={query_year}{str(month).zfill(2)}01&stockNo={stock_no}&response=json'

            try:
                res = r.get(url)
                stock_json = res.json()

                print(f"🔎 查詢 URL: {url}")
                print(f"🧾 筆數: {len(stock_json.get('data', []))}, 欄位: {stock_json.get('fields')}")

                if stock_json.get('data') and stock_json.get('fields'):
                    df = pd.DataFrame(data=stock_json['data'], columns=stock_json['fields'])
                    list1.append(df)
                else:
                    print(f"⚠️ 無有效資料或欄位：{url}")

            except Exception as e:
                print(f"❌ 發生錯誤：{url}, 原因：{e}")
                continue

    if not list1:
        print("🚨 最終結果：找不到任何資料")
        return pd.DataFrame()

    daily = pd.concat(list1, ignore_index=True)

    # 自動對應欄位名稱（模糊比對）
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

    # 確認必要欄位
    required_cols = ['Date', 'Volume', 'Open', 'High', 'Low', 'Close']
    if not all(col in daily.columns for col in required_cols):
        print("🚨 缺少必要欄位，目前欄位為：", daily.columns.tolist())
        return pd.DataFrame()

    # 日期轉換
    # 日期轉換：將民國年 113/01/03 → 西元 2024-01-03
    def convert_tw_date(date_str):
        try:
            y, m, d = date_str.split('/')
            year = int(y) + 1911
            return f"{year}-{m}-{d}"
        except:
            return None

    daily['Date'] = daily['Date'].apply(convert_tw_date)
    daily = daily.dropna(subset=['Date'])
    daily['Date'] = pd.to_datetime(daily['Date'])
    daily = daily.dropna(subset=['Date'])

    # 數值欄位清理
    for col in ['Volume', 'Open', 'High', 'Low', 'Close']:
        daily[col] = daily[col].replace(',', '', regex=True)
        daily[col] = pd.to_numeric(daily[col], errors='coerce')

    daily = daily.dropna(subset=['Volume', 'Open', 'High', 'Low', 'Close'])
    daily = daily.set_index('Date')

    print("🧪 前幾筆資料（轉換後）:")
    print(daily.head())
    print("✅ 整理後資料筆數：", len(daily))

    return daily
