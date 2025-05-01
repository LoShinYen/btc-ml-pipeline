import requests
import pandas as pd
import os
from src.config import RAW_DIR, FEAR_GREED_RAW_FILE

def fetch_fear_greed_data():
    """
    抓取 Fear & Greed Index 全次次次次資料，儲存為 CSV
    """
    url = 'https://api.alternative.me/fng/?limit=0&format=json'
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f"API 抽取失敗，確認網路狀況，status_code={response.status_code}")

    data = response.json()
    records = data['data']

    # 轉成 DataFrame
    df = pd.DataFrame(records)

    # timestamp 轉成人語日期
    df['timestamp'] = df['timestamp'].astype(int)
    df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.tz_localize('UTC')

    # 重排序，最早在前
    df = df.sort_values('date').reset_index(drop=True)

    # 选擇有用的欄位
    df = df[['date', 'value', 'value_classification']]

    return df

def upsample_to_hourly(df_daily):
    """
    將每日 Fear & Greed Index 數據拉高至 1 小時額，保持繼承值（forward fill）
    Args:
        df_daily (pd.DataFrame): 每日數據

    Returns:
        pd.DataFrame: 每小時的 Fear & Greed Index
    """
    df = df_daily.set_index('date')
    df_hourly = df.resample('1h').ffill()
    df_hourly = df_hourly.reset_index()
    return df_hourly

def main():
    try:
        os.makedirs(RAW_DIR, exist_ok=True)

        # 抓資料
        df_daily = fetch_fear_greed_data()
        print(f"✨ 成功抓取 {len(df_daily)} 筆每日 Fear & Greed Index 資料")

        # 上重進一步，抽高成 1hr
        df_hourly = upsample_to_hourly(df_daily)
        print(f"⏰ 已經變換為 {len(df_hourly)} 筆 1hr Fear & Greed Index 資料")

        # 儲存
        output_path = os.path.join(RAW_DIR, FEAR_GREED_RAW_FILE)
        df_hourly.to_csv(output_path, index=False, encoding='utf-8-sig')

        print(f"✅ 資料已成功儲存到 {output_path}")

    except Exception as e:
        print(f"❌ 發生錯誤: {e}")

if __name__ == '__main__':
    main()
