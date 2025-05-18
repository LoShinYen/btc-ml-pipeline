import ccxt
import time
import os
import pandas as pd
from datetime import datetime, timezone
from src.config import EXCHANGE_NAME, SYMBOL, TIMEFRAME, SLEEP_SECONDS, RAW_DIR, KLINE_ONE_H_RAW_FILE

def get_exchange():
    """
    初始化交易所連線。
    
    Returns:
        exchange (ccxt.Exchange): 初始化後的交易所物件
    """
    exchange_class = getattr(ccxt, EXCHANGE_NAME)
    exchange = exchange_class({
        'enableRateLimit': True  # 啟用限速
    })
    return exchange

def fetch_all_ohlcv(exchange, symbol, timeframe, since_timestamp):
    """
    從指定時間點起，抓取所有歷史K線資料直到現在。
    
    Args:
        exchange (ccxt.Exchange): 已初始化的交易所物件
        symbol (str): 幣種代碼，例如 'BTC/USDT'
        timeframe (str): K線時間範圍，例如 '1h'
        since_timestamp (int): 起始時間戳（毫秒）
        
    Returns:
        pd.DataFrame: 所有歷史K線資料
    """
    all_ohlcv = []
    now = exchange.milliseconds()

    print(f"從 {datetime.fromtimestamp(since_timestamp/1000, tz=timezone.utc)} 開始抓取資料...")

    while since_timestamp < now:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_timestamp, limit=1000)
            if not ohlcv:
                print("沒有更多資料了，停止抓取。")
                break

            all_ohlcv.extend(ohlcv)
            print(f"抓取到 {len(all_ohlcv)} 筆，目前最新時間：{datetime.fromtimestamp(ohlcv[-1][0]/1000, tz=timezone.utc)}")

            # 更新下一次抓取的起點
            since_timestamp = ohlcv[-1][0] + 1  # 加1毫秒，避免重複
            
            # 尊重交易所限速
            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            print(f"抓取失敗，等待重新連線：{e}")
            time.sleep(10)  # 等10秒重試
    
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC')
    return df

def save_raw_data(df):
    """
    儲存原始K線資料到CSV。
    
    Args:
        df (pd.DataFrame): 原始K線資料
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    df.to_csv(os.path.join(RAW_DIR, KLINE_ONE_H_RAW_FILE), index=False, encoding='utf-8-sig')

    print(f"✅ 成功儲存 {KLINE_ONE_H_RAW_FILE}，共 {len(df)} 筆資料！")

def main():
    """
    主程式：初始化交易所、抓取K線、儲存到CSV。
    """
    try:
        # Step 1: 初始化交易所
        exchange = get_exchange()

        # Step 2: 抓取K線資料
        # 幣安最早的現貨資料大概從 2017-08 開始，所以這裡直接設定個合理值
        start_time = exchange.parse8601('2017-08-01T00:00:00Z')

        df = fetch_all_ohlcv(exchange, SYMBOL, TIMEFRAME, start_time)

        # Step 3: 儲存到CSV
        save_raw_data(df)
        
    except Exception as e:
        print(f"❌ 抓取失敗：{e}")

