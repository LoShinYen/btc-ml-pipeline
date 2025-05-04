import os
import time
import pandas as pd
import numpy as np

from src.config import RAW_DIR, PROCESSED_DIR, TRAIN_DIR, TEST_DIR, KLINE_ONE_H_RAW_FILE, PROCESSED_FILE, TRAIN_FILE, TEST_FILE, FEAR_GREED_RAW_FILE, FERD_FILE

def check_dir():
    """
    檢查所需資料夾是否存在，若無則建立。
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

def load_raw_data():
    """
    載入 BTC 原始 K 線資料。
    Returns:
        pd.DataFrame: 含時間戳與價格資訊之 K 線資料。
    """
    raw_path = os.path.join(RAW_DIR, KLINE_ONE_H_RAW_FILE)
    raw_df = pd.read_csv(raw_path)
    raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])
    print(f"讀取 {len(raw_df)} 筆 raw 資料")
    return raw_df

def prepare_training_data(raw_df):
    """
    執行特徵工程、合併外部指標與標記資料。
    Args:
        raw_df (pd.DataFrame): 原始 K 線資料。
    Returns:
        pd.DataFrame: 完整的訓練資料。
    """
    # 合併 BTC K 線資料與 Fear & Greed Index，並將每日指數展開至每小時，
    # 使其支援連續時間序列的 rolling 計算。
    raw_df = merge_fear_greed(raw_df)

    # 合併 FRED 提供的每月宏觀經濟資料（CPI, PCE, FEDFUNDS 等），
    # 使用 forward-fill 將其延伸至每小時，對齊 BTC 資料時間。
    raw_df = merge_macro_data(raw_df)   


    # 對 BTC 1hr K 線資料執行完整特徵工程，涵蓋技術分析指標、成交量、K 線形態、
    raw_df = feature_engineering(raw_df)
    
    # 根據未來報酬率標記是否應該買入。
    raw_df = generate_labels(
        raw_df,
        future_hours=6,
        take_profit=0.02,
        stop_loss=-0.01
    )
    
    raw_df = raw_df.dropna()
    
    print(f"移除NA後剩餘 {len(raw_df)} 筆資料")
    print("標記後資料比例：")
    print(raw_df['label'].value_counts(normalize=True))
    
    return raw_df

def merge_fear_greed(df):
    """
    合併 BTC K 線資料與 Fear & Greed Index，並將每日指數展開至每小時，
    使其支援連續時間序列的 rolling 計算。
    
    Args:
        df (pd.DataFrame): 含 timestamp 欄位之 K 線資料。
    
    Returns:
        pd.DataFrame: 加入 fear_greed_value, fear_greed_level, fear_greed_label 的資料。
    """
    fg_path = os.path.join(RAW_DIR, FEAR_GREED_RAW_FILE)
    fg_df = pd.read_csv(fg_path)

    # 轉換為 datetime（保留 UTC）
    fg_df['date'] = pd.to_datetime(fg_df['date'], utc=True)

    # 重命名欄位
    fg_df = fg_df.rename(columns={
        'value': 'fear_greed_value',
        'value_classification': 'fear_greed_level'
    })

    # 僅保留每日 00:00（即每日的真實更新點）
    fg_daily = fg_df[fg_df['date'].dt.hour == 0].copy()

    # 建立小時資料範圍，延伸每日資料成每小時階梯形式
    start_ts = df['timestamp'].min()
    end_ts = df['timestamp'].max()
    hourly_index = pd.date_range(start=start_ts, end=end_ts, freq='h', tz='UTC')
    fg_hourly = pd.DataFrame({'timestamp': hourly_index})
    
    # 將每日資料設為 index，進行 forward fill
    fg_daily.set_index('date', inplace=True)
    fg_hourly = fg_hourly.set_index('timestamp')
    fg_hourly = fg_hourly.join(fg_daily, how='left')
    fg_hourly = fg_hourly.ffill().reset_index()

    # 合併資料
    merged_df = pd.merge(df, fg_hourly, on='timestamp', how='left')

    # 文字轉為分類數值（例如：Fear = 0, Greed = 1, Extreme Greed = 2...）
    if 'fear_greed_level' in merged_df.columns:
        merged_df['fear_greed_label'] = merged_df['fear_greed_level'].astype('category').cat.codes

    return merged_df

def merge_macro_data(df):
    """
    合併 FRED 提供的每月宏觀經濟資料（CPI, PCE, FEDFUNDS 等），
    使用 forward-fill 將其延伸至每小時，對齊 BTC 資料時間。
    """
    macro_path = os.path.join(RAW_DIR, FERD_FILE)
    macro_df = pd.read_csv(macro_path)

    # 轉為 datetime 並保留 UTC
    macro_df.rename(columns={macro_df.columns[0]: "date"}, inplace=True)
    macro_df['date'] = pd.to_datetime(macro_df['date'], utc=True)

    # 設定 index 為 date，便於時間對齊
    macro_df = macro_df.set_index('date')

    # 建立小時時間序列（與 kline 資料對齊）
    hourly_range = pd.date_range(df['timestamp'].min(), df['timestamp'].max(), freq='h', tz='UTC')
    hourly_df = pd.DataFrame(index=hourly_range)
    
    # forward fill macro 數據
    macro_hourly = hourly_df.join(macro_df, how='left').ffill().reset_index()
    macro_hourly.rename(columns={"index": "timestamp"}, inplace=True)

    # 合併到主資料上
    df = pd.merge(df, macro_hourly, on="timestamp", how="left")
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    對 BTC 1hr K 線資料執行完整特徵工程，涵蓋技術分析指標、成交量、K 線形態、
    情緒變化、時間週期與交互特徵，支援後續機器學習模型訓練。
    """
    # === 📈 均線特徵 ===
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_100'] = df['close'].rolling(window=100).mean()
    df['sma_convergence'] = df['sma_20'] - df['sma_50']

    df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema_10_diff'] = df['ema_10'].diff()

    # === 🔄 MACD 指標 ===
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd_line'] = ema12 - ema26
    df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd_line'] - df['macd_signal']
    df['macd_cross_signal'] = ((df['macd_line'] > df['macd_signal']) &
                               (df['macd_line'].shift(1) <= df['macd_signal'].shift(1))).astype(int)

    # === 🌀 價格動能與 RSI ===
    df['roc'] = (df['close'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    df['rsi_momentum'] = df['rsi_14'].diff()

    # === 📉 波動性與布林通道 ===
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = (df['high'] - df['close'].shift()).abs()
    df['low_close'] = (df['low'] - df['close'].shift()).abs()
    df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr_14'] = df['true_range'].rolling(window=14).mean()

    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    df['bb_distance'] = (df['close'] - df['bb_middle']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
    df['bb_percent_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
    df['rsi_bb_upper_cross'] = ((df['rsi_14'] > 70) & (df['close'] > df['bb_upper'])).astype(int)

    # === 🔺 成交量特徵 ===
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma_20'] + 1e-8)

    # === ⛳ 價格突破與交互特徵 ===
    df['high_breakout_20'] = (df['high'] > df['high'].rolling(window=20).max().shift(1)).astype(int)
    df['low_breakout_20'] = (df['low'] < df['low'].rolling(window=20).min().shift(1)).astype(int)
    df['price_vs_sma20'] = (df['close'] - df['sma_20']) / (df['sma_20'] + 1e-8)
    df['price_vs_sma50_volume'] = df['price_vs_sma20'] * df['volume_ratio']

    # === 🧮 滾動標準化 (Z-Score) ===
    for col in ['close', 'open', 'high', 'low']:
        df[f'{col}_zscore'] = (df[col] - df[col].rolling(20).mean()) / (df[col].rolling(20).std() + 1e-8)

    # === 🔁 Lag 特徵 ===
    for col in ['close', 'roc', 'rsi_14', 'macd_hist']:
        df[f'{col}_lag_1'] = df[col].shift(1)

    # === ⏱️ 時間週期（原始 + 正弦/餘弦） ===
    df['hour'] = df['timestamp'].dt.hour
    df['weekday'] = df['timestamp'].dt.weekday
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)

    # === 🕯️ K 線結構與比率 ===
    df['candle_body'] = (df['close'] - df['open']).abs()
    df['candle_upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['candle_lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
    df['candle_ratio'] = df['candle_body'] / (df['high'] - df['low'] + 1e-8)
    df['upper_wick_ratio'] = df['candle_upper_wick'] / (df['high'] - df['low'] + 1e-8)
    df['lower_wick_ratio'] = df['candle_lower_wick'] / (df['high'] - df['low'] + 1e-8)

    # === 🔥 KD 指標 ===
    low_min = df['low'].rolling(window=14).min()
    high_max = df['high'].rolling(window=14).max()
    df['kd_k'] = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-8)
    df['kd_d'] = df['kd_k'].rolling(window=3).mean()

    # === 😱 恐懼貪婪指標（動能 + 反向訊號） ===
    if 'fear_greed_value' in df.columns:
        df['fg_diff'] = df['fear_greed_value'].diff()
        df['fg_ema_3'] = df['fear_greed_value'].ewm(span=3).mean()
        df['fg_ema_7'] = df['fear_greed_value'].ewm(span=7).mean()
        df['fg_zscore_30'] = (df['fear_greed_value'] - df['fear_greed_value'].rolling(30).mean()) / \
                             (df['fear_greed_value'].rolling(30).std() + 1e-8)
        df['greed_high_then_macd_drop'] = ((df['fear_greed_label'] == 1) &
                                           (df['macd_hist'].diff() < 0)).astype(int)

    # 🧹 清除暫存欄位
    df.drop(columns=['high_low', 'high_close', 'low_close', 'true_range'], inplace=True)

    return df

def generate_labels(df, future_hours=6, take_profit=0.02, stop_loss=-0.01):
    """
    根據未來報酬率標記是否應該買入。
    Args:
        df (pd.DataFrame): K 線資料。
        future_hours (int): 預測視窗小時數。
        take_profit (float): 預期最大報酬。
        stop_loss (float): 可容忍最大回撤。
    Returns:
        pd.DataFrame: 含 label 的資料。
    """
    close_prices = df['close'].values
    labels = []

    for i in range(len(close_prices)):
        future_window = close_prices[i+1:i+1+future_hours]
        if len(future_window) == 0:
            labels.append(0)
            continue

        future_max = future_window.max()
        future_min = future_window.min()
        current_price = close_prices[i]
        max_return = (future_max - current_price) / current_price
        min_return = (future_min - current_price) / current_price

        if max_return >= take_profit and min_return > stop_loss:
            labels.append(1)
        else:
            labels.append(0)

    df['label'] = labels
    return df

def save_processed_data(df):
    """
    將處理後資料儲存至 processed 目錄。
    """
    processed_path = os.path.join(PROCESSED_DIR, PROCESSED_FILE)
    df.to_csv(processed_path, index=False, encoding='utf-8-sig')
    print(f"成功儲存特徵工程後資料：{processed_path}")

def split_train_test(df):
    """
    時序方式切分訓練與測試資料。
    """
    train_size = 0.8
    split_index = int(len(df) * train_size)
    df_train = df.iloc[:split_index]
    df_test = df.iloc[split_index:]
    
    # 檢查 label 是否只有一類
    unique_test_labels = df_test['label'].unique()
    if len(unique_test_labels) < 2:
        print("⚠️ Warning: 測試集只有單一類別，可能導致 auc 無法計算。")
        print("Test Label 分佈：", df_test['label'].value_counts())
    
    return df_train, df_test

def save_train_test_data(df_train, df_test):
    """
    儲存訓練與測試資料至各自目錄。
    """
    df_train.to_csv(os.path.join(TRAIN_DIR, TRAIN_FILE), index=False, encoding='utf-8-sig')
    df_test.to_csv(os.path.join(TEST_DIR, TEST_FILE), index=False, encoding='utf-8-sig')
    print(f"✅ Train 集儲存成功 ({len(df_train)} 筆)")
    print(f"✅ Test 集儲存成功 ({len(df_test)} 筆)")

def main():
    """
    主流程：整合、處理與切分資料。
    """
    try:
        # step 1: 檢查資料夾
        check_dir()

        # step 2: 載入原始資料
        raw_df = load_raw_data()

        # step 3: 處理資料
        df = prepare_training_data(raw_df)

        # step 4: 儲存處理後資料
        save_processed_data(df)

        # step 5: 切分訓練與測試資料
        df_train, df_test = split_train_test(df)

        # step 6: 儲存訓練與測試資料
        save_train_test_data(df_train, df_test)
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")

