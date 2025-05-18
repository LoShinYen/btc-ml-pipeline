import os
from dotenv import load_dotenv

load_dotenv()  # 讀取 .env

# API_KEY 設定
FERD_API_KEY = os.getenv('FERD_API_KEY')

# 交易所設定
EXCHANGE_NAME = os.getenv('EXCHANGE_NAME', 'binance')                                       # 交易所名稱
SYMBOL = os.getenv('SYMBOL', 'BTC/USDT')                                                    # 交易對
TIMEFRAME = os.getenv('TIMEFRAME', '1h')                                                    # 時間週期
SLEEP_SECONDS = int(os.getenv('SLEEP_SECONDS', '1'))                                        # 抓取間隔

# 資料夾設定
RAW_DIR = os.getenv('RAW_DIR', 'data/raw')                                                  # 原始資料夾    
PROCESSED_DIR = os.getenv('PROCESSED_DIR', 'data/processed')                                # 處理後的資料夾
TRAIN_DIR = os.getenv('TRAIN_DIR', 'data/train')                                            # 訓練資料夾
TEST_DIR = os.getenv('TEST_DIR', 'data/test')                                               # 測試資料夾
FEATURE_IMPORTANCE_DIR = os.getenv('FEATURE_IMPORTANCE_DIR', 'output/feature_importance')   # 特徵重要性資料夾
EVALUATE_DIR = os.getenv('EVALUATE_DIR', 'output/evaluate')                                 # 評估資料夾
MODEL_DIR = os.getenv('MODEL_DIR', 'models/xgb_model.json')                                 # 模型路徑 
OUTPUT_PREDICT_DIR = os.getenv('OUTPUT_PREDICT_DIR', 'output/predict')                      # 預測資料夾

# 輸出檔案名稱
FERD_FILE = os.getenv('FERD_FILE', 'us_macro_data_2018_onwards.csv')                                                # FRED 資料
KLINE_ONE_H_RAW_FILE = os.getenv('KLINE_ONE_H_RAW_FILE', 'btc_1h_kline.csv')                                        # 原始K線資料
FEAR_GREED_RAW_FILE = os.getenv('FEAR_GREED_RAW_FILE', 'fear_greed_index.csv')                                      # 恐懼貪婪指數
PROCESSED_FILE = os.getenv('PROCESSED_FILE', 'btc_1h_kline_processed.csv')                                          # 處理後的K線資料
TRAIN_FILE = os.getenv('TRAIN_FILE', 'btc_1h_kline_train.csv')                                                      # 訓練資料
TEST_FILE = os.getenv('TEST_FILE', 'btc_1h_kline_test.csv')                                                         # 測試資料
FEATURE_IMPORTANCE_FILE = os.getenv('FEATURE_IMPORTANCE_FILE', 'feature_importance.csv')                            # 特徵重要性資料
FEATURE_IMPORTANCE_PLOT_FILE = os.getenv('FEATURE_IMPORTANCE_PLOT_FILE', 'feature_importance.png')                  # 特徵重要性圖表
EVALUATE_FILE = os.getenv('EVALUATE_FILE', 'evaluate.csv')                                                          # 評估資料
OUTPUT_PREDICT_FILE = os.getenv('OUTPUT_PREDICT_FILE', 'predict.csv')                                               # 預測資料
CONFUSION_DETAIL_FILE = os.getenv('CONFUSION_DETAIL_FILE', 'confusion_matrix.csv')                                  # 混淆矩陣詳細結果
PRECISION_RECALL_THRESHOLD_FILE = os.getenv('PRECISION_RECALL_THRESHOLD_FILE', 'precision_recall_threshold.png')    # Precision / Recall / F1 vs Threshold
BTC_ONCHAIN_FILE = os.getenv('BTC_ONCHAIN_FILE','btc_onchain_coinmetrics_daily_2009_2025.csv')                             # BTC 鏈上宏觀資料

# 這些欄位在訓練和預測時要排除
NON_FEATURE_COLS = ['timestamp','label', 'fear_greed_level']


TOP_SELECTED_FEATURES_30 = [
    'atr_14', 'bb_std', 'FlowInExNtv', 'bb_percent_b', 'price_vs_sma50_volume',
    'bb_distance', 'FlowOutExNtv', 'CPIAUCSL', 'bb_middle', 'FeeMeanUSD',
    'volume', 'CapRealUSD', 'PCEPI', 'FeeMedUSD', 'CPILFESL',
    'low', 'candle_body', 'SplyAct1yr', 'bb_upper', 'NVTAdjFF',
    'CapMVRVCur', 'ema_10', 'sma_50', 'fg_ema_3', 'CapMrktCurUSD',
    'rsi_14_lag_1', 'IssContUSD', 'fg_ema_7', 'sma_20', 'hour_cos'
]

TOP_SELECTED_FEATURES_40 = TOP_SELECTED_FEATURES_30 + [
    'FeeTotUSD', 'close', 'bb_lower', 'hour', 'price_vs_sma20',
    'AdrBalUSD100KCnt', 'kd_d', 'close_lag_1', 'fear_greed_value', 'weekday_sin'
]

TOP_SELECTED_FEATURES_50 = TOP_SELECTED_FEATURES_40 + [
    'high', 'weekday', 'AdrActCnt', 'rsi_14', 'NVTAdj',
    'weekday_cos', 'RevUSD', 'TxTfrCnt', 'open', 'volume_ma_20'
]

SELECTED_FEATURES = TOP_SELECTED_FEATURES_50

SELECTED_INITIAL_FEATURES = [
    "AdrActCnt", "TxCnt", "TxTfrCnt",
    "AdrBalUSD100KCnt", "AdrBalUSD1MCnt", "SplyAdrTop1Pct",
    "CapMVRVCur", "CapRealUSD", "CapMrktCurUSD",
    "NVTAdj", "NVTAdjFF",
    "FeeMeanUSD", "FeeTotUSD", "FeeMedUSD",
    "FlowInExNtv", "FlowOutExNtv",
    "RevUSD", "IssContUSD", "SplyAct1yr"
]