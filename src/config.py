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
FERD_FILE = os.getenv('FERD_FILE', 'us_macro_data_2018_onwards.csv')                                                # FRED 經濟指標資料    
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

# 這些欄位在訓練和預測時要排除
NON_FEATURE_COLS = ['timestamp','label', 'fear_greed_level']

# 選取特徵
TOP_SELECTED_FEATURES_30 = [
    'atr_14', 'CPIAUCSL', 'price_vs_sma50_volume', 'PCEPILFE', 'ema_10',
    'PCEPI', 'bb_std', 'fg_ema_3', 'CPILFESL', 'weekday_sin',
    'sma_20', 'fg_ema_7', 'close_lag_1', 'FEDFUNDS', 'sma_100',
    'low', 'bb_upper', 'weekday', 'fear_greed_label', 'close',
    'bb_lower', 'price_vs_sma20', 'sma_50', 'macd_signal', 'volume',
    'hour_cos', 'fg_zscore_30', 'hour', 'bb_middle', 'volume_ma_20'
]

TOP_SELECTED_FEATURES_40 = TOP_SELECTED_FEATURES_30 + [
    'rsi_14_lag_1', 'open_zscore', 'sma_convergence', 'fear_greed_value',
    'candle_body', 'bb_distance', 'macd_line', 'high_zscore', 'low_zscore',
    'ema_10_diff'
]