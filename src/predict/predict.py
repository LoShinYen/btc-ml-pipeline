import os
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_recall_curve

from src.config import (
    TEST_DIR, TEST_FILE, MODEL_DIR, OUTPUT_PREDICT_DIR, OUTPUT_PREDICT_FILE, 
    NON_FEATURE_COLS , SELECTED_FEATURES
)


def load_test_data():
    """
    載入測試資料

    Returns:
        X_test (pd.DataFrame): 特徵集
        y_true (pd.Series): 標籤集
        original_df (pd.DataFrame): 原始資料
    """
    test_path = os.path.join(TEST_DIR, TEST_FILE)
    df = pd.read_csv(test_path)
    
    y_true = df['label'] if 'label' in df.columns else None
    # X_test = df[SELECTED_FEATURES]

    print(f"✅ 移除非特徵欄位: {NON_FEATURE_COLS}")
    feature_cols = [col for col in df.columns if col not in NON_FEATURE_COLS]
    X_test = df[feature_cols]

    file_count = X_test.shape[0]
    feature_count = X_test.shape[1]
    print(f"✅ 初步載入資料，共 {file_count} 筆，欄位數量 {feature_count} 個")

    return X_test, y_true, df

def predict_with_threshold(X_test, y_true, original_df):
    """
    預測、轉換、計算指標

    Args:
        X_test (pd.DataFrame): 特徵集
        y_true (pd.Series): 標籤集
        original_df (pd.DataFrame): 原始資料（含 timestamp, 特徵欄位等）
    """
    # 載入模型
    model = xgb.XGBClassifier()
    model.load_model(MODEL_DIR)

    # 預測機率
    y_proba = model.predict_proba(X_test)[:, 1]  # 取 1 類的機率

    # 找最佳閥值
    threshold = find_optimal_threshold(y_true, y_proba)
    print(f"✅ 最佳閥值: {threshold:.4f}")

    # 根據閥值轉換為預測類別
    y_pred = (y_proba >= threshold).astype(int)

    # ✅ 加入真實標籤
    original_df['label'] = y_true
    original_df['predicted_probability'] = y_proba
    original_df['predicted_label'] = y_pred

    # 儲存 CSV
    os.makedirs(OUTPUT_PREDICT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_PREDICT_DIR, OUTPUT_PREDICT_FILE)
    original_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ 預測完成並已儲存到 {output_path}")



def find_optimal_threshold(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = f1.argmax()
    return thresholds[best_idx]



def main():
    try:
        # 載入測試資料
        X_test, y_true, original_df = load_test_data()
        
        # 預測
        predict_with_threshold(X_test, y_true, original_df)
        
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")
