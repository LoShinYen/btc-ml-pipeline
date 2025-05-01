import os
import pandas as pd
import src.visualization.plot_utils as plot_utils
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from src.config import EVALUATE_DIR, OUTPUT_PREDICT_DIR, OUTPUT_PREDICT_FILE, CONFUSION_DETAIL_FILE

def load_predict_data():
    """
    從 csv 檔案評估預測結果
    
    Returns:
        y_true (pd.Series): 真實標籤
        y_pred (pd.Series): 預測標籤
        y_proba (pd.Series): 預測機率
    """
    csv_path = os.path.join(OUTPUT_PREDICT_DIR, OUTPUT_PREDICT_FILE)
    df = pd.read_csv(csv_path)
    y_true = df['label']
    y_pred = df['predicted_label']
    y_proba = df['predicted_probability']
    return y_true, y_pred, y_proba

def evaluate_predictions(y_true, y_pred):
    """
    計算評估指標

    Args:
        y_true (pd.Series): 真實標籤
        y_pred (pd.Series): 預測標籤

    Returns:
        acc (float): 準確率
        prec (float): 精確率
        rec (float): 召回率
        f1 (float): F1 Score
    """
    if y_true is None:
        print("⚠️ 測試資料沒有 label，無法評估")
        return

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n📈 Evaluate Results:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\n詳細分類報告:")
    print(classification_report(y_true, y_pred, zero_division=0))

def save_confusion_detail(y_true, y_pred):
    """
    儲存混淆矩陣詳細結果
    """
    result = annotate_confusion_results(y_true, y_pred)
    
    os.makedirs(EVALUATE_DIR, exist_ok=True)
    result.to_csv(os.path.join(EVALUATE_DIR, CONFUSION_DETAIL_FILE), index=False)
    
    print(f"✅ 混淆矩陣詳細結果已儲存到 {os.path.join(EVALUATE_DIR, CONFUSION_DETAIL_FILE)}")   

def annotate_confusion_results(y_true, y_pred):
    """
    傳入 y_true, y_pred，回傳每筆資料的 TP/FP/FN/TN 分類標籤與統計報告。
    """
    result = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred
    })

    def classify(row):
        if row['y_true'] == 1 and row['y_pred'] == 1:
            return 'TP'
        elif row['y_true'] == 1 and row['y_pred'] == 0:
            return 'FN'
        elif row['y_true'] == 0 and row['y_pred'] == 1:
            return 'FP'
        else:
            return 'TN'

    result['confusion'] = result.apply(classify, axis=1)

    # 統計總表
    stats = result['confusion'].value_counts().to_frame(name='count')
    stats['rate (%)'] = (stats['count'] / len(result) * 100).round(2)

    print("📊 混淆矩陣分類結果統計：")
    print(stats)


    return result

def main():
    """
    評估預測結果
    """
    try:
        # 載入預測資料
        y_true, y_pred, y_proba = load_predict_data()

        # 評估預測結果
        evaluate_predictions(y_true, y_pred)

        # 儲存混淆矩陣詳細結果
        save_confusion_detail(y_true, y_pred)

        # 繪製 Precision / Recall / F1 vs Threshold
        plot_utils.plot_precision_recall_threshold(y_true, y_proba)
        
        # 繪製特徵重要性
        plot_utils.plot_feature_importance_csv()
        
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")


