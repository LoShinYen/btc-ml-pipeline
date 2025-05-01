import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

from src.config import FEATURE_IMPORTANCE_DIR, FEATURE_IMPORTANCE_FILE, FEATURE_IMPORTANCE_PLOT_FILE, EVALUATE_DIR, PRECISION_RECALL_THRESHOLD_FILE

def plot_feature_importance_csv(top_n=30):
    """
    繪製特徵重要性
    
    Args:
        top_n (int): 繪製前 N 個特徵
    """
    csv_path = os.path.join(FEATURE_IMPORTANCE_DIR, FEATURE_IMPORTANCE_FILE)
    df = pd.read_csv(csv_path)
    df = df.sort_values(by='importance', ascending=False)

    plt.figure(figsize=(10, 8))
    plt.barh(df['feature'][:top_n][::-1], df['importance'][:top_n][::-1])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importances')
    plt.tight_layout()
    plt.savefig(os.path.join(FEATURE_IMPORTANCE_DIR, FEATURE_IMPORTANCE_PLOT_FILE))
    plt.show()
    print(f"✅ 特徵重要性圖表已儲存到 {os.path.join(FEATURE_IMPORTANCE_DIR, FEATURE_IMPORTANCE_PLOT_FILE)}")
    
def plot_precision_recall_threshold(y_true, y_proba):
    """
    繪製 Precision / Recall / F1 vs Threshold
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    plt.figure(figsize=(10,6))
    plt.plot(thresholds, precision[:-1], label='Precision')
    plt.plot(thresholds, recall[:-1], label='Recall')
    plt.plot(thresholds, f1[:-1], label='F1 Score')
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision / Recall / F1 vs Threshold")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(EVALUATE_DIR, PRECISION_RECALL_THRESHOLD_FILE))
    plt.show()
    print(f"✅ Precision / Recall / F1 vs Threshold 已儲存到 {os.path.join(EVALUATE_DIR, PRECISION_RECALL_THRESHOLD_FILE)}")