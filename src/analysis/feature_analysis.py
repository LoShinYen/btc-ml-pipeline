import pandas as pd
import os
from src.config import FEATURE_IMPORTANCE_DIR, FEATURE_IMPORTANCE_FILE

def save_feature_importance_csv(model, feature_names):
    """
    儲存特徵重要性
    
    Args:
        model (xgboost.XGBClassifier): 訓練好的模型
        feature_names (list): 特徵名稱
    """
    importance_scores = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    }).sort_values(by='importance', ascending=False)

    os.makedirs(FEATURE_IMPORTANCE_DIR, exist_ok=True)
    save_path = os.path.join(FEATURE_IMPORTANCE_DIR, FEATURE_IMPORTANCE_FILE)
    feature_importance.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"✅ 特徵重要性已儲存到 {save_path}")

