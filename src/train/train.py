import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from src.config import TRAIN_DIR, TRAIN_FILE , NON_FEATURE_COLS, SELECTED_FEATURES
from imblearn.over_sampling import SMOTE
from src.analysis.feature_analysis import save_feature_importance_csv
from src.train.timeseries_cv_search import cross_validate_xgboost_with_early_stopping

def load_train_data():
    """
    讀取完整的訓練資料 DataFrame
    Returns:
        X (pd.DataFrame): 特徵集
        y (pd.Series): 標籤集
    """
    train_path = os.path.join(TRAIN_DIR, TRAIN_FILE)
    df = pd.read_csv(train_path)

    y = df['label']
    X = df[SELECTED_FEATURES]
    
    # feature_cols = [col for col in df.columns if col not in NON_FEATURE_COLS]
    print(f"✅ 初步載入資料，共 {df.shape[0]} 筆，欄位數量 {df.shape[1]} 個")
    # print(f"✅ 特徵欄位：{feature_cols}")
    # X = df[feature_cols]

    return X, y

def train_xgboost_with_cv(X, y):
    """
    使用 TimeSeriesSplit 搭配 GridSearchCV 自動調參訓練 XGBoost，並使用 early stopping。
    Args:
        X (pd.DataFrame): 特徵集
        y (pd.Series): 標籤集
    Returns:
        model: 訓練完成的最佳模型
    """
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)
    X['__index__'] = np.arange(len(X))
    X = X.sort_values('__index__').drop(columns='__index__')
    y = y.loc[X.index]
    
    base_params  = {
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'scale_pos_weight': 1,
        'early_stopping_rounds': 50,
        'random_state': 42,
        # 'verbose': 100
    }

    # 超參數搜尋空間
    param_grid_dict  = {
        'max_depth': [5, 6],
        'learning_rate': [0.01, 0.05],
        'n_estimators': [500, 1000],
        'subsample': [0.8],
        'colsample_bytree': [0.8, 1.0]
    }
    
    print(f"✅ 開始進行 TimeSeriesSplit 搜尋最佳參數")

    best_params = cross_validate_xgboost_with_early_stopping(
        X, y,
        param_grid_dict,
        base_params=base_params,
        n_splits=5,
        metric='aucpr'
    )

    print(f"✅ 開始訓練 final_model")
    
    # ⭐ 建立一個新的 final_model
    final_model = xgb.XGBClassifier( **base_params, **best_params)

    # 切分 Train/Val
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, shuffle=False)
    
    final_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    
    print('✅ Final Model 訓練完成')

    return final_model

def train_xgboost_fast(X, y):
    from imblearn.over_sampling import SMOTE

    # ✅ 時序切分（保留前 80%）
    split_index = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_val = y.iloc[:split_index], y.iloc[split_index:]

    # ✅ 模型參數
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'max_depth': 7,
        'learning_rate': 0.03,
        'n_estimators': 1500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': 1,
        'early_stopping_rounds': 50,
        'random_state': 42,
        "scale_pos_weight" : 12.6
    }

    model = xgb.XGBClassifier(**params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],  # ⚠️ 未平衡資料才可作為驗證指標
        verbose=100
    )

    print("✅ 簡易訓練完成（保留時序、訓練集平衡）")
    return model

def save_model_with_confirm(model):
    """
    儲存模型

    Args:
        model (xgboost.XGBClassifier): 訓練好的模型
    """
    os.makedirs('./models/', exist_ok=True)
    base_path = './models/xgb_model.json'

    # if os.path.exists(base_path):
    #     answer = input(f"⚠️ 模型檔案 {base_path} 已存在，是否要覆寫？(y/n): ").strip().lower()
    #     if answer != 'y':
    #         timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    #         base_name = f"./models/xgb_model_{timestamp}.json"
    #         print(f"📂 自動儲存新模型名：{base_name}")
    #         model.save_model(base_name)
    #         print(f"✅ 模型已成功儲存到 {base_name}")
    #         return
        
    model.save_model(base_path)
    print(f"✅ 模型已成功儲存到 {base_path}")

def main():
    """
    主流程
    """
    try:
        # Step 1: 載入訓練資料  
        X,y = load_train_data()

        # Step 2: 訓練模型
        # model = train_xgboost_with_cv(X,y)
        model = train_xgboost_fast(X,y)

        # Step 3: 儲存模型
        save_model_with_confirm(model)

        # Step 4: 儲存特徵重要性
        save_feature_importance_csv(model, X.columns.tolist())
        
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")
