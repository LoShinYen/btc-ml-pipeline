# cross_validate_xgboost.py
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
import itertools

def generate_param_combinations(param_grid_dict):
    keys = list(param_grid_dict.keys())
    values = list(param_grid_dict.values())
    combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    return combinations

def cross_validate_xgboost_with_early_stopping(
    X, y,
    param_grid_dict,
    base_params=None,
    n_splits=5,
    metric='aucpr'
):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_score = -np.inf
    best_params = None

    param_combinations = generate_param_combinations(param_grid_dict)
    print(f"\n✅ 共有 {len(param_combinations)} 組組合要測試")

    for idx, params in enumerate(param_combinations):
        fold_scores = []
        skip_count = 0

        for fold_id, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # 若驗證集中只有單一類別，跳過
            if len(np.unique(y_val)) < 2:
                print(f"⚠️ 第 {fold_id+1} fold 驗證集只有單一類別，已跳過")
                skip_count += 1
                continue

            model = xgb.XGBClassifier(
                **(base_params or {}),
                **params
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                # early_stopping_rounds=50,
                # eval_metric=metric,
                verbose=False
            )

            score = model.evals_result()['validation_0'][metric][-1]
            fold_scores.append(score)

        if skip_count == n_splits:
            print(f"❌ 所有 fold 都無效：參數 {params} 全部跳過\n")
            continue

        avg_score = np.mean(fold_scores)
        print(f"[{idx+1}/{len(param_combinations)}] 參數 {params} 平均 {metric}: {avg_score:.5f}")

        if avg_score > best_score:
            best_score = avg_score
            best_params = params

    if best_params is None:
        raise ValueError("❌ 無有效的交叉驗證結果，無法產生最佳參數。請檢查資料順序與 SMOTE 使用。")

    print(f"\n🏆 最佳參數: {best_params}，平均 {metric}: {best_score:.5f}")
    return best_params