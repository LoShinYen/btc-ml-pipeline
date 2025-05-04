import os
import pandas as pd
from fredapi import Fred
from src.config import RAW_DIR, FERD_FILE, FERD_API_KEY

def main():
    API_KEY = FERD_API_KEY
    fred = Fred(api_key=API_KEY)

    # 觀察起始時間
    START_DATE = '2018-01-01'

    # 🧠 各項經濟指標定義與 FRED 代碼
    indicators = {
        # ▶️ 通膨指標
        "CPIAUCSL": "CPI 全指數：美國城市消費者物價總指數（不季調）",
        "CPILFESL": "核心 CPI：排除食品與能源的 CPI（不季調）",
        
        # ▶️ 消費支出指標
        "PCEPI": "PCE 價格指數：個人消費支出價格（含食品能源）",
        "PCEPILFE": "核心 PCE：剔除食品與能源的個人消費價格指數",
        
        # ▶️ 利率相關
        "FEDFUNDS": "聯邦基金有效利率：美國基準短期利率",
        
        # ▶️ 經濟成長指標
        # "GDP": "美國實質 GDP（季調年率）",
        # "GDPC1": "實質 GDP（Chain-Weighted 美元）",
    }

    # 🚀 抓取所有指標資料
    df_all = pd.DataFrame()

    for code, desc in indicators.items():
        print(f"抓取中：{desc} ({code})")
        series = fred.get_series(code, observation_start=START_DATE)
        series.name = code
        df_all = pd.concat([df_all, series], axis=1)

    # 🧼 處理缺值（可視情況填補 / 保留）
    df_all = df_all.dropna(how='all')  # 移除全部為空的列

    # 💾 儲存為 CSV 供模型訓練使用
    df_all.to_csv(os.path.join(RAW_DIR, FERD_FILE))
    print(f"✅ 所有指標資料已儲存至 {FERD_FILE}")


if __name__ == "__main__":
    main()
