import time
import src.processing as processing
import src.train.train as train
import src.predict.predict as predict
import src.evaluate.evaluate_results as evaluate_results

def header_messge(message):
    print("================================================")
    print(f'-----------{message}-----------')
    print("================================================")


def run_prepare_data():
    """
    資料準備
    """
    header_messge('Step 0: 資料準備')
    processing.main()
    
def run_train():
    """
    訓練模型
    """
    header_messge('Step 1: 訓練模型')

    # 訓練模型
    train.main()

def run_predict():
    """
    預測模型
    """
    header_messge('Step 2: 預測模型')
    predict.main()

def run_evaluate():
    """
    模型評估
    """
    header_messge('Step 3: 模型評估')
    
    evaluate_results.main()


if __name__ == "__main__":
    try:
        start_time = time.time()
        
        # step 0 資料準備
        run_prepare_data()

        # step 1 訓練模型
        run_train()
    
        # step 2 預測模型
        run_predict()

        # step 3 模型評估
        run_evaluate()

        end_time = time.time()
        
        print(f"執行時間: {end_time - start_time} 秒")
    except Exception as e:
        print(f"發生錯誤: {e}")





