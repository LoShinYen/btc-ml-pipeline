from src.processing.fetch_ferd import main as load_FERD
from src.processing.fetch_kline_data import main as load_kline_data
from src.processing.fetch_fear_greed import main as load_fear_greed
from src.processing.generate_feature import main as load_generate_feature

def prepare_training_data():
    load_FERD()
    load_kline_data()
    load_fear_greed()
    load_generate_feature()

def main():
    prepare_training_data()
    
if __name__ == "__main__":
    main()
