import pandas as pd
import joblib
import logging
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 로그 디렉토리 설정
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# 로깅 설정
logging.basicConfig(
    filename=os.path.join(log_dir, 'stock_data_fetcher.log'),
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 콘솔 로그 출력 설정
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)

def fetch_stock_data():
    """주식 데이터를 가져오는 함수 (CSV 파일에서)."""
    logging.debug("주식 데이터를 가져오는 중...")
    try:
        file_path = os.path.join('data', 'stock_data_with_indicators.csv')
        
        dtype = {
            'Code': 'object',
            'Date': 'str',
            'Open': 'float',
            'High': 'float',
            'Low': 'float',
            'Close': 'float',
            'Volume': 'float',
            'Change': 'float',
            'MA5': 'float',
            'MA20': 'float',
            'MACD': 'float',
            'MACD_Signal': 'float',
            'Bollinger_High': 'float',
            'Bollinger_Low': 'float',
            'Stoch': 'float',
            'RSI': 'float',
            'ATR': 'float',
            'CCI': 'float',
            'EMA20': 'float',
            'EMA50': 'float'
        }

        df = pd.read_csv(file_path, dtype=dtype)
        logging.info(f"주식 데이터를 '{file_path}'에서 성공적으로 가져왔습니다.")

        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        return df
    except Exception as e:
        logging.error(f"주식 데이터 가져오기 중 오류 발생: {e}")
        return None

def train_model():
    """모델을 훈련시키고 저장하는 함수."""
    df = fetch_stock_data()  # 주식 데이터 가져오기
    if df is None:
        logging.error("데이터프레임이 None입니다. 모델 훈련을 중단합니다.")
        return

    try:
        features = ['MA5', 'MA20', 'RSI', 'MACD', 'Bollinger_High', 'Bollinger_Low', 'Stoch', 'ATR', 'CCI', 'EMA20', 'EMA50']
        df['Target'] = np.where(df['Close'].shift(-1) >= df['Close'] * 1.29, 1, 0)  # 29% 상승 여부

        # NaN 제거
        df.dropna(subset=features + ['Target'], inplace=True)

        X = df[features]
        y = df['Target']

        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 모델 훈련
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # 모델 저장
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(model, os.path.join(models_dir, 'stock_model.pkl'))
        logging.info("모델 훈련 완료 및 'models/stock_model.pkl'로 저장되었습니다.")

        # 모델 평가
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        logging.info(f"모델 성능 보고서:\n{report}")
        print(report)

    except Exception as e:
        logging.error(f"모델 훈련 중 오류 발생: {e}")

def predict_next_day():
    """다음 거래일의 상승 여부를 예측하는 함수."""
    df = fetch_stock_data()  # 주식 데이터 가져오기
    if df is None:
        logging.error("데이터프레임이 None입니다. 예측을 중단합니다.")
        return

    # 훈련된 모델 불러오기
    model = joblib.load(os.path.join('models', 'stock_model.pkl'))

    # 예측할 데이터 준비 (추가 피쳐 포함)
    features = ['MA5', 'MA20', 'RSI', 'MACD', 'Bollinger_High', 'Bollinger_Low', 'Stoch', 'ATR', 'CCI', 'EMA20', 'EMA50']
    predictions = {}

    for stock_code in df['Code'].unique():
        stock_data = df[df['Code'] == stock_code].tail(1)  # 마지막 하루 데이터 가져오기
        if not stock_data.empty:
            X_next = stock_data[features]
            pred = model.predict(X_next)
            predictions[stock_code] = pred[0]  # 예측 결과 저장

    # 예측 결과 출력
    for code, prediction in predictions.items():
        if prediction == 1:
            print(f"{code}는 다음 거래일에 29% 상승할 것으로 예측됩니다.")
        else:
            print(f"{code}는 다음 거래일에 상승하지 않을 것으로 예측됩니다.")

if __name__ == "__main__":
    logging.info("모델 훈련 스크립트 실행 중...")
    train_model()  # 모델 훈련
    logging.info("모델 훈련 스크립트 실행 완료.")

    logging.info("다음 거래일 예측 스크립트 실행 중...")
    predict_next_day()  # 다음 거래일 예측
    logging.info("다음 거래일 예측 스크립트 실행 완료.")
