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
    filename=os.path.join(log_dir, 'stock_data_fetcher.log'),  # 로그 파일 경로
    level=logging.DEBUG,  # 로깅 레벨을 DEBUG로 설정
    format='%(asctime)s - %(levelname)s - %(message)s'  # 로그 메시지 형식
)

# 콘솔 로그 출력 설정
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # 콘솔 로그 레벨을 DEBUG로 설정
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))  # 콘솔 로그 형식
logging.getLogger().addHandler(console_handler)  # 콘솔 핸들러 추가

def fetch_stock_data():
    """주식 데이터를 가져오는 함수 (CSV 파일에서)."""
    logging.debug("주식 데이터를 가져오는 중...")
    try:
        # 파일 경로 설정
        file_path = os.path.join('data', 'stock_data_with_indicators.csv')
        
        # 데이터 타입 지정
        dtype = {
            'Code': 'object',
            'Date': 'str',  # 날짜는 나중에 datetime으로 변환할 예정
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

        # 날짜 열을 datetime 형식으로 변환
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

        # 데이터프레임 정보 및 데이터 타입 로그
        logging.debug(f"데이터프레임 정보:\n{df.info()}")
        logging.debug(f"데이터프레임 첫 5행:\n{df.head()}")

        return df
    except FileNotFoundError:
        logging.error(f"파일 '{file_path}'을(를) 찾을 수 없습니다.")
        return None
    except pd.errors.EmptyDataError:
        logging.error("CSV 파일이 비어 있습니다.")
        return None
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
        features = ['MA5', 'MA20', 'RSI']
        # 다음 날 종가가 현재 종가의 1.29배 이상인 경우를 타겟으로 설정
        df['Target'] = np.where(df['Close'].shift(-1) >= df['Close'] * 1.29, 1, 0)  # 29% 상승 여부

        # NaN 제거
        df.dropna(subset=features + ['Target'], inplace=True)

        X = df[features]
        y = df['Target']

        # 피쳐와 타겟 변수의 데이터 타입 로그
        logging.debug(f"피쳐 데이터 타입:\n{X.dtypes}")
        logging.debug(f"타겟 변수 데이터 타입:\n{y.dtypes}")

        logging.debug(f"훈련에 사용된 피쳐: {features}")
        logging.debug(f"타겟 변수: 'Target'")

        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 모델 훈련
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # models 디렉토리 존재 여부 확인 및 생성
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)

        # 모델 저장
        joblib.dump(model, os.path.join(models_dir, 'stock_model.pkl'))  # models 디렉토리에 저장
        logging.info("모델 훈련 완료 및 'models/stock_model.pkl'로 저장되었습니다.")

        # 모델 평가
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        logging.info(f"모델 성능 보고서:\n{report}")
        print(report)

    except Exception as e:
        logging.error(f"모델 훈련 중 오류 발생: {e}")

if __name__ == "__main__":
    logging.info("모델 훈련 스크립트 실행 중...")
    train_model()  # 모델 훈련
    logging.info("모델 훈련 스크립트 실행 완료.")
