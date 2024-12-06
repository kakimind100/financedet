import pandas as pd
import joblib
import logging
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import holidays  # 공휴일 정보 가져오기

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
        logging.debug(f"CSV 파일 경로: {file_path}")

        df = pd.read_csv(file_path)
        logging.info(f"주식 데이터를 '{file_path}'에서 성공적으로 가져왔습니다.")
        logging.debug(f"데이터프레임 정보:\n{df.info()}")
        logging.debug(f"가져온 데이터 길이: {len(df)}")
        
        # 날짜 형식 변환
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        logging.debug(f"가져온 데이터의 첫 5행:\n{df.head()}")
        return df
    except Exception as e:
        logging.error(f"주식 데이터 가져오기 중 오류 발생: {e}")
        return None

def prepare_data(df):
    """데이터를 준비하고 훈련/검증/테스트 세트로 분할하는 함수."""
    # 피처와 타겟 변수 정의
    X = df.drop(columns=['Code', 'Date', 'Close'])  # Close는 예측 목표
    y = (df['Close'].shift(-1) > df['Close']).astype(int)  # 다음 날 종가 상승 여부

    # 데이터 분할
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def train_model_with_hyperparameter_tuning():
    """모델을 훈련시키고 하이퍼파라미터를 튜닝하는 함수."""
    logging.info("모델 훈련 시작...")
    
    # 데이터 가져오기
    df = fetch_stock_data()
    if df is None:
        logging.error("주식 데이터를 가져오는 데 실패했습니다.")
        return None

    # 데이터 준비 및 분할
    X_train, X_valid, X_test, y_train, y_valid, y_test = prepare_data(df)

    # 하이퍼파라미터 튜닝을 위한 GridSearchCV 설정
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               scoring='accuracy', cv=3, verbose=2, n_jobs=-1)
    
    logging.info("하이퍼파라미터 튜닝 시작...")
    grid_search.fit(X_train, y_train)  # 하이퍼파라미터 튜닝
    
    # 최적의 하이퍼파라미터 출력
    logging.info(f"최적의 하이퍼파라미터: {grid_search.best_params_}")
    return grid_search.best_estimator_  # 최적 모델 반환

def is_business_day(date):
    """주어진 날짜가 거래일인지 확인하는 함수."""
    kr_holidays = holidays.KR()  # 한국 공휴일 객체 생성
    is_weekday = date.weekday() < 5  # 월~금: 0~4
    is_holiday = date in kr_holidays  # 공휴일 확인
    return is_weekday and not is_holiday  # 거래일은 평일이면서 공휴일이 아님

def predict_future_trading_days(model, stock_code):
    """향후 26 거래일의 상승 여부를 예측하는 함수."""
    logging.info(f"{stock_code}에 대한 향후 26 거래일 예측 시작...")
    df = fetch_stock_data()
    if df is None or df.empty:
        logging.error("데이터프레임이 None이거나 비어 있습니다. 예측을 중단합니다.")
        return []

    features = ['RSI', 'MACD', 'Bollinger_High', 'Bollinger_Low', 'EMA20', 'EMA50', 'ATR', 'Volume']
    today_data = df[df['Code'] == stock_code].tail(1)
    if today_data.empty:
        logging.warning(f"{stock_code}의 오늘 데이터가 없습니다. 예측을 중단합니다.")
        return []

    future_predictions = []
    today_date = today_data['Date'].values[0]
    logging.info(f"{stock_code} 오늘 날짜: {today_date}")

    future_date = today_date
    for i in range(26):
        future_date += pd.DateOffset(days=1)
        while not is_business_day(future_date):
            logging.debug(f"{future_date}는 거래일이 아닙니다. 다음 날로 이동합니다.")
            future_date += pd.DateOffset(days=1)

        logging.info(f"유효한 거래일: {future_date}")

        if future_date not in df['Date'].dt.date.values:
            logging.warning(f"{future_date}에 대한 데이터가 존재하지 않습니다.")
            continue

        future_data = today_data.copy()
        future_data['Date'] = future_date

        future_features = future_data[features].values
        logging.debug(f"{stock_code} 예측에 사용될 피처 데이터 (날짜: {future_date}): {future_features}")

        try:
            prediction = model.predict(future_features)
            logging.info(f"{stock_code} {future_date}에 대한 예측 결과: {'상승' if prediction[0] == 1 else '하락'}")
            future_predictions.append((future_date, prediction[0]))
        except Exception as e:
            logging.error(f"예측 중 오류 발생: {e}")

    if future_predictions:
        logging.info(f"{stock_code} 향후 26 거래일 예측 완료. 예측 결과:")
        for date, pred in future_predictions:
            logging.info(f"{date}: {'상승' if pred == 1 else '하락'}")
    else:
        logging.warning(f"{stock_code} 예측 결과가 없습니다.")

    return future_predictions

def main():
    logging.info("모델 훈련 스크립트 실행 중...")
    best_model = train_model_with_hyperparameter_tuning()
    if best_model:
        stock_code = '000490'  # 예시 종목 코드, 필요에 따라 변경 가능
        future_predictions = predict_future_trading_days(best_model, stock_code)
        if future_predictions:
            for date, prediction in future_predictions:
                logging.info(f"{stock_code} {date.date()}: 예측된 상승 여부: {'상승' if prediction == 1 else '하락'}")

if __name__ == "__main__":
    main()
