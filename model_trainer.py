import pandas as pd
import joblib
import logging
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import holidays  # 한국 공휴일 정보 가져오기

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

        dtype = {
            'Code': 'object',
            'Date': 'str',  # 초기에는 문자열로 읽음
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
            'MACD_Hist': 'float',
            'Bollinger_High': 'float',
            'Bollinger_Low': 'float',
            'Stoch': 'float',
            'RSI': 'float',
            'ATR': 'float',
            'CCI': 'float',
            'EMA20': 'float',
            'EMA50': 'float',
            'Momentum': 'float',
            'Williams %R': 'float',
            'ADX': 'float',
            'Volume_MA20': 'float',
            'ROC': 'float',
            'CMF': 'float',
            'OBV': 'float',
        }

        df = pd.read_csv(file_path, dtype=dtype)
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

def predict_future_trading_days(model, stock_code):
    """향후 26 거래일의 상승 여부를 예측하는 함수."""
    logging.info(f"{stock_code}에 대한 향후 26 거래일 예측 시작...")
    df = fetch_stock_data()  # 주식 데이터 가져오기
    if df is None or df.empty:
        logging.error("데이터프레임이 None이거나 비어 있습니다. 예측을 중단합니다.")
        return []

    features = [
        'RSI', 'MACD', 'Bollinger_High', 'Bollinger_Low', 
        'EMA20', 'EMA50', 'ATR', 'Volume'
    ]

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

        # future_date에 대한 데이터 존재 여부 확인
        if future_date not in df['Date'].dt.date.values:
            logging.warning(f"{future_date}에 대한 데이터가 존재하지 않습니다.")
            continue

        # 복사본 생성
        future_data = today_data.copy()
        future_data['Date'] = future_date  # 예측할 날짜로 설정

        # 피처 데이터 준비
        future_features = future_data[features].values
        logging.debug(f"{stock_code} 예측에 사용될 피처 데이터 (날짜: {future_date}): {future_features}")

        # 예측 수행
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
    # 모델 훈련 및 예측 실행
    best_model = train_model_with_hyperparameter_tuning()
    if best_model:
        stock_code = '000490'  # 예시 종목 코드, 필요에 따라 변경 가능
        future_predictions = predict_future_trading_days(best_model, stock_code)  # 종목 코드 전달
        if future_predictions:
            for date, prediction in future_predictions:
                logging.info(f"{stock_code} {date.date()}: 예측된 상승 여부: {'상승' if prediction == 1 else '하락'}")
            
            # 매수 신호 식별
            buy_signals = identify_buy_signals(future_predictions)
            logging.info(f"{stock_code} 매수 신호: {buy_signals}")

            # 상승률 계산
            df = fetch_stock_data()  # 주식 데이터 가져오기
            if df is not None:
                top_stocks = calculate_returns(future_predictions, df)
                print("상승률이 가장 높은 종목 20개:")
                for stock in top_stocks:
                    logging.info(f"매수 시점: {stock[0].date()}, 매도 시점: {stock[1].date()}, 상승률: {stock[2]:.2f}%")
                    print(f"매수 시점: {stock[0].date()}, 매도 시점: {stock[1].date()}, 상승률: {stock[2]:.2f}%")

            # 주식 차트 그리기
            plot_stock_chart(df)

# 메인 함수 실행
if __name__ == "__main__":
    main()
