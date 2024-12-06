import pandas as pd
import numpy as np
import logging
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime

# 로그 디렉토리 설정
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# 로깅 설정
logging.basicConfig(
    filename=os.path.join(log_dir, 'lstm_stock_predictor.log'),
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def fetch_stock_data():
    """주식 데이터를 가져오는 함수 (CSV 파일에서)."""
    logging.debug("주식 데이터를 가져오는 중...")
    file_path = 'data/stock_data_with_indicators.csv'
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])  # 날짜 형식 변환
    logging.info("주식 데이터를 성공적으로 가져왔습니다. 데이터 크기: %s", df.shape)

    return df

def prepare_data(df):
    """데이터를 준비하는 함수."""
    logging.debug("데이터 준비 중...")
    df = df[['Date', 'Close']].set_index('Date')
    df = df.sort_index()

    # Min-Max 스케일링
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    # 데이터셋 생성
    x_data, y_data = [], []
    for i in range(60, len(scaled_data)):
        x_data.append(scaled_data[i-60:i, 0])  # 이전 60일 데이터
        y_data.append(scaled_data[i, 0])      # 오늘 가격

    x_data, y_data = np.array(x_data), np.array(y_data)
    x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))  # LSTM 입력 형식

    logging.info("데이터 준비 완료. 입력 데이터 크기: %s", x_data.shape)
    return x_data, y_data, scaler

def create_lstm_model():
    """LSTM 모델을 생성하는 함수."""
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(60, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # 가격 예측

    model.compile(optimizer='adam', loss='mean_squared_error')
    logging.info("LSTM 모델 생성 완료.")
    return model

def main():
    # 데이터 로드
    logging.info("LSTM 모델 훈련 스크립트 실행 중...")
    df = fetch_stock_data()

    # 데이터 준비
    x_data, y_data, scaler = prepare_data(df)

    # 모델 생성
    model = create_lstm_model()

    # 모델 훈련
    model.fit(x_data, y_data, epochs=50, batch_size=32)
    logging.info("모델 훈련 완료.")

    # 예측
    predictions = model.predict(x_data)
    predictions = scaler.inverse_transform(predictions)  # 원래 스케일로 복원

    # 예측 결과 로그
    for i in range(len(predictions)):
        logging.info("예측 날짜: %s, 예측 가격: %.2f", df.index[i + 60], predictions[i][0])

    logging.info("LSTM 모델 훈련 스크립트 실행 완료.")

if __name__ == "__main__":
    main()
