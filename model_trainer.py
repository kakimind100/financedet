import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def fetch_stock_data():
    """주식 데이터를 가져오는 함수 (CSV 파일에서)."""
    file_path = 'data/stock_data_with_indicators.csv'
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def prepare_data(df):
    """데이터를 준비하는 함수."""
    # 필요한 열 선택 (종가와 기술적 지표)
    df = df[['Date', 'Close', 'MA5', 'MA20', 'MACD', 'RSI', 'Bollinger_High', 'Bollinger_Low']].dropna().set_index('Date')
    df = df.sort_index()

    # Min-Max 스케일링
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    # 입력(X)과 출력(y) 데이터 생성
    x_data, y_data = [], []
    for i in range(60, len(scaled_data) - 26):
        x_data.append(scaled_data[i-60:i])  # 지난 60일 데이터
        y_data.append(scaled_data[i:i + 26, 1])  # 향후 26일 종가

    x_data, y_data = np.array(x_data), np.array(y_data)
    return x_data, y_data, scaler

def create_and_train_model(X_train, y_train):
    """XGBoost 모델을 생성하고 훈련하는 함수."""
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train)
    return model

def predict_future_prices(model, last_60_days):
    """향후 26일 가격을 예측하는 함수."""
    predictions = model.predict(last_60_days)
    return predictions

def generate_signals(predictions):
    """매수 및 매도 신호를 생성하는 함수."""
    buy_signals = []
    sell_signals = []

    # 최저점과 최고점 찾기
    min_price = np.min(predictions)
    max_price = np.max(predictions)

    # 최저점에서 매수 신호 생성
    min_index = np.where(predictions == min_price)[0][0]
    buy_signals.append(min_index)

    # 매수 신호 이후 최고점에서 매도 신호 생성
    for i in range(min_index + 1, len(predictions)):
        if predictions[i] >= max_price:
            sell_signals.append(i)
            break

    return buy_signals, sell_signals

def main():
    # 데이터 로드
    df = fetch_stock_data()

    # 데이터 준비
    x_data, y_data, scaler = prepare_data(df)

    # 훈련 및 테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    # 모델 훈련
    model = create_and_train_model(X_train, y_train)

    # 가장 최근 60일 데이터를 사용하여 향후 26일 가격 예측
    last_60_days = x_data[-1].reshape(1, -1)
    future_predictions = predict_future_prices(model, last_60_days)

    # 예측 결과를 원래 스케일로 복원
    future_prices = scaler.inverse_transform(future_predictions.reshape(-1, 1))

    # 매수 및 매도 신호 생성
    buy_signals, sell_signals = generate_signals(future_prices.flatten())

    print("향후 26일 가격 예측:", future_prices.flatten())
    print("매수 신호 발생일 (인덱스):", buy_signals)
    print("매도 신호 발생일 (인덱스):", sell_signals)

if __name__ == "__main__":
    main()
