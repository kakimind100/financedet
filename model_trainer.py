import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

def fetch_stock_data():
    """주식 데이터를 가져오는 함수 (CSV 파일에서)."""
    file_path = 'data/stock_data_with_indicators.csv'
    df = pd.read_csv(file_path, dtype={'Code': str})  # 'Code' 열을 문자열로 읽어오기
    df['Date'] = pd.to_datetime(df['Date'])
    print("데이터 로드 완료. 열 목록:")
    print(df.columns.tolist())  # 로드된 데이터의 열 목록 출력
    return df

def prepare_data(df):
    """현재 날짜 기준 최근 60일 데이터를 준비."""
    df = df[['Date', 'Close', 'Change', 'EMA20', 'EMA50', 'RSI', 'MACD', 
              'MACD_Signal', 'Bollinger_High', 'Bollinger_Low', 'Stoch', 
              'Momentum', 'ADX']].dropna()

    # 날짜를 기준으로 정렬
    df = df.sort_values('Date')

    # 현재 날짜 기준으로 최근 60일 데이터만 선택
    today = df['Date'].max()
    recent_data = df[df['Date'] > today - pd.Timedelta(days=60)]
    
    if len(recent_data) < 60:
        raise ValueError("최근 60일 데이터가 부족합니다. 훈련을 진행할 수 없습니다.")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(recent_data.set_index('Date'))

    x_data, y_data = [], []
    # 60일 데이터를 입력으로 사용하여 예측
    for i in range(60, len(scaled_data) - 26):
        x_data.append(scaled_data[i-60:i])  # 이전 60일 데이터
        y_data.append(scaled_data[i + 25, 0])  # 26일 후의 종가 (Close)

    x_data, y_data = np.array(x_data), np.array(y_data)
    print(f"준비된 데이터 샘플 수: x_data={len(x_data)}, y_data={len(y_data)}")
    return x_data, y_data.reshape(-1, 1), scaler

def create_and_train_model(X_train, y_train):
    """모델을 생성하고 훈련하는 함수."""
    print("모델 훈련 시작...")
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100,
                              colsample_bytree=0.3, learning_rate=0.1,
                              max_depth=5, alpha=10, n_jobs=-1)
    model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    print("모델 훈련 완료.")
    return model

def predict_future_prices(model, last_60_days):
    """모델을 사용하여 향후 26일 가격을 예측하는 함수."""
    predictions = model.predict(last_60_days.reshape(1, -1))
    print(f"예측된 미래 가격: {predictions}")
    return predictions

def generate_signals(predictions):
    """예측 결과를 기반으로 매수 및 매도 신호를 생성하는 함수."""
    # 음수 예측 값 제거
    predictions = np.maximum(predictions, 0)

    buy_index = np.argmin(predictions)  # 최저점 인덱스
    if buy_index + 1 >= len(predictions):
        return buy_index, buy_index  # 매수와 매도 시점을 동일하게 반환

    sell_index_candidates = predictions[buy_index + 1:]
    if sell_index_candidates.size == 0:
        return buy_index, buy_index

    sell_index = buy_index + np.argmax(sell_index_candidates) + 1

    print(f"매수 신호 인덱스: {buy_index}, 매도 신호 인덱스: {sell_index}")
    return buy_index, sell_index

def main():
    df = fetch_stock_data()
    stock_codes = df['Code'].unique()

    for code in stock_codes:
        stock_data = df[df['Code'] == code]
        current_price = stock_data['Close'].iloc[-1]
        print(f"\n종목 코드: {code}, 현재 가격: {current_price}")

        try:
            x_data, y_data, scaler = prepare_data(stock_data)
        except ValueError as e:
            print(f"종목 코드 {code} 데이터 준비 중 오류: {e}")
            continue

        if len(x_data) < 60:
            print(f"종목 코드 {code} 데이터 부족. 건너뜁니다.")
            continue

        X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
        model = create_and_train_model(X_train, y_train)
        last_60_days = x_data[-1].reshape(1, -1)
        future_predictions = predict_future_prices(model, last_60_days)

        future_prices = scaler.inverse_transform(
            np.hstack((future_predictions.reshape(-1, 1), np.zeros((future_predictions.shape[0], 11))))
        )

        buy_index, sell_index = generate_signals(future_prices.flatten())
        print(f"종목 코드 {code} - 매수 인덱스: {buy_index}, 매도 인덱스: {sell_index}")

if __name__ == '__main__':
    main()
