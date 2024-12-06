import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

def fetch_stock_data():
    """주식 데이터를 가져오는 함수 (CSV 파일에서)."""
    file_path = 'data/stock_data_with_indicators.csv'
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def prepare_data(df):
    """데이터를 준비하는 함수."""
    df = df[['Date', 'Close']].set_index('Date')
    df = df.sort_index()

    # Min-Max 스케일링
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    # 데이터셋 생성
    x_data, y_data = [], []
    for i in range(60, len(scaled_data)):
        x_data.append(scaled_data[i-60:i, 0])
        y_data.append(scaled_data[i, 0])

    x_data, y_data = np.array(x_data), np.array(y_data)
    return x_data, y_data, scaler

def main():
    # 데이터 로드
    df = fetch_stock_data()

    # 데이터 준비
    x_data, y_data, scaler = prepare_data(df)

    # 훈련 및 테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    # XGBoost 모델 생성
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)

    # 모델 훈련
    model.fit(X_train, y_train)

    # 예측
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))

    # 성능 평가
    mse = mean_squared_error(scaler.inverse_transform(y_test.reshape(-1, 1)), predictions)
    print(f'Mean Squared Error: {mse}')

if __name__ == "__main__":
    main()
