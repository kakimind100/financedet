import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def fetch_stock_data():
    """주식 데이터를 가져오는 함수 (CSV 파일에서)."""
    file_path = 'data/stock_data_with_indicators.csv'
    df = pd.read_csv(file_path, dtype={'Code': str})  # 'Code' 열을 문자열로 읽어오기
    df['Date'] = pd.to_datetime(df['Date'])
    print("데이터 로드 완료. 열 목록:")
    print(df.columns.tolist())  # 로드된 데이터의 열 목록 출력
    print(df.head())  # 데이터 샘플 확인
    return df


def prepare_data(df):
    """데이터를 준비하는 함수."""
    df = df[['Date', 'Close', 'Change', 'EMA20', 'EMA50', 'RSI', 'MACD',
             'MACD_Signal', 'Bollinger_High', 'Bollinger_Low', 'Stoch',
             'Momentum', 'ADX']].dropna().set_index('Date')
    
    df = df.sort_index()
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    x_data, y_data = [], []
    for i in range(60, len(scaled_data) - 26):
        x_data.append(scaled_data[i - 60:i])  # 이전 60일 데이터
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
    """모델을 사용하여 향후 가격을 예측하는 함수."""
    try:
        predictions = model.predict(last_60_days.reshape(1, -1))
        if predictions[0] < 0:
            print("예측된 가격이 음수입니다. 확인 필요.")
            return None
        print(f"예측된 미래 가격: {predictions}")
        return predictions
    except Exception as e:
        print(f"예측 중 오류 발생: {e}")
        return None


def generate_signals(predictions):
    """예측 결과를 기반으로 매수 및 매도 신호를 생성하는 함수."""
    try:
        buy_index = np.argmin(predictions)  # 최저점 인덱스
        sell_index = buy_index + np.argmax(predictions[buy_index + 1:]) + 1
        return buy_index, sell_index
    except Exception as e:
        print(f"신호 생성 중 오류 발생: {e}")
        return None, None


def plot_stock_data(stock_data, code):
    """특정 종목의 데이터를 시각화하는 함수."""
    try:
        plt.figure(figsize=(14, 12))
        # 종가 및 이동 평균 차트
        plt.subplot(4, 2, 1)
        plt.plot(stock_data['Close'], label='Close Price', color='blue')
        plt.plot(stock_data['EMA20'], label='EMA20', color='orange')
        plt.plot(stock_data['EMA50'], label='EMA50', color='green')
        plt.title(f'{code} - Close Price and Moving Averages')
        plt.legend()

        # RSI 차트
        plt.subplot(4, 2, 2)
        plt.plot(stock_data['RSI'], label='RSI', color='purple')
        plt.axhline(70, linestyle='--', color='red', label='Overbought (70)')
        plt.axhline(30, linestyle='--', color='green', label='Oversold (30)')
        plt.title(f'{code} - Relative Strength Index (RSI)')
        plt.legend()

        # MACD 차트
        plt.subplot(4, 2, 3)
        plt.plot(stock_data['MACD'], label='MACD', color='orange')
        plt.plot(stock_data['MACD_Signal'], label='MACD Signal', color='blue')
        plt.title(f'{code} - MACD')
        plt.legend()

        # 기타 지표 생략 가능...
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"시각화 중 오류 발생: {e}")


def main():
    # 데이터 로드
    df = fetch_stock_data()

    # 종목별 매수 및 매도 시점 저장
    results = []

    stock_codes = df['Code'].unique()

    for code in stock_codes:
        stock_data = df[df['Code'] == code]
        print(f"\n종목 코드: {code}")

        # 데이터 준비
        try:
            x_data, y_data, scaler = prepare_data(stock_data)
        except Exception as e:
            print(f"데이터 준비 중 오류: {e}")
            continue

        if len(x_data) < 60:
            print(f"데이터 부족으로 {code} 건너뜀.")
            continue

        X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
        model = create_and_train_model(X_train, y_train)

        last_60_days = x_data[-1]
        predictions = predict_future_prices(model, last_60_days)

        if predictions is None:
            continue

        future_prices = scaler.inverse_transform(np.hstack((predictions.reshape(-1, 1),
                                                              np.zeros((predictions.shape[0], 11)))))

        buy_index, sell_index = generate_signals(future_prices.flatten())
        if buy_index is not None and sell_index is not None:
            results.append((code, buy_index, sell_index))
            plot_stock_data(stock_data, code)

    print("분석 완료.")

if __name__ == "__main__":
    main()
