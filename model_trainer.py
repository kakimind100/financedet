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
    print("데이터 로드 완료. 열 목록:")
    print(df.columns.tolist())  # 로드된 데이터의 열 목록 출력
    return df

def prepare_data(df):
    """데이터를 준비하는 함수."""
    # 필요한 열 선택
    df = df[['Date', 'Close', 'Change', 'EMA20', 'EMA50', 'RSI', 'MACD', 
              'MACD_Signal', 'Bollinger_High', 'Bollinger_Low', 'Stoch', 
              'Momentum', 'ADX']].dropna().set_index('Date')
    df = df.sort_index()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    x_data, y_data = [], []
    for i in range(60, len(scaled_data) - 26):
        x_data.append(scaled_data[i-60:i])  # 이전 60일 데이터
        y_data.append(scaled_data[i + 25, 1])  # 26일 후의 종가 (Close)

    x_data, y_data = np.array(x_data), np.array(y_data)
    return x_data, y_data.reshape(-1, 1), scaler

def create_and_train_model(X_train, y_train):
    """모델을 생성하고 훈련하는 함수."""
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100,
                              colsample_bytree=0.3, learning_rate=0.1,
                              max_depth=5, alpha=10, n_jobs=-1)
    model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    return model

def predict_future_prices(model, last_60_days):
    """모델을 사용하여 향후 가격을 예측하는 함수."""
    predictions = model.predict(last_60_days.reshape(1, -1))
    return predictions

def generate_signals(predictions):
    """예측 결과를 기반으로 매수 및 매도 신호를 생성하는 함수."""
    buy_signals = []
    sell_signals = []

    min_price = np.min(predictions)
    max_price = np.max(predictions)

    min_index = np.where(predictions == min_price)[0][0]
    buy_signals.append(min_index)

    for i in range(min_index + 1, len(predictions)):
        if predictions[i] >= max_price:
            sell_signals.append(i)
            break

    return buy_signals, sell_signals

def main():
    # 데이터 로드
    df = fetch_stock_data()

    # 종목별 매수 및 매도 시점 저장
    results = []

    # 종목 리스트
    stock_codes = df['Code'].unique()  # 'Code' 열을 사용하여 종목 코드 가져오기

    for code in stock_codes:
        stock_data = df[df['Code'] == code]  # 각 종목의 데이터만 필터링

        # 데이터 준비
        try:
            x_data, y_data, scaler = prepare_data(stock_data)
        except KeyError as e:
            print(f"종목 코드 {code}의 데이터 준비 중 오류 발생: {e}")
            continue  # 오류 발생 시 다음 종목으로 넘어감

        # 샘플 수 확인
        print(f"종목 코드 {code}의 샘플 수: x_data={len(x_data)}, y_data={len(y_data)}")

        if len(x_data) < 1:  # 데이터가 충분하지 않으면 건너뜀
            print(f"종목 코드 {code}의 데이터가 충분하지 않습니다. 건너뜁니다.")
            continue

        # 훈련 및 테스트 데이터 분할
        if len(x_data) < 2:  # 데이터가 충분하지 않으면 건너뜀
            print(f"종목 코드 {code}의 데이터가 충분하지 않습니다. 샘플 수: {len(x_data)}. 건너뜁니다.")
            continue

        # 훈련 및 테스트 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

        # 모델 훈련
        model = create_and_train_model(X_train, y_train)

        # 가장 최근 60일 데이터를 사용하여 향후 26일 가격 예측
        last_60_days = x_data[-1].reshape(1, -1)
        future_predictions = predict_future_prices(model, last_60_days)

        # 예측 결과를 원래 스케일로 복원
        future_prices = scaler.inverse_transform(np.hstack((future_predictions.reshape(-1, 1),
                                                              np.zeros((future_predictions.shape[0], 11)))))

        # 매수 및 매도 신호 생성
        buy_signals, sell_signals = generate_signals(future_prices.flatten())

        if buy_signals and sell_signals:
            buy_price = future_prices[buy_signals[0]]  # 매수 가격
            sell_price = future_prices[sell_signals[0]]  # 매도 가격
            gap = sell_signals[0] - buy_signals[0]  # 매수와 매도 시점의 격차
            results.append((code, gap, buy_price[0], sell_price[0]))
            print(f"종목 코드 {code} - 매수 가격: {buy_price[0]}, 매도 가격: {sell_price[0]}, 격차: {gap}")

    # 격차가 큰 순서로 정렬
    results.sort(key=lambda x: x[1], reverse=True)

    # 결과 출력
    print("매수와 매도 시점의 격차가 큰 종목 순서:")
    for code, gap, buy_price, sell_price in results:
        print(f"종목 코드: {code}, 격차: {gap}, 매수 가격: {buy_price}, 매도 가격: {sell_price}")

if __name__ == "__main__":
    print("모델 훈련 스크립트 실행 중...")  # 실행 시작 메시지
    main()
    print("모델 훈련 스크립트 실행 완료.")  # 실행 완료 메시지
