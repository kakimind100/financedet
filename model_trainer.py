import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def fetch_stock_data():
    """주식 데이터를 가져오는 함수 (CSV 파일에서)."""
    file_path = 'data/stock_data_with_indicators.csv'
    df = pd.read_csv(file_path, dtype={'Code': str})  # 'Code' 열을 문자열로 읽어오기
    df['Date'] = pd.to_datetime(df['Date'])
    print("데이터 로드 완료. 열 목록:")
    print(df.columns.tolist())  # 로드된 데이터의 열 목록 출력
    return df

def prepare_data(df):
    """데이터를 준비하는 함수."""
    df = df[['Date', 'Close', 'Change', 'EMA20', 'EMA50', 'RSI', 'MACD', 
              'MACD_Signal', 'Bollinger_High', 'Bollinger_Low', 'Stoch', 
              'Momentum', 'ADX']].dropna().set_index('Date')
    
    df = df.sort_index()
    
    # 종가가 음수인지 확인
    if (df['Close'] < 0).any():
        print("종가에 음수 값이 포함되어 있습니다.")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    x_data, y_data = [], []
    for i in range(60, len(scaled_data) - 26):
        x_data.append(scaled_data[i-60:i])  # 이전 60일 데이터
        y_data.append(scaled_data[i + 25, 1])  # 26일 후의 종가 (Close)

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
    predictions = model.predict(last_60_days.reshape(1, -1))
    print(f"예측된 미래 가격: {predictions}")
    return predictions

def generate_signals(predictions):
    """예측 결과를 기반으로 매수 및 매도 신호를 생성하는 함수."""
    buy_index = np.argmin(predictions)  # 최저점 인덱스
    sell_index = buy_index + np.argmax(predictions[buy_index + 1:]) + 1  # 매수 후 최고점 인덱스

    # 매도 인덱스가 범위를 초과하지 않는지 확인
    if sell_index >= len(predictions):
        sell_index = buy_index  # 매도 인덱스를 매수 인덱스로 설정

    print(f"매수 신호 인덱스: {buy_index}, 매도 신호 인덱스: {sell_index}")
    return buy_index, sell_index

def main():
    # 데이터 로드
    df = fetch_stock_data()

    # 종목별 매수 및 매도 시점 저장
    results = []

    # 종목 리스트
    stock_codes = df['Code'].unique()  # 'Code' 열을 사용하여 종목 코드 가져오기

    for code in stock_codes:
        stock_data = df[df['Code'] == code]  # 각 종목의 데이터만 필터링

        # 현재 가격 가져오기
        current_price = stock_data['Close'].iloc[-1]  # 마지막 날의 종가
        print(f"\n종목 코드: {code}, 현재 가격: {current_price}")

        # 데이터 준비
        try:
            x_data, y_data, scaler = prepare_data(stock_data)
        except KeyError as e:
            print(f"종목 코드 {code}의 데이터 준비 중 오류 발생: {e}")
            continue  # 오류 발생 시 다음 종목으로 넘어감

        # 샘플 수 확인
        print(f"종목 코드 {code}의 샘플 수: x_data={len(x_data)}, y_data={len(y_data)}")

        if len(x_data) < 60:  # 예를 들어, 60일치 데이터가 필요하다면
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
        buy_index, sell_index = generate_signals(future_prices.flatten())

        # 매수 및 매도 신호가 생성되었는지 확인
        if buy_index < len(future_prices[0]) and sell_index < len(future_prices[0]):
            try:
                buy_price = future_prices[0][buy_index]  # 매수 가격
                sell_price = future_prices[0][sell_index]  # 매도 가격
                
                # 매수 가격 확인
                if buy_price < 0:
                    print(f"종목 코드 {code}에서 매수 가격이 음수입니다: {buy_price}.")
                    print(f"매수 인덱스: {buy_index}, 매도 인덱스: {sell_index}")
                    print(f"예측된 가격 배열: {future_prices.flatten()}")
                    continue  # 무시할지 여부 결정
                
                gap = sell_index - buy_index  # 매수와 매도 시점의 격차
                results.append((code, gap, buy_price, sell_price, current_price))
                print(f"종목 코드 {code} - 현재 가격: {current_price}, 매수 가격: {buy_price}, 매도 가격: {sell_price}, 격차: {gap}")
            except IndexError as e:
                print(f"종목 코드 {code}에서 매수 또는 매도 가격 접근 오류: {e}")
        else:
            print(f"종목 코드 {code}에서 매수 또는 매도 신호가 유효하지 않습니다.")

    # 격차가 큰 순서로 정렬
    results.sort(key=lambda x: x[1], reverse=True)

    # 결과 출력
    print("\n매수와 매도 시점의 격차가 큰 종목 순서:")
    if results:
        for code, gap, buy_price, sell_price, current_price in results:
            print(f"종목 코드: {code}, 현재 가격: {current_price}, 격차: {gap}, 매수 가격: {buy_price}, 매도 가격: {sell_price}")
    else:
        print("매수와 매도 시점의 격차가 있는 종목이 없습니다.")

if __name__ == "__main__":
    print("모델 훈련 스크립트 실행 중...")  # 실행 시작 메시지
    main()
    print("모델 훈련 스크립트 실행 완료.")  # 실행 완료 메시지
