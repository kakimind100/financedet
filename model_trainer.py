import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import yfinance as yf

def prepare_data(df):
    """현재 날짜 기준으로 이전 60일 데이터를 준비"""
    # 날짜 컬럼을 datetime 형식으로 변환 (만약 변환이 안 되어 있으면)
    df['Date'] = pd.to_datetime(df['Date'])

    # 오늘 날짜를 기준으로 60일 전 데이터만 필터링
    today = df['Date'].max()  # 데이터 중 가장 최신 날짜를 가져옴
    start_date = today - pd.Timedelta(days=60)  # 60일 전 날짜

    recent_data = df[df['Date'] > start_date]
    
    # 60일 이상의 데이터가 없으면 훈련할 수 없음
    if len(recent_data) < 60:
        print(f"종목 코드 {df['Code'].iloc[0]}의 데이터가 60일 미만입니다. 훈련을 건너뜁니다.")
        return None, None, None  # 데이터가 부족한 경우 None 반환

    # 필요한 컬럼만 선택
    recent_data = recent_data[['Date', 'Close', 'Change', 'EMA20', 'EMA50', 'RSI', 'MACD', 
                               'MACD_Signal', 'Bollinger_High', 'Bollinger_Low', 'Stoch', 
                               'Momentum', 'ADX']].dropna()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(recent_data.set_index('Date'))

    # x_data와 y_data 준비 (60일 데이터로 다음 26일 후 가격 예측)
    x_data, y_data = [], []
    for i in range(60, len(scaled_data) - 26):  # 60일 이상 데이터를 가진 샘플만
        x_data.append(scaled_data[i-60:i])  # 이전 60일 데이터
        y_data.append(scaled_data[i + 25, 0])  # 26일 후의 종가 (Close)

    x_data, y_data = np.array(x_data), np.array(y_data.reshape(-1, 1))
    print(f"준비된 데이터 샘플 수: x_data={len(x_data)}, y_data={len(y_data)}")
    
    return x_data, y_data, scaler

def train_model(x_data, y_data):
    """훈련 모델 (Random Forest)"""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x_data.reshape(x_data.shape[0], -1), y_data)
    return model

def predict(model, x_data, scaler):
    """미래 가격 예측"""
    prediction = model.predict(x_data.reshape(x_data.shape[0], -1))
    prediction = scaler.inverse_transform(np.concatenate((prediction.reshape(-1, 1), np.zeros((len(prediction), 11))), axis=1))[:,0]
    return prediction

def generate_signals(predictions):
    """매수/매도 신호 생성"""
    buy_index = np.argmin(predictions)  # 예측된 가격 중 최저점
    sell_index = buy_index + np.argmax(predictions[buy_index + 1:]) + 1  # 매수 후 최고점 인덱스
    return buy_index, sell_index

def main():
    stock_codes = ['000020', '003380', '003480']  # 테스트 종목 코드들 (여기서 원하는 종목을 추가)
    
    for code in stock_codes:
        print(f"종목 코드: {code}")

        # yfinance에서 데이터 다운로드 (최근 6개월, 60일 이상 데이터 확보)
        df = yf.download(code + '.KS', period='6mo', interval='1d')
        df['Code'] = code  # 종목 코드 추가

        # 데이터 준비
        x_data, y_data, scaler = prepare_data(df)
        if x_data is None or y_data is None:
            continue  # 데이터가 부족한 경우 건너뛰기

        print(f"종목 코드 {code}의 데이터 준비 완료")

        # 모델 훈련
        model = train_model(x_data, y_data)
        print(f"모델 훈련 완료")

        # 예측
        future_predictions = predict(model, x_data[-1:], scaler)
        print(f"예측된 미래 가격: {future_predictions[-1]}")

        # 매수/매도 신호 생성
        buy_index, sell_index = generate_signals(future_predictions)
        print(f"매수 신호 인덱스: {buy_index}, 매도 신호 인덱스: {sell_index}")

        # 종목 코드와 예측된 가격 배열 출력
        print(f"종목 코드 {code}의 예측된 가격 배열: {future_predictions}")

if __name__ == "__main__":
    main()
