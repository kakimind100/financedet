import pandas as pd
import logging
from xgboost import XGBRegressor
import numpy as np

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# 주식 데이터 로드 함수
def fetch_stock_data():
    """주식 데이터를 가져오는 함수 (CSV 파일에서)."""
    try:
        file_path = 'data/stock_data_with_indicators.csv'
        df = pd.read_csv(file_path, dtype={'Code': str})
        df['Date'] = pd.to_datetime(df['Date'])
        logging.info("데이터 로드 완료. 열 목록: %s", df.columns.tolist())
        return df
    except FileNotFoundError:
        logging.error("CSV 파일을 찾을 수 없습니다. 경로: %s", file_path)
        return None
    except Exception as e:
        logging.error("데이터 로드 중 오류 발생: %s", e)
        return None

# 데이터 전처리 함수
def preprocess_data(df, code):
    """지정된 종목 코드에 대한 데이터를 전처리하는 함수."""
    stock_data = df[df['Code'] == code]  # 종목 코드에 맞는 데이터 필터링
    stock_data = stock_data.sort_values('Date')  # 날짜순 정렬

    # 특징과 레이블 생성
    x_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20',   
                          'MACD', 'Bollinger_High', 'Bollinger_Low',   
                          'Stoch', 'RSI']]  
    y_data = stock_data['Close'].shift(-1)  # 다음 날 종가를 레이블로 사용
    x_data = x_data[:-1]  # 레이블 이동으로 인해 마지막 행을 제거
    y_data = y_data.dropna()  # NaN 값 제거

    return x_data.values, y_data.values

# 모델 훈련 함수
def train_model(x_data, y_data):
    """XGBoost 모델을 훈련시키는 함수."""
    model = XGBRegressor()
    model.fit(x_data, y_data)
    logging.info("모델 훈련 완료.")
    return model

# 미래 가격 예측 함수
def predict_future_prices(model, last_60_days):
    """모델을 사용하여 향후 26일 가격을 예측하는 함수."""
    predictions = model.predict(last_60_days.reshape(1, -1))
    logging.info(f"예측된 미래 가격: {predictions}")
    return predictions

# 신호 생성 함수
def generate_signals(predictions):
    """예측 결과를 기반으로 매수 및 매도 신호를 생성하는 함수."""
    predictions = predictions.flatten()  # 1차원 배열로 변환
    buy_index = np.argmin(predictions)  # 최저점 인덱스
    sell_index = buy_index + np.argmax(predictions[buy_index + 1:]) + 1  # 매수 후 최고점 인덱스

    # 매도 인덱스가 범위를 초과하지 않는지 확인
    if sell_index >= len(predictions):
        sell_index = buy_index  # 매도 인덱스를 매수 인덱스로 설정

    logging.info(f"매수 신호 인덱스: {buy_index}, 매도 신호 인덱스: {sell_index}")
    return buy_index, sell_index

# 메인 함수
def main():
    # 데이터 로드
    df = fetch_stock_data()
    if df is None:
        return  # 데이터 로드 실패 시 종료

    # 전체 종목 코드에 대해 반복
    unique_codes = df['Code'].unique()  # 전체 종목 코드 추출
    logging.info(f"전체 종목 코드 목록: {unique_codes}")

    for code in unique_codes:
        logging.info(f"종목 코드 {code} 처리 시작")

        # 오늘자 종가 가져오기
        today_data = df[(df['Code'] == code) & (df['Date'] == pd.to_datetime('today'))]
        if today_data.empty:
            logging.error(f"{code} - 오늘자 종가를 찾을 수 없습니다.")
            continue
        
        current_price = today_data['Close'].values[0]  # 오늘자 종가
        logging.info(f"종목 코드 {code}의 오늘자 종가는 {current_price}입니다.")

        # 데이터 전처리
        x_data, y_data = preprocess_data(df, code)
        logging.info(f"준비된 데이터 샘플 수: x_data={len(x_data)}, y_data={len(y_data)}")

        # 모델 훈련
        model = train_model(x_data, y_data)

        # 최근 60일 데이터 가져오기
        last_60_days = x_data[-1]  # 마지막 샘플

        # 미래 가격 예측
        future_predictions = predict_future_prices(model, last_60_days)

        # 매수 및 매도 신호 생성
        buy_index, sell_index = generate_signals(future_predictions)

        # 신호 유효성 확인 및 출력
        if buy_index < len(future_predictions[0]) and sell_index < len(future_predictions[0]):
            buy_price = future_predictions[0][buy_index]  # 매수 가격
            sell_price = future_predictions[0][sell_index]  # 매도 가격

            # 매수 가격 유효성 체크
            if buy_price < 0:
                logging.warning(f"종목 코드 {code}에서 매수 가격이 음수입니다: {buy_price}.")
                continue  # 종료

            gap = sell_index - buy_index  # 매수와 매도 시점의 격차
            logging.info(f"종목 코드 {code} - 현재 가격: {current_price}, 매수 가격: {buy_price}, 매도 가격: {sell_price}, 격차: {gap}")
        else:
            logging.warning(f"종목 코드 {code}에서 매수 또는 매도 신호가 유효하지 않습니다.")

# 실행
if __name__ == "__main__":
    main()
