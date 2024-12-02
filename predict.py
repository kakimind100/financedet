import pandas as pd
import joblib
import logging
import os

# 로그 디렉토리 설정
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# 로깅 설정
logging.basicConfig(
    filename=os.path.join(log_dir, 'stock_predictions.log'),
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def fetch_stock_data():
    """주식 데이터를 가져오는 함수 (CSV 파일에서)."""
    logging.debug("주식 데이터를 가져오는 중...")
    try:
        file_path = os.path.join('data', 'stock_data_with_indicators.csv')
        
        dtype = {
            'Code': 'object',
            'Date': 'str',
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
            'OBV': 'float'
        }

        df = pd.read_csv(file_path, dtype=dtype)
        logging.info(f"주식 데이터를 '{file_path}'에서 성공적으로 가져왔습니다.")
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        return df
    except Exception as e:
        logging.error(f"주식 데이터 가져오기 중 오류 발생: {e}")
        return None

def predict_next_day(model):
    """다음 거래일의 상승 여부를 예측하는 함수."""
    df = fetch_stock_data()  # 주식 데이터 가져오기
    if df is None:
        logging.error("데이터프레임이 None입니다. 예측을 중단합니다.")
        return

    # 오늘 종가가 29% 이상 상승한 종목 필터링
    today_rise_stocks = df[df['Close'] >= df['Open'] * 1.29]

    # 예측할 데이터 준비 (모든 기술적 지표 포함)
    features = ['MA5', 'MA20', 'RSI', 'MACD', 'Bollinger_High', 'Bollinger_Low', 
                'Stoch', 'ATR', 'CCI', 'EMA20', 'EMA50', 'Momentum', 
                'Williams %R', 'ADX', 'Volume_MA20', 'ROC', 'CMF', 'OBV']
    predictions = []

    # 최근 10거래일 데이터를 사용하여 예측하기
    for stock_code in today_rise_stocks['Code'].unique():
        recent_data = df[df['Code'] == stock_code].tail(10)  # 마지막 10일 데이터 가져오기
        if not recent_data.empty and len(recent_data) == 10:  # 데이터가 비어있지 않고 10일인 경우
            # 최근 10일 데이터를 사용하여 예측
            X_next = recent_data[features].values[-1].reshape(1, -1)  # 마지막 날의 피처로 2D 배열로 변환
            logging.debug(f"예측할 데이터 X_next: {X_next}")

            # 예측
            pred = model.predict(X_next)

            # 예측 결과와 함께 정보를 저장
            predictions.append({
                'Code': stock_code,
                'Prediction': pred[0],
                **recent_data[features].iloc[-1].to_dict()  # 마지막 날의 피처 값 추가
            })

    # 예측 결과를 데이터프레임으로 변환
    predictions_df = pd.DataFrame(predictions)

    # 29% 상승할 것으로 예측된 종목 필터링
    top_predictions = predictions_df[predictions_df['Prediction'] == 1]

    # 상위 20개 종목 정렬 (기본 피처와 추가 피처 기준으로 정렬)
    top_predictions = top_predictions.sort_values(
        by=['RSI', 'MACD', 'Volume', 'Momentum', 'MA5', 'MA20', 'ATR', 'OBV', 'Stoch', 'CCI', 'ADX'], 
        ascending=[True, False, True, False, False, False, False, False, True, True, True]
    ).head(20)

    # 예측 결과 출력
    print("다음 거래일에 29% 상승할 것으로 예측되는 상위 20개 종목:")
    for index, row in top_predictions.iterrows():
        print(f"{row['Code']} (MA5: {row['MA5']}, MA20: {row['MA20']}, RSI: {row['RSI']}, "
              f"MACD: {row['MACD']}, Bollinger_High: {row['Bollinger_High']}, "
              f"Bollinger_Low: {row['Bollinger_Low']}, Stoch: {row['Stoch']}, "
              f"ATR: {row['ATR']}, CCI: {row['CCI']}, EMA20: {row['EMA20']}, "
              f"EMA50: {row['EMA50']}, Momentum: {row['Momentum']}, "
              f"Williams %R: {row['Williams %R']}, ADX: {row['ADX']}, "
              f"Volume_MA20: {row['Volume_MA20']}, ROC: {row['ROC']}, "
              f"CMF: {row['CMF']}, OBV: {row['OBV']})")

    # 상위 20개 종목의 전체 날짜 데이터 추출
    all_data_with_top_stocks = df[df['Code'].isin(top_predictions['Code'])]

    # 예측 결과를 data/top_20_stocks_all_dates.csv 파일로 저장
    output_file_path = os.path.join('data', 'top_20_stocks_all_dates.csv')
    all_data_with_top_stocks.to_csv(output_file_path, index=False, encoding='utf-8-sig')  # CSV 파일로 저장
    logging.info(f"상위 20개 종목의 전체 데이터가 '{output_file_path}'에 저장되었습니다.")

if __name__ == "__main__":
    logging.info("모델 예측 스크립트 실행 중...")
    model = joblib.load('best_model.pkl')  # 저장된 모델 로드
    predict_next_day(model)  # 다음 거래일 예측
    logging.info("모델 예측 스크립트 실행 완료.")

