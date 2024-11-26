import pandas as pd
import joblib
import logging
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 로그 디렉토리 설정
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# 로깅 설정
logging.basicConfig(
    filename=os.path.join(log_dir, 'stock_data_fetcher.log'),
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 콘솔 로그 출력 설정
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)

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
            'Volume_MA20': 'float'
        }

        df = pd.read_csv(file_path, dtype=dtype)
        logging.info(f"주식 데이터를 '{file_path}'에서 성공적으로 가져왔습니다.")

        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        return df
    except Exception as e:
        logging.error(f"주식 데이터 가져오기 중 오류 발생: {e}")
        return None

def train_model():
    """모델을 훈련시키고 저장하는 함수."""
    df = fetch_stock_data()  # 주식 데이터 가져오기
    if df is None:
        logging.error("데이터프레임이 None입니다. 모델 훈련을 중단합니다.")
        return

    try:
        # 오늘 종가가 29% 이상 상승한 종목 필터링
        df['Today_Rise'] = df['Close'] >= df['Open'] * 1.29
        rising_stocks = df[df['Today_Rise']]

        # 타겟 설정: 다음 날 종가가 오늘 종가의 1.29배 이상일 경우
        rising_stocks['Target'] = np.where(rising_stocks['Close'].shift(-1) >= rising_stocks['Close'] * 1.29, 1, 0)

        # NaN 제거
        rising_stocks.dropna(subset=['Target'], inplace=True)

        # 기술적 지표를 피쳐로 사용
        features = ['MA5', 'MA20', 'RSI', 'MACD', 'Bollinger_High', 'Bollinger_Low', 
                    'Stoch', 'ATR', 'CCI', 'EMA20', 'EMA50', 'Momentum', 
                    'Williams %R', 'ADX', 'Volume_MA20']

        # NaN 제거
        rising_stocks.dropna(subset=features + ['Target'], inplace=True)

        # 훈련 데이터를 위한 리스트
        X = []
        y = []

        # 종목 코드별로 최근 5일 데이터 확인
        for stock_code in rising_stocks['Code'].unique():
            stock_data = rising_stocks[rising_stocks['Code'] == stock_code].tail(5)  # 최근 5일 데이터
            
            if len(stock_data) == 5:  # 최근 5일 데이터가 있는 경우
                # 기술적 지표와 타겟 추가
                X.append(stock_data[features].values.flatten())  # 5일의 피쳐를 1D 배열로 변환
                y.append(stock_data['Target'].values[-1])  # 마지막 날의 타겟 값

        X = np.array(X)
        y = np.array(y)

        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 모델 훈련
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # 모델 저장
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(model, os.path.join(models_dir, 'stock_model.pkl'))
        logging.info("모델 훈련 완료 및 'models/stock_model.pkl'로 저장되었습니다.")

        # 모델 평가
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        logging.info(f"모델 성능 보고서:\n{report}")
        print(report)

        # 혼동 행렬 출력
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)

    except Exception as e:
        logging.error(f"모델 훈련 중 오류 발생: {e}")

def predict_next_day():
    """다음 거래일의 상승 여부를 예측하는 함수."""
    df = fetch_stock_data()  # 주식 데이터 가져오기
    if df is None:
        logging.error("데이터프레임이 None입니다. 예측을 중단합니다.")
        return

    # 훈련된 모델 불러오기
    model = joblib.load(os.path.join('models', 'stock_model.pkl'))

    # 오늘 종가가 29% 이상 상승한 종목 필터링
    today_rise_stocks = df[df['Close'] >= df['Open'] * 1.29]

    # 예측할 데이터 준비 (모든 기술적 지표 포함)
    features = ['MA5', 'MA20', 'RSI', 'MACD', 'Bollinger_High', 'Bollinger_Low', 
                'Stoch', 'ATR', 'CCI', 'EMA20', 'EMA50', 'Momentum', 
                'Williams %R', 'ADX', 'Volume_MA20']
    predictions = []

    # 최근 5거래일 데이터를 사용하여 예측하기
    for stock_code in today_rise_stocks['Code'].unique():
        # 최근 5일간의 데이터를 가져오기
        recent_data = df[df['Code'] == stock_code].tail(5)  # 마지막 5일 데이터 가져오기
        if not recent_data.empty:  # 데이터가 비어있지 않은 경우
            # 최근 5일 데이터를 사용하여 예측
            X_next = recent_data[features].values.flatten().reshape(1, -1)  # 5일 데이터로 2D 배열로 변환
            pred = model.predict(X_next)

            # 예측 결과와 함께 정보를 저장
            predictions.append({
                'Code': stock_code,
                'Prediction': pred[0],
                'MA5': recent_data['MA5'].values[-1],
                'MA20': recent_data['MA20'].values[-1],
                'RSI': recent_data['RSI'].values[-1],
                'MACD': recent_data['MACD'].values[-1],
                'Bollinger_High': recent_data['Bollinger_High'].values[-1],
                'Bollinger_Low': recent_data['Bollinger_Low'].values[-1],
                'Stoch': recent_data['Stoch'].values[-1],
                'ATR': recent_data['ATR'].values[-1],
                'CCI': recent_data['CCI'].values[-1],
                'EMA20': recent_data['EMA20'].values[-1],
                'EMA50': recent_data['EMA50'].values[-1],
                'Momentum': recent_data['Momentum'].values[-1],
                'Williams %R': recent_data['Williams %R'].values[-1],
                'ADX': recent_data['ADX'].values[-1],
                'Volume_MA20': recent_data['Volume_MA20'].values[-1]
            })

    # 예측 결과를 데이터프레임으로 변환
    predictions_df = pd.DataFrame(predictions)

    # 29% 상승할 것으로 예측된 종목 필터링
    top_predictions = predictions_df[predictions_df['Prediction'] == 1]

    # 상위 20개 종목 정렬 (MA5, MACD, RSI, Stoch, Momentum 기준으로 정렬)
    top_predictions = top_predictions.sort_values(
        by=['MA5', 'MACD', 'Momentum', 'RSI', 'Stoch'], 
        ascending=[False, False, False, True, True]  # MA5, MACD, Momentum은 내림차순, RSI와 Stoch은 오름차순으로 정렬
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
              f"Volume_MA20: {row['Volume_MA20']})")

    # 상위 20개 종목의 전체 날짜 데이터 추출
    all_data_with_top_stocks = df[df['Code'].isin(top_predictions['Code'])]

    # 예측 결과를 data/top_20_stocks_all_dates.csv 파일로 저장
    output_file_path = os.path.join('data', 'top_20_stocks_all_dates.csv')
    all_data_with_top_stocks.to_csv(output_file_path, index=False, encoding='utf-8-sig')  # CSV 파일로 저장
    logging.info(f"상위 20개 종목의 전체 데이터가 '{output_file_path}'에 저장되었습니다.")

if __name__ == "__main__":
    logging.info("모델 훈련 스크립트 실행 중...")
    train_model()  # 모델 훈련
    logging.info("모델 훈련 스크립트 실행 완료.")

    logging.info("다음 거래일 예측 스크립트 실행 중...")
    predict_next_day()  # 다음 거래일 예측
    logging.info("다음 거래일 예측 스크립트 실행 완료.")

