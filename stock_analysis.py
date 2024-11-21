import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from tqdm import tqdm  # 진행 상황 표시를 위한 라이브러리

# 로그 및 JSON 파일 디렉토리 설정
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# 로깅 설정
logging.basicConfig(
    filename=os.path.join(log_dir, 'stock_analysis.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 콘솔 로그 출력 설정
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)

def fetch_stock_data(code, start_date, end_date):
    """주식 데이터를 가져오는 함수."""
    logging.info(f"{code} 데이터 가져오는 중...")
    df = fdr.DataReader(code, start_date, end_date)
    if df is not None and not df.empty:
        df.reset_index(inplace=True)  # 날짜를 칼럼으로 추가
        logging.info(f"{code} 데이터 가져오기 완료, 데이터 길이: {len(df)}")

        # NaN 값 확인
        if df.isnull().values.any():
            logging.warning(f"{code} 데이터에 NaN 값이 포함되어 있습니다.")
            logging.debug(f"{code} NaN 값 위치:\n{df[df.isnull().any(axis=1)]}")

        return code, df
    logging.warning(f"{code} 데이터가 비어 있거나 가져오기 실패")
    return code, None

def calculate_technical_indicators(df):
    """기술적 지표를 계산하는 함수."""
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    # NaN 발생 원인 로그
    if gain.isnull().values.any() or loss.isnull().values.any():
        logging.warning("기술적 지표 계산 중 NaN 값 발생")
        logging.debug(f"Gain NaN 위치:\n{gain[gain.isnull()]}")
        logging.debug(f"Loss NaN 위치:\n{loss[loss.isnull()]}")

    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Upper Band'] = df['MA20'] + (df['Close'].rolling(window=20).std() * 2)
    df['Lower Band'] = df['MA20'] - (df['Close'].rolling(window=20).std() * 2)
    df['Price Change'] = df['Close'].diff()

    # NaN 값 확인
    if df.isnull().values.any():
        logging.warning("기술적 지표 계산 후 NaN 값이 포함되어 있습니다.")
        logging.debug(f"NaN 값 위치:\n{df[df.isnull().any(axis=1)]}")

    return df

def preprocess_data(all_stocks_data):
    """데이터를 전처리하고 피처와 레이블을 준비하는 함수."""
    all_features = []
    all_targets = []

    for code, df in all_stocks_data.items():
        df = calculate_technical_indicators(df)  # NaN 값을 제거하지 않음
        if len(df) > 20:  # 충분한 데이터가 있는 경우
            df = df.copy()  # 복사본 생성
            df['Target'] = np.where(df['Price Change'] > 0, 1, 0)  # 종가 상승 여부
            features = ['MA5', 'MA20', 'RSI', 'MACD', 'Upper Band', 'Lower Band']
            X = df[features]
            y = df['Target']

            # NaN 값 확인
            if X.isnull().values.any() or y.isnull().values.any():
                logging.warning(f"{code}의 입력 피처 또는 타겟에 NaN 값이 포함되어 있습니다.")
                logging.debug(f"{code}의 X NaN 위치:\n{X[X.isnull().any(axis=1)]}")
                logging.debug(f"{code}의 y NaN 위치:\n{y[y.isnull()]}")

            all_features.append(X)
            all_targets.append(y)

    # 모든 종목 데이터를 하나로 합치기
    X_all = pd.concat(all_features)
    y_all = pd.concat(all_targets)

    return X_all, y_all

def train_model(X, y):
    """모델을 훈련시키고 저장하는 함수."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # 진행상황을 로그로 남기기 위해 tqdm 사용
    logging.info("모델 훈련 시작...")
    for _ in tqdm(range(1), desc="모델 훈련 진행 중", unit="iteration"):
        model.fit(X, y)
    
    joblib.dump(model, 'stock_model.pkl')  # 모델 저장
    logging.info("모델 훈련 완료 및 저장됨.")
    
    return model

def evaluate_model(model, X_test, y_test):
    """모델 성능을 평가하는 함수."""
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    
    # 성능 보고서 로그에 기록
    logging.info(f"모델 성능 보고서:\n{report}")
    print(report)

def main():
    # 사용 예
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)

    # 모든 종목 데이터 가져오기
    markets = ['KOSPI', 'KOSDAQ']
    all_stocks_data = {}

    logging.info("주식 데이터 가져오는 중...")
    with ThreadPoolExecutor(max_workers=20) as executor:  # 멀티스레딩: 최대 20개
        futures = {}
        for market in markets:
            codes = fdr.StockListing(market)['Code'].tolist()
            for code in codes:
                futures[executor.submit(fetch_stock_data, code, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))] = code

        for future in as_completed(futures):
            code, data = future.result()
            if data is not None:
                all_stocks_data[code] = data
                logging.info(f"{code} 데이터가 성공적으로 저장되었습니다.")
            else:
                logging.warning(f"{code} 데이터가 실패했습니다.")

    # 데이터 전처리
    X_all, y_all = preprocess_data(all_stocks_data)

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

    # 모델 훈련 및 저장
    model = train_model(X_train, y_train)

    # 모델 평가
    evaluate_model(model, X_test, y_test)

    # 상승 가능성이 있는 종목 찾기
    potential_stocks = []

    for code, df in all_stocks_data.items():
        df = calculate_technical_indicators(df)
        if len(df) > 20:  # 충분한 데이터가 있는 경우
            last_row = df.iloc[-1]
            features = ['MA5', 'MA20', 'RSI', 'MACD', 'Upper Band', 'Lower Band']
            X_new = last_row[features].values.reshape(1, -1)  # 2D 배열로 변환

            # NaN 값 체크
            if np.isnan(X_new).any():
                logging.warning(f"종목 코드 {code}의 입력 데이터에 NaN 값이 포함되어 있습니다.")
                continue  # NaN이 포함된 종목은 건너뜁니다.

            prediction = model.predict(X_new)

            # 내일 가격 예측 (가정: 오늘 종가에 상승률 적용)
            predicted_price = last_row['Close'] * 1.29  # 29% 상승 예측
            if prediction[0] == 1 and last_row['Close'] < predicted_price:  # 상승 예상 및 29% 이상 상승
                potential_stocks.append((code, last_row['Close'], df))

    # 상승 가능성이 있는 종목 정렬 및 상위 20개 선택
    top_stocks = sorted(potential_stocks, key=lambda x: x[1], reverse=True)[:20]

    # 결과 출력 및 최근 5일 치 데이터 로그 기록
    if top_stocks:
        print("내일 29% 이상 상승 가능성이 있는 종목:")
        for code, price, df in top_stocks:
            recent_data = df.tail(5)  # 최근 5일 치 데이터
            logging.info(f"종목 코드: {code}, 최근 5일 치 데이터:\n{recent_data[['Date', 'Open', 'Close', 'Volume', 'MA5', 'RSI', 'MACD', 'Upper Band', 'Lower Band', 'Price Change']]}")
            print(f"종목 코드: {code}, 현재 가격: {price}")
    else:
        print("29% 이상 상승 가능성이 있는 종목이 없습니다.")

    logging.info("주식 분석 스크립트 실행 완료.")

if __name__ == "__main__":
    main()
