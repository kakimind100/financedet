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
    df = fdr.DataReader(code, start_date, end_date)
    if df is not None and not df.empty:
        df.reset_index(inplace=True)  # 날짜를 칼럼으로 추가
        df['Code'] = code  # 종목 코드 추가

        # 'Date' 열 뒤로 이동하기 위해 열 순서 재배치
        cols = df.columns.tolist()
        cols.remove('Code')  # 'Code' 열을 리스트에서 제거
        cols.insert(cols.index('Date') + 1, 'Code')  # 'Date' 뒤에 'Code' 추가
        df = df[cols]  # 데이터프레임 열 순서 재배치

        logging.info(f"{code} 데이터 가져오기 완료, 데이터 길이: {len(df)}")

        # 최근 5 거래일 데이터 로그 기록
        logging.info(f"{code} 최근 5 거래일 데이터:\n{df.tail(5)}")

        # NaN 값 확인
        if df.isnull().values.any():
            logging.warning(f"{code} 데이터에 NaN 값이 포함되어 있습니다.")
            logging.debug(f"{code} NaN 값 위치:\n{df[df.isnull().any(axis=1)]}")

        return code, df
    logging.warning(f"{code} 데이터가 비어 있거나 가져오기 실패")
    return code, None

def fetch_stock_data(code, start_date, end_date):
    """주식 데이터를 가져오는 함수."""
    df = fdr.DataReader(code, start_date, end_date)
    if df is not None and not df.empty:
        df.reset_index(inplace=True)  # 날짜를 칼럼으로 추가
        df['Code'] = code  # 종목 코드 추가
        logging.info(f"{code} 데이터 가져오기 완료, 데이터 길이: {len(df)}")

        # 최근 5 거래일 데이터 로그 기록
        logging.info(f"{code} 최근 5 거래일 데이터:\n{df.tail(5)}")

        # NaN 값 확인
        if df.isnull().values.any():
            logging.warning(f"{code} 데이터에 NaN 값이 포함되어 있습니다.")
            logging.debug(f"{code} NaN 값 위치:\n{df[df.isnull().any(axis=1)]}")

        return code, df
    logging.warning(f"{code} 데이터가 비어 있거나 가져오기 실패")
    return code, None

def calculate_technical_indicators(df):
    """기술적 지표를 계산하는 함수."""
    logging.info(f"현재 데이터프레임 열: {df.columns.tolist()}")  # 열 확인 로그 추가

    # 데이터가 충분한지 확인하기 위해 날짜 기준으로 정렬
    df.sort_values(by='Date', inplace=True)

    # 충분한 거래일 데이터 확보
    trading_days = df['Date'].drop_duplicates().count()  # 중복 없는 거래일 수 카운트
    if trading_days < 30:
        logging.warning(f"{df['Code'].iloc[0]}: 데이터가 부족하여 기술적 지표 계산을 건너뜁니다. 필요한 거래일: 30일, 현재 거래일: {trading_days}일")
        return df  # NaN 발생을 방지하기 위해 원본 반환

    # 이동 평균 계산
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA30'] = df['Close'].rolling(window=30).mean()

    # NaN 체크 및 로그 기록
    if df[['MA5', 'MA20', 'MA30']].isnull().values.any():
        nan_dates = df[df[['MA5', 'MA20', 'MA30']].isnull().any(axis=1)]['Date'].tolist()
        logging.warning(f"{df['Code'].iloc[0]}: 이동 평균 계산 중 NaN 값 발생. NaN 발생 날짜: {nan_dates}")
    
    # 가격 변화 및 RSI 계산
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

    # Gain, Loss NaN 체크
    if gain.isnull().values.any() or loss.isnull().values.any():
        nan_dates_gain = df[gain.isnull()]['Date'].tolist()
        nan_dates_loss = df[loss.isnull()]['Date'].tolist()
        logging.warning(f"{df['Code'].iloc[0]}: 기술적 지표 계산 중 Gain 또는 Loss에서 NaN 값 발생. Gain NaN 발생 날짜: {nan_dates_gain}, Loss NaN 발생 날짜: {nan_dates_loss}")

    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # RSI NaN 체크
    if df['RSI'].isnull().values.any():
        nan_dates_rsi = df[df['RSI'].isnull()]['Date'].tolist()
        logging.warning(f"{df['Code'].iloc[0]}: RSI 계산 중 NaN 값 발생. NaN 발생 날짜: {nan_dates_rsi}")

    logging.info(f"{df['Code'].iloc[0]}: RSI 계산 완료.")

    # EMA 및 MACD 계산
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # MACD NaN 체크
    if df[['MACD', 'Signal Line']].isnull().values.any():
        nan_dates_macd = df[df[['MACD', 'Signal Line']].isnull().any(axis=1)]['Date'].tolist()
        logging.warning(f"{df['Code'].iloc[0]}: MACD 계산 중 NaN 값 발생. NaN 발생 날짜: {nan_dates_macd}")

    logging.info(f"{df['Code'].iloc[0]}: MACD 및 Signal Line 계산 완료.")

    # 볼린저 밴드 계산
    df['Upper Band'] = df['MA30'] + (df['Close'].rolling(window=30).std() * 2)
    df['Lower Band'] = df['MA30'] - (df['Close'].rolling(window=30).std() * 2)

    # 볼린저 밴드 NaN 체크
    if df[['Upper Band', 'Lower Band']].isnull().values.any():
        nan_dates_bollinger = df[df[['Upper Band', 'Lower Band']].isnull().any(axis=1)]['Date'].tolist()
        logging.warning(f"{df['Code'].iloc[0]}: 볼린저 밴드 계산 중 NaN 값 발생. NaN 발생 날짜: {nan_dates_bollinger}")

    logging.info(f"{df['Code'].iloc[0]}: 볼린저 밴드 계산 완료.")

    # 가격 변화 계산
    df['Price Change'] = df['Close'].diff()

    # NaN 값 확인
    if df.isnull().values.any():
        nan_dates_final = df[df.isnull().any(axis=1)]['Date'].tolist()
        logging.warning(f"{df['Code'].iloc[0]}: 기술적 지표 계산 후 NaN 값이 포함되어 있습니다. NaN 발생 날짜: {nan_dates_final}")

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

            # 충분한 데이터가 있는 경우 로그 기록
            logging.info(f"종목 코드 {code}의 충분한 데이터 확인: 마지막 5일 데이터\n{df.tail(5)}")

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
    logging.info("주식 분석 스크립트 실행 중...")  # 실행 시작 메시지
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

            # 데이터가 있는 경우 로그 기록
            logging.info(f"종목 코드 {code}의 마지막 데이터: {last_row[['Date', 'Close', 'MA5', 'MA20', 'RSI', 'MACD', 'Upper Band', 'Lower Band']]}")

            # X_new의 데이터 타입 확인 및 변환
            try:
                X_new = np.array(X_new, dtype=float)  # 명시적으로 float로 변환
            except Exception as e:
                logging.error(f"종목 코드 {code}의 입력 데이터 변환 실패: {e}")
                continue

            # NaN 값 체크
            if np.isnan(X_new).any():
                logging.warning(f"종목 코드 {code}의 입력 데이터에 NaN 값이 포함되어 있습니다.")
                continue  # NaN이 포함된 종목은 건너뜁니다.

            prediction = model.predict(X_new)
            if prediction[0] == 1:  # 상승 예상
                potential_stocks.append((code, last_row['Close'], df))

    # 상승 가능성이 있는 종목 정렬 및 상위 20개 선택
    top_stocks = sorted(potential_stocks, key=lambda x: x[1], reverse=True)[:20]

    # 결과 출력 및 최근 5일 치 데이터 로그 기록
    if top_stocks:
        print("내일 29% 이상 상승 가능성이 있는 종목:")
        for code, price, df in top_stocks[:5]:  # 5개 종목만 출력
            recent_data = df.tail(5)  # 최근 5일 치 데이터
            logging.info(f"종목 코드: {code}, 최근 5일 치 데이터:\n{recent_data[['Date', 'Open', 'Close', 'Volume', 'MA5', 'RSI', 'MACD', 'Upper Band', 'Lower Band', 'Price Change']]}")
            print(f"종목 코드: {code}, 현재 가격: {price}")
    else:
        print("29% 이상 상승 가능성이 있는 종목이 없습니다.")

    logging.info("주식 분석 스크립트 실행 완료.")
