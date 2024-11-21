import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    try:
        df = fdr.DataReader(code, start_date, end_date)

        if df is not None and not df.empty:
            logging.info(f"{code} 데이터 가져오기 성공, 데이터 길이: {len(df)}")
            df.reset_index(inplace=True)  # 인덱스를 리셋하여 Date 컬럼으로 추가
            return df
    except Exception as e:
        logging.error(f"{code} 데이터 가져오기 중 오류 발생: {e}")
    
    return None

def calculate_technical_indicators(df):
    """기술적 지표를 계산하는 함수."""
    # 이동 평균 (MA)
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()

    # 상대 강도 지수 (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 볼린저 밴드
    df['Upper Band'] = df['MA20'] + (df['Close'].rolling(window=20).std() * 2)
    df['Lower Band'] = df['MA20'] - (df['Close'].rolling(window=20).std() * 2)

    return df

def find_stocks_with_potential_increase(df, threshold=2.0):
    """상승 가능성이 있는 종목을 찾는 함수."""
    # 상승 가능성 계산
    df['Potential Increase (%)'] = (
        ((df['Close'] - df['Open']) / df['Open']) * 100 +
        df['Volume'].pct_change() * 100  # 거래량 변화 비율
    )
    
    # 기준 이상인 종목 필터링
    potential_stocks = df[df['Potential Increase (%)'] > threshold]
    
    # 추가 조건: RSI, MACD 신호 등
    potential_stocks = potential_stocks[
        (potential_stocks['RSI'] < 30) |  # 과매도 상태
        (potential_stocks['MACD'] > potential_stocks['Signal Line'])  # MACD가 신호선 위
    ]

    return potential_stocks

def fetch_all_stocks_data(start_date, end_date):
    """모든 주식 데이터를 가져오는 함수."""
    markets = ['KOSPI', 'KOSDAQ']
    all_codes = []

    for market in markets:
        codes = fdr.StockListing(market)['Code'].tolist()
        all_codes.extend(codes)

    all_data = {}

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_stock_data, code, start_date, end_date): code for code in set(all_codes)}
        for future in as_completed(futures):
            code = futures[future]
            data = future.result()
            if data is not None:
                all_data[code] = data.to_dict(orient='records')

    return all_data

# 사용 예
end_date = datetime.today()
start_date = end_date - timedelta(days=365)

logging.info("주식 분석 스크립트 실행 중...")
all_stocks_data = fetch_all_stocks_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

# 내일 상승할 가능성이 있는 종목 찾기
potential_increase_stocks = {}
for specific_code, specific_data in all_stocks_data.items():
    df_specific = pd.DataFrame(specific_data)

    # 기술적 지표 계산
    df_specific = calculate_technical_indicators(df_specific)

    # 오늘까지의 데이터로 상승 가능성 있는 종목 찾기
    potential_stocks = find_stocks_with_potential_increase(df_specific, threshold=2.0)
    
    if not potential_stocks.empty:
        potential_increase_stocks[specific_code] = potential_stocks

# 결과 출력
print("내일 상승할 가능성이 있는 종목:")
for code, result in potential_increase_stocks.items():
    print(f"종목 코드 {code}:")
    print(result[['Date', 'Open', 'Close', 'Volume', 'Potential Increase (%)', 'RSI', 'MACD', 'Upper Band', 'Lower Band']].tail())  # 마지막 몇 개 데이터 출력

# 특정 종목 코드에 대한 최근 5일 치 데이터 로그 기록
specific_code_input = input("특정 종목 코드를 입력하세요 (예: '065350'): ")
if specific_code_input in all_stocks_data:
    specific_data = all_stocks_data[specific_code_input]
    df_specific = pd.DataFrame(specific_data)

    # 최근 5일 치 데이터 출력 및 로그 기록
    recent_data = df_specific.tail(5)
    logging.info(f"종목 코드 {specific_code_input}의 최근 5일 치 데이터:\n{recent_data[['Date', 'Open', 'Close', 'Volume']]}")
    print(f"종목 코드 {specific_code_input}의 최근 5일 치 데이터:")
    print(recent_data[['Date', 'Open', 'Close', 'Volume']])
else:
    print(f"종목 코드 {specific_code_input}의 데이터가 없습니다.")
