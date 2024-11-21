import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta

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
        df.reset_index(inplace=True)  # 인덱스를 리셋하여 Date 컬럼으로 추가
        return df
    return None

def calculate_technical_indicators(df):
    """기술적 지표를 계산하는 함수."""
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Upper Band'] = df['MA20'] + (df['Close'].rolling(window=20).std() * 2)
    df['Lower Band'] = df['MA20'] - (df['Close'].rolling(window=20).std() * 2)
    return df

# 사용 예
end_date = datetime.today()
start_date = end_date - timedelta(days=365)

# 모든 종목 데이터 가져오기
markets = ['KOSPI', 'KOSDAQ']
all_stocks_data = {}
for market in markets:
    codes = fdr.StockListing(market)['Code'].tolist()
    for code in codes:
        data = fetch_stock_data(code, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        if data is not None:
            all_stocks_data[code] = data

# 특정 종목 코드에 대한 최근 5일 치 데이터 로그 기록
specific_code_input = '007810'  # 예시 종목 코드
if specific_code_input in all_stocks_data:
    df_specific = pd.DataFrame(all_stocks_data[specific_code_input])
    df_specific = calculate_technical_indicators(df_specific)

    # 최근 5일 치 데이터 출력 및 로그 기록
    recent_data = df_specific.tail(5)
    logging.info(f"종목 코드 {specific_code_input}의 최근 5일 치 데이터:\n{recent_data[['Date', 'Open', 'Close', 'Volume', 'MA5', 'RSI', 'MACD', 'Upper Band', 'Lower Band']]}")
    print(f"종목 코드 {specific_code_input}의 최근 5일 치 데이터:")
    print(recent_data[['Date', 'Open', 'Close', 'Volume', 'MA5', 'RSI', 'MACD', 'Upper Band', 'Lower Band']])
else:
    logging.warning(f"종목 코드 {specific_code_input}의 데이터가 없습니다.")
    print(f"종목 코드 {specific_code_input}의 데이터가 없습니다.")
