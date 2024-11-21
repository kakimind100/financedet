import FinanceDataReader as fdr
import pandas as pd
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
            return df
    except Exception as e:
        logging.error(f"{code} 데이터 가져오기 중 오류 발생: {e}")
    
    return None

def calculate_potential_increase(df):
    """상승 가능성을 계산하는 함수."""
    data_length = len(df)

    if data_length < 2:
        logging.warning("상승 가능성을 계산하기 위해 최소 2개의 데이터가 필요합니다.")
        return None

    # 사용 가능한 데이터 길이에 따라 계산할 개수 설정
    num_to_calculate = min(data_length, 100)

    # 필요한 데이터만 슬라이스
    df = df.tail(num_to_calculate)

    # 이전 거래량과 현재 거래량 계산
    df['Volume Previous'] = df['Volume'].shift(1)

    # 상승 가능성 계산
    df['Potential Increase (%)'] = (
        ((df['Close'] - df['Open']) / df['Open']) * 100 +
        df.apply(lambda x: ((x['Volume'] - x['Volume Previous']) / x['Volume Previous']) * 100 if x['Volume Previous'] > 0 else 0, axis=1)
    )

    # 첫 번째 행은 계산할 수 없으므로 제거
    df = df.dropna(subset=['Potential Increase (%)'])

    return df[['Date', 'Open', 'Close', 'Volume', 'Potential Increase (%)']]

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

# 전체 종목에 대해 상승 가능성 계산
results = {}
for specific_code, specific_data in all_stocks_data.items():
    df_specific = pd.DataFrame(specific_data)

    # 상승 가능성 계산
    result_df = calculate_potential_increase(df_specific)
    
    if result_df is not None and not result_df.empty:
        results[specific_code] = result_df  # 결과를 저장

# 결과 출력 (원하는 형태로 출력 가능)
for code, result in results.items():
    print(f"종목 코드 {code}의 상승 가능성:")
    print(result.head())  # 각 종목의 첫 5개 데이터 출력
