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
    level=logging.DEBUG,
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

        # 데이터가 정상적으로 가져와졌는지 확인
        if df is not None and not df.empty:
            logging.info(f"{code} 데이터 가져오기 성공, 데이터 길이: {len(df)}")

            # 날짜 인덱스 확인
            if df.index.empty:
                logging.warning(f"{code}에 유효한 날짜 데이터가 없습니다.")
            else:
                logging.info(f"{code}의 날짜 인덱스: {df.index}")

            # 데이터 내용 출력
            print(f"종목 코드: {code}")
            print("데이터 열:", df.columns.tolist())  # 데이터 열 확인
            print(df.head())  # 데이터의 첫 5개 행 출력

            return df
    except Exception as e:
        logging.error(f"{code} 데이터 가져오기 중 오류 발생: {e}")
    
    return None

def fetch_all_stocks_data(start_date, end_date):
    """모든 주식 데이터를 가져오는 함수."""
    markets = ['KOSPI', 'KOSDAQ']
    all_codes = []

    # 각 시장에서 종목 코드 가져오기
    for market in markets:
        codes = fdr.StockListing(market)['Code'].tolist()
        all_codes.extend(codes)

    all_data = {}

    # 멀티스레딩을 사용하여 데이터 가져오기
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_stock_data, code, start_date, end_date): code for code in set(all_codes)}  # 중복된 종목 코드 제거
        for future in as_completed(futures):
            code = futures[future]
            data = future.result()
            if data is not None:
                all_data[code] = data.to_dict(orient='records')

    return all_data

# 사용 예
end_date = datetime.today()  # 현재 날짜
start_date = end_date - timedelta(days=365)  # 1년 전 날짜

# 실행 시작 메시지
logging.info("주식 분석 스크립트 실행 중...")
all_stocks_data = fetch_all_stocks_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

# 특정 종목 데이터 확인 (예: '227610')
specific_code = '227610'
if specific_code in all_stocks_data:
    specific_data = all_stocks_data[specific_code]
    df_specific = pd.DataFrame(specific_data)
    print(f"종목 코드 {specific_code}의 데이터:")
    print(df_specific.head())  # 첫 5개 데이터 출력
else:
    print(f"종목 코드 {specific_code}의 데이터가 없습니다.")
