import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess

# 로그 및 JSON 파일 디렉토리 설정
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

json_dir = 'json_results'
os.makedirs(json_dir, exist_ok=True)

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

def fetch_stock_listing(market):
    """주식 종목 목록을 가져오는 함수."""
    try:
        logging.debug(f"{market} 종목 목록 가져오는 중...")
        return fdr.StockListing(market)['Code'].tolist()
    except Exception as e:
        logging.error(f"{market} 종목 목록 가져오기 중 오류 발생: {e}")
        return []

def fetch_and_save_stock_data(codes, start_date, end_date):
    """주식 데이터를 JSON 형식으로 가져와 저장하는 함수."""
    all_data = {}

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fdr.DataReader, code, start_date, end_date): code for code in codes}
        for future in as_completed(futures):
            code = futures[future]
            try:
                df = future.result()
                logging.info(f"{code} 데이터 가져오기 성공, 가져온 데이터 길이: {len(df)}")

                if 'Date' not in df.columns:
                    df['Date'] = pd.date_range(end=datetime.today(), periods=len(df), freq='B')
                    logging.info(f"{code} 데이터에 날짜 정보를 추가했습니다.")

                df['Date'] = pd.to_datetime(df['Date'])
                all_data[code] = df.to_dict(orient='records')
            except Exception as e:
                logging.error(f"{code} 처리 중 오류 발생: {e}")

    filename = os.path.join(json_dir, 'stock_data.json')
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, default=str, ensure_ascii=False, indent=4)
    logging.info(f"주식 데이터를 JSON 파일로 저장했습니다: {filename}")

def load_stock_data_from_json():
    """JSON 파일에서 주식 데이터를 로드하는 함수."""
    filename = os.path.join(json_dir, 'stock_data.json')
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_rise_probability(stock_data):
    df = pd.DataFrame(stock_data)

    # 최근 Open, Close, Volume 데이터 추출
    if len(df) < 2:
        return None  # 데이터가 충분하지 않으면 None 반환

    open_price = df['Open'].iloc[-1]
    close_price = df['Close'].iloc[-1]
    volume_current = df['Volume'].iloc[-1]
    volume_previous = df['Volume'].iloc[-2]

    # 가격과 거래량이 유효한지 체크
    if open_price <= 0 or close_price <= 0 or volume_previous <= 0:
        return None  # 유효하지 않은 경우

    # 상승 가능성 계산
    rise_probability = (
        ((close_price - open_price) / open_price) * 100 +
        ((volume_current - volume_previous) / volume_previous) * 100
    )

    # 상승 가능성이 비정상적으로 높은 경우 None 반환
    if rise_probability > 1000:  # 예를 들어 1000% 이상은 비정상적이라고 가정
        return None

    return rise_probability

def search_patterns_and_find_top(stocks_data):
    """각 패턴을 탐지하고, 모든 데이터가 저장된 후 가장 좋은 상태의 종목을 찾는 함수."""
    results = []

    for code, data in stocks_data.items():
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.set_index('Date', inplace=True)
        df['Code'] = code

        if df.index.empty:
            logging.warning(f"종목 코드: {code}에 유효한 날짜 데이터가 없습니다.")
            continue

        # 상승 가능성 계산
        rise_probability = calculate_rise_probability(df)

        # 패턴 결과 저장
        pattern_info = {
            'code': code,
            'rise_probability': rise_probability,
            'data': df.to_dict(orient='records')
        }

        results.append(pattern_info)

    # 상승 가능성이 높은 종목 필터링
    results = [res for res in results if res['rise_probability'] is not None]

    # 상승 가능성 기준으로 정렬하고 상위 20개 선택
    top_20_stocks = sorted(results, key=lambda x: x['rise_probability'], reverse=True)[:20]

    # top_20_stocks를 JSON 파일로 저장
    filename = os.path.join(json_dir, 'top_20_stocks.json')
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(top_20_stocks, f, default=str, ensure_ascii=False, indent=4)

    logging.info(f"상승 가능성이 높은 종목 20개를 JSON 파일로 저장했습니다: {filename}")

    return top_20_stocks

# 메인 실행 블록
if __name__ == "__main__":
    logging.info("주식 분석 스크립트 실행 중...")

    today = datetime.today()
    start_date = today - timedelta(days=365)
    end_date = today.strftime('%Y-%m-%d')
    start_date_str = start_date.strftime('%Y-%m-%d')

    markets = ['KOSPI', 'KOSDAQ']
    all_codes = []

    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_market = {executor.submit(fetch_stock_listing, market): market for market in markets}
        for future in as_completed(future_to_market):
            market = future_to_market[future]
            try:
                codes = future.result()
                all_codes.extend(codes)
                logging.info(f"{market} 종목 목록 가져오기 성공: {len(codes)}개")
            except Exception as e:
                logging.error(f"{market} 종목 목록 가져오기 중 오류 발생: {e}")

    fetch_and_save_stock_data(all_codes, start_date_str, end_date)

    stocks_data = load_stock_data_from_json()

    top_stocks = search_patterns_and_find_top(stocks_data)

    if top_stocks:
        for stock in top_stocks:
            logging.info(f"종목 코드: {stock['code']} - 상승 가능성: {stock['rise_probability']:.2f}%")
    else:
        logging.info("상승 가능성이 높은 종목이 없습니다.")
