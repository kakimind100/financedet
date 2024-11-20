import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# 로그 디렉토리 생성
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# JSON 파일 저장 디렉토리 생성
json_dir = 'json_results'
os.makedirs(json_dir, exist_ok=True)

# 로깅 설정
logging.basicConfig(
    filename=os.path.join(log_dir, 'stock_analysis.log'),
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 콘솔에도 로그 출력
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

def is_cup_with_handle(df):
    """컵과 핸들 패턴을 찾는 함수."""
    if len(df) < 60:
        logging.debug(f"데이터 길이가 60일 미만입니다. 종목 코드: {df['Code'].iloc[0]}")
        return False, None

    cup_bottom = df['Low'].min()
    cup_bottom_index = df['Low'].idxmin()
    cup_bottom_index = df.index.get_loc(cup_bottom_index)

    cup_top = df['Close'][:cup_bottom_index].max()
    handle_start_index = cup_bottom_index + 1
    handle_end_index = handle_start_index + 10

    if handle_end_index <= len(df):
        handle = df.iloc[handle_start_index:handle_end_index]
        handle_top = handle['Close'].max()

        if handle_top < cup_top and cup_bottom < handle_top:
            return True, df.index[-1]
    
    return False, None

def is_golden_cross(df):
    """골든 크로스 패턴을 찾는 함수."""
    if len(df) < 50:
        logging.debug(f"데이터 길이가 50일 미만입니다. 종목 코드: {df['Code'].iloc[0]}")
        return False, None

    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()

    if df['SMA50'].isnull().all() or df['SMA200'].isnull().all():
        logging.warning(f"종목 코드: {df['Code'].iloc[0]}의 이동 평균 데이터가 없습니다.")
        return False, None

    last_sma50 = df['SMA50'].iloc[-1]
    last_sma200 = df['SMA200'].iloc[-1]
    prev_sma50 = df['SMA50'].iloc[-2]
    prev_sma200 = df['SMA200'].iloc[-2]

    if prev_sma50 < prev_sma200 and last_sma50 > last_sma200:
        return True, df.index[-1]

    return False, None

def search_patterns(stocks_data):
    """컵과 핸들 및 골든 크로스 패턴을 찾는 함수."""
    results = []

    for code, data in stocks_data.items():
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.set_index('Date', inplace=True)
        df['Code'] = code

        if df.index.empty:
            logging.warning(f"종목 코드: {code}에 유효한 날짜 데이터가 없습니다.")
            continue

        # 패턴 탐지
        is_cup, cup_date = is_cup_with_handle(df)
        is_golden, cross_date = is_golden_cross(df)

        # 컵과 핸들 패턴이 발견된 경우
        if is_cup:
            # 전체 데이터 저장
            results.append({
                'code': code,
                'pattern': 'Cup with Handle',
                'pattern_date': cup_date.strftime('%Y-%m-%d'),
                'data': df.to_dict(orient='records')  # 전체 데이터 저장
            })

        # 골든 크로스 패턴이 발견된 경우
        if is_golden:
            # 전체 데이터 저장
            results.append({
                'code': code,
                'pattern': 'Golden Cross',
                'pattern_date': cross_date.strftime('%Y-%m-%d'),
                'data': df.to_dict(orient='records')  # 전체 데이터 저장
            })

    return results

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
            except
                logging.error(f"{market} 종목 목록 가져오기 중 오류 발생: {e}")

    fetch_and_save_stock_data(all_codes, start_date_str, end_date)

    stocks_data = load_stock_data_from_json()

    results = search_patterns(stocks_data)

    if results:
        for result in results:
            logging.info(f"종목 코드: {result['code']} - 패턴: {result['pattern']} (완성 날짜: {result['pattern_date']})")
    else:
        logging.info("Cup with Handle 또는 Golden Cross 패턴을 가진 종목이 없습니다.")

