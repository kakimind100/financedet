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

    if df.index.empty:
        logging.warning(f"종목 코드: {df['Code'].iloc[0]}에 날짜 데이터가 없습니다.")
        return False, None

    logging.debug(f"종목 코드: {df['Code'].iloc[0]}의 날짜 데이터: {df.index.tolist()}")

    cup_bottom = df['Low'].min()
    cup_bottom_index = df['Low'].idxmin()

    cup_bottom_index = df.index.get_loc(cup_bottom_index)

    if cup_bottom_index < 0 or cup_bottom_index >= len(df):
        logging.warning(f"컵 바닥 인덱스가 유효하지 않습니다. 종목 코드: {df['Code'].iloc[0]}, cup_bottom_index: {cup_bottom_index}")
        return False, None

    cup_top = df['Close'][:cup_bottom_index].max()

    if pd.isna(cup_top):
        logging.warning(f"컵 상단 값이 유효하지 않습니다. 종목 코드: {df['Code'].iloc[0]}, cup_bottom_index: {cup_bottom_index}")
        return False, None

    handle_start_index = cup_bottom_index + 1
    handle_end_index = handle_start_index + 10

    if handle_end_index <= len(df):
        handle = df.iloc[handle_start_index:handle_end_index]
        handle_top = handle['Close'].max()

        if pd.isna(handle_top):
            logging.warning(f"핸들 상단 값이 유효하지 않습니다. 종목 코드: {df['Code'].iloc[0]}, handle_start_index: {handle_start_index}, handle_end_index: {handle_end_index}")
            return False, None
    else:
        logging.warning(f"{df['Code'].iloc[0]} 핸들 데이터가 부족합니다. handle_start_index: {handle_start_index}, handle_end_index: {handle_end_index}, 데이터 길이: {len(df)}")
        return False, None

    if handle_top < cup_top and cup_bottom < handle_top:
        buy_price = cup_top * 1.01  # 매수 가격 설정 (컵 상단의 1% 상승)
        recent_volume = df['Volume'].iloc[cup_bottom_index - 1]
        average_volume = df['Volume'].rolling(window=5).mean().iloc[cup_bottom_index - 1]

        if recent_volume > average_volume:
            logging.info(f"종목 코드: {df['Code'].iloc[0]} - 매수 신호 발생! 매수 가격: {buy_price}, 현재 가격: {df['Close'].iloc[cup_bottom_index]}")
            return True, df.index[-1]
        else:
            logging.warning(f"종목 코드: {df['Code'].iloc[0]} - 거래량이 충분하지 않아 매수 신호가 없습니다.")
    else:
        logging.debug(f"종목 코드: {df['Code'].iloc[0]} - 패턴 미발견. handle_top: {handle_top}, cup_top: {cup_top}, cup_bottom: {cup_bottom}")

    return False, None

def search_cup_with_handle(stocks_data):
    """저장된 주식 데이터에서 Cup with Handle 패턴을 찾는 함수."""
    recent_cup_with_handle = None
    recent_date = None
    results = []

    for code, data in stocks_data.items():
        df = pd.DataFrame(data)

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.set_index('Date', inplace=True)
        df['Code'] = code

        if df.index.empty:
            logging.warning(f"종목 코드: {code}에 유효한 날짜 데이터가 없습니다.")
            continue

        logging.debug(f"종목 코드: {code}의 날짜 데이터: {df.index.tolist()}")

        is_pattern, pattern_date = is_cup_with_handle(df)
        if is_pattern:
            if recent_date is None or (pattern_date and pattern_date > recent_date):
                recent_date = pattern_date
                recent_cup_with_handle = code
                results.append({
                    'code': code,
                    'pattern_date': pattern_date.strftime('%Y-%m-%d') if pattern_date else None
                })

    return recent_cup_with_handle, recent_date, results

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

    recent_stock, date_found, results = search_cup_with_handle(stocks_data)
    if recent_stock:
        logging.info(f"가장 최근 Cup with Handle 패턴이 발견된 종목: {recent_stock} (완성 날짜: {date_found})")
    else:
        logging.info("Cup with Handle 패턴을 가진 종목이 없습니다.")

    result_filename = os.path.join(json_dir, 'cup_with_handle_results.json')
    with open(result_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    logging.info(f"결과를 JSON 파일로 저장했습니다: {result_filename}")
