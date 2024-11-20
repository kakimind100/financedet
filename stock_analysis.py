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

def calculate_rsi(df, window=14):
    """상대 강도 지수 (RSI)를 계산하는 함수."""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def is_cup_with_handle(df):
    """컵과 핸들 패턴을 찾는 함수."""
    if len(df) < 40:  # 데이터 길이를 40으로 완화
        logging.debug(f"데이터 길이가 40일 미만입니다. 종목 코드: {df['Code'].iloc[0]}")
        return False

    cup_bottom = df['Low'].min()
    cup_bottom_index = df['Low'].idxmin()  # 컵의 바닥 인덱스
    cup_top = df['Close'][:cup_bottom_index].max()  # 컵의 상단

    # 핸들 시작 인덱스
    handle_start_index = df.index.get_loc(cup_bottom_index) + 1  # 정수 인덱스 위치로 변환
    handle_length = min(10, len(df) - handle_start_index)
    handle = df.iloc[handle_start_index:handle_start_index + handle_length]

    if handle.empty:
        logging.warning(f"{df['Code'].iloc[0]} 핸들 데이터가 부족합니다.")
        return False

    handle_top = handle['Close'].max()
    cup_depth = (cup_top - cup_bottom) / cup_top
    handle_depth = (handle_top - cup_bottom) / cup_top

    # 컵 깊이와 핸들 깊이에 대한 조건 완화
    if cup_depth < 0.05 or handle_depth > 0.2:  # 조건 완화
        logging.warning(f"종목 코드: {df['Code'].iloc[0]} - 컵 또는 핸들 조건이 충족되지 않음.")
        return False

    # 핸들이 컵의 상단보다 낮아야 하며 컵의 바닥 위에 있어야 함
    if handle_top < cup_top and handle_top > cup_bottom:
        return True

    return False

def is_golden_cross(df):
    """골든 크로스 조건을 확인하는 함수."""
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    if len(df) > 200:
        if df['MA50'].iloc[-2] < df['MA200'].iloc[-2] and df['MA50'].iloc[-1] > df['MA200'].iloc[-1]:
            logging.info(f"{df['Code'].iloc[0]} - 골든 크로스 발생!")
            return True
    return False

def search_patterns(stocks_data):
    """저장된 주식 데이터에서 패턴을 찾는 함수."""
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

        # RSI 계산
        df['RSI'] = calculate_rsi(df)

        # 패턴 확인
        is_cup_handle = is_cup_with_handle(df)
        is_golden_cross_pattern = is_golden_cross(df)

        # 하나라도 만족하는 경우
        if is_cup_handle or is_golden_cross_pattern:
            results.append({
                'code': code,
                'data': df.astype(object).to_dict(orient='records')  # 데이터 전체를 JSON 직렬화 가능한 형식으로 변환
            })

    return results

# 메인 실행 블록
if __name__ == "__main__":
    logging.info("주식 분석 스크립트 실행 중...")

    today = datetime.today()
    start_date = today - timedelta(days=730)  # 2년으로 설정
    end_date = today.strftime('%Y-%m-%d')
    start_date_str = start_date.strftime('%Y-%m-%d')

    markets = ['KOSPI', 'KOSDAQ']  # 필요한 경우 다른 시장 추가 가능
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

    results = search_patterns(stocks_data)

    # 결과를 JSON 파일로 저장
    result_filename = os.path.join(json_dir, 'pattern_results.json')
    with open(result_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    logging.info(f"결과를 JSON 파일로 저장했습니다: {result_filename}")

    # Discord 웹훅으로 전송하는 부분은 discord_webhook.py에서 처리
