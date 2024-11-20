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
    """컵과 핸들 패턴을 찾는 함수. 조건을 완화했습니다."""
    if len(df) < 40:  # 데이터 길이를 40으로 완화
        logging.debug(f"데이터 길이가 40일 미만입니다. 종목 코드: {df['Code'].iloc[0]}")
        return False, None

    cup_bottom = df['Low'].min()
    cup_bottom_index = df['Low'].idxmin()
    cup_bottom_index = df.index.get_loc(cup_bottom_index)

    cup_top = df['Close'][:cup_bottom_index].max()
    handle_start_index = cup_bottom_index + 1

    # 핸들 정의: 길이를 5일에서 10일로 유연하게 설정
    handle_length = min(10, len(df) - handle_start_index)
    handle = df.iloc[handle_start_index:handle_start_index + handle_length]

    if handle.empty:
        logging.warning(f"{df['Code'].iloc[0]} 핸들 데이터가 부족합니다.")
        return False, None

    handle_top = handle['Close'].max()

    # 컵과 핸들 조건 강화
    cup_depth = (cup_top - cup_bottom) / cup_top  # 컵 깊이 비율
    handle_depth = (handle_top - cup_bottom) / cup_top  # 핸들 깊이 비율

    # 조건 완화: 컵 깊이를 0.15 이상으로 설정하고 핸들 깊이를 0.1 이하로 설정
    if cup_depth < 0.15 or handle_depth > 0.1:
        logging.warning(f"종목 코드: {df['Code'].iloc[0]} - 컵 또는 핸들 조건이 충족되지 않음. 컵 깊이: {cup_depth}, 핸들 깊이: {handle_depth}")
        return False, None

    # 매수 신호 조건 강화
    if handle_top < cup_top and cup_bottom < handle_top:
        buy_price = cup_top * 1.02  # 매수 가격을 컵 상단의 2% 상승으로 설정
        recent_volume = df['Volume'].iloc[cup_bottom_index - 1]
        average_volume = df['Volume'].rolling(window=5).mean().iloc[cup_bottom_index - 1]

        if recent_volume > average_volume * 1.5:  # 거래량이 평균의 1.5배 이상이어야 매수 신호
            logging.info(f"종목 코드: {df['Code'].iloc[0]} - 매수 신호 발생! 매수 가격: {buy_price}, 현재 가격: {df['Close'].iloc[cup_bottom_index]}")
            return True, df.index[-1]
        else:
            logging.warning(f"종목 코드: {df['Code'].iloc[0]} - 거래량이 충분하지 않아 매수 신호가 없습니다.")
    else:
        logging.debug(f"종목 코드: {df['Code'].iloc[0]} - 패턴 미발견. handle_top: {handle_top}, cup_top: {cup_top}, cup_bottom: {cup_bottom}")

    return False, None

def is_golden_cross(df):
    """골든 크로스 조건을 확인하는 함수."""
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    if len(df) > 200:
        if df['MA50'].iloc[-2] < df['MA200'].iloc[-2] and df['MA50'].iloc[-1] > df['MA200'].iloc[-1]:
            logging.info(f"{df['Code'].iloc[0]} - 골든 크로스 발생!")
            return True
    return False

def is_breakout(df):
    """돌파 패턴을 확인하는 함수."""
    if df['Close'].count() < 20:  # 거래일 기준으로 20일 이상이 필요
        logging.warning(f"{df['Code'].iloc[0]} - 데이터가 부족하여 돌파 패턴 확인 불가")
        return False

    resistance = df['Close'].rolling(window=20).max().iloc[-2]  # 20일 최고가
    if df['Close'].iloc[-1] > resistance:
        logging.info(f"{df['Code'].iloc[0]} - 돌파 패턴 발생!")
        return True
    return False

def search_patterns(stocks_data):
    """저장된 주식 데이터에서 다양한 패턴을 찾는 함수."""
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
        is_cup_handle, pattern_date = is_cup_with_handle(df)
        is_breakout_pattern = is_breakout(df)
        is_golden_cross_pattern = is_golden_cross(df)

        # 조건이 모두 만족할 경우
        if is_cup_handle and is_breakout_pattern and df['RSI'].iloc[-1] < 30:  # RSI가 30 이하일 때
            results.append({
                'code': code,
                'pattern_date': pattern_date.strftime('%Y-%m-%d') if pattern_date else None,
                'type': 'Cup with Handle and Breakout'
            })
        
        # 골든 크로스 패턴이 발견된 경우 추가
        elif is_golden_cross_pattern:
            results.append({
                'code': code,
                'pattern_date': df.index[-1].strftime('%Y-%m-%d'),
                'type': 'Golden Cross'
            })

        # 돌파 패턴이 발견된 경우 추가
        elif is_breakout_pattern:
            results.append({
                'code': code,
                'pattern_date': df.index[-1].strftime('%Y-%m-%d'),
                'type': 'Breakout'
            })

    return results  # 모든 패턴 확인 후 결과 반환

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

    # 최근 발견된 패턴을 가진 종목을 로그로 기록
    if results:
        for result in results:
            logging.info(f"발견된 종목: {result['code']} (완성 날짜: {result['pattern_date']}, 유형: {result['type']})")
    else:
        logging.info("발견된 패턴이 없습니다.")

    result_filename = os.path.join(json_dir, 'pattern_results.json')
    with open(result_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    logging.info(f"결과를 JSON 파일로 저장했습니다: {result_filename}")
               
