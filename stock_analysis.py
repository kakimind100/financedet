import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    if len(df) < 200:  # 200일 이동 평균을 위해 최소 200일의 데이터 필요
        logging.debug(f"거래일 기준 데이터 길이가 200일 미만입니다. 종목 코드: {df['Code'].iloc[0]}")
        return False, None

    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()

    # 이동 평균이 계산되지 않은 경우 경고
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

def is_bullish_divergence(df):
    """다이버전스 패턴을 찾는 함수."""
    if len(df) < 15:  # 충분한 데이터가 필요
        return False, None

    df['RSI'] = 100 - (100 / (1 + (df['Close'].diff(1).clip(lower=0).rolling(window=14).mean() /
                                      df['Close'].diff(1).clip(upper=0).abs().rolling(window=14).mean())))

    # 최근 두 개의 종가와 RSI를 비교
    if (df['Close'].iloc[-1] < df['Close'].iloc[-2] and 
        df['RSI'].iloc[-1] > df['RSI'].iloc[-2]):
        return True, df.index[-1]
    
    return False, None

def is_round_bottom(df):
    """원형 바닥 패턴을 찾는 함수."""
    if len(df) < 30:  # 충분한 데이터가 필요
        return False, None

    # 바닥 형성을 위한 간단한 기준
    recent_low = df['Low'].rolling(window=10).min().iloc[-1]
    recent_high = df['High'].rolling(window=10).max().iloc[-1]

    if df['Close'].iloc[-1] > recent_low and df['Close'].iloc[-1] < recent_high:
        return True, df.index[-1]

    return False, None

def evaluate_stock(stock_data):
    """주어진 종목 데이터에 대해 평가 기준을 적용하여 점수를 매기는 함수."""
    df = pd.DataFrame(stock_data)

    # 최근 상승폭
    if len(df) >= 11:
        recent_gain = (df['Close'].iloc[-1] - df['Close'].iloc[-10]) / df['Close'].iloc[-10] * 100
    else:
        recent_gain = float('-inf')  # 데이터 부족

    # 거래량 증가
    if len(df) >= 10:
        avg_volume = df['Volume'].iloc[-11:-1].mean()  # 최근 10일 평균 거래량
        current_volume = df['Volume'].iloc[-1]
        volume_increase = (current_volume - avg_volume) / avg_volume * 100
    else:
        volume_increase = float('-inf')  # 데이터 부족

    # RSI 계산
    if len(df) >= 15:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
    else:
        rsi = float('-inf')  # 데이터 부족

    # 평가 점수 계산 (가중치 조정 가능)
    score = recent_gain + volume_increase + (rsi - 50)  # RSI는 50을 기준으로 점수화

    return score

def search_patterns_and_find_top(stocks_data):
    """각 패턴을 탐지하고, 모든 데이터가 저장된 후 가장 좋은 상태의 종목 50개를 찾는 함수."""
    results = []

    for code, data in stocks_data.items():
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.set_index('Date', inplace=True)
        df['Code'] = code

        if df.index.empty:
            logging.warning(f"종목 코드: {code}에 유효한 날짜 데이터가 없습니다.")
            continue

        # 각 패턴 탐지
        is_cup, cup_date = is_cup_with_handle(df)
        is_golden, cross_date = is_golden_cross(df)
        is_divergence, divergence_date = is_bullish_divergence(df)
        is_round_bottom, round_bottom_date = is_round_bottom(df)

        # 패턴 결과 저장
        pattern_info = {
            'code': code,
            'cup': is_cup,
            'golden_cross': is_golden,
            'divergence': is_divergence,
            'round_bottom': is_round_bottom,
            'data': df.to_dict(orient='records')
        }

        results.append(pattern_info)

    # 모든 패턴이 발견된 종목 필터링
    all_patterns_found = [res for res in results if res['cup'] and res['golden_cross'] and res['divergence'] and res['round_bottom']]

    # 종목 평가 및 점수 계산
    for item in all_patterns_found:
        score = evaluate_stock(item['data'])
        item['score'] = score

    # 점수 기준으로 정렬하고 상위 50개 선택
    top_50_stocks = sorted(all_patterns_found, key=lambda x: x['score'], reverse=True)[:50]

    return top_50_stocks

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
            logging.info(f"종목 코드: {stock['code']} - 점수: {stock['score']}")
    else:
        logging.info("모든 패턴을 만족하는 종목이 없습니다.")

