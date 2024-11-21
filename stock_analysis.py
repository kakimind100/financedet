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

def is_bullish_divergence(df):
    """다이버전스 패턴을 찾는 함수."""
    if len(df) < 15:
        return False, None

    df['RSI'] = 100 - (100 / (1 + (df['Close'].diff(1).clip(lower=0).rolling(window=14).mean() /
                                      df['Close'].diff(1).clip(upper=0).abs().rolling(window=14).mean())))

    if (df['Close'].iloc[-1] < df['Close'].iloc[-2] and 
        df['RSI'].iloc[-1] > df['RSI'].iloc[-2]):
        return True, df.index[-1]
    
    return False, None

def is_round_bottom(df):
    """원형 바닥 패턴을 찾는 함수."""
    if len(df) < 30:
        return False, None

    recent_low = df['Low'].rolling(window=10).min().iloc[-1]
    recent_high = df['High'].rolling(window=10).max().iloc[-1]

    if df['Close'].iloc[-1] > recent_low and df['Close'].iloc[-1] < recent_high:
        return True, df.index[-1]

    return False, None

def calculate_moving_average(df, window):
    return df['Close'].rolling(window=window).mean()

def calculate_macd(df):
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def calculate_bollinger_bands(df, window=20):
    rolling_mean = df['Close'].rolling(window).mean()
    rolling_std = df['Close'].rolling(window).std()
    upper_band = rolling_mean + (rolling_std * 2)
    lower_band = rolling_mean - (rolling_std * 2)
    return upper_band, lower_band

def calculate_stochastic_oscillator(df, k_window=14, d_window=3):
    low_min = df['Low'].rolling(window=k_window).min()
    high_max = df['High'].rolling(window=k_window).max()
    stoch_k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    stoch_d = stoch_k.rolling(window=d_window).mean()
    return stoch_k, stoch_d

def calculate_cci(df, window=20):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    sma = typical_price.rolling(window).mean()
    mad = (typical_price - sma).abs().rolling(window).mean()
    cci = (typical_price - sma) / (0.015 * mad)
    return cci

def evaluate_stock(stock_data):
    df = pd.DataFrame(stock_data)

    # 기술적 지표 계산
    df['MA20'] = calculate_moving_average(df, 20)
    macd, signal = calculate_macd(df)
    df['MACD'] = macd
    df['Signal'] = signal

    df['Upper_Band'], df['Lower_Band'] = calculate_bollinger_bands(df)
    df['Stoch_K'], df['Stoch_D'] = calculate_stochastic_oscillator(df)
    df['CCI'] = calculate_cci(df)

    # 기존 평가 기준
    recent_gain = (df['Close'].iloc[-1] - df['Close'].iloc[-10]) / df['Close'].iloc[-10] * 100
    avg_volume = df['Volume'].iloc[-11:-1].mean()
    current_volume = df['Volume'].iloc[-1]
    volume_increase = (current_volume - avg_volume) / avg_volume * 100

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs)).iloc[-1]

    # 평가 점수 계산
    score = (recent_gain + volume_increase + (rsi - 50) +
             (df['MACD'].iloc[-1] > df['Signal'].iloc[-1]) * 10 +
             (df['Close'].iloc[-1] > df['Upper_Band'].iloc[-1]) * 5 +
             (df['Stoch_K'].iloc[-1] > df['Stoch_D'].iloc[-1]) * 5 +
             (df['CCI'].iloc[-1] > 100) * 5)

    return score

def should_buy(stock_data):
    df = pd.DataFrame(stock_data)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs)).iloc[-1]

    # 컵과 핸들 패턴 확인
    is_cup, _ = is_cup_with_handle(df)

    # 원형 바닥 패턴 확인
    is_round_bottom_found, _ = is_round_bottom(df)

    # 매수 조건
    if (rsi < 30) and (is_cup or is_round_bottom_found):
        return True, rsi
    return False, rsi

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

        # 각 패턴 탐지
        is_cup, cup_date = is_cup_with_handle(df)
        is_divergence, divergence_date = is_bullish_divergence(df)
        is_round_bottom_found, round_bottom_date = is_round_bottom(df)

        # 패턴 결과 저장
        pattern_info = {
            'code': code,
            'cup': is_cup,
            'divergence': is_divergence,
            'round_bottom': is_round_bottom_found,
            'data': df.to_dict(orient='records')
        }

        results.append(pattern_info)

    # 모든 패턴이 발견된 종목 필터링
    all_patterns_found = [res for res in results if res['cup'] and res['divergence'] and res['round_bottom']]

    # 종목 평가 및 점수 계산
    for item in all_patterns_found:
        score = evaluate_stock(item['data'])
        logging.info(f"종목 코드: {item['code']} - 점수: {score}")  # 점수 출력
        # 점수는 결과에 포함하지 않음

    # 점수 기준으로 정렬하고 상위 20개 선택
    top_20_stocks = sorted(all_patterns_found, key=lambda x: x['score'], reverse=True)[:20]

    # top_20_stocks를 JSON 파일로 저장 (점수 제외)
    filename = os.path.join(json_dir, 'top_20_stocks.json')
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(top_20_stocks, f, default=str, ensure_ascii=False, indent=4)

    # Discord 웹훅 스크립트로 데이터 전달
    subprocess.run(["python", "discord_webhook.py", filename])

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
            logging.info(f"종목 코드: {stock['code']} - 점수: {stock['score']}")
    else:
        logging.info("모든 패턴을 만족하는 종목이 없습니다.")

