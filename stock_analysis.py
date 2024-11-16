import FinanceDataReader as fdr
import pandas as pd
import logging
from datetime import datetime, timedelta
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# 로그 디렉토리 생성
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# 로깅 설정
logging.basicConfig(
    filename=os.path.join(log_dir, 'stock_analysis.log'),
    level=logging.DEBUG,  # DEBUG 레벨로 모든 로그 기록
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 콘솔에도 로그 출력
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # 콘솔에는 INFO 레벨 이상의 로그만 출력
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)

def calculate_williams_r(df, window=14):
    """Williams %R을 직접 계산하는 함수."""
    logging.debug(f"Calculating Williams %R with window size: {window}")
    highest_high = df['High'].rolling(window=window).max()
    lowest_low = df['Low'].rolling(window=window).min()
    williams_r = -100 * (highest_high - df['Close']) / (highest_high - lowest_low)
    return williams_r

def calculate_indicators(df):
    """지표를 계산하는 함수."""
    df['williams_r'] = calculate_williams_r(df, window=14)
    return df

def process_stock(code, start_date):
    """주식 데이터를 처리하는 함수."""
    logging.info(f"{code} 처리 시작")
    try:
        df = fdr.DataReader(code, start=start_date)
        logging.info(f"{code} 데이터 가져오기 성공, 가져온 데이터 길이: {len(df)}")
        
        # 데이터 길이 체크: 최소 26일 데이터
        if len(df) < 26:
            logging.warning(f"{code} 데이터가 26일 미만으로 건너뜁니다.")
            return None
        
        # 최근 30일 데이터 추출
        recent_data = df.iloc[-30:]  # 최근 30일 데이터
        last_close = recent_data['Close'].iloc[-1]  # 최근 종가
        prev_close = recent_data['Close'].iloc[-2]  # 이전 종가
        
        # 최근 20일 데이터 추출
        recent_20_days = recent_data.iloc[-20:]

        # 장대 양봉 조건 체크
        last_candle = recent_data.iloc[-1]
        previous_candle = recent_data.iloc[-2]

        # 장대 양봉 여부
        is_bullish_engulfing = (last_candle['Close'] > previous_candle['Close']) and \
                               ((last_candle['Close'] - last_candle['Open']) > (previous_candle['Close'] - previous_candle['Open'])) and \
                               (last_candle['Low'] < previous_candle['Close'])

        # 장대 양봉 이후 하향하는지 확인
        if is_bullish_engulfing and last_close < last_candle['Open']:
            logging.info(f"{code} 장대 양봉 이후 하향하는 종목: 최근 종가 {last_close}")

            # 윌리엄스 %R 계산
            df = calculate_indicators(df)  # 윌리엄스 %R 계산
            williams_r = df['williams_r'].iloc[-1]

            # 장대 양봉 이후 7% 이상 상승한 이력 확인
            bullish_after = recent_data.iloc[:-1]  # 마지막 봉을 제외한 데이터
            has_risen_7_percent = any(bullish_after['Close'].iloc[i] >= bullish_after['Close'].iloc[i-1] * 1.07 for i in range(1, len(bullish_after)))

            if not has_risen_7_percent:
                # 조건 확인
                if williams_r <= -90:
                    result = {
                        'Code': code,
                        'Last Close': last_close,
                        'Williams %R': williams_r
                    }
                    logging.info(f"{code} 조건 만족: {result}")
                    return result
                else:
                    logging.info(f"{code} 윌리엄스 %R 조건 불만족: Williams %R={williams_r}")
            else:
                logging.info(f"{code} 장대 양봉 이후 7% 이상 상승한 이력이 있어 제외됨.")

        return None
    except Exception as e:
        logging.error(f"{code} 처리 중 오류 발생: {e}")
        return None

def search_stocks(start_date):
    """주식 종목을 검색하는 함수."""
    logging.info("주식 검색 시작")
    
    try:
        kospi = fdr.StockListing('KOSPI')  # KRX 코스피 종목 목록
        logging.info("코스피 종목 목록 가져오기 성공")
        
        kosdaq = fdr.StockListing('KOSDAQ')  # KRX 코스닥 종목 목록
        logging.info("코스닥 종목 목록 가져오기 성공")
    except Exception as e:
        logging.error(f"종목 목록 가져오기 중 오류 발생: {e}")
        return []

    stocks = pd.concat([kospi, kosdaq])
    result = []

    # 멀티스레딩으로 주식 데이터 처리
    with ThreadPoolExecutor(max_workers=10) as executor:  # 최대 10개의 스레드 사용
        futures = {executor.submit(process_stock, code, start_date): code for code in stocks['Code']}
        for future in as_completed(futures):
            result_data = future.result()
            if result_data:
                result.append(result_data
