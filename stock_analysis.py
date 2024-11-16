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

        # 최고가가 29% 이상 상승한 종목 확인
        high_condition = recent_20_days['High'].max() >= recent_20_days['High'].iloc[0] * 1.29

        # 장대 양봉 조건 체크
        last_candle = recent_data.iloc[-1]
        previous_candle = recent_data.iloc[-2]

        # 장대 양봉 여부
        is_bullish_engulfing = (last_candle['Close'] > previous_candle['Close']) and \
                               ((last_candle['Close'] - last_candle['Open']) > (previous_candle['Close'] - previous_candle['Open'])) and \
                               (last_candle['Low'] < previous_candle['Close'])

        # 장대 양봉의 저가 체크
        if high_condition and is_bullish_engulfing:
            # 장대 양봉의 저가 아래로 떨어진 경우 체크
            if recent_data['Low'].iloc[-1] < last_candle['Low']:
                logging.info(f"{code} 장대 양봉의 저가 아래로 떨어져 제외됨.")
                return None

            # 윌리엄스 %R 계산
            df = calculate_indicators(df)  # 윌리엄스 %R 계산

            # 윌리엄스 %R 조건 확인
            williams_r = df['williams_r'].iloc[-1]
            if williams_r <= -90:
                result = {
                    'Code': code,
                    'Last Close': last_close,
                    'Williams %R': williams_r
                }
                logging.info(f"{code} 조건 만족: {result}")
                return result
            elif -30 <= williams_r < -90:
                # 윌리엄스 %R이 -30 이상인 경우, 5% 이상 상승 이력 확인
                if any(recent_20_days['Close'].iloc[i] >= recent_20_days['Close'].iloc[i-1] * 1.05 for i in range(1, len(recent_20_days))):
                    logging.info(f"{code} 윌리엄스 %R이 -30 이상이며 5% 이상 상승 이력이 있어 제외됨: Williams %R={williams_r}")
                    return None
                else:
                    result = {
                        'Code': code,
                        'Last Close': last_close,
                        'Williams %R': williams_r
                    }
                    logging.info(f"{code} 조건 만족: {result}")
                    return result
            else:
                logging.info(f"{code} 윌리엄스 %R 조건 불만족: Williams %R={williams_r}")

        return None
    except Exception as e:
        logging.error(f"{code} 처리 중 오류 발생: {e}")
        return None

def search
