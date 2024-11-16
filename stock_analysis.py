import FinanceDataReader as fdr
import pandas as pd
import ta  # 기술적 지표 계산을 위한 라이브러리
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

def calculate_williams_r(df, window=14):
    """Williams %R을 직접 계산하는 함수."""
    logging.debug(f"Calculating Williams %R with window size: {window}")
    highest_high = df['High'].rolling(window=window).max()
    lowest_low = df['Low'].rolling(window=window).min()
    williams_r = -100 * (highest_high - df['Close']) / (highest_high - lowest_low)
    logging.debug("Williams %R calculation completed.")
    return williams_r

def calculate_indicators(df):
    """MACD와 윌리엄스 %R을 계산하는 함수."""
    logging.debug("Calculating MACD and Williams %R indicators.")
    df['macd'] = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9).macd()
    df['williams_r'] = calculate_williams_r(df, window=14)
    logging.debug("Indicators calculated successfully.")
    return df

def process_stock(code, start_date):
    """주식 데이터를 처리하는 함수."""
    logging.info(f"Processing stock code: {code}")
    try:
        df = fdr.DataReader(code, start=start_date)
        logging.info(f"{code} data retrieved successfully, number of records: {len(df)}")
        
        # 데이터 길이 체크: 최소 26일 데이터
        if len(df) < 26:
            logging.warning(f"{code} data has less than 26 records, skipping.")
            return None
        
        # 최근 30일 데이터 추출
        recent_data = df.iloc[-30:]  # 최근 30일 데이터
        last_close = recent_data['Close'].iloc[-1]  # 최근 종가
        prev_close = recent_data['Close'].iloc[-2]  # 이전 종가

        # 최근 20일 중 29% 이상 상승한 종가 확인
        if any(recent_data['Close'].iloc[i] >= recent_data['Close'].iloc[i-1] * 1.29 for i in range(1, len(recent_data))):
            logging.info(f"{code} found a stock with over 29% rise within the last 20 days: last close {last_close}, previous close {prev_close}")
            
            # MACD 계산
            df = calculate_indicators(df)  # MACD와 윌리엄스 %R 계산

            # MACD 조건 확인
            if df['macd'].iloc[-1] <= 5:
                logging.info(f"{code} MACD condition satisfied: MACD={df['macd'].iloc[-1]}")
                
                # 윌리엄스 %R 조건 확인
                if df['williams_r'].iloc[-1] <= 0:
                    result = {
                        'Code': code,
                        'Last Close': last_close,
                        'MACD': df['macd'].iloc[-1],
                        'Williams %R': df['williams_r'].iloc[-1]
                    }
                    logging.info(f"{code} conditions satisfied: {result}")
                    return result
                else:
                    logging.info(f"{code} Williams %R condition not satisfied: Williams %R={df['williams_r'].iloc[-1]}")
            else:
                logging.info(f"{code} MACD condition not satisfied: MACD={df['macd'].iloc[-1]}")

        return None
    except Exception as e:
        logging.error(f"Error processing {code}: {str(e)}", exc_info=True)
        return None

def search_stocks(start_date):
    """주식 종목을 검색하는 함수."""
    logging.info("Starting stock search...")
    
    try:
        kospi = fdr.StockListing('KOSPI')  # KRX 코스피 종목 목록
        logging.info("Successfully retrieved KOSPI stock list.")
        
        kosdaq = fdr.StockListing('KOSDAQ')  # KRX 코스닥 종목 목록
        logging.info("Successfully retrieved KOSDAQ stock list.")
    except Exception as e:
        logging.error(f"Error retrieving stock lists: {str(e)}", exc_info=True)
        return []

    stocks = pd.concat([kospi, kosdaq])
    result = []

    # 멀티스레딩으로 주식 데이터 처리
    with ThreadPoolExecutor(max_workers=10) as executor:  # 최대 10개의 스레드 사용
        futures = {executor.submit(process_stock, code, start_date): code for code in stocks['Code']}
        for future in as_completed(futures):
            result_data = future.result()
            if result_data:
                result.append(result_data)

    logging.info("Stock search completed.")
    return result

if __name__ == "__main__":
    logging.info("Script execution started.")
    
    # 최근 40 거래일을 기준으로 시작 날짜 설정
    today = datetime.today()
    start_date = today - timedelta(days=40)  # 최근 40 거래일 전 날짜
    start_date_str = start_date.strftime('%Y-%m-%d')

    logging.info(f"Starting stock analysis from date: {start_date_str}")

    result = search_stocks(start_date_str)
    
    if result:
        for item in result:
            print(item)  # 콘솔에 결과 출력
    else:
        print("No stocks met the conditions.")
        logging.info("No stocks met the conditions.")

    logging.info("Script execution completed.")
