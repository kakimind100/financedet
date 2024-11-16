import FinanceDataReader as fdr
import pandas as pd
import ta  # 기술적 지표 계산을 위한 라이브러리
import logging
from datetime import datetime, timedelta
import os
import sqlite3
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

def calculate_indicators(df):
    """MACD와 윌리엄스 %R을 계산하는 함수."""
    df['macd'] = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9).macd()
    df['williams_r'] = ta.momentum.WilliamsRIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14).williams_r()
    return df

def process_stock(code, start_date):
    """주식 데이터를 처리하는 함수."""
    logging.info(f"{code} 처리 시작")
    try:
        df = fdr.DataReader(code, start=start_date)
        logging.info(f"{code} 데이터 가져오기 성공, 가져온 데이터 길이: {len(df)}")

        # 데이터 길이 체크: 최소 40일 데이터
        if len(df) < 40:
            logging.warning(f"{code} 데이터가 40일 미만으로 건너뜁니다.")
            return None

        # 최근 40일 데이터에서 마지막 26일 데이터 추출
        recent_data = df.iloc[-40:]  # 최근 40일 데이터
        last_26_days = recent_data.iloc[-26:]  # 마지막 26일 데이터
        last_close = last_26_days['Close'].iloc[-1]  # 최근 종가

        # 최근 26일 중 29% 이상 상승한 종가 확인
        if any(last_26_days['Close'].iloc[i] >= last_26_days['Close'].iloc[i-1] * 1.29 for i in range(1, len(last_26_days))):
            logging.info(f"{code} 최근 26일 내 29% 이상 상승한 종목 발견: 최근 종가 {last_close}")

            df = calculate_indicators(df)  # MACD와 윌리엄스 %R 계산
            
            macd_last = df['macd'].iloc[-1]
            williams_r_last = df['williams_r'].iloc[-1]

            # 조건 확인
            return {
                'Code': code,
                'Last Close': last_close,
                'MACD': macd_last,
                'Williams %R': williams_r_last
            }
        else:
            logging.info(f"{code} 최근 26일 내 29% 이상 상승한 종목 없음: 최근 종가 {last_close}")

        return None
    except Exception as e:
        logging.error(f"{code} 처리 중 오류 발생: {e}")
        return None

def get_trading_days(start_date, num_days):
    """주식 개장일을 기준으로 지정된 개수의 거래일을 반환하는 함수."""
    trading_days = []
    current_date = start_date

    # 공휴일 리스트 (예: 2024년 공휴일)
    holidays = [
        datetime(2024, 1, 1),   # 신정
        datetime(2024, 3, 1),   # 삼일절
        datetime(2024, 5, 1),   # 근로자의 날
        datetime(2024, 5, 5),   # 어린이날
        datetime(2024, 5, 27),  # 석가탄신일
        datetime(2024, 6, 6),   # 현충일
        datetime(2024, 8, 15),  # 광복절
        datetime(2024, 9, 30),  # 추석
        datetime(2024, 10, 3),  # 개천절
        datetime(2024, 10, 9),  # 한글날
        datetime(2024, 12, 25)  # 성탄절
    ]
    
    while len(trading_days) < num_days:
        # 주말 확인 (토요일: 5, 일요일: 6) 및 공휴일 확인
        if current_date.weekday() < 5 and current_date not in holidays:
            trading_days.append(current_date)
        current_date += timedelta(days=1)

    return trading_days

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
    results = []

    # 멀티스레딩으로 주식 데이터 처리
    with ThreadPoolExecutor(max_workers=10) as executor:  # 최대 10개의 스레드 사용
        futures = {executor.submit(process_stock, code, start_date): code for code in stocks['Code']}
        for future in as_completed(futures):
            result_data = future.result()
            if result_data:
                results.append(result_data)

    logging.info("주식 검색 완료")

    # 필터링: 조건에 맞지 않는 종목 제외
    filtered_results = []
    for item in results:
        if item['Last Close'] <= 1000000 and item['MACD'] <= 5 and item['Williams %R'] <= 0:  # 필터 조건
            filtered_results.append(item)

    return filtered_results

def save_to_database(results):
    """결과를 데이터베이스에 저장하는 함수."""
    conn = sqlite3.connect('stock_analysis.db')
    cursor = conn.cursor()
    
    # 테이블 생성
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS stock_results (
        Code TEXT PRIMARY KEY,
        Last_Close REAL,
        MACD REAL,
        Williams_R REAL
    )
    ''')

    # 데이터 삽입
    for item in results:
        cursor.execute('''
        INSERT OR REPLACE INTO stock_results (Code, Last_Close, MACD, Williams_R)
        VALUES (?, ?, ?, ?)
        ''', (item['Code'], item['Last Close'], item['MACD'], item['Williams %R']))

    conn.commit()
    conn.close()
    logging.info("결과를 데이터베이스에 저장 완료")

if __name__ == "__main__":
    logging.info("스크립트 실행 시작")
    
    # 시작 날짜 설정 (예: 오늘)
    today = datetime.today()

    # 최근 40 거래일 구하기
    trading_days = get_trading_days(today, 40)
    start_date_str = trading_days[-1].strftime('%Y-%m-%d')  # 가장 오래된 거래일

    logging.info(f"주식 분석 시작 날짜: {start_date_str}")

    result = search_stocks(start_date_str)
    
    if result:
        for item in result:
            print(item)  # 콘솔에 결과 출력
        save_to_database(result)  # 데이터베이스에 결과 저장
    else:
        print("조건에 맞는 종목이 없습니다.")
        logging.info("조건에 맞는 종목이 없습니다.")

    logging.info("스크립트 실행 완료")
