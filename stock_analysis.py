import FinanceDataReader as fdr
import pandas as pd
import logging
import os
import sqlite3
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# 로그 디렉토리 생성
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# 로깅 설정
logging.basicConfig(
    filename=os.path.join(log_dir, 'stock_analysis.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 데이터베이스 연결 및 테이블 생성
def create_database():
    conn = sqlite3.connect('stocks.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_data (
            code TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            PRIMARY KEY (code, date)
        )
    ''')
    conn.commit()
    conn.close()
    logging.info("데이터베이스 및 테이블 생성 완료.")

def save_to_database(data):
    conn = sqlite3.connect('stocks.db')
    cursor = conn.cursor()
    
    for item in data:
        code = item['Code']
        date = item['Date']
        open_price = item['Opening Price']
        high_price = item['Highest Price']
        low_price = item['Lowest Price']
        close_price = item['Last Close']
        volume = item.get('Volume', 0)  # 거래량이 없는 경우 기본값 0

        cursor.execute('''
            INSERT OR REPLACE INTO stock_data (code, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (code, date, open_price, high_price, low_price, close_price, volume))
    
    conn.commit()
    conn.close()
    logging.info(f"{len(data)}개의 데이터를 데이터베이스에 저장 완료.")

def analyze_stock(code, start_date):
    """주식 데이터를 분석하고 데이터베이스에 저장하는 함수."""
    logging.info(f"{code} 데이터 분석 시작")
    try:
        df = fdr.DataReader(code, start=start_date)
        logging.info(f"{code} 데이터 가져오기 성공, 가져온 데이터 길이: {len(df)}")

        if len(df) < 1:
            logging.warning(f"{code} 데이터가 없습니다.")
            return None

        # 최근 데이터 저장
        result = []
        for index, row in df.iterrows():
            result.append({
                'Code': str(code),
                'Date': index.strftime('%Y-%m-%d'),
                'Opening Price': float(row['Open']),
                'Highest Price': float(row['High']),
                'Lowest Price': float(row['Low']),
                'Last Close': float(row['Close']),
                'Volume': int(row['Volume'])  # 거래량
            })
            # 각 종목의 데이터 로그 추가
            logging.info(f"{code} - 날짜: {index.strftime('%Y-%m-%d')}, "
                         f"시가: {row['Open']}, 최고가: {row['High']}, "
                         f"최저가: {row['Low']}, 종가: {row['Close']}, "
                         f"거래량: {row['Volume']}")

        logging.info(f"{code} 데이터 처리 완료: {len(result)}개 항목.")
        return result  # 결과 반환

    except Exception as e:
        logging.error(f"{code} 처리 중 오류 발생: {e}")
        return None

def search_stocks(start_date):
    """주식 종목을 검색하고 데이터를 데이터베이스에 저장하는 함수."""
    logging.info("주식 검색 시작")

    try:
        kospi = fdr.StockListing('KOSPI')
        logging.info("코스피 종목 목록 가져오기 성공")
        
        kosdaq = fdr.StockListing('KOSDAQ')
        logging.info("코스닥 종목 목록 가져오기 성공")
    except Exception as e:
        logging.error(f"종목 목록 가져오기 중 오류 발생: {e}")
        return []

    stocks = pd.concat([kospi, kosdaq])
    all_results = []

    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(analyze_stock, code, start_date): code for code in stocks['Code']}
        for future in as_completed(futures):
            result_data = future.result()
            if result_data:
                all_results.extend(result_data)

    logging.info("주식 검색 완료")
    return all_results

# 메인 실행 블록
if __name__ == "__main__":
    logging.info("스크립트 실행 시작")
    create_database()  # 데이터베이스 및 테이블 생성

    today = datetime.today()
    start_date = today - timedelta(days=730)  # 최근 2년(730일) 전 날짜
    start_date_str = start_date.strftime('%Y-%m-%d')

    logging.info(f"주식 분석 시작 날짜: {start_date_str}")

    results = search_stocks(start_date_str)  # 결과를 변수에 저장
    if results:  # 결과가 있을 때만 데이터베이스에 저장
        save_to_database(results)  # 데이터베이스에 저장
        logging.info(f"저장된 종목 수: {len(results)}")
    else:
        logging.info("조건을 만족하는 종목이 없습니다.")
