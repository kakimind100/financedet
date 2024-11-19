import FinanceDataReader as fdr
import pandas as pd
import logging
import os
import sqlite3
from datetime import datetime, timedelta

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
        volume = item['Volume']

        cursor.execute('''
            INSERT OR REPLACE INTO stock_data (code, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (code, date, open_price, high_price, low_price, close_price, volume))
    
    conn.commit()
    conn.close()
    logging.info(f"{len(data)}개의 데이터를 데이터베이스에 저장 완료.")

def fetch_and_store_stock_data(code, start_date):
    """주식 데이터를 가져와서 데이터베이스에 저장하는 함수."""
    logging.info(f"{code} 데이터 가져오기 시작")
    try:
        df = fdr.DataReader(code, start=start_date)
        logging.info(f"{code} 데이터 가져오기 성공, 가져온 데이터 길이: {len(df)}")

        if len(df) < 1:
            logging.warning(f"{code} 데이터가 없습니다.")
            return []

        # 데이터 프레임을 리스트로 변환
        result = []
        for index, row in df.iterrows():
            result.append({
                'Code': str(code),
                'Date': index.strftime('%Y-%m-%d'),
                'Opening Price': float(row['Open']),
                'Highest Price': float(row['High']),
                'Lowest Price': float(row['Low']),
                'Last Close': float(row['Close']),
                'Volume': int(row['Volume'])
            })

        logging.info(f"{code} 데이터 처리 완료: {len(result)}개 항목.")
        return result

    except Exception as e:
        logging.error(f"{code} 처리 중 오류 발생: {e}")
        return []

def main():
    logging.info("스크립트 실행 시작")
    create_database()  # 데이터베이스 및 테이블 생성

    today = datetime.today()
    start_date = today - timedelta(days=730)  # 최근 2년(730일) 전 날짜
    start_date_str = start_date.strftime('%Y-%m-%d')

    logging.info(f"주식 분석 시작 날짜: {start_date_str}")

    # KOSPI와 KOSDAQ 종목 목록 가져오기
    try:
        kospi = fdr.StockListing('KOSPI')
        logging.info("코스피 종목 목록 가져오기 성공")
        
        kosdaq = fdr.StockListing('KOSDAQ')
        logging.info("코스닥 종목 목록 가져오기 성공")
    except Exception as e:
        logging.error(f"종목 목록 가져오기 중 오류 발생: {e}")
        return

    stocks = pd.concat([kospi, kosdaq])
    all_results = []

    # 각 종목에 대해 데이터를 가져와서 데이터베이스에 저장
    for code in stocks['Code']:
        stock_data = fetch_and_store_stock_data(code, start_date)
        if stock_data:
            all_results.extend(stock_data)

    if all_results:
        save_to_database(all_results)  # 데이터베이스에 저장
        logging.info(f"총 저장된 데이터 수: {len(all_results)}")
    else:
        logging.info("저장할 데이터가 없습니다.")

if __name__ == "__main__":
    main()
