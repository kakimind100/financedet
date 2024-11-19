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
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 콘솔에도 로그 출력
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)

# 데이터베이스 연결 및 테이블 생성
def create_database():
    # 데이터베이스를 저장할 디렉토리 경로
    db_directory = 'data'
    os.makedirs(db_directory, exist_ok=True)  # 디렉토리가 없으면 생성

    # 데이터베이스 파일 경로
    db_path = os.path.join(db_directory, 'stock_data.db')
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_data (
            code TEXT,
            date DATE,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            PRIMARY KEY (code, date)
        )
    ''')
    conn.commit()
    logging.info("데이터베이스 및 테이블 생성 완료.")

    # 730일이 지난 데이터 삭제
    delete_old_data(conn)
    conn.close()

def delete_old_data(conn):
    """730일이 지난 데이터를 삭제하는 함수."""
    cursor = conn.cursor()
    cutoff_date = datetime.now() - timedelta(days=730)
    cursor.execute('DELETE FROM stock_data WHERE date < ?', (cutoff_date.date(),))
    conn.commit()
    logging.info(f"730일이 지난 데이터 삭제 완료: {cursor.rowcount}개의 레코드 삭제됨.")

def fetch_existing_data(code):
    """기존 데이터를 조회하여 해당 주식 코드의 모든 데이터를 반환하는 함수."""
    conn = sqlite3.connect('data/stock_data.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT date FROM stock_data WHERE code = ?', (code,))
    existing_dates = cursor.fetchall()
    
    conn.close()
    
    return {date[0] for date in existing_dates}

def save_to_database(data):
    conn = sqlite3.connect('data/stock_data.db')
    cursor = conn.cursor()
    
    for item in data:
        code = item['Code']
        date = item['Date']
        open_price = item['Opening Price']
        high_price = item['Highest Price']
        low_price = item['Lowest Price']
        close_price = item['Last Close']
        volume = item['Volume']

        try:
            cursor.execute('''
                INSERT INTO stock_data (code, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(code, date) DO UPDATE SET
                    open = excluded.open,
                    high = excluded.high,
                    low = excluded.low,
                    close = excluded.close,
                    volume = excluded.volume
            ''', (code, date, open_price, high_price, low_price, close_price, volume))
            conn.commit()
            logging.info(f"{code} - {date} 데이터 저장 완료.")
        except Exception as e:
            logging.error(f"{code} - {date} 데이터 저장 실패: {e}")

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

        existing_dates = fetch_existing_data(code)

        result = []
        for index, row in df.iterrows():
            date_str = index.strftime('%Y-%m-%d')
            if date_str not in existing_dates:
                result.append({
                    'Code': str(code),
                    'Date': date_str,
                    'Opening Price': float(row['Open']),
                    'Highest Price': float(row['High']),
                    'Lowest Price': float(row['Low']),
                    'Last Close': float(row['Close']),
                    'Volume': int(row['Volume'])
                })
                logging.info(f"{code} - {date_str} 데이터 추가 대상.")

        logging.info(f"{code} 데이터 처리 완료: {len(result)}개 항목.")
        return result

    except Exception as e:
        logging.error(f"{code} 처리 중 오류 발생: {e}")
        return []

def initialize_database():
    """데이터베이스를 초기화하는 함수."""
    if not os.path.exists('data/stock_data.db'):
        create_database()
        logging.info("새 데이터베이스를 생성했습니다.")
    else:
        logging.info("기존 데이터베이스를 사용합니다.")

    # 현재 작업 디렉토리 출력
    logging.info(f"현재 작업 디렉토리: {os.getcwd()}")
    if os.path.exists('data/stock_data.db'):
        logging.info("데이터베이스 파일이 존재합니다.")
    else:
        logging.warning("데이터베이스 파일이 존재하지 않습니다.")

def main():
    logging.info("스크립트 실행 시작")
    initialize_database()

    today = datetime.today()
    start_date = today - timedelta(days=730)
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

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_and_store_stock_data, code, start_date): code for code in stocks['Code']}
        for future in as_completed(futures):
            stock_data = future.result()
            if stock_data:
                all_results.extend(stock_data)

    if all_results:
        save_to_database(all_results)
        logging.info(f"총 저장된 데이터 수: {len(all_results)}")
    else:
        logging.info("저장할 데이터가 없습니다.")

if __name__ == "__main__":
    main()
