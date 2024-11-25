import FinanceDataReader as fdr
import pandas as pd
import logging
import os
from datetime import datetime, timedelta
import threading

# 로그 디렉토리 설정
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)  # 로그 디렉토리가 없으면 생성

# 로깅 설정
logging.basicConfig(
    filename=os.path.join(log_dir, 'stock_data_fetcher.log'),  # 로그 파일 경로
    level=logging.INFO,  # 로깅 레벨 설정
    format='%(asctime)s - %(levelname)s - %(message)s'  # 로그 메시지 형식
)

# 콘솔 로그 출력 설정
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # 콘솔 로그 레벨 설정
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))  # 콘솔 로그 형식
logging.getLogger().addHandler(console_handler)  # 콘솔 핸들러 추가

# 거래 정지 종목 목록을 가져오는 함수
def get_suspended_stocks():
    """거래 정지 종목 목록을 가져오는 함수."""
    try:
        suspended_stocks = fdr.StockSuspension()  # 거래 정지 종목 데이터 가져오기
        return suspended_stocks['Code'].tolist()  # 종목 코드 리스트 반환
    except Exception as e:
        logging.error(f"거래 정지 종목 가져오기 중 오류 발생: {e}")
        return []

# 주식 데이터를 가져오는 스레드 함수
def fetch_single_stock_data(code, start_date, end_date, all_stocks_data, suspended_stocks):
    """주식 코드에 대한 데이터를 가져오는 함수."""
    if code in suspended_stocks:  # 거래 정지 종목 제외
        logging.info(f"{code}는 거래 정지 종목으로 제외됩니다.")  # 제외 로그
        return

    try:
        df = fdr.DataReader(code, start_date, end_date)  # 주식 데이터 가져오기
        if df is not None and not df.empty:  # 데이터가 유효한지 확인
            df.reset_index(inplace=True)  # 인덱스 초기화
            df['Code'] = code  # 주식 코드 추가
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')  # 날짜 형식 변경
            all_stocks_data[code] = df  # 가져온 데이터 저장
            logging.info(f"{code} 데이터 가져오기 완료, 데이터 길이: {len(df)}")  # 성공 로그
        else:
            logging.warning(f"{code} 데이터가 비어 있거나 가져오기 실패")  # 경고 로그
    except Exception as e:
        logging.error(f"{code} 데이터 가져오기 중 오류 발생: {e}")  # 오류 로그

def fetch_stock_data(markets, start_date, end_date):
    """주식 데이터를 가져오는 메인 함수."""
    all_stocks_data = {}  # 모든 주식 데이터를 저장할 딕셔너리
    suspended_stocks = get_suspended_stocks()  # 거래 정지 종목 목록 가져오기

    for market in markets:  # 여러 시장에 대해 반복
        codes = fdr.StockListing(market)['Code'].tolist()  # 주식 코드 리스트 가져오기
        threads = []  # 스레드 리스트 초기화

        for code in codes:
            # 주식 코드에 대해 새 스레드 생성
            thread = threading.Thread(target=fetch_single_stock_data, args=(code, start_date, end_date, all_stocks_data, suspended_stocks))
            threads.append(thread)  # 스레드 리스트에 추가
            thread.start()  # 스레드 시작

        for thread in threads:
            thread.join()  # 모든 스레드가 완료될 때까지 대기

    # 데이터프레임으로 변환 후 CSV로 저장
    if all_stocks_data:  # 데이터가 있는 경우
        data_dir = 'data'
        os.makedirs(data_dir, exist_ok=True)  # data 디렉토리가 없으면 생성
        all_data = pd.concat(all_stocks_data.values(), ignore_index=True)  # 모든 데이터 결합
        all_data.to_csv(os.path.join(data_dir, 'stock_data.csv'), index=False)  # CSV로 저장
        logging.info("주식 데이터가 'data/stock_data.csv'로 저장되었습니다.")  # 성공 로그
    else:
        logging.warning("가져온 주식 데이터가 없습니다.")  # 경고 로그

if __name__ == "__main__":
    end_date = datetime.today()  # 오늘 날짜
    start_date = end_date - timedelta(days=365)  # 1년 전 날짜
    fetch_stock_data(['KOSPI', 'KOSDAQ'], start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))  # 데이터 가져오기 실행
