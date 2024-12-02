import FinanceDataReader as fdr
import yfinance as yf
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

# 주식 데이터를 가져오는 스레드 함수
def fetch_single_stock_data(code, start_date, end_date, all_stocks_data):
    """주식 코드에 대한 데이터를 가져오는 함수."""
    try:
        df = fdr.DataReader(code, start_date, end_date)  # 주식 데이터 가져오기
        if df is not None and not df.empty:  # 데이터가 유효한지 확인
            # 최근 26일 데이터 확인
            recent_data = df.tail(26)  # 최근 26일 데이터

            # 오늘의 시작가 확인
            today_open = recent_data['Open'].iloc[-1]  # 오늘의 시작가

            # 시작가가 0인 경우 제외
            if today_open == 0:
                logging.warning(f"{code}의 오늘 시작가(Open)가 0이므로 데이터 제외.")  # 경고 로그
                return

            # 최근 26일 중 시작가, 고가, 저가, 종가가 모두 동일한 날이 있는지 확인
            identical_prices = (recent_data['Open'] == recent_data['High']) & \
                               (recent_data['High'] == recent_data['Low']) & \
                               (recent_data['Low'] == recent_data['Close'])

            if identical_prices.any():  # 한 번이라도 동일한 날이 있는 경우
                logging.warning(f"{code}의 최근 26일 이내에 시작가, 고가, 저가, 종가가 동일한 날이 있으므로 데이터 제외.")  # 경고 로그
                return
            
            # 최근 26일의 거래량 확인
            recent_volume = df['Volume'].tail(26)

            if recent_volume.sum() > 0:  # 최근 26일 간 거래량이 0이 아닌 경우
                if all(recent_data['Close'] >= 3000):  # 최근 26일 종가가 3000 이상인지 확인
                    df.reset_index(inplace=True)  # 인덱스 초기화
                    df['Code'] = code  # 주식 코드 추가
                    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')  # 날짜 형식 변경
                    all_stocks_data[code] = df  # 가져온 데이터 저장

                    # 한 종목의 데이터 로그 기록
                    logging.info(f"{code} 데이터 가져오기 완료, 데이터 길이: {len(df)}")
                    logging.info(f"가져온 데이터:\n{df.head()}")  # 가져온 데이터의 첫 5행 로그

                    # 재무제표 데이터 가져오기
                    financials = fetch_financials(code)
                    if not financials.empty:  # 재무제표 데이터가 있는지 확인
                        logging.info(f"{code} 재무제표 데이터:\n{financials}")  # 재무제표 로그 기록
                    else:
                        logging.warning(f"{code}의 재무제표 데이터가 비어 있습니다.")  # 경고 로그

                else:
                    logging.warning(f"{code}의 최근 26일 종가가 3000 미만입니다. 데이터 제외.")  # 경고 로그
            else:
                logging.warning(f"{code}의 최근 26일 거래량이 0입니다. 데이터 제외.")  # 경고 로그
        else:
            logging.warning(f"{code} 데이터가 비어 있거나 가져오기 실패")  # 경고 로그
    except Exception as e:
        logging.error(f"{code} 데이터 가져오기 중 오류 발생: {e}")  # 오류 로그

def fetch_financials(ticker):
    """지정한 주식의 재무제표 데이터를 가져오는 함수."""
    stock = yf.Ticker(ticker)
    financials = stock.financials
    return financials

def fetch_stock_data(markets, start_date, end_date):
    """주식 데이터를 가져오는 메인 함수."""
    all_stocks_data = {}  # 모든 주식 데이터를 저장할 딕셔너리

    for market in markets:  # 여러 시장에 대해 반복
        codes = fdr.StockListing(market)['Code'].tolist()  # 주식 코드 리스트 가져오기
        threads = []  # 스레드 리스트 초기화

        for code in codes:
            # 주식 코드에 대해 새 스레드 생성
            thread = threading.Thread(target=fetch_single_stock_data, args=(code, start_date, end_date, all_stocks_data))
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
