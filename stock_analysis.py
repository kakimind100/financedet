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

# 콘솔에도 로그 출력
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # 콘솔에는 INFO 레벨 이상의 로그만 출력
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)

def calculate_indicators(df):
    """MACD와 윌리엄스 %R을 계산하는 함수."""
    df['macd'] = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9).macd()
    df['williams_r'] = ta.momentum.WilliamsR(df['High'], df['Low'], df['Close'], window=14)
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

        # 최근 20일 중 29% 이상 상승한 종가 확인
        if any(recent_data['Close'].iloc[i] >= recent_data['Close'].iloc[i-1] * 1.29 for i in range(1, len(recent_data))):
            logging.info(f"{code} 최근 20일 내 29% 이상 상승한 종목 발견: 최근 종가 {last_close}, 이전 종가 {prev_close}")
            
            # 장대 양봉 조건 확인: 다음날이 전일보다 29% 이상 상승
            for i in range(1, len(recent_data)):
                if recent_data['Close'].iloc[i] >= recent_data['Close'].iloc[i-1] * 1.29:
                    logging.info(f"{code} 장대 양봉 조건 만족: {recent_data.index[i]} 종가 {recent_data['Close'].iloc[i]}, 이전 종가 {recent_data['Close'].iloc[i-1]}")
                    df = calculate_indicators(df)  # MACD와 윌리엄스 %R 계산
                    
                    # MACD와 윌리엄스 %R 조건 확인
                    if df['macd'].iloc[-1] <= 5 and df['williams_r'].iloc[-1] <= 0:
                        result = {
                            'Code': code,
                            'Last Close': df['Close'].iloc[i],
                            'MACD': df['macd'].iloc[-1],
                            'Williams %R': df['williams_r'].iloc[-1]
                        }
                        logging.info(f"{code} 조건 만족: {result}")
                        return result
                    else:
                        logging.info(f"{code} 조건 불만족: MACD={df['macd'].iloc[-1]}, Williams %R={df['williams_r'].iloc[-1]}")
                    break  # 조건을 만족한 경우, 반복문 종료
            else:
                logging.info(f"{code} 장대 양봉 조건 불만족: 20일 기간 내에서 상승 조건 없음")
        
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
                result.append(result_data)

    logging.info("주식 검색 완료")
    return result

if __name__ == "__main__":
    logging.info("스크립트 실행 시작")
    
    # 최근 40 거래일을 기준으로 시작 날짜 설정
    today = datetime.today()
    start_date = today - timedelta(days=40)  # 최근 40 거래일 전 날짜
    start_date_str = start_date.strftime('%Y-%m-%d')

    logging.info(f"주식 분석 시작 날짜: {start_date_str}")

    result = search_stocks(start_date_str)
    
    if result:
        for item in result:
            print(item) 
