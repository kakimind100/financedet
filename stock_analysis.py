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
    filename=os.path.join(log_dir, 'stock_analysis.log'),  # logs/stock_analysis.log에 저장
    level=logging.DEBUG,  # DEBUG 레벨로 로그 기록
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 콘솔에도 로그 출력
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)

def calculate_indicators(df):
    """MACD와 윌리엄스 %R을 계산하는 함수."""
    df['macd'] = ta.trend.MACD(df['Close']).macd()
    df['williams_r'] = ta.momentum.WilliamsR(df['High'], df['Low'], df['Close'], window=14)
    return df

def process_stock(code, start_date):
    """주식 데이터를 처리하는 함수."""
    logging.info(f"{code} 처리 시작")
    try:
        df = fdr.DataReader(code, start=start_date)
        logging.info(f"{code} 데이터 가져오기 성공, 가져온 데이터 길이: {len(df)}")  # 데이터 로드 성공 로깅
        
        if len(df) < 10:
            logging.warning(f"{code} 데이터가 10일 미만으로 건너뜁니다.")
            return None  # 데이터가 충분하지 않으면 None 반환
        
        recent_data = df.iloc[-10:]  # 최근 10일 데이터
        last_close = recent_data['Close'].iloc[-1]  # 최근 종가
        prev_close = recent_data['Close'].iloc[-2]  # 이전 종가

        # 장대 양봉 조건 확인
        if last_close >= prev_close * 1.3:  # 최근 종가가 이전 종가보다 30% 이상 상승
            df = calculate_indicators(df)  # MACD와 윌리엄스 %R 계산
            
            # MACD와 윌리엄스 %R 조건 완화
            if df['macd'].iloc[-1] <= 10 and df['williams_r'].iloc[-1] <= -20:
                return {
                    'Code': code,
                    'Last Close': last_close,
                    'MACD': df['macd'].iloc[-1],
                    'Williams %R': df['williams_r'].iloc[-1]
                }
        
        logging.info(f"{code} 조건 불만족")
        return None  # 조건을 만족하지 않으면 None 반환
    except Exception as e:
        logging.error(f"{code} 처리 중 오류 발생: {e}")
        return None  # 오류 발생 시 None 반환

def search_stocks(start_date):
    """주식 종목을 검색하는 함수."""
    logging.info("주식 검색 시작")
    
    try:
        kospi = fdr.StockListing('KOSPI')  # 코스피 종목 목록
        logging.info("코스피 종목 목록 가져오기 성공")
        
        kosdaq = fdr.StockListing('KOSDAQ')  # 코스닥 종목 목록
        logging.info("코스닥 종목 목록 가져오기 성공")
    except Exception as e:
        logging.error(f"종목 목록 가져오기 중 오류 발생: {e}")
        return pd.DataFrame()

    stocks = pd.concat([kospi, kosdaq])
    
    # 열 이름 확인
    column_names = stocks.columns.tolist()
    logging.info(f"종목 목록 열 이름: {column_names}")  # 열 이름 로깅

    result = []

    # 'Code' 열을 사용하도록 수정
    if 'Code' not in column_names:
        logging.error("'Code' 열이 존재하지 않습니다. 사용할 수 있는 열: {}".format(column_names))
        print("Error: 'Code' 열이 존재하지 않습니다.")
        return pd.DataFrame()

    # 멀티스레딩으로 주식 데이터 처리
    with ThreadPoolExecutor(max_workers=10) as executor:  # 최대 10개의 스레드 사용
        futures = {executor.submit(process_stock, code, start_date): code for code in stocks['Code']}
        for future in as_completed(futures):
            result_data = future.result()
            if result_data:
                result.append(result_data)
                # 5개 종목 찾으면 종료
                if len(result) >= 5:
                    logging.info("5개 종목을 찾았습니다. 분석 종료.")
                    return pd.DataFrame(result)

    logging.info("주식 검색 완료")
    return pd.DataFrame(result)

if __name__ == "__main__":
    logging.info("스크립트 실행 시작")
    
    # 최근 10 거래일을 기준으로 시작 날짜 설정
    today = datetime.today()
    start_date = today - timedelta(days=10)  # 최근 10 거래일 전 날짜
    start_date_str = start_date.strftime('%Y-%m-%d')

    logging.info(f"주식 분석 시작 날짜: {start_date_str}")

    result = search_stocks(start_date_str)
    
    if not result.empty:
        print(result)
        result.to_csv('stock_analysis_results.csv', index=False)
        logging.info("결과가 'stock_analysis_results.csv' 파일로 저장되었습니다.")
    else:
        print("조건에 맞는 종목이 없습니다.")
        logging.info("조건에 맞는 종목이 없습니다.")

    logging.info("스크립트 실행 완료")
