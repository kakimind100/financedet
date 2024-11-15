import FinanceDataReader as fdr
import pandas as pd
import ta  # 기술적 지표 계산을 위한 라이브러리
import logging
from datetime import datetime, timedelta

# 로깅 설정
logging.basicConfig(
    filename='stock_analysis.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def calculate_indicators(df):
    """MACD와 윌리엄스 %R을 계산하는 함수."""
    df['macd'] = ta.trend.MACD(df['Close']).macd()
    df['williams_r'] = ta.momentum.WilliamsR(df['High'], df['Low'], df['Close'], window=14)
    return df

def search_stocks(start_date):
    """주식 종목을 검색하는 함수."""
    logging.info("주식 검색 시작")
    kospi = fdr.StockListing('KOSPI')
    kosdaq = fdr.StockListing('KOSDAQ')
    
    stocks = pd.concat([kospi, kosdaq])
    
    # 열 이름 확인
    logging.info(f"종목 목록 열 이름: {stocks.columns.tolist()}")  # 열 이름 로깅

    result = []

    for symbol in stocks['Symbol']:
        logging.info(f"{symbol} 처리 시작")
        try:
            df = fdr.DataReader(symbol, start=start_date)
            if len(df) < 10:
                logging.warning(f"{symbol} 데이터가 10일 미만으로 건너뜁니다.")
                continue
            
            recent_data = df.iloc[-10:]  # 최근 10일 데이터
            last_close = recent_data['Close'].iloc[-1]  # 최근 종가
            prev_close = recent_data['Close'].iloc[-2]  # 이전 종가

            # 장대 양봉 조건 확인
            if last_close >= prev_close * 1.3:  # 최근 종가가 이전 종가보다 30% 이상 상승
                df = calculate_indicators(df)  # MACD와 윌리엄스 %R 계산
                
                # MACD와 윌리엄스 %R 조건 확인
                if df['macd'].iloc[-1] <= 5 and df['williams_r'].iloc[-1] <= 0:
                    logging.info(f"{symbol} 조건 만족: Last Close={last_close}, MACD={df['macd'].iloc[-1]}, Williams %R={df['williams_r'].iloc[-1]}")
                    result.append({
                        'Symbol': symbol,
                        'Name': stocks[stocks['Symbol'] == symbol]['Name'].values[0],
                        'Last Close': last_close,
                        'MACD': df['macd'].iloc[-1],
                        'Williams %R': df['williams_r'].iloc[-1]
                    })
                    continue
            
            logging.info(f"{symbol} 조건 불만족")
        except Exception as e:
            logging.error(f"{symbol} 처리 중 오류 발생: {e}")

    logging.info("주식 검색 완료")
    return pd.DataFrame(result)

if __name__ == "__main__":
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
