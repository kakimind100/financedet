import FinanceDataReader as fdr
import pandas as pd
import ta  # 기술적 지표 계산을 위한 라이브러리
import logging
from datetime import datetime

# 로깅 설정
logging.basicConfig(filename='stock_analysis.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_indicators(df):
    """MACD와 윌리엄스 %R을 계산하는 함수."""
    df['macd'] = ta.trend.MACD(df['Close']).macd()
    df['williams_r'] = ta.momentum.WilliamsR(df['High'], df['Low'], df['Close'], window=14)
    return df

def search_stocks(start_date):
    """주식 종목을 검색하는 함수."""
    kospi = fdr.StockListing('KOSPI')
    kosdaq = fdr.StockListing('KOSDAQ')
    
    # 종목 목록을 하나의 데이터프레임으로 합치기
    stocks = pd.concat([kospi, kosdaq])
    
    # 열 이름 확인
    print("열 이름:", stocks.columns)  # 열 이름 출력

    result = []

    for symbol in stocks['Symbol']:  # 'Symbol' 열이 실제로 존재하는지 확인
        try:
            df = fdr.DataReader(symbol, start=start_date)
            if len(df) < 10:
                continue

            recent_data = df.iloc[-10:]
            last_close = recent_data['Close'].iloc[-1]
            prev_close = recent_data['Close'].iloc[-2]

            if last_close >= prev_close * 1.3:
                df = calculate_indicators(df)
                
                if df['macd'].iloc[-1] <= 5 and df['williams_r'].iloc[-1] <= 0:
                    result.append({
                        'Symbol': symbol,
                        'Name': stocks[stocks['Symbol'] == symbol]['Name'].values[0],
                        'Last Close': last_close,
                        'MACD': df['macd'].iloc[-1],
                        'Williams %R': df['williams_r'].iloc[-1]
                    })
        except Exception as e:
            logging.error(f"{symbol} 처리 중 오류 발생: {e}")
            print(f"{symbol} 처리 중 오류 발생: {e}")
    
    return pd.DataFrame(result)

if __name__ == "__main__":
    start_date_input = input("분석 시작 날짜를 입력하세요 (YYYY-MM-DD): ")
    try:
        start_date = datetime.strptime(start_date_input, '%Y-%m-%d')
    except ValueError:
        print("날짜 형식이 올바르지 않습니다.")
        exit(1)

    result = search_stocks(start_date.strftime('%Y-%m-%d'))
    
    if not result.empty:
        print(result)
        result.to_csv('stock_analysis_results.csv', index=False)
        print("결과가 'stock_analysis_results.csv' 파일로 저장되었습니다.")
    else:
        print("조건에 맞는 종목이 없습니다.")
