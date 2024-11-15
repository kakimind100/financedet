import FinanceDataReader as fdr
import pandas as pd
import ta  # 기술적 지표 계산을 위한 라이브러리

def calculate_indicators(df):
    """MACD와 윌리엄스 %R을 계산하는 함수."""
    # MACD 계산
    df['macd'] = ta.trend.MACD(df['Close']).macd()
    
    # 윌리엄스 %R 계산
    df['williams_r'] = ta.momentum.WilliamsR(df['High'], df['Low'], df['Close'], window=14)

    return df

def search_stocks():
    """주식 종목을 검색하는 함수."""
    # 코스피와 코스닥 종목 목록 가져오기
    kospi = fdr.StockListing('KOSPI')
    kosdaq = fdr.StockListing('KOSDAQ')
    
    # 종목 목록을 하나의 데이터프레임으로 합치기
    stocks = pd.concat([kospi, kosdaq])
    result = []

    for symbol in stocks['Symbol']:
        try:
            # 최근 10일 데이터 가져오기
            df = fdr.DataReader(symbol, start='2023-11-01')
            if len(df) < 10:
                continue  # 데이터가 10일 미만인 경우 건너뛰기

            # 장대 양봉 조건 확인
            recent_data = df.iloc[-10:]
            last_close = recent_data['Close'].iloc[-1]
            prev_close = recent_data['Close'].iloc[-2]

            if last_close >= prev_close * 1.3:  # 장대 양봉 조건
                # MACD와 윌리엄스 %R 계산
                df = calculate_indicators(df)
                
                # MACD와 윌리엄스 %R 조건 확인
                if df['macd'].iloc[-1] <= 5 and df['williams_r'].iloc[-1] <= 0:
                    result.append({
                        'Symbol': symbol,
                        'Name': stocks[stocks['Symbol'] == symbol]['Name'].values[0],
                        'Last Close': last_close,
                        'MACD': df['macd'].iloc[-1],
                        'Williams %R': df['williams_r'].iloc[-1]
                    })
        except Exception as e:
            print(f"{symbol} 처리 중 오류 발생: {e}")
    
    return pd.DataFrame(result)

if __name__ == "__main__":
    result = search_stocks()
    
    if not result.empty:
        print(result)
    else:
        print("조건에 맞는 종목이 없습니다.")
