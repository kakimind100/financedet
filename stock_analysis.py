import pandas as pd
import yfinance as yf
import asyncio

async def fetch_stock_data(ticker):
    """주식 데이터를 비동기적으로 가져오는 함수"""
    data = yf.download(ticker, period='1d', interval='1m', progress=False)
    return data

def check_conditions(stock_data):
    """조건을 확인하는 함수"""
    if stock_data.empty:
        return False

    last_row = stock_data.iloc[-1]

    # MACD 계산
    stock_data['EMA12'] = stock_data['Close'].ewm(span=12, adjust=False).mean()
    stock_data['EMA26'] = stock_data['Close'].ewm(span=26, adjust=False).mean()
    stock_data['MACD'] = stock_data['EMA12'] - stock_data['EMA26']
    
    # William's R 계산
    high = stock_data['High'].rolling(window=14).max()
    low = stock_data['Low'].rolling(window=14).min()
    stock_data['WilliamsR'] = (high - last_row['Close']) / (high - low) * -100

    # 조건 확인
    macd_condition = stock_data['MACD'].iloc[-1] < 5
    williams_r_condition = stock_data['WilliamsR'].iloc[-1] < 0
    price_condition = last_row['Close'] > low.iloc[-2]  # 이전 저점보다 높은지 확인

    return macd_condition and williams_r_condition and price_condition

async def find_matching_stocks(tickers):
    """주어진 티커 리스트에서 조건을 충족하는 종목 찾기"""
    matching_stocks = []

    tasks = [fetch_stock_data(ticker) for ticker in tickers]
    results = await asyncio.gather(*tasks)

    for ticker, stock_data in zip(tickers, results):
        if check_conditions(stock_data):
            matching_stocks.append(ticker)

    return matching_stocks

if __name__ == "__main__":
    # 종목 리스트를 CSV 파일에서 읽어오기
    tickers_df = pd.read_csv('tickers.csv')  # 종목 리스트 파일
    tickers = tickers_df['Ticker'].tolist()

    matching_stocks = asyncio.run(find_matching_stocks(tickers))

    # 결과 출력
    print("Stocks meeting the conditions:")
    print(matching_stocks)
