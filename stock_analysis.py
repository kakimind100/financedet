import requests
from bs4 import BeautifulSoup
import pandas as pd
import yfinance as yf
import pandas_ta as ta

# KRX에서 코스피 및 코스닥 종목 코드 가져오기
def get_stock_codes():
    url = 'http://kind.krx.co.kr/corpgeneral/corpList.do'
    res = requests.get(url)
    
    if res.status_code != 200:
        print(f"Error fetching stock codes: {res.status_code}")
        return []

    soup = BeautifulSoup(res.text, 'html.parser')
    stock_table = soup.find('table', {'class': 'type_1'})
    
    # stock_table이 None인지 체크
    if stock_table is None:
        print("Error: Stock table not found.")
        return []

    rows = stock_table.find_all('tr')[1:]  # 첫 번째 행은 헤더이므로 제외
    stock_codes = []
    
    for row in rows:
        cols = row.find_all('td')
        if len(cols) > 1:
            code = cols[0].text.strip()  # 종목 코드
            stock_codes.append(code)

    return stock_codes

# 주식 데이터 분석 함수
def analyze_stocks(stocks):
    selected_stocks = []
    
    for stock in stocks:
        # Yahoo Finance에서 데이터 가져오기 (KOSPI 종목은 '.KS' 추가)
        data = yf.download(stock + '.KS', period='30d')

        # 30% 양봉 확인
        data['Body'] = data['Close'] - data['Open']
        data['Range'] = data['High'] - data['Low']
        data['Large_Candle'] = (data['Body'].abs() / data['Range']) > 0.3  # 30% 이상
        data['Large_Candle'] = data['Large_Candle'] & (data['Body'] > 0)  # 양봉인지 확인

        # 최근 10일간 장대양봉 발생 여부
        recent_large_candles = data['Large_Candle'].iloc[-10:].any()

        if recent_large_candles:
            # MACD 및 윌리엄스 %R 계산
            data['MACD'] = ta.macd(data['Close'])['macd']
            data['Williams'] = ta.williams(data['High'], data['Low'], data['Close'])

            # 조건 체크
            macd_condition = data['MACD'].iloc[-1] < 5
            williams_condition = data['Williams'].iloc[-1] < 0
            previous_low = data['Close'].min()  # 전저점
            current_close = data['Close'].iloc[-1]

            if macd_condition and williams_condition and current_close >= previous_low:
                selected_stocks.append(stock)

    return selected_stocks

# 종목 코드 가져오기
stocks = get_stock_codes()

# 조건을 만족하는 주식 찾기
selected_stocks = analyze_stocks(stocks)

# 결과 출력
print("조건을 만족하는 주식:", selected_stocks)
