import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta

# KRX 종목 코드 크롤링 (이전 코드와 동일)
def get_stock_codes():
    url = 'http://kind.krx.co.kr/corpgeneral/corpList.do'
    res = requests.get(url)
    soup = BeautifulSoup(res.text, 'html.parser')

    stock_table = soup.find('table', {'class': 'type_1'})
    rows = stock_table.find_all('tr')[1:]

    stock_codes = []
    for row in rows:
        cols = row.find_all('td')
        if len(cols) > 1:
            code = cols[0].text.strip()
            stock_codes.append(code)

    return stock_codes

# 주식 데이터 분석
def analyze_stocks(stocks):
    selected_stocks = []
    
    for stock in stocks:
        data = yf.download(stock + '.KS', period='30d')

        # 10일 이내 장대양봉 발생 여부 확인
        data['Body'] = data['Close'] - data['Open']
        data['Range'] = data['High'] - data['Low']
        data['Large_Candle'] = (data['Body'].abs() / data['Range']) > 0.3

        if data['Large_Candle'].iloc[-10:].any():
            # MACD 및 윌리엄스 %R 계산
            data['MACD'] = ta.macd(data['Close'])['macd']
            data['Williams'] = ta.williams(data['High'], data['Low'], data['Close'])

            # 조건 체크
            macd_condition = data['MACD'].iloc[-1] < 5
            williams_condition = data['Williams'].iloc[-1] < 0
            previous_low = data['Close'].iloc[-11]
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
