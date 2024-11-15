import requests
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd

# KRX에서 KOSPI 및 KOSDAQ 종목 코드 가져오기
def get_stock_codes():
    url = 'http://kind.krx.co.kr/corpgeneral/corpList.do'
    res = requests.get(url)
    
    if res.status_code != 200:
        print(f"Error fetching stock codes: {res.status_code}")
        return [], []

    soup = BeautifulSoup(res.text, 'html.parser')
    stock_table = soup.find('table', {'class': 'type_1'})
    
    if stock_table is None:
        print("Error: Stock table not found.")
        return [], []

    rows = stock_table.find_all('tr')[1:]  # 첫 번째 행은 헤더이므로 제외
    kospi_codes = []
    kosdaq_codes = []
    
    for row in rows:
        cols = row.find_all('td')
        if len(cols) > 1:
            code = cols[0].text.strip()  # 종목 코드
            market_type = cols[1].text.strip()  # 시장 구분 (KOSPI 또는 KOSDAQ)
            if market_type == 'KOSPI':
                kospi_codes.append(code)
            elif market_type == 'KOSDAQ':
                kosdaq_codes.append(code)

    return kospi_codes, kosdaq_codes

def fetch_stock_data(stock_code):
    """주식 데이터를 가져오는 함수"""
    try:
        data = yf.download(stock_code + '.KS', period='1y')  # KOSPI 종목 코드에 '.KS' 추가
        return data
    except Exception as e:
        print(f"Error fetching data for {stock_code}: {e}")
        return None

def analyze_stock(data):
    """주식 데이터를 분석하고 조건을 만족하는지 확인하는 함수"""
    data['SMA_50'] = data['Close'].rolling(window=50).mean()  # 50일 이동 평균
    data['SMA_200'] = data['Close'].rolling(window=200).mean()  # 200일 이동 평균

    # MACD 계산
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # 조건: SMA 50이 SMA 200을 상회하고 MACD가 0 이상일 때
    if data['SMA_50'].iloc[-1] > data['SMA_200'].iloc[-1] and data['MACD'].iloc[-1] > 0:
        return True
    return False

# KRX에서 KOSPI 및 KOSDAQ 종목 코드 가져오기
kospi_codes, kosdaq_codes = get_stock_codes()

# 조건을 만족하는 종목을 찾기
selected_stocks = []

for stock_code in kospi_codes + kosdaq_codes:  # KOSPI와 KOSDAQ 종목 모두 포함
    print(f"Fetching data for {stock_code}...")
    stock_data = fetch_stock_data(stock_code)
    if stock_data is not None and analyze_stock(stock_data):
        selected_stocks.append(stock_code)

# 결과 출력
if selected_stocks:
    print("조건을 만족하는 종목:", selected_stocks)
else:
    print("조건을 만족하는 종목이 없습니다.")
