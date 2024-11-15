import FinanceDataReader as fdr
import pandas as pd
import plotly.graph_objects as go

def get_kospi_kosdaq_tickers():
    # KOSPI와 KOSDAQ 종목 목록 가져오기
    kospi = fdr.StockListing('KOSPI')
    kosdaq = fdr.StockListing('KOSDAQ')
    
    return kospi['Code'].tolist(), kosdaq['Code'].tolist()

def fetch_stock_data(tickers):
    stock_data = {}
    for ticker in tickers:
        try:
            # 최근 5거래일 데이터 가져오기
            data = fdr.DataReader(ticker, '2023-11-01', '2023-11-15')  # 날짜는 필요에 따라 조정
            stock_data[ticker] = data
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    return stock_data

def plot_stock_data(stock_data):
    for ticker, data in stock_data.items():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines+markers', name='Close Price'))
        fig.update_layout(title=f'Stock Price for {ticker}', xaxis_title='Date', yaxis_title='Price')
        fig.show()

# KOSPI와 KOSDAQ 종목 리스트 가져오기
kospi_tickers, kosdaq_tickers = get_kospi_kosdaq_tickers()

# 전체 티커 리스트
tickers = kospi_tickers + kosdaq_tickers

# 종목 데이터 가져오기
stock_data = fetch_stock_data(tickers)

# 주가 데이터 시각화
plot_stock_data(stock_data)
