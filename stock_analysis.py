import pandas as pd
import yfinance as yf
import pandas_ta as ta
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# CSV 파일 경로
csv_file_path = 'path/to/your/stock_list.csv'  # CSV 파일의 경로를 수정하세요.

# CSV 파일에서 종목 코드 가져오기
def get_stock_codes_from_csv():
    try:
        df = pd.read_csv(csv_file_path)
        return df['종목코드'].tolist()  # '종목코드' 컬럼 이름에 맞게 변경
    except Exception as e:
        logging.error(f"Error reading stock codes from CSV: {e}")
        return []

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

    if not selected_stocks:
        logging.info("No stocks met the selection criteria.")
    else:
        logging.info(f"Selected stocks: {selected_stocks}")

    return selected_stocks

# 종목 코드 가져오기
stocks = get_stock_codes_from_csv()

# 조건을 만족하는 주식 찾기
selected_stocks = analyze_stocks(stocks)

# 결과 출력
print("조건을 만족하는 주식:", selected_stocks)
