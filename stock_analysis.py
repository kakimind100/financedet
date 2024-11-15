import os
import openai
import requests
import pandas as pd

# OpenAI API 키 설정
openai.api_key = os.getenv('OPENAI_API_KEY')  # 환경 변수에서 API 키 가져오기

# KOSPI와 KOSDAQ 종목 리스트 가져오기
def get_stock_list():
    # KOSPI 종목 가져오기
    kospi_url = 'https://kind.krx.co.kr/corpgeneral/corpList.do'
    kospi_response = requests.get(kospi_url)
    kospi_data = pd.read_html(kospi_response.text)[0]
    kospi_tickers = kospi_data['종목코드'].tolist()
    kospi_names = kospi_data['회사명'].tolist()
    
    # KOSDAQ 종목 가져오기
    kosdaq_url = 'https://kind.krx.co.kr/corpgeneral/corpList.do'
    kosdaq_response = requests.get(kosdaq_url)
    kosdaq_data = pd.read_html(kosdaq_response.text)[0]
    kosdaq_tickers = kosdaq_data['종목코드'].tolist()
    kosdaq_names = kosdaq_data['회사명'].tolist()
    
    return kospi_tickers + kosdaq_tickers, kospi_names + kosdaq_names

# OpenAI API에 종목 조건 전달
def find_matching_stocks(tickers, names):
    results = []
    for ticker, name in zip(tickers, names):
        # OpenAI API를 통해 조건 확인
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": f"Does the stock {name} (ticker: {ticker}) meet the following conditions? "
                               f"1. There should be a large bullish candle near the upper limit (30%). "
                               f"2. MACD should drop below 5 after a large bullish candle. "
                               f"3. William's R should be below 0. "
                               f"4. The stock price should be above the previous low before the upper limit."
                }
            ]
        )
        
        # 조건 충족 여부 확인
        if response['choices'][0]['message']['content'].lower() == 'yes':
            results.append(name)

    return results

# 메인 실행 부분
if __name__ == "__main__":
    tickers, names = get_stock_list()
    matching_stocks = find_matching_stocks(tickers, names)
    
    # 결과 출력
    print("Matching stocks:", matching_stocks)
