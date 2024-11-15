import FinanceDataReader as fdr
import yfinance as yf
import pandas as pd

# 한국 주식 종목 리스트 가져오기 (KOSPI와 KOSDAQ)
kospi = fdr.StockListing('KOSPI')
kosdaq = fdr.StockListing('KOSDAQ')

# KOSPI와 KOSDAQ의 종목 코드를 합칩니다.
tickers = kospi['Code'].tolist() + kosdaq['Code'].tolist()

# 결과를 저장할 리스트
final_results = []

# 각 종목에 대해 데이터 가져오기
for ticker in tickers:
    try:
        # 최근 5거래일 데이터 가져오기
        data = yf.download(ticker + '.KS', period='5d')  # KOSPI 종목에 '.KS' 추가
        # KOSDAQ 종목에 대해서도 '.KQ'를 추가해야 합니다.
        if ticker in kosdaq['Code'].tolist():
            data = yf.download(ticker + '.KQ', period='5d')  # KOSDAQ 종목에 '.KQ' 추가

        # 상한가 계산 (예시로 5% 상승)
        upper_limit = data['Close'].shift(1) * 1.05

        # 장대 양봉 조건 확인
        data['Is Upper Limit'] = data['Close'] >= upper_limit
        data['Is Large Bullish'] = (data['Close'] > data['Open']) & ((data['Close'] - data['Open']) > (data['High'] - data['Low']) * 0.5)

        # MACD 계산
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()  # 시그널 라인을 9로 설정

        # 전저가 확인
        previous_low = data['Low'].min()  # 최근 5일 중 최저가

        # William's R 계산
        high14 = data['High'].rolling(window=14).max()
        low14 = data['Low'].rolling(window=14).min()
        data['Williams R'] = -100 * (high14 - data['Close']) / (high14 - low14)

        # 조건에 맞는 데이터 필터링
        if not data[data['Is Upper Limit'] & data['Is Large Bullish']].empty:
            # MACD가 5 이하인 조건
            if (data['MACD'] <= 5).any():
                # 가격 하락 조건 확인 (종가가 이전 거래일 종가보다 낮은 경우)
                price_decline = (data['Close'].diff() < 0).any()
                
                # 전저가 이하로 내려가지 않은 경우
                if (data['Close'].min() >= previous_low):
                    # William's R이 0 이하인 경우
                    if (data['Williams R'] <= 0).any():
                        final_results.append({'Ticker': ticker})

    except Exception as e:
        print(f"Error processing {ticker}: {e}")

# 결과 출력
final_results_df = pd.DataFrame(final_results)
print(final_results_df)
