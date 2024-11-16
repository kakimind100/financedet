import FinanceDataReader as fdr
import pandas as pd
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# 로그 디렉토리 생성
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# 로깅 설정
logging.basicConfig(
    filename=os.path.join(log_dir, 'stock_analysis.log'),
    level=logging.DEBUG,  # DEBUG 레벨로 모든 로그 기록
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 콘솔에도 로그 출력
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # 콘솔에는 INFO 레벨 이상의 로그만 출력
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)

def calculate_williams_r(df, window=14):
    """Williams %R을 직접 계산하는 함수."""
    logging.debug(f"Calculating Williams %R with window size: {window}")
    highest_high = df['High'].rolling(window=window).max()
    lowest_low = df['Low'].rolling(window=window).min()
    williams_r = -100 * (highest_high - df['Close']) / (highest_high - lowest_low)
    logging.debug(f"Williams %R 계산 완료: {williams_r.tail()}")
    return williams_r

def calculate_rsi(df, window=14):
    """RSI를 계산하는 함수."""
    logging.debug(f"Calculating RSI with window size: {window}")
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    logging.debug(f"RSI 계산 완료: {rsi.tail()}")
    return rsi

def calculate_macd(df):
    """MACD를 계산하는 함수."""
    logging.debug("Calculating MACD")
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    logging.debug(f"MACD 계산 완료: {macd.tail()}")
    return macd, signal

def calculate_cci(df, window=5):
    """CCI (Commodity Channel Index)를 계산하는 함수."""
    logging.debug(f"Calculating CCI with window size: {window}")
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    sma = typical_price.rolling(window=window).mean()
    mean_deviation = (typical_price - sma).abs().rolling(window=window).mean()
    cci = (typical_price - sma) / (0.015 * mean_deviation)
    logging.debug(f"CCI 계산 완료: {cci.tail()}")
    return cci

def analyze_stock(code, start_date):
    """주식 데이터를 분석하는 함수."""
    logging.info(f"{code} 데이터 분석 시작")
    try:
        df = fdr.DataReader(code, start=start_date)
        logging.info(f"{code} 데이터 가져오기 성공, 가져온 데이터 길이: {len(df)}")

        # 데이터 길이 체크: 최소 26일 데이터
        if len(df) < 26:
            logging.warning(f"{code} 데이터가 26일 미만으로 건너뜁니다.")
            return None

        # 최근 20일 데이터 추출
        recent_data = df.iloc[-20:]  # 최근 20일 데이터
        last_close = recent_data['Close'].iloc[-1]  # 최근 종가
        opening_price = recent_data['Open'].iloc[-1]  # 최근 시작가 (개장가)

        # 가격 상승 조건 체크 (장대 양봉)
        price_increase_condition = False

        for i in range(len(recent_data)):
            daily_low = recent_data['Low'].iloc[i]  # 당일 최저가
            daily_high = recent_data['High'].iloc[i]  # 당일 최고가
            daily_open = recent_data['Open'].iloc[i]  # 당일 시가
            daily_close = recent_data['Close'].iloc[i]  # 당일 종가

            # 장대 양봉 조건: 당일 최고가가 최저가의 1.29배 초과하고, 종가가 시가보다 높은지 확인
            if daily_high > daily_low * 1.29 and daily_close > daily_open:
                price_increase_condition = True
                logging.info(f"{code} - {recent_data.index[i].date()}일: 장대 양봉 발생, 최고가 {daily_high}가 최저가 {daily_low}의 29% 초과")

        # CCI 계산
        df['cci'] = calculate_cci(df, window=5)
        cci_current = df['cci'].iloc[-1]  # 현재 CCI 값
        cci_condition = cci_current < -90  # CCI 조건

        # Williams %R 계산
        df['williams_r'] = calculate_williams_r(df)
        williams_r = df['williams_r'].iloc[-1]

        # RSI 계산
        rsi = calculate_rsi(df)
        rsi_current = rsi.iloc[-1]  # 현재 RSI 값
        rsi_condition = rsi_current < 50  # RSI 조건

        # 현재 RSI와 CCI 값을 로그에 기록
        logging.info(f"{code} - 현재 RSI: {rsi_current}, 현재 CCI: {cci_current}")

        # MACD 계산
        macd, signal = calculate_macd(df)
        macd_condition = macd.iloc[-1] <= 5  # MACD가 5 이하일 경우

        # 지지선 확인: 마지막 날의 종가가 최근 저점의 1% 초과인지 확인
        overall_low = df['Low'].min()  # 전체 기간의 저점
        support_condition = last_close > overall_low * 1.02  # 최근 종가가 전체 저점의 2% 초과
        
        # 조건 확인: 가격 상승 조건, CCI, Williams %R, RSI, MACD, 지지선 확인
        if (price_increase_condition and 
            williams_r <= -90 and 
            rsi_condition and 
            cci_condition and 
            support_condition and 
            macd_condition):  # MACD 조건
            result = {
                'Code': code,
                'Last Close': last_close,
                'Opening Price': opening_price,
                'Lowest Price': overall_low,
                'Highest Price': recent_data['High'].max(),
                'Williams %R': williams_r,
                'CCI': cci_current,  # 현재 CCI 값 추가
                'Support Condition': support_condition  # 지지선 조건 추가
            }
            logging.info(f"{code} 조건 만족: {result}")
            return result
        else:
            logging.info(f"{code} 조건 불만족: "
                         f"가격 상승 조건: {price_increase_condition}, "
                         f"Williams %R: {williams_r}, "
                         f"RSI: {rsi_current}, "
                         f"CCI: {cci_current}, "
                         f"MACD: {macd_condition}, "
                         f"지지선 확인: {support_condition}")

    except Exception as e:
        logging.error(f"{code} 처리 중 오류 발생: {e}")
        return None

def search_stocks(start_date):
    """주식 종목을 검색하는 함수."""
    logging.info("주식 검색 시작")
    
    try:
        kospi = fdr.StockListing('KOSPI')  # KRX 코스피 종목 목록
        logging.info("코스피 종목 목록 가져오기 성공")
        
        kosdaq = fdr.StockListing('KOSDAQ')  # KRX 코스닥 종목 목록
        logging.info("코스닥 종목 목록 가져오기 성공")
    except Exception as e:
        logging.error(f"종목 목록 가져오기 중 오류 발생: {e}")
        return []

    stocks = pd.concat([kospi, kosdaq])
    result = []

    # 멀티스레딩으로 주식 데이터 처리
    with ThreadPoolExecutor(max_workers=10) as executor:  # 최대 10개의 스레드 사용
        futures = {executor.submit(analyze_stock, code, start_date): code for code in stocks['Code']}
        for future in as_completed(futures):
            result_data = future.result()
            if result_data:
                result.append(result_data)

    logging.info("주식 검색 완료")
    return result

if __name__ == "__main__":
    logging.info("스크립트 실행 시작")
    
    # 최근 40 거래일을 기준으로 시작 날짜 설정
    today = datetime.today()
    start_date = today - timedelta(days=40)  # 최근 40 거래일 전 날짜
    start_date_str = start_date.strftime('%Y-%m-%d')

    logging.info(f"주식 분석 시작 날짜: {start_date_str}")

    result = search_stocks(start_date_str)
   
