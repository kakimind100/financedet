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
    return williams_r

def calculate_moving_average(df, window):
    """이동 평균을 계산하는 함수."""
    return df['Close'].rolling(window=window).mean()

def calculate_macd(df):
    """MACD를 계산하는 함수."""
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def calculate_rsi(df, window=14):
    """RSI를 계산하는 함수."""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

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
        recent_low = recent_data['Low'].min()  # 최근 저점

        # 29% 이상 상승 여부 확인
        high_condition = False
        volume_increase = False

        # 최근 20일 내에서 전일 대비 29% 상승한 날 찾기
        for i in range(1, len(recent_data)):
            if recent_data['Close'].iloc[i] >= recent_data['Close'].iloc[i - 1] * 1.29:
                high_condition = True
                # 해당 날의 거래량과 전날 거래량 확인
                current_volume = recent_data['Volume'].iloc[i]  # 29% 상승한 날의 거래량
                previous_volume = recent_data['Volume'].iloc[i - 1]  # 전날 거래량
                volume_increase = current_volume >= previous_volume * 2  # 전날 대비 200% 이상 증가 여부
                logging.info(f"{code} 거래량 증가: 전날 대비 {current_volume / previous_volume * 100 - 100:.2f}%")
                break  # 조건 만족 시 루프 종료

        # 장대 양봉 여부 체크
        is_bullish_engulfing = False
        if high_condition:
            last_candle = recent_data.iloc[-1]
            is_bullish_engulfing = (last_candle['Close'] > last_candle['Open']) and \
                                   (last_candle['Low'] < last_candle['Close'])

        # Williams %R 계산
        df['williams_r'] = calculate_williams_r(df)
        williams_r = df['williams_r'].iloc[-1]

        # RSI 계산
        rsi = calculate_rsi(df)
        rsi_condition = rsi.iloc[-1] < 30  # 최근 RSI가 30 이하

        # 이동 평균선 계산
        short_ma = calculate_moving_average(df, window=5).iloc[-1]
        long_ma = calculate_moving_average(df, window=20).iloc[-1]
        ma_condition = short_ma > long_ma * 1.01  # 단기 이동 평균이 장기 이동 평균보다 1% 이상 높아야 함

        # MACD 계산
        macd, signal = calculate_macd(df)
        macd_condition = macd.iloc[-1] <= 5  # MACD가 5 이하일 경우

        # 지지선 확인: 마지막 날의 종가가 최근 저점 이하인지 확인
        support_condition = last_close >= recent_low  # 최근 저점 이하로 내려가지 않았으면 True

        # 조건 확인: 이동 평균선과 MACD 중 하나만 True, 지지선 확인은 True여야 함
        if (high_condition and 
            williams_r <= -80 and 
            rsi_condition and 
            volume_increase and 
            support_condition and 
            (ma_condition or macd_condition)):  # 이동 평균선 또는 MACD 조건
            result = {
                'Code': code,
                'Last Close': last_close,
                'Williams %R': williams_r,
                'Bullish Engulfing': is_bullish_engulfing
            }
            logging.info(f"{code} 조건 만족: {result}")
            return result
        else:
            logging.info(f"{code} 조건 불만족: "
                         f"29% 상승: {high_condition}, "
                         f"Williams %R: {williams_r}, "
                         f"거래량 증가: {volume_increase}, "
                         f"이동 평균선: {ma_condition}, "
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
    
    if result:
        for item in result:
            print(item)  # 콘솔에 결과 출력
    else:
        print("조건에 맞는 종목
