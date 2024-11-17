import FinanceDataReader as fdr
import pandas as pd
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json

# JSON 파일로 결과를 저장하는 함수
def save_results_to_json(data, filename='results.json'):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

# 로그 디렉토리 생성
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# 로깅 설정
logging.basicConfig(
    filename=os.path.join(log_dir, 'stock_analysis.log'),
    level=logging.INFO,  # INFO 레벨로 로그 기록
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 콘솔에도 로그 출력
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # 콘솔에서도 INFO 레벨 이상의 로그 출력
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

def calculate_obv(df):
    """OBV (On-Balance Volume)를 계산하는 함수."""
    logging.debug("Calculating OBV")
    obv = [0]  # 첫 번째 OBV 값은 0으로 초기화
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i - 1]:  # 종가가 이전 종가보다 높을 때
            obv.append(obv[-1] + df['Volume'].iloc[i])  # 현재 거래량을 추가
        elif df['Close'].iloc[i] < df['Close'].iloc[i - 1]:  # 종가가 이전 종가보다 낮을 때
            obv.append(obv[-1] - df['Volume'].iloc[i])  # 현재 거래량을 빼기
        else:
            obv.append(obv[-1])  # 종가가 변하지 않으면 이전 OBV 유지
    logging.debug(f"OBV 계산 완료: {obv[-5:]}")  # 마지막 5개 OBV 값 로그
    return pd.Series(obv, index=df.index)

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
        bullish_candle_index = None  # 장대 양봉 발생 인덱스 초기화

        for i in range(len(recent_data)):
            daily_low = recent_data['Low'].iloc[i]  # 당일 최저가
            daily_high = recent_data['High'].iloc[i]  # 당일 최고가
            daily_open = recent_data['Open'].iloc[i]  # 당일 시가
            daily_close = recent_data['Close'].iloc[i]  # 당일 종가

            # 장대 양봉 조건: 당일 최고가가 최저가의 1.29배 초과하고, 종가가 시가보다 높은지 확인
            if daily_high > daily_low * 1.29 and daily_close > daily_open:
                price_increase_condition = True
                bullish_candle_index = i  # 장대 양봉 발생 인덱스 저장
                logging.info(f"{code} - {recent_data.index[i].date()}일: 장대 양봉 발생, 최고가 {daily_high}가 최저가 {daily_low}의 29% 초과")

        # OBV 계산
        df['obv'] = calculate_obv(df)
        obv_current = df['obv'].iloc[-1]  # 현재 OBV 값
        previous_obv = df['obv'].iloc[-2] if len(df['obv']) > 1 else 0  # 이전 OBV 값

        # OBV 세력 판단 조건
        obv_strength_condition = obv_current > previous_obv  # 현재 OBV가 이전 OBV보다 클 때

        # Williams %R 계산
        df['williams_r'] = calculate_williams_r(df)
        williams_r = df['williams_r'].iloc[-1]

        # RSI 계산
        rsi = calculate_rsi(df)
        rsi_current = rsi.iloc[-1]  # 현재 RSI 값
        rsi_condition = rsi_current < 51  # RSI 조건

        # 현재 RSI와 OBV 값을 로그에 기록
        logging.info(f"{code} - 현재 RSI: {rsi_current}, 현재 OBV: {obv_current}")

        # MACD 계산
        macd, signal = calculate_macd(df)
        macd_condition = macd.iloc[-1] <= 5  # MACD가 5 이하일 경우

        # 장대 양봉 이후의 데이터에서 저점 계산
        if bullish_candle_index is not None:
            filtered_data = recent_data.iloc[:bullish_candle_index]  # 장대 양봉 이전 데이터만 필터링
        else:
            filtered_data = recent_data  # 장대 양봉이 없으면 전체 데이터 사용

        # 전체 기간의 저점 계산 (당일 포한 2일제외)
        overall_low = df.iloc[:-2]['Low'].min()  # 전체 기간에서 마지막 행(당일) 제외하고 저점 계산
        if len(filtered_data) > 0:
            overall_low = min(overall_low, filtered_data['Low'].min())  # 필터링된 데이터의 저점과 비교

        # 지지선 조건 (당일 제외)
        support_condition = last_close > overall_low * 1.01  # 최근 종가가 전체 저점의 1% 초과
        
        # 장대 양봉 발생 시의 OBV 값 저장
        if bullish_candle_index is not None:
            obv_at_bullish_candle = df['obv'].iloc[bullish_candle_index]  # 장대 양봉 발생 시의 OBV

            # 조건 확인: 가격 상승 조건, Williams %R, RSI, MACD, 지지선 확인
            if (price_increase_condition and 
                williams_r <= -80 and 
                rsi_condition and 
                support_condition and 
                macd_condition and 
                obv_current > obv_at_bullish_candle):  # OBV 세력 조건 추가
                result = {
                    'Code': code,
                    'Last Close': int(last_close),  # int로 변환
                    'Opening Price': int(opening_price),  # int로 변환
                    'Lowest Price': int(overall_low),  # int로 변환
                    'Highest Price': int(recent_data['High'].max()),  # int로 변환
                    'Williams %R': float(williams_r),  # float으로 변환
                    'OBV': int(obv_current),  # int로 변환
                    'Support Condition': bool(support_condition),  # bool로 변환
                    'OBV Strength Condition': bool(obv_current > obv_at_bullish_candle),  # bool로 변환
                }
                logging.info(f"{code} 조건 만족: {result}")
                print(f"만족한 종목 코드: {code}")  # 만족한 종목 코드
                return result  # 조건을 만족하는 경우 결과 반환
            else:
                logging.info(f"{code} 조건 불만족: "
                             f"가격 상승 조건: {price_increase_condition}, "
                             f"Williams %R: {williams_r}, "
                             f"RSI: {rsi_current}, "
                             f"OBV: {obv_current}, "
                             f"MACD: {macd_condition}, "
                             f"지지선 확인: {support_condition}, "
                             f"OBV 세력 확인: {obv_current > obv_at_bullish_candle}")

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
    with ThreadPoolExecutor(max_workers=15) as executor:  # 최대 15개의 스레드 사용
        futures = {executor.submit(analyze_stock, code, start_date): code for code in stocks['Code']}
        for future in as_completed(futures):
            result_data = future.result()
            if result_data:
                result.append(result_data)

    logging.info("주식 검색 완료")
    return result
    
# 메인 실행 블록에서 결과 저장 호출 추가
if __name__ == "__main__":
    logging.info("스크립트 실행 시작")
    
    # 최근 40 거래일을 기준으로 시작 날짜 설정
    today = datetime.today()
    start_date = today - timedelta(days=40)  # 최근 40 거래일 전 날짜
    start_date_str = start_date.strftime('%Y-%m-%d')

    logging.info(f"주식 분석 시작 날짜: {start_date_str}")

    results = search_stocks(start_date_str)  # 결과를 변수에 저장
    if results:  # 결과가 있을 때만 출력
        logging.info(f"만족한 종목 리스트: {[stock['Code'] for stock in results]}")
        save_results_to_json(results)  # JSON 파일로 저장
    else:
        logging.info("조건을 만족하는 종목이 없습니다.")
