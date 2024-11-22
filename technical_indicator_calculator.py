import pandas as pd
import logging
import os
import pandas_ta as ta  # pandas_ta 라이브러리 임포트

# 로그 디렉토리 설정
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)  # 로그 디렉토리가 없으면 생성

# 로깅 설정
logging.basicConfig(
    filename=os.path.join(log_dir, 'technical_indicator_calculator.log'),  # 로그 파일 경로
    level=logging.DEBUG,  # 로깅 레벨을 DEBUG로 설정
    format='%(asctime)s - %(levelname)s - %(message)s'  # 로그 메시지 형식
)

# 콘솔 로그 출력 설정
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # 콘솔 로그 레벨 설정
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))  # 콘솔 로그 형식
logging.getLogger().addHandler(console_handler)  # 콘솔 핸들러 추가

def calculate_technical_indicators():
    """기술적 지표를 계산하는 함수."""
    # CSV 파일 읽기
    data_dir = 'data'
    dtype = {
        'Date': 'str',  # 날짜는 문자열로 읽기
        'Open': 'float', 
        'High': 'float', 
        'Low': 'float', 
        'Close': 'float', 
        'Volume': 'float'  # 거래량도 float로 처리
    }
    
    try:
        df = pd.read_csv(os.path.join(data_dir, 'stock_data.csv'), dtype=dtype)  # dtype 지정
        logging.debug(f"CSV 파일 '{os.path.join(data_dir, 'stock_data.csv')}'을(를) 성공적으로 읽었습니다.")
        
        # 날짜 열을 datetime 형식으로 변환
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')  # 날짜 형식 변환
        df.set_index('Date', inplace=True)  # 날짜를 인덱스로 설정

        # 데이터 타입 로그
        logging.info("데이터프레임 열의 데이터 타입:")
        for column, dtype in df.dtypes.items():
            logging.info(f"{column}: {dtype}")

    except FileNotFoundError:
        logging.error(f"파일 '{os.path.join(data_dir, 'stock_data.csv')}'을(를) 찾을 수 없습니다.")
        return
    except pd.errors.EmptyDataError:
        logging.error("CSV 파일이 비어 있습니다.")
        return
    except Exception as e:
        logging.error(f"CSV 파일 읽기 중 오류 발생: {e}")
        return

    # 이동 평균 계산
    df['MA5'] = df['Close'].rolling(window=5).mean()  # 5일 이동 평균
    df['MA20'] = df['Close'].rolling(window=20).mean()  # 20일 이동 평균
    logging.debug("이동 평균(MA5, MA20)을 계산했습니다.")

    # 다양한 기술적 지표 추가
    df['RSI'] = ta.rsi(df['Close'], length=14)  # 상대 강도 지수
    macd = ta.macd(df['Close'])  # MACD 계산
    df['MACD'] = macd['MACD']  # MACD
    df['MACD_Signal'] = macd['MACDh']  # MACD Signal
    df['Bollinger_High'], df['Bollinger_Low'] = ta.bbands(df['Close'], length=20, std=2).iloc[:, 0:2].T  # Bollinger Bands
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)  # Average True Range
    df['Stoch'] = ta.stoch(df['High'], df['Low'], df['Close'])['STOCHK']  # Stochastic Oscillator
    df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], length=20)  # Commodity Channel Index
    df['EMA20'] = ta.ema(df['Close'], length=20)  # 20일 지수 이동 평균
    df['EMA50'] = ta.ema(df['Close'], length=50)  # 50일 지수 이동 평균
    logging.debug("기술적 지표(RSI, MACD, Bollinger Bands 등)를 계산했습니다.")

    # 계산된 데이터프레임을 CSV로 저장
    output_file = os.path.join(data_dir, 'stock_data_with_indicators.csv')  # 결과 저장 경로
    df.to_csv(output_file)  # CSV로 저장
    logging.info("기술적 지표가 'stock_data_with_indicators.csv'로 저장되었습니다.")  # 성공 로그

if __name__ == "__main__":
    calculate_technical_indicators()  # 함수 실행
