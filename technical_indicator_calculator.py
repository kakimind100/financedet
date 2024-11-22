import pandas as pd
import logging
import os
import pandas_ta as ta  # pandas_ta 라이브러리 임포트

# 로그 디렉토리 설정
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# 로깅 설정
logging.basicConfig(
    filename=os.path.join(log_dir, 'technical_indicator_calculator.log'),
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 콘솔 로그 출력 설정
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)

def calculate_technical_indicators():
    """기술적 지표를 계산하는 함수."""
    data_dir = 'data'
    dtype = {
        'Date': 'str',
        'Open': 'float',
        'High': 'float',
        'Low': 'float',
        'Close': 'float',
        'Volume': 'float',
        'Change': 'float',
        'Code': 'object'
    }

    try:
        df = pd.read_csv(os.path.join(data_dir, 'stock_data.csv'), dtype=dtype)
        logging.debug(f"CSV 파일 '{os.path.join(data_dir, 'stock_data.csv')}'을(를) 성공적으로 읽었습니다.")

        # 날짜 열을 datetime 형식으로 변환
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        df.set_index(['Code', 'Date'], inplace=True)  # 종목 코드와 날짜를 인덱스로 설정

        # 중복된 데이터 처리: 종목 코드와 날짜로 그룹화하여 평균값으로 대체
        df = df.groupby(['Code', df.index.get_level_values('Date')]).mean()

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
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    logging.debug("이동 평균(MA5, MA20)을 계산했습니다.")

    # MACD 계산
    macd = ta.macd(df['Close'])
    if 'MACD' in macd.columns:
        df['MACD'] = macd['MACD']
        df['MACD_Signal'] = macd['MACDh']
        logging.info("MACD 계산 완료.")
    else:
        logging.error("MACD 계산 결과에 'MACD' 열이 없습니다.")
        logging.debug(f"MACD 결과: {macd}")
        return

    # Bollinger Bands 계산
    try:
        bollinger_bands = ta.bbands(df['Close'], length=20, std=2)
        df['Bollinger_High'] = bollinger_bands['BBH']
        df['Bollinger_Low'] = bollinger_bands['BBL']
        logging.info("Bollinger Bands 계산 완료.")
    except Exception as e:
        logging.error(f"Bollinger Bands 계산 중 오류 발생: {e}")
        return

    # 기술적 지표 추가
    try:
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['Stoch'] = ta.stoch(df['High'], df['Low'], df['Close'])['STOCHK']
        df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], length=20)
        df['EMA20'] = ta.ema(df['Close'], length=20)
        df['EMA50'] = ta.ema(df['Close'], length=50)
        logging.info("기술적 지표(RSI, ATR, Stochastic, CCI, EMA 등)를 계산했습니다.")
    except Exception as e:
        logging.error(f"기술적 지표 계산 중 오류 발생: {e}")
        return

    # 계산된 데이터프레임을 CSV로 저장
    output_file = os.path.join(data_dir, 'stock_data_with_indicators.csv')
    df.to_csv(output_file)
    logging.info("기술적 지표가 'stock_data_with_indicators.csv'로 저장되었습니다.")

if __name__ == "__main__":
    calculate_technical_indicators()
