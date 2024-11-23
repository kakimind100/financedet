import pandas as pd
import logging
import os
import pandas_ta as ta
import json  # JSON 파일 처리를 위한 라이브러리

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
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)

def calculate_technical_indicators(target_code):
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
        df.set_index(['Code', 'Date'], inplace=True)

        # 중복된 데이터 처리
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

    # 기술적 지표 계산
    try:
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        macd = ta.macd(df['Close'])
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_Signal'] = macd['MACDh_12_26_9']
        bollinger_bands = ta.bbands(df['Close'], length=20, std=2)
        df['Bollinger_High'] = bollinger_bands['BBU_20_2.0']
        df['Bollinger_Low'] = bollinger_bands['BBL_20_2.0']
        stoch = ta.stoch(df['High'], df['Low'], df['Close'])
        df['Stoch'] = stoch['STOCHk_14_3_3']
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], length=20)
        df['EMA20'] = ta.ema(df['Close'], length=20)
        df['EMA50'] = ta.ema(df['Close'], length=50)

        logging.info("기술적 지표를 계산했습니다.")
    except Exception as e:
        logging.error(f"기술적 지표 계산 중 오류 발생: {e}")
        return

    # NaN 값이 있는 행 제거
    df.dropna(inplace=True)

    # 상위 20개 종목 선택 (예: MA20 기준으로 정렬)
    top_20 = df.groupby(level=0).last().nlargest(20, 'MA20')

    # 상위 20개 종목의 기술적 지표를 JSON으로 저장
    top_20_dict = top_20.reset_index().to_dict(orient='records')
    output_json_file = os.path.join(data_dir, 'top_20_stocks.json')
    
    with open(output_json_file, 'w') as json_file:
        json.dump(top_20_dict, json_file, indent=4)
    
    logging.info(f"상위 20개 종목의 기술적 지표가 '{output_json_file}'로 저장되었습니다.")

if __name__ == "__main__":
    target_code = '006280'  # 특정 종목 코드를 입력하세요.
    logging.info("기술 지표 계산 스크립트 실행 중...")
    calculate_technical_indicators(target_code)
    logging.info("기술 지표 계산 스크립트 실행 완료.")
