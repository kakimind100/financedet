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

def calculate_technical_indicators():
    """기술적 지표를 계산하는 함수."""
    # CSV 파일 읽기
    data_dir = 'data'
    df = pd.read_csv(os.path.join(data_dir, 'stock_data.csv'))  # 수정된 경로
    logging.debug(f"CSV 파일 '{os.path.join(data_dir, 'stock_data.csv')}'을(를) 성공적으로 읽었습니다.")

    # 이동 평균 계산
    df['MA5'] = df['Close'].rolling(window=5).mean()  # 5일 이동 평균
    df['MA20'] = df['Close'].rolling(window=20).mean()  # 20일 이동 평균
    logging.debug("이동 평균(MA5, MA20)을 계산했습니다.")

    # 다양한 기술적 지표 추가
    df['RSI'] = ta.rsi(df['Close'], length=14)  # 상대 강도 지수
    df['MACD'] = ta.macd(df['Close'])['MACD']  # MACD
    df['MACD_Signal'] = ta.macd(df['Close'])['MACDh']  # MACD Signal
    df['Bollinger_High'], df['Bollinger_Low'] = ta.bbands(df['Close'], length=20, std=2).iloc[:, 0:2].T  # Bollinger Bands
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)  # Average True Range
    df['Stoch'] = ta.stoch(df['High'], df['Low'], df['Close'])['STOCHK']  # Stochastic Oscillator
    df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], length=20)  # Commodity Channel Index
    df['EMA20'] = ta.ema(df['Close'], length=20)  # 20일 지수 이동 평균
    df['EMA50'] = ta.ema(df['Close'], length=50)  # 50일 지수 이동 평균
    logging.debug("기술적 지표(RSI, MACD, Bollinger Bands 등)를 계산했습니다.")

    # NaN 값 처리
    df.fillna(0, inplace=True)  # NaN 값을 0으로 대체
    logging.debug("NaN 값을 0으로 대체했습니다.")

    # 계산된 데이터프레임을 CSV로 저장
    output_file = os.path.join(data_dir, 'stock_data_with_indicators.csv')  # 결과 저장 경로
    df.to_csv(output_file, index=False)  # CSV로 저장
    logging.info("기술적 지표가 'stock_data_with_indicators.csv'로 저장되었습니다.")  # 성공 로그

if __name__ == "__main__":
    calculate_technical_indicators()  # 함수 실행
