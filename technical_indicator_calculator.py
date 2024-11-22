# technical_indicator_calculator.py
import pandas as pd
import logging
import os

# 로그 디렉토리 설정
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# 로깅 설정
logging.basicConfig(
    filename=os.path.join(log_dir, 'technical_indicator_calculator.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def compute_rsi(series, period=14):
    """상대 강도 지수(RSI)를 계산하는 함수."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_technical_indicators():
    """기술적 지표를 계산하는 함수."""
    df = pd.read_csv('stock_data.csv')
    
    # 이동 평균 계산
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = compute_rsi(df['Close'])

    # NaN 값 처리
    df.fillna(0, inplace=True)

    # 계산된 데이터프레임을 CSV로 저장
    df.to_csv('stock_data_with_indicators.csv', index=False)
    logging.info("기술적 지표가 'stock_data_with_indicators.csv'로 저장되었습니다.")

if __name__ == "__main__":
    calculate_technical_indicators()
