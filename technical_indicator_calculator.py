import pandas as pd
import logging
import os

# 로그 디렉토리 설정
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)  # 로그 디렉토리가 없으면 생성

# 로깅 설정
logging.basicConfig(
    filename=os.path.join(log_dir, 'technical_indicator_calculator.log'),  # 로그 파일 경로
    level=logging.INFO,  # 로깅 레벨 설정
    format='%(asctime)s - %(levelname)s - %(message)s'  # 로그 메시지 형식
)

def compute_rsi(series, period=14):
    """상대 강도 지수(RSI)를 계산하는 함수."""
    delta = series.diff()  # 가격 변화량 계산
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()  # 상승분 평균
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()  # 하락분 평균
    rs = gain / loss  # 상대 강도 계산
    rsi = 100 - (100 / (1 + rs))  # RSI 계산
    return rsi

def calculate_technical_indicators():
    """기술적 지표를 계산하는 함수."""
    # CSV 파일 읽기
    data_dir = 'data'
    df = pd.read_csv(os.path.join(data_dir, 'stock_data.csv'))  # 수정된 경로
    
    # 이동 평균 계산
    df['MA5'] = df['Close'].rolling(window=5).mean()  # 5일 이동 평균
    df['MA20'] = df['Close'].rolling(window=20).mean()  # 20일 이동 평균
    df['RSI'] = compute_rsi(df['Close'])  # RSI 계산

    # NaN 값 처리
    df.fillna(0, inplace=True)  # NaN 값을 0으로 대체

    # 계산된 데이터프레임을 CSV로 저장
    output_file = os.path.join(data_dir, 'stock_data_with_indicators.csv')  # 결과 저장 경로
    df.to_csv(output_file, index=False)  # CSV로 저장
    logging.info("기술적 지표가 'stock_data_with_indicators.csv'로 저장되었습니다.")  # 성공 로그

if __name__ == "__main__":
    calculate_technical_indicators()  # 함수 실행
