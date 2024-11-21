import FinanceDataReader as fdr
import pandas as pd
import logging
import os
from datetime import datetime, timedelta

# 로그 및 JSON 파일 디렉토리 설정
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# 로깅 설정
logging.basicConfig(
    filename=os.path.join(log_dir, 'stock_analysis.log'),
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 콘솔 로그 출력 설정
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)

def fetch_stock_data(code, start_date, end_date):
    """주식 데이터를 가져오는 함수."""
    try:
        df = fdr.DataReader(code, start_date, end_date)
        logging.info(f"{code} 데이터 가져오기 성공, 데이터 길이: {len(df)}")

        # 날짜 인덱스 확인
        if df.index.empty:
            logging.warning(f"{code}에 유효한 날짜 데이터가 없습니다.")
        else:
            logging.info(f"{code}의 날짜 인덱스: {df.index}")

        return df
    except Exception as e:
        logging.error(f"{code} 데이터 가져오기 중 오류 발생: {e}")
        return None

# 사용 예
end_date = datetime.today()  # 현재 날짜
start_date = end_date - timedelta(days=365)  # 1년 전 날짜

# 실행 시작 메시지
logging.info("주식 분석 스크립트 실행 중...")
fetch_stock_data('AAPL', start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
