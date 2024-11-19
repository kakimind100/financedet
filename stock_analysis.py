import FinanceDataReader as fdr
import pandas as pd
import logging
from datetime import datetime, timedelta
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
console_handler.setLevel(logging.INFO)  # 콘솔에는 INFO 레벨 이상만 출력
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)

def is_cup_with_handle(df):
    """컵과 핸들 패턴을 찾는 함수."""
    if len(df) < 60:  # 최소 60일의 데이터 필요
        logging.debug(f"데이터 길이가 60일 미만입니다. 종목 코드: {df['Code'].iloc[0] if 'Code' in df.columns else 'N/A'}")
        return False, None
    
    # 컵의 저점과 핸들의 저점 찾기
    cup_bottom = df['Low'].min()
    cup_bottom_index = df['Low'].idxmin()
    
    # 컵의 높이
    cup_top = df['Close'][:cup_bottom_index].max()
    
    # 핸들 형성 여부 확인 (컵 끝에서 약간의 조정)
    handle = df.iloc[cup_bottom_index:cup_bottom_index + 10]  # 핸들 데이터 (10일)
    handle_top = handle['Close'].max()
    
    # 컵과 핸들 패턴 조건
    if handle_top < cup_top and cup_bottom < handle_top:
        logging.debug(f"패턴 발견! 종목 코드: {df['Code'].iloc[0] if 'Code' in df.columns else 'N/A'}")
        return True, df.index[-1]  # 최근 날짜 반환
    return False, None

def search_stocks(start_date, end_date):
    """주식 종목을 검색하고 가장 최근 Cup with Handle 패턴을 찾는 함수."""
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
    recent_cup_with_handle = None
    recent_date = None

    for code in stocks['Code']:
        try:
            logging.debug(f"종목 코드 {code} 데이터 가져오는 중...")
            df = fdr.DataReader(code, start=start_date, end=end_date)  # 시작 및 종료 날짜를 사용하여 데이터 가져오기
            
            # 'Code' 컬럼을 DataFrame에 추가
            df['Code'] = code
            
            is_pattern, pattern_date = is_cup_with_handle(df)
            if is_pattern:
                if recent_date is None or pattern_date > recent_date:
                    recent_date = pattern_date
                    recent_cup_with_handle = code
                    logging.info(f"{code}에서 최근 Cup with Handle 패턴 발견 (완성 날짜: {pattern_date})")
            else:
                logging.debug(f"{code}에서 Cup with Handle 패턴 발견하지 못함.")
        except Exception as e:
            logging.error(f"{code} 처리 중 오류 발생: {e}")

    logging.info("주식 검색 완료")
    return recent_cup_with_handle, recent_date

# 메인 실행 블록
if __name__ == "__main__":
    logging.info("주식 분석 스크립트 실행 중...")
    
    # 최근 1년을 기준으로 시작 날짜 설정
    today = datetime.today()
    start_date = today - timedelta(days=365)  # 최근 1년 전 날짜
    end_date = today.strftime('%Y-%m-%d')  # 오늘 날짜
    start_date_str = start_date.strftime('%Y-%m-%d')

    logging.info(f"주식 분석 시작 날짜: {start_date_str}")

    recent_stock, date_found = search_stocks(start_date_str, end_date)  # 결과를 변수에 저장
    if recent_stock:  # 최근 패턴이 발견된 경우
        logging.info(f"가장 최근 Cup with Handle 패턴이 발견된 종목: {recent_stock} (완성 날짜: {date_found})")
    else:
        logging.info("Cup with Handle 패턴을 가진 종목이 없습니다.")
