import FinanceDataReader as fdr
import pandas as pd
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
import openai

# OpenAI API 키 설정
openai.api_key = os.getenv('OPENAI_API_KEY')  # 환경 변수에서 API 키 가져오기

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
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 콘솔에도 로그 출력
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)

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
    result = {}

    # 멀티스레딩으로 주식 데이터 처리
    with ThreadPoolExecutor(max_workers=20) as executor:  # 최대 20개의 스레드 사용
        futures = {executor.submit(fdr.DataReader, code, start_date): code for code in stocks['Code']}
        for future in as_completed(futures):
            code = futures[future]
            try:
                df = future.result()
                logging.info(f"{code} 데이터 가져오기 성공, 가져온 데이터 길이: {len(df)}")
                
                # 날짜별 데이터 저장
                result[code] = df.to_dict(orient='records')  # 리스트 형태의 딕셔너리로 변환
            except Exception as e:
                logging.error(f"{code} 처리 중 오류 발생: {e}")

    logging.info("주식 검색 완료")
    return result

def analyze_stocks(stock_codes):
    """주식 데이터를 AI에게 분석하여 상승 가능성이 높은 종목 추천."""
    
    # AI에게 전체 종목 코드를 요청
    prompt = (
        "다음은 전체 주식 종목 코드입니다. "
        "각 종목의 상승 가능성을 0%에서 100%로 점수화하고, "
        "이유를 20자 내외로 작성한 후, 점수를 기준으로 상위 5개 종목을 추천해 주세요:\n\n"
        + "\n".join(stock_codes)  # 모든 종목 코드를 한 번에 추가
    )
    
    # OpenAI API를 통해 분석 요청
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # AI의 응답을 recommendations 리스트에 바로 추가
    recommendations = response['choices'][0]['message']['content'].strip().split('\n')

    return recommendations

# 메인 실행 블록에서 결과 저장 호출 추가
if __name__ == "__main__":
    logging.info("스크립트 실행 시작")
    
    # 최근 730 거래일을 기준으로 시작 날짜 설정
    today = datetime.today()
    start_date = today - timedelta(days=730)  # 최근 730 거래일 전 날짜
    start_date_str = start_date.strftime('%Y-%m-%d')

    logging.info(f"주식 분석 시작 날짜: {start_date_str}")

    results = search_stocks(start_date_str)  # 결과를 변수에 저장
    if results:  # 결과가 있을 때만 출력
        logging.info("가져온 종목 리스트:")
        for i in range(0, len(results), 10):  # 10개씩 나누어 출력
            logging.info(list(results.keys())[i:i+10])  # 종목 코드 리스트에서 10개씩 출력
        
        # AI 분석 및 추천
        stock_codes = list(results.keys())  # 종목 코드 리스트
        recommendations = analyze_stocks(stock_codes)  # AI에게 추천 요청
        logging.info("AI 추천 종목:")
        for rec in recommendations:
            logging.info(rec)  # 추천 종목 출력

        save_results_to_json(results)  # JSON 파일로 저장
    else:
        logging.info("조건을 만족하는 종목이 없습니다.")
