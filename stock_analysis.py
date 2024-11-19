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

                # 날짜 정보가 포함되어 있는지 확인
                if 'Date' not in df.columns:
                    df['Date'] = pd.date_range(end=datetime.today(), periods=len(df), freq='B')  # 비즈니스 일 기준으로 날짜 추가
                    logging.info(f"{code} 데이터에 날짜 정보를 추가했습니다.")
                
                # 종목별로 데이터 저장
                result[code] = df.to_dict(orient='records')  # 리스트 형태의 딕셔너리로 변환
            except Exception as e:
                logging.error(f"{code} 처리 중 오류 발생: {e}")

    logging.info("주식 검색 완료")
    return result  # 결과 반환

def send_stock_analysis_to_ai(stock_data):
    """AI에게 주식 데이터 분석을 요청하는 함수."""
    analysis_request = (
        "다음은 최근 40일간의 주식 데이터입니다:\n"
    )
    
    for code, data in stock_data.items():
        # 최근 40일의 종가 정보만 포함
        recent_40_days = data[-40:]  # 최근 40일의 데이터
        recent_dates = [record['Date'] for record in recent_40_days]
        recent_closes = [record['Close'] for record in recent_40_days]

        analysis_request += (
            f"{code}: 최근 40일 종가: {', '.join([f'{date}: {close:.2f}' for date, close in zip(recent_dates, recent_closes)])}\n"
        )
    
    analysis_request += (
        "주식의 상승 가능성을 %로 표기하고, 상위 5개 종목의 이유를 20자 내외로 작성해 주세요."
    )
    
    response = openai.ChatCompletion.create(
        model="gpt-4-o",  # GPT-4O 미니 모델로 변경
        messages=[
            {"role": "system", "content": "당신은 최고의 주식 전문 투자자입니다."},
            {"role": "user", "content": analysis_request}
        ],
        max_tokens=150  # 최대 토큰 수 설정
    )
    return response['choices'][0]['message']['content']

# 메인 실행 블록에서 결과 저장 호출 추가
if __name__ == "__main__":
    logging.info("주식 분석 스크립트 실행 중...")
    
    # 최근 40일을 기준으로 시작 날짜 설정
    today = datetime.today()
    start_date = today - timedelta(days=40)  # 최근 40일 전 날짜
    start_date_str = start_date.strftime('%Y-%m-%d')

    logging.info(f"주식 분석 시작 날짜: {start_date_str}")

    results = search_stocks(start_date_str)  # 주식 데이터를 가져옴
    if results:  # 결과가 있을 때만 출력
        logging.info("가져온 종목 리스트:")
        for i in range(0, len(results), 10):  # 10개씩 나누어 출력
            logging.info(list(results.keys())[i:i+10])  # 종목 코드 리스트에서 10개씩 출력
        
        # AI에게 주식 데이터 분석 요청
        insights = send_stock_analysis_to_ai(results)  # results가 stock_data로 전달됨
        logging.info("AI의 주식 분석 결과:")
        logging.info(insights)  # AI의 응답 출력

        save_results_to_json(results)  # JSON 파일로 저장
    else:
        logging.info("조건을 만족하는 종목이 없습니다.")
