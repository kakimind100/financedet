import FinanceDataReader as fdr
import pandas as pd
import logging
import os
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# OpenAI API 설정
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  # 시크릿에서 API 키를 가져옵니다.

# 로그 디렉토리 생성
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# 로깅 설정
logging.basicConfig(
    filename=os.path.join(log_dir, 'stock_analysis.log'),
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# CSV 파일에 데이터 저장
def save_to_csv(data, filename='stock_data.csv'):
    df = pd.DataFrame(data)
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)  # 기존 파일에 추가
        logging.info(f"{filename}에 기존 데이터 추가 완료.")
    else:
        df.to_csv(filename, index=False)  # 새 파일 생성
        logging.info(f"{filename} 새 파일 생성 완료.")
    logging.info(f"{len(data)}개의 데이터를 {filename}에 저장 완료.")

def fetch_and_store_stock_data(code, start_date):
    """주식 데이터를 가져와서 CSV 파일에 저장하는 함수."""
    logging.info(f"{code} 데이터 가져오기 시작")
    try:
        df = fdr.DataReader(code, start=start_date)
        logging.info(f"{code} 데이터 가져오기 성공, 가져온 데이터 길이: {len(df)}")

        if len(df) < 1:
            logging.warning(f"{code} 데이터가 없습니다.")
            return []

        result = []
        for index, row in df.iterrows():
            result.append({
                'Code': str(code),
                'Date': index.strftime('%Y-%m-%d'),
                'Opening Price': float(row['Open']),
                'Highest Price': float(row['High']),
                'Lowest Price': float(row['Low']),
                'Last Close': float(row['Close']),
                'Volume': int(row['Volume'])
            })
            logging.info(f"{code} - {index.strftime('%Y-%m-%d')} 데이터 추가 완료.")

        logging.info(f"{code} 데이터 처리 완료: {len(result)}개 항목.")
        return result

    except Exception as e:
        logging.error(f"{code} 처리 중 오류 발생: {e}", exc_info=True)
        return []

def analyze_stocks(data):
    """OpenAI API를 사용하여 기술적 분석을 수행하고 상승 예측을 반환하는 함수."""
    messages = [
        {"role": "system", "content": "당신은 금융 분석가입니다."},
        {"role": "user", "content": (
            "다음은 주식 데이터입니다:\n"
            f"{data}\n\n"
            "이 데이터를 분석하여 다음 거래일에 가장 많이 상승할 것으로 예상되는 "
            "종목을 0%~100%까지의 비율로 순위를 매기고, "
            "상위 5개 주식에 대해 각 종목의 종목 코드와 추천 이유를 20자 내외로 작성해 주세요."
        )}
    ]

    try:
        response = requests.post('https://api.openai.com/v1/chat/completions', headers={
            'Authorization': f'Bearer {OPENAI_API_KEY}',
            'Content-Type': 'application/json'
        }, json={"model": "gpt-3.5-turbo", "messages": messages})

        if response.status_code == 200:
            logging.info("OpenAI API 요청 성공.")
            return response.json()['choices'][0]['message']['content']
        else:
            logging.error(f"OpenAI API 요청 실패: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        logging.error(f"OpenAI API 요청 중 오류 발생: {e}", exc_info=True)
        return None

def main():
    logging.info("스크립트 실행 시작")

    today = datetime.today()
    start_date = today - timedelta(days=730)
    start_date_str = start_date.strftime('%Y-%m-%d')

    logging.info(f"주식 분석 시작 날짜: {start_date_str}")

    # KOSPI와 KOSDAQ 종목 목록 가져오기
    try:
        kospi = fdr.StockListing('KOSPI')
        logging.info("코스피 종목 목록 가져오기 성공, 종목 수: {}".format(len(kospi)))
        
        kosdaq = fdr.StockListing('KOSDAQ')
        logging.info("코스닥 종목 목록 가져오기 성공, 종목 수: {}".format(len(kosdaq)))
    except Exception as e:
        logging.error(f"종목 목록 가져오기 중 오류 발생: {e}", exc_info=True)
        return

    stocks = pd.concat([kospi, kosdaq])
    all_results = []

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_and_store_stock_data, code, start_date): code for code in stocks['Code']}
        for future in as_completed(futures):
            stock_data = future.result()
            if stock_data:
                all_results.extend(stock_data)

    if all_results:
        save_to_csv(all_results)
        logging.info(f"총 저장된 데이터 수: {len(all_results)}")

        # OpenAI API를 통한 기술적 분석 및 상승 예측
        analysis_result = analyze_stocks(all_results)
        if analysis_result:
            logging.info(f"상승 예측 결과:\n{analysis_result}")
        else:
            logging.info("분석 결과를 가져오는 데 실패했습니다.")
    else:
        logging.info("저장할 데이터가 없습니다.")

if __name__ == "__main__":
    main()
