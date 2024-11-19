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
    level=logging.DEBUG,  # DEBUG로 설정하여 모든 로그를 기록
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def fetch_and_store_stock_data(code, start_date):
    """주식 데이터를 가져오는 함수."""
    logging.info(f"{code} 데이터 가져오기 시작")
    try:
        df = fdr.DataReader(code, start=start_date)
        logging.info(f"{code} 데이터 가져오기 성공, 데이터 길이: {len(df)}")

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
            logging.info(f"{code} - {index.strftime('%Y-%m-%d')} 데이터 추가 완료. "
                         f"시가: {row['Open']}, 종가: {row['Close']}, 거래량: {row['Volume']}")

        logging.info(f"{code} 데이터 처리 완료: {len(result)}개 항목.")
        return result

    except Exception as e:
        logging.error(f"{code} 처리 중 오류 발생: {e}", exc_info=True)
        return []

def analyze_stocks(data):
    """OpenAI API를 사용하여 기술적 분석을 수행하고 상승 예측을 반환하는 함수."""
    data_string = "\n".join([f"종목 코드: {item['Code']}, 날짜: {item['Date']}, "
                              f"시가: {item['Opening Price']}, 종가: {item['Last Close']}, "
                              f"거래량: {item['Volume']}" for item in data])

    messages = [
        {"role": "system", "content": "당신은 금융 분석가입니다."},
        {"role": "user", "content": (
            "이 데이터를 분석하여 다음 거래일에 가장 많이 상승할 것으로 예상되는 "
            "종목을 0%~100%까지의 비율로 순위를 매기고, "
            "상위 5개 주식에 대해 각 종목의 종목 코드와 추천 이유를 20자 내외로 작성해 주세요."
        )}
    ]

    try:
        response = requests.post('https://api.openai.com/v1/chat/completions', headers={
            'Authorization': f'Bearer {OPENAI_API_KEY}',
            'Content-Type': 'application/json'
        }, json={
            "model": "gpt-3.5-turbo",
            "messages": messages,
            "max_tokens": 30  # 최대 토큰 수를 30으로 설정
        })

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
    logging.info("주식 분석 스크립트가 시작되었습니다.")  # 스크립트 시작 로그

    today = datetime.today()
    start_date = today - timedelta(days=730)
    start_date_str = start_date.strftime('%Y-%m-%d')

    logging.info(f"주식 분석 시작 날짜: {start_date_str}")

    # KOSPI와 KOSDAQ 종목 목록 가져오기
    try:
        logging.info("KOSPI 종목 목록 가져오기 시작")
        kospi = fdr.StockListing('KOSPI')
        logging.info(f"코스피 종목 목록 가져오기 성공, 종목 수: {len(kospi)}")
        
        logging.info("KOSDAQ 종목 목록 가져오기 시작")
        kosdaq = fdr.StockListing('KOSDAQ')
        logging.info(f"코스닥 종목 목록 가져오기 성공, 종목 수: {len(kosdaq)}")
    except Exception as e:
        logging.error(f"종목 목록 가져오기 중 오류 발생: {e}", exc_info=True)
        return

    stocks = pd.concat([kospi, kosdaq])
    all_results = []

    logging.info("주식 데이터 수집 시작")  # 데이터 수집 시작 로그

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_and_store_stock_data, code, start_date): code for code in stocks['Code']}
        for future in as_completed(futures):
            stock_data = future.result()
            if stock_data:
                all_results.extend(stock_data)

    if all_results:
        logging.info(f"총 수집된 데이터 수: {len(all_results)}")

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
