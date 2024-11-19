import FinanceDataReader as fdr
import pandas as pd
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
import openai
import matplotlib
import matplotlib.font_manager as fm
matplotlib.use('Agg')  # Agg 백엔드 사용
import matplotlib.pyplot as plt

# 한글 글꼴 설정
font_path = "C:/Windows/Fonts/malgun.ttf"  # 본인의 환경에 맞는 경로로 수정
font_prop = fm.FontProperties(fname=font_path, size=12)
plt.rc('font', family=font_prop.get_name())

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
                    # 날짜 정보를 추가
                    df['Date'] = pd.date_range(end=datetime.today(), periods=len(df), freq='B')  # 비즈니스 일 기준으로 날짜 추가
                    logging.info(f"{code} 데이터에 날짜 정보를 추가했습니다.")
                
                # 종목별로 데이터 저장
                result[code] = df.to_dict(orient='records')  # 리스트 형태의 딕셔너리로 변환
            except Exception as e:
                logging.error(f"{code} 처리 중 오류 발생: {e}")

    logging.info("주식 검색 완료")
    return result

def visualize_stock_data(code, records):
    """주식 데이터를 시각화하여 이미지를 저장하는 함수."""
    df = pd.DataFrame(records)  # 리스트를 데이터프레임으로 변환
    df['Date'] = pd.to_datetime(df['Date'])  # 날짜 형식 변환
    df.set_index('Date', inplace=True)  # 날짜를 인덱스로 설정

    plt.figure(figsize=(12, 6))  # 그래프 크기 설정

    # 종가 그래프 그리기
    plt.plot(df.index, df['Close'], label=f'{code} 종가', color='blue')  # 종목 코드 포함

    # 거래량 그래프 그리기 (second y-axis)
    ax2 = plt.gca().twinx()  # 두 번째 y축 생성
    ax2.bar(df.index, df['Volume'], alpha=0.3, label=f'{code} 거래량', color='gray')

    plt.title(f'{code} 주식 종가 및 거래량 (2년간)')
    plt.xlabel('날짜')
    plt.ylabel('종가')
    ax2.set_ylabel('거래량')
    plt.legend(loc='upper left')  # 범례에 종목 코드 포함
    plt.xticks(rotation=45)
    plt.tight_layout()  # 레이아웃 조정

    # 그래프를 이미지 파일로 저장
    plt.savefig(f'{code}_stock_prices_and_volume.png')
    logging.info(f"{code} 그래프 이미지 저장 완료: '{code}_stock_prices_and_volume.png'")
    plt.close()  # 그래프 닫기

def send_graph_description_to_ai(graph_description):
    """AI에게 그래프 설명을 요청하는 함수."""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": graph_description}
        ],
        max_tokens=150  # 최대 토큰 수 설정
    )
    return response['choices'][0]['message']['content']

# 메인 실행 블록에서 결과 저장 호출 추가
if __name__ == "__main__":
    logging.info("주식 분석 스크립트 실행 중...")
    
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
        
        # 멀티스레딩으로 주식 데이터 시각화 (그래프 이미지화)
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = {executor.submit(visualize_stock_data, code, records): code for code, records in results.items()}
            for future in as_completed(futures):
                try:
                    future.result()  # 각 스레드의 작업 완료 대기
                except Exception as e:
                    logging.error(f"{code} 그래프 이미지화 중 오류 발생: {e}")

        # 그래프에 대한 설명
        graph_description = (
            "다음은 2년간의 주식 종가 및 거래량 그래프입니다. "
            "다음 그래프를 보고 다음 거래일에 가장 많이 상승 가능성을 %로 표기하고 "
            "상위 5개 종목을 이유 20자 내외로 함께 작성해 주세요."
        )

        # AI에게 그래프 설명 요청
        insights = send_graph_description_to_ai(graph_description)
        logging.info("AI의 그래프 분석 결과:")
        logging.info(insights)  # AI의 응답 출력

        save_results_to_json(results)  # JSON 파일로 저장
    else:
        logging.info("조건을 만족하는 종목이 없습니다.")
