import sys
import logging
import requests
import openai
import os
import pandas as pd
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 콘솔 로그 출력 설정
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)

def send_discord_message(webhook_url, message):
    """Discord 웹훅으로 메시지를 전송하는 함수."""
    data = {
        "content": message
    }
    response = requests.post(webhook_url, json=data)
    if response.status_code != 204:
        logging.error(f"메시지 전송 실패: {response.status_code} {response.text}")

def get_ai_response(api_key, prompt):
    """AI에게 질문을 하고 응답을 받는 함수."""
    openai.api_key = api_key
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # 모델을 GPT-4o-mini로 설정
            messages=[
                {"role": "system", "content": "당신은 최고의 전문 기술 투자자입니다."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300  # 최대 토큰 수 설정
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        logging.error(f"AI 응답 오류: {str(e)}")
        return None

def main():
    # 환경 변수에서 설정 로드
    discord_webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
    openai_api_key = os.getenv('OPENAI_API_KEY')

    # 인수로 전달된 파일 경로 확인
    if len(sys.argv) < 2:
        logging.error("파일 경로가 제공되지 않았습니다.")
        return

    filename = sys.argv[1]
    logging.info(f"파일 경로: {filename}")

    try:
        # CSV 파일 읽기
        top_stocks = pd.read_csv(filename)
        logging.info(f"읽어온 데이터 개수: {len(top_stocks)}개")

        # 데이터 확인
        if top_stocks.empty:
            logging.error("읽어온 데이터가 비어 있습니다.")
            return
        
        # 전체 데이터 프레임 로그
        logging.info("전체 데이터 프레임:")
        logging.info(top_stocks.to_string())  # 전체 데이터 프레임 출력

        # 현재 날짜를 가져오는 부분
        current_date = datetime.today()

        # 최근 26 거래일 데이터만 남기기
        filtered_stocks = []
        for code in top_stocks['Code'].unique():
            stock_data = top_stocks[top_stocks['Code'] == code]

            # 거래일 필터링
            trading_days = 0
            for record_index, record in stock_data.iterrows():
                record_date = datetime.strptime(record['Date'], '%Y-%m-%d')
                if trading_days < 26:  # 최근 26 거래일만 남기기
                    filtered_stocks.append(record)
                    trading_days += 1

        # 필터링된 데이터로 DataFrame 생성
        filtered_df = pd.DataFrame(filtered_stocks)
        logging.info(f"필터링된 데이터 개수: {len(filtered_df)}개")

        # AI에게 전달할 분석 프롬프트
        analysis_prompt = (
            f"주식 데이터는 다음과 같습니다:\n{filtered_df.to_json(orient='records', force_ascii=False)}\n"
            f"각 종목 코드에 대한 다음 거래일(현재 날짜: {current_date.strftime('%Y-%m-%d')})에 상승 가능성을 예측해주세요. "
            f"예측은 최근 가격, 거래량, 그리고 기술적 지표(예: 이동 평균, RSI 등)를 기반으로 하며, "
            f"상승 가능성이 70% 이상인 종목 코드와 그 상승 가능성을 높은 순서로 나열해 주세요. "
            f"상승 가능성이 70% 이상인 경우, 그 이유를 기술적 지표와 함께 분석하여 설명해 주세요.\n"
            f"예시 출력 형식: 종목 코드: 상승 가능성 %, 이유 (예: 기술적 지표에 따른 분석)"
        )

        ai_response = get_ai_response(openai_api_key, analysis_prompt)

        if ai_response:
            logging.info(f"AI 응답: {ai_response}")
            # AI의 응답을 Discord 웹훅으로 전송
            send_discord_message(discord_webhook_url, f"상승 가능성 예측:\n{ai_response}")

    except FileNotFoundError:
        logging.error("파일을 찾을 수 없습니다.")
    except pd.errors.EmptyDataError:
        logging.error("CSV 파일이 비어 있습니다.")
    except pd.errors.ParserError:
        logging.error("CSV 파일을 읽는 중 오류 발생.")
    except Exception as e:
        logging.error(f"오류 발생: {str(e)}")

if __name__ == "__main__":
    main()
