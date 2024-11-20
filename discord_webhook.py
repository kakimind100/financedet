import sys
import json
import logging
import requests
import openai
import yaml
from datetime import datetime, timedelta

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_config():
    """YAML 파일에서 설정을 로드하는 함수."""
    with open("config.yml", 'r') as file:
        config = yaml.safe_load(file)
    return config

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
            model="gpt-4",  # 모델을 GPT-4로 변경
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
    # 설정 로드
    config = load_config()
    discord_webhook_url = config['discord']['webhook_url']
    openai_api_key = config['openai']['api_key']

    # 인수로 전달된 파일 경로 확인
    if len(sys.argv) < 2:
        logging.error("파일 경로가 제공되지 않았습니다.")
        return

    filename = sys.argv[1]
    logging.info(f"파일 경로: {filename}")

    try:
        # JSON 파일 읽기
        with open(filename, 'r', encoding='utf-8') as f:
            top_stocks = json.load(f)

        logging.info(f"읽어온 데이터 개수: {len(top_stocks)}개")

        # 데이터 확인 및 날짜 추가
        for stock in top_stocks:
            code = stock.get('code', '알 수 없음')
            score = stock.get('score', '알 수 없음')

            # 날짜 데이터가 없으면 기본 날짜 추가
            if 'data' in stock:
                for index, record in enumerate(stock['data']):
                    if 'Date' not in record or not record['Date']:
                        record['Date'] = (datetime.today() - timedelta(days=index)).strftime('%Y-%m-%d')
                        logging.info(f"종목 코드: {code} - 날짜 추가: {record['Date']}")

        # 상승 가능성 예측을 위한 프롬프트 생성
        analysis_prompt = "다음 거래일에 상승 가능성을 %로 예측하고 상위 5종목에 대해 20자 내외의 이유를 작성해 주세요."
        ai_response = get_ai_response(openai_api_key, analysis_prompt)

        if ai_response:
            logging.info(f"AI 응답: {ai_response}")
            send_discord_message(discord_webhook_url, f"상승 가능성 예측:\n{ai_response}")

    except FileNotFoundError:
        logging.error("파일을 찾을 수 없습니다.")
    except json.JSONDecodeError:
        logging.error("JSON 파일을 읽는 중 오류 발생.")
    except Exception as e:
        logging.error(f"오류 발생: {str(e)}")

if __name__ == "__main__":
    main()
