import sys
import json
import logging
import requests
import openai
import os
from datetime import datetime, timedelta

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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
            model="gpt-4o-mini",  # 모델을 GPT-4로 변경
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
        # JSON 파일 읽기
        with open(filename, 'r', encoding='utf-8') as f:
            top_stocks = json.load(f)

        logging.info(f"읽어온 데이터 개수: {len(top_stocks)}개")

        # 현재 날짜
        current_date = datetime.today()

        # 데이터 확인 및 날짜 추가
        for stock in top_stocks:
            code = stock.get('code', '알 수 없음')
            score = stock.get('score', '알 수 없음')

            # 날짜 데이터가 없으면 기본 날짜 추가
            if 'data' in stock:
                for index, record in enumerate(stock['data']):
                    if 'Date' not in record or not record['Date']:
                        record['Date'] = (current_date - timedelta(days=index)).strftime('%Y-%m-%d')
                        logging.info(f"종목 코드: {code} - 날짜 추가: {record['Date']}")

                # 최근 26 거래일 데이터만 남기기
                trading_days = 0
                filtered_data = []
                for record in reversed(stock['data']):
                    record_date = datetime.strptime(record['Date'], '%Y-%m-%d')
                    if trading_days < 26:
                        filtered_data.append(record)
                        if record_date.weekday() < 5:  # 0-4: 월-금
                            trading_days += 1
                stock['data'] = list(reversed(filtered_data))  # 원래 순서로 복원
                logging.info(f"종목 코드: {code} - 최근 26 거래일 데이터 개수: {len(stock['data'])}")

        # AI에게 JSON 파일의 데이터를 기반으로 분석 요청
        analysis_prompt = (
            f"다음 거래일에 상승 가능성이 높은 5개 종목을 분석해 주세요.\n"
            f"주식 데이터는 다음과 같습니다:\n{json.dumps(top_stocks, ensure_ascii=False)}\n"
            f"가져온 데이터에서 점수: {score}는 15%의 비중만 주세요.나저지 데이터는 85%의 비중을 주세요"
            f"각 종목에 대한 상승 가능성을 %로 예측하고, 간단한 이유를 20자 내외로 작성해 주세요."
        )
        
        ai_response = get_ai_response(openai_api_key, analysis_prompt)

        if ai_response:
            logging.info(f"AI 응답: {ai_response}")
            # AI의 응답을 Discord 웹훅으로 전송
            send_discord_message(discord_webhook_url, f"상승 가능성 예측:\n{ai_response}")

    except FileNotFoundError:
        logging.error("파일을 찾을 수 없습니다.")
    except json.JSONDecodeError:
        logging.error("JSON 파일을 읽는 중 오류 발생.")
    except Exception as e:
        logging.error(f"오류 발생: {str(e)}")

if __name__ == "__main__":
    main()
