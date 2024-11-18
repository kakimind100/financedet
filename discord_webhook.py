import os
import json
import openai
import requests
import logging
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")  # 환경 변수에서 API 키를 가져옴
logging.info("OpenAI API 키를 설정했습니다.")

# 한국의 공휴일을 반환하는 함수
def get_holidays(year):
    holidays = [
        f"{year}-01-01",  # 신정
        f"{year}-03-01",  # 삼일절
        f"{year}-05-01",  # 노동절
        f"{year}-06-06",  # 현충일
        # 추가 공휴일...
    ]
    return holidays

# 주식 시장이 열리는 날인지 확인하는 함수
def is_market_open(date, manual_run):
    if manual_run:
        return True
    if date.weekday() >= 5:  # 5: 토요일, 6: 일요일
        return False
    holidays = get_holidays(date.year)
    if date.strftime("%Y-%m-%d") in holidays:
        return False
    return True

# 웹훅을 통해 메시지를 디스코드 채널로 보내는 함수
def send_to_discord_webhook(webhook_url, message):
    data = {"content": message}
    response = requests.post(webhook_url, json=data)
    if response.status_code == 204:
        logging.info("메시지가 성공적으로 전송되었습니다.")
    else:
        logging.error(f"메시지 전송 실패: {response.status_code} - {response.text}")

# AI를 사용하여 주식 분석 결과를 생성하는 함수
def generate_ai_response(stock_data):
    prompt = (
        "주어진 주식 데이터를 기반으로 다음 거래일에 가장 많이 상승할 가능성이 있는 종목을 우선순위에 따라 추천해 주세요. "
        "각 종목의 우선순위는 Williams %R(20% 비중), OBV(40% 비중), 가격 변동성(10% 비중), RSI(30% 비중)의 조합에 따라 결정되며, "
        "각 종목의 상승 가능성을 0에서 100 사이의 점수로 평가하고 추천 이유를 설명해 주세요. "
        "결과는 점수가 높은 5개 종목으로 한정'종목 코드: [종목 코드], 추천 이유: [이유는 간략하게 10자내외로 작성해 주세요], 상승 가능성: [점수]' 형식으로 줄바꿈 없이 작성해 주세요."
    )
    
    for stock in stock_data:
        prompt += (f"종목 코드: {stock['Code']}, "
                   f"마지막 종가: {stock['Last Close']}, "
                   f"개장가: {stock['Opening Price']}, "
                   f"최저가: {stock['Lowest Price']}, "
                   f"최고가: {stock['Highest Price']}, "
                   f"Williams %R: {stock['Williams %R']}, "
                   f"OBV: {stock['OBV']}, "
                   f"RSI: {stock['RSI']}, "  # RSI 추가
                   f"지지선 확인: {stock['Support Condition']}, "
                   f"OBV 세력 확인: {stock['OBV Strength Condition']}\n")

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "이 시스템은 최고의 주식 분석 시스템입니다."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.3
        )
        logging.info("AI의 응답을 성공적으로 받았습니다.")
        result = response['choices'][0]['message']['content']
        
        return result.replace("\n", "\n")
    except Exception as e:
        logging.error(f"API 호출 중 오류 발생: {e}")
        return None

# 메인 함수
def main():
    logging.info("스크립트 실행 시작.")
    
    today = datetime.today()
    manual_run = os.getenv("MANUAL_RUN", "false").lower() == "true"

    if not is_market_open(today, manual_run):
        logging.info("오늘은 주식 시장이 열리지 않습니다. 스크립트를 종료합니다.")
        return

    filename = 'results.json'
    if not os.path.exists(filename):
        logging.error(f"{filename} 파일이 존재하지 않습니다.")
        return

    logging.info(f"{filename} 파일을 열고 데이터를 읽고 있습니다.")
    with open(filename, 'r') as f:
        results = json.load(f)

    if isinstance(results, str):
        try:
            results = json.loads(results)
            logging.info("결과를 문자열에서 JSON으로 변환했습니다.")
        except json.JSONDecodeError:
            logging.error("결과가 올바른 JSON 형식이 아닙니다.")
            return

    if not isinstance(results, list):
        logging.error("결과가 리스트 형식이 아닙니다.")
        return

    logging.info(f"총 {len(results)}개의 종목 데이터가 로드되었습니다.")
    logging.info(f"로드된 종목 데이터: {json.dumps(results, ensure_ascii=False, indent=2)}")

    ai_response = generate_ai_response(results)

    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if webhook_url:
        message = f"AI 분석 결과:\n{ai_response}"
        logging.info("전송할 메시지 생성 완료.")
        logging.info(f"전송할 메시지: {message}")
        send_to_discord_webhook(webhook_url, message)
    else:
        logging.error("웹훅 URL이 설정되어 있지 않습니다.")

if __name__ == "__main__":
    main()
