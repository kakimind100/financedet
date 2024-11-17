import os
import json
import openai
import requests
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")  # 환경 변수에서 API 키를 가져옴

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
    prompt = "다음 주식 데이터에 대해 기술적 분석을 기반으로 투자 전략과 조언을 제공해 주세요:\n"
    
    for stock in stock_data:
        # 각 종목에 대한 데이터를 추가
        prompt += (f"종목 코드: {stock['Code']}, 마지막 종가: {stock['Last Close']}, "
                   f"개장가: {stock['Opening Price']}, 최저가: {stock['Lowest Price']}, "
                   f"최고가: {stock['Highest Price']}, Williams %R: {stock['Williams %R']}, "
                   f"OBV: {stock['OBV']}, 지지선 확인: {stock['Support Condition']}, "
                   f"OBV 세력 확인: {stock['OBV Strength Condition']}\n")

    prompt += "각 종목에 대한 간단한 분석도 포함해 주세요."

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # 사용할 모델 설정
            messages=[
                {"role": "system", "content": "이 시스템은 최고의 주식 분석 시스템입니다."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300  # 응답의 최대 토큰 수
        )
        return response['choices'][0]['message']['content']  # 응답 내용 반환
    except Exception as e:
        logging.error(f"API 호출 중 오류 발생: {e}")
        return None

# 메인 함수
def main():
    # JSON 파일에서 결과 읽기
    filename = 'results.json'
    if not os.path.exists(filename):
        logging.error(f"{filename} 파일이 존재하지 않습니다.")
        return

    with open(filename, 'r') as f:
        results = json.load(f)

    # results가 문자열인지 확인하고 리스트로 변환
    if isinstance(results, str):
        try:
            results = json.loads(results)  # 문자열을 JSON으로 변환
        except json.JSONDecodeError:
            logging.error("결과가 올바른 JSON 형식이 아닙니다.")
            return

    # results가 리스트인지 확인
    if not isinstance(results, list):
        logging.error("결과가 리스트 형식이 아닙니다.")
        return

    # AI 분석 결과 생성
    ai_response = generate_ai_response(results)

    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")  # 환경 변수에서 웹훅 URL을 가져옴
    if webhook_url:
        message = f"조건을 만족하는 종목 리스트: {results}\nAI 분석 결과: {ai_response}"
        
        # 메시지 확인 로그
        logging.info("전송할 메시지: ")
        logging.info(message)
        
        send_to_discord_webhook(webhook_url, message)  # 웹훅으로 결과 전송
    else:
        logging.error("웹훅 URL이 설정되어 있지 않습니다.")

if __name__ == "__main__":
    main()
