import os
import json
import openai
import requests

# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")  # 환경 변수에서 API 키를 가져옴

# 웹훅을 통해 메시지를 디스코드 채널로 보내는 함수
def send_to_discord_webhook(webhook_url, message):
    data = {"content": message}
    response = requests.post(webhook_url, json=data)
    if response.status_code == 204:
        print("메시지가 성공적으로 전송되었습니다.")
    else:
        print(f"메시지 전송 실패: {response.status_code} - {response.text}")

# AI를 사용하여 주식 분석 결과를 생성하는 함수
def generate_ai_response(stock_codes):
    prompt = f"다음 종목 코드에 대해 분석하여 다음 거래일에 가장 많이 오를 것 같은 3개의 종목을 찾아주세요: {stock_codes}"
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # 사용할 모델 설정
            messages=[
                {"role": "system", "content": "이 시스템은 최고의 주식 분석 시스템입니다."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150  # 응답의 최대 토큰 수
        )
        return response['choices'][0]['message']['content']  # 응답 내용 반환
    except Exception as e:
        print(f"API 호출 중 오류 발생: {e}")
        return None

# 메인 함수
def main():
    # JSON 파일에서 결과 읽기
    filename = 'results.json'
    if not os.path.exists(filename):
        print(f"{filename} 파일이 존재하지 않습니다.")
        return

    with open(filename, 'r') as f:
        results = json.load(f)

    # results가 문자열인지 확인하고 리스트로 변환
    if isinstance(results, str):
        try:
            results = json.loads(results)  # 문자열을 JSON으로 변환
        except json.JSONDecodeError:
            print("결과가 올바른 JSON 형식이 아닙니다.")
            return

    # results가 리스트인지 확인
    if not isinstance(results, list):
        print("결과가 리스트 형식이 아닙니다.")
        return

    # AI 분석 결과 생성
    ai_response = generate_ai_response([result['Code'] for result in results if 'Code' in result])

    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")  # 환경 변수에서 웹훅 URL을 가져옴
    if webhook_url:
        message = f"조건을 만족하는 종목 리스트: {results}\nAI 분석 결과: {ai_response}"
        send_to_discord_webhook(webhook_url, message)  # 웹훅으로 결과 전송
    else:
        print("웹훅 URL이 설정되어 있지 않습니다.")

if __name__ == "__main__":
    main()
