import os
import openai
import json
import requests

# JSON 파일에서 결과를 읽어오는 함수
def load_results_from_json(filename='results.json'):
    with open(filename, 'r') as f:
        results = json.load(f)
    return results

# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")  # 환경 변수에서 API 키를 가져옴

# OpenAI API를 호출하는 함수
def generate_response(stock_codes):
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

# 웹훅을 통해 메시지를 디스코드 채널로 보내는 함수
def send_to_discord_webhook(webhook_url, message):
    data = {"content": message}
    response = requests.post(webhook_url, json=data)
    if response.status_code == 204:
        print("메시지가 성공적으로 전송되었습니다.")
    else:
        print(f"메시지 전송 실패: {response.status_code} - {response.text}")

# 메인 함수
def main():
    results = load_results_from_json()  # JSON 파일에서 결과 읽기
    print(f"전송된 종목 리스트: {results}")  # 결과 출력

    # API 호출 및 응답 출력
    response = generate_response(results)
    if response:
        webhook_url = os.getenv("DISCORD_WEBHOOK_URL")  # 환경 변수에서 웹훅 URL을 가져옴
        send_to_discord_webhook(webhook_url, f"OpenAI 응답: {response}")  # 웹훅으로 응답 전송
    else:
        print("주식 분석 요청 중 오류가 발생했습니다.")

if __name__ == "__main__":
    main()
