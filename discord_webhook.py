import json
import requests
import os

# Discord 웹훅 URL 설정
WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL')  # 환경 변수에서 웹훅 URL 가져오기

def send_to_discord(data):
    """Discord 웹훅으로 데이터를 전송하는 함수."""
    embed = {
        "title": "주식 패턴 발견 결과",
        "description": "다음은 발견된 주식 패턴입니다.",
        "color": 5814783,  # 색상 코드 (16진수)
        "fields": []
    }

    for item in data:
        fields = {
            "name": f"종목 코드: {item['code']}",
            "value": f"완성 날짜: {item['pattern_date']}, 유형: {item['type']}",
            "inline": False
        }
        embed["fields"].append(fields)

    payload = {
        "embeds": [embed]
    }

    response = requests.post(WEBHOOK_URL, json=payload)
    
    if response.status_code == 204:
        print("메시지가 성공적으로 전송되었습니다.")
    else:
        print(f"메시지 전송 실패: {response.status_code}, {response.text}")

def load_results():
    """저장된 JSON 파일에서 결과를 로드하는 함수."""
    filename = os.path.join('json_results', 'pattern_results.json')
    if not os.path.exists(filename):
        print("결과 파일이 존재하지 않습니다.")
        return []

    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

if __name__ == "__main__":
    results = load_results()
    if results:
        send_to_discord(results)  # Discord로 전송
    else:
        print("발견된 패턴이 없습니다.")
