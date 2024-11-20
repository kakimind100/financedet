import json
import requests
import os
from datetime import datetime, timedelta

def filter_recent_data(data, days=40):
    """최근 n일의 데이터만 필터링하는 함수."""
    cutoff_date = datetime.today() - timedelta(days=days)
    recent_data = []

    for item in data:
        pattern_date = datetime.strptime(item['pattern_date'], '%Y-%m-%d')
        if pattern_date >= cutoff_date:
            recent_data.append(item)

    return recent_data

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

    # 웹훅 호출 부분은 주석 처리했습니다. 필요 시 주석을 해제하세요.
    # response = requests.post(WEBHOOK_URL, json=payload)
    
    # if response.status_code == 204:
    #     print("메시지가 성공적으로 전송되었습니다.")
    # else:
    #     print(f"메시지 전송 실패: {response.status_code}, {response.text}")

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
        recent_results = filter_recent_data(results)  # 최근 40일 데이터 필터링
        if recent_results:
            # 웹훅 전송 부분은 주석 처리
            # send_to_discord(recent_results)  # Discord로 전송
            print("최근 40일 데이터:", recent_results)  # 필터링된 데이터 출력
        else:
            print("최근 40일 내의 발견된 패턴이 없습니다.")
    else:
        print("발견된 패턴이 없습니다.")
