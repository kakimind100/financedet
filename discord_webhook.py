import json
import os
import logging
from datetime import datetime, timedelta
import requests  # requests 라이브러리가 필요합니다.

# 로그 디렉토리 생성
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# 로깅 설정
logging.basicConfig(
    filename=os.path.join(log_dir, 'discord_webhook.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def filter_recent_data(data, days=40):
    """최근 n일의 데이터만 필터링하는 함수."""
    cutoff_date = datetime.today() - timedelta(days=days)
    recent_data = []

    for item in data:
        # 올바른 키를 사용하여 날짜를 가져옵니다.
        pattern_date = datetime.strptime(item['date'], '%Y-%m-%d')  # 'pattern_date' 대신 'date' 사용
        if pattern_date >= cutoff_date:
            recent_data.append(item)

    return recent_data

def log_recent_data(data):
    """최근 데이터를 로그에 기록하는 함수."""
    for item in data:
        logging.info(f"종목 코드: {item['code']}, 완성 날짜: {item['date']}, 유형: {item['type']}")

def send_to_discord(data):
    """Discord 웹훅으로 데이터를 전송하는 함수."""
    WEBHOOK_URL = "YOUR_DISCORD_WEBHOOK_URL"  # 여기에 실제 웹훅 URL을 입력하세요.
    
    embed = {
        "title": "주식 패턴 발견 결과",
        "description": "다음은 발견된 주식 패턴입니다.",
        "color": 5814783,  # 색상 코드 (16진수)
        "fields": []
    }

    for item in data:
        fields = {
            "name": f"종목 코드: {item['code']}",
            "value": f"완성 날짜: {item['date']}, 유형: {item['type']}",
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
    print("Discord 웹훅 스크립트 실행 중...")  # 실행 시작 메시지
    results = load_results()
    if results:
        recent_results = filter_recent_data(results)  # 최근 40일 데이터 필터링
        if recent_results:
            log_recent_data(recent_results)  # 최근 데이터 로그 기록
            send_to_discord(recent_results)  # Discord로 전송
            print("최근 40일 데이터:", recent_results)  # 필터링된 데이터 출력
        else:
            print("최근 40일 내의 발견된 패턴이 없습니다.")
    else:
        print("발견된 패턴이 없습니다.")
