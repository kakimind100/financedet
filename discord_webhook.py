import json
import os
from datetime import datetime, timedelta

def load_results():
    """저장된 JSON 파일에서 결과를 로드하는 함수."""
    filename = os.path.join('json_results', 'pattern_results.json')
    if not os.path.exists(filename):
        print("결과 파일이 존재하지 않습니다.")
        return []

    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def filter_results(results):
    """결과에서 40일 이전의 데이터 제외하고 날짜 추가하는 함수."""
    filtered_results = []
    today = datetime.today()

    for item in results:
        code = item['code']
        data = item['data']
        
        # 40일 이전의 날짜 계산
        cutoff_date = today - timedelta(days=40)

        # 필터링된 데이터 생성
        filtered_data = [entry for entry in data if pd.to_datetime(entry['Date']) > cutoff_date]

        if filtered_data:
            filtered_results.append({
                'code': code,
                'data': filtered_data
            })

    return filtered_results

def save_results(filtered_results):
    """필터링된 결과를 JSON 파일로 저장하는 함수."""
    filename = os.path.join('json_results', 'filtered_pattern_results.json')
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(filtered_results, f, ensure_ascii=False, indent=4)
    print(f"필터링된 결과를 JSON 파일로 저장했습니다: {filename}")

def log_results(filtered_results):
    """저장된 결과를 로그로 출력하는 함수."""
    if filtered_results:
        for item in filtered_results:
            print(f"종목 코드: {item['code']}")
            for entry in item['data']:
                print(f"  날짜: {entry['Date']}, 종가: {entry['Close']}, RSI: {entry['RSI']}")
    else:
        print("발견된 패턴이 없습니다.")

if __name__ == "__main__":
    results = load_results()
    if results:
        # 결과 필터링
        filtered_results = filter_results(results)
        
        # 필터링된 결과 저장
        save_results(filtered_results)

        # 저장된 결과 로그 출력
        log_results(filtered_results)
    else:
        print("발견된 패턴이 없습니다.")
