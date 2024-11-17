import json

# JSON 파일에서 결과를 읽어오는 함수
def load_results_from_json(filename='results.json'):
    with open(filename, 'r') as f:
        results = json.load(f)
    return results

# 메인 실행 블록
if __name__ == "__main__":
    results = load_results_from_json()  # JSON 파일에서 결과 읽기
    print(f"전송된 종목 리스트: {results}")  # 결과 출력
