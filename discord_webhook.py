import json
import os

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
        # 결과의 첫 번째 항목을 출력하여 키를 확인합니다.
        print("첫 번째 항목:", results[0])
        # 모든 항목의 키를 출력합니다.
        for item in results:
            print("항목 키:", item.keys())
    else:
        print("발견된 패턴이 없습니다.")
