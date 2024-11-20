import json
import os

def load_results():
    """저장된 JSON 파일에서 결과를 로드하는 함수."""
    filename = os.path.join('json_results', 'pattern_results.json')
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("파일을 찾을 수 없습니다. 경로를 확인하세요.")
        return []
    except json.JSONDecodeError:
        print("JSON 파일을 읽는 중 오류가 발생했습니다.")
        return []

if __name__ == "__main__":
    results = load_results()
    if results:
        print("결과를 성공적으로 로드했습니다:", results)
    else:
        print("로드된 결과가 없습니다.")
