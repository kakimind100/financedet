import openai
import os

# OpenAI API 키 설정
openai.api_key = os.getenv('OPENAI_API_KEY')  # 환경 변수에서 API 키를 가져옵니다.

def get_stock_list():
    """KOSPI와 KOSDAQ의 종목 리스트를 가져오는 함수"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "Please provide the stock codes and names for KOSPI and KOSDAQ."
            }
        ]
    )
    return response['choices'][0]['message']['content']

def find_matching_stocks():
    """종목 리스트를 가져오는 함수"""
    stock_data = get_stock_list()
    results = {'KOSPI': [], 'KOSDAQ': []}

    # AI에게 종목 데이터 요청
    data = stock_data  # AI가 반환한 종목 데이터
    
    # KOSPI와 KOSDAQ의 종목 리스트를 결과에 추가
    for market in ['KOSPI', 'KOSDAQ']:
        results[market] = [stock['name'] for stock in data[market]]

    return results

if __name__ == "__main__":
    matching_stocks = find_matching_stocks()
    
    # 결과 출력
    print("Matching stocks:")
    for market, names in matching_stocks.items():
        print(f"{market}: {names}")
