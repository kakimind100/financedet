import openai
import os

# OpenAI API 키 설정
openai.api_key = os.getenv('OPENAI_API_KEY')  # 환경 변수에서 API 키를 가져옵니다.

def find_stocks_meeting_conditions():
    """AI에게 조건을 충족하는 종목을 요청하는 함수"""
    conditions = (
        "Please find stocks that meet the following conditions:\n"
        "1. There should be a large bullish candle near the upper limit (30%).\n"
        "2. MACD should drop below 5 after a large bullish candle.\n"
        "3. William's R should be below 0.\n"
        "4. The stock price should be above the previous low before the upper limit."
    )
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": conditions
            }
        ]
    )
    
    return response['choices'][0]['message']['content']

if __name__ == "__main__":
    # AI에게 조건을 충족하는 종목 데이터 요청 및 결과 출력
    matching_stocks = find_stocks_meeting_conditions()
    print("Stocks meeting the conditions:")
    print(matching_stocks)
