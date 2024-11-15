import openai

# OpenAI API 키 설정
openai.api_key = 'YOUR_API_KEY'  # 여기에 API 키를 입력하세요

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

def check_conditions_with_ai(ticker, name, data):
    """AI를 통해 조건을 확인하는 함수"""
    message = (
        f"Does the stock {name} (ticker: {ticker}) meet the following conditions?\n"
        f"1. There should be a large bullish candle near the upper limit (30%).\n"
        f"2. MACD should drop below 5 after a large bullish candle.\n"
        f"3. William's R should be below 0.\n"
        f"4. The stock price should be above the previous low before the upper limit.\n"
        f"Here is the data: {data}"
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": message}]
    )

    return response['choices'][0]['message']['content'].lower() == 'yes'

def find_matching_stocks():
    """조건을 충족하는 종목을 찾는 함수"""
    stock_data = get_stock_list()
    results = {'KOSPI': [], 'KOSDAQ': []}

    # AI에게 종목 데이터 요청
    data = stock_data  # AI가 반환한 종목 데이터

    # AI에게 각 종목에 대해 조건 확인
    for market in ['KOSPI', 'KOSDAQ']:
        for stock in data[market]:
            ticker = stock['code']
            name = stock['name']
            if check_conditions_with_ai(ticker, name, data):
                results[market].append(name)

    return results

if __name__ == "__main__":
    matching_stocks = find_matching_stocks()
    
    # 결과 출력
    print("Matching stocks:")
    for market, names in matching_stocks.items():
        print(f"{market}: {names}")
