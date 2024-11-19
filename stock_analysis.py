def send_stock_analysis_to_ai(stock_data):
    """AI에게 주식 데이터 분석을 요청하는 함수."""
    # stock_data에서 필요한 정보를 문자열로 변환
    stock_info = ""
    total_stocks = len(stock_data)  # 전체 종목 수
    for index, (code, data) in enumerate(stock_data.items()):
        # 최근 6개월간의 데이터에서 요약 정보 계산
        closing_prices = [record['Close'] for record in data]
        volumes = [record['Volume'] for record in data]

        # 요약 정보 계산
        average_close = sum(closing_prices) / len(closing_prices) if closing_prices else 0
        max_close = max(closing_prices) if closing_prices else 0
        min_close = min(closing_prices) if closing_prices else 0
        total_volume = sum(volumes)

        # 특정 날짜의 종가 (예: 최근 5일)
        recent_dates = [record['Date'] for record in data[-5:]]
        recent_closes = closing_prices[-5:]

        # 요약 정보 추가
        stock_info += (
            f"{code}: "
            f"평균 종가: {average_close:.2f}, "
            f"최고가: {max_close:.2f}, "
            f"최저가: {min_close:.2f}, "
            f"총 거래량: {total_volume}, "
            f"최근 5일 종가: {', '.join([f'{date}: {close:.2f}' for date, close in zip(recent_dates, recent_closes)])}\n"
        )

        # 진행 상황 표시
        progress = (index + 1) / total_stocks * 100
        print(f"진행 상황: {progress:.2f}% ({index + 1}/{total_stocks})")

    # 요청 메시지 구성 (for 루프 밖)
    analysis_request = (
        f"다음은 최근 6개월간의 주식 데이터 요약입니다:\n{stock_info}\n"
        "주식의 상승 가능성을 %로 표기하고, 상위 5개 종목의 이유를 20자 내외로 작성해 주세요."
    )
    
    # AI에게 요청 (for 루프 밖)
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "당신은 최고의 주식 전문 투자자입니다."},
            {"role": "user", "content": analysis_request}
        ],
        max_tokens=150  # 최대 토큰 수 설정
    )
    return response['choices'][0]['message']['content']
