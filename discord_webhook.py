import logging
import pandas as pd

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def search_patterns_and_find_top(stocks_data):
    """각 패턴을 탐지하고, 모든 데이터가 저장된 후 가장 좋은 상태의 종목 50개를 찾는 함수."""
    results = []

    for code, data in stocks_data.items():
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.set_index('Date', inplace=True)
        df['Code'] = code

        if df.index.empty:
            logging.warning(f"종목 코드: {code}에 유효한 날짜 데이터가 없습니다.")
            continue

        # 각 패턴 탐지
        is_cup, cup_date = is_cup_with_handle(df)
        is_divergence, divergence_date = is_bullish_divergence(df)
        is_round_bottom_found, round_bottom_date = is_round_bottom(df)

        # 패턴 결과 저장
        pattern_info = {
            'code': code,
            'cup': is_cup,
            'divergence': is_divergence,
            'round_bottom': is_round_bottom_found,
            'data': df.to_dict(orient='records')
        }

        results.append(pattern_info)

    # 모든 패턴이 발견된 종목 필터링
    all_patterns_found = [res for res in results if res['cup'] and res['divergence'] and res['round_bottom']]

    # 종목 평가 및 점수 계산
    for item in all_patterns_found:
        score = evaluate_stock(item['data'])
        item['score'] = score

    # 점수 기준으로 정렬하고 상위 50개 선택
    top_50_stocks = sorted(all_patterns_found, key=lambda x: x['score'], reverse=True)[:50]

    # 상위 50개 종목 데이터 출력
    logging.info("상위 50개 종목:")
    for stock in top_50_stocks:
        logging.info(f"종목 코드: {stock['code']}, 점수: {stock['score']}, 컵과 핸들: {stock['cup']}, "
                     f"다이버전스: {stock['divergence']}, 원형 바닥: {stock['round_bottom']}")

    return top_50_stocks

# 메인 실행 블록
if __name__ == "__main__":
    logging.info("주식 분석 스크립트 실행 중...")

    # 주식 데이터 예시 (여기에 실제 데이터를 넣어야 함)
    stocks_data = {
        '038530': [{'Date': '2024-11-01', 'Close': 1000, 'Volume': 1000},
                    {'Date': '2024-11-02', 'Close': 1020, 'Volume': 1200}],
        '011560': [{'Date': '2024-11-01', 'Close': 2000, 'Volume': 800},
                    {'Date': '2024-11-02', 'Close': 1980, 'Volume': 900}],
        # 추가 종목 데이터...
    }

    top_stocks = search_patterns_and_find_top(stocks_data)

    if top_stocks:
        logging.info("데이터 분석 완료. 상위 종목이 확인되었습니다.")
    else:
        logging.info("모든 패턴을 만족하는 종목이 없습니다.")
