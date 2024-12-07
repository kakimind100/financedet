import numpy as np
import pandas as pd
from joblib import Parallel, delayed

def process_stock_data(predictions, start_date, code):
    """종목별로 매수 및 매도 신호를 처리하는 함수."""
    try:
        # 매수 및 매도 신호 생성
        buy_index, sell_index, buy_date, sell_date = generate_signals(predictions, start_date)
        
        if buy_index is None or sell_index is None:
            print(f"종목 코드 {code}: 신호를 생성하지 못했습니다. 건너뜁니다.")
            return None
        
        # 결과 반환
        return {
            "code": code,
            "buy_index": buy_index,
            "sell_index": sell_index,
            "buy_date": buy_date,
            "sell_date": sell_date
        }

    except Exception as e:
        print(f"종목 코드 {code} 처리 중 오류 발생: {e}")
        return None

def generate_signals(predictions, start_date):
    """예측 결과를 기반으로 매수 및 매도 신호를 생성하는 함수."""
    try:
        # 매수 시점 찾기
        buy_index = np.argmin(predictions)
        
        # 매수 이후의 예측 데이터를 확인
        remaining_predictions = predictions[buy_index + 1:]
        if len(remaining_predictions) == 0:
            raise ValueError("매수 이후 예측 데이터가 비어 있습니다.")

        # 매도 시점 찾기
        sell_index = np.argmax(remaining_predictions) + buy_index + 1

        # 날짜 계산
        buy_date = start_date + pd.Timedelta(days=buy_index)
        sell_date = start_date + pd.Timedelta(days=sell_index)

        print(f"매수 신호 날짜: {buy_date}, 매도 신호 날짜: {sell_date}")
        return buy_index, sell_index, buy_date, sell_date

    except Exception as e:
        print(f"신호 생성 중 오류 발생: {e}")
        return None, None, None, None

def main():
    """메인 함수: 전체 종목 데이터를 처리."""
    # 샘플 데이터 생성 (사용 시 실제 데이터를 연결해야 함)
    num_stocks = 10  # 종목 수
    prediction_length = 30  # 예측 길이
    start_date = pd.Timestamp("2023-01-01")  # 데이터 시작 날짜
    codes = [f"Stock_{i}" for i in range(num_stocks)]  # 종목 코드
    
    # 랜덤 예측 데이터 생성
    predictions = {code: np.random.rand(prediction_length) for code in codes}

    # 병렬 처리를 통해 종목별로 데이터 처리
    results = Parallel(n_jobs=10, backend="loky")(
        delayed(process_stock_data)(predictions[code], start_date, code) for code in codes
    )

    # 유효한 결과만 필터링
    valid_results = [result for result in results if result is not None]

    # 결과 출력
    for result in valid_results:
        print(f"종목 코드: {result['code']}, 매수 날짜: {result['buy_date']}, 매도 날짜: {result['sell_date']}")

if __name__ == "__main__":
    main()
