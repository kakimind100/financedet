def prepare_data(df):
    """데이터를 준비하는 함수 (최근 60일 데이터를 학습, 향후 26일 예측)."""
    df = df[['Date', 'Close', 'Change', 'EMA20', 'EMA50', 'RSI', 'MACD', 
              'MACD_Signal', 'Bollinger_High', 'Bollinger_Low', 'Stoch', 
              'Momentum', 'ADX']].dropna().set_index('Date')
    
    df = df.sort_index()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)  # fit()을 학습 데이터에 적용

    x_data, y_data = [], []
    # 최근 60일 데이터를 사용하여 향후 26일 예측
    for i in range(60, len(scaled_data) - 26):
        x_data.append(scaled_data[i-60:i])  # 이전 60일 데이터
        y_data.append(scaled_data[i + 25, 0])  # 26일 후의 종가 (Close)

    x_data, y_data = np.array(x_data), np.array(y_data)
    print(f"준비된 데이터 샘플 수: x_data={len(x_data)}, y_data={len(y_data)}")
    return x_data, y_data.reshape(-1, 1), scaler  # scaler를 반환

def predict_future_prices(model, last_60_days, scaler):
    """모델을 사용하여 향후 26일 가격을 예측하는 함수."""
    predictions = model.predict(last_60_days.reshape(1, -1))
    print(f"예측된 미래 가격: {predictions}")
    
    # 예측된 가격을 원래 스케일로 복원
    future_prices = scaler.inverse_transform(np.hstack((predictions.reshape(-1, 1), 
                                                          np.zeros((predictions.shape[0], 11)))))
    return future_prices

def main():
    # 데이터 로드
    df = fetch_stock_data()

    # 종목별 매수 및 매도 시점 저장
    results = []

    # 종목 리스트
    stock_codes = df['Code'].unique()  # 'Code' 열을 사용하여 종목 코드 가져오기

    for code in stock_codes:
        stock_data = df[df['Code'] == code]  # 각 종목의 데이터만 필터링

        # 현재 가격 가져오기
        current_price = stock_data['Close'].iloc[-1]  # 마지막 날의 종가
        print(f"\n종목 코드: {code}, 현재 가격: {current_price}")

        # 데이터 준비
        try:
            x_data, y_data, scaler = prepare_data(stock_data)
        except KeyError as e:
            print(f"종목 코드 {code}의 데이터 준비 중 오류 발생: {e}")
            continue  # 오류 발생 시 다음 종목으로 넘어감

        # 훈련 및 테스트 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

        # 모델 훈련
        model = create_and_train_model(X_train, y_train)

        # 가장 최근 60일 데이터를 사용하여 향후 26일 가격 예측
        last_60_days = x_data[-1].reshape(1, -1)
        future_prices = predict_future_prices(model, last_60_days, scaler)

        # 매수 및 매도 신호 생성
        buy_index, sell_index = generate_signals(future_prices.flatten())

        # 매수 및 매도 신호가 생성되었는지 확인
        if buy_index < len(future_prices[0]) and sell_index < len(future_prices[0]):
            try:
                # 매수 및 매도 인덱스에 해당하는 모든 데이터 출력
                print(f"종목 코드 {code}의 예측된 가격 배열: {future_prices.flatten()}")
                
                # 매수 및 매도 인덱스에 해당하는 모든 특성 가져오기
                buy_data_index = buy_index + 60  # 60일 전 데이터
                sell_data_index = sell_index + 60  # 60일 전 데이터

                if buy_data_index < len(stock_data) and sell_data_index < len(stock_data):
                    all_data_at_buy_index = stock_data.iloc[buy_data_index]
                    all_data_at_sell_index = stock_data.iloc[sell_data_index]
                    
                    print(f"매수 인덱스 {buy_index}의 모든 데이터: {all_data_at_buy_index.to_dict()}")
                    print(f"매도 인덱스 {sell_index}의 모든 데이터: {all_data_at_sell_index.to_dict()}")

                buy_price = future_prices[0][buy_index]  # 매수 가격
                sell_price = future_prices[0][sell_index]  # 매도 가격
                
                # 매수 가격 확인
                if buy_price < 0:
                    print(f"종목 코드 {code}에서 매수 가격이 음수입니다: {buy_price}.")
                    continue  # 무시할지 여부 결정
                
                gap = sell_index - buy_index  # 매수와 매도 시점의 격차
                results.append((code, gap, buy_price, sell_price, current_price))
                print(f"종목 코드 {code} - 현재 가격: {current_price}, 매수 가격: {buy_price}, 매도 가격: {sell_price}, 격차: {gap}")
            except IndexError as e:
                print(f"종목 코드 {code}에서 매수 또는 매도 가격 접근 오류: {e}")
        else:
            print(f"종목 코드 {code}에서 매수 또는 매도 신호가 유효하지 않습니다.")

    # 격차가 큰 순서로 정렬
    results.sort(key=lambda x: x[1], reverse=True)

    # 결과 출력
    print("\n매수와 매도 시점의 격차가 큰 종목 순서:")
    if results:
        for code, gap, buy_price, sell_price, current_price in results:
            print(f"종목 코드: {code}, 현재 가격: {current_price}, 격차: {gap}, 매수 가격: {buy_price}, 매도 가격: {sell_price}")
    else:
        print("매수와 매도 시점의 격차가 큰 종목이 없습니다.")

# 실행
main()
