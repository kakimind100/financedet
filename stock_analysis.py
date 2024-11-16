def process_stock(code, start_date):
    """주식 데이터를 처리하는 함수."""
    logging.info(f"{code} 처리 시작")
    try:
        df = fdr.DataReader(code, start=start_date)
        logging.info(f"{code} 데이터 가져오기 성공, 가져온 데이터 길이: {len(df)}")
        
        # 데이터 길이 체크: 최소 26일 데이터
        if len(df) < 26:
            logging.warning(f"{code} 데이터가 26일 미만으로 건너뜁니다.")
            return None
        
        # 최근 30일 데이터 추출
        recent_data = df.iloc[-30:]  # 최근 30일 데이터
        last_close = recent_data['Close'].iloc[-1]  # 최근 종가
        prev_close = recent_data['Close'].iloc[-2]  # 이전 종가
        
        # 최근 20일 데이터 추출
        recent_20_days = recent_data.iloc[-20:]

        # 최고가가 29% 이상 상승한 종목 확인
        high_condition = recent_20_days['High'].max() >= recent_20_days['High'].iloc[0] * 1.29

        # 장대 양봉 조건 체크
        last_candle = recent_data.iloc[-1]
        previous_candle = recent_data.iloc[-2]

        # 장대 양봉 여부
        is_bullish_engulfing = (last_candle['Close'] > previous_candle['Close']) and \
                               ((last_candle['Close'] - last_candle['Open']) > (previous_candle['Close'] - previous_candle['Open'])) and \
                               (last_candle['Low'] < previous_candle['Close'])

        # 장대 양봉의 저가 체크
        if high_condition and is_bullish_engulfing:
            # 장대 양봉의 저가 아래로 떨어진 경우 체크
            if recent_data['Low'].iloc[-1] < last_candle['Low']:
                logging.info(f"{code} 장대 양봉의 저가 아래로 떨어져 제외됨.")
                return None

            logging.info(f"{code} 최근 20일 내 최고가 29% 이상 상승한 종목 발견: 최근 종가 {last_close}, 이전 종가 {prev_close}")
            
            # 지표 계산
            df = calculate_indicators(df)  # 윌리엄스 %R 계산

            # 윌리엄스 %R 조건 확인
            if df['williams_r'].iloc[-1] <= -90:
                result = {
                    'Code': code,
                    'Last Close': last_close,
                    'Williams %R': df['williams_r'].iloc[-1]
                }
                logging.info(f"{code} 조건 만족: {result}")
                return result
            else:
                logging.info(f"{code} 윌리엄스 %R 조건 불만족: Williams %R={df['williams_r'].iloc[-1]}")

        return None
    except Exception as e:
        logging.error(f"{code} 처리 중 오류 발생: {e}")
        return None
