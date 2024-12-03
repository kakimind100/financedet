import pandas as pd
import joblib
import logging
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE  # SMOTE 임포트 추가

# 로그 디렉토리 설정
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# 로깅 설정
logging.basicConfig(
    filename=os.path.join(log_dir, 'stock_data_fetcher.log'),
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 콘솔 로그 출력 설정
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)

def fetch_stock_data():
    """주식 데이터를 가져오는 함수 (CSV 파일에서)."""
    logging.debug("주식 데이터를 가져오는 중...")
    try:
        file_path = os.path.join('data', 'stock_data_with_indicators.csv')
        
        dtype = {
            'Code': 'object',
            'Date': 'str',
            'Open': 'float',
            'High': 'float',
            'Low': 'float',
            'Close': 'float',
            'Volume': 'float',
            'Change': 'float',
            'MA5': 'float',
            'MA20': 'float',
            'MACD': 'float',
            'MACD_Signal': 'float',
            'Bollinger_High': 'float',
            'Bollinger_Low': 'float',
            'Stoch': 'float',
            'RSI': 'float',
            'ATR': 'float',
            'CCI': 'float',
            'EMA20': 'float',
            'EMA50': 'float',
            'Momentum': 'float',
            'Williams %R': 'float',
            'ADX': 'float',
            'Volume_MA20': 'float',
            'ROC': 'float',
            'CMF': 'float',
            'OBV': 'float'
            'Anomaly': 'int'
        }

        df = pd.read_csv(file_path, dtype=dtype)
        logging.info(f"주식 데이터를 '{file_path}'에서 성공적으로 가져왔습니다.")

        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        return df
    except Exception as e:
        logging.error(f"주식 데이터 가져오기 중 오류 발생: {e}")
        return None

def prepare_data(df):
    """데이터를 준비하고 분할하는 함수."""
    # 매수 중심으로 우선순위를 조정한 기술적 지표 리스트
    features = [
        'RSI',                  # 과매도 상태를 나타내는 지표
        'MACD',                 # 추세 반전을 나타내는 지표
        'Stoch',                # 과매수/과매도 신호를 나타내는 지표
        'Bollinger_High',       # 가격의 상한선을 나타내는 지표
        'Bollinger_Low',        # 가격의 하한선을 나타내는 지표
        'MA5',                  # 단기 이동 평균
        'MA20',                 # 중기 이동 평균
        'EMA20',                # 지수 이동 평균
        'EMA50',                # 지수 이동 평균
        'CCI',                  # 가격의 과매수/과매도 상태를 나타내는 지표
        'ATR',                  # 변동성 지표
        'Momentum',             # 가격 변화의 속도를 나타내는 지표
        'ADX',                  # 추세의 강도를 나타내는 지표
        'Williams %R',          # 과매수/과매도 신호를 나타내는 지표
        'Volume_MA20',          # 거래량의 이동 평균
        'ROC',                  # 가격 변화율
        'CMF',                  # 자금 흐름 지표
        'OBV',                  # 거래량 기반의 지표
        'Anomaly'               # Anomaly 추가
    ]

    X = []
    y = []
    stock_codes = []

    for stock_code in df['Code'].unique():
        stock_data = df[df['Code'] == stock_code].tail(11)

        if len(stock_data) == 11:
            open_price = stock_data['Open'].iloc[-1]
            low_price = stock_data['Low'].min()
            high_price = stock_data['High'].max()

            # 타겟 설정: 오늘 최저가에서 최고가가 29% 이상 상승했는지 여부
            target_today = 1 if high_price > low_price * 1.29 else 0

            # 마지막 날의 피처와 타겟을 함께 추가
            X.append(stock_data[features].values[-1])  # 마지막 날의 피처 사용
            y.append(target_today)  # 오늘의 타겟 값 사용
            stock_codes.append(stock_code)  # 종목 코드 추가

    X = np.array(X)
    y = np.array(y)

    # 클래스 분포 확인
    logging.info(f"타겟 클래스 분포: {np.bincount(y)}")

    # SMOTE 적용
    if len(np.unique(y)) > 1:  # 클래스가 2개 이상인 경우에만 SMOTE 적용
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # stock_codes에 대한 재조정
        # SMOTE는 X, y에 대해서만 작동하므로, stock_codes는 원래 데이터에 기반하여 다시 설정
        stock_codes_resampled = []
        for i in range(len(y_resampled)):
            stock_codes_resampled.append(stock_codes[i % len(stock_codes)])  # 다시 원본 stock_codes에서 순환

    else:
        logging.warning("타겟 클래스가 1개만 존재합니다. SMOTE를 적용하지 않습니다.")
        X_resampled, y_resampled = X, y  # 원본 데이터 유지
        stock_codes_resampled = stock_codes  # 원본 stock_codes 유지

    # 데이터 분할
    X_train, X_temp, y_train, y_temp, stock_codes_train, stock_codes_temp = train_test_split(
        X_resampled, y_resampled, stock_codes_resampled, test_size=0.3, random_state=42
    )

    logging.info(f"훈련에 사용된 종목 코드: {stock_codes_train}")

    X_valid, X_test, y_valid, y_test, stock_codes_valid, stock_codes_test = train_test_split(
        X_temp, y_temp, stock_codes_temp, test_size=0.5, random_state=42
    )

    return X_train, X_valid, X_test, y_train, y_valid, y_test, stock_codes_train, stock_codes_valid, stock_codes_test

def train_model_with_hyperparameter_tuning():
    """모델을 훈련시키고 하이퍼파라미터를 튜닝하는 함수."""
    df = fetch_stock_data()  # 주식 데이터 가져오기
    if df is None:
        logging.error("데이터프레임이 None입니다. 모델 훈련을 중단합니다.")
        return None, None  # None 반환

    # 데이터 준비 및 분할
    X_train, X_valid, X_test, y_train, y_valid, y_test, stock_codes_train, stock_codes_valid, stock_codes_test = prepare_data(df)

    # 하이퍼파라미터 튜닝을 위한 GridSearchCV 설정
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               scoring='accuracy', cv=3, verbose=2, n_jobs=-1)
    
    grid_search.fit(X_train, y_train)  # 하이퍼파라미터 튜닝
    
    # 최적의 하이퍼파라미터 출력
    logging.info(f"최적의 하이퍼파라미터: {grid_search.best_params_}")
    print(f"최적의 하이퍼파라미터: {grid_search.best_params_}")

    # 최적의 모델로 재훈련
    best_model = grid_search.best_estimator_

    # 모델 평가
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred)
    logging.info(f"모델 성능 보고서:\n{report}")
    print(report)

    # 테스트 세트 종목 코드 로깅
    logging.info(f"테스트 세트 종목 코드: {stock_codes_test}")

    return best_model, stock_codes_test  # 최적 모델과 테스트 종목 코드 반환

def predict_next_day(model, stock_codes_test):
    """다음 거래일의 상승 여부를 예측하는 함수."""
    df = fetch_stock_data()  # 주식 데이터 가져오기
    if df is None:
        logging.error("데이터프레임이 None입니다. 예측을 중단합니다.")
        return

    # 오늘 종가가 29% 이상 상승한 종목 필터링
    today_rise_stocks = df[df['Close'] >= df['Open'] * 1.29]

    # 예측할 데이터 준비 (모든 기술적 지표 포함)
    features = [
        'RSI',                  # 과매도 상태를 나타내는 지표
        'MACD',                 # 추세 반전을 나타내는 지표
        'Stoch',                # 과매수/과매도 신호를 나타내는 지표
        'Bollinger_High',       # 가격의 상한선을 나타내는 지표
        'Bollinger_Low',        # 가격의 하한선을 나타내는 지표
        'MA5',                  # 단기 이동 평균
        'MA20',                 # 중기 이동 평균
        'EMA20',                # 지수 이동 평균
        'EMA50',                # 지수 이동 평균
        'CCI',                  # 가격의 과매수/과매도 상태를 나타내는 지표
        'ATR',                  # 변동성 지표
        'Momentum',             # 가격 변화의 속도를 나타내는 지표
        'ADX',                  # 추세의 강도를 나타내는 지표
        'Williams %R',          # 과매수/과매도 신호를 나타내는 지표
        'Volume_MA20',          # 거래량의 이동 평균
        'ROC',                  # 가격 변화율
        'CMF',                  # 자금 흐름 지표
        'OBV',                  # 거래량 기반의 지표
        'Anomaly'               # Anomaly 추가
    ]

    predictions = []  # 예측 결과를 저장할 리스트

    # 테스트 데이터와 예측 데이터의 중복 체크
    overlapping_stocks = today_rise_stocks['Code'].unique()
    common_stocks = set(stock_codes_test).intersection(set(overlapping_stocks))
    
    if common_stocks:
        logging.warning(f"예측 데이터와 테스트 데이터가 겹치는 종목: {common_stocks}")

    # 최근 10거래일 데이터를 사용하여 예측하기
    for stock_code in today_rise_stocks['Code'].unique():
        if stock_code in stock_codes_test:  # 테스트 데이터에 포함된 종목만 예측
            recent_data = df[df['Code'] == stock_code].tail(10)  # 마지막 10일 데이터 가져오기
            if not recent_data.empty and len(recent_data) == 10:  # 데이터가 비어있지 않고 10일인 경우
                # 최근 10일 데이터를 사용하여 예측
                X_next = recent_data[features].values[-1].reshape(1, -1)  # 마지막 날의 피처로 2D 배열로 변환
                logging.debug(f"예측할 데이터 X_next: {X_next}")

                # 예측
                pred = model.predict(X_next)

                # 예측 결과와 함께 정보를 저장
                predictions.append({
                    'Code': stock_code,
                    'Prediction': pred[0],
                    # 피처 값도 저장할 수 있음
                    **recent_data[features].iloc[-1].to_dict()  # 마지막 날의 피처 값 추가
                })

    # 예측 결과를 데이터프레임으로 변환
    predictions_df = pd.DataFrame(predictions)

    # 29% 상승할 것으로 예측된 종목 필터링
    top_predictions = predictions_df[predictions_df['Prediction'] == 1]

    # 상위 20개 종목 정렬 (기본 피처와 추가 피처 기준으로 정렬)
    top_predictions = top_predictions.sort_values(
        by=['RSI', 'MACD', 'Stoch', 'Momentum', 'Volume_MA20', 
            'Bollinger_Low', 'ATR', 'CMF', 'OBV', 
            'CCI', 'ADX', 'ROC', 'MA5', 'MA20'], 
        ascending=[True, False, True, False, False, 
                   True, True, True, False, 
                   True, True, True, True, True]
    ).head(20)

    # 예측 결과 출력
    print("다음 거래일에 29% 상승할 것으로 예측되는 상위 20개 종목:")
    for index, row in top_predictions.iterrows():
        print(f"{row['Code']} (MA5: {row['MA5']}, MA20: {row['MA20']}, RSI: {row['RSI']}, "
              f"MACD: {row['MACD']}, Bollinger_High: {row['Bollinger_High']}, "
              f"Bollinger_Low: {row['Bollinger_Low']}, Stoch: {row['Stoch']}, "
              f"ATR: {row['ATR']}, CCI: {row['CCI']}, EMA20: {row['EMA20']}, "
              f"EMA50: {row['EMA50']}, Momentum: {row['Momentum']}, "
              f"Williams %R: {row['Williams %R']}, ADX: {row['ADX']}, "
              f"Volume_MA20: {row['Volume_MA20']}, ROC: {row['ROC']}, "
              f"CMF: {row['CMF']}, OBV: {row['OBV']})")

    # 상위 20개 종목의 전체 날짜 데이터 추출
    all_data_with_top_stocks = df[df['Code'].isin(top_predictions['Code'])]

    # 예측 결과를 data/top_20_stocks_all_dates.csv 파일로 저장
    output_file_path = os.path.join('data', 'top_20_stocks_all_dates.csv')
    all_data_with_top_stocks.to_csv(output_file_path, index=False, encoding='utf-8-sig')  # CSV 파일로 저장
    logging.info(f"상위 20개 종목의 전체 데이터가 '{output_file_path}'에 저장되었습니다.")

if __name__ == "__main__":
    logging.info("모델 훈련 스크립트 실행 중...")
    model, stock_codes_test = train_model_with_hyperparameter_tuning()  # 하이퍼파라미터 튜닝 모델 훈련
    logging.info("모델 훈련 스크립트 실행 완료.")

    if model is not None and stock_codes_test is not None:  # 모델과 테스트 데이터가 있는 경우에만 예측 실행
        logging.info("다음 거래일 예측 스크립트 실행 중...")
        predict_next_day(model, stock_codes_test)  # 다음 거래일 예측
        logging.info("다음 거래일 예측 스크립트 실행 완료.")
    else:
        logging.error("모델 훈련에 실패했습니다. 예측을 수행할 수 없습니다.")
