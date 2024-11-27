import pandas as pd
import joblib
import logging
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

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
            'Open': 'float64',
            'High': 'float64',
            'Low': 'float64',
            'Close': 'float64',
            'Volume': 'float64',
            'Change': 'float64',
            'MA5': 'float64',
            'MA20': 'float64',
            'MACD': 'float64',
            'MACD_Signal': 'float64',
            'Bollinger_High': 'float64',
            'Bollinger_Low': 'float64',
            'Stoch': 'float64',
            'RSI': 'float64',
            'ATR': 'float64',
            'CCI': 'float64',
            'EMA20': 'float64',
            'EMA50': 'float64',
            'Momentum': 'float64',
            'Williams %R': 'float64',
            'ADX': 'float64',
            'Volume_MA20': 'float64',
            'ROC': 'float64',
            'CMF': 'float64',
            'OBV': 'float64'
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
    # 오늘 종가 기준으로 29% 이상 상승 여부를 타겟으로 설정
    df['Target'] = np.where(df['Close'].shift(-1) >= df['Close'] * 1.29, 1, 0)  # 다음 날 종가 기준
    
    # NaN 제거
    df.dropna(subset=['Target'], inplace=True)

    # 기술적 지표를 피쳐로 사용
    features = ['MA5', 'MA20', 'RSI', 'MACD', 'Bollinger_High', 'Bollinger_Low', 
                'Stoch', 'ATR', 'CCI', 'EMA20', 'EMA50', 'Momentum', 
                'Williams %R', 'ADX', 'Volume_MA20', 'ROC', 'CMF', 'OBV']

    # NaN 제거
    df.dropna(subset=features + ['Target'], inplace=True)

    # 훈련 데이터를 위한 리스트
    X = []
    y = []
    stock_codes = []  # 종목 코드를 저장할 리스트 추가

    # 종목 코드별로 최근 5일 데이터 확인
    for stock_code in df['Code'].unique():
        stock_data = df[df['Code'] == stock_code].tail(26)  # 최근 26일 데이터
        
        if len(stock_data) == 26:  # 최근 26일 데이터가 있는 경우
            X.append(stock_data[features].values.flatten())  # 26일의 피쳐를 1D 배열로 변환
            y.append(stock_data['Target'].values[-1])  # 마지막 날의 타겟 값
            stock_codes.append(stock_code)  # 종목 코드 추가

    X = np.array(X)
    y = np.array(y)

    # 데이터 분할
    X_train, X_temp, y_train, y_temp, stock_codes_train, stock_codes_temp = train_test_split(
        X, y, stock_codes, test_size=0.3, random_state=42
    )

    # 검증 및 테스트 세트 분할
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
    features = ['MA5', 'MA20', 'RSI', 'MACD', 'Bollinger_High', 'Bollinger_Low', 
                'Stoch', 'ATR', 'CCI', 'EMA20', 'EMA50', 'Momentum', 
                'Williams %R', 'ADX', 'Volume_MA20', 'ROC', 'CMF', 'OBV']
    predictions = []

    # 테스트 데이터와 예측 데이터의 중복 체크
    overlapping_stocks = today_rise_stocks['Code'].unique()
    common_stocks = set(stock_codes_test).intersection(set(overlapping_stocks))
    
    if common_stocks:
        logging.warning(f"예측 데이터와 테스트 데이터가 겹치는 종목: {common_stocks}")

    # 최근 5거래일 데이터를 사용하여 예측하기
    for stock_code in today_rise_stocks['Code'].unique():
        if stock_code in stock_codes_test:  # 테스트 데이터에 포함된 종목만 예측
            recent_data = df[df['Code'] == stock_code].tail(26)  # 마지막 26일 데이터 가져오기
            if not recent_data.empty:  # 데이터가 비어있지 않은 경우
                # 최근 26일 데이터를 사용하여 예측
                X_next = recent_data[features].values.flatten().reshape(1, -1)  # 26일 데이터로 2D 배열로 변환
                pred = model.predict(X_next)

                # 예측 결과와 함께 정보를 저장
                predictions.append({
                    'Code': stock_code,
                    'Prediction': pred[0],
                    'MA5': recent_data['MA5'].values[-1],
                    'MA20': recent_data['MA20'].values[-1],
                    'RSI': recent_data['RSI'].values[-1],
                    'MACD': recent_data['MACD'].values[-1],
                    'Bollinger_High': recent_data['Bollinger_High'].values[-1],
                    'Bollinger_Low': recent_data['Bollinger_Low'].values[-1],
                    'Stoch': recent_data['Stoch'].values[-1],
                    'ATR': recent_data['ATR'].values[-1],
                    'CCI': recent_data['CCI'].values[-1],
                    'EMA20': recent_data['EMA20'].values[-1],
                    'EMA50': recent_data['EMA50'].values[-1],
                    'Momentum': recent_data['Momentum'].values[-1],
                    'Williams %R': recent_data['Williams %R'].values[-1],
                    'ADX': recent_data['ADX'].values[-1],
                    'Volume_MA20': recent_data['Volume_MA20'].values[-1],
                    'ROC': recent_data['ROC'].values[-1],
                    'CMF': recent_data['CMF'].values[-1],
                    'OBV': recent_data['OBV'].values[-1]
                })

    # 예측 결과를 데이터프레임으로 변환
    predictions_df = pd.DataFrame(predictions)

    # 29% 상승할 것으로 예측된 종목 필터링
    top_predictions = predictions_df[predictions_df['Prediction'] == 1]

    # 상위 20개 종목 정렬 (MA5, MACD, RSI, Stoch 기준으로 정렬)
    top_predictions = top_predictions.sort_values(
        by=['MA5', 'MACD', 'RSI', 'Stoch'], 
        ascending=[False, False, True, True]  # MA5와 MACD는 내림차순, RSI와 Stoch은 오름차순으로 정렬
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
