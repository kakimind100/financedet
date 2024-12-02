import pandas as pd
import joblib
import logging
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE  # SMOTE 임포트 추가
from skopt import BayesSearchCV  # Bayesian Optimization을 위한 임포트

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
    features = ['MA5', 'MA20', 'RSI', 'MACD', 'Bollinger_High', 'Bollinger_Low', 
                'Stoch', 'ATR', 'CCI', 'EMA20', 'EMA50', 'Momentum', 
                'Williams %R', 'ADX', 'Volume_MA20', 'ROC', 'CMF', 'OBV']

    X = []
    y = []
    stock_codes = []

    for stock_code in df['Code'].unique():
        stock_data = df[df['Code'] == stock_code].tail(27)

        if len(stock_data) == 27:
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

def score_function(weights, top_predictions):
    """가중치를 사용하여 점수를 계산하는 함수."""
    weights_dict = {
        'MA5': weights[0],
        'MA20': weights[1],
        'MACD': weights[2],
        'RSI': weights[3],
        'ATR': weights[4],
        'OBV': weights[5],
        'Stoch': weights[6],
        'CCI': weights[7],
        'ADX': weights[8]
    }

    # 점수 계산
    top_predictions['Score'] = (
        weights_dict['MA5'] * top_predictions['MA5'] +
        weights_dict['MA20'] * top_predictions['MA20'] +
        weights_dict['MACD'] * top_predictions['MACD'] +
        weights_dict['RSI'] * (100 - top_predictions['RSI']) +
        weights_dict['ATR'] * top_predictions['ATR'] +
        weights_dict['OBV'] * top_predictions['OBV'] +
        weights_dict['Stoch'] * (100 - top_predictions['Stoch']) +  # Stoch도 낮을수록 좋음
        weights_dict['CCI'] * top_predictions['CCI'] +
        weights_dict['ADX'] * top_predictions['ADX']
    )

    # 점수의 평균을 반환
    return top_predictions['Score'].mean()

def optimize_weights(top_predictions):
    """가중치를 최적화하는 함수."""
    from skopt import gp_minimize  # Bayesian Optimization을 위한 임포트

    # 가중치의 범위 설정 (0.0에서 1.0 사이)
    space = [
        (0.0, 1.0),  # MA5
        (0.0, 1.0),  # MA20
        (0.0, 1.0),  # MACD
        (0.0, 1.0),  # RSI
        (0.0, 1.0),  # ATR
        (0.0, 1.0),  # OBV
        (0.0, 1.0),  # Stoch
        (0.0, 1.0),  # CCI
        (0.0, 1.0)   # ADX
    ]

    # 최적화 수행
    res = gp_minimize(lambda w: -score_function(w, top_predictions),  # 점수를 최소화하는 것이므로 부호 반전
                       space,
                       n_calls=50,  # 호출 횟수
                       random_state=42)

    return res.x  # 최적의 가중치 반환

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

    # 최근 26거래일 데이터를 사용하여 예측하기
    for stock_code in today_rise_stocks['Code'].unique():
        if stock_code in stock_codes_test:  # 테스트 데이터에 포함된 종목만 예측
            recent_data = df[df['Code'] == stock_code].tail(26)  # 마지막 26일 데이터 가져오기
            if not recent_data.empty and len(recent_data) == 26:  # 데이터가 비어있지 않고 26일인 경우
                # 최근 26일 데이터를 사용하여 예측
                X_next = recent_data[features].values[-1].reshape(1, -1)  # 마지막 날의 피처로 2D 배열로 변환
                logging.debug(f"예측할 데이터 X_next: {X_next}")

                # 예측
                pred = model.predict(X_next)

                # 예측 결과와 함께 정보를 저장
                predictions.append({
                    'Code': stock_code,
                    'Prediction': pred[0],
                    **recent_data[features].iloc[-1].to_dict()  # 마지막 날의 피처 값 추가
                })

    # 예측 결과를 데이터프레임으로 변환
    predictions_df = pd.DataFrame(predictions)

    # 29% 상승할 것으로 예측된 종목 필터링
    top_predictions = predictions_df[predictions_df['Prediction'] == 1]

    # 최적의 가중치 찾기
    optimal_weights = optimize_weights(top_predictions)

    # 점수 계산
    top_predictions['Score'] = (
        optimal_weights[0] * top_predictions['MA5'] +
        optimal_weights[1] * top_predictions['MA20'] +
        optimal_weights[2] * top_predictions['MACD'] +
        optimal_weights[3] * (100 - top_predictions['RSI']) +
        optimal_weights[4] * top_predictions['ATR'] +
        optimal_weights[5] * top_predictions['OBV'] +
        optimal_weights[6] * (100 - top_predictions['Stoch']) +
        optimal_weights[7] * top_predictions['CCI'] +
        optimal_weights[8] * top_predictions['ADX']
    )

    # 점수로 정렬
    top_predictions = top_predictions.sort_values(by='Score', ascending=False).head(20)

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
