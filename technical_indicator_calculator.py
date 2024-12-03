import FinanceDataReader as fdr
import pandas as pd
import logging
import os
from datetime import datetime, timedelta
import threading
from sklearn.ensemble import IsolationForest

# 로그 디렉토리 설정
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# 로깅 설정
logging.basicConfig(
    filename=os.path.join(log_dir, 'stock_data_fetcher.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 콘솔 로그 출력 설정
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)

# 주식 데이터를 가져오는 스레드 함수
def fetch_single_stock_data(code, start_date, end_date, all_stocks_data):
    """주식 코드에 대한 데이터를 가져오는 함수."""
    try:
        df = fdr.DataReader(code, start_date, end_date)
        if df is not None and not df.empty:
            recent_data = df.tail(26)
            today_open = recent_data['Open'].iloc[-1]

            if today_open == 0:
                logging.warning(f"{code}의 오늘 시작가(Open)가 0이므로 데이터 제외.")
                return

            identical_prices = (recent_data['Open'] == recent_data['High']) & \
                               (recent_data['High'] == recent_data['Low']) & \
                               (recent_data['Low'] == recent_data['Close'])

            if identical_prices.any():
                logging.warning(f"{code}의 최근 26일 이내에 시작가, 고가, 저가, 종가가 동일한 날이 있으므로 데이터 제외.")
                return
            
            recent_volume = df['Volume'].tail(26)

            if recent_volume.sum() > 0:
                if all(recent_data['Close'] >= 3000) and all(recent_data['Close'] <= 300000):
                    df.reset_index(inplace=True)
                    df['Code'] = code
                    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
                    all_stocks_data[code] = df
                    logging.info(f"{code} 데이터 가져오기 완료, 데이터 길이: {len(df)}")
                else:
                    logging.warning(f"{code}의 최근 26일 종가가 3000 미만이거나 30만원 초과입니다. 데이터 제외.")
            else:
                logging.warning(f"{code}의 최근 26일 거래량이 0입니다. 데이터 제외.")
        else:
            logging.warning(f"{code} 데이터가 비어 있거나 가져오기 실패")
    except Exception as e:
        logging.error(f"{code} 데이터 가져오기 중 오류 발생: {e}")

def fetch_stock_data(markets, start_date, end_date):
    """주식 데이터를 가져오는 메인 함수."""
    all_stocks_data = {}

    for market in markets:
        codes = fdr.StockListing(market)['Code'].tolist()
        threads = []

        for code in codes:
            thread = threading.Thread(target=fetch_single_stock_data, args=(code, start_date, end_date, all_stocks_data))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

    if all_stocks_data:
        data_dir = 'data'
        os.makedirs(data_dir, exist_ok=True)
        all_data = pd.concat(all_stocks_data.values(), ignore_index=True)

        # Anomaly 컬럼 추가 (조정 상태 탐지)
        isolation_forest = IsolationForest(contamination=0.05, random_state=42)
        all_data['Anomaly'] = isolation_forest.fit_predict(all_data[['Close', 'Open', 'High', 'Low', 'Volume']])

        # 조정 상태 해석: -1은 조정 상태로 해석
        all_data['Adjustment'] = np.where(all_data['Anomaly'] == -1, '조정', '정상')

        all_data.to_csv(os.path.join(data_dir, 'stock_data.csv'), index=False)
        logging.info("주식 데이터가 'data/stock_data.csv'로 저장되었습니다.")
    else:
        logging.warning("가져온 주식 데이터가 없습니다.")

if __name__ == "__main__":
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)
    fetch_stock_data(['KOSPI', 'KOSDAQ'], start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
