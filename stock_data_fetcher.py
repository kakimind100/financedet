# stock_data_fetcher.py
import FinanceDataReader as fdr
import pandas as pd
import logging
import os

# 로그 디렉토리 설정
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# 로깅 설정
logging.basicConfig(
    filename=os.path.join(log_dir, 'stock_data_fetcher.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def fetch_stock_data(market, start_date, end_date):
    """주식 데이터를 가져오는 함수."""
    all_stocks_data = {}
    codes = fdr.StockListing(market)['Code'].tolist()

    for code in codes:
        try:
            df = fdr.DataReader(code, start_date, end_date)
            if df is not None and not df.empty:
                df.reset_index(inplace=True)
                df['Code'] = code
                all_stocks_data[code] = df
                logging.info(f"{code} 데이터 가져오기 완료, 데이터 길이: {len(df)}")
            else:
                logging.warning(f"{code} 데이터가 비어 있거나 가져오기 실패")
        except Exception as e:
            logging.error(f"{code} 데이터 가져오기 중 오류 발생: {e}")

    # 데이터프레임으로 변환 후 CSV로 저장
    all_data = pd.concat(all_stocks_data.values(), ignore_index=True)
    all_data.to_csv('stock_data.csv', index=False)
    logging.info("주식 데이터가 'stock_data.csv'로 저장되었습니다.")

if __name__ == "__main__":
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)
    fetch_stock_data('KOSPI', start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
