# Stock Database for Autocomplete Functionality
import pandas as pd
import requests
import io

def get_nse_stocks_list():
    """
    Returns a list of dictionaries, each containing stock symbol and company name.
    Tries to fetch from NSE API, falls back to local CSV if API fails.
    Returns:
        list: List of dictionaries with 'SYMBOL' and 'NAME OF COMPANY' keys
    """
    nse_url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        s = requests.Session()
        s.headers.update(headers)
        r = s.get(nse_url, timeout=10)
        s.close()
        r.raise_for_status()
        df_nse = pd.read_csv(io.BytesIO(r.content))
    except Exception as e:
        print(f"Error fetching NSE stocks from API: {e}. Falling back to local CSV.")
        # Fallback to local CSV file
        try:
            df_nse = pd.read_csv("equityList/EQUITY_L.csv")
        except Exception as e2:
            print(f"Error reading local CSV: {e2}")
            return []
    # Create a list of dictionaries for each stock
    stock_dict_list = []
    for index, row in df_nse.iterrows():
        stock_dict = {
            row['NAME OF COMPANY']: row['SYMBOL']
        }
        stock_dict_list.append(stock_dict)
    return stock_dict_list

def get_stock_database():
    """
    Returns a comprehensive database of Indian stocks with their names and symbols
    from the NSE exchange.
    """
    # Initialize an empty list for stocks
    stocks_list = []
    
    # Get the complete NSE stocks list
    try:
        nse_stocks = get_nse_stocks_list()
        
        # Add NSE stocks to the list
        for stock_dict in nse_stocks:
            for company_name, symbol in stock_dict.items():
                stocks_list.append({
                    'company_name': company_name,
                    'symbol': symbol,
                    'search_text': f"{company_name} {symbol}".lower()  # For easier searching
                })
    except Exception as e:
        print(f"Error fetching NSE stocks: {e}")
        # Return empty DataFrame if there's an error
        return pd.DataFrame(columns=['company_name', 'symbol', 'search_text'])
    
    return pd.DataFrame(stocks_list)

def search_stocks(query):
    """
    Search for stocks based on a partial query string.
    Returns matching stocks as a list of dictionaries with company name and symbol.
    """
    if not query or len(query) < 2:
        return []
    
    # Get the stock database
    stocks_df = get_stock_database()
    
    # Convert query to lowercase for case-insensitive search
    query = query.lower()
    
    # Search in both company names and symbols
    matches = stocks_df[stocks_df['search_text'].str.contains(query)]
    
    # Return as a list of dictionaries
    results = []
    for _, row in matches.iterrows():
        results.append({
            'company_name': row['company_name'],
            'symbol': row['symbol'],
            'display_text': f"{row['company_name']} ({row['symbol']})"
        })
    
    return results