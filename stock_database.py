# Stock Database for Autocomplete Functionality
import pandas as pd

def get_stock_database():
    """
    Returns a comprehensive database of Indian stocks with their names and symbols
    for both NSE and BSE exchanges.
    """
    # Dictionary of major Indian stocks (Nifty 50 and other popular stocks)
    nifty_sensex_stocks = {
        "Reliance Industries": "RELIANCE", 
        "Tata Consultancy Services": "TCS", 
        "HDFC Bank": "HDFCBANK",
        "Infosys": "INFY", 
        "ICICI Bank": "ICICIBANK", 
        "Kotak Mahindra Bank": "KOTAKBANK", 
        "Larsen & Toubro": "LT", 
        "Axis Bank": "AXISBANK",
        "Hindustan Unilever": "HINDUNILVR", 
        "State Bank of India": "SBIN", 
        "Bajaj Finance": "BAJFINANCE", 
        "Bharti Airtel": "BHARTIARTL",
        "Tata Steel": "TATASTEEL", 
        "ITC": "ITC", 
        "Maruti Suzuki": "MARUTI", 
        "Asian Paints": "ASIANPAINT", 
        "HCL Technologies": "HCLTECH",
        "Wipro": "WIPRO", 
        "Tech Mahindra": "TECHM", 
        "UltraTech Cement": "ULTRACEMCO", 
        "Sun Pharma": "SUNPHARMA", 
        "Titan Company": "TITAN",
        "Bajaj Auto": "BAJAJ-AUTO", 
        "Power Grid Corp": "POWERGRID", 
        "NTPC": "NTPC", 
        "Grasim Industries": "GRASIM", 
        "IndusInd Bank": "INDUSINDBK",
        "Tata Motors": "TATAMOTORS", 
        "Nestle India": "NESTLEIND", 
        "Mahindra & Mahindra": "M&M", 
        "Dr. Reddy's Laboratories": "DRREDDY",
        "JSW Steel": "JSWSTEEL", 
        "Cipla": "CIPLA", 
        "Adani Enterprises": "ADANIENT", 
        "Adani Ports": "ADANIPORTS", 
        "Eicher Motors": "EICHERMOT",
        "HDFC Life Insurance": "HDFCLIFE", 
        "SBI Life Insurance": "SBILIFE", 
        "Bajaj Finserv": "BAJAJFINSV", 
        "Britannia Industries": "BRITANNIA",
        "Hindalco Industries": "HINDALCO", 
        "Divi's Laboratories": "DIVISLAB", 
        "Apollo Hospitals": "APOLLOHOSP", 
        "UPL": "UPL",
        "Oil & Natural Gas Corp": "ONGC", 
        "Coal India": "COALINDIA", 
        "Tata Consumer Products": "TATACONSUM",
        # Additional popular stocks
        "Adani Green Energy": "ADANIGREEN",
        "Adani Power": "ADANIPOWER",
        "Adani Transmission": "ADANITRANS",
        "Ambuja Cements": "AMBUJACEM",
        "Ashok Leyland": "ASHOKLEY",
        "Aurobindo Pharma": "AUROPHARMA",
        "Bajaj Holdings": "BAJAJHLDNG",
        "Bandhan Bank": "BANDHANBNK",
        "Bank of Baroda": "BANKBARODA",
        "Berger Paints": "BERGEPAINT",
        "Bharat Electronics": "BEL",
        "Bharat Petroleum": "BPCL",
        "Biocon": "BIOCON",
        "Bosch": "BOSCHLTD",
        "Canara Bank": "CANBK",
        "Cholamandalam Investment": "CHOLAFIN",
        "Colgate Palmolive": "COLPAL",
        "Container Corporation": "CONCOR",
        "Dabur India": "DABUR",
        "DLF": "DLF",
        "Federal Bank": "FEDERALBNK",
        "GAIL India": "GAIL",
        "Godrej Consumer Products": "GODREJCP",
        "Havells India": "HAVELLS",
        "Hero MotoCorp": "HEROMOTOCO",
        "Hindustan Aeronautics": "HAL",
        "Hindustan Petroleum": "HINDPETRO",
        "Hindustan Zinc": "HINDZINC",
        "IDFC First Bank": "IDFCFIRSTB",
        "Indian Oil Corporation": "IOC",
        "Indus Towers": "INDUSTOWER",
        "Interglobe Aviation": "INDIGO",
        "Jindal Steel": "JINDALSTEL",
        "LIC Housing Finance": "LICHSGFIN",
        "Lupin": "LUPIN",
        "Marico": "MARICO",
        "MRF": "MRF",
        "NMDC": "NMDC",
        "ONGC": "ONGC",
        "Page Industries": "PAGEIND",
        "Petronet LNG": "PETRONET",
        "Pidilite Industries": "PIDILITIND",
        "Punjab National Bank": "PNB",
        "Shree Cement": "SHREECEM",
        "Siemens": "SIEMENS",
        "SRF": "SRF",
        "Tata Chemicals": "TATACHEM",
        "Tata Power": "TATAPOWER",
        "Torrent Pharmaceuticals": "TORNTPHARM",
        "United Breweries": "UBL",
        "United Spirits": "MCDOWELL-N",
        "Vedanta": "VEDL",
        "Voltas": "VOLTAS",
        "Zee Entertainment": "ZEEL"
    }
    
    # Create a DataFrame for easier searching
    stocks_list = []
    for company_name, symbol in nifty_sensex_stocks.items():
        stocks_list.append({
            'company_name': company_name,
            'symbol': symbol,
            'search_text': f"{company_name} {symbol}".lower()  # For easier searching
        })
    
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