import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px

def get_stock_data(ticker, months=6):
    period = f"{months}mo"
    stock = yf.Ticker(ticker)
    data = stock.history(period=period, interval='1d')
    return data

def get_fundamental_metrics(ticker):
    stock = yf.Ticker(ticker)
    info = stock.get_info()
    
    def format_currency(value):
        return f"₹{value:,}" if isinstance(value, (int, float)) else "N/A"
    
    def format_percentage(value):
        return f"{value * 100:.2f}%" if isinstance(value, (int, float)) else "N/A"
    
    return {
        "Market Cap": format_currency(info.get('marketCap', 0)),
        "P/E Ratio": info.get('trailingPE', 'N/A'),
        "EPS": info.get('trailingEps', 'N/A'),
        "P/B Ratio": info.get('priceToBook', 'N/A'),
        "Dividend Yield": format_percentage(info.get('dividendYield', 0)),
        "ROE": format_percentage(info.get('returnOnEquity', 0)),
        "ROA": format_percentage(info.get('returnOnAssets', 0)),
        "Debt-to-Equity Ratio": info.get('debtToEquity', 'N/A'),
        "Current Ratio": info.get('currentRatio', 'N/A'),
        "Profit Margins": format_percentage(info.get('profitMargins', 0)),
        "Revenue": format_currency(info.get('totalRevenue', 0)),
        "Net Income": format_currency(info.get('netIncomeToCommon', 0)),
        "Operating Cash Flow": format_currency(info.get('operatingCashflow', 0)),
        "Free Cash Flow": format_currency(info.get('freeCashflow', 0))
    }

def get_dividends_splits(ticker):
    stock = yf.Ticker(ticker)
    dividends = stock.dividends.tail(5)
    splits = stock.splits.tail(5)
    
    if dividends.empty:
        dividends = pd.Series(["No Dividends"], index=[pd.Timestamp.today().date()])
    if splits.empty:
        splits = pd.Series(["No Split"], index=[pd.Timestamp.today().date()])
    
    dividends.index = dividends.index.date
    splits.index = splits.index.date
    
    return dividends, splits

def evaluate_stock(price_history):
    recent_prices = price_history['Close'].tail(30)
    if recent_prices.empty:
        return "Insufficient Data"
    avg_price = recent_prices.mean()
    last_price = recent_prices.iloc[-1]
    if last_price > avg_price * 1.1:
        return "Overvalued"
    elif last_price < avg_price * 0.9:
        return "Undervalued"
    else:
        return "Fairly Valued"

def main():
    st.set_page_config(page_title="Indian Stock Visualizer", layout="wide")
    st.title("📈 Indian Stock Data Visualizer (NSE/BSE)")
    
    with st.sidebar:
        stock_list = {
            "Reliance Industries": "RELIANCE.NS", "Tata Consultancy Services": "TCS.NS", "Infosys": "INFY.NS", "HDFC Bank": "HDFCBANK.NS",
            "ICICI Bank": "ICICIBANK.NS", "Kotak Mahindra Bank": "KOTAKBANK.NS", "Larsen & Toubro": "LT.NS", "Axis Bank": "AXISBANK.NS",
            "Hindustan Unilever": "HINDUNILVR.NS", "State Bank of India": "SBIN.NS", "Bajaj Finance": "BAJFINANCE.NS", "Bharti Airtel": "BHARTIARTL.NS",
            "Tata Steel": "TATASTEEL.NS", "ITC": "ITC.NS", "Maruti Suzuki": "MARUTI.NS", "Asian Paints": "ASIANPAINT.NS", "HCL Technologies": "HCLTECH.NS",
            "Wipro": "WIPRO.NS", "Tech Mahindra": "TECHM.NS", "UltraTech Cement": "ULTRACEMCO.NS", "Sun Pharma": "SUNPHARMA.NS", "Titan Company": "TITAN.NS",
            "Bajaj Auto": "BAJAJ-AUTO.NS", "Power Grid Corp": "POWERGRID.NS", "NTPC": "NTPC.NS", "Grasim Industries": "GRASIM.NS", "IndusInd Bank": "INDUSINDBK.NS",
            "Tata Motors": "TATAMOTORS.NS", "Nestle India": "NESTLEIND.NS", "Mahindra & Mahindra": "M&M.NS", "Dr. Reddy's Laboratories": "DRREDDY.NS",
            "JSW Steel": "JSWSTEEL.NS", "Cipla": "CIPLA.NS", "Adani Enterprises": "ADANIENT.NS", "Adani Ports": "ADANIPORTS.NS", "Eicher Motors": "EICHERMOT.NS",
            "HDFC Life Insurance": "HDFCLIFE.NS", "SBI Life Insurance": "SBILIFE.NS", "Bajaj Finserv": "BAJAJFINSV.NS", "Britannia Industries": "BRITANNIA.NS",
            "Hindalco Industries": "HINDALCO.NS", "Divi's Laboratories": "DIVISLAB.NS", "Apollo Hospitals": "APOLLOHOSP.NS", "UPL": "UPL.NS",
            "Oil & Natural Gas Corp": "ONGC.NS", "Coal India": "COALINDIA.NS", "Tata Consumer Products": "TATACONSUM.NS"
        }
        sorted_stocks = sorted(stock_list.keys())
        stock_selection = st.multiselect("Select Stocks:", sorted_stocks, default=[sorted_stocks[0]])
        months = st.slider("Select Months of Historical Data:", 1, 60, 6)
        exchange = st.radio("Select Exchange:", ["NSE", "BSE"])
    
    all_data = []
    if stock_selection:
        for company_name in stock_selection:
            stock_symbol = stock_list[company_name]
            if exchange == "BSE":
                stock_symbol = stock_symbol.replace(".NS", ".BO")
            
            with st.expander(f"📌 {company_name}"):
                with st.spinner(f"Fetching data for {company_name}..."):
                    try:
                        df = get_stock_data(stock_symbol, months)
                        if df.empty:
                            st.error(f"No data found for {company_name}.")
                            continue
                        
                        fundamentals = get_fundamental_metrics(stock_symbol)
                        dividends, splits = get_dividends_splits(stock_symbol)
                        df['Stock'] = company_name
                        all_data.append(df)
                        
                        st.subheader(f"Stock Data for {company_name}")
                        fig = px.line(df, x=df.index, y=['Open', 'Close'], title=f"Open & Close Prices of {company_name}")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.subheader("📊 Fundamental Metrics")
                        st.table(pd.DataFrame(fundamentals.items(), columns=["Metric", "Value"]))
                        
                        evaluation = evaluate_stock(df)
                        st.subheader("Stock Evaluation")
                        st.markdown(f"**This stock is currently: {evaluation}**")
                        
                        st.subheader("Dividends & Stock Splits")
                        st.table(pd.DataFrame({
                            "Date": dividends.index.tolist() + splits.index.tolist(),
                            "Dividend": dividends.tolist() + ["-"] * len(splits),
                            "Stock Split": ["-"] * len(dividends) + splits.tolist()
                        }).sort_values(by="Date", ascending=False).reset_index(drop=True))
                    except Exception as e:
                        st.error(f"Error fetching data: {e}")
    
    if all_data:
        combined_df = pd.concat(all_data)
        with st.expander("📊 Stock Comparison & Volume Analysis"):
            comparison_fig = px.line(combined_df, x=combined_df.index, y='Close', color='Stock', title="Stock Comparison")
            st.plotly_chart(comparison_fig, use_container_width=True)
            
            volume_fig = px.line(combined_df, x=combined_df.index, y='Volume', color='Stock', title="Volume of Shares Traded")
            st.plotly_chart(volume_fig, use_container_width=True)

if __name__ == "__main__":
    main()
