import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from stock_database import search_stocks
from prophet import Prophet
from prophet.plot import plot_plotly
import warnings
warnings.filterwarnings('ignore')

# Helper functions
def get_stock_data(ticker, period="6mo", interval="1d"):
    """
    Fetch stock data from Yahoo Finance
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        
        # Get company info
        info = stock.info
        
        # Create summary metrics
        metrics = {
            'Current Price': info.get('currentPrice', info.get('regularMarketPrice', 'N/A')),
            'Market Cap': info.get('marketCap', 'N/A'),
            'PE Ratio': info.get('trailingPE', 'N/A'),
            'Dividend Yield': info.get('dividendYield', 'N/A'),
            '52 Week High': info.get('fiftyTwoWeekHigh', 'N/A'),
            '52 Week Low': info.get('fiftyTwoWeekLow', 'N/A'),
            'Volume': info.get('volume', 'N/A'),
            'Average Volume': info.get('averageVolume', 'N/A'),
            'Beta': info.get('beta', 'N/A'),
            'EPS': info.get('trailingEps', 'N/A')
        }
        
        return {
            'history': data,
            'metrics': metrics,
            'info': info
        }
    except Exception as e:
        raise Exception(f"Error fetching data: {str(e)}")

def get_dividends_splits(ticker):
    """
    Get dividend and split history for a stock
    """
    stock = yf.Ticker(ticker)
    dividends = stock.dividends.tail(5)
    splits = stock.splits.tail(5)
    
    if not dividends.empty:
        dividends.index = dividends.index.date
        dividends = dividends.apply(lambda x: f"₹{x:.2f}")
    
    if not splits.empty:
        splits.index = splits.index.date
        splits = splits.apply(lambda x: f"{int(x)}:1" if x > 1 else "No Split")
        
    return dividends, splits

def format_large_number(number):
    """
    Format large numbers to human-readable format
    """
    if not isinstance(number, (int, float)) or pd.isna(number):
        return 'N/A'
    
    if number >= 1e12:
        return f"₹{number/1e12:.2f}T"
    elif number >= 1e9:
        return f"₹{number/1e9:.2f}B"
    elif number >= 1e6:
        return f"₹{number/1e6:.2f}M"
    else:
        return f"₹{number:,.2f}"

def evaluate_stock(df):
    """
    Simple stock evaluation based on price relative to mean and standard deviation
    """
    if df.empty:
        return "No data available"
    
    recent_close = df['Close'].iloc[-1]
    avg_close = df['Close'].mean()
    std_dev = df['Close'].std()
    
    if recent_close > avg_close + std_dev:
        return "🔴 Overvalued - The stock price is significantly above its average. Consider market trends and financial reports before investing."
    elif recent_close < avg_close - std_dev:
        return "🟢 Undervalued - The stock price is significantly below its average. If the company has strong future prospects, this could be a good buying opportunity."
    else:
        return "🟡 Fairly Valued - The stock price is within a reasonable range. Check future growth prospects and company fundamentals before making a decision."

def calculate_technical_indicators(df):
    """
    Calculate technical indicators for stock analysis
    """
    if df.empty or len(df) < 20:
        return df
    
    try:
        # Calculate moving averages
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        
        # Calculate Bollinger Bands
        df['MA20_std'] = df['Close'].rolling(window=20).std()
        df['Upper_Band'] = df['MA20'] + (df['MA20_std'] * 2)
        df['Lower_Band'] = df['MA20'] - (df['MA20_std'] * 2)
        
        # Calculate RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD (Moving Average Convergence Divergence)
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
        
        return df
    except Exception as e:
        st.error(f"Error calculating technical indicators: {e}")
        return df

# Main application
def main():
    st.set_page_config(page_title="Indian Stock Visualizer", page_icon="📈", layout="wide")
    
    # Custom CSS
    st.markdown("""
        <style>
        .stAlert {
            padding: 1rem;
            margin: 1rem 0;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        /* Improved tab list styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background-color: #f8f9fa;
            padding: 8px 8px 0 8px;
            border-radius: 8px 8px 0 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        /* Enhanced individual tab styling */
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f0f2f6;
            border-radius: 8px 8px 0 0;
            gap: 1px;
            padding: 10px 16px;
            margin-right: 2px;
            font-weight: 500;
            border: 1px solid #e1e4e8;
            border-bottom: none;
            transition: all 0.2s ease;
        }
        /* Hover effect for tabs */
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #e9ecef;
            cursor: pointer;
        }
        /* Selected tab styling */
        .stTabs [aria-selected="true"] {
            background-color: #ffffff;
            border-bottom: 3px solid #4da6ff;
            box-shadow: 0 -2px 6px rgba(0,0,0,0.05);
        }
        /* Tab content panel styling */
        .stTabs [data-baseweb="tab-panel"] {
            background-color: #ffffff;
            border-radius: 0 0 8px 8px;
            padding: 16px;
            border: 1px solid #e1e4e8;
            border-top: none;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Title and description
    st.title("📈 Indian Stock Data Visualizer")
    st.markdown("An interactive tool for analyzing Indian stocks from NSE/BSE")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Exchange selection
        exchange = st.radio("Select Exchange:", ("NSE", "BSE"), index=0)
        exchange_suffix = ".NS" if exchange == "NSE" else ".BO"
        
        # Analysis mode
        analysis_mode = st.radio("Analysis Mode:", ("Single Stock Analysis", "Multi-Stock Comparison"), index=0)
        
        # Time period selection
        time_options = {
            "1 Month": "1mo",
            "3 Months": "3mo",
            "6 Months": "6mo",
            "1 Year": "1y",
            "2 Years": "2y",
            "5 Years": "5y",
            "Max": "max"
        }
        selected_period = st.selectbox("Select Time Period:", list(time_options.keys()), index=3)
        period = time_options[selected_period]
        
        # Interval selection (only show for periods > 1 month)
        interval_options = {
            "Daily": "1d",
            "Weekly": "1wk",
            "Monthly": "1mo"
        }
        interval = "1d"  # Default
        if period in ["1y", "2y", "5y", "max"]:
            selected_interval = st.selectbox("Select Interval:", list(interval_options.keys()), index=0)
            interval = interval_options[selected_interval]
        
        # Get stock list from stock_database
        from stock_database import get_stock_database
        
        # Get the stock database
        stocks_df = get_stock_database()
        
        # Create a dictionary of company names to symbols
        nifty_sensex_stocks = {}
        
        # Check if we got any stocks from the database
        if not stocks_df.empty:
            for _, row in stocks_df.iterrows():
                nifty_sensex_stocks[row['company_name']] = row['symbol']
        else:
            # Fallback to a basic list of major Indian stocks if database fetch fails
            st.warning("Could not fetch complete stock list. Using a limited list of major stocks.")
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
                "State Bank of India": "SBIN"
            }
        
        # Stock selection is now done exclusively from the database
        # Custom stock input has been removed as we now have all listed stocks
        
        # Stock selection based on mode - simplified without custom stock input
        if analysis_mode == "Single Stock Analysis":
            selected_stock_name = st.selectbox("Select a Stock:", sorted(nifty_sensex_stocks.keys()))
            selected_stock = nifty_sensex_stocks[selected_stock_name]
            
            stock_symbols = [f"{selected_stock}{exchange_suffix}"]
            stock_names = [selected_stock_name]
        else:  # Multi-Stock Comparison
            selected_stock_names = st.multiselect("Select Stocks to Compare:", 
                                               sorted(nifty_sensex_stocks.keys()), 
                                               default=[list(nifty_sensex_stocks.keys())[0]])
            
            selected_stocks = [nifty_sensex_stocks[name] for name in selected_stock_names]
            stock_symbols = [f"{stock}{exchange_suffix}" for stock in selected_stocks]
            stock_names = selected_stock_names
    
    # Main content area
    if not stock_symbols:
        st.info("👈 Please select at least one stock from the sidebar to begin analysis")
        return
    
    # Process data based on mode
    if analysis_mode == "Single Stock Analysis":
        single_stock_analysis(stock_symbols[0], stock_names[0], period, interval)
    else:
        multi_stock_comparison(stock_symbols, stock_names, period, interval)
    
    # Add stock analyzer link in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Related Tools")
    st.sidebar.markdown(
    """
    **Want to analyze mutual funds?**

    🔗 [Indian Mutual Fund Anazlyzer](https://indian-mutual-fund-analyzer.streamlit.app)

    Analyze mutual funds with technical indicators, and fund data.
    """,
    unsafe_allow_html=True
    )
    st.sidebar.markdown("---")

def single_stock_analysis(stock_symbol, stock_name, period, interval):
    """Detailed analysis for a single stock"""
    try:
        with st.spinner(f"Fetching data for {stock_name}..."):
            # Get stock data
            data = get_stock_data(stock_symbol, period, interval)
            df = data['history']
            info = data['info']
            metrics = data['metrics']
            
            # Get dividends and splits
            dividends, splits = get_dividends_splits(stock_symbol)
            
            # Calculate technical indicators
            df_with_indicators = calculate_technical_indicators(df.copy())
            
        # Display company info
        st.header(f"{info.get('longName', stock_name)}")
        if 'longBusinessSummary' in info:
            with st.expander("Company Overview", expanded=False):
                st.markdown(info.get('longBusinessSummary', ''))
        
        # Key metrics in columns
        st.subheader("Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", format_large_number(metrics['Current Price']))
        with col2:
            st.metric("Market Cap", format_large_number(metrics['Market Cap']))
        with col3:
            st.metric("P/E Ratio", f"{metrics['PE Ratio']:.2f}" if isinstance(metrics['PE Ratio'], (int, float)) and not pd.isna(metrics['PE Ratio']) else 'N/A')
        with col4:
            st.metric("Dividend Yield", f"{metrics['Dividend Yield']*100:.2f}%" if isinstance(metrics['Dividend Yield'], (int, float)) and not pd.isna(metrics['Dividend Yield']) else 'N/A')
        
        # Valuation status
        valuation = evaluate_stock(df)
        st.markdown(f"### 🏆 **Valuation Status:**")
        st.markdown(f"<div style='background-color:#f0f2f6;padding:10px;border-radius:5px;'>{valuation}</div>", unsafe_allow_html=True)
        
        # Create tabs for different charts
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Price Chart", "📈 Technical Indicators", "📉 Volume", "📋 Financial Data", "🔮 Price Forecast"])
        
        with tab1:
            # Price chart with candlestick
            st.subheader("Stock Price History")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='OHLC'
            ))
            
            fig.update_layout(
                title=f"{stock_name} Stock Price",
                yaxis_title='Stock Price (₹)',
                xaxis_title='Date',
                template='plotly_white',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Technical indicators
            st.subheader("Technical Indicators")
            
            # Moving Averages
            ma_fig = go.Figure()
            ma_fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
            
            if 'MA20' in df_with_indicators.columns:
                ma_fig.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['MA20'], mode='lines', name='20-day MA', line=dict(color='blue')))
            
            if 'MA50' in df_with_indicators.columns:
                ma_fig.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['MA50'], mode='lines', name='50-day MA', line=dict(color='orange')))
            
            if 'MA200' in df_with_indicators.columns:
                ma_fig.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['MA200'], mode='lines', name='200-day MA', line=dict(color='red')))
            
            ma_fig.update_layout(
                title="Moving Averages",
                yaxis_title='Price (₹)',
                template='plotly_white',
                height=400
            )
            st.plotly_chart(ma_fig, use_container_width=True)
            
            # Bollinger Bands
            if 'Upper_Band' in df_with_indicators.columns and 'Lower_Band' in df_with_indicators.columns:
                bb_fig = go.Figure()
                bb_fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
                bb_fig.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['Upper_Band'], mode='lines', name='Upper Band', line=dict(dash='dash')))
                bb_fig.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['MA20'], mode='lines', name='20-day MA'))
                bb_fig.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['Lower_Band'], mode='lines', name='Lower Band', line=dict(dash='dash')))
                
                bb_fig.update_layout(
                    title="Bollinger Bands",
                    yaxis_title='Price (₹)',
                    template='plotly_white',
                    height=400
                )
                st.plotly_chart(bb_fig, use_container_width=True)
            
            # RSI
            if 'RSI' in df_with_indicators.columns:
                rsi_fig = go.Figure()
                rsi_fig.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['RSI'], mode='lines', name='RSI'))
                
                # Add overbought/oversold lines
                rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                
                rsi_fig.update_layout(
                    title="Relative Strength Index (RSI)",
                    yaxis_title='RSI Value',
                    template='plotly_white',
                    height=300
                )
                st.plotly_chart(rsi_fig, use_container_width=True)
            
            # MACD
            if all(x in df_with_indicators.columns for x in ['MACD', 'Signal_Line', 'MACD_Histogram']):
                macd_fig = go.Figure()
                macd_fig.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['MACD'], mode='lines', name='MACD'))
                macd_fig.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['Signal_Line'], mode='lines', name='Signal Line'))
                
                # Add MACD histogram
                colors = ['red' if x < 0 else 'green' for x in df_with_indicators['MACD_Histogram']]
                macd_fig.add_trace(go.Bar(x=df_with_indicators.index, y=df_with_indicators['MACD_Histogram'], name='Histogram', marker_color=colors))
                
                macd_fig.update_layout(
                    title="MACD (Moving Average Convergence Divergence)",
                    yaxis_title='Value',
                    template='plotly_white',
                    height=300
                )
                st.plotly_chart(macd_fig, use_container_width=True)
        
        with tab3:
            # Volume chart
            st.subheader("Trading Volume")
            volume_fig = go.Figure()
            volume_fig.add_trace(go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume'
            ))
            
            volume_fig.update_layout(
                title=f"{stock_name} Trading Volume",
                yaxis_title='Volume',
                xaxis_title='Date',
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(volume_fig, use_container_width=True)
        
        with tab4:
            # Financial metrics table
            st.subheader("Financial Metrics")
            metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
            metrics_df.index.name = 'Metric'
            st.dataframe(metrics_df)
            
            # Dividends and splits
            col1, col2 = st.columns(2)
            
            with col1:
                if not dividends.empty:
                    st.subheader("Recent Dividend Declarations")
                    st.dataframe(dividends)
                else:
                    st.info("No recent dividends found")
            
            with col2:
                if not splits.empty:
                    st.subheader("Recent Stock Splits")
                    st.dataframe(splits)
                else:
                    st.info("No recent stock splits found")
            
            # Download button for CSV
            csv = df.to_csv()
            st.download_button(
                label="Download historical data as CSV",
                data=csv,
                file_name=f"{stock_symbol}_historical_data.csv",
                mime='text/csv',
            )
            
        with tab5:
            # Stock Price Forecasting
            st.subheader("Stock Price Forecast")
            
            # Forecasting method selection
            forecast_method = "Prophet"  # Only Prophet is available now
            
            # Forecast period selection
            forecast_days = st.slider(
                "Forecast Period (Days):",
                min_value=7,
                max_value=365,
                value=30,
                step=7
            )
            
            # Forecast button
            if st.button("Generate Forecast"):
                with st.spinner(f"Generating {forecast_days} days forecast using Prophet..."):
                    try:
                        # Prepare data for forecasting
                        forecast_data = df[['Close']].reset_index()
                        forecast_data.columns = ['ds', 'y']
                        # Remove timezone information from datetime column to avoid Prophet errors
                        forecast_data['ds'] = forecast_data['ds'].dt.tz_localize(None)
                        
                        # Prophet model
                        m = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
                        m.fit(forecast_data)
                        
                        # Create future dataframe
                        future = m.make_future_dataframe(periods=forecast_days)
                        forecast = m.predict(future)
                        
                        # Plot forecast
                        fig = plot_plotly(m, forecast)
                        fig.update_layout(
                            title=f"{stock_name} Stock Price Forecast ({forecast_days} days)",
                            yaxis_title='Stock Price (₹)',
                            xaxis_title='Date',
                            template='plotly_white',
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show forecast components
                        st.subheader("Forecast Components")
                        fig2 = m.plot_components(forecast)
                        st.pyplot(fig2)
                    
                    except Exception as e:
                        st.error(f"Error generating forecast: {str(e)}")
                        st.info("Try adjusting the forecast period.")
            
            # Explanation of forecasting method
            with st.expander("About Prophet Forecasting"):
                st.markdown("""
                ### Prophet
                - Developed by Facebook
                - Handles seasonality and holiday effects
                - Good for time series with strong seasonal patterns
                - Automatically detects changepoints in the time series
                - Robust to missing data and outliers
                
                **Note:** All forecasting methods have limitations and should be used as one of many tools for investment decisions. Past performance is not indicative of future results.
                """)
    
    except Exception as e:
        st.error(f"Error analyzing {stock_name}: {str(e)}")
        st.warning("Please check if the stock symbol is correct and try again.")


def multi_stock_comparison(stock_symbols, stock_names, period, interval):
    """Compare multiple stocks side by side"""
    try:
        # Create a progress bar
        progress_bar = st.progress(0)
        all_data = []
        all_metrics = []
        
        # Fetch data for all stocks
        for i, (symbol, name) in enumerate(zip(stock_symbols, stock_names)):
            with st.spinner(f"Fetching data for {name} ({i+1}/{len(stock_symbols)})..."):
                try:
                    data = get_stock_data(symbol, period, interval)
                    df = data['history']
                    
                    if not df.empty:
                        # Add stock name as a column for identification
                        df['Stock'] = name
                        all_data.append(df)
                        
                        # Store metrics for comparison
                        metrics = data['metrics']
                        metrics['Stock'] = name
                        all_metrics.append(metrics)
                except Exception as e:
                    st.error(f"Error fetching data for {name}: {str(e)}")
            
            # Update progress bar
            progress_bar.progress((i + 1) / len(stock_symbols))
        
        # Remove progress bar after completion
        progress_bar.empty()
        
        if not all_data:
            st.error("No data available for the selected stocks.")
            return
            
        # Combine all stock data
        combined_df = pd.concat(all_data)
        
        # Create tabs for different comparisons
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Price Comparison", "📈 Performance", "📉 Volume", "📋 Metrics"])
        
        with tab1:
            st.subheader("Stock Price Comparison")
            
            # Normalized price chart (for better comparison)
            st.markdown("### Normalized Price (Base 100)")
            st.markdown("*Prices normalized to 100 at the beginning of the period for better comparison*")
            
            # Create a copy of the dataframe for normalization
            normalized_df = combined_df.copy()
            
            # Group by stock and normalize the Close price
            for stock in stock_names:
                stock_data = normalized_df[normalized_df['Stock'] == stock]
                if not stock_data.empty:
                    base_value = stock_data['Close'].iloc[0]
                    normalized_df.loc[normalized_df['Stock'] == stock, 'Normalized'] = \
                        normalized_df.loc[normalized_df['Stock'] == stock, 'Close'] / base_value * 100
            
            # Plot normalized prices
            norm_fig = px.line(
                normalized_df, 
                x=normalized_df.index, 
                y='Normalized', 
                color='Stock',
                title="Normalized Stock Price Comparison (Base 100)"
            )
            norm_fig.update_layout(
                yaxis_title='Normalized Price (Base 100)',
                xaxis_title='Date',
                template='plotly_white',
                height=500
            )
            st.plotly_chart(norm_fig, use_container_width=True)
            
            # Absolute price chart
            st.markdown("### Absolute Price Comparison")
            price_fig = px.line(
                combined_df, 
                x=combined_df.index, 
                y='Close', 
                color='Stock',
                title="Stock Price Comparison"
            )
            price_fig.update_layout(
                yaxis_title='Price (₹)',
                xaxis_title='Date',
                template='plotly_white',
                height=500
            )
            st.plotly_chart(price_fig, use_container_width=True)
        
        with tab2:
            st.subheader("Performance Analysis")
            
            # Calculate returns for different periods
            returns_data = []
            
            for stock in stock_names:
                stock_data = combined_df[combined_df['Stock'] == stock]
                if len(stock_data) > 0:
                    # Calculate returns
                    latest_price = stock_data['Close'].iloc[-1]
                    earliest_price = stock_data['Close'].iloc[0]
                    total_return = ((latest_price - earliest_price) / earliest_price) * 100
                    
                    # Calculate volatility (standard deviation of daily returns)
                    daily_returns = stock_data['Close'].pct_change().dropna()
                    volatility = daily_returns.std() * 100
                    
                    # Add to returns data
                    returns_data.append({
                        'Stock': stock,
                        'Total Return (%)': round(total_return, 2),
                        'Volatility (%)': round(volatility, 2),
                        'Max Price': round(stock_data['High'].max(), 2),
                        'Min Price': round(stock_data['Low'].min(), 2),
                        'Avg Volume': int(stock_data['Volume'].mean())
                    })
            
            # Create returns dataframe
            if returns_data:
                returns_df = pd.DataFrame(returns_data)
                returns_df.set_index('Stock', inplace=True)
                
                # Display returns table
                st.markdown("### Performance Metrics")
                st.dataframe(returns_df)
                
                # Plot returns comparison
                returns_fig = px.bar(
                    returns_data,
                    x='Stock',
                    y='Total Return (%)',
                    title="Total Return Comparison",
                    color='Total Return (%)',
                    color_continuous_scale='RdYlGn'
                )
                returns_fig.update_layout(
                    yaxis_title='Return (%)',
                    template='plotly_white',
                    height=400
                )
                st.plotly_chart(returns_fig, use_container_width=True)
                
                # Plot volatility comparison
                vol_fig = px.bar(
                    returns_data,
                    x='Stock',
                    y='Volatility (%)',
                    title="Volatility Comparison",
                    color='Volatility (%)',
                    color_continuous_scale='Viridis'
                )
                vol_fig.update_layout(
                    yaxis_title='Volatility (%)',
                    template='plotly_white',
                    height=400
                )
                st.plotly_chart(vol_fig, use_container_width=True)
        
        with tab3:
            st.subheader("Volume Analysis")
            
            # Volume comparison chart
            volume_fig = px.line(
                combined_df,
                x=combined_df.index,
                y='Volume',
                color='Stock',
                title="Trading Volume Comparison"
            )
            volume_fig.update_layout(
                yaxis_title='Volume',
                xaxis_title='Date',
                template='plotly_white',
                height=500
            )
            st.plotly_chart(volume_fig, use_container_width=True)
            
            # Average daily volume comparison
            avg_volumes = combined_df.groupby('Stock')['Volume'].mean().reset_index()
            avg_vol_fig = px.bar(
                avg_volumes,
                x='Stock',
                y='Volume',
                title="Average Daily Trading Volume",
                color='Volume',
                color_continuous_scale='Blues'
            )
            avg_vol_fig.update_layout(
                yaxis_title='Average Volume',
                template='plotly_white',
                height=400
            )
            st.plotly_chart(avg_vol_fig, use_container_width=True)
        
        with tab4:
            st.subheader("Financial Metrics Comparison")
            
            if all_metrics:
                # Create a metrics comparison dataframe
                metrics_df = pd.DataFrame(all_metrics)
                metrics_df.set_index('Stock', inplace=True)
                
                # Select key metrics for comparison
                key_metrics = ['Current Price', 'Market Cap', 'PE Ratio', 'Dividend Yield', 'Beta', 'EPS']
                display_metrics = metrics_df[key_metrics].copy()
                
                # Format the metrics for display
                for metric in key_metrics:
                    if metric in ['Current Price', 'EPS']:
                        display_metrics[metric] = display_metrics[metric].apply(
                            lambda x: format_large_number(x) if isinstance(x, (int, float)) and not pd.isna(x) else 'N/A'
                        )
                    elif metric == 'Market Cap':
                        display_metrics[metric] = display_metrics[metric].apply(
                            lambda x: format_large_number(x) if isinstance(x, (int, float)) and not pd.isna(x) else 'N/A'
                        )
                    elif metric == 'Dividend Yield':
                        display_metrics[metric] = display_metrics[metric].apply(
                            lambda x: f"{x*100:.2f}%" if isinstance(x, (int, float)) and not pd.isna(x) else 'N/A'
                        )
                    elif metric in ['PE Ratio', 'Beta']:
                        display_metrics[metric] = display_metrics[metric].apply(
                            lambda x: f"{x:.2f}" if isinstance(x, (int, float)) and not pd.isna(x) else 'N/A'
                        )
                
                # Display the metrics table
                st.dataframe(display_metrics)
                
                # Create download button for metrics CSV
                csv = metrics_df.to_csv()
                st.download_button(
                    label="Download metrics as CSV",
                    data=csv,
                    file_name="stock_metrics_comparison.csv",
                    mime='text/csv',
                )
                
                # Create download button for historical data CSV
                hist_csv = combined_df.to_csv()
                st.download_button(
                    label="Download historical data as CSV",
                    data=hist_csv,
                    file_name="stock_historical_data.csv",
                    mime='text/csv',
                )
    
    except Exception as e:
        st.error(f"Error in multi-stock comparison: {str(e)}")

if __name__ == "__main__":
    main()
           