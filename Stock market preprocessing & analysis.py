import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('C:/Users/deepa/OneDrive/Desktop/stock_details_5_years.csv/stock_details_5_years.csv')

# Basic info
print(df.info())
print(df.head())

# summary statistics
print(df.describe())

# Data cleaning steps
df['Date'] = pd.to_datetime(df['Date'].str.split(' ').str[0])

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_convert(None)

# Check for missing values
print(df.isnull().sum())

# Check for duplicates
print(f"Duplicate rows: {df.duplicated().sum()}")

# Add time features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# Calculate returns
df['Daily_Return'] = df.groupby('Company')['Close'].pct_change()

# Calculate moving averages
df['MA_20'] = df.groupby('Company')['Close'].transform(lambda x: x.rolling(window=20).mean())
df['MA_50'] = df.groupby('Company')['Close'].transform(lambda x: x.rolling(window=50).mean())

# Calculate volatility
df['Volatility'] = df.groupby('Company')['Close'].transform(lambda x: x.pct_change().rolling(window=20).std())

# Calculate cumulative returns
df['Cumulative_Return'] = df.groupby('Company')['Close'].transform(lambda x: (  x / x.iloc[0]) - 1)

# Select top 5 companies by market cap (assuming higher close price = larger cap)
top_companies = df.groupby('Company')['Close'].max().sort_values(ascending=False).head(5).index

# Stock Price Trends Over Time
plt.figure(figsize=(14, 7))
for company in top_companies:
    company_data = df[df['Company'] == company]
    plt.plot(company_data['Date'], company_data['Close'], label=company, linewidth=2)

plt.title('Stock Price Trends (Top 5 Companies)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Closing Price ($)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Trading Volume Analysis
plt.figure(figsize=(14, 7))
for company in top_companies:
    company_data = df[df['Company'] == company]
    plt.plot(company_data['Date'], company_data['Volume'], label=company, alpha=0.7)

plt.title('Trading Volume Trends (Top 5 Companies)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Volume (Millions)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.yscale('log')
plt.show()

# Volatility Analysis (Risk Assessment)
volatility = df.groupby('Company')['Daily_Return'].std().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
volatility.head(20).plot(kind='bar', color='coral')
plt.title('Top 20 Most Volatile Stocks', fontsize=16)
plt.xlabel('Company', fontsize=12)
plt.ylabel('Standard Deviation (Daily Returns)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Correlation Between Price Metrics
# Example for AAPL
aapl = df[df['Company'] == 'AAPL'][['Open', 'High', 'Low', 'Close', 'Volume']]

plt.figure(figsize=(10, 8))
sns.heatmap(aapl.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('AAPL Price Correlations', fontsize=16)
plt.show()


# Dividend vs. Non-Dividend Stocks
dividend_stocks = df[df['Dividends'] > 0]
non_dividend = df[df['Dividends'] == 0]

plt.figure(figsize=(10, 6))
sns.boxplot(
    x=dividend_stocks['Dividends'] > 0,
    y=dividend_stocks['Daily_Return'],
    palette='Set2'
)
plt.title('Daily Returns: Dividend vs. Non-Dividend Stocks', fontsize=16)
plt.xlabel('Dividend-Paying Stock?', fontsize=12)
plt.ylabel('Daily Return', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Moving Averages (Trend Analysis)
# Example for AAPL
aapl = df[df['Company'] == 'AAPL'].set_index('Date')

aapl['MA_50'] = aapl['Close'].rolling(50).mean()
aapl['MA_200'] = aapl['Close'].rolling(200).mean()

plt.figure(figsize=(14, 7))
plt.plot(aapl['Close'], label='AAPL Close Price', alpha=0.7)
plt.plot(aapl['MA_50'], label='50-Day MA', linestyle='--')
plt.plot(aapl['MA_200'], label='200-Day MA', linestyle='--')
plt.title('AAPL Moving Averages', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price ($)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Intraday Price Movement Analysis
df['Daily_Spread'] = df['High'] - df['Low']
df['Open_Close_Change'] = (df['Close'] - df['Open']) / df['Open'] * 100

plt.figure(figsize=(14, 6))
sns.boxplot(data=df[df['Company'].isin(top_companies)],
            x='Company',
            y='Daily_Spread',
            palette='viridis')
plt.title('Daily Price Range (High - Low) by Company', fontsize=16)
plt.ylabel('Price Spread ($)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Volume-Price Relationship
for company in top_companies[:3]:
    company_data = df[df['Company'] == company]
    plt.figure(figsize=(14, 5))
    plt.scatter(company_data['Daily_Return'],
                company_data['Volume'],
                alpha=0.5,
                label=company)
    plt.title(f'Volume vs. Daily Return: {company}', fontsize=16)
    plt.xlabel('Daily Return (%)', fontsize=12)
    plt.ylabel('Log Volume', fontsize=12)
    plt.yscale('log')
    plt.axvline(0, color='red', linestyle='--')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# Rolling Volatility (30-day Window)
for company in top_companies[:3]:
    company_data = df[df['Company'] == company].set_index('Date')
    company_data['30D_Volatility'] = company_data['Daily_Return'].rolling(30).std() * np.sqrt(252)

    plt.figure(figsize=(14, 5))
    company_data['30D_Volatility'].plot(label=f'{company} 30D Volatility')
    plt.title(f'30-Day Rolling Volatility: {company}', fontsize=16)
    plt.ylabel('Annualized Volatility', fontsize=12)
    plt.axhline(y=company_data['30D_Volatility'].mean(),
                color='red',
                linestyle='--',
                label='Mean Volatility')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# Relative Strength Analysis
# Assuming we have SPY data, else use one stock as benchmark
benchmark = 'AAPL'
for company in top_companies[1:3]:
    company_data = df[df['Company'] == company].set_index('Date')['Close']
    benchmark_data = df[df['Company'] == benchmark].set_index('Date')['Close']

    relative_strength = (company_data / benchmark_data) * 100

    plt.figure(figsize=(14, 5))
    relative_strength.plot(label=f'{company} Relative to {benchmark}')
    plt.title(f'Relative Strength: {company} vs {benchmark}', fontsize=16)
    plt.ylabel('Relative Strength (%)', fontsize=12)
    plt.axhline(y=100, color='red', linestyle='--', label='Parity')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# Candlestick Pattern Recognition
import mplfinance as mpf

for company in top_companies[:2]:
    company_data = df[df['Company'] == company].set_index('Date')
    ohlc_data = company_data.loc[:, ['Open', 'High', 'Low', 'Close', 'Volume']]
    ohlc_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    plt.figure(figsize=(14, 7))
    mpf.plot(ohlc_data[-90:],
             type='candle',
             style='charles',
             title=f'{company} Candlestick Chart (Last 90 Days)',
             volume=True,
             mav=(20,50))
    plt.show()

# Sector Rotation Analysis
# Mock sector analysis (replace with actual sector data if available)
sectors = {
    'AAPL': 'Technology',
    'MSFT': 'Technology',
    'JPM': 'Financial',
    'XOM': 'Energy',
    'WMT': 'Consumer Staples'
}

df['Sector'] = df['Company'].map(sectors)

if 'Sector' in df.columns:
    sector_perf = df.groupby(['Sector', 'Year'])['Daily_Return'].mean().unstack()

    plt.figure(figsize=(14, 7))
    sns.heatmap(sector_perf,
                annot=True,
                cmap='RdYlGn',
                center=0,
                fmt='.2%')
    plt.title('Annual Sector Performance (%)', fontsize=16)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Sector', fontsize=12)
    plt.show()

# Monte Carlo Simulation for Risk Assessment
from scipy.stats import norm

def monte_carlo_simulation(ticker, days=30, simulations=1000):
    company_data = df[df['Company'] == ticker].set_index('Date')['Close']
    log_returns = np.log(1 + company_data.pct_change())

    mu = log_returns.mean()
    sigma = log_returns.std()

    last_price = company_data.iloc[-1]

    plt.figure(figsize=(14, 7))
    for i in range(simulations):
        price_path = [last_price]
        for _ in range(days):
            daily_return = norm.rvs(loc=mu, scale=sigma)
            price_path.append(price_path[-1] * (1 + daily_return))
        plt.plot(price_path, alpha=0.05, color='blue')

    plt.title(f'{ticker} Monte Carlo Simulation ({days} Days)', fontsize=16)
    plt.xlabel('Trading Days', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

monte_carlo_simulation('AAPL')

# Pairs Trading Opportunity Identification
from statsmodels.tsa.stattools import coint

stock_pairs = [('AAPL', 'MSFT'), ('JPM', 'BAC'), ('XOM', 'CVX')]

for stock1, stock2 in stock_pairs:
    s1 = df[df['Company'] == stock1].set_index('Date')['Close']
    s2 = df[df['Company'] == stock2].set_index('Date')['Close']

    # Align dates
    aligned = pd.concat([s1, s2], axis=1).dropna()
    s1_aligned = aligned.iloc[:, 0]
    s2_aligned = aligned.iloc[:, 1]

    # Calculate spread
    spread = s1_aligned - s2_aligned

    # Test for cointegration
    score, pvalue, _ = coint(s1_aligned, s2_aligned)

    plt.figure(figsize=(14, 5))
    spread.plot(title=f'{stock1}-{stock2} Spread (p-value: {pvalue:.4f})')
    plt.axhline(y=spread.mean(), color='red', linestyle='--')
    plt.fill_between(spread.index,
                     spread.mean() - spread.std(),
                     spread.mean() + spread.std(),
                     color='red', alpha=0.1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

