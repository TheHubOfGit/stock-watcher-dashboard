# Stock Dashboard

Real-time stock market dashboard with technical indicators, signals, and performance tracking.

🔗 **Live Site**: [www.michaelbrancazio-stockboard.pages.dev](https://www.michaelbrancazio-stockboard.pages.dev)

## Features

- **Real-time Data**: Automatically updated hourly via GitHub Actions
- **Technical Indicators**: RSI, EMA signals, Z-Score mean reversion
- **Performance Tracking**: Year-to-date alpha vs SPY benchmark
- **Interactive Charts**: Hover graphs for historical analysis
- **Asset Grouping**: Organized by ETFs, Stocks, and Crypto

## Technology Stack

- **Frontend**: Vanilla JavaScript with Chart.js
- **Data Generation**: Python with yfinance, pandas, numpy
- **Deployment**: Cloudflare Pages
- **Automation**: GitHub Actions (hourly data updates)

## Local Development

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Generate data locally:
```bash
python generate_data.py
```

3. Serve frontend:
```bash
cd frontend
python -m http.server 8000
```

4. Open http://localhost:8000

## Deployment

Data is automatically generated hourly by GitHub Actions and deployed to Cloudflare Pages. The workflow:

1. GitHub Actions runs `generate_data.py` every hour
2. Updates `frontend/data.json` with fresh market data
3. Commits changes to repository
4. Cloudflare Pages auto-deploys the updated site

## License

MIT
