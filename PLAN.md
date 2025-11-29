# Stock Dashboard Project Plan

## 1. Overview & Goal

Build a modern, real-time visual stock dashboard website.

**Key Features:**
*   Displays data for a predefined list of stocks and cryptocurrencies (BA, ABNB, SHOP, WDAY, SWBI, COIN, QCOM, AMD, NVDA, PLTR, EQIX, DIS, PFE, PSEC, SOXX, SPXL, TQQQ, TSLA, MSFT, GOOG, AMZN, AAPL, V, UAL, DAL, BTC-USD, LTC-USD).
*   Calculates and displays: Latest Price, Daily Change %, RSI 14, EMA 13, EMA 21, Max Drawdown %.
*   Data updates automatically every 1 minute.
*   Includes a manual refresh button.
*   (Optional) Includes news sentiment analysis for each stock.
*   Visual Style: Data-dense table layout, dark theme, smooth "ease-in-out" transitions for data updates, inspired by Robinhood.

## 2. Architecture

*   **Backend:** Python (using Flask framework)
    *   Handles data fetching from external sources.
    *   Performs indicator calculations.
    *   Provides a JSON API for the frontend.
    *   Serves the frontend static files.
*   **Frontend:** Standard HTML, CSS, and JavaScript
    *   Renders the user interface (table layout).
    *   Fetches data from the backend API periodically and on demand.
    *   Updates the UI dynamically with smooth transitions.
*   **Data Source:** `yfinance` library (initially). Monitor for rate limits; may need to switch to a dedicated API (e.g., Finnhub, Alpha Vantage) if issues arise.

## 3. Detailed Steps

### Step 1: Project Setup
*   Create project root directory: `stock-dashboard/`.
*   Create subdirectories: `backend/` and `frontend/`.
*   Set up Python virtual environment (`venv`) within `backend/`.
*   Install necessary Python libraries (`Flask`, `yfinance`, `pandas`, `numpy`) in the `venv`.

### Step 2: Backend Development (Python/Flask)
*   Create main Flask application file (`backend/app.py`).
*   Adapt the provided indicator calculation script into functions within the Flask app.
    *   Address potential pandas errors noted during planning (e.g., Series ambiguity).
*   Create an API endpoint (e.g., `/api/dashboard-data`) that:
    *   Accepts symbol list (or uses predefined list).
    *   Fetches data via `yfinance`.
    *   Calculates indicators.
    *   Returns data in JSON format (Symbol, Type, Price, Change%, RSI, EMA13, EMA21, Drawdown%, etc.).
*   Implement basic caching or data storage if needed (consider performance/rate limits).
*   Configure Flask to serve static files from the `frontend/` directory.

### Step 3: Frontend Development (HTML/CSS/JS)
*   **HTML (`frontend/index.html`):**
    *   Structure using a semantic `<table>`.
    *   Include `<thead>` for headers (Symbol, Price, Change%, RSI, EMA13, EMA21, Drawdown%, etc.).
    *   Include `<tbody>` for dynamic data rows (use `data-symbol` attributes on `<tr>`).
    *   Add a manual refresh button (`<button id="refresh-btn">Refresh</button>`).
*   **CSS (`frontend/style.css`):**
    *   Implement a **dark theme** (dark backgrounds, light text).
    *   Style the table for a **data-dense**, Robinhood-inspired look (compact padding, clear fonts).
    *   Use distinct colors (e.g., green/red) for positive/negative changes.
    *   Define CSS classes for visual cues (`.positive`, `.negative`, `.flash`).
    *   Implement **smooth transitions** (`transition: property duration ease-in-out;`) for relevant properties (e.g., `background-color`, `color`) on table cells (`<td>`) that update.
*   **JavaScript (`frontend/script.js`):**
    *   Define the list of stock/crypto symbols.
    *   Implement `fetchDashboardData()` to call the backend `/api/dashboard-data` API.
    *   Implement `updateDashboardUI(data)`:
        *   Iterate through received data.
        *   Find/create table rows (`<tr>`) and update cell (`<td>`) content.
        *   Apply/remove CSS classes for visual feedback on updates (e.g., color changes, brief flashes).
    *   Set up auto-refresh using `setInterval(fetchDashboardData, 60000)`.
    *   Add event listener to the manual refresh button to call `fetchDashboardData()`.
    *   Implement loading state indicators.
    *   Include error handling for API calls.

### Step 4: Integration & Refinement
*   Run the Flask backend.
*   Open `index.html` in the browser (served by Flask).
*   Test the complete data flow and UI updates.
*   Refine styles, transitions, and layout for polish.
*   Monitor `yfinance` performance and address rate limit issues if they occur.

### Step 5: (Optional) News Sentiment Analysis
*   **Backend:** Integrate a news API and sentiment analysis tool. Add sentiment score to the JSON response.
*   **Frontend:** Add a "Sentiment" column to the table and update it via JavaScript.

## 4. Architecture Diagram (Mermaid)

```mermaid
graph TD
    User[User Browser] -- Requests HTML/CSS/JS --> Backend[Python/Flask Server]
    Backend -- Serves Static Files --> User

    subgraph Frontend (Browser - Dark Theme, Table Layout)
        HTML[index.html - Table Structure]
        CSS[style.css - Dark Theme, Dense Layout, Transitions]
        JavaScript[script.js - API Calls, DOM Updates, Refresh Logic]
    end

    subgraph Backend (Server)
        Flask[Flask App - app.py]
        API[API Endpoint /api/dashboard-data]
        Logic[Data Fetching & Calculation Logic]
        yFinance[yfinance Library]
    end

    User -- 1. API Call (JS Fetch - every 60s / manual) --> API
    API -- 2. Trigger Logic --> Logic
    Logic -- 3. Fetch Data --> yFinance
    yFinance -- 4. Stock Data --> Logic
    Logic -- 5. Calculate & Format --> API
    API -- 6. Return JSON Data --> User
    User -- 7. Update Table UI (JS DOM Manipulation w/ CSS Transitions) --> HTML