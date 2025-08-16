import { useState, useEffect } from "react";
import axios from "axios";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
} from "recharts";

export default function PortfolioForm() {
  const [riskLevel, setRiskLevel] = useState("medium");
  const [tickers, setTickers] = useState("AAPL, MSFT, GOOGL");
  const [result, setResult] = useState(null);
  const [priceHistory, setPriceHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [inputError, setInputError] = useState("");
  const [portfolioHistory, setPortfolioHistory] = useState([]);

  // Load portfolio history from localStorage on component mount
  useEffect(() => {
    const saved = localStorage.getItem("portfolioHistory");
    if (saved) {
      try {
        setPortfolioHistory(JSON.parse(saved));
      } catch (e) {
        console.error("Failed to load portfolio history:", e);
      }
    }
  }, []);

  // Save portfolio to history
  const saveToHistory = (portfolioData) => {
    const newEntry = {
      id: Date.now(),
      timestamp: new Date().toISOString(),
      tickers: tickers,
      riskLevel: riskLevel,
      ...portfolioData,
    };
    
    const updatedHistory = [newEntry, ...portfolioHistory.slice(0, 9)]; // Keep last 10
    setPortfolioHistory(updatedHistory);
    localStorage.setItem("portfolioHistory", JSON.stringify(updatedHistory));
  };

  // Validate ticker input
  const validateTickers = (tickerString) => {
    const tickerList = tickerString.split(",").map((t) => t.trim().toUpperCase());
    const errors = [];

    if (tickerList.length === 0) {
      errors.push("At least one ticker is required");
    }

    if (tickerList.length > 20) {
      errors.push("Maximum 20 tickers allowed");
    }

    const tickerPattern = /^[A-Z]{1,5}$/;
    const invalidTickers = tickerList.filter(t => !tickerPattern.test(t));
    if (invalidTickers.length > 0) {
      errors.push(`Invalid ticker format: ${invalidTickers.join(", ")}`);
    }

    return { isValid: errors.length === 0, errors, tickers: tickerList };
  };

  const handleTickerChange = (e) => {
    const value = e.target.value;
    setTickers(value);
    
    if (value.trim()) {
      const validation = validateTickers(value);
      setInputError(validation.isValid ? "" : validation.errors[0]);
    } else {
      setInputError("");
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    // Validate input before sending
    const validation = validateTickers(tickers);
    if (!validation.isValid) {
      setInputError(validation.errors[0]);
      setLoading(false);
      return;
    }

    try {
      // Try local development first, fallback to production
      const apiUrl = window.location.hostname === 'localhost' 
        ? "http://localhost:8001/optimize"
        : "https://ai-portfolio-backend.onrender.com/optimize";
        
      const response = await axios.post(apiUrl, {
        risk_level: riskLevel,
        tickers: validation.tickers,
      });

      setResult(response.data);
      setPriceHistory(response.data.price_history || []);
      saveToHistory(response.data);
      setInputError("");
    } catch (err) {
      console.error("Optimization request failed:", err);
      if (err.response?.data?.detail) {
        setError(err.response.data.detail);
      } else {
        setError("Failed to fetch optimization results. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  };

  const downloadCSV = () => {
    if (!result || !result.weights) return;

    const header = "Ticker,Weight,Expected_Return,Volatility,Sharpe_Ratio\n";
    const rows = Object.entries(result.weights)
      .map(([ticker, weight]) => 
        `${ticker},${(weight * 100).toFixed(2)}%,${(result.expected_return * 100).toFixed(2)}%,${(result.expected_volatility * 100).toFixed(2)}%,${result.sharpe_ratio.toFixed(2)}`
      )
      .join("\n");

    const csv = header + rows;

    const blob = new Blob([csv], { type: "text/csv" });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.setAttribute("hidden", "");
    a.setAttribute("href", url);
    a.setAttribute("download", "portfolio_analysis.csv");
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  // Prepare data for pie chart
  const preparePieData = () => {
    if (!result?.weights) return [];
    return Object.entries(result.weights)
      .filter(([_, weight]) => weight > 0.01) // Only show significant allocations
      .map(([ticker, weight]) => ({
        name: ticker,
        value: weight * 100,
        weight: weight,
      }))
      .sort((a, b) => b.value - a.value);
  };

  // Colors for pie chart
  const COLORS = [
    '#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8',
    '#82CA9D', '#FFC658', '#FF7C7C', '#8DD1E1', '#D084D0'
  ];

  // Load a portfolio from history
  const loadPortfolio = (portfolio) => {
    setTickers(portfolio.tickers);
    setRiskLevel(portfolio.riskLevel);
    setResult(portfolio);
    setPriceHistory(portfolio.price_history || []);
  };

  return (
    <div className="max-w-6xl mx-auto p-6 mt-10 space-y-6">
      <div className="bg-white rounded-xl shadow p-6">
        <h1 className="text-3xl font-bold mb-6 text-center text-gray-800">
          AI Portfolio Optimizer
        </h1>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="tickers" className="block mb-2 font-medium text-gray-700">
              Enter Tickers (comma-separated):
            </label>
            <input
              id="tickers"
              type="text"
              value={tickers}
              onChange={handleTickerChange}
              placeholder="e.g. AAPL, MSFT, GOOGL, TSLA"
              className={`w-full border rounded-lg p-3 transition-colors ${
                inputError ? 'border-red-500 bg-red-50' : 'border-gray-300 focus:border-blue-500'
              }`}
            />
            {inputError && (
              <p className="mt-1 text-sm text-red-600">{inputError}</p>
            )}
            <p className="mt-1 text-sm text-gray-500">
              Maximum 20 tickers, 1-5 letter symbols only
            </p>
          </div>

          <div>
            <label htmlFor="risk" className="block mb-2 font-medium text-gray-700">
              Select Risk Level:
            </label>
            <select
              id="risk"
              value={riskLevel}
              onChange={(e) => setRiskLevel(e.target.value)}
              className="w-full border border-gray-300 rounded-lg p-3 focus:border-blue-500 transition-colors"
            >
              <option value="low">Low Risk (Conservative)</option>
              <option value="medium">Medium Risk (Balanced)</option>
              <option value="high">High Risk (Aggressive)</option>
            </select>
          </div>

          <button
            type="submit"
            className={`w-full py-3 rounded-lg font-medium transition-all ${
              loading || inputError
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-blue-600 hover:bg-blue-700 text-white shadow-lg hover:shadow-xl'
            }`}
            disabled={loading || inputError}
          >
            {loading ? (
              <span className="flex items-center justify-center">
                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Optimizing Portfolio...
              </span>
            ) : (
              "Optimize Portfolio"
            )}
          </button>
        </form>

        {error && (
          <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-red-700 font-medium">{error}</p>
          </div>
        )}
      </div>

      {/* Portfolio History Sidebar */}
      {portfolioHistory.length > 0 && (
        <div className="bg-white rounded-xl shadow p-6">
          <h3 className="text-lg font-semibold mb-4 text-gray-800">Recent Portfolios</h3>
          <div className="space-y-2 max-h-60 overflow-y-auto">
            {portfolioHistory.map((portfolio) => (
              <div
                key={portfolio.id}
                className="p-3 bg-gray-50 rounded-lg cursor-pointer hover:bg-gray-100 transition-colors"
                onClick={() => loadPortfolio(portfolio)}
              >
                <div className="flex justify-between items-center">
                  <span className="font-medium text-sm">{portfolio.tickers}</span>
                  <span className="text-xs text-gray-500">
                    {new Date(portfolio.timestamp).toLocaleDateString()}
                  </span>
                </div>
                <div className="text-xs text-gray-600 mt-1">
                  {portfolio.riskLevel} risk • Sharpe: {portfolio.sharpe_ratio?.toFixed(2)}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {result && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Portfolio Metrics */}
          <div className="bg-white rounded-xl shadow p-6">
            <h2 className="text-2xl font-bold mb-6 text-gray-800">Portfolio Analysis</h2>
            
            <div className="grid grid-cols-2 gap-4 mb-6">
              <div className="bg-blue-50 p-4 rounded-lg">
                <h4 className="text-sm font-medium text-blue-600 mb-1">Expected Return</h4>
                <p className="text-2xl font-bold text-blue-900">
                  {(result.expected_return * 100).toFixed(2)}%
                </p>
              </div>
              
              <div className="bg-green-50 p-4 rounded-lg">
                <h4 className="text-sm font-medium text-green-600 mb-1">Volatility</h4>
                <p className="text-2xl font-bold text-green-900">
                  {(result.expected_volatility * 100).toFixed(2)}%
                </p>
              </div>
              
              <div className="bg-purple-50 p-4 rounded-lg">
                <h4 className="text-sm font-medium text-purple-600 mb-1">Sharpe Ratio</h4>
                <p className="text-2xl font-bold text-purple-900">
                  {result.sharpe_ratio.toFixed(2)}
                </p>
              </div>
              
              <div className="bg-orange-50 p-4 rounded-lg">
                <h4 className="text-sm font-medium text-orange-600 mb-1">Assets</h4>
                <p className="text-2xl font-bold text-orange-900">
                  {result.num_assets || Object.keys(result.weights).length}
                </p>
              </div>
            </div>

            {/* Additional Risk Metrics */}
            {(result.var_95 || result.max_drawdown) && (
              <div className="border-t pt-4">
                <h4 className="font-semibold text-gray-700 mb-3">Risk Metrics</h4>
                <div className="space-y-2 text-sm">
                  {result.var_95 && (
                    <div className="flex justify-between">
                      <span className="text-gray-600">Value at Risk (95%):</span>
                      <span className="font-medium">{(result.var_95 * 100).toFixed(2)}%</span>
                    </div>
                  )}
                  {result.max_drawdown && (
                    <div className="flex justify-between">
                      <span className="text-gray-600">Max Drawdown:</span>
                      <span className="font-medium">{(result.max_drawdown * 100).toFixed(2)}%</span>
                    </div>
                  )}
                  {result.diversification_ratio && (
                    <div className="flex justify-between">
                      <span className="text-gray-600">Diversification Ratio:</span>
                      <span className="font-medium">{result.diversification_ratio.toFixed(2)}</span>
                    </div>
                  )}
                </div>
              </div>
            )}
            
            <button
              onClick={downloadCSV}
              className="mt-6 w-full bg-green-600 text-white py-3 rounded-lg hover:bg-green-700 transition-colors font-medium"
            >
              📊 Export Portfolio Analysis
            </button>
          </div>

          {/* Portfolio Allocation Visualization */}
          <div className="bg-white rounded-xl shadow p-6">
            <h3 className="text-xl font-bold mb-4 text-gray-800">Portfolio Allocation</h3>
            
            {/* Pie Chart */}
            <div className="mb-6">
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={preparePieData()}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={120}
                    paddingAngle={2}
                    dataKey="value"
                  >
                    {preparePieData().map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => `${value.toFixed(2)}%`} />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>

            {/* Weights List */}
            <div>
              <h4 className="font-semibold text-gray-700 mb-3">Detailed Weights</h4>
              <div className="space-y-2 max-h-48 overflow-y-auto">
                {Object.entries(result.weights)
                  .filter(([_, weight]) => weight > 0.001)
                  .sort(([,a], [,b]) => b - a)
                  .map(([ticker, weight], index) => (
                    <div key={ticker} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                      <div className="flex items-center">
                        <div 
                          className="w-4 h-4 rounded mr-3"
                          style={{ backgroundColor: COLORS[index % COLORS.length] }}
                        ></div>
                        <span className="font-medium">{ticker}</span>
                      </div>
                      <span className="text-gray-600">{(weight * 100).toFixed(2)}%</span>
                    </div>
                  ))
                }
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Price History Chart */}
      {priceHistory.length > 0 && (
        <div className="bg-white rounded-xl shadow p-6">
          <h3 className="text-xl font-bold mb-4 text-gray-800">
            Normalized Price Performance (% Change)
          </h3>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={priceHistory}>
              <XAxis 
                dataKey="Date" 
                tick={{ fontSize: 12 }}
                angle={-45}
                textAnchor="end"
                height={60}
              />
              <YAxis 
                label={{ value: 'Price (%)', angle: -90, position: 'insideLeft' }}
                tick={{ fontSize: 12 }}
              />
              <Tooltip 
                labelFormatter={(value) => `Date: ${value}`}
                formatter={(value, name) => [`${value.toFixed(2)}%`, name]}
              />
              <Legend />
              {result.valid_tickers?.map((ticker, idx) => (
                <Line
                  key={ticker}
                  type="monotone"
                  dataKey={ticker}
                  stroke={COLORS[idx % COLORS.length]}
                  strokeWidth={2}
                  dot={false}
                  connectNulls
                />
              )) || 
              tickers.split(",").map((ticker, idx) => (
                <Line
                  key={ticker.trim()}
                  type="monotone"
                  dataKey={ticker.trim().toUpperCase()}
                  stroke={COLORS[idx % COLORS.length]}
                  strokeWidth={2}
                  dot={false}
                  connectNulls
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
