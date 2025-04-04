import { useState } from "react";
import axios from "axios";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

export default function PortfolioForm() {
  const [riskLevel, setRiskLevel] = useState("medium");
  const [tickers, setTickers] = useState("AAPL, MSFT, GOOGL");
  const [result, setResult] = useState(null);
  const [priceHistory, setPriceHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(
        "https://ai-portfolio-backend.onrender.com/optimize",
        {
          risk_level: riskLevel,
          tickers: tickers.split(",").map((t) => t.trim().toUpperCase()),
        }
      );

      setResult(response.data);
      setPriceHistory(response.data.price_history || []);
    } catch (err) {
      console.error("Optimization request failed:", err);
      setError("Failed to fetch optimization results. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const downloadCSV = () => {
    if (!result || !result.weights) return;

    const header = "Ticker,Weight\n";
    const rows = Object.entries(result.weights)
      .map(([ticker, weight]) => `${ticker},${(weight * 100).toFixed(2)}%`)
      .join("\n");

    const csv = header + rows;

    const blob = new Blob([csv], { type: "text/csv" });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.setAttribute("hidden", "");
    a.setAttribute("href", url);
    a.setAttribute("download", "portfolio_weights.csv");
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="max-w-xl mx-auto p-6 mt-10 bg-white rounded-xl shadow">
      <h1 className="text-2xl font-bold mb-6 text-center">AI Portfolio Optimizer</h1>

      <form onSubmit={handleSubmit}>
        <label htmlFor="tickers" className="block mb-2 font-medium">
          Enter Tickers (comma-separated):
        </label>
        <input
          id="tickers"
          type="text"
          value={tickers}
          onChange={(e) => setTickers(e.target.value)}
          placeholder="e.g. AAPL, MSFT, TSLA"
          className="w-full border border-gray-300 rounded p-2 mb-4"
        />

        <label htmlFor="risk" className="block mb-2 font-medium">
          Select Risk Level:
        </label>
        <select
          id="risk"
          value={riskLevel}
          onChange={(e) => setRiskLevel(e.target.value)}
          className="w-full border border-gray-300 rounded p-2 mb-4"
        >
          <option value="low">Low</option>
          <option value="medium">Medium</option>
          <option value="high">High</option>
        </select>

        <button
          type="submit"
          className="w-full bg-blue-600 text-white py-2 rounded hover:bg-blue-700 transition"
          disabled={loading}
        >
          {loading ? "Optimizing..." : "Optimize Portfolio"}
        </button>
      </form>

      {error && (
        <div className="mt-4 text-red-600 font-medium">{error}</div>
      )}

      {result && (
        <div className="mt-6">
          <h2 className="text-xl font-semibold mb-2">Results</h2>
          <p>
            <strong>Expected Return:</strong>{" "}
            {(result.expected_return * 100).toFixed(2)}%
          </p>
          <p>
            <strong>Volatility:</strong>{" "}
            {(result.expected_volatility * 100).toFixed(2)}%
          </p>
          <p>
            <strong>Sharpe Ratio:</strong> {result.sharpe_ratio.toFixed(2)}
          </p>

          <h3 className="mt-4 font-semibold">Portfolio Weights:</h3>
          <ul className="list-disc ml-6">
            {Object.entries(result.weights).map(([ticker, weight]) => (
              <li key={ticker}>
                {ticker}: {(weight * 100).toFixed(2)}%
              </li>
            ))}
          </ul>

          <button
            onClick={downloadCSV}
            className="mt-4 bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700 transition"
          >
            Export to CSV for Power BI
          </button>

          {priceHistory.length > 0 && (
            <div className="mt-8">
              <h3 className="text-lg font-semibold mb-2">Price History</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={priceHistory}>
                  <XAxis dataKey="Date" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  {tickers.split(",").map((ticker, idx) => (
                    <Line
                      key={ticker.trim()}
                      type="monotone"
                      dataKey={ticker.trim().toUpperCase()}
                      stroke={`hsl(${(idx * 60) % 360}, 70%, 50%)`}
                      strokeWidth={2}
                      dot={false}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
