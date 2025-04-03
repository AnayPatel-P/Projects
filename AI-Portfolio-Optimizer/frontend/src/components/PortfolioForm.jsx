import { useState } from "react";
import axios from "axios";

export default function PortfolioForm() {
  const [riskLevel, setRiskLevel] = useState("medium");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post("http://localhost:8000/optimize", {
        risk_level: riskLevel,
      });
      setResult(response.data);
    } catch (err) {
      console.error("Optimization request failed:", err);
      setError("Failed to fetch optimization results. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-xl mx-auto p-6 mt-10 bg-white rounded-xl shadow">
      <h1 className="text-2xl font-bold mb-6 text-center">AI Portfolio Optimizer</h1>

      <form onSubmit={handleSubmit}>
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
        <div className="mt-4 text-red-600 font-medium">
          {error}
        </div>
      )}

      {result && (
        <div className="mt-6">
          <h2 className="text-xl font-semibold mb-2">Results</h2>
          <p><strong>Expected Return:</strong> {(result.expected_return * 100).toFixed(2)}%</p>
          <p><strong>Volatility:</strong> {(result.expected_volatility * 100).toFixed(2)}%</p>
          <p><strong>Sharpe Ratio:</strong> {result.sharpe_ratio.toFixed(2)}</p>

          <h3 className="mt-4 font-semibold">Portfolio Weights:</h3>
          <ul className="list-disc ml-6">
            {Object.entries(result.weights).map(([ticker, weight]) => (
              <li key={ticker}>
                {ticker}: {(weight * 100).toFixed(2)}%
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
