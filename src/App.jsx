import { useState } from "react";
import "./App.css";

const initialMetrics = {
  cpu_requested: 0.5,
  memory_requested: 0.7,
  cpu_utilization: 0.85,
  memory_utilization: 0.75,
  machine_cpu_utilization: 0.3,
  machine_memory_utilization: 0.4,
  queue_length: 58,
  priority: 7,
  scheduling_class: 2,
  network_latency_ms: 60,
  estimated_runtime_sec: 500,
  resource_contention: 0.8,
};

const fields = [
  { key: "cpu_requested", label: "CPU requested", min: 0, max: 1, step: 0.01 },
  { key: "memory_requested", label: "Memory requested", min: 0, max: 1, step: 0.01 },
  { key: "cpu_utilization", label: "CPU utilization", min: 0, max: 1, step: 0.01 },
  { key: "memory_utilization", label: "Memory utilization", min: 0, max: 1, step: 0.01 },
  { key: "machine_cpu_utilization", label: "Machine CPU load", min: 0, max: 1, step: 0.01 },
  { key: "machine_memory_utilization", label: "Machine memory load", min: 0, max: 1, step: 0.01 },
  { key: "queue_length", label: "Queue length", min: 0, max: 100, step: 1 },
  { key: "priority", label: "Task priority", min: 1, max: 10, step: 1 },
  { key: "scheduling_class", label: "Scheduling class", min: 0, max: 3, step: 1 },
  { key: "network_latency_ms", label: "Network latency (ms)", min: 0, max: 200, step: 1 },
  { key: "estimated_runtime_sec", label: "Estimated runtime (sec)", min: 1, max: 1000, step: 1 },
  { key: "resource_contention", label: "Resource contention", min: 0, max: 1, step: 0.01 },
];

function App() {
  const [metrics, setMetrics] = useState(initialMetrics);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  function updateMetric(key, value) {
    setMetrics((previous) => ({
      ...previous,
      [key]: Number(value),
    }));
    setResult(null);
  }

  async function predictViolation() {
    setLoading(true);
    setResult(null);

    try {
      const response = await fetch("http://127.0.0.1:8001/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(metrics),
      });

      if (!response.ok) {
        const details = await response.text();
        throw new Error(details || `Request failed (${response.status})`);
      }

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Prediction error:", error);

      setResult({
        error:
          "Unable to get a prediction. Check that your backend is running on port 8000 and that the API is accessible.",
      });
    } finally {
      setLoading(false);
    }
  }

  function resetInputs() {
    setMetrics({ ...initialMetrics });
    setResult(null);
  }

  const riskCards = [
    {
      label: "CPU utilization",
      value: `${(metrics.cpu_utilization * 100).toFixed(0)}%`,
      color: "blue",
    },
    {
      label: "Memory utilization",
      value: `${(metrics.memory_utilization * 100).toFixed(0)}%`,
      color: "purple",
    },
    {
      label: "Queue length",
      value: metrics.queue_length,
      color: "orange",
    },
    {
      label: "Network latency",
      value: `${metrics.network_latency_ms} ms`,
      color: "green",
    },
  ];

  const predictionLabel = result
    ? result.result ??
      (result.prediction === 1
        ? "SLA Violation Predicted"
        : result.prediction === 0
        ? "No SLA Violation Predicted"
        : result.Prediction ?? "Prediction received")
    : "";

  const probability = result
    ? Number(
        result.violation_probability ??
          result["Violation Probability (%)"] ??
          0
      )
    : 0;

  const isViolation =
    result?.prediction === 1 ||
    result?.result === "SLA Violation Predicted";

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">S</div>
          <div>
            <h2>SLA Guard</h2>
            <span>Cloud intelligence</span>
          </div>
        </div>

        <div className="nav-label">WORKSPACE</div>
        <div className="nav-item active">
          <span>◫</span> Overview
        </div>
        <div className="nav-item">
          <span>◈</span> SLA predictions
        </div>
        <div className="nav-item">
          <span>▥</span> Resource monitor
        </div>
        <div className="nav-item">
          <span>◷</span> Model insights
        </div>

        <div className="sidebar-bottom">
          <div className="status-dot" />
          <div>
            <strong>Prediction workspace</strong>
            <p>Development environment</p>
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <div className="breadcrumb">Workspace / Overview</div>
            <h1>Cloud SLA Overview</h1>
          </div>

          <div className="topbar-right">
            <span className="environment-badge">● Prototype</span>
            <div className="avatar">SG</div>
          </div>
        </header>

        <section className="welcome-section">
          <div>
            <p className="eyebrow">PREDICTIVE CLOUD MONITORING</p>
            <h2>Anticipate risk. Protect service reliability.</h2>
            <p className="welcome-text">
              Analyze workload conditions and assess the risk of a potential
              SLA violation.
            </p>
          </div>

          <div className="welcome-graphic">
            <div className="graphic-ring ring-one" />
            <div className="graphic-ring ring-two" />
            <div className="graphic-core">SLA</div>
          </div>
        </section>

        <section className="metrics-grid">
          {riskCards.map((card) => (
            <div className="metric-card" key={card.label}>
              <div className={`metric-icon ${card.color}`}>◈</div>
              <p>{card.label}</p>
              <h3>{card.value}</h3>
              <span className="metric-caption">Current input value</span>
            </div>
          ))}
        </section>

        <div className="section-heading">
          <div>
            <h2>Prediction workspace</h2>
            <p>Adjust task and machine parameters to assess SLA risk.</p>
          </div>

          <span className="model-badge">Random Forest · ML</span>
        </div>

        <section className="workspace-grid">
          <div className="panel input-panel">
            <div className="panel-heading">
              <div>
                <h3>Workload parameters</h3>
                <p>Configure the cloud task characteristics.</p>
              </div>
              <span className="panel-number">01</span>
            </div>

            <div className="input-grid">
              {fields.map((field) => (
                <label className="field" key={field.key}>
                  <span>{field.label}</span>

                  <div className="field-control">
                    <input
                      type="number"
                      min={field.min}
                      max={field.max}
                      step={field.step}
                      value={metrics[field.key]}
                      onChange={(event) =>
                        updateMetric(field.key, event.target.value)
                      }
                    />
                  </div>
                </label>
              ))}
            </div>

            <div className="panel-actions">
              <button
                className="primary-button"
                onClick={predictViolation}
                disabled={loading}
              >
                {loading ? "Analyzing workload..." : "Run SLA prediction →"}
              </button>

              <button
                className="secondary-button"
                onClick={resetInputs}
                disabled={loading}
              >
                Reset inputs
              </button>
            </div>
          </div>

          <div className="panel result-panel">
            <div className="panel-heading">
              <div>
                <h3>Prediction result</h3>
                <p>Model assessment for the current workload.</p>
              </div>
              <span className="panel-number">02</span>
            </div>

            {!result && !loading && (
              <div className="empty-result">
                <div className="result-orbit">
                  <span>↗</span>
                </div>

                <h3>Ready to analyze</h3>
                <p>
                  Adjust the workload parameters and run a prediction to see
                  the model response here.
                </p>

                <div className="result-placeholder">
                  <span>Prediction status</span>
                  <strong>Waiting for input</strong>
                </div>
              </div>
            )}

            {loading && (
              <div className="empty-result">
                <div className="result-orbit">
                  <span>...</span>
                </div>
                <h3>Analyzing workload</h3>
                <p>
                  Sending the current workload parameters to the Random Forest
                  prediction API.
                </p>
              </div>
            )}

            {result && (
              <div className="prediction-output">
                {result.error ? (
                  <div className="error-box">
                    <strong>Prediction request failed</strong>
                    <p>{result.error}</p>
                  </div>
                ) : (
                  <>
                    <div
                      className={`result-orbit ${
                        isViolation ? "risk-result" : "safe-result"
                      }`}
                    >
                      <span>{isViolation ? "!" : "✓"}</span>
                    </div>

                    <p className="eyebrow">MODEL RESPONSE</p>
                    <h2>{predictionLabel}</h2>

                    <div className="probability-card">
                      <span>Violation probability</span>
                      <strong>{probability.toFixed(1)}%</strong>
                    </div>

                    <div className="probability-track">
                      <div
                        className="probability-fill"
                        style={{
                          width: `${Math.min(100, Math.max(0, probability))}%`,
                        }}
                      />
                    </div>

                    {result.threshold !== undefined && (
                      <p className="result-note">
                        Decision threshold:{" "}
                        {Number(result.threshold) <= 1
                          ? `${(Number(result.threshold) * 100).toFixed(1)}%`
                          : `${Number(result.threshold).toFixed(1)}%`}
                      </p>
                    )}

                    <p className="result-note">
                      This is a model estimate, not a guarantee of an actual
                      SLA breach. Its reliability depends on the training data
                      and validation results.
                    </p>
                  </>
                )}
              </div>
            )}

            <div className="model-info">
              <span className="status-dot" />
              <div>
                <strong>
                  {result && !result.error
                    ? `Connected · ${result.model ?? "Random Forest"}`
                    : "Prediction API"}
                </strong>
                <p>
                  {result && !result.error
                    ? "A response was received from your ML backend."
                    : "Run a prediction to receive the model response."}
                </p>
              </div>
            </div>
          </div>
        </section>

        <footer>
          <span>SLA Guard · Research prototype</span>
          <span>
            Predictions require validation against real cloud trace data.
          </span>
        </footer>
      </main>
    </div>
  );
}

export default App;
