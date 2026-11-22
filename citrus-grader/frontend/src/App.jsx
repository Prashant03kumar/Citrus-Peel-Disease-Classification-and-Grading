import { useCallback, useEffect, useRef, useState } from 'react'
import {
  getClasses,
  getHealth,
  getMetrics,
} from './api'
import ClassGallery from './components/ClassGallery'
import ConfusionMatrix from './components/ConfusionMatrix'
import HistoryList from './components/HistoryList'
import MetricsTable from './components/MetricsTable'
import StatCard from './components/StatCard'
import Uploader from './components/Uploader'
import './App.css'

function percentage(value) {
  return `${(value * 100).toFixed(1)}%`
}

function DashboardSkeleton() {
  return (
    <div className="dashboard-skeleton" aria-label="Loading dashboard data">
      {Array.from({ length: 3 }, (_, index) => (
        <div className="skeleton-panel" key={index}>
          <span className="skeleton-line skeleton-line-short" />
          <span className="skeleton-line skeleton-line-long" />
          <span className="skeleton-line skeleton-line-medium" />
        </div>
      ))}
    </div>
  )
}

function App() {
  const [health, setHealth] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [classes, setClasses] = useState(null)
  const [connection, setConnection] = useState('checking')
  const [loading, setLoading] = useState(true)
  const [dashboardError, setDashboardError] = useState('')
  const [history, setHistory] = useState([])
  const historyUrls = useRef(new Set())

  const refreshDashboard = useCallback(async () => {
    setLoading(true)
    setDashboardError('')
    try {
      const [nextHealth, nextMetrics, nextClasses] = await Promise.all([
        getHealth(),
        getMetrics(),
        getClasses(),
      ])
      if (!nextHealth.model_loaded) {
        throw new Error('The API is available, but its model has not loaded yet.')
      }
      setHealth(nextHealth)
      setMetrics(nextMetrics)
      setClasses(nextClasses)
      setConnection('connected')
    } catch (error) {
      setConnection('offline')
      setDashboardError(error.message || 'The dashboard could not reach the API.')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    Promise.resolve().then(refreshDashboard)
  }, [refreshDashboard])

  useEffect(() => {
    if (!health) return undefined

    let active = true
    async function checkConnection() {
      try {
        const currentHealth = await getHealth()
        if (!active) return
        setConnection(currentHealth.model_loaded ? 'connected' : 'offline')
        if (!currentHealth.model_loaded) {
          setDashboardError('The API is available, but its model has not loaded yet.')
        }
      } catch (error) {
        if (!active) return
        setConnection('offline')
        setDashboardError(error.message || 'The dashboard could not reach the API.')
      }
    }

    const timer = setInterval(checkConnection, 4000)
    return () => {
      active = false
      clearInterval(timer)
    }
  }, [health])

  useEffect(() => {
    const activeUrls = new Set(history.map((prediction) => prediction.thumbnail))
    for (const url of historyUrls.current) {
      if (!activeUrls.has(url)) {
        URL.revokeObjectURL(url)
        historyUrls.current.delete(url)
      }
    }
  }, [history])

  useEffect(
    () => () => {
      for (const url of historyUrls.current) URL.revokeObjectURL(url)
    },
    [],
  )

  function addPrediction(prediction, file) {
    const thumbnail = URL.createObjectURL(file)
    historyUrls.current.add(thumbnail)
    setHistory((current) =>
      [
        {
          ...prediction,
          id: `${Date.now()}-${thumbnail}`,
          thumbnail,
        },
        ...current,
      ].slice(0, 10),
    )
  }

  function clearHistory() {
    setHistory([])
  }

  const connectionLabel =
    connection === 'connected'
      ? 'Backend connected'
      : connection === 'checking'
        ? 'Checking backend'
        : 'Backend not connected'

  return (
    <div className="app-shell">
      <header className="site-header">
        <a className="brand" href="#top" aria-label="Citrus Disease Grader home">
          <span className="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 36 36" role="presentation">
              <circle cx="18" cy="19" r="13" />
              <path d="M18 7c2-4 6-5 10-4-1 4-4 7-9 7M18 11c0 8-4 13-9 16" />
              <path d="M18 11c4 1 7 4 9 8" />
            </svg>
          </span>
          <span>Citrus Disease Grader</span>
        </a>
        <div
          className={`connection-badge connection-${connection}`}
          role="status"
          aria-live="polite"
        >
          <span className="connection-dot" />
          {connectionLabel}
        </div>
      </header>

      <main id="top">
        <section className="page-intro" aria-labelledby="page-title">
          <div className="intro-copy">
            <p className="eyebrow">QUALITY INSIGHTS · CITRUS SORTING</p>
            <h1 id="page-title">A clearer view of every peel.</h1>
            <p>
              Explore model performance, understand the dataset, and classify a
              fruit photo for a quick grading recommendation.
            </p>
          </div>
          <div className="intro-fruit" aria-hidden="true">
            <span className="fruit-leaf" />
            <span className="fruit-orbit fruit-orbit-one" />
            <span className="fruit-orbit fruit-orbit-two" />
            <span className="fruit-highlight" />
          </div>
        </section>

        {dashboardError ? (
          <section className="dashboard-error" role="alert">
            <div className="error-icon" aria-hidden="true">!</div>
            <div>
              <h2>We can’t load the dashboard right now</h2>
              <p>{dashboardError} Check that the backend is running, then retry.</p>
            </div>
            <button
              className="button button-secondary retry-button"
              type="button"
              onClick={refreshDashboard}
              disabled={loading}
            >
              {loading ? 'Retrying…' : 'Retry connection'}
            </button>
          </section>
        ) : loading || !metrics || !health || !classes ? (
          <DashboardSkeleton />
        ) : (
          <>
            <section className="dashboard-section" aria-labelledby="performance-title">
              <div className="section-heading">
                <div>
                  <p className="eyebrow">MODEL PERFORMANCE</p>
                  <h2 id="performance-title">Reliable at a glance</h2>
                </div>
                <p className="section-note">
                  Evaluation metrics from the current test split
                </p>
              </div>
              <div className="stats-grid performance-grid">
                <StatCard
                  label="Test accuracy"
                  value={percentage(metrics.accuracy)}
                  note="Correct predictions"
                  accent="orange"
                />
                <StatCard
                  label="Macro precision"
                  value={percentage(metrics.macro_precision)}
                  note="Across all classes"
                />
                <StatCard
                  label="Macro recall"
                  value={percentage(metrics.macro_recall)}
                  note="Across all classes"
                />
                <StatCard
                  label="Macro F1"
                  value={percentage(metrics.macro_f1)}
                  note="Balanced model score"
                />
                <StatCard
                  label="CPU inference"
                  value={`${metrics.speed.cpu_ms_per_image.toFixed(1)} ms`}
                  note={`${metrics.speed.cpu_images_per_sec.toFixed(1)} images per second`}
                  accent="green"
                />
              </div>
            </section>

            <section className="dashboard-section" aria-labelledby="dataset-title">
              <div className="section-heading">
                <div>
                  <p className="eyebrow">DATASET</p>
                  <h2 id="dataset-title">The data behind the model</h2>
                </div>
              </div>
              <div className="stats-grid dataset-grid">
                <StatCard label="Training images" value={metrics.dataset.train.toLocaleString()} />
                <StatCard label="Validation images" value={metrics.dataset.val.toLocaleString()} />
                <StatCard label="Test images" value={metrics.dataset.test.toLocaleString()} />
                <StatCard label="Classes" value={health.classes.length.toLocaleString()} />
              </div>
              <div className="panel split-panel">
                <div className="panel-heading">
                  <div>
                    <h3>Images per class and split</h3>
                    <p>Distribution across training, validation, and test data</p>
                  </div>
                  <div className="split-legend" aria-label="Dataset split legend">
                    <span><i className="legend-training" />Training</span>
                    <span><i className="legend-validation" />Validation</span>
                    <span><i className="legend-test" />Test</span>
                  </div>
                </div>
                <div className="class-split-list">
                  {Object.entries(metrics.dataset.per_class).map(([className, counts]) => {
                    const maxCount = Math.max(
                      ...Object.values(metrics.dataset.per_class).flatMap((values) =>
                        Object.values(values),
                      ),
                      1,
                    )
                    return (
                      <div className="class-split-row" key={className}>
                        <span className="split-class-name">
                          {className.replaceAll('-', ' ')}
                        </span>
                        <div className="split-bars">
                          {[
                            ['train', 'training'],
                            ['val', 'validation'],
                            ['test', 'test'],
                          ].map(([split, tone]) => (
                            <div className="split-bar-line" key={split}>
                              <span className="split-value">{counts[split].toLocaleString()}</span>
                              <span
                                className={`split-bar split-bar-${tone}`}
                                role="img"
                                aria-label={`${className} ${split}: ${counts[split]} images`}
                              >
                                <span style={{ width: `${(counts[split] / maxCount) * 100}%` }} />
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            </section>

            <section className="dashboard-section analysis-grid" aria-label="Model analysis">
              <div className="panel table-panel">
                <div className="panel-heading">
                  <div>
                    <p className="eyebrow">CLASS BREAKDOWN</p>
                    <h2>Per-class metrics</h2>
                  </div>
                </div>
                <MetricsTable metrics={metrics.per_class} />
              </div>
              <div className="panel confusion-panel">
                <div className="panel-heading">
                  <div>
                    <p className="eyebrow">PREDICTION DETAIL</p>
                    <h2>Confusion matrix</h2>
                  </div>
                </div>
                <ConfusionMatrix confusion={metrics.confusion_matrix} />
              </div>
            </section>

            {metrics.ablation !== null && (
              <section className="dashboard-section" aria-label="CLAHE ablation results">
                <div className="ablation-card">
                  <div className="ablation-mark" aria-hidden="true">↗</div>
                  <div>
                    <p className="eyebrow">PREPROCESSING COMPARISON</p>
                    <h2>CLAHE impact</h2>
                    <p className="ablation-summary">
                      Without CLAHE {percentage(metrics.ablation.baseline_accuracy)}
                      <span aria-hidden="true"> → </span>
                      With CLAHE {percentage(metrics.ablation.clahe_accuracy)}
                    </p>
                  </div>
                  <div className="ablation-difference">
                    {`${metrics.ablation.clahe_accuracy >= metrics.ablation.baseline_accuracy ? '+' : ''}${((metrics.ablation.clahe_accuracy - metrics.ablation.baseline_accuracy) * 100).toFixed(1)} pp`}
                    <span>difference</span>
                  </div>
                </div>
              </section>
            )}

            <section className="dashboard-section" aria-labelledby="gallery-title">
              <div className="section-heading">
                <div>
                  <p className="eyebrow">VISUAL REFERENCE</p>
                  <h2 id="gallery-title">Know the signs</h2>
                </div>
                <p className="section-note">Examples from each disease class</p>
              </div>
              <ClassGallery classes={classes} />
            </section>

            <section className="dashboard-section" aria-labelledby="upload-title">
              <div className="section-heading">
                <div>
                  <p className="eyebrow">PHOTO CLASSIFICATION</p>
                  <h2 id="upload-title">Check a citrus fruit</h2>
                </div>
                <p className="section-note">Upload a clear peel photo to get a grading suggestion</p>
              </div>
              <Uploader classes={classes} onPrediction={addPrediction} />
              <HistoryList predictions={history} onClear={clearHistory} />
            </section>
          </>
        )}
      </main>

      <footer className="site-footer">
        <span className="footer-mark" aria-hidden="true">●</span>
        Model: EfficientNet-B0 + CLAHE
      </footer>
    </div>
  )
}

export default App
