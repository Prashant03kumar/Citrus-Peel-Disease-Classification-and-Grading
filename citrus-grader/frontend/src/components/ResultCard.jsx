function gradeTone(grade) {
  const knownGrades = ['A', 'B', 'Reject', 'Review']
  return knownGrades.includes(grade) ? grade.toLowerCase() : 'neutral'
}

function ResultCard({ result, classes }) {
  const classInfo = classes.find((item) => item.name === result.predicted_class)
  const displayName =
    classInfo?.display_name || result.predicted_class.replaceAll('-', ' ')
  const confidencePercent = result.confidence * 100

  return (
    <section className="result-card" aria-live="polite" aria-labelledby="result-title">
      <div className="result-topline">
        <p className="eyebrow">CLASSIFICATION RESULT</p>
        <span className={`grade-badge grade-${gradeTone(result.grade)}`}>
          {result.grade}
        </span>
      </div>
      <h3 id="result-title" className="result-class">{displayName}</h3>
      <p className="result-decision">{result.decision}</p>

      <div className="confidence-block">
        <div className="confidence-label">
          <span>Confidence</span>
          <strong>{confidencePercent.toFixed(1)}%</strong>
        </div>
        <div
          className="progress-track"
          role="progressbar"
          aria-label="Prediction confidence"
          aria-valuemin="0"
          aria-valuemax="100"
          aria-valuenow={confidencePercent}
        >
          <span style={{ width: `${confidencePercent}%` }} />
        </div>
      </div>

      <div className="probability-list">
        <h4>Class probabilities</h4>
        {Object.entries(result.probabilities).map(([className, probability]) => (
          <div className="probability-row" key={className}>
            <span>{className.replaceAll('-', ' ')}</span>
            <span className="probability-track">
              <span style={{ width: `${probability * 100}%` }} />
            </span>
            <strong>{(probability * 100).toFixed(1)}%</strong>
          </div>
        ))}
      </div>
      <p className="inference-time">
        Inference time <strong>{result.inference_ms.toFixed(1)} ms</strong>
      </p>
    </section>
  )
}

export default ResultCard
