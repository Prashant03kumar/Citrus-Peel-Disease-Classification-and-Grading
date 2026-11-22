function HistoryList({ predictions, onClear }) {
  return (
    <section className="history-section" aria-labelledby="history-title">
      <div className="history-heading">
        <div>
          <p className="eyebrow">THIS SESSION</p>
          <h3 id="history-title">Recent predictions</h3>
        </div>
        {predictions.length > 0 && (
          <button className="text-button" type="button" onClick={onClear}>
            Clear history
          </button>
        )}
      </div>
      {predictions.length === 0 ? (
        <p className="history-empty">Your classifications will appear here.</p>
      ) : (
        <ul className="history-list">
          {predictions.map((prediction) => (
            <li className="history-item" key={prediction.id}>
              <img src={prediction.thumbnail} alt="" />
              <div className="history-info">
                <strong>{prediction.predicted_class.replaceAll('-', ' ')}</strong>
                <span>{(prediction.confidence * 100).toFixed(1)}% confidence</span>
              </div>
              <span className={`grade-badge grade-${prediction.grade.toLowerCase()}`}>
                {prediction.grade}
              </span>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}

export default HistoryList
