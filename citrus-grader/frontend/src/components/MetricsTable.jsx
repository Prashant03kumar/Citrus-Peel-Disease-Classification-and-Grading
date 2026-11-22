function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`
}

function f1Tone(value) {
  if (value >= 0.9) return 'f1-strong'
  if (value >= 0.8) return 'f1-watch'
  return 'f1-low'
}

function MetricsTable({ metrics }) {
  return (
    <div className="table-scroll">
      <table className="data-table">
        <thead>
          <tr>
            <th scope="col">Class</th>
            <th scope="col">Precision</th>
            <th scope="col">Recall</th>
            <th scope="col">F1</th>
            <th scope="col">Support</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(metrics).map(([className, values]) => (
            <tr key={className}>
              <th scope="row">{className.replaceAll('-', ' ')}</th>
              <td>{formatPercent(values.precision)}</td>
              <td>{formatPercent(values.recall)}</td>
              <td>
                <span className={`metric-pill ${f1Tone(values.f1)}`}>
                  {formatPercent(values.f1)}
                </span>
              </td>
              <td>{values.support.toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default MetricsTable
