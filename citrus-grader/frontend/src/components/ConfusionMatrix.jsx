function ConfusionMatrix({ confusion }) {
  const values = confusion.matrix.flat()
  const maximum = Math.max(...values, 1)

  return (
    <div className="confusion-scroll">
      <table className="confusion-table">
        <caption>Rows are actual classes; columns are predicted classes.</caption>
        <thead>
          <tr>
            <th scope="col">Actual ↓ / Predicted →</th>
            {confusion.labels.map((label) => (
              <th scope="col" key={label}>{label.replaceAll('-', ' ')}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {confusion.matrix.map((row, rowIndex) => (
            <tr key={confusion.labels[rowIndex]}>
              <th scope="row">{confusion.labels[rowIndex].replaceAll('-', ' ')}</th>
              {row.map((count, columnIndex) => {
                const intensity = count / maximum
                return (
                  <td
                    key={`${rowIndex}-${columnIndex}`}
                    style={{
                      backgroundColor: `rgba(55, 119, 71, ${0.1 + intensity * 0.72})`,
                      color: intensity > 0.55 ? '#fff' : '#28342b',
                    }}
                    aria-label={`${confusion.labels[rowIndex]} predicted as ${confusion.labels[columnIndex]}: ${count}`}
                  >
                    {count.toLocaleString()}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default ConfusionMatrix
