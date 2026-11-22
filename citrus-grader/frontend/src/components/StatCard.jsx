function StatCard({ label, value, note, accent = 'neutral' }) {
  return (
    <article className={`stat-card stat-card-${accent}`}>
      <p className="stat-label">{label}</p>
      <p className="stat-value">{value}</p>
      {note && <p className="stat-note">{note}</p>}
    </article>
  )
}

export default StatCard
