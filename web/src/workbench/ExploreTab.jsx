import { useState } from 'react'
import StatTile from '../components/StatTile'
import { fmtCell } from './ui'

const PREVIEW = 20

const distinct = (values) => new Set(values.filter((v) => v !== null && !(typeof v === 'number' && Number.isNaN(v))).map((v) => `${typeof v}:${v}`)).size

// Streamlit's Dataset Explorer tab: pick an uploaded dataset, see its size, a preview, its columns and the
// generator manifest whose file name matches it.
export default function ExploreTab({ datasets, manifests }) {
  const [name, setName] = useState(null)
  const [all, setAll] = useState(false)
  const current = datasets.find((d) => d.name === name) || datasets[0]
  if (!current) {
    return (
      <div className="empty">
        Add one or more synthetic CSV files above to begin. A generator writes them to <code>output/synthetic/</code>, for example{' '}
        <code>python -m generators.generate --generator gaussian_copula --seed 42</code>.
      </div>
    )
  }
  const f = current.frame
  const missing = f.columns.reduce((a, c) => a + f.data[c].filter((v) => v === null).length, 0)
  const rows = Math.min(f.n, all ? f.n : PREVIEW)
  const stem = current.name.replace(/\.csv$/i, '')
  const match = manifests.filter((m) => !m.error && m.name.includes(stem))

  return (
    <div className="stack">
      <div className="filters">
        <label className="field">
          <span>Dataset</span>
          <select className="select" value={current.name} onChange={(e) => { setName(e.target.value); setAll(false) }}>
            {datasets.map((d) => (
              <option key={d.name} value={d.name}>{d.name}</option>
            ))}
          </select>
        </label>
      </div>

      <div className="grid three">
        <StatTile label="Rows" value={f.n} />
        <StatTile label="Columns" value={f.columns.length} />
        <StatTile label="Missing values" value={missing} />
      </div>

      <figure className="card" style={{ margin: 0 }}>
        <div className="chart-head">
          <div>
            <h3>Preview</h3>
            <p className="sub">{all ? `All ${f.n} rows` : `First ${rows} of ${f.n} rows`} of <span className="num">{current.name}</span>.</p>
          </div>
          {f.n > PREVIEW && (
            <div className="chart-tools">
              <button className="tbtn" onClick={() => setAll(!all)}>{all ? `Show first ${PREVIEW}` : `Show all ${f.n}`}</button>
            </div>
          )}
        </div>
        <div className="sheet" tabIndex={0} role="region" aria-label={`Rows of ${current.name}, scrollable`}>
          <table>
            <thead>
              <tr>
                <th scope="col" className="n rownum" />
                {f.columns.map((c) => (
                  <th key={c} scope="col">{c}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Array.from({ length: rows }, (_, i) => (
                <tr key={i}>
                  <td className="n rownum">{i}</td>
                  {f.columns.map((c) => (
                    <td key={c} className={typeof f.data[c][i] === 'number' ? 'n' : undefined}>{fmtCell(f.data[c][i], f.dtypes[c])}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </figure>

      <div className="grid two">
        <div className="card">
          <h3>Columns</h3>
          <div className="sheet short" style={{ marginTop: 12 }} tabIndex={0} role="region" aria-label="Columns, scrollable">
            <table>
              <thead>
                <tr>
                  <th scope="col">Column</th>
                  <th scope="col">Data type</th>
                  <th scope="col" className="n">Missing values</th>
                  <th scope="col" className="n">Distinct values</th>
                </tr>
              </thead>
              <tbody>
                {f.columns.map((c) => (
                  <tr key={c}>
                    <td>{c}</td>
                    <td className="muted">{f.dtypes[c]}</td>
                    <td className="n">{f.data[c].filter((v) => v === null).length}</td>
                    <td className="n">{distinct(f.data[c])}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="card">
          <h3>Generator manifest</h3>
          {match.length ? (
            <>
              <p className="muted small" style={{ marginTop: 4 }}>
                From <span className="num">{match[0].name}</span>{match.length > 1 ? `, the first of ${match.length} files that match this dataset's name` : ''}.
              </p>
              <pre className="json">{JSON.stringify(match[0].json, null, 2)}</pre>
            </>
          ) : (
            <p className="muted small" style={{ marginTop: 6 }}>
              {manifests.length
                ? `None of the uploaded manifests has "${stem}" in its file name.`
                : `Add the manifest a generator wrote next to this file (${stem}.manifest.json) to see how it was made.`}
            </p>
          )}
        </div>
      </div>
    </div>
  )
}
