import { useMemo, useState } from 'react'
import ChartCard from '../components/ChartCard'
import SegmentedControl from '../components/SegmentedControl'
import { subset } from './csv'
import { checkContract } from './contract'
import { comparableColumns, correlationGap, describe, fidelityTable, officialFidelity, profileMatchRate, valueShares } from './fidelity'
import { Check, f4, pct } from './ui'

const TOP = 15
const SHARE_TOP = 12
const REFS = [
  { value: 'train', label: 'Train split' },
  { value: 'holdout', label: 'Holdout split' },
  { value: 'all', label: 'All rows' },
]
const STATS = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']

const attempt = (fn) => {
  try {
    return { value: fn(), error: null }
  } catch (e) {
    return { value: null, error: e.message || String(e) }
  }
}

function Bar({ v, max, color = 'var(--c-blue)' }) {
  return (
    <span className="track">
      <i style={{ width: `${max > 0 && Number.isFinite(v) ? Math.min(100, (v / max) * 100) : 0}%`, background: color }} />
    </span>
  )
}

// Streamlit's Compare to Real Data tab: the Stage 2 contract check, fidelity against the chosen real reference,
// a memorization signal, per-column divergence and a column-by-column comparison.
export default function CompareTab({ datasets, real }) {
  const [name, setName] = useState(null)
  const [ref, setRef] = useState('train')
  const [allCols, setAllCols] = useState(false)
  const [col, setCol] = useState(null)
  const current = datasets.find((d) => d.name === name) || datasets[0]

  const result = useMemo(() => {
    if (!current || !real) return null
    const synth = current.frame
    const hasSplit = real.frame.columns.includes('split')
    const reference = hasSplit && ref !== 'all' ? subset(real.frame, (i) => real.frame.data.split[i] === ref) : real.frame
    const contract = attempt(() => checkContract(synth, real.frame))
    const table = attempt(() => fidelityTable(reference, synth))
    const { categorical, numeric } = comparableColumns(reference, synth)
    const own = {
      meanJs: table.value ? mean(table.value.map((r) => r.js).filter((v) => !Number.isNaN(v))) : NaN,
      corrGap: attempt(() => correlationGap(reference, synth, numeric)),
    }
    const official = attempt(() => officialFidelity(reference, synth))
    const copied = profileMatchRate(reference, synth, categorical)
    const splits = hasSplit ? new Set(real.frame.data.split) : new Set()
    const baseline =
      splits.has('train') && splits.has('holdout')
        ? profileMatchRate(
            subset(real.frame, (i) => real.frame.data.split[i] === 'train'),
            subset(real.frame, (i) => real.frame.data.split[i] === 'holdout'),
            categorical,
          )
        : null
    return { synth, reference, hasSplit, contract, table, categorical, own, official, copied, baseline }
  }, [current, real, ref])

  if (!current) return <div className="empty">Add a synthetic dataset above.</div>
  if (!real) {
    return (
      <div className="empty">
        Add the real cleaned dataset above to compare against it. It is <code>output/mimic_demo_clean.csv</code> in the repository, with its{' '}
        <code>split</code> column, which lets the page compare against the training rows, the held-out rows, or both.
      </div>
    )
  }
  const r = result
  const meanJs = r.official.value ? r.official.value.meanJs : r.own.meanJs
  const corrGap = r.official.value ? r.official.value.corrGap : r.own.corrGap.value
  const rows = r.table.value || []
  const shown = allCols ? rows : rows.slice(0, TOP)
  const maxJs = Math.max(0.001, ...rows.map((x) => (Number.isFinite(x.js) ? x.js : 0)))
  const detailName = rows.some((x) => x.column === col) ? col : rows[0]?.column
  const detail = rows.find((x) => x.column === detailName)

  return (
    <div className="stack">
      <div className="filters">
        <label className="field">
          <span>Dataset to compare</span>
          <select className="select" value={current.name} onChange={(e) => setName(e.target.value)}>
            {datasets.map((d) => (
              <option key={d.name} value={d.name}>{d.name}</option>
            ))}
          </select>
        </label>
        {r.hasSplit && <SegmentedControl label="Compare against" value={ref} onChange={setRef} options={REFS} />}
        {r.hasSplit && <span className="small muted">{r.reference.n} real rows in the selected reference.</span>}
      </div>

      {r.contract.error ? (
        <Check kind="warn" lead="The contract check could not run on this file:">{r.contract.error}</Check>
      ) : r.contract.value.length ? (
        <Check kind="fail" lead={`Fails the Stage 2 output contract (${r.contract.value.length} issue${r.contract.value.length > 1 ? 's' : ''}):`}>
          <ul>
            {r.contract.value.map((m) => (
              <li key={m}>{m}</li>
            ))}
          </ul>
        </Check>
      ) : (
        <Check kind="pass" lead="Passes the Stage 2 output contract.">
          Checked with a port of Track 1&rsquo;s <code>generators.validate</code>, against the real file&rsquo;s training rows.
        </Check>
      )}

      {r.official.error && (
        <Check kind="warn" lead="Track 2's metric could not run on this file">({r.official.error}); showing the page&rsquo;s own fidelity numbers.</Check>
      )}

      <div className="grid three">
        <div className="card tile">
          <div className="label">Mean JS divergence ↓︎</div>
          <div className="value num">{f4(meanJs)}</div>
          <div className="note">{r.official.value ? "Track 2's evaluation.fidelity_metrics" : "the page's own binned JS"}</div>
        </div>
        <div className="card tile">
          <div className="label">Correlation gap ↓︎</div>
          <div className="value num">{f4(corrGap)}</div>
          <div className="note">mean absolute difference of the two correlation matrices</div>
        </div>
        <div className="card tile">
          <div className="label">Rows matching a real patient ↓︎</div>
          <div className="value num">{pct(r.copied)}</div>
          {r.baseline != null && !Number.isNaN(r.copied) && (
            <div className="delta">
              <span aria-hidden="true">{r.copied > r.baseline ? '▲' : r.copied < r.baseline ? '▼' : '='}</span>
              {`${r.copied - r.baseline >= 0 ? '+' : ''}${((r.copied - r.baseline) * 100).toFixed(1)}% vs real unseen rows`}
            </div>
          )}
        </div>
      </div>
      <p className="small muted" style={{ maxWidth: 900 }}>
        Lower divergence means the columns look more like the real ones. The correlation gap is the mean absolute difference between the two correlation
        matrices. The last number is the share of rows whose {r.categorical.slice(0, 4).join(', ')} and the other categorical fields exactly match a real patient
        {r.baseline != null
          ? `; real rows the generator never saw match at ${pct(r.baseline)}, so that is the level to expect from a generator that isn't copying.`
          : '. Add the real dataset with its split column to see the level real unseen rows reach.'}{' '}
        It is a memorization signal, not the full privacy evaluation.
      </p>

      {r.table.error ? (
        <Check kind="warn" lead="Per-column divergence could not be computed:">{r.table.error}</Check>
      ) : (
        <ChartCard
          title="Per-column divergence"
          subtitle={`Jensen-Shannon divergence of each column against the ${ref === 'all' ? 'whole real file' : `real ${ref} split`}, worst first.`}
          better="lower"
          table={{
            caption: 'JS divergence per column',
            columns: [
              { key: 'c', label: 'Column' },
              { key: 'k', label: 'Kind' },
              { key: 'v', label: 'JS divergence ↓︎', numeric: true },
            ],
            rows: rows.map((x) => ({ c: x.column, k: x.kind, v: f4(x.js) })),
          }}
        >
          <div className="barlist">
            {shown.map((x) => (
              <div className="row" key={x.column}>
                <span className="name">
                  {x.column}
                  <span className="kindpill">{x.kind === 'numeric' ? 'num' : 'cat'}</span>
                </span>
                <Bar v={x.js} max={maxJs} />
                <span className="val">{f4(x.js)}</span>
              </div>
            ))}
          </div>
          {rows.length > TOP && (
            <p style={{ marginTop: 10 }}>
              <button className="tbtn" onClick={() => setAllCols(!allCols)}>{allCols ? `Show worst ${TOP}` : `Show all ${rows.length} columns`}</button>
            </p>
          )}
        </ChartCard>
      )}

      {detail && <ColumnDetail rows={rows} name={detailName} kind={detail.kind} onPick={setCol} reference={r.reference} synth={r.synth} />}
    </div>
  )
}

function ColumnDetail({ rows, name, kind, onPick, reference, synth }) {
  const numeric = kind === 'numeric'
  const real = numeric ? describe(reference.data[name]) : valueShares(reference.data[name], reference.dtypes[name])
  const syn = numeric ? (synth.columns.includes(name) ? describe(synth.data[name]) : null) : valueShares(synth.data[name], synth.dtypes[name])

  let levels = []
  if (!numeric) {
    const all = [...new Set([...Object.keys(real), ...Object.keys(syn)])].sort(
      (a, b) => Math.max(real[b] || 0, syn[b] || 0) - Math.max(real[a] || 0, syn[a] || 0) || (a < b ? -1 : 1),
    )
    const top = all.slice(0, SHARE_TOP)
    const rest = all.slice(SHARE_TOP)
    levels = top.map((k) => [k, real[k] || 0, syn[k] || 0])
    if (rest.length) levels.push([`Other (${rest.length} more)`, rest.reduce((a, k) => a + (real[k] || 0), 0), rest.reduce((a, k) => a + (syn[k] || 0), 0)])
  }
  const allLevels = numeric ? [] : [...new Set([...Object.keys(real), ...Object.keys(syn)])].sort()

  return (
    <ChartCard
      title="Column detail"
      subtitle={numeric ? 'Summary statistics, real against synthetic.' : 'Share of rows in each category, real against synthetic.'}
      table={
        numeric
          ? null
          : {
              caption: `Shares of ${name}`,
              columns: [
                { key: 'k', label: 'Category' },
                { key: 'r', label: 'Real', numeric: true },
                { key: 's', label: 'Synthetic', numeric: true },
              ],
              rows: allLevels.map((k) => ({ k, r: pct(real[k] || 0), s: pct(syn[k] || 0) })),
            }
      }
    >
      <div className="detail-pick">
        <label className="field">
          <span>Column</span>
          <select className="select" value={name} onChange={(e) => onPick(e.target.value)}>
            {rows.map((x) => (
              <option key={x.column} value={x.column}>{x.column}</option>
            ))}
          </select>
        </label>
        <span className="small muted">JS divergence {f4(rows.find((x) => x.column === name)?.js)}</span>
      </div>
      {numeric ? (
        <div className="tablewrap">
          <table className="statcompare">
            <thead>
              <tr>
                <th scope="col">Statistic</th>
                <th scope="col" className="n">Real</th>
                <th scope="col" className="n">Synthetic</th>
              </tr>
            </thead>
            <tbody>
              {STATS.map((k) => (
                <tr key={k}>
                  <td className="muted">{k}</td>
                  <td className="n">{k === 'count' ? real[k] : f4(real[k])}</td>
                  <td className="n">{syn ? (k === 'count' ? syn[k] : f4(syn[k])) : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <>
          <div className="share-head">
            <span />
            <span>Real</span>
            <span>Synthetic</span>
          </div>
          {levels.map(([k, a, b]) => (
            <div className="sharebar" key={k}>
              <span className="lab" title={k}>{k}</span>
              <span className="pair">
                <span className="track real"><i style={{ width: `${a * 100}%` }} /></span>
                <span className="v">{pct(a)}</span>
              </span>
              <span className="pair">
                <span className="track synth"><i style={{ width: `${b * 100}%` }} /></span>
                <span className="v">{pct(b)}</span>
              </span>
            </div>
          ))}
        </>
      )}
    </ChartCard>
  )
}

function mean(xs) {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN
}
