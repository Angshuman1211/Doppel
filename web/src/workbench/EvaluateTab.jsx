import { useMemo, useState } from 'react'
import ChartCard from '../components/ChartCard'
import DotPlot from '../charts/DotPlot'
import ScatterPlot from '../charts/ScatterPlot'
import Legend from '../components/Legend'
import { ARM_STYLE, armStyle } from '../lib/arms'
import { Swatch } from '../lib/marks'
import { niceTicks } from '../lib/format'
import { generatorSummary } from './reports'
import { Check, FileChips, FileDrop, YesNo, f4 } from './ui'

const AXES = [
  { key: 'fidelityJs', title: 'Fidelity', label: 'Mean JS divergence', better: 'lower', refs: [] },
  { key: 'utilityTstr', title: 'Utility', label: 'Mean TSTR AUROC', better: 'higher', refs: [{ value: 0.5, label: 'Guessing 0.5' }] },
  {
    key: 'privacyWorstMia',
    title: 'Privacy',
    label: 'Worst-case membership AUROC',
    better: 'lower',
    refs: [
      { value: 0.5, label: 'Guessing 0.5' },
      { value: 0.65, label: 'Fail line 0.65' },
    ],
  },
]

// A domain that holds every mean ± sd and every reference line, with round ticks.
function domainFor(rows, key, refs, floorZero) {
  const lo = Math.min(...rows.map((r) => r[key] - r.sd[key]), ...refs.map((r) => r.value))
  const hi = Math.max(...rows.map((r) => r[key] + r.sd[key]), ...refs.map((r) => r.value))
  const pad = Math.max((hi - lo) * 0.12, 0.005)
  const a = floorZero ? 0 : lo - pad
  const b = hi + pad
  const ticks = niceTicks(a, b, 5)
  return { domain: [Math.min(a, ticks[0]), Math.max(b, ticks[ticks.length - 1])], ticks }
}

const ORDER = Object.keys(ARM_STYLE)
const rankOf = (g) => (ORDER.includes(g) ? ORDER.indexOf(g) : ORDER.length)
const bySeed = (a, b) => rankOf(a.generator) - rankOf(b.generator) || (a.generator < b.generator ? -1 : a.generator > b.generator ? 1 : 0) || Number(a.seed) - Number(b.seed) || (a.seed < b.seed ? -1 : 1)

// Streamlit's Evaluation & Pareto tab: upload result JSONs, compare generators, and find the frontier.
export default function EvaluateTab({ reports, onAdd, onRemove, onClear }) {
  const good = reports.filter((r) => !r.report.error).map((r) => r.report)
  const broken = reports.filter((r) => r.report.error)
  const summary = useMemo(() => generatorSummary(good), [good])
  const order = Object.keys(ARM_STYLE)
  const rank = (g) => (order.includes(g) ? order.indexOf(g) : order.length)
  const styled = summary.map((s) => ({ ...s, style: armStyle(s.generator) })).sort((a, b) => rank(a.generator) - rank(b.generator))

  return (
    <div className="stack">
      <div>
        <FileDrop
          id="wb-reports"
          title="Evaluation reports (JSON)"
          hint="One file per generator and seed, as evaluation.eval_runner writes them to results/."
          accept=".json"
          multiple
          onFiles={onAdd}
        />
        <FileChips items={reports.map((r) => ({ name: r.name, error: r.report.error }))} onRemove={onRemove} onClear={onClear} />
      </div>

      {broken.length > 0 && (
        <details className="card report-list">
          <summary>
            {broken.length} report{broken.length > 1 ? 's' : ''} could not be read
          </summary>
          <ul className="file-errors">
            {broken.map((b) => (
              <li key={b.name}>
                <b>{b.name}</b>: {b.report.error}
              </li>
            ))}
          </ul>
        </details>
      )}

      {!good.length ? (
        <div className="empty">
          Add one or more evaluation report JSON files to compare generators. <code>python -m evaluation.eval_runner --generator gaussian_copula --seed 42</code> writes{' '}
          <code>results/gaussian_copula_seed42.json</code>; add a few seeds of several generators to see the frontier.
        </div>
      ) : (
        <>
          <p className="small muted">
            {good.length} report{good.length > 1 ? 's' : ''} loaded from {summary.length} generator{summary.length > 1 ? 's' : ''}. Metrics reflect the uploaded reports and seeds.
          </p>

          <ChartCard
            title="Generator comparison"
            subtitle="One row per uploaded report. Lower JS divergence is better fidelity, higher TSTR AUROC is better utility, lower membership AUROC means the attack did worse."
            table={null}
          >
            <div className="sheet short" tabIndex={0} role="region" aria-label="Uploaded reports, scrollable">
              <table>
                <thead>
                  <tr>
                    <th scope="col">Generator</th>
                    <th scope="col" className="n">Seed</th>
                    <th scope="col" className="n">Mean JS divergence ↓︎</th>
                    <th scope="col" className="n">Mean TSTR AUROC ↑︎</th>
                    <th scope="col" className="n">Worst-case membership AUROC ↓︎</th>
                  </tr>
                </thead>
                <tbody>
                  {[...good].sort(bySeed).map((r) => {
                    const s = armStyle(r.generator)
                    return (
                      <tr key={r.file}>
                        <td>
                          <span className="namecell" title={r.generator}><Swatch shape={s.marker} color={s.color} />{s.label}</span>
                        </td>
                        <td className="n">{r.seed}</td>
                        <td className="n">{f4(r.fidelityJs)}</td>
                        <td className="n">{f4(r.utilityTstr)}</td>
                        <td className="n">{f4(r.privacyWorstMia)}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </ChartCard>

          <div className="grid three">
            {AXES.map((ax) => {
              const { domain, ticks } = domainFor(summary, ax.key, ax.refs, ax.key === 'fidelityJs')
              return (
                <ChartCard
                  key={ax.key}
                  title={ax.title}
                  subtitle={`${ax.label}, mean over the uploaded seeds; whiskers are ±1 sd.`}
                  better={ax.better}
                  table={{
                    caption: `${ax.label} by generator`,
                    columns: [
                      { key: 'g', label: 'Generator' },
                      { key: 'm', label: 'Mean', numeric: true },
                      { key: 's', label: '± 1 sd', numeric: true },
                      { key: 'n', label: 'Seeds', numeric: true },
                    ],
                    rows: styled.map((s) => ({ g: s.style.label, m: f4(s[ax.key]), s: f4(s.sd[ax.key]), n: s.seeds })),
                  }}
                >
                  <DotPlot
                    rows={styled.map((s) => ({
                      key: s.generator,
                      label: s.style.label,
                      short: s.style.short,
                      shape: s.style.marker,
                      color: s.style.color,
                      mean: s[ax.key],
                      sd: s.sd[ax.key],
                      tip: [['Mean', f4(s[ax.key])], ['Spread (± 1 sd)', f4(s.sd[ax.key])], ['Seeds', String(s.seeds)]],
                    }))}
                    domain={domain}
                    ticks={ticks}
                    refs={ax.refs.filter((r) => r.value >= domain[0] && r.value <= domain[1])}
                    format={(v) => v.toFixed(3)}
                    label={`${ax.label} by generator`}
                  />
                </ChartCard>
              )
            })}
          </div>

          <Frontier rows={styled} />

          <details className="card report-list">
            <summary>
              <h3 style={{ display: 'inline' }}>Detailed metrics</h3>
              <span className="small muted" style={{ fontWeight: 400 }}>every metric each of the {good.length} reports carries, as the evaluation wrote it</span>
            </summary>
            {[...good].sort(bySeed).map((r) => (
              <details key={r.file}>
                <summary>
                  <Swatch shape={armStyle(r.generator).marker} color={armStyle(r.generator).color} />
                  {armStyle(r.generator).label} — seed {r.seed}
                </summary>
                <p className="small muted" style={{ marginTop: 8 }}>TSTR AUROC by classifier</p>
                <pre>{JSON.stringify(r.tstrDetails, null, 2)}</pre>
                <p className="small muted" style={{ marginTop: 8 }}>Full evaluation metrics</p>
                <pre>{JSON.stringify(r.metrics, null, 2)}</pre>
              </details>
            ))}
          </details>
        </>
      )}
    </div>
  )
}

function Frontier({ rows }) {
  const [hidden, setHidden] = useState(new Set())
  const toggle = (id) => setHidden((h) => { const n = new Set(h); n.has(id) ? n.delete(id) : n.add(id); return n })
  const single = rows.every((r) => r.seeds === 1)
  const ux = domainFor(rows, 'privacyWorstMia', [], false)
  const uy = domainFor(rows, 'utilityTstr', [], false)
  const points = rows.filter((r) => !hidden.has(r.generator)).map((r) => ({
    key: r.generator,
    label: r.style.label,
    short: r.style.short,
    shape: r.style.marker,
    color: r.style.color,
    x: r.privacyWorstMia,
    y: r.utilityTstr,
    xsd: r.sd.privacyWorstMia,
    ysd: r.sd.utilityTstr,
    tip: [
      ['Worst-case membership AUROC', f4(r.privacyWorstMia)],
      ['Mean TSTR AUROC', f4(r.utilityTstr)],
      ['Mean JS divergence', f4(r.fidelityJs)],
      ['On the frontier', r.onFrontier ? 'Yes' : 'No'],
    ],
  }))
  return (
    <ChartCard
      title="Pareto frontier"
      subtitle="A generator is on the frontier when no other generator is at least as good on fidelity, utility and privacy together, and strictly better on at least one."
      table={null}
      takeaway={
        <>
          <b>{rows.filter((r) => r.onFrontier).length} of {rows.length} generators are on the frontier.</b> The chart shows two of the three axes; the table and the
          frontier itself use all three. Frontier membership is computed from the uploaded reports; bootstrap frequencies are not.
        </>
      }
    >
      {single && (
        <div style={{ marginBottom: 12 }}>
          <Check kind="warn" lead="Each generator has one seed here,">
            so frontier membership reflects those single runs. Upload more seeds to see how much of it is seed noise.
          </Check>
        </div>
      )}
      <ScatterPlot
        points={points}
        label="Utility against privacy leakage, one point per uploaded generator"
        xAxis={{ domain: ux.domain, ticks: ux.ticks, format: (v) => v.toFixed(2), label: 'Worst-case membership AUROC (lower is better)', better: 'lower', short: 'leakage' }}
        yAxis={{ domain: uy.domain, ticks: uy.ticks, format: (v) => v.toFixed(2), label: 'Mean TSTR AUROC (higher is better)', better: 'higher', short: 'utility' }}
      />
      <Legend arms={rows.map((r) => ({ id: r.generator, label: r.style.label, marker: r.style.marker, family: r.style.family }))} hidden={hidden} onToggle={toggle} />
      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              <th scope="col">Generator</th>
              <th scope="col" className="n">Seeds</th>
              <th scope="col" className="n">Mean JS divergence</th>
              <th scope="col" className="n">Mean TSTR AUROC</th>
              <th scope="col" className="n">Worst-case membership AUROC</th>
              <th scope="col">On Pareto frontier</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.generator}>
                <td>
                  <span className="namecell" title={r.generator}><Swatch shape={r.style.marker} color={r.style.color} />{r.style.label}</span>
                </td>
                <td className="n">{r.seeds}</td>
                <td className="n">{f4(r.fidelityJs)}</td>
                <td className="n">{f4(r.utilityTstr)}</td>
                <td className="n">{f4(r.privacyWorstMia)}</td>
                <td><YesNo yes={r.onFrontier} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </ChartCard>
  )
}
