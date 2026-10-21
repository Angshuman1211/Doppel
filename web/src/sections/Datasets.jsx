import { useEffect, useState } from 'react'
import { useData } from '../lib/hooks'
import { FAMILY_COLOR, Swatch } from '../lib/marks'

const PREVIEW_ROWS = 10
const BASE = import.meta.env.BASE_URL

function cell(v) {
  if (v === null) return <span className="nil">—</span>
  if (typeof v === 'boolean') return v ? 'True' : 'False'
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : String(Math.round(v * 1000) / 1000)
  return v.length > 26 ? <span title={v}>{v.slice(0, 25)}…</span> : v
}

function toCsv(columns, rows) {
  const esc = (v) => {
    if (v === null) return ''
    if (typeof v === 'boolean') return v ? 'True' : 'False'
    const s = String(v)
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
  }
  return [columns.map((c) => esc(c.name)).join(','), ...rows.map((r) => r.map(esc).join(','))].join('\n')
}

function download(name, columns, rows) {
  const url = URL.createObjectURL(new Blob([toCsv(columns, rows)], { type: 'text/csv' }))
  const a = document.createElement('a')
  a.href = url
  a.download = name
  a.click()
  URL.revokeObjectURL(url)
}

function Bar({ v, max }) {
  return (
    <div className="track">
      <i style={{ width: `${max > 0 ? Math.min(100, (v / max) * 100) : 0}%` }} />
    </div>
  )
}

const FIDELITY_PREVIEW = 10

export default function Datasets({ arms }) {
  const index = useData(`${BASE}data/datasets/index.json`)
  const real = useData(`${BASE}data/datasets/real.json`)
  const [arm, setArm] = useState(null)
  const [seed, setSeed] = useState(null)
  const [all, setAll] = useState(false)
  const [allCols, setAllCols] = useState(false)
  const [detailCol, setDetailCol] = useState(null)

  const list = index.data?.arms
  useEffect(() => {
    if (list?.length && !arm) {
      setArm(list[list.length - 1].id)
      setSeed(list[list.length - 1].seeds[0])
    }
  }, [list, arm])

  const current = list?.find((a) => a.id === arm)
  const table = useData(arm && seed != null ? `${BASE}data/datasets/${arm}_seed${seed}.json` : null)
  const t = arm && seed != null ? table.data : null
  const meta = arms.find((a) => a.id === arm)

  const pickArm = (id) => {
    const next = list.find((a) => a.id === id)
    setArm(id)
    setSeed(next.seeds.includes(seed) ? seed : next.seeds[0])
    setAll(false)
    setAllCols(false)
  }
  const shown = t ? (all ? t.rows : t.rows.slice(0, PREVIEW_ROWS)) : []

  const fidCols = t?.fidelity?.columns ?? []
  const fidShown = allCols ? fidCols : fidCols.slice(0, FIDELITY_PREVIEW)
  const maxJs = Math.max(0.001, ...fidCols.map((c) => c.js))
  useEffect(() => {
    if (fidCols.length && (!detailCol || !fidCols.some((c) => c.name === detailCol))) setDetailCol(fidCols[0].name)
  }, [t, fidCols, detailCol])
  const detail = detailCol && t && real.data ? { kind: fidCols.find((c) => c.name === detailCol)?.kind, synth: t.column_detail[detailCol], real: real.data.columns[detailCol] } : null

  return (
    <section id="data" aria-labelledby="h-data">
      <div className="wrap">
        <div className="section-head">
          <span className="eyebrow">Data</span>
          <h2 id="h-data">Look at the synthetic records</h2>
          <p>
            Pick a generator and a seed to see the records it produced and where they differ from the real ones. To check a file of your own, use Upload below.
          </p>
        </div>

        {index.error && (
          <div className="notice" role="note">
            <span>
              <b>The dataset tables are not in the repository.</b> They are generated files, so they are left out of git. Generate the evaluation datasets,
              then run <code>npm run export-data</code> in <code>web/</code> and reload.
            </span>
          </div>
        )}
        {!index.error && !list && <div className="loading small">Loading datasets…</div>}

        {list && current && (
          <>
            <div className="filters">
              <label className="field">
                <span>Generator</span>
                <select className="select" value={arm} onChange={(e) => pickArm(e.target.value)}>
                  {list.map((a) => (
                    <option key={a.id} value={a.id}>{a.label}</option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>Random seed</span>
                <select className="select" value={seed} onChange={(e) => { setSeed(Number(e.target.value)); setAll(false); setAllCols(false) }}>
                  {current.seeds.map((s) => (
                    <option key={s} value={s}>{s}</option>
                  ))}
                </select>
              </label>
              {meta && (
                <span className="field pick-note">
                  <Swatch shape={meta.marker} color={FAMILY_COLOR[meta.family]} />
                  <span className="small muted">{meta.family_label}</span>
                </span>
              )}
            </div>

            {table.error && <div className="loading small" role="alert">Could not load this dataset ({String(table.error.message)}).</div>}
            {!t && !table.error && <div className="loading small">Loading table…</div>}

            {t && (
              <div className="stack">
                <figure className="card" style={{ margin: 0 }}>
                  <div className="chart-head">
                    <div>
                      <h3>Dataset preview</h3>
                      <p className="sub">
                        {all ? `All ${t.n_rows} rows` : `First ${Math.min(PREVIEW_ROWS, t.n_rows)} of ${t.n_rows} rows`} of <span className="num">{t.file}</span>. Scroll sideways for more columns.
                      </p>
                    </div>
                    <div className="chart-tools">
                      {t.n_rows > PREVIEW_ROWS && (
                        <button className="tbtn" onClick={() => setAll(!all)}>{all ? `Show first ${PREVIEW_ROWS}` : `Show all ${t.n_rows}`}</button>
                      )}
                      <button className="tbtn" onClick={() => download(t.file, t.columns, t.rows)}>Download CSV</button>
                    </div>
                  </div>
                  <div className="sheet" tabIndex={0} role="region" aria-label={`Rows of ${t.file}, scrollable`}>
                    <table>
                      <thead>
                        <tr>
                          <th scope="col" className="n rownum">#</th>
                          {t.columns.map((c) => (
                            <th key={c.name} scope="col">{c.name}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {shown.map((r, i) => (
                          <tr key={i}>
                            <td className="n rownum">{i}</td>
                            {r.map((v, j) => (
                              <td key={j} className={typeof v === 'number' ? 'n' : undefined}>{cell(v)}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </figure>

                {t.fidelity && (
                  <div className="card">
                    <div className="chart-head">
                      <div>
                        <h3>Fidelity against the real training rows</h3>
                        <p className="sub">
                          Per-column distance between this file and the {real.data?.n_train ?? 94} real training rows, using Track 2&rsquo;s own fidelity code (the Compare section&rsquo;s method), so the mean below matches it. Worst column first.
                        </p>
                      </div>
                      <div className="chart-tools">
                        <span className="chip">Mean JS {t.fidelity.mean_js?.toFixed(4)}</span>
                        <span className="chip">Corr. gap {t.fidelity.corr_diff?.toFixed(4)}</span>
                      </div>
                    </div>
                    <div className="barlist">
                      {fidShown.map((c) => (
                        <div className="row" key={c.name}>
                          <span className="name">
                            {c.name}
                            <span className="kindpill">{c.kind === 'numeric' ? 'num' : 'cat'}</span>
                          </span>
                          <Bar v={c.js} max={maxJs} />
                          <span className="val">{c.js.toFixed(4)}</span>
                        </div>
                      ))}
                    </div>
                    {fidCols.length > FIDELITY_PREVIEW && (
                      <p style={{ marginTop: 10 }}>
                        <button className="tbtn" onClick={() => setAllCols(!allCols)}>
                          {allCols ? `Show worst ${FIDELITY_PREVIEW}` : `Show all ${fidCols.length} columns`}
                        </button>
                      </p>
                    )}
                    <p className="takeaway">
                      Computed from the file above with the same code and the same training rows as the evaluation, so these numbers match that seed&rsquo;s result exactly. The Compare section shows the mean over all 20 seeds.
                    </p>
                  </div>
                )}

                {detail && (
                  <div className="card">
                    <h3>Column detail: real vs this dataset</h3>
                    <div className="detail-pick">
                      <label className="field">
                        <span>Column</span>
                        <select className="select" value={detailCol} onChange={(e) => setDetailCol(e.target.value)}>
                          {fidCols.map((c) => (
                            <option key={c.name} value={c.name}>{c.name}</option>
                          ))}
                        </select>
                      </label>
                      <span className="small muted">JS divergence {fidCols.find((c) => c.name === detailCol)?.js.toFixed(4)}</span>
                    </div>

                    {detail.kind === 'numeric' ? (
                      <table className="statcompare">
                        <thead>
                          <tr>
                            <th>Statistic</th>
                            <th className="n">Real training rows</th>
                            <th className="n">This dataset</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.keys(detail.real.stats).map((k) => (
                            <tr key={k}>
                              <td className="muted">{k}</td>
                              <td className="n">{detail.real.stats[k]}</td>
                              <td className="n">{detail.synth.stats[k]}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    ) : (
                      <>
                        <div className="share-head"><span /><span>Real training rows</span><span>This dataset</span></div>
                        {Object.keys({ ...detail.real.shares, ...detail.synth.shares }).map((k) => (
                          <div className="sharebar" key={k}>
                            <span className="lab">{k}</span>
                            <span className="pair">
                              <span className="track real"><i style={{ width: `${(detail.real.shares[k] ?? 0) * 100}%` }} /></span>
                              <span className="v">{((detail.real.shares[k] ?? 0) * 100).toFixed(0)}%</span>
                            </span>
                            <span className="pair">
                              <span className="track synth"><i style={{ width: `${(detail.synth.shares[k] ?? 0) * 100}%` }} /></span>
                              <span className="v">{((detail.synth.shares[k] ?? 0) * 100).toFixed(0)}%</span>
                            </span>
                          </div>
                        ))}
                      </>
                    )}
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </section>
  )
}
