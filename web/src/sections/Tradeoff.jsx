import { useState } from 'react'
import ChartCard from '../components/ChartCard'
import SegmentedControl from '../components/SegmentedControl'
import Legend from '../components/Legend'
import ScatterPlot from '../charts/ScatterPlot'
import { FAMILY_COLOR, Swatch } from '../lib/marks'
import { f2, f3 } from '../lib/format'

const METRICS = {
  privacy: { label: 'Privacy leakage: attack score', short: 'leakage', better: 'lower', get: (r) => r.privacy.worst, domain: [0.6, 0.82], ticks: [0.6, 0.65, 0.7, 0.75, 0.8] },
  utility: { label: 'Utility: prediction score', short: 'utility', better: 'higher', get: (r) => r.utility.tstr_mean, domain: [0.3, 0.9], ticks: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] },
  fidelity: { label: 'Fidelity: distance from real data', short: 'distance', better: 'lower', get: (r) => r.fidelity.jsd, domain: [0, 0.42], ticks: [0, 0.1, 0.2, 0.3, 0.4] },
}
const OPTIONS = [
  { value: 'privacy', label: 'Privacy' },
  { value: 'utility', label: 'Utility' },
  { value: 'fidelity', label: 'Fidelity' },
]
const COLS = [
  ['main', 'Main axes'],
  ['+ correlation axis', '+ Correlation'],
  ['utility = LR only', 'Utility: logistic'],
  ['utility = RF only', 'Utility: forest'],
  ['privacy = 4 numeric', 'Privacy: 4 columns'],
  ['privacy = Gower', 'Privacy: all columns'],
  ['privacy = ICD-9 codes', 'Privacy: ICD-9'],
  ['no utility axis', 'No utility axis'],
]

export default function Tradeoff({ data }) {
  const [xKey, setX] = useState('privacy')
  const [yKey, setY] = useState('utility')
  const [hidden, setHidden] = useState(new Set())
  const { arms, results: r, pareto } = data

  const pick = (setter, other, otherSetter) => (v) => {
    setter(v)
    if (v === other) otherSetter(OPTIONS.find((o) => o.value !== v && o.value !== other)?.value ?? OPTIONS.find((o) => o.value !== v).value)
  }
  const toggle = (id) =>
    setHidden((h) => {
      const n = new Set(h)
      n.has(id) ? n.delete(id) : n.add(id)
      return n
    })

  const X = METRICS[xKey]
  const Y = METRICS[yKey]
  const third = OPTIONS.find((o) => o.value !== xKey && o.value !== yKey).value
  const T = METRICS[third]
  const points = arms
    .filter((a) => !hidden.has(a.id))
    .map((a) => {
      const x = X.get(r[a.id])
      const y = Y.get(r[a.id])
      const t = T.get(r[a.id])
      return {
        key: a.id, label: a.label, short: a.short, shape: a.marker, color: FAMILY_COLOR[a.family],
        x: x.mean, y: y.mean, xsd: x.sd, ysd: y.sd,
        tip: [[`${X.short} (mean ± sd)`, `${f3(x.mean)} ± ${f3(x.sd)}`], [`${Y.short} (mean ± sd)`, `${f3(y.mean)} ± ${f3(y.sd)}`], [`${T.short}`, f3(t.mean)]],
      }
    })

  const freq = pareto.frontier_freq
  const allOn = Object.values(pareto.frontier).every(Boolean)
  const sens = pareto.sensitivity
  const tvNoUtil = sens['no utility axis'].tvae
  const tvWeak = sens['privacy = 4 numeric'].tvae
  const leakyNoUtil = Math.max(sens['no utility axis'].tvae, sens['no utility axis'].tabular_diffusion, sens['no utility axis'].tabular_diffusion_reg)
  const off = arms.filter((a) => !pareto.frontier[a.id]).map((a) => a.label)

  const table = {
    caption: 'Position of each generator on the chosen measures, mean ± one standard deviation over 20 seeds',
    columns: [
      { key: 'g', label: 'Generator' },
      { key: 'x', label: X.label, numeric: true },
      { key: 'y', label: Y.label, numeric: true },
    ],
    rows: arms.map((a) => ({
      g: (
        <span className="namecell">
          <Swatch shape={a.marker} color={FAMILY_COLOR[a.family]} />
          {a.label}
        </span>
      ),
      x: `${f3(X.get(r[a.id]).mean)} ± ${f3(X.get(r[a.id]).sd)}`,
      y: `${f3(Y.get(r[a.id]).mean)} ± ${f3(Y.get(r[a.id]).sd)}`,
    })),
  }

  return (
    <section id="tradeoffs" aria-labelledby="h-trade">
      <div className="wrap">
        <div className="section-head">
          <span className="eyebrow">Trade-offs</span>
          <h2 id="h-trade">Is any generator simply the best?</h2>
          <p>No generator wins on realism, usefulness and privacy at once. Pick any two measures to see the trade-off.</p>
        </div>

        <div className="filters">
          <SegmentedControl label="Horizontal" value={xKey} onChange={pick(setX, yKey, setY)} options={OPTIONS} />
          <SegmentedControl label="Vertical" value={yKey} onChange={pick(setY, xKey, setX)} options={OPTIONS} />
        </div>

        <div className="stack">
          <ChartCard
            title={`${Y.label.split(':')[0]} against ${X.label.split(':')[0].toLowerCase()}`}
            subtitle="Each point is a generator; whiskers show how much it varied over 20 seeds."
            table={table}
            legend={<Legend arms={arms} hidden={hidden} onToggle={toggle} />}
            takeaway={
              <>
                <b>Nothing sits in the best corner.</b> TVAE and the two diffusion settings buy utility with leakage; CTGAN and the floor leak least but predict least; the copulas
                sit in between, the tuned one (shrink 0.10) closest to the corner.
              </>
            }
          >
            <ScatterPlot
              points={points}
              label={`${Y.label} against ${X.label}`}
              xAxis={{ domain: X.domain, ticks: X.ticks, format: f2, label: `${X.label} (${X.better} is better)`, better: X.better, short: X.short }}
              yAxis={{ domain: Y.domain, ticks: Y.ticks, format: f2, label: `${Y.label} (${Y.better} is better)`, better: Y.better, short: Y.short }}
            />
          </ChartCard>

          <div className="card">
            <h3>Can the generators be ranked?</h3>
            <ul className="list" style={{ marginTop: 10 }}>
              <li>
                <b>{allOn ? `All ${arms.length}` : `${arms.length - off.length} of ${arms.length}`} generators are on the trade-off frontier</b>: no other generator beats them on realism, usefulness
                and privacy at once.{allOn ? '' : ` Only ${off.join(' and ')} is beaten, by its tuned version.`} So the frontier shows which way each one trades, not which is best.
              </li>
              <li>
                <b>TVAE and the diffusion models are there only because of their prediction score.</b> Without that measure they are on the frontier at most{' '}
                {Math.round(leakyNoUtil * 100)}% of the time, and that score comes from copying and from the wrong rate of deaths.
              </li>
              <li>
                <b>The tuned copula is the steadiest choice.</b> It stays on the frontier in {Math.round(Math.min(...Object.entries(sens).filter(([k]) => k !== 'no utility axis').map(([, v]) => v.gaussian_copula_shrink010)) * 100)}%{' '}
                or more of resamples under every choice of measures that keeps the prediction score, while leaking less than TVAE and both diffusion models.
              </li>
            </ul>
          </div>
        </div>
      </div>
    </section>
  )
}
