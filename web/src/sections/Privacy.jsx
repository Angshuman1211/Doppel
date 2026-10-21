import ChartCard from '../components/ChartCard'
import DotPlot from '../charts/DotPlot'
import { FAMILY_COLOR } from '../lib/marks'
import { f3 } from '../lib/format'
import { NOT_LEAKY, pct0, range } from '../lib/groups'

// The one privacy result a reader most needs (how many real patients an attacker can pick out), and in three
// sentences why the attacks can be trusted. The full attack-by-attack evidence is in final_evaluation_result.md.
export default function Privacy({ data }) {
  const { arms, calibration: c } = data
  const REF = 'var(--c-floor)'
  const tpr = (a) => c.tpr1[a].gower
  const rows = [
    ...arms.map((a) => ({
      key: a.id, label: a.label, short: a.short, shape: a.marker, color: FAMILY_COLOR[a.family], mean: tpr(a.id).mean, sd: tpr(a.id).sd,
      tip: [['Patients found', pct0(tpr(a.id).mean)], ['Times guessing', `${(tpr(a.id).mean / 0.01).toFixed(1)}×`]],
    })),
    { key: 'floor', label: 'Real rows it never saw (floor)', short: 'Unseen rows', shape: 'circle', color: REF, hollow: true, mean: tpr('disjoint_real').mean, sd: tpr('disjoint_real').sd, tip: [['Patients found', pct0(tpr('disjoint_real').mean)]] },
  ]
  const top = Math.max(...rows.map((r) => r.mean + r.sd))
  const axis = top > 0.2 ? { domain: [0, 0.44], ticks: [0, 0.1, 0.2, 0.3, 0.4] } : { domain: [0, 0.09], ticks: [0, 0.02, 0.04, 0.06, 0.08] }
  const [lo, hi] = range(NOT_LEAKY, (a) => tpr(a).mean)
  const once = arms.map((a) => c.auroc[a.id].codes_once.mean)

  return (
    <section id="privacy" aria-labelledby="h-privacy">
      <div className="wrap">
        <div className="section-head">
          <span className="eyebrow">Privacy</span>
          <h2 id="h-privacy">How many real patients can an attacker pick out?</h2>
          <p>The attacker is allowed to wrongly accuse only 1 in 100 patients who were never used. How many of the real training patients does it catch?</p>
        </div>

        <div className="stack">
          <ChartCard
            title="Training patients identified at 1% false alarms"
            subtitle="All-columns membership attack, mean over 20 seeds. Guessing finds 1%."
            better="lower"
            table={{
              caption: 'Share of training patients identified at 1% false alarms',
              columns: [{ key: 'g', label: 'Generator' }, { key: 'm', label: 'Found', numeric: true }, { key: 's', label: '± 1 sd', numeric: true }],
              rows: rows.map((r) => ({ g: r.label, m: pct0(r.mean), s: f3(r.sd) })),
            }}
            takeaway={
              <>
                <b>The diffusion model gives away {pct0(tpr('tabular_diffusion').mean)} of its training patients</b>, the regularized one {pct0(tpr('tabular_diffusion_reg').mean)} and
                TVAE {pct0(tpr('tvae').mean)}. The generators that do not copy stay at {pct0(lo)} to {pct0(hi)}, close to guessing.
              </>
            }
          >
            <DotPlot rows={rows} domain={axis.domain} ticks={axis.ticks} refs={[{ value: 0.01, label: 'Guessing 1%' }]} format={(v) => `${Math.round(v * 100)}%`} label="Training patients identified at 1% false alarms" />
          </ChartCard>

          <div className="card">
            <h3>Why these attacks can be trusted</h3>
            <ul className="list" style={{ marginTop: 10 }}>
              <li>
                <b>They are checked against known answers.</b> A generator that copies its training rows scores {f3(c.auroc.exact_copy.gower.mean)}, and real rows it never saw score{' '}
                {f3(c.auroc.disjoint_real.gower.mean)}. Every generator falls between the two.
              </li>
              <li>
                <b>The privacy score is the worst of three attacks.</b> A weak attack on four columns scores TVAE {f3(c.auroc.tvae.numeric4.mean)}, the same as guessing, even though
                TVAE copies its training data.
              </li>
              <li>
                <b>Rare diagnosis codes leak through every generator.</b> A code only one training patient has gives that patient away, so even generators that learned little score{' '}
                {f3(Math.min(...once))} to {f3(Math.max(...once))} on those codes.
              </li>
            </ul>
          </div>
        </div>
      </div>
    </section>
  )
}
