import { useState } from 'react'
import ChartCard from '../components/ChartCard'
import SegmentedControl from '../components/SegmentedControl'
import DotPlot from '../charts/DotPlot'
import { FAMILY_COLOR, Swatch } from '../lib/marks'
import { f3, niceTicks } from '../lib/format'
import { LEAKY, NEURAL, NOT_LEAKY, argmax, domainFor, pct0, range } from '../lib/groups'

const UTILITY = [
  { value: 'tstr_rf', label: 'Random forest' },
  { value: 'tstr_lr', label: 'Logistic regression' },
]
const ATTACKS = [
  { value: 'worst', label: 'Worst case' },
  { value: 'gower', label: 'All columns' },
  { value: 'codes', label: 'ICD-9 codes' },
  { value: 'numeric4', label: '4 columns' },
]
const ATTACK_NAME = { worst: 'worst case of three attacks', gower: 'the all-columns attack', codes: 'the ICD-9 code attack', numeric4: 'the 4-column attack' }

const tableFor = (arms, rows, caption, valueLabel) => ({
  caption,
  columns: [
    { key: 'g', label: 'Generator' },
    { key: 'm', label: valueLabel, numeric: true },
    { key: 's', label: 'Spread (± 1 sd)', numeric: true },
  ],
  rows: arms.map((a, i) => ({
    g: (
      <span className="namecell">
        <Swatch shape={a.marker} color={FAMILY_COLOR[a.family]} />
        {a.label}
      </span>
    ),
    m: f3(rows[i].mean),
    s: f3(rows[i].sd),
  })),
})

export default function Compare({ data }) {
  const [util, setUtil] = useState('tstr_rf')
  const [attack, setAttack] = useState('worst')
  const { arms, results: r, refs, mortality } = data
  const name = (id) => arms.find((a) => a.id === id)?.label ?? id

  const armRow = (a, m, tip) => ({ key: a.id, label: a.label, short: a.short, shape: a.marker, color: FAMILY_COLOR[a.family], mean: m.mean, sd: m.sd, tip })

  const fid = arms.map((a) => r[a.id].fidelity.jsd)
  const fidRows = arms.map((a, i) => armRow(a, fid[i], [['Distance (mean)', f3(fid[i].mean)], ['Spread (± 1 sd)', f3(fid[i].sd)], ['Correlation error', f3(r[a.id].fidelity.corr_diff.mean)]]))

  const ceiling = util === 'tstr_rf' ? refs.trtr_rf : refs.trtr_lr
  const ut = arms.map((a) => r[a.id].utility[util])
  const utRows = arms.map((a, i) => armRow(a, ut[i], [['Score (mean)', f3(ut[i].mean)], ['Spread (± 1 sd)', f3(ut[i].sd)], ['Gap to real-data ceiling', f3(ceiling - ut[i].mean)]]))

  const pr = arms.map((a) => r[a.id].privacy[attack])
  const prRows = arms.map((a, i) => armRow(a, pr[i], [['Attack score (mean)', f3(pr[i].mean)], ['Spread (± 1 sd)', f3(pr[i].sd)], ['Above guessing (0.5)', f3(pr[i].mean - 0.5)]]))

  const copJsd = r.gaussian_copula.fidelity.jsd.mean
  const [nLo, nHi] = range(NEURAL, (a) => r[a].fidelity.jsd.mean)
  const farthest = argmax(NEURAL, (a) => r[a].fidelity.jsd.mean)
  const fidAxis = domainFor(fid, [], { floorZero: true, niceTicks })
  const utAxis = domainFor(ut, [0.5, ceiling], { niceTicks, count: 7 })
  const prAxis = domainFor(pr, [0.5, refs.fail_line], { niceTicks, count: 7 })

  const priv = (a) => r[a].privacy[attack].mean
  const top = argmax(arms.map((a) => a.id), priv)
  const [restLo, restHi] = range(NOT_LEAKY, priv)
  const blind = !LEAKY.includes(top) || priv('tvae') <= restLo
  const topUtil = argmax(arms.map((a) => a.id), (a) => r[a].utility[util].mean)
  const bestHonest = argmax(NOT_LEAKY, (a) => r[a].utility[util].mean)

  return (
    <section id="compare" aria-labelledby="h-compare">
      <div className="wrap">
        <div className="section-head">
          <span className="eyebrow">Compare</span>
          <h2 id="h-compare">How do the generators compare?</h2>
          <p>
            Each dot is the average over {r.tvae.n_seeds} runs, and the whisker shows how much the runs varied. Where whiskers overlap, the
            difference could be chance.
          </p>
        </div>

        <div className="filters">
          <SegmentedControl label="Prediction model" value={util} onChange={setUtil} options={UTILITY} />
          <SegmentedControl label="Privacy attack" value={attack} onChange={setAttack} options={ATTACKS} />
        </div>

        <div className="stack">
          <ChartCard
            title="Fidelity: how close to the real data"
            subtitle="Average distance between each column's real and synthetic distribution."
            better="lower"
            table={tableFor(arms, fid, 'Fidelity distance, mean and spread over 20 seeds', 'Distance')}
            takeaway={
              <>
                <b>The neural generators are {(nLo / copJsd).toFixed(1)} to {Math.round(nHi / copJsd)}× further from the real data</b> than the copulas ({f3(nLo)} to{' '}
                {f3(nHi)} against {f3(copJsd)}), {name(farthest)} furthest. The floor ties with the copulas because it copies each column exactly.
              </>
            }
          >
            <DotPlot rows={fidRows} domain={fidAxis.domain} ticks={fidAxis.ticks} label="Fidelity distance by generator" />
          </ChartCard>

          <ChartCard
            title="Utility: how well a model trained on it predicts"
            subtitle="Train on synthetic data, test on real held-out patients, predicting in-hospital death."
            better="higher"
            table={tableFor(arms, ut, `Prediction score (AUROC, ${util === 'tstr_rf' ? 'random forest' : 'logistic regression'}), mean and spread over 20 seeds`, 'AUROC')}
            takeaway={
              <>
                <b>The top score, {name(topUtil)}&rsquo;s {f3(r[topUtil].utility[util].mean)}, is not real utility.</b> It generates {pct0(mortality[topUtil].mean)} deaths against
                a real {pct0(data.dataset.train_mortality_rate)}, and TVAE ({pct0(mortality.tvae.mean)} deaths) reaches the real-data ceiling of {f3(ceiling)} by copying. Of the generators
                that do not leak, {name(bestHonest)} scores highest ({f3(r[bestHonest].utility[util].mean)}). The spread between runs, up to{' '}
                {f3(Math.max(...ut.map((u) => u.sd)))}, is bigger than most gaps.
              </>
            }
          >
            <DotPlot
              rows={utRows}
              domain={utAxis.domain}
              ticks={utAxis.ticks}
              refs={[
                { value: 0.5, label: 'Guessing 0.5' },
                { value: ceiling, label: 'Real-data ceiling' },
              ]}
              label="Prediction score by generator"
            />
          </ChartCard>

          <ChartCard
            title="Privacy: how much the attack learns"
            subtitle={`Attacker guesses which real records trained the generator, using ${ATTACK_NAME[attack]}.`}
            better="lower"
            table={tableFor(arms, pr, `Membership attack score (${ATTACK_NAME[attack]}), mean and spread over 20 seeds`, 'Attack score')}
            takeaway={
              blind ? (
                <>
                  <b>This attack barely sees the leaks:</b> it scores TVAE at {f3(priv('tvae'))} and the regularized diffusion model at {f3(priv('tabular_diffusion_reg'))},
                  no higher than the generators that do not leak ({f3(restLo)} to {f3(restHi)}). Switch to &ldquo;Worst case&rdquo; to see them.
                </>
              ) : (
                <>
                  <b>{name(top)} leaks most</b> ({f3(priv(top))}), then {LEAKY.filter((a) => a !== top).map((a) => `${name(a)} (${f3(priv(a))})`).join(' and ')}, against{' '}
                  {f3(restLo)} to {f3(restHi)} for the rest. A score of 0.5 means the attacker is only guessing; 0.65 is the protocol&rsquo;s fail line.
                </>
              )
            }
          >
            <DotPlot
              rows={prRows}
              domain={prAxis.domain}
              ticks={prAxis.ticks}
              refs={[
                { value: 0.5, label: 'Guessing 0.5' },
                { value: refs.fail_line, label: 'Fail line 0.65' },
              ]}
              label="Membership attack score by generator"
            />
          </ChartCard>
        </div>
      </div>
    </section>
  )
}

