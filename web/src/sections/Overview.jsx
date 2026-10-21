import StatTile from '../components/StatTile'
import { f3, pval } from '../lib/format'
import { COPULAS, NEURAL, NOT_LEAKY, pct0, range } from '../lib/groups'

export default function Overview({ data }) {
  const { results: r, refs, meta, pareto, tests, coverage, mortality, arms } = data
  const label = (id) => arms.find((a) => a.id === id)?.label ?? id
  const worst = (a) => r[a].privacy.worst.mean
  const [lo, hi] = range(NOT_LEAKY, worst)
  const onFrontier = Object.values(pareto.frontier).filter(Boolean).length
  const total = Object.keys(pareto.frontier).length
  const offFrontier = Object.keys(pareto.frontier).filter((a) => !pareto.frontier[a]).map(label)
  const copulaJsd = r.gaussian_copula.fidelity.jsd.mean
  const [nLo, nHi] = range(NEURAL, (a) => r[a].fidelity.jsd.mean)
  const diff = r.tabular_diffusion
  const maxP = Math.max(...Object.keys(r).filter((a) => a !== 'tabular_diffusion').map((a) => tests[`tabular_diffusion_vs_${a}_worst_p`]))
  const best = r.gaussian_copula_shrink010
  const [cLo] = range(COPULAS, (a) => r[a].fidelity.corr_diff.mean)
  const realDeaths = data.dataset.train_mortality_rate
  const topUtil = ['tabular_diffusion', 'tabular_diffusion_reg'].reduce((a, b) => (r[a].utility.tstr_rf.mean > r[b].utility.tstr_rf.mean ? a : b))

  return (
    <section id="overview" aria-labelledby="h-overview">
      <div className="wrap">
        <div className="hero">
          <div>
            <span className="eyebrow">Project Doppel · Synthetic health-record benchmark</span>
            <h1 id="h-overview">Which synthetic-data generator is realistic, useful and private?</h1>
            <p className="lead">
              Hospitals want to share patient data for research without exposing patients. Doppel tests generators of synthetic records on three
              things at once: how realistic they are, how useful they are for a prediction task, and how much they reveal about the real patients
              they learned from.
            </p>
            <div className="cta">
              <a className="btn primary" href="#compare">Compare the generators</a>
              <a className="btn" href="#tradeoffs">See the trade-offs</a>
            </div>
          </div>
          <div className="card hero-figure">
            <div className="value num">{f3(worst('tabular_diffusion'))}</div>
            <div className="label">Membership-attack score for the diffusion model</div>
            <p className="muted">
              This diffusion model copies most of its training rows. The attack scores 0.5 when it is guessing and 1.0 when it always wins.
              The six generators that do not leak score {f3(lo)} to {f3(hi)}.
            </p>
          </div>
        </div>

        <div className="grid four" style={{ marginTop: 26 }}>
          <StatTile label="Evaluation runs" value={meta.runs} note={`${arms.length} generator settings × ${r.tvae.n_seeds} random seeds each`} />
          <StatTile label="Closest to the real data" value={f3(copulaJsd)} note={`The copulas and the floor tie. The neural generators are ${(nLo / copulaJsd).toFixed(1)} to ${Math.round(nHi / copulaJsd)}× further away.`} />
          <StatTile
            label="Highest prediction score"
            value={f3(r[topUtil].utility.tstr_rf.mean)}
            note={`${label(topUtil)}, above models trained on real data (${f3(refs.trtr_rf)}), because it generates ${pct0(mortality[topUtil].mean)} deaths against a real ${pct0(realDeaths)}.`}
          />
          <StatTile label="On the trade-off frontier" value={`${onFrontier} of ${total}`} note={`Only ${offFrontier.join(' and ')} is off it, so the data cannot rank the rest.`} />
        </div>

        <div className="section-head" style={{ marginTop: 56 }}>
          <h2>What we found</h2>
          <p>Three results from the full comparison, each with the numbers behind it.</p>
        </div>
        <div className="grid three">
          <article className="card finding">
            <span className="tag">Privacy</span>
            <h3>The generators that copy leak most, and the attacks catch them</h3>
            <p>
              TVAE and the 3000-step diffusion model copy their training rows by Track 1&rsquo;s own check, and the attacks rank both above every generator that
              does not. The regularized diffusion model passed that check and still leaks second most.
            </p>
            <div className="evidence">
              Diffusion <b className="num">{f3(worst('tabular_diffusion'))}</b>, regularized <b className="num">{f3(worst('tabular_diffusion_reg'))}</b>, TVAE{' '}
              <b className="num">{f3(worst('tvae'))}</b>, against <b className="num">{f3(lo)}–{f3(hi)}</b> for the rest ({pval(maxP)} for the diffusion model against each).
              At 1% false alarms the attack finds <b>{pct0(diff.privacy.tpr1_gower.mean)}</b> of its training patients.
            </div>
          </article>
          <article className="card finding">
            <span className="tag">Fidelity</span>
            <h3>No neural generator works at this size</h3>
            <p>
              With 94 training rows, each neural family fails one of two ways. CTGAN learns no relationships between columns, even after tuning. TVAE and the
              diffusion model reproduce their training patients instead of learning from them.
            </p>
            <div className="evidence">
              Distance from real data <b className="num">{f3(copulaJsd)}</b> for the copula against <b className="num">{f3(nLo)}–{f3(nHi)}</b> for the neural generators.
              Tuned CTGAN&rsquo;s correlation error <b className="num">{f3(r.ctgan_small.fidelity.corr_diff.mean)}</b> is worse than the floor&rsquo;s{' '}
              <b className="num">{f3(r.independent_marginals.fidelity.corr_diff.mean)}</b>. TVAE&rsquo;s output sits nearest to only{' '}
              <b>{coverage.tvae.distinct.mean.toFixed(0)} of 94</b> training rows.
            </div>
          </article>
          <article className="card finding">
            <span className="tag">Trade-off</span>
            <h3>The tuned copula is the best trade-off here</h3>
            <p>
              The copula with the least shrinkage keeps the most of the real correlations, predicts about as well as TVAE, and leaks less than every generator that
              copies. It needs no copying to get there.
            </p>
            <div className="evidence">
              Correlation error <b className="num">{f3(cLo)}</b>, the best of any generator. Prediction score <b className="num">{f3(best.utility.tstr_mean.mean)}</b> vs TVAE{' '}
              <b className="num">{f3(r.tvae.utility.tstr_mean.mean)}</b> ({pval(tests.shrink010_vs_tvae_utility_p)}). Leakage{' '}
              <b className="num">{f3(best.privacy.worst.mean)}</b> vs <b className="num">{f3(worst('tvae'))}</b> and up.
            </div>
          </article>
        </div>
      </div>
    </section>
  )
}

