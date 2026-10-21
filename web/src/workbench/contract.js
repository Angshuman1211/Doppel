// generators/validate.check_contract, line for line, with the same messages. Streamlit runs the Python version
// when it is started from the repo; in the browser this port is what runs.
import { kindOf, pyStr, subset } from './csv.js'

const REAL_CSV = 'output/mimic_demo_clean.csv'
const SPLIT_COL = 'split'
const TRAIN_SPLIT = 'train'
const SYNTH_SPLIT = 'synthetic'
const ID_COLS = ['subject_id', 'hadm_id']
const CODES_COL = 'icd9_codes'
const PRIMARY_COL = 'icd9_primary'
const N_DIAG_COL = 'n_diagnoses'
const CATEGORICAL = ['admission_type', 'ethnicity', 'first_careunit', 'gender', PRIMARY_COL]
const BINARY = ['hospital_expire_flag', 'readmit_30d', 'age_89_plus']
const AGE_COL = 'age'
const AGE_FLAG_COL = 'age_89_plus'
const AGE_CAP = 89

// Python's repr of a list of strings, e.g. ['a', "b's"].
const pyList = (items) =>
  '[' + items.map((s) => (s.includes("'") && !s.includes('"') ? `"${s}"` : `'${s.replace(/\\/g, '\\\\').replace(/'/g, "\\'")}'`)).join(', ') + ']'

const isMissing = (v) => v === null || (typeof v === 'number' && Number.isNaN(v))
// pandas truthiness for astype(bool): missing is True, '' is False, 0 is False.
const truthy = (v) => (v === null ? true : typeof v === 'string' ? v.length > 0 : Boolean(v))
// Python equality as pandas applies it: True == 1, 3 == 3.0.
const num = (v) => (typeof v === 'boolean' ? Number(v) : v)

// Python's json.loads is a little more lenient than JSON.parse: it accepts NaN, Infinity and -Infinity.
function loadsJson(raw) {
  if (typeof raw !== 'string') throw new TypeError('not a string')
  try {
    return JSON.parse(raw)
  } catch {
    const relaxed = raw.replace(/(?<![\w"])-?(NaN|Infinity)(?![\w"])/g, 'null')
    if (relaxed === raw) throw new SyntaxError('invalid JSON')
    return JSON.parse(relaxed)
  }
}

function codeProblem(raw, primary, nDiag, trainCodes) {
  let codes
  try {
    codes = loadsJson(raw)
  } catch {
    return 'not valid JSON'
  }
  if (!Array.isArray(codes) || !codes.length || !codes.every((c) => typeof c === 'string')) {
    return 'not a non-empty JSON list of strings'
  }
  if (new Set(codes).size !== codes.length) return 'duplicate codes in one admission'
  if (codes[0] !== primary) return 'first code differs from icd9_primary'
  if (isMissing(nDiag) || codes.length !== num(nDiag)) return 'length differs from n_diagnoses'
  if (!codes.every((c) => trainCodes.has(c))) return 'code never seen in train'
  return null
}

// A hashable key for set membership, matching how Python compares values in a set (1 == 1.0 == True).
const setKey = (v) => (isMissing(v) ? 'nan' : typeof v === 'boolean' || typeof v === 'number' ? `n:${num(v)}` : `s:${v}`)

export function checkContract(synth, real) {
  const same = synth.columns.length === real.columns.length && synth.columns.every((c, i) => c === real.columns[i])
  if (!same) {
    const missing = real.columns.filter((c) => !synth.columns.includes(c)).sort()
    const extra = synth.columns.filter((c) => !real.columns.includes(c)).sort()
    return [`columns must match ${REAL_CSV} in name and order (missing=${pyList(missing)}, extra=${pyList(extra)})`]
  }
  if (!synth.n) return ['no rows']

  const train = subset(real, (i) => real.data[SPLIT_COL][i] === TRAIN_SPLIT)
  const errors = []

  for (const c of real.columns) {
    const r = kindOf(real.dtypes[c])
    const s = kindOf(synth.dtypes[c])
    if (r !== s) errors.push(`${c}: dtype kind '${s}' but real is '${r}'`)
  }

  if (!synth.data[SPLIT_COL].every((v) => v === SYNTH_SPLIT)) errors.push(`split must be '${SYNTH_SPLIT}' on every row`)
  for (const c of ID_COLS) {
    const realIds = new Set(real.data[c].map(setKey))
    const collide = new Set(synth.data[c].map(setKey).filter((k) => realIds.has(k)))
    // NaN never equals NaN in a Python set intersection
    collide.delete('nan')
    if (collide.size) errors.push(`${c}: ${collide.size} values collide with real IDs`)
  }
  const hadm = synth.data[ID_COLS[1]].map(setKey)
  if (new Set(hadm).size !== hadm.length) errors.push(`${ID_COLS[1]} is not unique`)

  for (const c of CATEGORICAL) {
    const seen = new Set(train.data[c].map((v) => pyStr(v, train.dtypes[c])))
    const unseen = [...new Set(synth.data[c].map((v) => pyStr(v, synth.dtypes[c])))].filter((v) => !seen.has(v)).sort()
    if (unseen.length) errors.push(`${c}: ${unseen.length} levels never seen in train, e.g. ${pyList(unseen.slice(0, 3))}`)
  }
  for (const c of BINARY) {
    if (!synth.data[c].every((v) => !isMissing(v) && (num(v) === 0 || num(v) === 1))) errors.push(`${c}: values outside {0, 1}`)
  }

  const age = synth.data[AGE_COL]
  if (!age.every((v) => !isMissing(v) && typeof num(v) === 'number' && num(v) >= 0 && num(v) <= AGE_CAP)) {
    errors.push(`${AGE_COL}: values outside [0, ${AGE_CAP}]`)
  }
  const flag = synth.data[AGE_FLAG_COL]
  if (age.some((v, i) => truthy(flag[i]) && (isMissing(v) || num(v) !== AGE_CAP))) {
    errors.push(`${AGE_FLAG_COL} rows must have ${AGE_COL} == ${AGE_CAP}`)
  }

  const continuous = real.columns.filter((c) => c.startsWith('los_') || c.startsWith('lab_'))
  for (const c of continuous) {
    const v = synth.data[c]
    if (v.some((x) => x === Infinity || x === -Infinity)) errors.push(`${c}: infinite values`)
    if (v.some(isMissing) && !train.data[c].some(isMissing)) errors.push(`${c}: missing values, but this column is never missing in train`)
    if (v.some((x) => !isMissing(x) && num(x) < 0)) errors.push(`${c}: negative values`)
    if (c.endsWith('_abnormal_frac') && v.some((x) => !isMissing(x) && num(x) > 1)) errors.push(`${c}: fraction above 1`)
  }
  const labIds = real.columns.filter((c) => c.startsWith('lab_') && c.endsWith('_mean')).map((c) => c.slice(4, -5))
  for (const id of labIds) {
    const m = synth.data[`lab_${id}_mean`]
    const f = synth.data[`lab_${id}_abnormal_frac`]
    if (m.some((v, i) => isMissing(v) !== isMissing(f[i]))) errors.push(`lab ${id}: mean and abnormal_frac must be missing together`)
  }

  const trainCodes = new Set(train.data[PRIMARY_COL].filter((v) => !isMissing(v)))
  for (const raw of train.data[CODES_COL]) for (const c of loadsJson(raw)) trainCodes.add(c)
  const problems = new Map()
  for (let i = 0; i < synth.n; i++) {
    const p = codeProblem(
      synth.data[CODES_COL][i],
      pyStr(synth.data[PRIMARY_COL][i], synth.dtypes[PRIMARY_COL]),
      synth.data[N_DIAG_COL][i],
      trainCodes,
    )
    if (p) problems.set(p, (problems.get(p) || 0) + 1)
  }
  for (const [p, n] of problems) errors.push(`${CODES_COL}: ${n} rows with ${p}`)
  return errors
}
