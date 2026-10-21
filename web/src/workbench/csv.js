// Reads a CSV the way the Streamlit app's pd.read_csv(..., dtype={"icd9_primary": str}) does, so every number the
// workbench shows matches the Python one. A frame is { columns, data: {name: values[]}, dtypes: {name: dtype}, n }.
// Missing cells are null. dtype is one of int64, float64, bool, object, as pandas would report it.

// pandas' default strings that read as missing (read_csv na_values).
const NA = new Set(['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN',
  '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'None', 'n/a', 'nan', 'null'])
const TRUE = new Set(['True', 'TRUE', 'true'])
const FALSE = new Set(['False', 'FALSE', 'false'])
const INT = /^[+-]?\d+$/
const FLOAT = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/
const INF = /^[+-]?(inf|infinity)$/i

// Columns read as text no matter what they look like ('0389' read as a number becomes 389, another code, P-005).
export const PINNED_TEXT = ['icd9_primary']

// pandas' default float reader (float_precision='high', precise_xstrtod in pandas/_libs/src/parser/tokenizer.c).
// It is not correctly rounded: '0.42000000000000004' reads as 0.42, where JavaScript's Number() gives the exact
// double. A value one bit off can land in a different histogram bin, so fidelity only matches pandas with this port.
const POW10 = Array.from({ length: 309 }, (_, k) => Number(`1e${k}`))
export function preciseXstrtod(text) {
  const s = text.trim()
  let p = 0
  let negative = false
  if (s[p] === '-' || s[p] === '+') {
    negative = s[p] === '-'
    p++
  }
  const isDigit = (ch) => ch >= '0' && ch <= '9'
  let number = 0
  let exponent = 0
  let numDigits = 0
  let numDecimals = 0
  const maxDigits = 17
  while (p < s.length && isDigit(s[p])) {
    if (numDigits < maxDigits) {
      number = number * 10 + (s.charCodeAt(p) - 48)
      numDigits++
    } else exponent++
    p++
  }
  if (s[p] === '.') {
    p++
    while (numDigits < maxDigits && p < s.length && isDigit(s[p])) {
      number = number * 10 + (s.charCodeAt(p) - 48)
      p++
      numDigits++
      numDecimals++
    }
    if (numDigits >= maxDigits) while (p < s.length && isDigit(s[p])) p++
    exponent -= numDecimals
  }
  if (numDigits === 0) return NaN
  if (negative) number = -number
  if (s[p] === 'e' || s[p] === 'E') {
    let neg = false
    p++
    if (s[p] === '-' || s[p] === '+') {
      neg = s[p] === '-'
      p++
    }
    let n = 0
    while (p < s.length && isDigit(s[p])) {
      n = n * 10 + (s.charCodeAt(p) - 48)
      p++
    }
    exponent += neg ? -n : n
  }
  if (exponent > 308) return negative ? -Infinity : Infinity
  if (exponent > 0) number *= POW10[exponent]
  else if (exponent < -308) number = exponent < -616 ? 0 : number / POW10[-308 - exponent] / POW10[308]
  else number /= POW10[-exponent]
  return number
}

// RFC 4180 parsing: quoted fields, doubled quotes, commas and newlines inside quotes, CRLF or LF.
export function parseCsv(text) {
  if (text.charCodeAt(0) === 0xfeff) text = text.slice(1)
  const rows = []
  let row = []
  let field = ''
  let quoted = false
  for (let i = 0; i < text.length; i++) {
    const ch = text[i]
    if (quoted) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          field += '"'
          i++
        } else quoted = false
      } else field += ch
    } else if (ch === '"') quoted = true
    else if (ch === ',') {
      row.push(field)
      field = ''
    } else if (ch === '\n' || ch === '\r') {
      if (ch === '\r' && text[i + 1] === '\n') i++
      row.push(field)
      rows.push(row)
      row = []
      field = ''
    } else field += ch
  }
  if (field !== '' || row.length) {
    row.push(field)
    rows.push(row)
  }
  // pandas skips blank lines
  return rows.filter((r) => !(r.length === 1 && r[0] === ''))
}

// pandas renames repeated headers a, a.1, a.2 ...
function dedupe(header) {
  const seen = new Map()
  return header.map((h) => {
    if (!seen.has(h)) {
      seen.set(h, 0)
      return h
    }
    let k = seen.get(h) + 1
    while (seen.has(`${h}.${k}`)) k++
    seen.set(h, k)
    seen.set(`${h}.${k}`, 0)
    return `${h}.${k}`
  })
}

function inferColumn(cells, pinned) {
  const present = cells.filter((c) => !NA.has(c))
  const hasNa = present.length < cells.length
  if (pinned) return { dtype: 'object', values: cells.map((c) => (NA.has(c) ? null : c)) }
  // pandas: a header with no rows gives object columns; rows that are all empty give float64
  if (cells.length === 0) return { dtype: 'object', values: [] }
  if (present.length === 0) return { dtype: 'float64', values: cells.map(() => null) }
  if (present.every((c) => TRUE.has(c) || FALSE.has(c))) {
    return { dtype: hasNa ? 'object' : 'bool', values: cells.map((c) => (NA.has(c) ? null : TRUE.has(c))) }
  }
  if (present.every((c) => INT.test(c))) {
    return { dtype: hasNa ? 'float64' : 'int64', values: cells.map((c) => (NA.has(c) ? null : Number(c))) }
  }
  if (present.every((c) => INT.test(c) || FLOAT.test(c) || INF.test(c))) {
    return {
      dtype: 'float64',
      values: cells.map((c) => (NA.has(c) ? null : INF.test(c) ? (c.startsWith('-') ? -Infinity : Infinity) : preciseXstrtod(c))),
    }
  }
  return { dtype: 'object', values: cells.map((c) => (NA.has(c) ? null : c)) }
}

export function frameFromCsv(text) {
  const rows = parseCsv(text)
  if (!rows.length) throw new Error('the file is empty')
  const columns = dedupe(rows[0])
  const body = rows.slice(1)
  const width = columns.length
  for (let r = 0; r < body.length; r++) {
    if (body[r].length > width) throw new Error(`Error tokenizing data. Expected ${width} fields in line ${r + 2}, saw ${body[r].length}`)
  }
  const data = {}
  const dtypes = {}
  columns.forEach((name, j) => {
    const cells = body.map((r) => (j < r.length ? r[j] : ''))
    const { dtype, values } = inferColumn(cells, PINNED_TEXT.includes(name))
    data[name] = values
    dtypes[name] = dtype
  })
  return { columns, data, dtypes, n: body.length }
}

export const isNumericDtype = (dtype) => dtype === 'int64' || dtype === 'float64' || dtype === 'bool'
export const kindOf = (dtype) => ({ int64: 'i', float64: 'f', bool: 'b', object: 'O' })[dtype]

// Python's repr of a float: shortest round-trip digits, exponent form outside 1e-4 <= |x| < 1e16.
export function pyFloat(x) {
  if (Number.isNaN(x)) return 'nan'
  if (x === Infinity) return 'inf'
  if (x === -Infinity) return '-inf'
  if (x === 0) return Object.is(x, -0) ? '-0.0' : '0.0'
  const [mant, expStr] = Math.abs(x).toExponential().split('e')
  const exp = Number(expStr)
  const digits = mant.replace('.', '')
  const sign = x < 0 ? '-' : ''
  if (exp >= -4 && exp < 16) {
    let s
    if (exp >= digits.length - 1) s = digits + '0'.repeat(exp - digits.length + 1) + '.0'
    else if (exp >= 0) s = digits.slice(0, exp + 1) + '.' + digits.slice(exp + 1)
    else s = '0.' + '0'.repeat(-exp - 1) + digits
    return sign + s
  }
  const m = digits.length > 1 ? digits[0] + '.' + digits.slice(1) : digits
  return `${sign}${m}e${exp < 0 ? '-' : '+'}${String(Math.abs(exp)).padStart(2, '0')}`
}

// Series.astype(str) for one cell of a column with the given dtype. Missing becomes 'nan', as in pandas.
export function pyStr(v, dtype) {
  if (v === null || v === undefined) return 'nan'
  if (typeof v === 'boolean') return v ? 'True' : 'False'
  if (typeof v === 'number') return dtype === 'float64' ? pyFloat(v) : String(v)
  return v
}

export function subset(frame, keep) {
  const idx = []
  for (let i = 0; i < frame.n; i++) if (keep(i)) idx.push(i)
  const data = {}
  for (const c of frame.columns) data[c] = idx.map((i) => frame.data[c][i])
  return { columns: frame.columns, data, dtypes: frame.dtypes, n: idx.length }
}
