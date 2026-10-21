import { useRef, useState } from 'react'

// A drop zone that is also a file picker, so it works by mouse, keyboard and drag and drop.
export function FileDrop({ id, title, hint, accept, multiple, onFiles }) {
  const [over, setOver] = useState(false)
  const exts = accept.split(',').map((a) => a.trim().toLowerCase())
  const keep = (files) => files.filter((f) => exts.some((e) => f.name.toLowerCase().endsWith(e)))
  return (
    <label
      htmlFor={id}
      className={`drop${over ? ' over' : ''}`}
      onDragOver={(e) => {
        e.preventDefault()
        setOver(true)
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault()
        setOver(false)
        const files = keep([...e.dataTransfer.files])
        if (files.length) onFiles(multiple ? files : files.slice(0, 1))
      }}
    >
      <input
        id={id}
        type="file"
        accept={accept}
        multiple={multiple}
        className="sr-only"
        onChange={(e) => {
          onFiles([...e.target.files])
          e.target.value = ''
        }}
      />
      <span className="drop-title">{title}</span>
      <span className="small muted">{hint}</span>
      <span className="drop-cta">{multiple ? 'Choose files' : 'Choose a file'} or drop {multiple ? 'them' : 'it'} here</span>
    </label>
  )
}

export function FileChips({ items, onRemove, onClear, collapseAfter = 6 }) {
  const [open, setOpen] = useState(false)
  if (!items.length) return null
  const many = items.length > collapseAfter
  const visible = many && !open ? items.filter((it) => it.error) : items
  return (
    <div className="files">
      {many && (
        <span className="filechip summary">
          <span>{items.length} files</span>
          <button type="button" className="tbtn" onClick={() => setOpen(!open)} aria-expanded={open}>{open ? 'Hide list' : 'Show all'}</button>
          {onClear && <button type="button" className="tbtn" onClick={onClear}>Remove all</button>}
        </span>
      )}
      {visible.map((it) => (
        <span key={it.name} className={`filechip${it.error ? ' bad' : ''}`} title={it.error || it.name}>
          <span>{it.name}</span>
          <button type="button" aria-label={`Remove ${it.name}`} onClick={() => onRemove(it.name)}>
            ×
          </button>
        </span>
      ))}
    </div>
  )
}

// Tabs with the ARIA pattern: arrow keys move between them, only the selected one is in the tab order.
export function Tabs({ tabs, value, onChange, idPrefix }) {
  const refs = useRef([])
  const onKey = (e, i) => {
    const step = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0
    if (!step) return
    e.preventDefault()
    const next = (i + step + tabs.length) % tabs.length
    onChange(tabs[next].id)
    refs.current[next]?.focus()
  }
  return (
    <div className="tabs" role="tablist" aria-label="Workbench views">
      {tabs.map((t, i) => (
        <button
          key={t.id}
          ref={(el) => (refs.current[i] = el)}
          role="tab"
          id={`${idPrefix}-tab-${t.id}`}
          aria-controls={`${idPrefix}-panel-${t.id}`}
          aria-selected={value === t.id}
          tabIndex={value === t.id ? 0 : -1}
          onClick={() => onChange(t.id)}
          onKeyDown={(e) => onKey(e, i)}
        >
          <span className="long">{t.label}</span>
          <span className="short" aria-hidden="true">{t.short || t.label}</span>
          {t.count != null && <span className="count">{t.count}</span>}
        </button>
      ))}
    </div>
  )
}

const ICONS = {
  pass: <path d="M4 9.2l3.2 3.2L14 5.5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />,
  fail: <path d="M5 5l8 8M13 5l-8 8" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />,
  warn: <><path d="M9 2.5l7 12.5H2z" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinejoin="round" /><path d="M9 7.2v3.6M9 13v.1" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" /></>,
}

// A status line: color is never the only signal, every state has its icon and its own words.
export function Check({ kind, lead, children }) {
  return (
    <div className={`check ${kind}`} role={kind === 'fail' ? 'alert' : 'status'}>
      <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true" style={{ color: `var(--${kind === 'pass' ? 'good' : kind === 'fail' ? 'crit' : 'warn'}-ink)` }}>
        {ICONS[kind]}
      </svg>
      <div>
        <span className="lead">{lead}</span> {children}
      </div>
    </div>
  )
}

export function YesNo({ yes }) {
  return (
    <span className={`yesno ${yes ? 'yes' : 'no'}`}>
      <svg width="14" height="14" viewBox="0 0 18 18" aria-hidden="true">{yes ? ICONS.pass : <circle cx="9" cy="9" r="5.5" fill="none" stroke="currentColor" strokeWidth="1.8" />}</svg>
      {yes ? 'Yes' : 'No'}
    </span>
  )
}

// How a cell reads in the preview tables, close to Streamlit's st.dataframe.
export function fmtCell(v, dtype) {
  if (v === null || v === undefined) return <span className="nil">None</span>
  if (typeof v === 'boolean') return v ? 'True' : 'False'
  if (typeof v === 'number') {
    if (!Number.isFinite(v)) return v > 0 ? 'inf' : v < 0 ? '-inf' : 'NaN'
    return dtype === 'float64' ? v.toFixed(4) : String(v)
  }
  return v.length > 28 ? <span title={v}>{v.slice(0, 27)}…</span> : v
}

export const f4 = (x) => (x === null || x === undefined || Number.isNaN(x) ? '—' : x.toFixed(4))
export const pct = (x) => (Number.isNaN(x) ? '—' : `${(x * 100).toFixed(1)}%`)
