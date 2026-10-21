import { useState } from 'react'
import { frameFromCsv } from '../workbench/csv'
import { loadReport } from '../workbench/reports'
import { FileChips, FileDrop, Tabs } from '../workbench/ui'
import ExploreTab from '../workbench/ExploreTab'
import CompareTab from '../workbench/CompareTab'
import EvaluateTab from '../workbench/EvaluateTab'

// Everything the Streamlit dashboard (dashboard/app.py) does, in the browser: upload synthetic datasets, the real
// cleaned dataset, generator manifests and evaluation reports, then explore, compare and rank them. Nothing leaves
// the page; files are read with the File API and every number is computed here.
const upsert = (list, items) => {
  const names = new Set(items.map((i) => i.name))
  return [...list.filter((x) => !names.has(x.name)), ...items]
}

async function readCsv(file) {
  try {
    return { name: file.name, frame: frameFromCsv(await file.text()) }
  } catch (e) {
    return { name: file.name, error: e.message }
  }
}

async function readJson(file) {
  try {
    return { name: file.name, json: JSON.parse(await file.text()) }
  } catch (e) {
    return { name: file.name, error: e.message }
  }
}

export default function Workbench() {
  const [synth, setSynth] = useState([])
  const [real, setReal] = useState(null)
  const [manifests, setManifests] = useState([])
  const [reports, setReports] = useState([])
  const [tab, setTab] = useState('explore')

  const addSynth = async (files) => {
    const items = await Promise.all(files.map(readCsv))
    setSynth((s) => upsert(s, items))
  }
  const addReal = async (files) => setReal(await readCsv(files[0]))
  const addManifests = async (files) => {
    const items = await Promise.all(files.map(readJson))
    setManifests((m) => upsert(m, items))
  }
  const addReports = async (files) => {
    const items = await Promise.all(files.map(async (f) => ({ name: f.name, report: loadReport(f.name, await f.text()) })))
    setReports((r) => upsert(r, items))
  }

  const datasets = synth.filter((d) => !d.error)
  const realOk = real && !real.error ? real : null
  const errors = [...synth, ...(real ? [real] : []), ...manifests].filter((x) => x.error)
  const tabs = [
    { id: 'explore', label: 'Dataset explorer', short: 'Explore', count: datasets.length || null },
    { id: 'compare', label: 'Compare to real data', short: 'Compare' },
    { id: 'evaluate', label: 'Evaluation and Pareto', short: 'Evaluate', count: reports.filter((r) => !r.report.error).length || null },
  ]

  return (
    <section id="upload" aria-labelledby="h-upload">
      <div className="wrap">
        <div className="section-head">
          <span className="eyebrow">Upload</span>
          <h2 id="h-upload">Check your own files</h2>
          <p>
            Upload a synthetic dataset to explore it, add the real cleaned dataset to score fidelity against it, and upload evaluation reports to compare generators.
            This does what the Streamlit dashboard does, and the files never leave your browser.
          </p>
        </div>

        <div className="uploads">
          <div>
            <FileDrop
              id="wb-synth"
              title="Synthetic datasets (CSV)"
              hint="Output of python -m generators.generate, or any CSV with the same columns."
              accept=".csv"
              multiple
              onFiles={addSynth}
            />
            <FileChips items={synth} onRemove={(n) => setSynth((s) => s.filter((x) => x.name !== n))} onClear={() => setSynth([])} />
          </div>
          <div>
            <FileDrop
              id="wb-real"
              title="Real cleaned dataset (CSV, optional)"
              hint="output/mimic_demo_clean.csv. Needed to score fidelity and to check the output contract."
              accept=".csv"
              onFiles={addReal}
            />
            <FileChips items={real ? [real] : []} onRemove={() => setReal(null)} />
          </div>
          <div>
            <FileDrop
              id="wb-manifest"
              title="Generator manifests (JSON, optional)"
              hint="The .manifest.json a generator writes next to each CSV. Matched to a dataset by file name."
              accept=".json"
              multiple
              onFiles={addManifests}
            />
            <FileChips items={manifests} onRemove={(n) => setManifests((m) => m.filter((x) => x.name !== n))} onClear={() => setManifests([])} />
          </div>
        </div>
        {errors.length > 0 && (
          <ul className="file-errors" role="alert">
            {errors.map((e) => (
              <li key={e.name}>
                <b>{e.name}</b>: {e.error}
              </li>
            ))}
          </ul>
        )}
        <p className="engine-note">
          Running in the browser: fidelity is a port of Track 2&rsquo;s <code>evaluation.fidelity_metrics</code> and the contract check a port of Track 1&rsquo;s{' '}
          <code>generators.validate</code>, both checked against the Python on 28,429 values from the project&rsquo;s own files.
        </p>

        <Tabs tabs={tabs} value={tab} onChange={setTab} idPrefix="wb" />
        <div role="tabpanel" id={`wb-panel-${tab}`} aria-labelledby={`wb-tab-${tab}`}>
          {tab === 'explore' && <ExploreTab datasets={datasets} manifests={manifests} />}
          {tab === 'compare' && <CompareTab datasets={datasets} real={realOk} />}
          {tab === 'evaluate' && (
            <EvaluateTab reports={reports} onAdd={addReports} onRemove={(n) => setReports((r) => r.filter((x) => x.name !== n))} onClear={() => setReports([])} />
          )}
        </div>
      </div>
    </section>
  )
}
