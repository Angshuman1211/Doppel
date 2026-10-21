// Static words. Every number on the page comes from data/dashboard.json, not from this file.

export const STATUS_LABEL = { done: 'Done', partial: 'Partly done', none: 'Not started' }

export const STEPS = [
  { n: 1, title: 'Prepare the data', text: 'Turn 100 MIMIC-III Demo patients into 129 clean hospital admissions, split by patient into 94 training and 35 held-out rows.', owner: 'Track 3', status: 'done' },
  { n: 2, title: 'Generate synthetic records', text: 'Each generator learns from the 94 training rows only and produces a synthetic table of the same shape.', owner: 'Track 1', status: 'done' },
  { n: 3, title: 'Evaluate on one protocol', text: 'Score every generator for realism, usefulness and privacy, over 20 random seeds each, against fixed reference points.', owner: 'Track 2', status: 'done' },
  { n: 4, title: 'Compare and deliver', text: 'Show the trade-offs, run the pipeline in containers, and write it up.', owner: 'Track 4', status: 'partial' },
]

export const TERMS = [
  ['Fidelity', 'How close the synthetic records are to the real ones, column by column. Lower is better.'],
  ['Utility', 'How well a model trained on synthetic records predicts in-hospital death for real patients it never saw. 0.5 is guessing; higher is better.'],
  ['Membership attack', 'An attacker tries to tell which real patients were used to train the generator. 0.5 means guessing, 1.0 means it always wins. Lower is better.'],
  ['Positive control', 'A generator known to copy its training data, used to check that the attacks can catch a leak. Here, TVAE and the 3000-step diffusion model.'],
  ['Pareto frontier', 'The generators that no other generator beats on all three measures at once. It shows trade-offs; it does not rank.'],
  ['Seeds and whiskers', 'Every generator was run 20 times. Dots show the average, whiskers show how much the runs varied.'],
]

export const LIMITS = [
  ['The dataset is tiny.', '94 training patients and 35 held-out ones, only 6 of whom died. The 20 runs vary the generators, not the patients.'],
  ['Prediction scores cannot rank the generators.', 'The highest belong to generators that copy or that produce the wrong rate of deaths.'],
  ['A low attack score is not proof of safety.', 'It means these attacks found nothing, not that nothing can be found.'],
]

export const DOCS = [
  ['Final evaluation result', 'docs/final_evaluation_result.md', 'All nine generators, every measure and test'],
  ['Evaluation section of the report', 'docs/report_evaluation_section.md', 'The method and results, written up'],
  ['Evaluation protocol', 'docs/eval_protocol.md', 'Formulas and choices behind every measure'],
  ['Project README', 'README.md', 'Architecture and how to run everything'],
]
