# Data License

The data files in this repository are derived from the **MIMIC-III Clinical Database Demo v1.4**. They are distributed under the [Open Data Commons Open Database License v1.0 (ODbL)](https://opendatacommons.org/licenses/odbl/1-0/), the license the source dataset is published under.

This covers:

- `output/mimic_demo_clean.csv`, `output/icd9_lookup.csv`, `output/lab_item_lookup.csv`
- any synthetic data produced by `generators/` (written to `output/synthetic/`, not committed), since it is derived from `output/mimic_demo_clean.csv`

The code in this repository is licensed separately under MIT. See [`LICENSE`](LICENSE).

## Source and citation

Johnson, A., Pollard, T., & Mark, R. (2019). MIMIC-III Clinical Database Demo (version 1.4). *PhysioNet*. https://doi.org/10.13026/C2HM2Q

Source page: https://physionet.org/content/mimiciii-demo/1.4/

PhysioNet also asks users to cite the original MIMIC-III publication and the PhysioNet platform:

- Johnson, A. E. W., Pollard, T. J., Shen, L., Lehman, L. H., Feng, M., Ghassemi, M., Moody, B., Szolovits, P., Celi, L. A., & Mark, R. G. (2016). MIMIC-III, a freely accessible critical care database. *Scientific Data*, 3, 160035.
- Pollard, T., Moody, B. E., Lehman, L., Gow, B., Fernandes, C., Xie, C., Johnson, A., Mark, R. G., & Heldt, T. (2026). PhysioNet as a global platform for biomedical research. *Nature Health*, 1(8), 792–795. https://doi.org/10.1038/s44360-026-00096-z
