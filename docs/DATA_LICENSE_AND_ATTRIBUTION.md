# Data licence and attribution

## The data

- **Dataset:** Pappalardo, L. & Massucco, E. (2019). *Soccer match event dataset.* figshare. Collection. [doi:10.6084/m9.figshare.c.4415000.v5](https://doi.org/10.6084/m9.figshare.c.4415000.v5).
- **Paper:** Pappalardo, L., Cintia, P., Rossi, A., Massucco, E., Ferragina, P., Pedreschi, D. & Giannotti, F. (2019). A public data set of spatio-temporal match events in soccer competitions. *Scientific Data* 6:236. [doi:10.1038/s41597-019-0247-7](https://doi.org/10.1038/s41597-019-0247-7). The collection description also asks users to cite the *PlayeRank* paper (ACM Transactions on Intelligent Systems and Technology 10(5), Article 59, [doi:10.1145/3343172](https://doi.org/10.1145/3343172)).
- **Original data provider:** Wyscout collected and provided the underlying event data.
- **Licence of the dataset:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/), as stated on the Figshare articles and in the paper. The grant is made by the dataset authors. The chain of rights from Wyscout to them was not independently verified in this project.
- **Files used** (pinned in `src/expected_goals_lab/wyscout.py` by Figshare file ID, size and MD5, and verified on every load): `events.zip`, `matches.zip`, `competitions.json`, `eventid2name.csv`, `tags2name.csv`. Only the five domestic league files (England, France, Germany, Italy, Spain, 2017/18) are read.

## What this repository does with it

- **Nothing from the dataset is redistributed.** Raw archives, event files, caches and the locally built shot table are never stored here; the code downloads the public files from Figshare on the user's machine and checks them against the pinned checksums. The `.gitignore` and `tests/test_repository_hygiene.py` are written to keep it that way.
- **Adaptations (the CC BY 4.0 "indication of changes").** The data were filtered to the five domestic league files and to one event class (regular shots), parsed into flat rows, quarantined by quality rules, and augmented with derived columns (metric coordinates, distance, angle). Models, figures and aggregate tables were then computed from those rows. Every figure and report carries the credit line.
- **Published material derived from the data** consists of aggregate tables, figures, one validation-window match map (shot positions and model probabilities for the 20 regular shots of a single match) and short event-level excerpts in the audit reports (a few events quoted to document data issues). These are adaptations of the dataset and are shared under the attribution above, with the changes indicated as described.
- **No endorsement.** Neither the dataset authors nor Wyscout have reviewed, endorsed or are affiliated with this project or its results. Wyscout and any other names are the property of their owners and are used only to identify the source.

## The code and the data licence are separate

The original source code, tests and documentation in this repository are licensed under the MIT licence in [`LICENSE`](../LICENSE). That licence does not apply to the dataset or to the dataset authors' publications, and it does not override any third-party right. The CC BY 4.0 grant for the dataset does not make the code CC BY, and does not extend to any other brand, tool or upstream provider material.

## No other provider's data

No data, code, assets or outputs from any other football-data provider are used in this repository.

## Suggested citation of this work

See [`CITATION.cff`](../CITATION.cff). Please also cite the dataset and the paper above.
