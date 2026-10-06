# Data (not included)

This repository contains **no survey data**. The Demographic and Health Survey (DHS) data are available free of charge for registered users, but the DHS Program's terms do not allow redistribution, so please do not upload data files here.

## How to obtain the file

1. Register for access on the DHS Program website (dhsprogram.com) and describe your intended use.
2. Request the **2025 Rwanda Standard DHS** dataset.
3. Download the **Births Recode** file in Stata format, named `RWBR91FL.dta`.
4. Place `RWBR91FL.dta` in this `data/` folder.

## Files created by the code

Running the two Stata do-files writes these analytic files into this folder. They are derived from the survey data and must also stay out of the repository (they are excluded by `.gitignore`):

- `model1_analytic_data_improved.dta`
- `model2_analytic_data_improved.dta`
