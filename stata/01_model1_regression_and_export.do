* ----------------------------------------------------------------------
* 01_model1_regression_and_export.do
* Model 1: sample, outcome, survey-adjusted logistic regression (Tables 1 and 2), AUC, export for Python.
*
* INPUT : $datadir/RWBR91FL.dta   (2025 Rwanda DHS Births Recode; obtain it from the DHS Program, see data/README.md)
* OUTPUT: $datadir/model1_analytic_data_improved.dta  (analytic file read by the Python pipeline)
*         regression tables and AUC are printed in the Results window; use `log using` to save them.
* RUN   : in Stata, `cd` to the repository folder, set `global datadir` in the Command window only if your data are elsewhere (see below), then `do stata/01_model1_regression_and_export.do`
* Stata version used for the paper: 18
* ----------------------------------------------------------------------

* Folder that contains RWBR91FL.dta (forward slashes work on all systems)
* Default: a data/ folder inside the repository. To use another folder WITHOUT editing this file,
* type this in the Command window first (then run the whole do-file):
*     global datadir "C:/full/path/to/folder"
if "$datadir" == "" global datadir "data"

* Stop with a clear message if the survey file cannot be found in that folder
capture confirm file "$datadir/RWBR91FL.dta"
if _rc {
    display as error "Cannot find RWBR91FL.dta in the folder given by global datadir: $datadir"
    display as error "Either use cd to move to the repository folder (and put the file in data/), or type in the Command window: global datadir followed by the folder path in quotes, then run the whole do-file again."
    exit 601
}

* ================================================================
* NEONATAL MORTALITY PREDICTION - RWANDA 2025 RDHS
* Model 1: Literature-justified predictors, full 5-year sample
* ================================================================


* ================================================================
* SECTION 1: LOAD DATA AND RESTRICT TO 5-YEAR RECALL PERIOD
* ================================================================

use "$datadir/RWBR91FL.dta", clear

gen birth_recency = v008 - b3
keep if birth_recency >= 1 & birth_recency <= 60
* The month of interview (birth_recency == 0) is excluded, following the
* standard DHS convention for mortality reference periods. This leaves
* 7,214 births in the five-year window.


* ================================================================
* SECTION 2: CONSTRUCT OUTCOME VARIABLE (neonatal_death)
* 0-27 completed days, using b5 (alive/dead) and b6 (age at death)
* ================================================================

gen died = (b5 == 0)
label define died_lbl 0 "Alive" 1 "Dead"
label values died died_lbl

gen age_death_days = .
replace age_death_days = b6 - 100 if b6 >= 100 & b6 <= 131            // died in days
replace age_death_days = (b6 - 200) * 30.4 if b6 >= 200 & b6 <= 259   // died in months, full u5 range

gen neonatal_death = 0
replace neonatal_death = 1 if died == 1 & age_death_days <= 27
replace neonatal_death = . if died == 1 & missing(age_death_days)     // unknown timing -> missing, not 0

label define neo_lbl 0 "Not neonatal death" 1 "Neonatal death"
label values neonatal_death neo_lbl

* Sanity checks
tab neonatal_death, m
tab died neonatal_death, m


* ================================================================
* SECTION 3: SURVEY DESIGN + VALIDATION AGAINST OFFICIAL NNMR
* ================================================================

gen wt = v005 / 1000000
svyset v021 [pw=wt], strata(v023)

* Weighted prevalence - should closely match official NNMR (17/1,000 = 1.7%)
svy: tab neonatal_death, percent ci


* ================================================================
* SECTION 4: PREPARE PREDICTORS
* ================================================================

* --- 4a. Maternal age: non-linear relationship, use categories ---
gen matage_cat = .
replace matage_cat = 1 if v012 < 20
replace matage_cat = 2 if v012 >= 20 & v012 < 35
replace matage_cat = 3 if v012 >= 35
label define ma_lbl 1 "<20" 2 "20-34" 3 "35+"
label values matage_cat ma_lbl

* --- 4b. Birth order: collapse to avoid empty/separated categories ---
gen bord_cat = .
replace bord_cat = 1 if bord == 1
replace bord_cat = 2 if bord >= 2 & bord <= 4
replace bord_cat = 3 if bord >= 5
label define bord_lbl 1 "1st" 2 "2nd-4th" 3 "5th+"
label values bord_cat bord_lbl

* --- 4c. Birth interval: first births are structurally missing on b11,
*         recode as own category so they aren't dropped ---
gen birth_interval_cat = .
replace birth_interval_cat = 0 if bord == 1
replace birth_interval_cat = 1 if bord > 1 & b11 < 24 & !missing(b11)
replace birth_interval_cat = 2 if bord > 1 & b11 >= 24 & !missing(b11)
* (31 cases remain genuinely missing: bord>1 with no recorded b11)

* Collapse to binary to avoid collinearity with bord_cat/b0 in the multivariable model
gen short_interval = (birth_interval_cat == 1)
replace short_interval = . if missing(birth_interval_cat)
label define si_lbl 0 "Not short/NA" 1 "Short interval (<24mo)"
label values short_interval si_lbl

* --- 4d. Remaining predictors used as-is (b0, v190, v106, b4, v501 - all complete) ---

* --- 4e. Collapsed wealth and marital status - LOGISTIC REGRESSION ONLY.
* Original v190 (5 categories) and v501 (6 categories) are kept unchanged
* for the ML pipeline (Section 8 export); these collapsed versions are
* used only in Sections 5, 5b, and 6 below, to address sparse cells and
* separation risk in the regression (e.g., widowed = 2 events nationally).
* Confirm v190's underlying numeric coding before trusting this recode. ---

tab v190
gen wealth_cat3 = .
replace wealth_cat3 = 1 if inlist(v190, 1, 2)   // poorest + poorer -> poor
replace wealth_cat3 = 2 if v190 == 3             // middle
replace wealth_cat3 = 3 if inlist(v190, 4, 5)   // richer + richest -> rich
label define wc3_lbl 1 "Poor" 2 "Middle" 3 "Rich"
label values wealth_cat3 wc3_lbl
tab v190 wealth_cat3, m

gen marital_cat3 = .
replace marital_cat3 = 1 if v501 == 0                          // never in union
replace marital_cat3 = 2 if inlist(v501, 1, 2)                 // married + living with partner -> currently in union
replace marital_cat3 = 3 if inlist(v501, 3, 4, 5)               // widowed + divorced + separated -> formerly in union
label define mc3_lbl 1 "Never in union" 2 "Currently in union" 3 "Formerly in union"
label values marital_cat3 mc3_lbl
tab v501 marital_cat3, m


* ================================================================
* SECTION 4b: COMPLETE-CASE SAMPLE (8 final predictors)
* Explicit restriction so Table 1 and crude ORs (Section 5b) are
* computed on the SAME sample as the final regression (Section 6),
* rather than relying on svy: logistic's implicit listwise deletion
* ================================================================

gen complete_case = 1
foreach v in matage_cat bord_cat short_interval b0 wealth_cat3 v106 b4 marital_cat3 neonatal_death {
    replace complete_case = 0 if missing(`v')
}

count if complete_case == 1
count if complete_case == 1 & neonatal_death == 1
* Result: 7,167 births with 121 neonatal deaths. The 47 exclusions are counted
* here predictor-first (31 missing preceding birth interval, then 16 with
* unresolved age at death); the paper counts the same 47 outcome-first (17, then 30).


* ================================================================
* SECTION 5: BIVARIATE SCREEN (descriptive only - all retained
* regardless of significance per literature justification)
* ================================================================

preserve
keep if complete_case == 1

tab bord_cat neonatal_death, row chi2
tab b0 neonatal_death, row chi2
tab wealth_cat3 neonatal_death, row chi2
tab v106 neonatal_death, row chi2
tab b4 neonatal_death, row chi2
tab marital_cat3 neonatal_death, row chi2
tab short_interval neonatal_death, row chi2
tab matage_cat neonatal_death, row chi2


* ================================================================
* SECTION 5b: TABLE 1 DATA (weighted n and % for baseline characteristics
* table, in n(%) format) AND CRUDE (UNADJUSTED) ODDS RATIOS
* ================================================================

* ---- Weighted N (count) and % (col) together, for n(%) reporting ----
* Run both for every variable so weighted N and % can be combined
* directly into "n(%)" format without separate lookups.

foreach v in bord_cat b0 wealth_cat3 v106 b4 marital_cat3 short_interval matage_cat {
    di "=========================================="
    di "VARIABLE: `v'"
    di "=========================================="
    svy: tab `v' neonatal_death, count
    svy: tab `v' neonatal_death, col
}

svy: tab neonatal_death, count
* Use this total (weighted N for Died and Survived) to confirm each
* variable's category counts sum to approximately the same total -
* a quick way to catch any misread category before it reaches the
* manuscript table.

* ---- Crude (unadjusted) odds ratios, one predictor at a time ----
svy: logistic neonatal_death i.bord_cat
svy: logistic neonatal_death i.b0
svy: logistic neonatal_death i.wealth_cat3
svy: logistic neonatal_death i.v106
svy: logistic neonatal_death i.b4
svy: logistic neonatal_death i.marital_cat3
svy: logistic neonatal_death i.short_interval
svy: logistic neonatal_death i.matage_cat

restore


* ================================================================
* SECTION 6: FINAL MULTIVARIABLE LOGISTIC REGRESSION (MODEL 1)
* ================================================================

keep if complete_case == 1

svy: logistic neonatal_death i.bord_cat i.b0 i.wealth_cat3 i.v106 i.b4 i.marital_cat3 i.short_interval i.matage_cat

estimates store model1_logit_final


* ================================================================
* SECTION 7: MODEL DISCRIMINATION (AUC)
* ================================================================

predict phat, pr
roctab neonatal_death phat
* Result: apparent AUC = 0.694 (95% CI 0.643-0.745)

sum phat, detail


* ================================================================
* END OF MODEL 1
* Main adjusted odds ratios: multiple birth 6.2 to 50.5, female sex 0.63,
* maternal age 20-34 0.18 and 35+ 0.21 (vs <20). Apparent AUC = 0.694.
* ================================================================


* ================================================================
* SECTION 8: EXPORT ANALYTIC DATASET FOR ML (PYTHON)
* .dta only - preserves exact numeric encoding, missing values,
* and variable/value labels; avoids CSV parsing ambiguity
* ================================================================

* Work on a copy so the in-memory Model 1 dataset (with phat etc.) is untouched
preserve

keep neonatal_death bord_cat b0 v190 v106 b4 v501 short_interval matage_cat wt v021 v023

* Same listwise deletion as the svy: logistic model used
drop if missing(neonatal_death)
drop if missing(bord_cat) | missing(b0) | missing(v190) | missing(v106) ///
     | missing(b4) | missing(v501) | missing(short_interval) | missing(matage_cat)

* Confirm final sample size matches Model 1 corrected run (n=7,167, 121 events)
count
tab neonatal_death

save "$datadir/model1_analytic_data_improved.dta", replace

restore
* ================================================================
* END OF EXPORT - proceed to Python for ML comparators
* (Random Forest, XGBoost, SHAP) on model1_analytic_data.dta
* ================================================================
