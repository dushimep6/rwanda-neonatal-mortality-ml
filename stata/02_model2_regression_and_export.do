* ----------------------------------------------------------------------
* 02_model2_regression_and_export.do
* Model 2 (sensitivity): 3-year window, extra predictors, complete-case exclusions (Supplementary Table S4), Tables 1 and 4, AUC, export for Python.
*
* INPUT : $datadir/RWBR91FL.dta   (2025 Rwanda DHS Births Recode; obtain it from the DHS Program, see data/README.md)
* OUTPUT: $datadir/model2_analytic_data_improved.dta  (analytic file read by the Python pipeline)
*         regression tables and AUC are printed in the Results window; use `log using` to save them.
* RUN   : in Stata, `cd` to the repository folder, set `global datadir` in the Command window only if your data are elsewhere (see below), then `do stata/02_model2_regression_and_export.do`
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
* MODEL 2: 3-year restricted sample, richer clinical/care predictors
* ================================================================


* ================================================================
* SECTION 1: LOAD DATA AND CONSTRUCT OUTCOME (identical logic to Model 1)
* ================================================================

use "$datadir/RWBR91FL.dta", clear

gen birth_recency = v008 - b3

gen died = (b5 == 0)
gen age_death_days = .
replace age_death_days = b6 - 100 if b6 >= 100 & b6 <= 131
replace age_death_days = (b6 - 200) * 30.4 if b6 >= 200 & b6 <= 259

gen neonatal_death = 0
replace neonatal_death = 1 if died == 1 & age_death_days <= 27
replace neonatal_death = . if died == 1 & missing(age_death_days)

* DEFINITION NOTE: neonatal death is defined here as death within 27
* completed days of life (0-27 days, i.e. the first 28 days), following the
* standard WHO/clinical definition used throughout the study's literature
* base. This is NOT the same as the DHS Guide to Statistics' 0-30-day
* mortality indicator convention used for some published DHS mortality rates.

label define neo_lbl 0 "Not neonatal death" 1 "Neonatal death"
label values neonatal_death neo_lbl


* ================================================================
* SECTION 2: RESTRICT TO 3-YEAR WINDOW
* (m-series delivery/ANC module only administered for these births)
* ================================================================

keep if birth_recency >= 1 & birth_recency <= 36
* The month of interview (birth_recency == 0) is excluded, consistent with the
* standard DHS mortality reference period convention. This leaves 4,345 births
* in the three-year window.

count
tab neonatal_death, m
* Result: 4,345 births, of which 5 have unresolved age at death.


* ================================================================
* SECTION 3: MODEL 1 PREDICTORS - identical recodes, validated earlier
* ================================================================

gen matage_cat = .
replace matage_cat = 1 if v012 < 20
replace matage_cat = 2 if v012 >= 20 & v012 < 35
replace matage_cat = 3 if v012 >= 35
label define ma_lbl 1 "<20" 2 "20-34" 3 "35+"
label values matage_cat ma_lbl

gen bord_cat = .
replace bord_cat = 1 if bord == 1
replace bord_cat = 2 if bord >= 2 & bord <= 4
replace bord_cat = 3 if bord >= 5
label define bord_lbl 1 "1st" 2 "2nd-4th" 3 "5th+"
label values bord_cat bord_lbl

* Preceding birth interval (short_interval) is retained in the final Model 2
* regression (Section 7), as in Model 1, although it was not significant at
* the bivariate stage (p=0.897).
gen birth_interval_cat = .
replace birth_interval_cat = 0 if bord == 1
replace birth_interval_cat = 1 if bord > 1 & b11 < 24 & !missing(b11)
replace birth_interval_cat = 2 if bord > 1 & b11 >= 24 & !missing(b11)
gen short_interval = (birth_interval_cat == 1)
replace short_interval = . if missing(birth_interval_cat)
label define si_lbl 0 "Not short/NA" 1 "Short interval (<24mo)"
label values short_interval si_lbl

* Multiple birth: collapsed to binary to avoid separation
* (3rd of multiple = 1 case, guarantees perfect separation otherwise)
gen multiple_birth = (b0 != 0) if !missing(b0)
label define mb_lbl 0 "Single birth" 1 "Multiple birth"
label values multiple_birth mb_lbl

* Marital status: sparse categories (widowed, divorced, separated)
* collapsed into "Formerly partnered" to avoid separation
* (widowed = 0 events among 27 cases, guarantees perfect separation)
* This 4-category version feeds the ML pipeline (Section 9 export) -
* left unchanged so existing ML results remain valid.
gen marital_cat = .
replace marital_cat = 1 if v501 == 0
replace marital_cat = 2 if v501 == 1
replace marital_cat = 3 if v501 == 2
replace marital_cat = 4 if inlist(v501, 3, 4, 5)
label define mc_lbl 1 "Never in union" 2 "Married" 3 "Living with partner" 4 "Formerly partnered"
label values marital_cat mc_lbl

* --- Collapsed wealth and marital status - LOGISTIC REGRESSION ONLY.
* Used only in Sections 6, 6b, and 7 below, matching the same collapse
* applied in Model 1, for consistency between the two regression
* analyses. v190 numeric coding verified via codebook (1=poorest,
* 5=richest, 0 missing). ML pipeline (Section 9) is untouched. ---

gen wealth_cat3 = .
replace wealth_cat3 = 1 if inlist(v190, 1, 2)   // poorest + poorer -> poor
replace wealth_cat3 = 2 if v190 == 3             // middle
replace wealth_cat3 = 3 if inlist(v190, 4, 5)   // richer + richest -> rich
label define wc3_lbl 1 "Poor" 2 "Middle" 3 "Rich"
label values wealth_cat3 wc3_lbl

gen marital_cat3 = .
replace marital_cat3 = 1 if v501 == 0                // never in union
replace marital_cat3 = 2 if inlist(v501, 1, 2)       // married + living with partner -> currently in union
replace marital_cat3 = 3 if inlist(v501, 3, 4, 5)     // widowed + divorced + separated -> formerly in union
label define mc3_lbl 1 "Never in union" 2 "Currently in union" 3 "Formerly in union"
label values marital_cat3 mc3_lbl


* ================================================================
* SECTION 4: NEW MODEL 2 PREDICTORS - facility delivery, mode of
* delivery, birth size. (ANC adequacy and breastfeeding initiation
* were tested but EXCLUDED from the final model: missingness on these
* two variables was strongly outcome-dependent: ANC adequacy was missing for
* 34 of 82 neonatal deaths (41.5%) and breastfeeding timing for 55 (67.1%),
* versus 7.8% and 2.7% of 4,258 survivors - consistent with a structural
* mechanism (e.g. death occurring before breastfeeding could be established).
* Requiring both would leave only 16 of the 82 deaths (19.5%) in a complete-case
* sample. See stata/03_optional_antenatal_breastfeeding_missingness.do.)
* ================================================================

* VERIFICATION: confirm every observed
* m15 code is correctly classified before trusting this recode - DHS
* allows country-specific m15 codes, so this must be checked against
* the actual Rwanda 2025 file, not assumed from standard DHS coding.
tab m15, missing
tab m15, missing nolabel

gen facility_delivery = .
replace facility_delivery = 1 if inlist(m15, 21, 22, 23, 24, 31, 32)
replace facility_delivery = 0 if inlist(m15, 11, 12)
label define fd_lbl 0 "Home" 1 "Facility"
label values facility_delivery fd_lbl
* CHECK: does every observed m15 code fall into one of the two groups
* above? Any code not covered will be silently set to missing - this
* tab confirms whether that happened and for how many cases.
tab m15 facility_delivery, missing

label define m17_lbl 0 "Vaginal" 1 "Caesarean"
label values m17 m17_lbl

gen birth_size_cat = .
replace birth_size_cat = 1 if inlist(m18, 4, 5)
replace birth_size_cat = 2 if m18 == 3
replace birth_size_cat = 3 if inlist(m18, 1, 2)
label define bs_lbl 1 "Small" 2 "Average" 3 "Large"
label values birth_size_cat bs_lbl


* ================================================================
* SECTION 5: COMPLETE-CASE SAMPLE (11 final predictors)
* The loop checks every variable that enters the final regression (Section 7).
* Exclusions are counted sequentially, in the order shown, so each birth is
* counted once (Supplementary Table S4).
* ================================================================

gen complete_case = 1
foreach v in neonatal_death bord_cat multiple_birth wealth_cat3 v106 b4 ///
    marital_cat3 short_interval matage_cat facility_delivery m17 birth_size_cat {
    replace complete_case = 0 if missing(`v')
}

tab complete_case neonatal_death, row
count if complete_case == 1
count if complete_case == 1 & neonatal_death == 1
* Result: 4,210 births with 72 events. multiple_birth, wealth_cat3 and
* marital_cat3 are derived from b0/v190/v501, which had zero missingness,
* so they add no further exclusions.


* ================================================================
* SURVEY DESIGN - declared here, before any svy: command is used
* ================================================================

gen wt = v005 / 1000000
svyset v021 [pw=wt], strata(v023)


* ================================================================
* SECTION 6: BIVARIATE SCREEN (complete-case sample only)
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
tab facility_delivery neonatal_death, row chi2
tab m17 neonatal_death, row chi2
tab birth_size_cat neonatal_death, row chi2

restore


* ================================================================
* SECTION 6b: TABLE 1 DATA (weighted n and % for baseline characteristics
* table, in n(%) format) AND CRUDE (UNADJUSTED) ODDS RATIOS
* ================================================================

preserve
keep if complete_case == 1

* ---- Weighted N (count) and % (col) together, for n(%) reporting ----
foreach v in bord_cat multiple_birth wealth_cat3 v106 b4 marital_cat3 short_interval ///
    matage_cat facility_delivery m17 birth_size_cat {
    di "=========================================="
    di "VARIABLE: `v'"
    di "=========================================="
    svy: tab `v' neonatal_death, count
    svy: tab `v' neonatal_death, col
}

svy: tab neonatal_death, count
* Use this total (weighted N for Died and Survived) to confirm each
* variable's category counts sum to approximately the same total.

* ---- Crude (unadjusted) odds ratios, one predictor at a time ----
svy: logistic neonatal_death i.bord_cat
svy: logistic neonatal_death i.multiple_birth
svy: logistic neonatal_death i.wealth_cat3
svy: logistic neonatal_death i.v106
svy: logistic neonatal_death i.b4
svy: logistic neonatal_death i.marital_cat3
svy: logistic neonatal_death i.short_interval
svy: logistic neonatal_death i.matage_cat
svy: logistic neonatal_death i.facility_delivery
svy: logistic neonatal_death i.m17
svy: logistic neonatal_death i.birth_size_cat

restore


* ================================================================
* SECTION 7: FINAL MULTIVARIABLE LOGISTIC REGRESSION
* (multiple_birth and marital_cat collapsed to avoid separation)
* wt and svyset already declared earlier in the do-file
* ================================================================

keep if complete_case == 1

svy: logistic neonatal_death i.bord_cat i.multiple_birth i.wealth_cat3 i.v106 i.b4 ///
    i.marital_cat3 i.short_interval i.matage_cat i.facility_delivery i.m17 ///
    i.birth_size_cat

estimates store model2_final


* ================================================================
* SECTION 8: MODEL DISCRIMINATION (AUC)
* CAVEAT: roctab is an ordinary ROC procedure and does NOT incorporate the DHS
* survey design (weights, clustering, stratification) the way svy: logistic does.
* The AUC below is unweighted, non-survey-adjusted discrimination, distinct from
* the survey-adjusted regression coefficients above.
* ================================================================

predict phat2, pr
roctab neonatal_death phat2
* This AUC is NOT survey-weighted - see caveat above.

sum phat2, detail

* ---- Additional classification metrics at a fixed threshold,
* using the outcome's own prevalence as the cutoff (since 0.5 is
* meaningless for an outcome this rare) ----
sum neonatal_death
local cutoff = r(mean)
gen predicted_death = (phat2 >= `cutoff') if !missing(phat2)

tab predicted_death neonatal_death, cell
* From this 2x2 table, compute manually or via -diagt- if installed:
* sensitivity, specificity, PPV, NPV. Note this classification table
* is also unweighted, same caveat as the AUC above - a fully
* survey-weighted version would require weighting the 2x2 cell counts
* directly rather than using unweighted case counts.


* ================================================================
* SECTION 9: EXPORT ANALYTIC DATASET FOR ML (PYTHON)
* ================================================================

preserve

keep neonatal_death bord_cat multiple_birth v190 v106 b4 marital_cat ///
     short_interval matage_cat facility_delivery m17 birth_size_cat ///
     wt v021 v023

misstable summarize
* Expected: no output (zero missingness across all kept variables)

count
tab neonatal_death
* Confirmed Model 2 corrected run: n=4,210, 72 events

save "$datadir/model2_analytic_data_improved.dta", replace

restore

* ================================================================
* END OF MODEL 2
* Main adjusted odds ratios (4,210 births, 72 deaths): multiple birth 6.80,
* maternal age 20-34 0.12 and 35+ 0.12 (vs <20), birth size average 0.43 and
* large 0.37 (vs small). Marital status was not significant at the 5% level.
* AUC = 0.6986 [0.635, 0.763] (apparent, unweighted).
* ================================================================
