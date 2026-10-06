* ----------------------------------------------------------------------
* 03_optional_antenatal_breastfeeding_missingness.do   (OPTIONAL SNIPPET - not a stand-alone script)
* Reports how often antenatal care adequacy and early breastfeeding initiation are
* missing among neonatal deaths versus survivors (the reason these two variables were
* excluded; reported in the Methods and Results).
*
* HOW TO USE: paste the commands below into stata/02_model2_regression_and_export.do,
* after the predictors have been built (end of Section 4) and before the complete-case
* loop (Section 5), then run that do-file. It needs neonatal_death, m14 and m34.
* ----------------------------------------------------------------------

* Run inside neonatal_mortality_model2_clean.do, after Section 4 (variables built)
* and before the complete_case loop. Uses the corrected 3-year window (birth_recency 1-36).

capture drop anc_adequate bf_early miss_anc miss_bf miss_either

gen anc_adequate = .
replace anc_adequate = 0 if m14 < 4
replace anc_adequate = 1 if m14 >= 4 & m14 <= 20
gen bf_early = .
replace bf_early = 1 if inlist(m34, 0, 100, 101)
replace bf_early = 0 if m34 > 101 & m34 <= 223

gen miss_anc    = missing(anc_adequate)
gen miss_bf     = missing(bf_early)
gen miss_either = miss_anc | miss_bf

* sanity check: should show 4,345 births in the corrected window (5 with unknown outcome)
tab neonatal_death, missing

tab miss_anc    neonatal_death, col
tab miss_bf     neonatal_death, col
tab miss_either neonatal_death, col

count if neonatal_death == 1                    // deaths in the corrected window
count if neonatal_death == 1 & miss_either == 0 // deaths kept if BOTH variables were required
