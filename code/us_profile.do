* Churn Ken Lee

version 16
clear all
capture log close

global homepath "C:\Users\churn\Documents\UCSD\2020_winter\econ281\blinder_weiss_1976"
global datapath "$homepath/data"
global outputpath "$homepath/output"
global codepath "$homepath/code"

cd $codepath


use "$datapath/us_profile.dta"

* Keep men only
keep if sex == 1

* Keep older men only
keep if inrange(age, 16, 64)

* Keep only men who were full-time workers
keep if wkswork2 == 6
keep if hrswork2 >= 4 | uhrswork >= 35

* Cohort birth data
gen birth_year = year - age

* Education dummies
gen educ_level = 0
gen yrs_sch = 0
replace yrs_sch = educ + 6 if educ >= 3
lab var yrs_sch "Years of schooling"

gen hs_dropout = 0 
replace hs_dropout = 1 if inrange(yrs_sch, 9, 11)
replace educ_level = 1 if inrange(yrs_sch, 9, 11)

gen hs_grad = 0 
replace hs_grad = 1 if yrs_sch == 12
replace educ_level = 2 if yrs_sch == 12

gen col_some = 0
replace col_some = 1 if inrange(yrs_sch, 13, 15)
replace educ_level = 3 if inrange(yrs_sch, 13, 15)

gen col_grad = 0
replace col_grad = 1 if yrs_sch == 16
replace educ_level = 4 if yrs_sch == 16

gen post_grad = 0
replace post_grad = 1 if yrs_sch > 16
replace educ_level = 5 if yrs_sch > 16

* Drop missing education levels
drop if educ_level == 0

* Years of experience
gen yrs_exp = age - yrs_sch - 6
lab var yrs_exp "Potential experience"

* Year of entry
gen yr_entry = year - yrs_exp
/*
* Generate employment trends
keep if inrange(age, 25, 55)
drop if empstat == 0
generate pop = perwt if inrange(empstat, 1, 3)
generate nonparticipating = perwt if empstat == 3

gcollapse (sum) nonparticipant_num = nonparticipating (sum) educ_level_num = pop , by(year educ_level)

generate nonparticipant_share = nonparticipant_num/educ_level_num

graph twoway (line nonparticipant_share year if educ_level == 1)


restore
*/

*********************************
* Generate data for  cross-sectional experience-wage profile
*********************************
gen lincwage = log(incwage)

* Collapse by census-year and years of experience
gen yr_group = year
replace yr_group = 2010 if inrange(year, 2001, 2010)
replace yr_group = 2019 if inrange(year, 2011, 2019) 

preserve
gcollapse (mean) mean_lincwage = lincwage [weight = perwt], by(yr_group yrs_exp educ_level)
save "$datapath/us_profile_cross.dta", replace
restore


*********************************
* Generate data for cohort experience-wage profile
*********************************
* Generate cohort identifiers
* Keep decadal years, and 2019
preserve
generate decadal_remainder = mod(yr_entry, 10)
keep if decadal_remainder == 0 & yr_entry >= 1960

gcollapse (mean) mean_lincwage = lincwage [weight = perwt], by(yr_entry yrs_exp educ_level)
save "$datapath/us_profile_cohort.dta", replace


*********************************
* Create figures
*********************************
* Cross-section
use "$datapath/us_profile_cross.dta", clear

keep if inrange(yrs_exp, 0, 30)

bysort yr_group educ_level: gegen mean_lincwage_0 = min(mean_lincwage)
generate normalized_mean_lincwage = mean_lincwage - mean_lincwage_0
lab var normalized_mean_lincwage "Normalized log income"

glevelsof educ_level, local(educ_level_list)

sort yr_group educ_level yrs_exp
/*
graph twoway (line normalized_mean_lincwage yrs_exp if yr_group == 1960, lcolor(gs0)) ///
(line normalized_mean_lincwage yrs_exp if yr_group == 1970, lcolor(gs2)) ///
(line normalized_mean_lincwage yrs_exp if yr_group == 1980, lcolor(gs4)) ///
(line normalized_mean_lincwage yrs_exp if yr_group == 1990, lcolor(gs6)) ///
(line normalized_mean_lincwage yrs_exp if yr_group == 2000, lcolor(gs8)) ///
(line normalized_mean_lincwage yrs_exp if yr_group == 2010, lcolor(gs10)) ///
(line normalized_mean_lincwage yrs_exp if yr_group == 2019, lcolor(gs12) legend(label(1 "1960") label(2 "1970") label(3 "1980") label(4 "1990") label(5 "2000") label(6 "2010") label(7 "2019") pos(3))), ///
 by(educ_level)
*/

foreach i of local educ_level_list {
    graph twoway (line normalized_mean_lincwage yrs_exp if yr_group == 1960 & educ_level == `i', lcolor(gs0) legend(label(1 "1960") label(2 "1970") label(3 "1980") label(4 "1990") label(5 "2000") label(6 "2010") label(7 "2019") bmargin(large))) ///
    (line normalized_mean_lincwage yrs_exp if yr_group == 1970 & educ_level == `i', lcolor(gs2)) ///
    (line normalized_mean_lincwage yrs_exp if yr_group == 1980 & educ_level == `i', lcolor(gs4)) ///
    (line normalized_mean_lincwage yrs_exp if yr_group == 1990 & educ_level == `i', lcolor(gs6)) ///
    (line normalized_mean_lincwage yrs_exp if yr_group == 2000 & educ_level == `i', lcolor(gs8)) ///
    (line normalized_mean_lincwage yrs_exp if yr_group == 2010 & educ_level == `i', lcolor(gs10)) ///
    (line normalized_mean_lincwage yrs_exp if yr_group == 2019 & educ_level == `i', lcolor(gs12)), ///
    saving(educ`i', replace)

    graph export "$outputpath/exp_wage_profile_cross_educ`i'.pdf", replace
}

grc1leg educ1.gph educ2.gph educ3.gph educ4.gph educ5.gph, position(4) ring(0)
graph export "$outputpath/exp_wage_profile_cross.pdf", replace


* Cohort
use "$datapath/us_profile_cohort.dta", clear

keep if inrange(yrs_exp, 0, 30)

bysort yr_entry educ_level: gegen mean_lincwage_0 = min(mean_lincwage)
generate normalized_mean_lincwage = mean_lincwage - mean_lincwage_0
lab var normalized_mean_lincwage "Normalized log income"

glevelsof educ_level, local(educ_level_list)

sort yr_entry educ_level yrs_exp

foreach i of local educ_level_list {
    graph twoway (line normalized_mean_lincwage yrs_exp if yr_entry == 1960 & educ_level == `i', lcolor(gs0) legend(label(1 "1960") label(2 "1970") label(3 "1980") label(4 "1990") label(5 "2000") label(6 "2010") bmargin(large))) ///
    (line normalized_mean_lincwage yrs_exp if yr_entry == 1970 & educ_level == `i', lcolor(gs2)) ///
    (line normalized_mean_lincwage yrs_exp if yr_entry == 1980 & educ_level == `i', lcolor(gs4)) ///
    (line normalized_mean_lincwage yrs_exp if yr_entry == 1990 & educ_level == `i', lcolor(gs6)) ///
    (line normalized_mean_lincwage yrs_exp if yr_entry == 2000 & educ_level == `i', lcolor(gs8)) ///
    (line normalized_mean_lincwage yrs_exp if yr_entry == 2010 & educ_level == `i', lcolor(gs10)), ///
    legend(label(1 "1960") label(2 "1970") label(3 "1980") label(4 "1990") label(5 "2000") label(6 "2010")) ///
    saving(educ`i', replace)

    graph export "$outputpath/exp_wage_profile_cohort_educ`i'.pdf", replace
}

grc1leg educ1.gph educ2.gph educ3.gph educ4.gph educ5.gph, position(4) ring(0)
graph export "$outputpath/exp_wage_profile_cohort.pdf", replace



