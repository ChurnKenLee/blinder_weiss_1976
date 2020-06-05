* Churn Ken Lee

version 16
clear all
capture log close

global homepath "C:\Users\churn\Documents\UCSD\2020_winter\econ281\blinder_weiss_1976"
global datapath "$homepath/data"
global codepath "$homepath/code"

cd $codepath

use "$datapath/emp_hours.dta"

generate age_90 = age - (year - 1990)
generate age_


