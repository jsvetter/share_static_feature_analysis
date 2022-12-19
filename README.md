# Steps necessary to run:

### Change these values in the numbers file.
- OP Indication time Pat 5962483 and 5398110: remove date, only time should be there
- Reason Pat 4333506: "3/2a" -> "2a"
- OP Indication date Pat 4743593: 20:14 -> 4/5/2020
- Nicotine to Hypertonia Pat 5440529; move value from Nicotine to Hypertonia

Not absolutely necessary:
- Removed empty rows.
- All rows in one line.

**Then export .numbers file as .csv!**

Change ```folder_path``` and ```csv_name``` to your values.
Run the script.

# Example output from single run:

```
pHa classification
Imbalance: 0.7347480106100795
Average Test AuROC: 0.5760540788119546
Std Test AuROC: 0.03450453291426004
Average Train AuROC: 0.6398369531727398
Std Train AuROC: 0.009653761192807742

delivery classification
Imbalance: 0.42793987621573826
Average Test AuROC: 0.7875815512032243
Std Test AuROC: 0.026414648053785364
Average Train AuROC: 0.805242593361802
Std Train AuROC: 0.007400258517516268

pHa regression
Linear regression coefficients
[[ 0.00689219  0.03323316 -0.00124495 -0.08155025  0.05363434 -0.22169501
   0.11303833  0.11993614  0.07292329 -0.08358952  0.06401649 -0.03730806
   0.0300173   0.05760403  0.11783761  0.05339413  0.06547524  0.01951503
  -0.03343838 -0.03723022  0.01974598  0.09098876  0.08918553  0.01141139]]
 ```


