# You Need A Budget

[You Need A Budget](https://www.youneedabudget.com/) (YNAB) is a macOS app for
budgeting and tracking transactions from bank accounts. *(Actually YNAB is now
moving towards a SAAS model with a rolling subscription and cloud-based apps,
but I haven't made the upgrade yet. I use YNAB 4).

Aggregating data from YNAB can be achieved automatically by reading the app's
local data files. From what I can tell, the process is pretty straightforward -
the data we're interested in stored in per-budget JSON files with a simple
schema. The three entries we're interested in are:

```
{
  'transactions': [ ... ],
  'masterCategories': [ ... ],
  'monthlyBudgets': [ ... ],
}
```
