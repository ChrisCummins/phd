# Copyright 2018, 2019 Chris Cummins <chrisc.101@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""A module to generate random YNAB datasets."""
import pathlib
import random
import time
import typing

from labm8.py import app
from labm8.py import jsonutil

FLAGS = app.FLAGS


class RandomDatasetGenerator(object):
  def __init__(
    self,
    start_date_seconds_since_epoch: float,
    categories: typing.Dict[str, typing.List],
  ):
    self.start_date_seconds_since_epoch = start_date_seconds_since_epoch
    self.categories = categories

  def Sample(self, num_rows: int) -> typing.Dict[str, typing.Any]:
    data = {
      "transactions": [],
      "masterCategories": [],
      "monthlyBudgets": [],
    }

    # Create the categories.
    category_ids = []
    for master_category in self.categories:
      cat = {
        "name": master_category,
        "type": "OUTFLOW",
        "subCategories": [],
      }
      for s in self.categories[master_category]:
        cat_id = "A{}".format(random.randint(1, int(1e9)))
        category_ids.append(cat_id)
        cat["subCategories"].append(
          {"type": "OUTFLOW", "name": s, "entityId": cat_id,}
        )
      data["masterCategories"].append(cat)

    # Create the transactions.
    time_offset_seconds = 0
    for _ in range(num_rows):
      time_offset_seconds += random.randint(0, 2) * 24 * 3600
      date = time.strftime(
        "%Y-%m-%d",
        time.localtime(
          self.start_date_seconds_since_epoch + time_offset_seconds
        ),
      )
      data["transactions"].append(
        {
          "importedPayee": "unused",
          "date": date,
          "YNABID": "unused",
          "FITID": "unused",
          "source": "unused",
          "entityId": "unused",
          "entityType": "transaction",
          "categoryId": random.choice(category_ids),
          "entityVersion": "unused",
          "amount": round(random.randint(-10000, 10000) / 100, 2),
          "accountId": "unused",
          "payeeId": "unused",
          "memo": "unused",
          "cleared": "Cleared",
          "accepted": True if random.random() > 0.5 else False,
        }
      )

    time_offset_seconds = 0
    for _ in range(12):
      time_offset_seconds += random.randint(0, 1) * 24 * 3600 * 31
      date = time.strftime(
        "%Y-%m-%d",
        time.localtime(
          self.start_date_seconds_since_epoch + time_offset_seconds
        ),
      )
      month = {"month": date, "monthlySubCategoryBudgets": []}
      for category in category_ids:
        month["monthlySubCategoryBudgets"].append(
          {
            "budgeted": random.randint(0, 10000),
            "categoryId": category,
            "overspendingHandling": None,
            "isTombstone": True,
            "entityVersion": "unused",
            "entityId": "unused",
            "parentMonthlyBudgetId": "unused",
            "entityType": "unused",
          }
        )
      data["monthlyBudgets"].append(month)

    return data

  def SampleFile(self, path: pathlib.Path, *sample_args) -> pathlib.Path:
    with open(path, "w") as f:
      f.write(jsonutil.format_json(self.Sample(*sample_args)))
    return path


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Unrecognized command line flags.")

  generator = RandomDatasetGenerator(
    start_date_seconds_since_epoch=time.mktime(
      time.strptime("1/1/2018", "%m/%d/%Y")
    ),
    categories={
      "Rainy Day": ["Savings", "Pension"],
      "Everyday Expenses": ["Groceries", "Clothes"],
    },
  )
  print(jsonutil.format_json(generator.Sample(30)))


if __name__ == "__main__":
  app.RunWithArgs(main)
