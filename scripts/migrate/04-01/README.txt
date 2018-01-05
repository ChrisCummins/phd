27 July 2017 - Database refactor.

"Old" database structure was grinding to a halt as the dataset size increased.
Chokepoint was a single >100GB ClgenResults table. Restructuring to a more
normalized layout, and reducing data duplication.
