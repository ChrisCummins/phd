
class Split(Base, sqlutil.PluralTablenameFromCamelCapsClassNameMixin):
  id: int = sql.Column(sql.Integer, nullable, primary_key=True)

  # A string name to split the graphs in a database into discrete buckets, e.g.
  # "train", "val", "test"; or "1", "2", ... k for k-fold classification.
  split: str = sql.Column(sql.String(8), nullable=False, index=True)
