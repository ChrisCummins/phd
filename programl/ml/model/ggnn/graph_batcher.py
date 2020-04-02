# TODO: Inherit from base class
class GgnnBatchBuilder(object):
  def MakeBatch(
    self,
    epoch_type: epoch.EpochType,
    graphs: Iterable[graph_tuple_database.GraphTuple],
    ctx: progress.ProgressContext = progress.NullContext,
  ) -> BatchData:
    """Create a mini-batch of data from an iterator of graphs.

    Returns:
      A single batch of data for feeding into RunBatch(). A batch consists of a
      list of graph IDs and a model-defined blob of data. If the list of graph
      IDs is empty, the batch is discarded and not fed into RunBatch().
    """
    # TODO(github.com/ChrisCummins/ProGraML/issues/24): The new graph batcher
    # implementation is not well suited for reading the graph IDs, hence this
    # somewhat clumsy iterator wrapper. A neater approach would be to create
    # a graph batcher which returns a list of graphs in the batch.
    class GraphIterator(object):
      """A wrapper around a graph iterator which records graph IDs."""

      def __init__(self, graphs: Iterable[graph_tuple_database.GraphTuple]):
        self.input_graphs = graphs
        self.graphs_read: List[graph_tuple_database.GraphTuple] = []

      def __iter__(self):
        return self

      def __next__(self):
        graph: graph_tuple_database.GraphTuple = next(self.input_graphs)
        self.graphs_read.append(graph)
        return graph.tuple

    graph_iterator = GraphIterator(graphs)

    # Create a disjoint graph out of one or more input graphs.
    batcher = graph_batcher.GraphBatcher.CreateFromFlags(
      graph_iterator, ctx=ctx
    )

    try:
      disjoint_graph = next(batcher)
    except StopIteration:
      # We have run out of graphs.
      return BatchData.CreateEndOfBatches()

    # Workaround for the fact that graph batcher may read one more graph than
    # actually gets included in the batch.
    if batcher.last_graph:
      graphs = graph_iterator.graphs_read[:-1]
    else:
      graphs = graph_iterator.graphs_read

    # Discard single-graph batches during training when there are graph
    # features. This is because we use batch normalization on incoming features,
    # and batch normalization requires > 1 items to normalize.
    if (
      len(graphs) <= 1
      and epoch_type == epoch.EpochType.TRAIN
      and disjoint_graph.graph_x_dimensionality
    ):
      return BatchData.CreateEmptyBatch()

    return BatchData(
      graph_ids=[graph.id for graph in graphs],
      model_data=GgnnBatchData(disjoint_graph=disjoint_graph, graphs=graphs),
    )

  def GraphReader(
    self,
    epoch_type: epoch.EpochType,
    graph_db: graph_tuple_database.Database,
    filters: Optional[List[Callable[[], bool]]] = None,
    limit: Optional[int] = None,
    ctx: progress.ProgressContext = progress.NullContext,
  ) -> graph_database_reader.BufferedGraphReader:
    """Construct a buffered graph reader.

    Args:
      epoch_type: The type of graph reader to return a graph reader for.
      graph_db: The graph database to read graphs from.
      filters: A list of filters to impose on the graph database reader.
      limit: The maximum number of rows to read.
      ctx: A logging context.

    Returns:
      A buffered graph reader instance.
    """
    filters = filters or []

    # Only read graphs with data_flow_steps <= message_passing_step_count if
    # --limit_max_data_flow_steps is set.
    if FLAGS.limit_max_data_flow_steps and self.graph_db.has_data_flow:
      filters.append(
        lambda: graph_tuple_database.GraphTuple.data_flow_steps
        <= self.message_passing_step_count
      )

    # If we are batching my maximum node count and skipping graphs that are
    # larger than this, we can apply that filter to the SQL query now, rather
    # than reading the graphs and ignoring them later. This ensures that when
    # --max_{train,val}_per_epoch is set, the number of graphs that get used
    # matches the limit.
    if (
      FLAGS.graph_batch_node_count
      and FLAGS.max_node_count_limit_handler == "skip"
    ):
      filters.append(
        lambda: (
          graph_tuple_database.GraphTuple.node_count
          <= FLAGS.graph_batch_node_count
        )
      )

    return super(Ggnn, self).GraphReader(
      epoch_type=epoch_type,
      graph_db=graph_db,
      filters=filters,
      limit=limit,
      ctx=ctx,
    )
