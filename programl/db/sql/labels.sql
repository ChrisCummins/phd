-- The type of labels for a graph.

CREATE TYPE LABELS_TYPE AS ENUM(--
 'REACHABILITY', -- Control-flow reachability labels.
 'DOMTREE', -- Dominator tree labels.
 'LIVENESS', -- Variable liveness.
 'SUBEXPRESSIONS', -- Global common subexpressions.
 'OPENCL_DEVMAP', -- OpenCL heterogeneous device mapping.
 'ALGORITHM_CLASS'-- Algorithm classification.
);


CREATE TABLE labels
  (-- A set of labels for a graph.
 labels_id BIGSERIAL PRIMARY KEY, -- A numeric ID for this set of graph labels.
 TYPE LABELS_TYPE NOT NULL, --
 ir_id BIGSERIAL REFERENCES graph(ir_id) ON DELETE CASCADE ON UPDATE CASCADE, -- The graph which this set of labels belongs to.
 graph_y FLOAT [], node_y FLOAT [], -- An array of graph-level or node-level labels. Only one of these two columns should be set.
 date_added TIMESTAMP NOT NULL DEFAULT NOW(), -- The date at which these labels were generated.
 UNIQUE(TYPE, ir_id) -- A set of labels of a given type are unique to an IR.
);
