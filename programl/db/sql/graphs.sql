-- A ProGraML graph.
--
-- This is a table of ProgramGraph protocol buffers.

CREATE TABLE graph
  (-- The IR used to generate graph.
 ir_id BIGSERIAL PRIMARY KEY REFERENCES ir(ir_id) ON DELETE CASCADE ON UPDATE CASCADE, -- The data of a ProgramGraph protocol buffer, as JSON.
 DATA JSONB, -- The state of the repository used to generate this graph.
 repo_state BIGSERIAL REFERENCES repo(repo_id), -- The date that the graph was generated.
 date_added TIMESTAMP NOT NULL DEFAULT NOW());
