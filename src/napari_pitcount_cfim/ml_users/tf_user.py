from pathlib import Path
import tensorflow as tf
base_dir = Path(__file__).resolve().parent
input_path = base_dir / "net_out" / "ac152c5c-b616-431c-ae2a-ea0ec307ce76.model"

print(f"Path: {input_path}")
# 1) Read and parse the GraphDef
graph_def = tf.compat.v1.GraphDef()
with open(input_path, 'rb') as f:
    graph_def.ParseFromString(f.read())

# 2) Import into a fresh Graph
graph = tf.compat.v1.Graph()
with graph.as_default():
    tf.compat.v1.import_graph_def(graph_def, name="")  # no name-prefix

# 3) (Optional) Inspect ops to find your input/output tensor names
for op in graph.get_operations():
    print(op.name)