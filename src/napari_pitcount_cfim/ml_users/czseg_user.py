# python
import tensorflow as tf
from pathlib import Path

base_dir = Path(__file__).resolve().parent
input_path = base_dir / "net_out" / "ac152c5c-b616-431c-ae2a-ea0ec307ce76.model"

with open(input_path, 'rb') as f:
    binary_content = f.read()

graph_def = tf.compat.v1.GraphDef()
try:
    graph_def.ParseFromString(binary_content)
    print("GraphDef parsed successfully. Node names:")
    for node in graph_def.node:
        print(node.name)
except Exception as e:
    print("Error parsing GraphDef:", e)