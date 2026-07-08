from graphviz import Digraph

def trace(root):
    # walk backwards from root and collect every node and edge in the graph
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})  # LR = left to right
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # rectangle showing the node's data and its gradient
        dot.node(name=uid, label="{%s | data %.4f | grad %.4f}" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            # small oval for the operation that produced this node
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)
    for n1, n2 in edges:
        # connect each input to the op that consumed it
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    return dot

if __name__ == "__main__":
    from engine import Value
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = a * b; c.label = 'c'
    d = (c + 1.0).tanh(); d.label = 'd'
    d.backward()
    draw_dot(d).render('graph', view=False)
    print("saved graph.svg")
