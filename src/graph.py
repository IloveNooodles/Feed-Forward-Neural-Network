import graphviz


class Graph:
    def __init__(self) -> None:
        pass

    def add_node(self):
        pass

    def add_edge(self):
        pass


if __name__ == "__main__":
    ps = graphviz.Digraph('petshop', node_attr={
                          'shape': 'plaintext'}, format='png')
    ps.node('x1')
    ps.node('x2')
    ps.node('x3')
    ps.node('h1')
    ps.edge('x1', 'h1', label='20')
    ps.edge('x2', 'h1', label='10')
    ps.edge('x3', 'h1', label='30')
    print(ps.render(directory="/img", view=True))
