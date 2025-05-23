import numpy as np


class Node:
    # Each token _is_ a node

    def __init__(self):
        # Vector of this node
        # This data would be the embedding of the word
        # "Cat" -> Some vector of len 512 in reality
        self.data = np.random.randn(20)

        # These are projection matrices that create different "views" of the node (token).
        # They project the node's internal data into the key, query, and value spaces.
        # In real Transformers, these matrices are learned and shared across all tokens.
        # Each node uses the same key/query/value matrices to produce its key, query, and value vectors.
        # They are matrices (20,20) to enable the views that are made up of linear combinations of the data
        self._key = np.random.randn(20, 20)
        self._query = np.random.randn(20, 20)
        self._value = np.random.randn(20, 20)

    def key(self):
        # How nodes "see" what I represent
        # The key is an abstraction of the type of data that is help by the node
        # It might encode things like: "Animal", "Noun", "Person" etc
        # It's used to detemine how much "ATTENTION" other nodes need to pay to it
        # This is achieved using the dot product
        # For example, A Person may well be Walking so if person was a query and walking was a key,
        # these two might create strong attention
        # On the other hand, a Person is unlikely to be associated with Green, so the
        # dot product of the person query with a green key would be low = low attention
        return self._key @ self.data

    def query(self):
        # "What am I looking for"
        # Used to define the kinds of things that might be of interest to this node
        # For example, a Person node might be interested in:
        # Other people, activites, food, actions
        return self._query @ self.data

    def value(self):
        # What exactly i am if picked. For example, if my key indicates i am a "Fruit"
        # My value might encode in some way that I am an apple
        return self._value @ self.data


class Graph:
    def __init__(self, num_nodes=10, num_edges=40):
        # make 10 nodes
        self.nodes = [Node() for _ in range(num_nodes)]
        # make 40 edges that randomly like the nodes (from, to)
        randi = lambda: np.random.randint(len(self.nodes))
        self.edges = [[randi(), randi()] for _ in range(num_edges)]

    def run(self):
        updates = []
        for i, n in enumerate(self.nodes):
            # What is the node looking for?
            q = n.query()

            # find all the nodes that input to this nodes, based on the edges
            # e_from = edge start
            # e_to = end end
            # a(e_from) -> (e_to)b
            inputs = [self.nodes[e_from] for e_from, e_to in self.edges if e_to == i]

            if len(inputs) == 0:
                # No nodes connect to this node
                continue

            # Get the keys of the input nodes
            keys = [i.key() for i in inputs]

            # ATTENTION STEP: How much ATTENTION should I be paying to this nodes value
            scores = [k.dot(q) for k in keys]
            # Softmax the scores [0, 1]
            scores = np.exp(scores)
            scores = scores / np.sum(scores)

            # Get the values and weight them based on the attention
            values = [i.value() for i in inputs]
            update = sum([s * v] for s, v in zip(scores, values))
            updates.append(update)

        # Update each node's internal data
        for n, u in zip(self.nodes, updates):
            n.data = n.data + u  # residual connection
