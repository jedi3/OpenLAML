package ml.graph;

import ml.utils.*;

public class Edge<K extends Comparable<K>>
{
    Pair<Vertex<K>, Vertex<K>> edge;
    K weight;
    
    public Edge(final Pair<Vertex<K>, Vertex<K>> edge, final K weight) {
        this.edge = edge;
        this.weight = weight;
    }
    
    public Edge(final Vertex<K> u, final Vertex<K> v, final K weight) {
        this.edge = Pair.of(u, v);
        this.weight = weight;
    }
}
