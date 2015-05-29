package ml.graph;

import ml.utils.*;
import java.util.*;

public class Graph<K extends Comparable<K>>
{
    LinkedList<Vertex<K>> vertices;
    
    public Graph() {
        this.vertices = new LinkedList<Vertex<K>>();
    }
    
    public void addVertex(final Vertex<K> v) {
        this.vertices.add(v);
    }
    
    public static void main(final String[] args) {
        final Vertex<Double> A = new Vertex<Double>(Double.valueOf(3.5));
        final Vertex<Double> B = new Vertex<Double>(Double.valueOf(2.5));
        final Vertex<Double> C = new Vertex<Double>(Double.valueOf(5.5));
        final Vertex<Double> D = new Vertex<Double>(Double.valueOf(1.5));
        final MinPQueue<Vertex<Double>, Double> minQueue = new MinPQueue<Vertex<Double>, Double>();
        minQueue.insert(A);
        minQueue.insert(B);
        minQueue.insert(C);
        minQueue.insert(D);
        System.out.printf("Min key: %f\n", minQueue.delMin().key);
        minQueue.update(A, 12.0);
        System.out.printf("Min key: %f\n", minQueue.delMin().key);
        System.out.printf("Min key: %f\n", minQueue.delMin().key);
        System.out.printf("Min key: %f\n", minQueue.delMin().key);
        D.key = 6.5;
        final Graph<Double> G = new Graph<Double>();
        final Vertex<Double> a = new Vertex<Double>("a");
        final Vertex<Double> b = new Vertex<Double>(Double.class, "b");
        final Vertex<Double> c = new Vertex<Double>(Double.class, "c");
        final Vertex<Double> d = new Vertex<Double>(Double.class, "d");
        final Vertex<Double> e = new Vertex<Double>(Double.class, "e");
        final Vertex<Double> f = new Vertex<Double>(Double.class, "f");
        final Vertex<Double> g = new Vertex<Double>(Double.class, "g");
        final Vertex<Double> h = new Vertex<Double>(Double.class, "h");
        final Vertex<Double> i = new Vertex<Double>(Double.class, "i");
        a.addToAdjList(b, 4.0);
        a.addToAdjList(h, 8.0);
        b.addToAdjList(a, 4.0);
        b.addToAdjList(c, 8.0);
        b.addToAdjList(h, 11.0);
        c.addToAdjList(b, 8.0);
        c.addToAdjList(d, 7.0);
        c.addToAdjList(f, 4.0);
        c.addToAdjList(i, 2.0);
        d.addToAdjList(c, 7.0);
        d.addToAdjList(e, 9.0);
        d.addToAdjList(f, 14.0);
        e.addToAdjList(d, 9.0);
        e.addToAdjList(f, 10.0);
        f.addToAdjList(c, 4.0);
        f.addToAdjList(d, 14.0);
        f.addToAdjList(e, 10.0);
        f.addToAdjList(g, 2.0);
        g.addToAdjList(f, 2.0);
        g.addToAdjList(h, 1.0);
        g.addToAdjList(i, 6.0);
        h.addToAdjList(a, 8.0);
        h.addToAdjList(b, 11.0);
        h.addToAdjList(g, 1.0);
        h.addToAdjList(i, 7.0);
        i.addToAdjList(c, 2.0);
        i.addToAdjList(g, 6.0);
        i.addToAdjList(h, 7.0);
        G.addVertex(a);
        G.addVertex(b);
        G.addVertex(c);
        G.addVertex(d);
        G.addVertex(e);
        G.addVertex(f);
        G.addVertex(g);
        G.addVertex(h);
        G.addVertex(i);
        for (final Vertex<Double> v : G.vertices) {
            System.out.println(v);
        }
        final LinkedList<Edge<Double>> MST = minimumSpanningTreePrim(G);
        for (final Edge<Double> edge : MST) {
            System.out.print(edge.edge);
            System.out.printf(": %f\n", edge.weight);
        }
        final Vertex<Double> s = a;
        final LinkedList<Vertex<Double>> S = shortestPathDijkstra(G, s);
        for (final Vertex<Double> v2 : S) {
            if (v2 == s) {
                continue;
            }
            System.out.printf("delta(%s, %s): %s\tw(%s->%s): %s \n", s, v2, v2.key, v2.parent, v2, v2.parent.adjcencyMap.get(v2));
        }
    }
    
    public static <K extends Comparable<K>> LinkedList<Edge<K>> minimumSpanningTreePrim(final Graph<K> G) {
        final Vertex<K> r = G.vertices.element();
        for (final Vertex<K> v : G.vertices) {
            if (v.typeClass == null) {
                v.key = null;
            }
            else if (v.typeClass.equals(Double.class)) {
                v.key = (K)new Double(Double.POSITIVE_INFINITY);
            }
            else if (v.typeClass.equals(Integer.class)) {
                v.key = (K)new Integer(Integer.MAX_VALUE);
            }
            else if (v.typeClass.equals(Float.class)) {
                v.key = (K)new Float(Float.POSITIVE_INFINITY);
            }
            v.parent = null;
        }
        if (r.key instanceof Double) {
            r.key = (K)new Double(0.0);
        }
        else if (r.key instanceof Integer) {
            r.key = (K)new Integer(0);
        }
        else if (r.key instanceof Float) {
            r.key = (K)new Float(0.0f);
        }
        else {
            r.key = null;
        }
        final MinPQueue<Vertex<K>, K> Q = new MinPQueue<Vertex<K>, K>();
        for (final Vertex<K> v2 : G.vertices) {
            Q.insert(v2);
        }
        while (!Q.isEmpty()) {
            final Vertex<K> u = Q.delMin();
            for (final Pair<Vertex<K>, K> edge : u.adjcencyList) {
                final Vertex<K> v3 = edge.first;
                final K w = edge.second;
                if (Q.containsKey(v3) && w.compareTo(v3.key) < 0) {
                    v3.parent = u;
                    Q.update(v3, w);
                }
            }
        }
        final LinkedList<Edge<K>> MST = new LinkedList<Edge<K>>();
        for (final Vertex<K> v4 : G.vertices) {
            if (v4 != r) {
                MST.add(new Edge<K>(v4.parent, v4, v4.key));
            }
        }
        return MST;
    }
    
    public static <K extends Comparable<K>> LinkedList<Vertex<K>> shortestPathDijkstra(final Graph<K> G, final Vertex<K> s) {
        for (final Vertex<K> v : G.vertices) {
            if (v.typeClass == null) {
                v.key = null;
            }
            else if (v.typeClass.equals(Double.class)) {
                v.key = (K)new Double(Double.POSITIVE_INFINITY);
            }
            else if (v.typeClass.equals(Integer.class)) {
                v.key = (K)new Integer(Integer.MAX_VALUE);
            }
            else if (v.typeClass.equals(Float.class)) {
                v.key = (K)new Float(Float.POSITIVE_INFINITY);
            }
            v.parent = null;
        }
        if (s.key instanceof Double) {
            s.key = (K)new Double(0.0);
        }
        else if (s.key instanceof Integer) {
            s.key = (K)new Integer(0);
        }
        else if (s.key instanceof Float) {
            s.key = (K)new Float(0.0f);
        }
        else {
            s.key = null;
        }
        final MinPQueue<Vertex<K>, K> Q = new MinPQueue<Vertex<K>, K>();
        for (final Vertex<K> v2 : G.vertices) {
            Q.insert(v2);
        }
        final LinkedList<Vertex<K>> S = new LinkedList<Vertex<K>>();
        while (!Q.isEmpty()) {
            final Vertex<K> u = Q.delMin();
            S.add(u);
            for (final Map.Entry<Vertex<K>, K> edge : u.adjcencyMap.entrySet()) {
                final Vertex<K> v3 = edge.getKey();
                final K w = edge.getValue();
                final K temp = add(u.key, w);
                if (temp.compareTo(v3.key) < 0) {
                    v3.parent = u;
                    Q.update(v3, temp);
                }
            }
        }
        return S;
    }
    
    public static <K> K add(final K k1, final K k2) {
        K res = null;
        if (k1 instanceof Double) {
            res = (K)new Double((double)k1 + (double)k2);
        }
        else if (k1 instanceof Float) {
            res = (K)new Float((float)k1 + (float)k2);
        }
        else if (k1 instanceof Integer) {
            res = (K)new Integer((int)k1 + (int)k2);
        }
        else {
            res = k2;
        }
        return res;
    }
}
