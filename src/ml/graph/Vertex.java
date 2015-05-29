package ml.graph;

import ml.utils.*;
import java.util.*;

public class Vertex<K extends Comparable<K>> implements Comparable<Vertex<K>>, Updater<K>
{
    String name;
    K key;
    Vertex<K> parent;
    LinkedList<Pair<Vertex<K>, K>> adjcencyList;
    HashMap<Vertex<K>, K> adjcencyMap;
    Class<K> typeClass;
    
    public Vertex(final K key, final String name) {
        this.name = name;
        this.key = key;
        this.typeClass = (Class<K>)key.getClass();
        this.parent = null;
        this.adjcencyList = new LinkedList<Pair<Vertex<K>, K>>();
        this.adjcencyMap = new HashMap<Vertex<K>, K>();
    }
    
    public Vertex(final K key) {
        this(key, "");
    }
    
    public Vertex(final Vertex<K> v) {
        this(v.key, v.name);
    }
    
    public Vertex(final Class<K> typeClass, final String name) {
        this.typeClass = typeClass;
        if (typeClass == null) {
            this.key = null;
        }
        else if (typeClass.equals(Double.class)) {
            this.key = (K)new Double(Double.POSITIVE_INFINITY);
        }
        else if (typeClass.equals(Integer.class)) {
            this.key = (K)new Integer(Integer.MAX_VALUE);
        }
        else if (typeClass.equals(Float.class)) {
            this.key = (K)new Float(Float.POSITIVE_INFINITY);
        }
        else {
            this.key = null;
        }
        this.name = name;
        this.parent = null;
        this.adjcencyList = new LinkedList<Pair<Vertex<K>, K>>();
        this.adjcencyMap = new HashMap<Vertex<K>, K>();
    }
    
    public Vertex(final Class<K> typeClass) {
        this(typeClass, "");
    }
    
    public Vertex() {
        this((Class)null);
    }
    
    public Vertex(final String name) {
        this((Class)null, name);
    }
    
    public void addToAdjList(final Vertex<K> v, final K w) {
        if (this.typeClass == null) {
            this.typeClass = (Class<K>)w.getClass();
        }
        this.adjcencyList.add(Pair.of(v, w));
        this.adjcencyMap.put(v, w);
    }
    
    @Override
    public String toString() {
        if (this.name.isEmpty()) {
            return super.toString();
        }
        return String.valueOf(this.name) + ':' + this.key;
    }
    
    @Override
    public int compareTo(final Vertex<K> o) {
        int cmp = 0;
        if (o == null) {
            cmp = 1;
        }
        else if (this.key == null) {
            cmp = ((o.key == null) ? 0 : -1);
        }
        else if (o.key == null) {
            cmp = ((this.key != null) ? 1 : 0);
        }
        else {
            cmp = this.key.compareTo(o.key);
        }
        return cmp;
    }
    
    @Override
    public void update(final K key) {
        this.key = key;
    }
}
