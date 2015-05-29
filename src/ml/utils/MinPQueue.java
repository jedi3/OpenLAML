package ml.utils;

import java.util.*;

public class MinPQueue<Key extends Comparable, K> implements Iterable<Key>
{
    private Comparable[] pq;
    private int N;
    private HashMap<Key, Integer> keyToIndexMap;
    
    public MinPQueue(final int initCapacity) {
        this.keyToIndexMap = new HashMap<Key, Integer>();
        this.pq = new java.lang.Comparable[initCapacity + 1];
        this.N = 0;
    }
    
    public MinPQueue() {
        this(1);
    }
    
    public MinPQueue(final Key[] keys) {
        this.keyToIndexMap = new HashMap<Key, Integer>();
        this.N = keys.length;
        this.pq = new java.lang.Comparable[keys.length + 1];
        for (int i = 0; i < this.N; ++i) {
            this.pq[i + 1] = (java.lang.Comparable)keys[i];
            this.keyToIndexMap.put((Key)this.pq[i + 1], i + 1);
        }
        for (int k = this.N / 2; k >= 1; --k) {
            this.sink(k);
        }
        assert this.isMinHeap();
    }
    
    public boolean isEmpty() {
        return this.N == 0;
    }
    
    public int size() {
        return this.N;
    }
    
    public Key min() {
        if (this.isEmpty()) {
            throw new NoSuchElementException("Priority queue underflow");
        }
        return (Key)this.pq[1];
    }
    
    private void resize(final int capacity) {
        assert capacity > this.N;
        final java.lang.Comparable[] temp = new java.lang.Comparable[capacity];
        for (int i = 1; i <= this.N; ++i) {
            temp[i] = this.pq[i];
        }
        this.pq = temp;
    }
    
    public boolean containsKey(final Key key) {
        return this.keyToIndexMap.containsKey(key);
    }
    
    public void insert(final Key x) {
        if (this.N == this.pq.length - 1) {
            this.resize(2 * this.pq.length);
        }
        this.pq[++this.N] = (java.lang.Comparable)x;
        this.keyToIndexMap.put(x, this.N);
        this.swim(this.N);
        assert this.isMinHeap();
    }
    
    public Key delMin() {
        if (this.isEmpty()) {
            throw new NoSuchElementException("Priority queue underflow");
        }
        this.exch(1, this.N);
        final Key min = (Key)this.pq[this.N--];
        this.keyToIndexMap.remove(min);
        this.sink(1);
        this.pq[this.N + 1] = null;
        if (this.N > 0 && this.N == (this.pq.length - 1) / 4) {
            this.resize(this.pq.length / 2);
        }
        assert this.isMinHeap();
        return min;
    }
    
    public void delete(final Key x) {
        if (!this.keyToIndexMap.containsKey(x)) {
            return;
        }
        if (this.isEmpty()) {
            throw new NoSuchElementException("Priority queue underflow");
        }
        final int idx = this.keyToIndexMap.get(x);
        this.exch(idx, this.N);
        --this.N;
        this.keyToIndexMap.remove(x);
        this.sink(idx);
        this.pq[this.N + 1] = null;
        if (this.N > 0 && this.N == (this.pq.length - 1) / 4) {
            this.resize(this.pq.length / 2);
        }
        assert this.isMinHeap();
    }
    
    public void update(final Key key, final K k) {
        final int index = this.keyToIndexMap.get(key);
        ((Updater)key).update(k);
        if (index > 1 && this.greater(index / 2, index)) {
            this.swim(index);
        }
        else {
            this.sink(index);
        }
    }
    
    private void swim(int k) {
        while (k > 1 && this.greater(k / 2, k)) {
            this.exch(k, k / 2);
            k /= 2;
        }
    }
    
    private void sink(int k) {
        while (2 * k <= this.N) {
            int j = 2 * k;
            if (j < this.N && this.greater(j, j + 1)) {
                ++j;
            }
            if (!this.greater(k, j)) {
                break;
            }
            this.exch(k, j);
            k = j;
        }
    }
    
    private boolean greater(final int i, final int j) {
        return this.pq[i].compareTo(this.pq[j]) > 0;
    }
    
    private void exch(final int i, final int j) {
        final Key swap = (Key)this.pq[i];
        this.pq[i] = this.pq[j];
        this.pq[j] = (java.lang.Comparable)swap;
        this.keyToIndexMap.put((Key)this.pq[i], i);
        this.keyToIndexMap.put((Key)this.pq[j], j);
    }
    
    private boolean isMinHeap() {
        return this.isMinHeap(1);
    }
    
    private boolean isMinHeap(final int k) {
        if (k > this.N) {
            return true;
        }
        final int left = 2 * k;
        final int right = 2 * k + 1;
        return (left > this.N || !this.greater(k, left)) && (right > this.N || !this.greater(k, right)) && (this.isMinHeap(left) && this.isMinHeap(right));
    }
    
    @Override
    public Iterator<Key> iterator() {
        return new HeapIterator();
    }
    
    private class HeapIterator implements Iterator<Key>
    {
        private MinPQueue<Key, K> copy;
        
        public HeapIterator() {
            this.copy = new MinPQueue<Key, K>(MinPQueue.this.size());
            for (int i = 1; i <= MinPQueue.this.N; ++i) {
                this.copy.insert((Key)MinPQueue.this.pq[i]);
            }
        }
        
        @Override
        public boolean hasNext() {
            return !this.copy.isEmpty();
        }
        
        @Override
        public void remove() {
            throw new UnsupportedOperationException();
        }
        
        @Override
        public Key next() {
            if (!this.hasNext()) {
                throw new NoSuchElementException();
            }
            return this.copy.delMin();
        }
    }
}
