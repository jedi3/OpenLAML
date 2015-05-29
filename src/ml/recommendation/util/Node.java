package ml.recommendation.util;

import java.util.*;

public class Node
{
    public int idx;
    public int parentIdx;
    public TreeMap<Integer, Node> children;
    
    public Node() {
        this.idx = 0;
        this.parentIdx = 0;
        this.children = null;
    }
    
    public Node(final int idx, final int parentIdx) {
        this.idx = idx;
        this.parentIdx = parentIdx;
        this.children = null;
    }
}
