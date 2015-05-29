package ml.recommendation.util;

import java.util.*;
import java.io.*;

public class Tree
{
    public Node root;
    
    public Tree() {
        this.root = new Node(0, -1);
    }
    
    public void insertTopCencept(final int idx) {
        if (this.root.children == null) {
            this.root.children = new TreeMap<Integer, Node>();
        }
        final TreeMap<Integer, Node> children = this.root.children;
        if (!children.containsKey(idx)) {
            final Node node = new Node(idx, this.root.idx);
            children.put(idx, node);
        }
    }
    
    public void insertEdge(final Node parent, final int pIdx, final int idx) {
        if (parent.idx == pIdx) {
            if (parent.children == null) {
                parent.children = new TreeMap<Integer, Node>();
            }
            final TreeMap<Integer, Node> children = parent.children;
            if (!children.containsKey(idx)) {
                final Node node = new Node(idx, this.root.idx);
                children.put(idx, node);
            }
            return;
        }
        for (final Node child : parent.children.values()) {
            this.insertEdge(child, pIdx, idx);
        }
    }
    
    public void insertPath(final LinkedList<Integer> path) {
        this.insertPath(this.root, path);
    }
    
    public void insertPath(final Node parent, final LinkedList<Integer> path) {
        if (path.isEmpty()) {
            return;
        }
        if (parent.children == null) {
            parent.children = new TreeMap<Integer, Node>();
        }
        final TreeMap<Integer, Node> children = parent.children;
        Node child = null;
        final int index = path.pop();
        if (!children.containsKey(index)) {
            child = new Node(index, parent.idx);
            children.put(index, child);
        }
        else {
            child = children.get(index);
        }
        this.insertPath(child, path);
    }
    
    public void insertLocation(final int countryIdx, final int stateIdx, final int cityIdx) {
        if (this.root.children == null) {
            this.root.children = new TreeMap<Integer, Node>();
        }
        final TreeMap<Integer, Node> countries = this.root.children;
        Node country = null;
        if (!countries.containsKey(countryIdx)) {
            country = new Node(countryIdx, this.root.idx);
            countries.put(countryIdx, country);
        }
        else {
            country = countries.get(countryIdx);
        }
        if (country.children == null) {
            country.children = new TreeMap<Integer, Node>();
        }
        final TreeMap<Integer, Node> states = country.children;
        Node state = null;
        if (!states.containsKey(stateIdx)) {
            state = new Node(stateIdx, country.idx);
            states.put(stateIdx, state);
        }
        else {
            state = states.get(stateIdx);
        }
        if (state.children == null) {
            state.children = new TreeMap<Integer, Node>();
        }
        final TreeMap<Integer, Node> cities = state.children;
        Node city = null;
        if (!cities.containsKey(cityIdx)) {
            city = new Node(cityIdx, state.idx);
            cities.put(cityIdx, city);
        }
        else {
            city = cities.get(cityIdx);
        }
    }
    
    public void print() {
        final String indent = "";
        final String indentUnit = "    ";
        this.print(indent, indentUnit);
    }
    
    public void print(final String indent, final String indentUnit) {
        this.print(this.root, indent, indentUnit, 0);
    }
    
    public void print(final Node node, String indent, final String indentUnit, int level) {
        if (node == null) {
            return;
        }
        System.out.print(indent);
        final String type = (level == 0) ? "root" : String.format("level %d", level);
        System.out.printf("%d (%s)\n", node.idx, type);
        final TreeMap<Integer, Node> children = node.children;
        if (children == null) {
            return;
        }
        indent = String.valueOf(indent) + indentUnit;
        ++level;
        for (final int idx : children.keySet()) {
            this.print(children.get(idx), indent, indentUnit, level);
        }
    }
    
    public void save(final String filePath) {
        final String indent = "";
        final String indentUnit = "    ";
        this.save(filePath, indent, indentUnit);
    }
    
    public void save(final String filePath, final String indent, final String indentUnit) {
        PrintWriter pw = null;
        try {
            pw = new PrintWriter(new BufferedWriter(new FileWriter(filePath)), true);
        }
        catch (IOException e) {
            e.printStackTrace();
            System.exit(1);
        }
        this.save(pw, this.root, indent, indentUnit, 0);
        pw.close();
    }
    
    public void save(final PrintWriter pw, final Node node, String indent, final String indentUnit, int level) {
        if (node == null) {
            return;
        }
        pw.print(indent);
        pw.printf("%d\n", node.idx);
        final TreeMap<Integer, Node> children = node.children;
        if (children == null) {
            return;
        }
        indent = String.valueOf(indent) + indentUnit;
        ++level;
        for (final int idx : children.keySet()) {
            this.save(pw, children.get(idx), indent, indentUnit, level);
        }
    }
    
    public void saveGeoTree(final String filePath) {
        final String indent = "";
        final String indentUnit = "    ";
        this.saveGeoTree(filePath, indent, indentUnit);
    }
    
    public void saveGeoTree(final String filePath, final String indent, final String indentUnit) {
        PrintWriter pw = null;
        try {
            pw = new PrintWriter(new BufferedWriter(new FileWriter(filePath)), true);
        }
        catch (IOException e) {
            e.printStackTrace();
            System.exit(1);
        }
        this.saveGeoTree(pw, this.root, indent, indentUnit, 0);
        pw.close();
    }
    
    public void saveGeoTree(final PrintWriter pw, final Node node, String indent, final String indentUnit, int level) {
        if (node == null) {
            return;
        }
        pw.print(indent);
        String type = "";
        switch (level) {
            case 0: {
                type = "root";
                break;
            }
            case 1: {
                type = "country";
                break;
            }
            case 2: {
                type = "state";
                break;
            }
            case 3: {
                type = "city";
                break;
            }
        }
        pw.printf("%d (%s)\n", node.idx, type);
        final TreeMap<Integer, Node> children = node.children;
        if (children == null) {
            return;
        }
        indent = String.valueOf(indent) + indentUnit;
        ++level;
        for (final int idx : children.keySet()) {
            this.saveGeoTree(pw, children.get(idx), indent, indentUnit, level);
        }
    }
    
    public void saveWithLevelTags(final String filePath) {
        final String indent = "";
        final String indentUnit = "    ";
        this.saveWithLevelTags(filePath, indent, indentUnit);
    }
    
    public void saveWithLevelTags(final String filePath, final String indent, final String indentUnit) {
        PrintWriter pw = null;
        try {
            pw = new PrintWriter(new BufferedWriter(new FileWriter(filePath)), true);
        }
        catch (IOException e) {
            e.printStackTrace();
            System.exit(1);
        }
        this.saveWithLevelTags(pw, this.root, indent, indentUnit, 0);
        pw.close();
    }
    
    public void saveWithLevelTags(final PrintWriter pw, final Node node, String indent, final String indentUnit, int level) {
        if (node == null) {
            return;
        }
        pw.print(indent);
        final String type = (level == 0) ? "root" : String.format("level %d", level);
        pw.printf("%d (%s)\n", node.idx, type);
        final TreeMap<Integer, Node> children = node.children;
        if (children == null) {
            return;
        }
        indent = String.valueOf(indent) + indentUnit;
        ++level;
        for (final int idx : children.keySet()) {
            this.saveWithLevelTags(pw, children.get(idx), indent, indentUnit, level);
        }
    }
}
