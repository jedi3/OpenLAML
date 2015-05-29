package ml.recommendation.util;

import java.util.*;

import la.vector.*;
import la.vector.Vector;

public class Data
{
    public int M;
    public int N;
    public int T;
    public int Pu;
    public int Pv;
    public int Pe;
    public double[] Yij;
    public double[][] Xij;
    public int[] UserIndices;
    public int[] ItemIndices;
    public HashMap<Integer, LinkedList<Integer>> CUser;
    public HashMap<Integer, LinkedList<Integer>> CItem;
    public HashMap<Integer, LinkedList<Integer>> User2EventIndexSetMap;
    public HashMap<Integer, LinkedList<Integer>> Item2EventIndexSetMap;
    public Vector[] Xi;
    public Vector[] Xj;
    
    public Data(final int M, final int N, final int T, final double[] Yij, final double[][] Xij, final int[] UserIndices, final int[] ItemIndices, final HashMap<Integer, LinkedList<Integer>> CUser, final HashMap<Integer, LinkedList<Integer>> CItem, final HashMap<Integer, LinkedList<Integer>> User2EventIndexSetMap, final HashMap<Integer, LinkedList<Integer>> Item2EventIndexSetMap, final Vector[] Xi, final Vector[] Xj, final int Pu, final int Pv, final int Pe) {
        this.M = 0;
        this.N = 0;
        this.T = 0;
        this.Pu = 0;
        this.Pv = 0;
        this.Pe = 0;
        this.Yij = null;
        this.Xij = null;
        this.UserIndices = null;
        this.ItemIndices = null;
        this.CUser = new HashMap<Integer, LinkedList<Integer>>();
        this.CItem = new HashMap<Integer, LinkedList<Integer>>();
        this.User2EventIndexSetMap = new HashMap<Integer, LinkedList<Integer>>();
        this.Item2EventIndexSetMap = new HashMap<Integer, LinkedList<Integer>>();
        this.Xi = null;
        this.Xj = null;
        this.M = M;
        this.N = N;
        this.T = T;
        this.Yij = Yij;
        this.Xij = Xij;
        this.UserIndices = UserIndices;
        this.ItemIndices = ItemIndices;
        this.CUser = CUser;
        this.CItem = CItem;
        this.User2EventIndexSetMap = User2EventIndexSetMap;
        this.Item2EventIndexSetMap = Item2EventIndexSetMap;
        this.Xi = Xi;
        this.Xj = Xj;
        this.Pu = Pu;
        this.Pv = Pv;
        this.Pe = Pe;
    }
    
    public Data() {
        this.M = 0;
        this.N = 0;
        this.T = 0;
        this.Pu = 0;
        this.Pv = 0;
        this.Pe = 0;
        this.Yij = null;
        this.Xij = null;
        this.UserIndices = null;
        this.ItemIndices = null;
        this.CUser = new HashMap<Integer, LinkedList<Integer>>();
        this.CItem = new HashMap<Integer, LinkedList<Integer>>();
        this.User2EventIndexSetMap = new HashMap<Integer, LinkedList<Integer>>();
        this.Item2EventIndexSetMap = new HashMap<Integer, LinkedList<Integer>>();
        this.Xi = null;
        this.Xj = null;
    }
}
