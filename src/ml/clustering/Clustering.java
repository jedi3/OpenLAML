package ml.clustering;

import ml.options.*;
import java.util.*;
import la.matrix.*;

public abstract class Clustering
{
    public int nClus;
    public int nFeature;
    public int nExample;
    protected Matrix dataMatrix;
    protected Matrix indicatorMatrix;
    protected Matrix centers;
    
    public Clustering() {
        this.nClus = 0;
    }
    
    public Clustering(final ClusteringOptions clusteringOptions) {
        this.nClus = clusteringOptions.nClus;
    }
    
    public Clustering(final int nClus) {
        if (nClus < 1) {
            System.err.println("Number of clusters less than one!");
            System.exit(1);
        }
        this.nClus = nClus;
    }
    
    public void feedData(final Matrix dataMatrix) {
        this.dataMatrix = dataMatrix;
        this.nExample = dataMatrix.getRowDimension();
        this.nFeature = dataMatrix.getColumnDimension();
    }
    
    public void feedData(final double[][] data) {
        this.feedData(new DenseMatrix(data));
    }
    
    public void initialize(final Matrix G0) {
        if (G0 != null) {
            this.indicatorMatrix = G0;
            return;
        }
        final List<Integer> indList = new ArrayList<Integer>();
        for (int i = 0; i < this.nExample; ++i) {
            indList.add(i);
        }
        final Random rdn = new Random(System.currentTimeMillis());
        Collections.shuffle(indList, rdn);
        this.indicatorMatrix = new SparseMatrix(this.nExample, this.nClus);
        for (int j = 0; j < this.nClus; ++j) {
            this.indicatorMatrix.setEntry(indList.get(j), j, 1.0);
        }
    }
    
    public abstract void clustering();
    
    public void clustering(final Matrix G0) {
        this.initialize(G0);
        this.clustering();
    }
    
    public Matrix getData() {
        return this.dataMatrix;
    }
    
    public Matrix getCenters() {
        return this.centers;
    }
    
    public Matrix getIndicatorMatrix() {
        return this.indicatorMatrix;
    }
    
    public static double getAccuracy(final Matrix G, final Matrix groundTruth) {
        System.out.println("Sorry, this function has not been implemented yet...");
        return 0.0;
    }
}
