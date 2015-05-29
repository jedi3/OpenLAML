package ml.clustering;

import ml.options.*;
import la.matrix.*;
import la.vector.*;
import ml.utils.*;
import la.io.*;
import la.io.IO;

public class KMeans extends Clustering
{
    KMeansOptions options;
    
    public KMeans(final int nClus) {
        super(nClus);
        this.options.maxIter = 100;
        this.options.verbose = false;
    }
    
    public KMeans(final int nClus, final int maxIter) {
        super(nClus);
        this.options.maxIter = maxIter;
        this.options.verbose = false;
    }
    
    public KMeans(final int nClus, final int maxIter, final boolean verbose) {
        super(nClus);
        this.options.maxIter = maxIter;
        this.options.verbose = verbose;
    }
    
    public KMeans(final KMeansOptions options) {
        super(options.nClus);
        this.options = options;
    }
    
    @Override
    public void clustering() {
        if (this.indicatorMatrix == null) {
            this.initialize(null);
        }
        final Vector[] maxRes = Matlab.max(this.indicatorMatrix, 2);
        final double[] indicators = new double[this.nExample];
        for (int i = 0; i < this.nExample; ++i) {
            if (maxRes[0].get(i) != 0.0) {
                indicators[i] = maxRes[1].get(i);
            }
            else {
                indicators[i] = -1.0;
            }
        }
        final double[] clusterSizes = ArrayOperator.allocate1DArray(this.nClus, 0.0);
        Vector[] examples = null;
        if (this.dataMatrix instanceof SparseMatrix) {
            examples = Matlab.sparseMatrix2SparseRowVectors(this.dataMatrix);
        }
        else if (this.dataMatrix instanceof DenseMatrix) {
            final double[][] data = ((DenseMatrix)this.dataMatrix).getData();
            examples = new Vector[this.nExample];
            for (int j = 0; j < this.nExample; ++j) {
                examples[j] = DenseVector.buildDenseVector(data[j]);
            }
        }
        final Vector[] centers = new Vector[this.nClus];
        if (this.dataMatrix instanceof DenseMatrix) {
            for (int k = 0; k < this.nClus; ++k) {
                centers[k] = new DenseVector(this.nFeature);
            }
        }
        else {
            for (int k = 0; k < this.nClus; ++k) {
                centers[k] = new SparseVector(this.nFeature);
            }
        }
        for (int j = 0; j < this.nExample; ++j) {
            final int l = (int)indicators[j];
            if (l != -1) {
                InPlaceOperator.plusAssign(centers[l], examples[j]);
                final double[] array = clusterSizes;
                final int n = l;
                ++array[n];
            }
        }
        for (int k = 0; k < this.nClus; ++k) {
            InPlaceOperator.timesAssign(centers[k], 1.0 / clusterSizes[k]);
        }
        int cnt = 0;
        Matrix DistMatrix = null;
        double mse = 0.0;
        while (cnt < this.options.maxIter) {
            final Matrix indOld = this.indicatorMatrix;
            final long start = System.currentTimeMillis();
            DistMatrix = Matlab.l2DistanceSquare(centers, examples);
            final Vector[] minRes = Matlab.min(DistMatrix);
            final Vector minVals = minRes[0];
            final Vector IX = minRes[1];
            this.indicatorMatrix = new SparseMatrix(this.nExample, this.nClus);
            for (int m = 0; m < this.nExample; ++m) {
                this.indicatorMatrix.setEntry(m, (int)IX.get(m), 1.0);
                indicators[m] = IX.get(m);
            }
            mse = Matlab.sum(minVals) / this.nExample;
            if (Matlab.norm(indOld.minus(this.indicatorMatrix), "fro") == 0.0) {
                System.out.println("KMeans complete.");
                break;
            }
            final double elapsedTime = (System.currentTimeMillis() - start) / 1000.0;
            ++cnt;
            if (this.options.verbose) {
                System.out.format("Iter %d: mse = %.3f (%.3f secs)\n", cnt, mse, elapsedTime);
            }
            InPlaceOperator.clear(clusterSizes);
            for (int k2 = 0; k2 < this.nClus; ++k2) {
                centers[k2].clear();
            }
            for (int i2 = 0; i2 < this.nExample; ++i2) {
                final int k3 = (int)indicators[i2];
                InPlaceOperator.plusAssign(centers[k3], examples[i2]);
                final double[] array2 = clusterSizes;
                final int n2 = k3;
                ++array2[n2];
            }
            for (int k2 = 0; k2 < this.nClus; ++k2) {
                InPlaceOperator.timesAssign(centers[k2], 1.0 / clusterSizes[k2]);
            }
        }
        if (this.dataMatrix instanceof SparseMatrix) {
            this.centers = Matlab.sparseRowVectors2SparseMatrix(centers);
        }
        else if (this.dataMatrix instanceof DenseMatrix) {
            this.centers = Matlab.denseRowVectors2DenseMatrix(centers);
        }
    }
    
    public static void main(final String[] args) {
        runKMeans();
        final int K = 3;
        final int maxIter = 100;
        final boolean verbose = true;
        final KMeansOptions options = new KMeansOptions(K, maxIter, verbose);
        final Clustering KMeans = new KMeans(options);
        final double[][] matrixData2 = { { 1.0, 0.0, 3.0, 2.0, 0.0 }, { 2.0, 5.0, 3.0, 1.0, 0.0 }, { 4.0, 1.0, 0.0, 0.0, 1.0 }, { 3.0, 0.0, 1.0, 0.0, 2.0 }, { 2.0, 5.0, 3.0, 1.0, 6.0 } };
        final Matrix dataMatrix = new DenseMatrix(matrixData2);
        Printer.printMatrix(dataMatrix);
        final Matrix X = IO.loadMatrix("CNNTest-TrainingData.txt");
        Matrix X2 = Matlab.normalizeByColumns(Matlab.getTFIDF(X));
        X2 = X2.transpose();
        KMeans.feedData(X2);
        Matrix initializer = IO.loadMatrix("indicators");
        initializer = null;
        KMeans.initialize(initializer);
        KMeans.clustering();
        System.out.println("Indicator Matrix:");
        Printer.printMatrix(Matlab.full(KMeans.getIndicatorMatrix()));
    }
    
    public static void runKMeans() {
        final double[][] data = { { 3.5, 5.3, 0.2, -1.2 }, { 4.4, 2.2, 0.3, 0.4 }, { 1.3, 0.5, 4.1, 3.2 } };
        final KMeansOptions options = new KMeansOptions();
        options.nClus = 2;
        options.verbose = true;
        options.maxIter = 100;
        final KMeans KMeans = new KMeans(options);
        KMeans.feedData(data);
        Matrix initializer = null;
        initializer = new SparseMatrix(3, 2);
        initializer.setEntry(0, 0, 1.0);
        initializer.setEntry(1, 1, 1.0);
        initializer.setEntry(2, 0, 1.0);
        KMeans.clustering(initializer);
        System.out.println("Indicator Matrix:");
        Printer.printMatrix(Matlab.full(KMeans.getIndicatorMatrix()));
    }
}
