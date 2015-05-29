package ml.clustering;

import la.io.*;
import la.io.IO;
import la.matrix.*;
import ml.options.*;
import ml.utils.*;

public class NMF extends L1NMF
{
    public static void main(final String[] args) {
        runNMF();
        final String dataMatrixFilePath = "CNN - DocTermCount.txt";
        Time.tic();
        Matrix X = IO.loadMatrixFromDocTermCountFile(dataMatrixFilePath);
        X = Matlab.getTFIDF(X);
        X = Matlab.normalizeByColumns(X);
        X = X.transpose();
        final KMeansOptions kMeansOptions = new KMeansOptions();
        kMeansOptions.nClus = 10;
        kMeansOptions.maxIter = 50;
        kMeansOptions.verbose = true;
        final KMeans KMeans = new KMeans(kMeansOptions);
        KMeans.feedData(X);
        KMeans.initialize(null);
        KMeans.clustering();
        final Matrix G0 = KMeans.getIndicatorMatrix();
        final NMFOptions NMFOptions = new NMFOptions();
        NMFOptions.maxIter = 300;
        NMFOptions.verbose = true;
        NMFOptions.calc_OV = false;
        NMFOptions.epsilon = 1.0E-5;
        final Clustering NMF = new NMF(NMFOptions);
        NMF.feedData(X);
        NMF.initialize(G0);
        NMF.clustering();
        System.out.format("Elapsed time: %.3f seconds\n", Time.toc());
        IO.saveDenseMatrix("F.txt", NMF.centers);
        IO.saveDenseMatrix("G.txt", NMF.indicatorMatrix);
    }
    
    public NMF(final Options options) {
        super(options);
        this.gamma = 0.0;
        this.mu = 0.0;
    }
    
    public NMF(final NMFOptions NMFOptions) {
        this.nClus = NMFOptions.nClus;
        this.maxIter = NMFOptions.maxIter;
        this.epsilon = NMFOptions.epsilon;
        this.verbose = NMFOptions.verbose;
        this.calc_OV = NMFOptions.calc_OV;
        this.gamma = 0.0;
        this.mu = 0.0;
    }
    
    public NMF() {
        final Options options = new Options();
        this.nClus = options.nClus;
        this.maxIter = options.maxIter;
        this.epsilon = options.epsilon;
        this.verbose = options.verbose;
        this.calc_OV = options.calc_OV;
        this.gamma = 0.0;
        this.mu = 0.0;
    }
    
    public static void runNMF() {
        final double[][] data = { { 3.5, 4.4, 1.3 }, { 5.3, 2.2, 0.5 }, { 0.2, 0.3, 4.1 }, { 1.2, 0.4, 3.2 } };
        final KMeansOptions options = new KMeansOptions();
        options.nClus = 2;
        options.verbose = true;
        options.maxIter = 100;
        final KMeans KMeans = new KMeans(options);
        KMeans.feedData(data);
        KMeans.initialize(null);
        KMeans.clustering();
        final Matrix G0 = KMeans.getIndicatorMatrix();
        final NMFOptions NMFOptions = new NMFOptions();
        NMFOptions.nClus = 2;
        NMFOptions.maxIter = 50;
        NMFOptions.verbose = true;
        NMFOptions.calc_OV = false;
        NMFOptions.epsilon = 1.0E-5;
        final Clustering NMF = new NMF(NMFOptions);
        NMF.feedData(data);
        NMF.clustering(G0);
        System.out.println("Basis Matrix:");
        Printer.printMatrix(Matlab.full(NMF.getCenters()));
        System.out.println("Indicator Matrix:");
        Printer.printMatrix(Matlab.full(NMF.getIndicatorMatrix()));
    }
}
