package ml.clustering;

import la.matrix.*;
import ml.manifold.*;
import ml.options.*;
import la.vector.*;
import ml.utils.*;

public class SpectralClustering extends Clustering
{
    public SpectralClusteringOptions options;
    
    public SpectralClustering() {
        this.options = new SpectralClusteringOptions();
    }
    
    public SpectralClustering(final int nClus) {
        super(nClus);
        this.options = new SpectralClusteringOptions(nClus);
    }
    
    public SpectralClustering(final ClusteringOptions options) {
        super(options);
        this.options = new SpectralClusteringOptions(options);
    }
    
    public SpectralClustering(final SpectralClusteringOptions options) {
        this.options = options;
    }
    
    @Override
    public void initialize(final Matrix G0) {
    }
    
    @Override
    public void clustering() {
        final Matrix X = this.dataMatrix;
        final String TYPE = this.options.graphType;
        double PARAM = this.options.graphParam;
        PARAM = Math.ceil(Math.log(Matlab.size(X, 1)) + 1.0);
        if (PARAM == Matlab.size(X, 1)) {
            --PARAM;
        }
        final String DISTANCEFUNCTION = this.options.graphDistanceFunction;
        Matrix A = Manifold.adjacencyDirected(X, TYPE, PARAM, DISTANCEFUNCTION);
        final Vector Z = Matlab.max(A, 2)[0];
        double WEIGHTPARAM = this.options.graphWeightParam;
        WEIGHTPARAM = Matlab.sum(Z) / Z.getDim();
        A = Matlab.max(A, A.transpose());
        final Matrix W = A.copy();
        final FindResult findResult = Matlab.find(A);
        final int[] A_i = findResult.rows;
        final int[] A_j = findResult.cols;
        final double[] A_v = findResult.vals;
        final String WEIGHTTYPE = this.options.graphWeightType;
        if (WEIGHTTYPE.equals("distance")) {
            for (int i = 0; i < A_i.length; ++i) {
                W.setEntry(A_i[i], A_j[i], A_v[i]);
            }
        }
        else if (WEIGHTTYPE.equals("inner")) {
            for (int i = 0; i < A_i.length; ++i) {
                W.setEntry(A_i[i], A_j[i], 1.0 - A_v[i] / 2.0);
            }
        }
        else if (WEIGHTTYPE.equals("binary")) {
            for (int i = 0; i < A_i.length; ++i) {
                W.setEntry(A_i[i], A_j[i], 1.0);
            }
        }
        else if (WEIGHTTYPE.equals("heat")) {
            final double t = -2.0 * WEIGHTPARAM * WEIGHTPARAM;
            for (int j = 0; j < A_i.length; ++j) {
                W.setEntry(A_i[j], A_j[j], Math.exp(A_v[j] * A_v[j] / t));
            }
        }
        else {
            System.err.println("Unknown Weight Type.");
            System.exit(1);
        }
        final Vector D = Matlab.sum(W, 2);
        final Matrix Dsqrt = Matlab.diag(Matlab.dotDivide(1.0, Matlab.sqrt(D)));
        final Matrix L_sym = Matlab.speye(Matlab.size(W, 1)).minus(Dsqrt.mtimes(W).mtimes(Dsqrt));
        final Matrix[] eigRes = Matlab.eigs(L_sym, this.options.nClus, "sm");
        final Matrix V = eigRes[0];
        final Matrix U = Dsqrt.mtimes(V);
        final KMeansOptions kMeansOptions = new KMeansOptions();
        kMeansOptions.nClus = this.options.nClus;
        kMeansOptions.maxIter = this.options.maxIter;
        kMeansOptions.verbose = this.options.verbose;
        final KMeans KMeans = new KMeans(kMeansOptions);
        KMeans.feedData(U);
        KMeans.initialize(null);
        KMeans.clustering();
        this.indicatorMatrix = KMeans.indicatorMatrix;
        System.out.println("Spectral clustering complete.");
    }
    
    public static void main(final String[] args) {
        Time.tic();
        final int nClus = 2;
        final boolean verbose = false;
        final int maxIter = 100;
        final String graphType = "nn";
        final double graphParam = 6.0;
        final String graphDistanceFunction = "euclidean";
        final String graphWeightType = "heat";
        final double graphWeightParam = 1.0;
        final ClusteringOptions options = new SpectralClusteringOptions(nClus, verbose, maxIter, graphType, graphParam, graphDistanceFunction, graphWeightType, graphWeightParam);
        final Clustering spectralClustering = new SpectralClustering(options);
        final double[][] data = { { 3.5, 5.3, 0.2, -1.2 }, { 4.4, 2.2, 0.3, 0.4 }, { 1.3, 0.5, 4.1, 3.2 } };
        spectralClustering.feedData(data);
        spectralClustering.clustering(null);
        Printer.display(Matlab.full(spectralClustering.getIndicatorMatrix()));
        System.out.format("Elapsed time: %.3f seconds\n", Time.toc());
    }
}
