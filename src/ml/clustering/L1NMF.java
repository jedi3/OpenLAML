package ml.clustering;

import java.util.*;

import ml.options.*;
import la.matrix.*;
import la.vector.*;
import la.vector.Vector;
import ml.utils.*;
import la.io.*;
import la.io.IO;

public class L1NMF extends Clustering
{
    public double epsilon;
    public int maxIter;
    public double gamma;
    public double mu;
    public boolean calc_OV;
    public boolean verbose;
    public ArrayList<Double> valueList;
    Matrix initializer;
    
    public L1NMF(final Options options) {
        this.initializer = null;
        this.maxIter = options.maxIter;
        this.epsilon = options.epsilon;
        this.gamma = options.gamma;
        this.mu = options.mu;
        this.verbose = options.verbose;
        this.calc_OV = options.calc_OV;
        this.nClus = options.nClus;
    }
    
    public L1NMF(final L1NMFOptions L1NMFOptions) {
        this.initializer = null;
        this.maxIter = L1NMFOptions.maxIter;
        this.epsilon = L1NMFOptions.epsilon;
        this.gamma = L1NMFOptions.gamma;
        this.mu = L1NMFOptions.mu;
        this.verbose = L1NMFOptions.verbose;
        this.calc_OV = L1NMFOptions.calc_OV;
        this.nClus = L1NMFOptions.nClus;
    }
    
    public L1NMF() {
        this.initializer = null;
        final L1NMFOptions options = new L1NMFOptions();
        this.maxIter = options.maxIter;
        this.epsilon = options.epsilon;
        this.gamma = options.gamma;
        this.mu = options.mu;
        this.verbose = options.verbose;
        this.calc_OV = options.calc_OV;
        this.nClus = options.nClus;
    }
    
    @Override
    public void initialize(final Matrix G0) {
        if (G0 != null) {
            this.initializer = G0;
            return;
        }
        final KMeansOptions kMeansOptions = new KMeansOptions();
        kMeansOptions.nClus = this.nClus;
        kMeansOptions.maxIter = 50;
        kMeansOptions.verbose = true;
        System.out.println("Using KMeans to initialize...");
        final Clustering KMeans = new KMeans(kMeansOptions);
        KMeans.feedData(this.dataMatrix);
        KMeans.clustering();
        this.initializer = KMeans.getIndicatorMatrix();
    }
    
    @Override
    public void clustering() {
        if (this.initializer == null) {
            this.initialize(null);
        }
        this.clustering(this.initializer);
    }
    
    @Override
    public void clustering(Matrix G0) {
        if (G0 == null) {
            this.initialize(null);
            G0 = this.initializer;
        }
        final Matrix X = this.dataMatrix;
        Matrix G = G0;
        Matrix F = Matlab.mldivide(G.transpose().mtimes(G), G.transpose().mtimes(X));
        G = Matlab.full(G);
        final ArrayList<Double> J = new ArrayList<Double>();
        final Matrix F_pos = Matlab.subplus(F);
        F = F_pos.plus(0.2 * Matlab.sumAll(F_pos) / Matlab.find(F_pos).rows.length);
        final Matrix E_F = Matlab.ones(Matlab.size(F)).times(this.gamma / 2.0);
        final Matrix E_G = Matlab.ones(Matlab.size(G)).times(this.mu / 2.0);
        if (this.calc_OV) {
            J.add(this.f(X, F, G, E_F, E_G));
        }
        int ind = 0;
        final Matrix G_old = new DenseMatrix(Matlab.size(G));
        double d = 0.0;
        do {
            InPlaceOperator.assign(G_old, G);
            G = this.UpdateG(X, F, this.mu, G);
            F = this.UpdateF(X, G, this.gamma, F);
            if (++ind > this.maxIter) {
                System.out.println("Maximal iterations");
                break;
            }
            d = Matlab.norm(G.minus(G_old), "fro");
            if (this.calc_OV) {
                J.add(this.f(X, F, G, E_F, E_G));
            }
            if (ind % 10 == 0 && this.verbose) {
                if (this.calc_OV) {
                    System.out.println(String.format("Iteration %d, delta G: %f, J: %f", ind, d, J.get(J.size() - 1)));
                }
                else {
                    System.out.println(String.format("Iteration %d, delta G: %f", ind, d));
                }
            }
            if (this.calc_OV) {
                if (Math.abs(J.get(J.size() - 2) - J.get(J.size() - 1)) < this.epsilon && d < this.epsilon) {
                    System.out.println("Converge successfully!");
                    break;
                }
                continue;
            }
            else {
                if (d < this.epsilon) {
                    System.out.println("Converge successfully!");
                    break;
                }
                continue;
            }
        } while (Matlab.sumAll(Matlab.isnan(G)) <= 0.0);
        this.centers = F;
        this.indicatorMatrix = G;
        this.valueList = J;
    }
    
    private Matrix UpdateG(final Matrix X, final Matrix F, final double mu, final Matrix G0) {
        final int MaxIter = 10000;
        final double epsilon = 0.1;
        final int K = Matlab.size(F, 1);
        final int NExample = Matlab.size(X, 1);
        final Matrix S = F.mtimes(F.transpose());
        final Matrix C = X.mtimes(F.transpose());
        InPlaceOperator.timesAssign(C, -1.0);
        InPlaceOperator.plusAssign(C, mu / 2.0);
        final double[] D = ((DenseVector)Matlab.diag(S).getColumnVector(0)).getPr();
        int ind = 0;
        double d = 0.0;
        final Matrix G_old = new DenseMatrix(Matlab.size(G0));
        final Vector GSPlusCj = new DenseVector(NExample);
        final Vector[] SColumns = Matlab.denseMatrix2DenseColumnVectors(S);
        final Vector[] CColumns = Matlab.denseMatrix2DenseColumnVectors(C);
        final double[][] GData = ((DenseMatrix)G0).getData();
        double[] pr = null;
        do {
            InPlaceOperator.assign(G_old, G0);
            for (int j = 0; j < K; ++j) {
                InPlaceOperator.operate(GSPlusCj, G0, SColumns[j]);
                InPlaceOperator.plusAssign(GSPlusCj, CColumns[j]);
                InPlaceOperator.timesAssign(GSPlusCj, 1.0 / D[j]);
                pr = ((DenseVector)GSPlusCj).getPr();
                for (int i = 0; i < NExample; ++i) {
                    GData[i][j] = Math.max(GData[i][j] - pr[i], 0.0);
                }
            }
            if (++ind > MaxIter) {
                break;
            }
            d = Matlab.sumAll(Matlab.abs(G0.minus(G_old)));
        } while (d >= epsilon);
        return G0;
    }
    
    private Matrix UpdateF(final Matrix X, final Matrix G, final double gamma, final Matrix F0) {
        final int MaxIter = 10000;
        final double epsilon = 0.1;
        final int K = Matlab.size(G, 2);
        final int NFea = Matlab.size(X, 2);
        final Matrix S = G.transpose().mtimes(G);
        final Matrix C = G.transpose().mtimes(X);
        InPlaceOperator.timesAssign(C, -1.0);
        InPlaceOperator.plusAssign(C, gamma / 2.0);
        final double[] D = ((DenseVector)Matlab.diag(S).getColumnVector(0)).getPr();
        int ind = 0;
        double d = 0.0;
        final Matrix F_old = new DenseMatrix(Matlab.size(F0));
        final Vector SFPlusCi = new DenseVector(NFea);
        final Vector[] SRows = Matlab.denseMatrix2DenseRowVectors(S);
        final Vector[] CRows = Matlab.denseMatrix2DenseRowVectors(C);
        final double[][] FData = ((DenseMatrix)F0).getData();
        double[] FRow = null;
        double[] pr = null;
        do {
            InPlaceOperator.assign(F_old, F0);
            for (int i = 0; i < K; ++i) {
                InPlaceOperator.operate(SFPlusCi, SRows[i], F0);
                InPlaceOperator.plusAssign(SFPlusCi, CRows[i]);
                InPlaceOperator.timesAssign(SFPlusCi, 1.0 / D[i]);
                pr = ((DenseVector)SFPlusCi).getPr();
                FRow = FData[i];
                for (int j = 0; j < NFea; ++j) {
                    FRow[j] = Math.max(FRow[j] - pr[j], 0.0);
                }
            }
            if (++ind > MaxIter) {
                break;
            }
            d = Matlab.sumAll(Matlab.abs(F0.minus(F_old)));
        } while (d >= epsilon);
        return F0;
    }
    
    private double f(final Matrix X, final Matrix F, final Matrix G, final Matrix E_F, final Matrix E_G) {
        final double fval = Math.pow(Matlab.norm(X.minus(G.mtimes(F)), "fro"), 2.0) + 2.0 * Matlab.sumAll(E_F.times(F)) + 2.0 * Matlab.sumAll(E_G.times(G));
        return fval;
    }
    
    public static void main(final String[] args) {
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
        KMeans.clustering();
        final Matrix G0 = KMeans.getIndicatorMatrix();
        final L1NMFOptions L1NMFOptions = new L1NMFOptions();
        L1NMFOptions.nClus = 10;
        L1NMFOptions.gamma = 1.0E-4;
        L1NMFOptions.mu = 0.1;
        L1NMFOptions.maxIter = 50;
        L1NMFOptions.verbose = true;
        L1NMFOptions.calc_OV = false;
        L1NMFOptions.epsilon = 1.0E-5;
        final Clustering L1NMF = new L1NMF(L1NMFOptions);
        L1NMF.feedData(X);
        L1NMF.clustering(G0);
        System.out.format("Elapsed time: %.3f seconds\n", Time.toc());
        IO.saveMatrix("F.txt", L1NMF.centers);
        IO.saveMatrix("G.txt", L1NMF.indicatorMatrix);
    }
}
