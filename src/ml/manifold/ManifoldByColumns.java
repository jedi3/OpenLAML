package ml.manifold;

import la.io.*;
import la.io.IO;
import ml.options.*;
import ml.utils.*;
import la.matrix.*;
import ml.kernel.*;
import la.vector.*;

public class ManifoldByColumns
{
    public static void main(final String[] args) {
        final String filePath = "CNN - DocTermCount.txt";
        Matrix X = IO.loadMatrixFromDocTermCountFile(filePath);
        final int NSample = Math.min(20, X.getColumnDimension());
        X = X.getSubMatrix(0, X.getRowDimension() - 1, 0, NSample - 1);
        System.out.println(String.format("%d samples loaded", X.getColumnDimension()));
        final GraphOptions options = new GraphOptions();
        options.graphType = "nn";
        final String type = options.graphType;
        double NN = options.graphParam;
        System.out.println(String.format("Graph type: %s with NN: %d", type, (int)NN));
        options.kernelType = "cosine";
        options.graphDistanceFunction = "cosine";
        options.graphNormalize = true;
        options.graphWeightType = "heat";
        final boolean show = true;
        Time.tic();
        final String DISTANCEFUNCTION = options.graphDistanceFunction;
        final Matrix A = adjacency(X, type, NN, DISTANCEFUNCTION);
        System.out.format("Elapsed time: %.2f seconds.%n", Time.toc());
        final String adjacencyFilePath = "adjacency.txt";
        IO.saveMatrix(adjacencyFilePath, A);
        if (show) {
            Printer.disp(A.getSubMatrix(0, 9, 0, 9));
        }
        Time.tic();
        final Matrix L = laplacian(X, type, options);
        System.out.format("Elapsed time: %.2f seconds.%n", Time.toc());
        final String LaplacianFilePath = "Laplacian.txt";
        IO.saveMatrix(LaplacianFilePath, L);
        if (show) {
            Printer.disp(L.getSubMatrix(0, 9, 0, 9));
        }
        NN = options.graphParam;
        final String DISTFUNC = options.graphDistanceFunction;
        final String KernelType = options.kernelType;
        final double KernelParam = options.kernelParam;
        final double lambda = 0.001;
        Time.tic();
        final Matrix LLR_text = calcLLR(X, NN, DISTFUNC, KernelType, KernelParam, lambda);
        System.out.format("Elapsed time: %.2f seconds.%n", Time.toc());
        final String LLRFilePath = "localLearningRegularization.txt";
        IO.saveMatrix(LLRFilePath, LLR_text);
        if (show) {
            Printer.display(LLR_text.getSubMatrix(0, 9, 0, 9));
        }
    }
    
    public static Matrix laplacian(final Matrix X, final String type, final GraphOptions options) {
        System.out.println("Computing Graph Laplacian...");
        final double NN = options.graphParam;
        final String DISTANCEFUNCTION = options.graphDistanceFunction;
        final String WEIGHTTYPE = options.graphWeightType;
        final double WEIGHTPARAM = options.graphWeightParam;
        final boolean NORMALIZE = options.graphNormalize;
        if (WEIGHTTYPE.equals("inner") && !DISTANCEFUNCTION.equals("cosine")) {
            System.err.println("WEIGHTTYPE and DISTANCEFUNCTION mismatch.");
        }
        final Matrix A = adjacency(X, type, NN, DISTANCEFUNCTION);
        final Matrix W = A.copy();
        final FindResult findResult = Matlab.find(A);
        final int[] A_i = findResult.rows;
        final int[] A_j = findResult.cols;
        final double[] A_v = findResult.vals;
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
        }
        Matrix D = null;
        final Vector V = Matlab.sum(W, 2);
        Matrix L = null;
        if (!NORMALIZE) {
            L = Matlab.diag(V).minus(W);
        }
        else {
            D = Matlab.diag(Matlab.dotDivide(1.0, Matlab.sqrt(V)));
            L = Matlab.speye(Matlab.size(W, 1)).minus(D.mtimes(W).mtimes(D));
        }
        return L;
    }
    
    public static Matrix adjacency(final Matrix X, final String type, final double param, final String distFunc) {
        final Matrix A = adjacencyDirected(X, type, param, distFunc);
        return Matlab.max(A, A.transpose());
    }
    
    public static Matrix adjacencyDirected(final Matrix X, final String type, final double param, final String distFunc) {
        System.out.println("Computing directed adjacency graph...");
        final int n = Matlab.size(X, 2);
        if (type.equals("nn")) {
            System.out.println(String.format("Creating the adjacency matrix. Nearest neighbors, N = %d.", (int)param));
        }
        else if (type.equals("epsballs") || type.equals("eps")) {
            System.out.println(String.format("Creating the adjacency matrix. Epsilon balls, eps = %f.", param));
        }
        else {
            System.err.println("type should be either \"nn\" or \"epsballs\" (\"eps\")");
            System.exit(1);
        }
        final Matrix A = new SparseMatrix(n, n);
        Matrix dt = null;
        for (int i = 0; i < n; ++i) {
            if (distFunc.equals("euclidean")) {
                dt = euclideanByColumns(X.getColumnMatrix(i), X);
            }
            else if (distFunc.equals("cosine")) {
                dt = cosineByColumns(X.getColumnMatrix(i), X);
            }
            final Matrix[] sortResult = Matlab.sort(dt, 2);
            final Matrix Z = sortResult[0];
            final double[][] IX = ((DenseMatrix)sortResult[1]).getData();
            if (type.equals("nn")) {
                for (int j = 0; j <= param; ++j) {
                    if ((int)IX[0][j] != i) {
                        A.setEntry(i, (int)IX[0][j], Z.getEntry(0, j) + Matlab.eps);
                    }
                }
            }
            else if (type.equals("epsballs") || type.equals("eps")) {
                for (int j = 0; Z.getEntry(0, j) <= param; ++j) {
                    if ((int)IX[0][j] != i) {
                        A.setEntry(i, (int)IX[0][j], Z.getEntry(0, j) + Matlab.eps);
                    }
                }
            }
        }
        return A;
    }
    
    public static Matrix cosineByColumns(final Matrix A, final Matrix B) {
        final double[] AA = Matlab.sum(Matlab.times(A, A)).getPr();
        final double[] BB = Matlab.sum(Matlab.times(B, B)).getPr();
        Matrix AB = A.transpose().mtimes(B);
        final int M = AB.getRowDimension();
        final int N = AB.getColumnDimension();
        double v = 0.0;
        if (AB instanceof DenseMatrix) {
            final double[][] resData = ((DenseMatrix)AB).getData();
            double[] resRow = null;
            for (int i = 0; i < M; ++i) {
                resRow = resData[i];
                for (int j = 0; j < N; ++j) {
                    v = resRow[j];
                    resRow[j] = 1.0 - v / Math.sqrt(AA[i] * BB[j]);
                }
            }
        }
        else if (AB instanceof SparseMatrix) {
            final double[] pr = ((SparseMatrix)AB).getPr();
            final int[] ir = ((SparseMatrix)AB).getIr();
            final int[] jc = ((SparseMatrix)AB).getJc();
            for (int j = 0; j < N; ++j) {
                for (int k = jc[j]; k < jc[j + 1]; ++k) {
                    final double[] array = pr;
                    final int n = k;
                    array[n] /= -Math.sqrt(AA[ir[k]] * BB[j]);
                }
            }
            AB = AB.plus(1.0);
        }
        return AB;
    }
    
    public static Matrix euclideanByColumns(final Matrix A, final Matrix B) {
        return Matlab.l2DistanceByColumns(A, B);
    }
    
    public static Matrix calcLLR(final Matrix X, final double NN, final String distFunc, final String kernelType, final double kernelParam, final double lambda) {
        final String type = "nn";
        final Matrix A = adjacencyDirected(X, type, NN, distFunc);
        final Matrix K = KernelByColumns.calcKernel(kernelType, kernelParam, X);
        final int NSample = Matlab.size(X, 2);
        final int n_i = (int)NN;
        final Matrix I_i = Matlab.eye(n_i);
        final Matrix I = Matlab.speye(NSample);
        Matrix G = A.copy();
        int[] neighborIndices_i = null;
        Vector[] neighborhood_X_i = null;
        Matrix K_i = null;
        Matrix k_i = null;
        Vector x_i = null;
        Matrix alpha_i = null;
        final Vector[] Vs = Matlab.sparseMatrix2SparseRowVectors(A);
        Vector[] Xs = null;
        if (X instanceof DenseMatrix) {
            Xs = Matlab.denseMatrix2DenseColumnVectors(X);
        }
        else if (X instanceof SparseMatrix) {
            Xs = Matlab.sparseMatrix2SparseColumnVectors(X);
        }
        final Vector[] Gs = new Vector[NSample];
        for (int i = 0; i < NSample; ++i) {
            neighborIndices_i = Matlab.find(Vs[i]);
            neighborhood_X_i = new Vector[neighborIndices_i.length];
            for (int k = 0; k < neighborIndices_i.length; ++k) {
                neighborhood_X_i[k] = Xs[neighborIndices_i[k]];
            }
            K_i = K.getSubMatrix(neighborIndices_i, neighborIndices_i);
            x_i = Xs[i];
            k_i = KernelByColumns.calcKernel(kernelType, kernelParam, new Vector[] { x_i }, neighborhood_X_i);
            alpha_i = Matlab.mrdivide(k_i, I_i.times(n_i * lambda).plus(K_i));
            Gs[i] = new SparseVector(neighborIndices_i, ((DenseMatrix)alpha_i).getData()[0], neighborIndices_i.length, NSample);
        }
        G = Matlab.sparseRowVectors2SparseMatrix(Gs);
        final Matrix T = G.minus(I);
        final Matrix L = T.transpose().mtimes(T);
        return L;
    }
}
