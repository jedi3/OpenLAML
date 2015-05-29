package ml.subspace;

import la.matrix.*;
import ml.manifold.*;
import la.vector.*;
import ml.utils.*;

public class LLE extends DimensionalityReduction
{
    int K;
    
    public static void main(final String[] args) {
        final double[][] data = { { 0.0, 2.0, 3.0, 4.0 }, { 2.0, 0.0, 4.0, 5.0 }, { 3.0, 4.1, 5.0, 6.0 }, { 2.0, 7.0, 1.0, 6.0 } };
        final Matrix X = new DenseMatrix(data).transpose();
        final int K = 3;
        final int r = 3;
        final Matrix R = run(X, K, r);
        Printer.disp("Original Data:");
        Printer.disp(X);
        Printer.disp("Reduced Data:");
        Printer.disp(R);
    }
    
    public LLE(final int r) {
        super(r);
    }
    
    public LLE(final int r, final int K) {
        super(r);
        this.K = K;
    }
    
    @Override
    public void run() {
        this.R = run(this.X, this.K, this.r);
    }
    
    public static Matrix run(final Matrix X, final int K, final int r) {
        final String type = "nn";
        final double param = K;
        final Matrix A = Manifold.adjacencyDirected(X, type, param, "euclidean");
        final int N = Matlab.size(X, 1);
        Matrix X_i = null;
        Matrix C_i = null;
        Matrix C = null;
        Matrix w = null;
        Matrix W = Matlab.gt(A, 0.0);
        Matrix M = null;
        final Matrix Ones = Matlab.ones(K, 1);
        final Matrix OnesT = Matlab.ones(1, K);
        final Matrix I = Matlab.eye(N);
        int[] neighborIndices = null;
        final Vector[] Ws = new Vector[N];
        for (int i = 0; i < N; ++i) {
            neighborIndices = Matlab.find(A.getRowVector(i));
            X_i = X.getRows(neighborIndices);
            C_i = X_i.minus(Ones.mtimes(Matlab.getRows(X, i)));
            C = C_i.mtimes(C_i.transpose());
            C = C.plus(Matlab.diag(Matlab.diag(C)));
            w = Matlab.mrdivide(OnesT, C);
            InPlaceOperator.timesAssign(w, 1.0 / Matlab.sumAll(w));
            Ws[i] = new SparseVector(neighborIndices, ((DenseMatrix)w).getData()[0], neighborIndices.length, N);
        }
        W = Matlab.sparseRowVectors2SparseMatrix(Ws);
        M = I.minus(W);
        M = M.transpose().mtimes(M);
        final Matrix U = Matlab.eigs(M, r + 1, "sm")[0];
        return Matlab.times(Math.sqrt(N), Matlab.getColumns(U, ArrayOperator.colon(1, r)));
    }
}
