package ml.subspace;

import la.matrix.*;
import ml.utils.*;

public class PCA extends DimensionalityReduction
{
    public static void main(final String[] args) {
        final double[][] data = { { 0.0, 2.0, 3.0, 4.0 }, { 2.0, 0.0, 4.0, 5.0 }, { 3.0, 4.1, 5.0, 6.0 }, { 2.0, 7.0, 1.0, 6.0 } };
        final Matrix X = new DenseMatrix(data).transpose();
        final int r = 3;
        final Matrix R = run(X, r);
        Printer.disp("Original Data:");
        Printer.disp(X);
        Printer.disp("Reduced Data:");
        Printer.disp(R);
    }
    
    public PCA(final int r) {
        super(r);
    }
    
    @Override
    public void run() {
        this.R = run(this.X, this.r);
    }
    
    public static Matrix run(Matrix X, final int r) {
        final int n = Matlab.size(X, 1);
        final double[] S = Matlab.sum(X).getPr();
        ArrayOperator.divideAssign(S, n);
        X = X.copy();
        final int d = X.getColumnDimension();
        double s = 0.0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < d; ++j) {
                s = S[j];
                if (s != 0.0) {
                    X.setEntry(i, j, X.getEntry(i, j) - s);
                }
            }
        }
        final Matrix XT = X.transpose();
        final Matrix Psi = XT.mtimes(X);
        return X.mtimes(Matlab.eigs(Psi, r, "lm")[0]);
    }
}
