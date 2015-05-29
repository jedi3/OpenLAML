package ml.subspace;

import la.matrix.*;
import ml.manifold.*;
import ml.utils.*;

public class Isomap extends DimensionalityReduction
{
    int K;
    
    public static void main(final String[] args) {
        final double[][] data = { { 0.0, 2.0, 3.0, 4.0 }, { 2.0, 0.0, 4.0, 5.0 }, { 3.0, 4.1, 5.0, 6.0 }, { 2.0, 7.0, 1.0, 6.0 } };
        Matrix X = new DenseMatrix(data);
        X = X.transpose();
        final int K = 3;
        final int r = 3;
        final Matrix R = run(X, K, r);
        Printer.disp("Original Data:");
        Printer.disp(X);
        Printer.disp("Reduced Data:");
        Printer.disp(R);
    }
    
    public Isomap(final int r) {
        super(r);
    }
    
    public Isomap(final int r, final int K) {
        super(r);
        this.K = K;
    }
    
    @Override
    public void run() {
        this.R = run(this.X, this.K, this.r);
    }
    
    public static Matrix run(final Matrix X, final int K, final int r) {
        final Matrix D = Manifold.adjacency(X, "nn", K, "euclidean");
        Matlab.logicalIndexingAssignment(D, Matlab.eq(D, 0.0), Double.POSITIVE_INFINITY);
        Matlab.logicalIndexingAssignment(D, Matlab.speye(Matlab.size(D)), 0.0);
        for (int d = Matlab.size(D, 1), k = 0; k < d; ++k) {
            for (int i = 0; i < d; ++i) {
                for (int j = 0; j < d; ++j) {
                    D.setEntry(i, j, Math.min(D.getEntry(i, j), D.getEntry(i, k) + D.getEntry(k, j)));
                }
            }
        }
        return MDS.run(D, r);
    }
}
