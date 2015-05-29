package ml.subspace;

import la.matrix.*;
import ml.utils.*;

public class MDS extends DimensionalityReduction
{
    public static void main(final String[] args) {
        final double[][] data = { { 0.0, 2.0, 3.0, 4.0 }, { 2.0, 0.0, 4.0, 5.0 }, { 3.0, 4.1, 5.0, 6.0 }, { 2.0, 7.0, 1.0, 6.0 } };
        final Matrix O = new DenseMatrix(data).transpose();
        final Matrix D = Matlab.l2Distance(O, O);
        final Matrix X = run(D, 3);
        Printer.disp("Reduced X:");
        Printer.disp(X);
    }
    
    public MDS(final int r) {
        super(r);
    }
    
    @Override
    public void run() {
    }
    
    public static Matrix run(final Matrix D, final int p) {
        if (Matlab.norm(D.minus(D.transpose())) > 1.0E-12) {
            System.err.println("The dissimilarity matrix should be symmetric!");
            System.exit(1);
        }
        final int n = D.getColumnDimension();
        final Matrix A = Matlab.times(-0.5, Matlab.times(D, D));
        final Matrix H = Matlab.eye(n).minus(Matlab.rdivide(Matlab.ones(n), n));
        Matrix B = H.mtimes(A).mtimes(H);
        B = Matlab.rdivide(Matlab.plus(B, B.transpose()), 2.0);
        Matrix[] eigRes;
        int k;
        for (eigRes = Matlab.eigs(B, n, "lm"), k = 0, k = p - 1; k >= 0 && eigRes[1].getEntry(k, k) <= 0.0; --k) {}
        final double[][] eigData = ((DenseMatrix)eigRes[0]).getData();
        final double[][] resData = ArrayOperator.allocate2DArray(n, k + 1);
        double[] resRow = null;
        final double[] s = new double[k + 1];
        for (int j = 0; j <= k; ++j) {
            s[j] = Math.sqrt(eigRes[1].getEntry(j, j));
        }
        for (int i = 0; i < n; ++i) {
            resRow = resData[i];
            System.arraycopy(eigData[i], 0, resRow, 0, k + 1);
            ArrayOperator.timesAssign(resRow, s);
        }
        return new DenseMatrix(resData);
    }
}
