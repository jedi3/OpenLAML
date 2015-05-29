package ml.subspace;

import la.matrix.*;
import ml.kernel.*;
import ml.utils.*;

public class KernelPCA extends DimensionalityReduction
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
    
    public KernelPCA(final int r) {
        super(r);
    }
    
    @Override
    public void run() {
        this.R = run(this.X, this.r);
    }
    
    public static Matrix run(final Matrix X, final int r) {
        final int N = Matlab.size(X, 1);
        final Matrix H = Matlab.eye(N).minus(Matlab.rdivide(Matlab.ones(N, N), N));
        final double sigma = 1.0;
        final Matrix K = Kernel.calcKernel("rbf", sigma, X);
        final Matrix Psi = H.mtimes(K).mtimes(H);
        final Matrix[] UD = Matlab.eigs(Psi, r, "lm");
        final Matrix U = UD[0];
        final Matrix D = UD[1];
        final double[] s = new double[r];
        for (int j = 0; j < r; ++j) {
            s[j] = 1.0 / Math.sqrt(D.getEntry(j, j));
        }
        final double[][] eigData = ((DenseMatrix)U).getData();
        for (int i = 0; i < N; ++i) {
            ArrayOperator.timesAssign(eigData[i], s);
        }
        return K.mtimes(U);
    }
}
