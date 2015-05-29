package ml.random;

import la.matrix.*;
import ml.utils.*;
import java.util.*;

public class MultivariateGaussianDistribution
{
    public static void main(final String[] args) {
        final int n = 10;
        final int d = 2;
        final Matrix t = Matlab.rand(d);
        Matrix SIGMA = Matlab.plus(t.mtimes(t.transpose()), Matlab.times(Matlab.diag(Matlab.rand(d, 1)), Matlab.eye(d)));
        final double theta = Matlab.rand(1).getEntry(0, 0) * 3.141592653589793;
        final Matrix P = new DenseMatrix(new double[][] { { Math.cos(theta), -Math.sin(theta) }, { Math.sin(theta), Math.cos(theta) } });
        SIGMA = P.mtimes(SIGMA).mtimes(P.transpose());
        final Matrix MU = Matlab.times(3.0, Matlab.rand(1, d));
        final Matrix X = mvnrnd(MU, SIGMA, n);
        Printer.disp(X);
    }
    
    public static Matrix mvnrnd(final Matrix MU, final Matrix SIGMA, final int cases) {
        final int d = MU.getColumnDimension();
        if (MU.getRowDimension() != 1) {
            System.err.printf("MU is expected to be 1 x %d matrix!\n", d);
        }
        if (Matlab.norm(SIGMA.transpose().minus(SIGMA)) > 1.0E-10) {
            System.err.printf("SIGMA should be a %d x %d real symmetric matrix!\n", d);
        }
        final Matrix[] eigenDecompostion = Matlab.eigs(SIGMA, d, "lm");
        final Matrix B = eigenDecompostion[0];
        final Matrix Lambda = eigenDecompostion[1];
        final Matrix X = new DenseMatrix(d, cases);
        final Random generator = new Random();
        double sigma = 0.0;
        for (int i = 0; i < d; ++i) {
            sigma = Lambda.getEntry(i, i);
            if (sigma == 0.0) {
                X.setRowMatrix(i, Matlab.zeros(1, cases));
            }
            else {
                if (sigma < 0.0) {
                    System.err.printf("Covariance matrix should be positive semi-definite!\n", new Object[0]);
                    System.exit(1);
                }
                for (int n = 0; n < cases; ++n) {
                    X.setEntry(i, n, generator.nextGaussian() * Math.pow(sigma, 0.5));
                }
            }
        }
        final Matrix Y = Matlab.plus(Matlab.mtimes(B, X), Matlab.repmat(MU.transpose(), 1, cases)).transpose();
        return Y;
    }
    
    public static Matrix mvnrnd(final double[] MU, final double[][] SIGMA, final int cases) {
        return mvnrnd(new DenseMatrix(MU, 2), new DenseMatrix(SIGMA), cases);
    }
    
    public static Matrix mvnrnd(final double[] MU, final double[] SIGMA, final int cases) {
        return mvnrnd(new DenseMatrix(MU, 2), Matlab.diag(SIGMA), cases);
    }
}
