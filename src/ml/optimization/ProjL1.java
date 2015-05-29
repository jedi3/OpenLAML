package ml.optimization;

import la.matrix.*;
import la.vector.*;
import ml.utils.*;

public class ProjL1 implements Projection
{
    public static void main(final String[] args) {
        final double[][] data = { { -3.5 }, { 2.4 }, { 1.2 }, { -0.9 } };
        final Matrix X = new DenseMatrix(data);
        final double t = 1.5;
        Printer.display(new ProjL1().compute(t, X));
    }
    
    @Override
    public Matrix compute(final double t, final Matrix X) {
        if (t < 0.0) {
            System.err.println("The first input should be a nonnegative real scalar.");
            System.exit(-1);
        }
        if (X.getColumnDimension() > 1) {
            System.err.println("The second input should be a vector.");
            System.exit(-1);
        }
        Matrix res = X.copy();
        final Matrix U = X.copy();
        Matrix V = Matlab.abs(X);
        if (Matlab.sum(Matlab.sum(V)) <= t) {
            res = Matlab.zeros(Matlab.size(V));
        }
        final int d = Matlab.size(X)[0];
        V = Matlab.sort(V)[0];
        final double[] Delta = new double[d - 1];
        for (int k = 0; k < d - 1; ++k) {
            Delta[k] = V.getEntry(k + 1, 0) - V.getEntry(k, 0);
        }
        final double[] S = ArrayOperator.times(Delta, ArrayOperator.colon(d - 1, -1.0, 1.0));
        double a = V.getEntry(d - 1, 0);
        double n = 1.0;
        double sum = S[d - 2];
        for (int j = d - 1; j >= 1 && sum < t; --j) {
            if (j > 1) {
                sum += S[j - 2];
            }
            a += V.getEntry(j - 1, 0);
            ++n;
        }
        final double alpha = (a - t) / n;
        V = U;
        InPlaceOperator.minus(res, Matlab.abs(V), alpha);
        InPlaceOperator.subplusAssign(res);
        InPlaceOperator.timesAssign(res, Matlab.sign(V));
        return res;
    }
    
    @Override
    public void compute(Matrix res, final double t, final Matrix X) {
        if (t < 0.0) {
            System.err.println("The first input should be a nonnegative real scalar.");
            System.exit(-1);
        }
        if (X.getColumnDimension() > 1) {
            System.err.println("The second input should be a vector.");
            System.exit(-1);
        }
        final Matrix U = X.copy();
        Matrix V = Matlab.abs(X);
        if (Matlab.sum(Matlab.sum(V)) <= t) {
            res = Matlab.zeros(Matlab.size(V));
        }
        final int d = Matlab.size(X)[0];
        V = Matlab.sort(V)[0];
        final double[] Delta = new double[d - 1];
        for (int k = 0; k < d - 1; ++k) {
            Delta[k] = V.getEntry(k + 1, 0) - V.getEntry(k, 0);
        }
        final double[] S = ArrayOperator.times(Delta, ArrayOperator.colon(d - 1, -1.0, 1.0));
        double a = V.getEntry(d - 1, 0);
        double n = 1.0;
        double sum = S[d - 2];
        for (int j = d - 1; j >= 1 && sum < t; --j) {
            if (j > 1) {
                sum += S[j - 2];
            }
            a += V.getEntry(j - 1, 0);
            ++n;
        }
        final double alpha = (a - t) / n;
        V = U;
        InPlaceOperator.minus(res, Matlab.abs(V), alpha);
        InPlaceOperator.subplusAssign(res);
        InPlaceOperator.timesAssign(res, Matlab.sign(V));
    }
}
