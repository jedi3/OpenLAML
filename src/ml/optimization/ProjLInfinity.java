package ml.optimization;

import la.matrix.*;
import ml.utils.*;

public class ProjLInfinity implements Projection
{
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
        return Matlab.times(Matlab.sign(X), Matlab.min(Matlab.abs(X), t));
    }
    
    @Override
    public void compute(final Matrix res, final double t, final Matrix X) {
        if (t < 0.0) {
            System.err.println("The first input should be a nonnegative real scalar.");
            System.exit(-1);
        }
        if (X.getColumnDimension() > 1) {
            System.err.println("The second input should be a vector.");
            System.exit(-1);
        }
        InPlaceOperator.times(res, Matlab.sign(X), Matlab.min(Matlab.abs(X), t));
    }
}
