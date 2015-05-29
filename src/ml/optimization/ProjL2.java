package ml.optimization;

import la.matrix.*;
import ml.utils.*;

public class ProjL2 implements Projection
{
    @Override
    public Matrix compute(final double t, final Matrix X) {
        final double norm = Matlab.norm(X, "fro");
        if (norm <= t) {
            return X;
        }
        return Matlab.times(t / norm, X);
    }
    
    @Override
    public void compute(final Matrix res, final double t, final Matrix X) {
        final double norm = Matlab.norm(X, "fro");
        if (norm <= t) {
            InPlaceOperator.assign(res, X);
        }
        else {
            InPlaceOperator.times(res, t / norm, X);
        }
    }
}
