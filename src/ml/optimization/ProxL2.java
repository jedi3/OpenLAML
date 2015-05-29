package ml.optimization;

import la.matrix.*;
import ml.utils.*;

public class ProxL2 implements ProximalMapping
{
    private double lambda;
    
    public ProxL2(final double lambda) {
        this.lambda = lambda;
    }
    
    @Override
    public Matrix compute(double t, final Matrix X) {
        t *= this.lambda;
        final double norm = Matlab.norm(X, "fro");
        if (norm <= t) {
            return Matlab.zeros(Matlab.size(X));
        }
        final Matrix res = X.copy();
        InPlaceOperator.times(res, 1.0 - t / norm, X);
        return res;
    }
    
    @Override
    public void compute(final Matrix res, double t, final Matrix X) {
        t *= this.lambda;
        final double norm = Matlab.norm(X, "fro");
        if (norm <= t) {
            res.clear();
        }
        else {
            InPlaceOperator.times(res, 1.0 - t / norm, X);
        }
    }
}
