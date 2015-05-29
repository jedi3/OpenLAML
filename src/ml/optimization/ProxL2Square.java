package ml.optimization;

import la.matrix.*;
import ml.utils.*;

public class ProxL2Square implements ProximalMapping
{
    private double lambda;
    
    public ProxL2Square(final double lambda) {
        this.lambda = lambda;
    }
    
    @Override
    public Matrix compute(double t, final Matrix X) {
        t *= this.lambda;
        final Matrix res = X.copy();
        InPlaceOperator.times(res, 1.0 / (1.0 + 2.0 * t), X);
        return res;
    }
    
    @Override
    public void compute(final Matrix res, double t, final Matrix X) {
        t *= this.lambda;
        InPlaceOperator.times(res, 1.0 / (1.0 + 2.0 * t), X);
    }
}
