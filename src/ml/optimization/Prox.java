package ml.optimization;

import la.matrix.*;
import ml.utils.*;

public class Prox implements ProximalMapping
{
    @Override
    public Matrix compute(final double t, final Matrix X) {
        return X;
    }
    
    @Override
    public void compute(final Matrix res, final double t, final Matrix X) {
        InPlaceOperator.assign(res, X);
    }
}
