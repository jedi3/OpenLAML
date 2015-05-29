package ml.optimization;

import la.matrix.*;
import ml.utils.*;

public class ProxPlus implements ProximalMapping
{
    @Override
    public Matrix compute(final double t, final Matrix X) {
        return Matlab.subplus(X);
    }
    
    @Override
    public void compute(final Matrix res, final double t, final Matrix X) {
        InPlaceOperator.subplus(res, X);
    }
}
