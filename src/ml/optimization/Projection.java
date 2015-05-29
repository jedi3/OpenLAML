package ml.optimization;

import la.matrix.*;

public interface Projection
{
    Matrix compute(double p0, Matrix p1);
    
    void compute(Matrix p0, double p1, Matrix p2);
}
