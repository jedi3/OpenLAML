package la.vector;

import la.matrix.*;

public interface Vector
{
    int getDim();
    
    Vector copy();
    
    Vector times(Vector p0);
    
    Vector times(double p0);
    
    Vector plus(Vector p0);
    
    Vector minus(Vector p0);
    
    double get(int p0);
    
    void set(int p0, double p1);
    
    Vector operate(Matrix p0);
    
    void clear();
}
