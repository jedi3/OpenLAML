package ml.optimization;

import la.matrix.*;

public class PhaseIResult
{
    public boolean feasible;
    public Matrix optimizer;
    public double optimum;
    
    public PhaseIResult(final Matrix optimizer, final double optimum) {
        this.optimizer = optimizer;
        this.optimum = optimum;
    }
    
    public PhaseIResult(final boolean feasible, final Matrix optimizer, final double optimum) {
        this.feasible = feasible;
        this.optimizer = optimizer;
        this.optimum = optimum;
    }
}
