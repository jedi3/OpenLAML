package ml.optimization;

import la.matrix.*;

public class QPSolution
{
    public Matrix optimizer;
    public Matrix lambda_opt;
    public Matrix nu_opt;
    public double optimum;
    
    public QPSolution(final Matrix optimizer, final Matrix lambda_opt, final Matrix nu_opt, final double optimum) {
        this.optimizer = optimizer;
        this.lambda_opt = lambda_opt;
        this.nu_opt = nu_opt;
        this.optimum = optimum;
    }
}
