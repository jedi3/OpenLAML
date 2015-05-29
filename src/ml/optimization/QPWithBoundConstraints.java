package ml.optimization;

import la.matrix.*;
import ml.utils.*;

public class QPWithBoundConstraints
{
    public static void main(final String[] args) {
        final int n = 5;
        final Matrix x = Matlab.rand(n);
        final Matrix Q = Matlab.minus(x.mtimes(x.transpose()), Matlab.times(Matlab.rand(1).getEntry(0, 0), Matlab.eye(n)));
        final Matrix c = Matlab.plus(-2.0, Matlab.times(2.0, Matlab.rand(n, 1)));
        final double l = 0.0;
        final double u = 1.0;
        final double epsilon = 1.0E-6;
        final QPSolution S = solve(Q, c, l, u, epsilon);
        Printer.disp("Q:");
        Printer.disp(Q);
        Printer.disp("c:");
        Printer.disp(c);
        Printer.fprintf("Optimum: %g\n", S.optimum);
        Printer.fprintf("Optimizer:\n", new Object[0]);
        Printer.display(S.optimizer.transpose());
    }
    
    public static QPSolution solve(final Matrix Q, final Matrix c, final double l, final double u, final double epsilon) {
        return solve(Q, c, l, u, epsilon, null);
    }
    
    public static QPSolution solve(final Matrix Q, final Matrix c, final double l, final double u, final double epsilon, final Matrix x0) {
        final int d = Q.getColumnDimension();
        double fval = 0.0;
        Matrix x = null;
        if (x0 != null) {
            x = x0;
        }
        else {
            x = Matlab.plus((l + u) / 2.0, Matlab.zeros(d, 1));
        }
        Matrix Grad = Q.mtimes(x).plus(c);
        fval = Matlab.innerProduct(x, Q.mtimes(x)) / 2.0 + Matlab.innerProduct(c, x);
        boolean[] flags = null;
        while (true) {
            flags = BoundConstrainedPLBFGS.run(Grad, fval, l, u, epsilon, x);
            if (flags[0]) {
                break;
            }
            fval = Matlab.innerProduct(x, Q.mtimes(x)) / 2.0 + Matlab.innerProduct(c, x);
            if (!flags[1]) {
                continue;
            }
            Grad = Q.mtimes(x).plus(c);
        }
        return new QPSolution(x, null, null, fval);
    }
}
