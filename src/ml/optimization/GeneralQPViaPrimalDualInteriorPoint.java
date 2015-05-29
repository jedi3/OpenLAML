package ml.optimization;

import la.io.*;
import la.io.IO;
import la.matrix.*;
import ml.utils.*;

public class GeneralQPViaPrimalDualInteriorPoint
{
    public static void main(final String[] args) {
        final int n = 5;
        final int m = 6;
        final int p = 3;
        Matrix x = null;
        Matrix Q = null;
        Matrix c = null;
        Matrix A = null;
        Matrix b = null;
        Matrix B = null;
        Matrix d = null;
        final double rou = -2.0;
        final double HasEquality = 1.0;
        final boolean generate = false;
        if (generate) {
            x = Matlab.rand(n, n);
            Q = x.mtimes(x.transpose()).plus(Matlab.times(Matlab.rand(1), Matlab.eye(n)));
            c = Matlab.rand(n, 1);
            A = Matlab.times(HasEquality, Matlab.rand(p, n));
            x = Matlab.rand(n, 1);
            b = A.mtimes(x);
            B = Matlab.rand(m, n);
            d = Matlab.plus(B.mtimes(x), Matlab.times(rou, Matlab.ones(m, 1)));
            IO.saveMatrix("Q", Q);
            IO.saveMatrix("c", c);
            IO.saveMatrix("A", A);
            IO.saveMatrix("b2", b);
            IO.saveMatrix("B", B);
            IO.saveMatrix("d", d);
        }
        else {
            Q = IO.loadMatrix("Q");
            c = IO.loadMatrix("c");
            A = IO.loadMatrix("A");
            b = IO.loadMatrix("b2");
            B = IO.loadMatrix("B");
            d = IO.loadMatrix("d");
        }
        solve(Q, c, A, b, B, d);
    }
    
    public static QPSolution solve(final Matrix Q, final Matrix c, final Matrix A, final Matrix b, final Matrix B, final Matrix d) {
        Printer.fprintf("Phase I:\n\n", new Object[0]);
        final PhaseIResult phaseIResult = GeneralQP.phaseI(A, b, B, d);
        if (phaseIResult.feasible) {
            Printer.fprintf("Phase II:\n\n", new Object[0]);
            final Matrix x0 = phaseIResult.optimizer;
            return phaseII(Q, c, A, b, B, d, x0);
        }
        System.err.println("The QP problem is infeasible!\n");
        return null;
    }
    
    private static QPSolution phaseII(final Matrix Q, final Matrix c, final Matrix A, final Matrix b, final Matrix B, final Matrix d, final Matrix x0) {
        final Matrix x = x0.copy();
        final Matrix l = new DenseMatrix(B.getRowDimension(), 1);
        final Matrix v = new DenseMatrix(A.getRowDimension(), 1);
        final Matrix H_x = null;
        Matrix F_x = null;
        final Matrix DF_x = null;
        Matrix G_f_x = null;
        double fval = 0.0;
        fval = Matlab.innerProduct(x, Q.mtimes(x)) / 2.0 + Matlab.innerProduct(c, x);
        F_x = B.mtimes(x).minus(d);
        G_f_x = Q.mtimes(x).plus(c);
        boolean[] flags = null;
        int k = 0;
        Time.tic();
        while (true) {
            flags = PrimalDualInteriorPoint.run(A, b, Q, F_x, B, G_f_x, fval, x, l, v);
            if (flags[0]) {
                break;
            }
            fval = Matlab.innerProduct(x, Q.mtimes(x)) / 2.0 + Matlab.innerProduct(c, x);
            F_x = B.mtimes(x).minus(d);
            if (!flags[1]) {
                continue;
            }
            ++k;
            G_f_x = Q.mtimes(x).plus(c);
        }
        final double t_primal_dual_interior_point = Time.toc();
        final double fval_primal_dual_interior_point = fval;
        final Matrix x_primal_dual_interior_point = x;
        final Matrix lambda_primal_dual_interior_point = l;
        final Matrix v_primal_dual_interior_point = v;
        Printer.fprintf("Optimal objective function value: %g\n\n", fval_primal_dual_interior_point);
        Printer.fprintf("Optimizer:\n", new Object[0]);
        Printer.disp(x_primal_dual_interior_point.transpose());
        final Matrix e = B.mtimes(x).minus(d);
        Printer.fprintf("B * x - d:\n", new Object[0]);
        Printer.disp(e.transpose());
        Printer.fprintf("lambda:\n", new Object[0]);
        Printer.disp(lambda_primal_dual_interior_point.transpose());
        Printer.fprintf("nu:\n", new Object[0]);
        Printer.disp(v_primal_dual_interior_point.transpose());
        Printer.fprintf("norm(A * x - b, \"fro\"): %f\n\n", Matlab.norm(A.mtimes(x_primal_dual_interior_point).minus(b), "fro"));
        Printer.fprintf("Computation time: %f seconds\n\n", t_primal_dual_interior_point);
        return new QPSolution(x, l, v, fval);
    }
}
