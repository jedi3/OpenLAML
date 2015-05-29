package ml.optimization;

import la.matrix.*;
import la.io.*;
import la.io.IO;
import ml.utils.*;

public class GeneralQP
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
        final PhaseIResult phaseIResult = phaseI(A, b, B, d);
        if (phaseIResult.feasible) {
            Printer.fprintf("Phase II:\n\n", new Object[0]);
            final Matrix x0 = phaseIResult.optimizer;
            return phaseII(Q, c, A, b, B, d, x0);
        }
        System.err.println("The QP problem is infeasible!\n");
        return null;
    }
    
    public static PhaseIResult phaseI(Matrix A, final Matrix b, Matrix B, Matrix d) {
        int n = A.getColumnDimension();
        final int p = A.getRowDimension();
        int m = B.getRowDimension();
        final Matrix A_ori = A;
        final Matrix B_ori = B;
        final Matrix d_ori = d;
        final Matrix c = Matlab.vertcat(Matlab.zeros(n, 1), Matlab.ones(m, 1));
        A = Matlab.horzcat(A, Matlab.zeros(p, m));
        B = Matlab.vertcat(Matlab.horzcat(B, Matlab.uminus(Matlab.eye(m))), Matlab.horzcat(Matlab.zeros(m, n), Matlab.uminus(Matlab.eye(m))));
        d = Matlab.vertcat(d, Matlab.zeros(m, 1));
        final int n_ori = n;
        final int m_ori = m;
        n += m;
        m *= 2;
        Matrix x0 = Matlab.ones(n_ori, 1);
        final Matrix s0 = B_ori.mtimes(x0).minus(d_ori).plus(Matlab.ones(m_ori, 1));
        x0 = Matlab.vertcat(x0, s0);
        final Matrix v0 = Matlab.zeros(p, 1);
        final double mu = 1.8;
        final double epsilon = 1.0E-10;
        final double epsilon_feas = 1.0E-10;
        final double alpha = 0.1;
        final double beta = 0.98;
        Time.tic();
        final Matrix l0 = Matlab.rdivide(Matlab.ones(m, 1), m);
        final Matrix x = x0;
        final Matrix i = l0;
        final Matrix v = v0;
        final Matrix F_x_0 = B.mtimes(x).minus(d);
        double eta_t = -Matlab.innerProduct(F_x_0, l0);
        double t = 1.0;
        double f_x = 0.0;
        Matrix G_f_x = null;
        Matrix F_x = null;
        Matrix DF_x = null;
        final Matrix H_x = Matlab.times(1.0E-10, Matlab.eye(n));
        Matrix r_prim = null;
        Matrix r_dual = null;
        Matrix r_cent = null;
        Matrix Matrix = null;
        Matrix Vector = null;
        double residual = 0.0;
        double residual_prim = 0.0;
        double residual_dual = 0.0;
        Matrix z_pd = null;
        Matrix x_nt = null;
        Matrix l_nt = null;
        Matrix v_nt = null;
        final Matrix x_s = Matlab.zeros(Matlab.size(x0));
        final Matrix l_s = Matlab.zeros(Matlab.size(l0));
        final Matrix v_s = Matlab.zeros(Matlab.size(v0));
        double s = 0.0;
        Matrix G_f_x_s = null;
        Matrix F_x_s = null;
        Matrix DF_x_s = null;
        Matrix r_prim_s = null;
        Matrix r_dual_s = null;
        Matrix r_cent_s = null;
        double residual_s = 0.0;
        while (true) {
            t = mu * m / eta_t;
            f_x = Matlab.innerProduct(c, x);
            G_f_x = c;
            F_x = B.mtimes(x).minus(d);
            DF_x = B;
            r_prim = A.mtimes(x).minus(b);
            r_dual = G_f_x.plus(DF_x.transpose().mtimes(i)).plus(A.transpose().mtimes(v));
            r_cent = Matlab.uminus(Matlab.times(i, F_x)).minus(Matlab.rdivide(Matlab.ones(m, 1), t));
            Matrix = Matlab.vertcat(Matlab.horzcat(H_x, DF_x.transpose(), A.transpose()), Matlab.horzcat(Matlab.uminus(Matlab.mtimes(Matlab.diag(i), DF_x)), Matlab.uminus(Matlab.diag(F_x)), Matlab.zeros(m, p)), Matlab.horzcat(A, Matlab.zeros(p, m), Matlab.zeros(p, p)));
            Vector = Matlab.uminus(Matlab.vertcat(r_dual, r_cent, r_prim));
            residual = Matlab.norm(Vector);
            residual_prim = Matlab.norm(r_prim);
            residual_dual = Matlab.norm(r_dual);
            eta_t = -Matlab.innerProduct(F_x, i);
            if (residual_prim <= epsilon_feas && residual_dual <= epsilon_feas && eta_t <= epsilon) {
                break;
            }
            z_pd = Matlab.mldivide(Matrix, Vector);
            x_nt = Matlab.getRows(z_pd, 0, n - 1);
            l_nt = Matlab.getRows(z_pd, n, n + m - 1);
            v_nt = Matlab.getRows(z_pd, n + m, n + m + p - 1);
            s = 1.0;
            while (true) {
                InPlaceOperator.affine(l_s, s, l_nt, '+', i);
                if (Matlab.sumAll(Matlab.lt(l_s, 0.0)) <= 0.0) {
                    break;
                }
                s *= beta;
            }
            while (true) {
                InPlaceOperator.affine(x_s, s, x_nt, '+', x);
                if (Matlab.sumAll(Matlab.lt(d.minus(B.mtimes(x_s)), 0.0)) <= 0.0) {
                    break;
                }
                s *= beta;
            }
            while (true) {
                InPlaceOperator.affine(x_s, s, x_nt, '+', x);
                InPlaceOperator.affine(l_s, s, l_nt, '+', i);
                InPlaceOperator.affine(v_s, s, v_nt, '+', v);
                G_f_x_s = c;
                F_x_s = B.mtimes(x_s).minus(d);
                DF_x_s = B;
                r_prim_s = A.mtimes(x_s).minus(b);
                r_dual_s = G_f_x_s.plus(DF_x_s.transpose().mtimes(l_s)).plus(A.transpose().mtimes(v_s));
                r_cent_s = Matlab.uminus(Matlab.times(l_s, F_x_s)).minus(Matlab.rdivide(Matlab.ones(m, 1), t));
                residual_s = Matlab.norm(Matlab.vertcat(r_dual_s, r_cent_s, r_prim_s));
                if (residual_s <= (1.0 - alpha * s) * residual) {
                    break;
                }
                s *= beta;
            }
            InPlaceOperator.assign(x, x_s);
            InPlaceOperator.assign(i, l_s);
            InPlaceOperator.assign(v, v_s);
        }
        Printer.fprintf("Terminate successfully.\n\n", new Object[0]);
        final double t_sum_of_inequalities = Time.toc();
        final Matrix x_opt = Matlab.getRows(x, 0, n_ori - 1);
        Printer.fprintf("x_opt:\n", new Object[0]);
        Printer.disp(x_opt.transpose());
        final Matrix s_opt = Matlab.getRows(x, n_ori, n - 1);
        Printer.fprintf("s_opt:\n", new Object[0]);
        Printer.disp(s_opt.transpose());
        final Matrix lambda_s = Matlab.getRows(i, m_ori, m - 1);
        Printer.fprintf("lambda for the inequalities s_i >= 0:\n", new Object[0]);
        Printer.disp(lambda_s.transpose());
        final Matrix e = B_ori.mtimes(x_opt).minus(d_ori);
        Printer.fprintf("B * x - d:\n", new Object[0]);
        Printer.disp(e.transpose());
        final Matrix lambda_ineq = Matlab.getRows(i, 0, m_ori - 1);
        Printer.fprintf("lambda for the inequalities fi(x) <= s_i:\n", new Object[0]);
        Printer.disp(lambda_ineq.transpose());
        final Matrix v_opt = v;
        Printer.fprintf("nu for the equalities A * x = b:\n", new Object[0]);
        Printer.disp(v_opt.transpose());
        Printer.fprintf("residual: %g\n\n", residual);
        Printer.fprintf("A * x - b:\n", new Object[0]);
        Printer.disp(A_ori.mtimes(x_opt).minus(b).transpose());
        Printer.fprintf("norm(A * x - b, \"fro\"): %f\n\n", Matlab.norm(A_ori.mtimes(x_opt).minus(b), "fro"));
        final double fval_opt = f_x;
        Printer.fprintf("fval_opt: %g\n\n", fval_opt);
        boolean feasible = false;
        if (fval_opt <= epsilon) {
            feasible = true;
            Printer.fprintf("The problem is feasible.\n\n", new Object[0]);
        }
        else {
            feasible = false;
            Printer.fprintf("The problem is infeasible.\n\n", new Object[0]);
        }
        Printer.fprintf("Computation time: %f seconds\n\n", t_sum_of_inequalities);
        x0 = x_opt;
        final int pause_time = 1;
        Printer.fprintf("halt execution temporarily in %d seconds...\n\n", pause_time);
        Time.pause(pause_time);
        return new PhaseIResult(feasible, x_opt, fval_opt);
    }
    
    public static QPSolution phaseII(final Matrix Q, final Matrix c, final Matrix A, final Matrix b, final Matrix B, final Matrix d, final Matrix x0) {
        final int n = A.getColumnDimension();
        final int p = A.getRowDimension();
        final int m = B.getRowDimension();
        final Matrix v0 = Matlab.zeros(p, 1);
        final double mu = 1.8;
        final double epsilon = 1.0E-10;
        final double epsilon_feas = 1.0E-10;
        final double alpha = 0.1;
        final double beta = 0.98;
        Time.tic();
        final Matrix i;
        final Matrix l0 = i = Matlab.rdivide(Matlab.ones(m, 1), m);
        final Matrix v = v0;
        final Matrix F_x_0 = B.mtimes(x0).minus(d);
        double eta_t = -Matlab.innerProduct(F_x_0, l0);
        double t = 1.0;
        double f_x = 0.0;
        Matrix G_f_x = null;
        Matrix F_x = null;
        final Matrix DF_x = null;
        Matrix r_prim = null;
        Matrix r_dual = null;
        Matrix r_cent = null;
        Matrix Matrix = null;
        Matrix Vector = null;
        double residual = 0.0;
        double residual_prim = 0.0;
        double residual_dual = 0.0;
        Matrix z_pd = null;
        Matrix x_nt = null;
        Matrix l_nt = null;
        Matrix v_nt = null;
        final Matrix x_s = Matlab.zeros(Matlab.size(x0));
        final Matrix l_s = Matlab.zeros(Matlab.size(l0));
        final Matrix v_s = Matlab.zeros(Matlab.size(v0));
        double s = 0.0;
        Matrix G_f_x_s = null;
        Matrix F_x_s = null;
        final Matrix DF_x_s = null;
        Matrix r_prim_s = null;
        Matrix r_dual_s = null;
        Matrix r_cent_s = null;
        double residual_s = 0.0;
        while (true) {
            t = mu * m / eta_t;
            f_x = Matlab.innerProduct(x0, Q.mtimes(x0)) / 2.0 + Matlab.innerProduct(c, x0);
            G_f_x = Q.mtimes(x0).plus(c);
            F_x = B.mtimes(x0).minus(d);
            r_prim = A.mtimes(x0).minus(b);
            r_dual = G_f_x.plus(B.transpose().mtimes(i)).plus(A.transpose().mtimes(v));
            r_cent = Matlab.uminus(Matlab.times(i, F_x)).minus(Matlab.rdivide(Matlab.ones(m, 1), t));
            Matrix = Matlab.vertcat(Matlab.horzcat(Q, B.transpose(), A.transpose()), Matlab.horzcat(Matlab.uminus(Matlab.mtimes(Matlab.diag(i), B)), Matlab.uminus(Matlab.diag(F_x)), Matlab.zeros(m, p)), Matlab.horzcat(A, Matlab.zeros(p, m), Matlab.zeros(p, p)));
            Vector = Matlab.uminus(Matlab.vertcat(r_dual, r_cent, r_prim));
            residual = Matlab.norm(Vector);
            residual_prim = Matlab.norm(r_prim);
            residual_dual = Matlab.norm(r_dual);
            eta_t = -Matlab.innerProduct(F_x, i);
            if (residual_prim <= epsilon_feas && residual_dual <= epsilon_feas && eta_t <= epsilon) {
                break;
            }
            z_pd = Matlab.mldivide(Matrix, Vector);
            x_nt = Matlab.getRows(z_pd, 0, n - 1);
            l_nt = Matlab.getRows(z_pd, n, n + m - 1);
            v_nt = Matlab.getRows(z_pd, n + m, n + m + p - 1);
            s = 1.0;
            while (true) {
                InPlaceOperator.affine(l_s, s, l_nt, '+', i);
                if (Matlab.sumAll(Matlab.lt(l_s, 0.0)) <= 0.0) {
                    break;
                }
                s *= beta;
            }
            while (true) {
                InPlaceOperator.affine(x_s, s, x_nt, '+', x0);
                if (Matlab.sumAll(Matlab.lt(d.minus(B.mtimes(x_s)), 0.0)) <= 0.0) {
                    break;
                }
                s *= beta;
            }
            while (true) {
                InPlaceOperator.affine(x_s, s, x_nt, '+', x0);
                InPlaceOperator.affine(l_s, s, l_nt, '+', i);
                InPlaceOperator.affine(v_s, s, v_nt, '+', v);
                G_f_x_s = Q.mtimes(x_s).plus(c);
                F_x_s = B.mtimes(x_s).minus(d);
                r_prim_s = A.mtimes(x_s).minus(b);
                r_dual_s = G_f_x_s.plus(B.transpose().mtimes(l_s)).plus(A.transpose().mtimes(v_s));
                r_cent_s = Matlab.uminus(Matlab.times(l_s, F_x_s)).minus(Matlab.rdivide(Matlab.ones(m, 1), t));
                residual_s = Matlab.norm(Matlab.vertcat(r_dual_s, r_cent_s, r_prim_s));
                if (residual_s <= (1.0 - alpha * s) * residual) {
                    break;
                }
                s *= beta;
            }
            InPlaceOperator.assign(x0, x_s);
            InPlaceOperator.assign(i, l_s);
            InPlaceOperator.assign(v, v_s);
        }
        Printer.fprintf("Terminate successfully.\n\n", new Object[0]);
        final double t_primal_dual_interior_point = Time.toc();
        final double fval_primal_dual_interior_point = f_x;
        final Matrix lambda_primal_dual_interior_point = i;
        final Matrix v_primal_dual_interior_point = v;
        Printer.fprintf("residual: %g\n\n", residual);
        Printer.fprintf("Optimal objective function value: %g\n\n", fval_primal_dual_interior_point);
        Printer.fprintf("Optimizer:\n", new Object[0]);
        Printer.disp(x0.transpose());
        final Matrix e = B.mtimes(x0).minus(d);
        Printer.fprintf("B * x - d:\n", new Object[0]);
        Printer.disp(e.transpose());
        Printer.fprintf("lambda:\n", new Object[0]);
        Printer.disp(lambda_primal_dual_interior_point.transpose());
        Printer.fprintf("nu:\n", new Object[0]);
        Printer.disp(v_primal_dual_interior_point.transpose());
        Printer.fprintf("norm(A * x - b, \"fro\"): %f\n\n", Matlab.norm(A.mtimes(x0).minus(b), "fro"));
        Printer.fprintf("Computation time: %f seconds\n\n", t_primal_dual_interior_point);
        return new QPSolution(x0, i, v, f_x);
    }
}
