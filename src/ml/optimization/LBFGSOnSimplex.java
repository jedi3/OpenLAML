package ml.optimization;

import java.io.*;

import la.io.*;
import la.io.IO;
import la.vector.*;
import ml.utils.*;
import la.matrix.*;

import java.util.*;

public class LBFGSOnSimplex
{
    private static Matrix G;
    private static Matrix G_pre;
    private static Matrix X;
    private static Matrix X_pre;
    private static double fval;
    private static boolean gradientRequired;
    private static boolean converge;
    private static int state;
    private static double t;
    private static int k;
    private static double alpha;
    private static double beta;
    private static int m;
    private static double H;
    private static Matrix s_k;
    private static Matrix y_k;
    private static double rou_k;
    private static LinkedList<Matrix> s_ks;
    private static LinkedList<Matrix> y_ks;
    private static LinkedList<Double> rou_ks;
    private static Matrix z;
    private static Matrix z_t;
    private static Matrix p_z;
    private static Matrix I_z;
    private static Matrix G_z;
    private static Matrix PG_z;
    private static int i;
    private static double tol;
    private static ArrayList<Double> J;
    
    static {
        LBFGSOnSimplex.G = null;
        LBFGSOnSimplex.G_pre = null;
        LBFGSOnSimplex.X = null;
        LBFGSOnSimplex.X_pre = null;
        LBFGSOnSimplex.fval = 0.0;
        LBFGSOnSimplex.gradientRequired = false;
        LBFGSOnSimplex.converge = false;
        LBFGSOnSimplex.state = 0;
        LBFGSOnSimplex.t = 1.0;
        LBFGSOnSimplex.k = 0;
        LBFGSOnSimplex.alpha = 0.2;
        LBFGSOnSimplex.beta = 0.75;
        LBFGSOnSimplex.m = 30;
        LBFGSOnSimplex.H = 0.0;
        LBFGSOnSimplex.s_k = null;
        LBFGSOnSimplex.y_k = null;
        LBFGSOnSimplex.s_ks = new LinkedList<Matrix>();
        LBFGSOnSimplex.y_ks = new LinkedList<Matrix>();
        LBFGSOnSimplex.rou_ks = new LinkedList<Double>();
        LBFGSOnSimplex.z = null;
        LBFGSOnSimplex.z_t = null;
        LBFGSOnSimplex.p_z = null;
        LBFGSOnSimplex.I_z = null;
        LBFGSOnSimplex.G_z = null;
        LBFGSOnSimplex.PG_z = null;
        LBFGSOnSimplex.i = -1;
        LBFGSOnSimplex.tol = 1.0;
        LBFGSOnSimplex.J = new ArrayList<Double>();
    }
    
    public static void main(final String[] args) {
        final int n = 10;
        final Matrix t = Matlab.rand(n);
        Matrix C = Matlab.minus(t.mtimes(t.transpose()), Matlab.times(0.05, Matlab.eye(n)));
        Matrix y = Matlab.times(3.0, Matlab.rand(n, 1));
        final double epsilon = 1.0E-6;
        final double gamma = 0.01;
        final String path = "C:/Aaron/My Codes/Matlab/Convex Optimization";
        C = IO.loadMatrix(String.valueOf(path) + File.separator + "C.txt");
        y = IO.loadMatrix(String.valueOf(path) + File.separator + "y.txt");
        final long start = System.currentTimeMillis();
        final Matrix x0 = Matlab.rdivide(Matlab.ones(n, 1), n);
        final Matrix x = x0.copy();
        Matrix r_x = null;
        double f_x = 0.0;
        double phi_x = 0.0;
        double fval = 0.0;
        r_x = C.mtimes(x).minus(y);
        f_x = Matlab.norm(r_x);
        phi_x = Matlab.norm(x);
        fval = f_x + gamma * phi_x;
        Matrix Grad_f_x = null;
        Matrix Grad_phi_x = null;
        Matrix Grad = null;
        Grad_f_x = Matlab.rdivide(C.transpose().mtimes(r_x), f_x);
        Grad_phi_x = Matlab.rdivide(x, phi_x);
        Grad = Matlab.plus(Grad_f_x, Matlab.times(gamma, Grad_phi_x));
        boolean[] flags = null;
        int k = 0;
        final int maxIter = 1000;
        while (true) {
            flags = run(Grad, fval, epsilon, x);
            if (flags[0]) {
                break;
            }
            if (Matlab.sum(Matlab.sum(Matlab.isnan(x))) > 0.0) {
                int a = 1;
                ++a;
            }
            r_x = C.mtimes(x).minus(y);
            f_x = Matlab.norm(r_x);
            phi_x = Matlab.norm(x);
            fval = f_x + gamma * phi_x;
            if (!flags[1]) {
                continue;
            }
            if (++k > maxIter) {
                break;
            }
            Grad_f_x = Matlab.rdivide(C.transpose().mtimes(r_x), f_x);
            Grad_phi_x = Matlab.rdivide(x, phi_x);
            Grad = Matlab.plus(Grad_f_x, Matlab.times(gamma, Grad_phi_x));
        }
        final Matrix x_projected_LBFGS_Armijo = x;
        final double f_projected_LBFGS_Armijo = fval;
        Printer.fprintf("fval_projected_LBFGS_Armijo: %g\n\n", f_projected_LBFGS_Armijo);
        Printer.fprintf("x_projected_LBFGS_Armijo:\n", new Object[0]);
        Printer.display(x_projected_LBFGS_Armijo.transpose());
        final double elapsedTime = (System.currentTimeMillis() - start) / 1000.0;
        Printer.fprintf("Elapsed time: %.3f seconds\n", elapsedTime);
    }
    
    public static boolean[] run(final Matrix Grad_t, final double fval_t, final double epsilon, final Matrix X_t) {
        if (LBFGSOnSimplex.state == 4) {
            LBFGSOnSimplex.s_ks.clear();
            LBFGSOnSimplex.y_ks.clear();
            LBFGSOnSimplex.rou_ks.clear();
            LBFGSOnSimplex.J.clear();
            LBFGSOnSimplex.z_t = null;
            LBFGSOnSimplex.state = 0;
        }
        if (LBFGSOnSimplex.state == 0) {
            LBFGSOnSimplex.X = X_t.copy();
            if (Grad_t == null) {
                System.err.println("Gradient is required on the first call!");
                System.exit(1);
            }
            LBFGSOnSimplex.G = Grad_t.copy();
            LBFGSOnSimplex.fval = fval_t;
            if (Double.isNaN(LBFGSOnSimplex.fval)) {
                System.err.println("Object function value is nan!");
                System.exit(1);
            }
            System.out.format("Initial ofv: %g\n", LBFGSOnSimplex.fval);
            LBFGSOnSimplex.tol = epsilon * Matlab.norm(LBFGSOnSimplex.G);
            LBFGSOnSimplex.k = 0;
            LBFGSOnSimplex.state = 1;
        }
        if (LBFGSOnSimplex.state == 1) {
            Matrix I_k = null;
            Matrix I_k_com = null;
            if (LBFGSOnSimplex.k == 0) {
                LBFGSOnSimplex.H = 1.0;
            }
            else {
                LBFGSOnSimplex.H = Matlab.innerProduct(LBFGSOnSimplex.s_k, LBFGSOnSimplex.y_k) / Matlab.innerProduct(LBFGSOnSimplex.y_k, LBFGSOnSimplex.y_k);
            }
            Matrix s_k_i = null;
            Matrix y_k_i = null;
            Double rou_k_i = null;
            Iterator<Matrix> iter_s_ks = null;
            Iterator<Matrix> iter_y_ks = null;
            Iterator<Double> iter_rou_ks = null;
            final double[] a = new double[LBFGSOnSimplex.m];
            double b = 0.0;
            Matrix q = null;
            Matrix r = null;
            q = LBFGSOnSimplex.G.copy();
            iter_s_ks = LBFGSOnSimplex.s_ks.descendingIterator();
            iter_y_ks = LBFGSOnSimplex.y_ks.descendingIterator();
            iter_rou_ks = LBFGSOnSimplex.rou_ks.descendingIterator();
            for (int i = LBFGSOnSimplex.s_ks.size() - 1; i >= 0; --i) {
                s_k_i = iter_s_ks.next();
                y_k_i = iter_y_ks.next();
                rou_k_i = iter_rou_ks.next();
                InPlaceOperator.minusAssign(q, a[i] = rou_k_i * Matlab.innerProduct(s_k_i, q), y_k_i);
            }
            r = Matlab.times(LBFGSOnSimplex.H, q);
            iter_s_ks = LBFGSOnSimplex.s_ks.iterator();
            iter_y_ks = LBFGSOnSimplex.y_ks.iterator();
            iter_rou_ks = LBFGSOnSimplex.rou_ks.iterator();
            for (int i = 0; i < LBFGSOnSimplex.s_ks.size(); ++i) {
                s_k_i = iter_s_ks.next();
                y_k_i = iter_y_ks.next();
                rou_k_i = iter_rou_ks.next();
                b = rou_k_i * Matlab.innerProduct(y_k_i, r);
                InPlaceOperator.plusAssign(r, a[i] - b, s_k_i);
            }
            LBFGSOnSimplex.i = (int)Matlab.max(LBFGSOnSimplex.X, 1)[1].get(0);
            final int n = Matlab.size(LBFGSOnSimplex.X, 1);
            final Matrix A = new SparseMatrix(n, n - 1);
            for (int j = 0; j < LBFGSOnSimplex.i; ++j) {
                A.setEntry(j, j, 1.0);
            }
            for (int j = 0; j < n - 1; ++j) {
                A.setEntry(LBFGSOnSimplex.i, j, -1.0);
            }
            for (int j = LBFGSOnSimplex.i; j < n - 1; ++j) {
                A.setEntry(j + 1, j, 1.0);
            }
            final Matrix b_z = new SparseMatrix(n, 1);
            b_z.setEntry(LBFGSOnSimplex.i, 0, 1.0);
            LBFGSOnSimplex.I_z = Matlab.not(b_z);
            LBFGSOnSimplex.z = Matlab.logicalIndexing(LBFGSOnSimplex.X, LBFGSOnSimplex.I_z);
            if (LBFGSOnSimplex.z_t == null) {
                LBFGSOnSimplex.z_t = LBFGSOnSimplex.z.copy();
            }
            LBFGSOnSimplex.G_z = A.transpose().mtimes(LBFGSOnSimplex.G);
            I_k = Matlab.or(Matlab.lt(LBFGSOnSimplex.G_z, 0.0), Matlab.gt(LBFGSOnSimplex.z, 0.0));
            I_k_com = Matlab.not(I_k);
            Matlab.logicalIndexingAssignment(LBFGSOnSimplex.PG_z = LBFGSOnSimplex.G_z.copy(), I_k_com, 0.0);
            final double norm_PGrad_z = Matlab.norm(LBFGSOnSimplex.PG_z);
            if (norm_PGrad_z < LBFGSOnSimplex.tol) {
                LBFGSOnSimplex.converge = true;
                LBFGSOnSimplex.gradientRequired = false;
                LBFGSOnSimplex.state = 4;
                System.out.printf("PLBFGS on simplex converges with norm(PGrad_z) %f\n", norm_PGrad_z);
                return new boolean[] { LBFGSOnSimplex.converge, LBFGSOnSimplex.gradientRequired };
            }
            final Matrix HG_z = A.transpose().mtimes(r);
            I_k = Matlab.or(Matlab.lt(HG_z, 0.0), Matlab.gt(LBFGSOnSimplex.z, 0.0));
            I_k_com = Matlab.not(I_k);
            final Matrix PHG_z = HG_z.copy();
            Matlab.logicalIndexingAssignment(PHG_z, I_k_com, 0.0);
            if (Matlab.innerProduct(PHG_z, LBFGSOnSimplex.G_z) <= 0.0) {
                LBFGSOnSimplex.p_z = Matlab.uminus(LBFGSOnSimplex.PG_z);
            }
            else {
                LBFGSOnSimplex.p_z = Matlab.uminus(PHG_z);
            }
            LBFGSOnSimplex.t = 1.0;
            while (true) {
                InPlaceOperator.affine(LBFGSOnSimplex.z_t, LBFGSOnSimplex.t, LBFGSOnSimplex.p_z, '+', LBFGSOnSimplex.z);
                InPlaceOperator.subplusAssign(LBFGSOnSimplex.z_t);
                if (Matlab.sum(Matlab.sum(LBFGSOnSimplex.z_t)) <= 1.0) {
                    break;
                }
                LBFGSOnSimplex.t *= LBFGSOnSimplex.beta;
            }
            LBFGSOnSimplex.state = 2;
            Matlab.logicalIndexingAssignment(X_t, LBFGSOnSimplex.I_z, LBFGSOnSimplex.z_t);
            X_t.setEntry(LBFGSOnSimplex.i, 0, 1.0 - Matlab.sum(Matlab.sum(LBFGSOnSimplex.z_t)));
            LBFGSOnSimplex.converge = false;
            LBFGSOnSimplex.gradientRequired = false;
            return new boolean[] { LBFGSOnSimplex.converge, LBFGSOnSimplex.gradientRequired };
        }
        else {
            if (LBFGSOnSimplex.state == 2) {
                LBFGSOnSimplex.converge = false;
                if (fval_t <= LBFGSOnSimplex.fval + LBFGSOnSimplex.alpha * Matlab.innerProduct(LBFGSOnSimplex.G_z, Matlab.minus(LBFGSOnSimplex.z_t, LBFGSOnSimplex.z))) {
                    LBFGSOnSimplex.gradientRequired = true;
                    LBFGSOnSimplex.state = 3;
                }
                else {
                    LBFGSOnSimplex.t *= LBFGSOnSimplex.beta;
                    InPlaceOperator.affine(LBFGSOnSimplex.z_t, LBFGSOnSimplex.t, LBFGSOnSimplex.p_z, '+', LBFGSOnSimplex.z);
                    InPlaceOperator.subplusAssign(LBFGSOnSimplex.z_t);
                    Matlab.logicalIndexingAssignment(X_t, LBFGSOnSimplex.I_z, LBFGSOnSimplex.z_t);
                    X_t.setEntry(LBFGSOnSimplex.i, 0, 1.0 - Matlab.sum(Matlab.sum(LBFGSOnSimplex.z_t)));
                    LBFGSOnSimplex.gradientRequired = false;
                }
                return new boolean[] { LBFGSOnSimplex.converge, LBFGSOnSimplex.gradientRequired };
            }
            if (LBFGSOnSimplex.state == 3) {
                if (LBFGSOnSimplex.X_pre == null) {
                    LBFGSOnSimplex.X_pre = LBFGSOnSimplex.X.copy();
                }
                else {
                    InPlaceOperator.assign(LBFGSOnSimplex.X_pre, LBFGSOnSimplex.X);
                }
                if (LBFGSOnSimplex.G_pre == null) {
                    LBFGSOnSimplex.G_pre = LBFGSOnSimplex.G.copy();
                }
                else {
                    InPlaceOperator.assign(LBFGSOnSimplex.G_pre, LBFGSOnSimplex.G);
                }
                LBFGSOnSimplex.fval = fval_t;
                LBFGSOnSimplex.J.add(LBFGSOnSimplex.fval);
                System.out.format("Iter %d, ofv: %g, norm(PGrad_z): %g\n", LBFGSOnSimplex.k + 1, LBFGSOnSimplex.fval, Matlab.norm(LBFGSOnSimplex.PG_z));
                InPlaceOperator.assign(LBFGSOnSimplex.X, X_t);
                InPlaceOperator.assign(LBFGSOnSimplex.G, Grad_t);
                if (LBFGSOnSimplex.k >= LBFGSOnSimplex.m) {
                    LBFGSOnSimplex.s_k = LBFGSOnSimplex.s_ks.removeFirst();
                    LBFGSOnSimplex.y_k = LBFGSOnSimplex.y_ks.removeFirst();
                    LBFGSOnSimplex.rou_ks.removeFirst();
                    InPlaceOperator.minus(LBFGSOnSimplex.s_k, LBFGSOnSimplex.X, LBFGSOnSimplex.X_pre);
                    InPlaceOperator.minus(LBFGSOnSimplex.y_k, LBFGSOnSimplex.G, LBFGSOnSimplex.G_pre);
                }
                else {
                    LBFGSOnSimplex.s_k = LBFGSOnSimplex.X.minus(LBFGSOnSimplex.X_pre);
                    LBFGSOnSimplex.y_k = LBFGSOnSimplex.G.minus(LBFGSOnSimplex.G_pre);
                }
                LBFGSOnSimplex.rou_k = 1.0 / Matlab.innerProduct(LBFGSOnSimplex.y_k, LBFGSOnSimplex.s_k);
                LBFGSOnSimplex.s_ks.add(LBFGSOnSimplex.s_k);
                LBFGSOnSimplex.y_ks.add(LBFGSOnSimplex.y_k);
                LBFGSOnSimplex.rou_ks.add(LBFGSOnSimplex.rou_k);
                ++LBFGSOnSimplex.k;
                LBFGSOnSimplex.state = 1;
            }
            LBFGSOnSimplex.converge = false;
            LBFGSOnSimplex.gradientRequired = false;
            return new boolean[] { LBFGSOnSimplex.converge, LBFGSOnSimplex.gradientRequired };
        }
    }
}
