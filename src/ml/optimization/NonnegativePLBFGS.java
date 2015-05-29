package ml.optimization;

import la.matrix.*;
import ml.utils.*;
import java.util.*;

public class NonnegativePLBFGS
{
    private static Matrix G;
    private static Matrix PG;
    private static Matrix G_pre;
    private static Matrix X;
    private static Matrix X_pre;
    private static Matrix p;
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
    private static double tol;
    private static ArrayList<Double> J;
    
    static {
        NonnegativePLBFGS.G = null;
        NonnegativePLBFGS.PG = null;
        NonnegativePLBFGS.G_pre = null;
        NonnegativePLBFGS.X = null;
        NonnegativePLBFGS.X_pre = null;
        NonnegativePLBFGS.p = null;
        NonnegativePLBFGS.fval = 0.0;
        NonnegativePLBFGS.gradientRequired = false;
        NonnegativePLBFGS.converge = false;
        NonnegativePLBFGS.state = 0;
        NonnegativePLBFGS.t = 1.0;
        NonnegativePLBFGS.k = 0;
        NonnegativePLBFGS.alpha = 0.2;
        NonnegativePLBFGS.beta = 0.75;
        NonnegativePLBFGS.m = 30;
        NonnegativePLBFGS.H = 0.0;
        NonnegativePLBFGS.s_k = null;
        NonnegativePLBFGS.y_k = null;
        NonnegativePLBFGS.s_ks = new LinkedList<Matrix>();
        NonnegativePLBFGS.y_ks = new LinkedList<Matrix>();
        NonnegativePLBFGS.rou_ks = new LinkedList<Double>();
        NonnegativePLBFGS.tol = 1.0;
        NonnegativePLBFGS.J = new ArrayList<Double>();
    }
    
    public static void main(final String[] args) {
    }
    
    public static boolean[] run(final Matrix Grad_t, final double fval_t, final double epsilon, final Matrix X_t) {
        if (NonnegativePLBFGS.state == 4) {
            NonnegativePLBFGS.s_ks.clear();
            NonnegativePLBFGS.y_ks.clear();
            NonnegativePLBFGS.rou_ks.clear();
            NonnegativePLBFGS.J.clear();
            NonnegativePLBFGS.X_pre = null;
            NonnegativePLBFGS.G_pre = null;
            NonnegativePLBFGS.PG = null;
            NonnegativePLBFGS.state = 0;
        }
        if (NonnegativePLBFGS.state == 0) {
            NonnegativePLBFGS.X = X_t.copy();
            if (Grad_t == null) {
                System.err.println("Gradient is required on the first call!");
                System.exit(1);
            }
            NonnegativePLBFGS.G = Grad_t.copy();
            NonnegativePLBFGS.fval = fval_t;
            if (Double.isNaN(NonnegativePLBFGS.fval)) {
                System.err.println("Object function value is nan!");
                System.exit(1);
            }
            System.out.format("Initial ofv: %g\n", NonnegativePLBFGS.fval);
            NonnegativePLBFGS.tol = epsilon * Matlab.norm(NonnegativePLBFGS.G, Matlab.inf);
            NonnegativePLBFGS.k = 0;
            NonnegativePLBFGS.state = 1;
        }
        if (NonnegativePLBFGS.state == 1) {
            Matrix I_k = null;
            Matrix I_k_com = null;
            I_k = Matlab.or(Matlab.lt(NonnegativePLBFGS.G, 0.0), Matlab.gt(NonnegativePLBFGS.X, 0.0));
            I_k_com = Matlab.not(I_k);
            if (NonnegativePLBFGS.PG == null) {
                NonnegativePLBFGS.PG = NonnegativePLBFGS.G.copy();
            }
            else {
                InPlaceOperator.assign(NonnegativePLBFGS.PG, NonnegativePLBFGS.G);
            }
            Matlab.logicalIndexingAssignment(NonnegativePLBFGS.PG, I_k_com, 0.0);
            final double norm_PGrad = Matlab.norm(NonnegativePLBFGS.PG, Matlab.inf);
            if (norm_PGrad < NonnegativePLBFGS.tol) {
                NonnegativePLBFGS.converge = true;
                NonnegativePLBFGS.gradientRequired = false;
                NonnegativePLBFGS.state = 4;
                System.out.printf("PLBFGS converges with norm(PGrad) %f\n", norm_PGrad);
                return new boolean[] { NonnegativePLBFGS.converge, NonnegativePLBFGS.gradientRequired };
            }
            if (NonnegativePLBFGS.k == 0) {
                NonnegativePLBFGS.H = 1.0;
            }
            else {
                NonnegativePLBFGS.H = Matlab.innerProduct(NonnegativePLBFGS.s_k, NonnegativePLBFGS.y_k) / Matlab.innerProduct(NonnegativePLBFGS.y_k, NonnegativePLBFGS.y_k);
            }
            Matrix s_k_i = null;
            Matrix y_k_i = null;
            Double rou_k_i = null;
            Iterator<Matrix> iter_s_ks = null;
            Iterator<Matrix> iter_y_ks = null;
            Iterator<Double> iter_rou_ks = null;
            final double[] a = new double[NonnegativePLBFGS.m];
            double b = 0.0;
            Matrix q = null;
            Matrix r = null;
            q = NonnegativePLBFGS.G.copy();
            iter_s_ks = NonnegativePLBFGS.s_ks.descendingIterator();
            iter_y_ks = NonnegativePLBFGS.y_ks.descendingIterator();
            iter_rou_ks = NonnegativePLBFGS.rou_ks.descendingIterator();
            for (int i = NonnegativePLBFGS.s_ks.size() - 1; i >= 0; --i) {
                s_k_i = iter_s_ks.next();
                y_k_i = iter_y_ks.next();
                rou_k_i = iter_rou_ks.next();
                InPlaceOperator.minusAssign(q, a[i] = rou_k_i * Matlab.innerProduct(s_k_i, q), y_k_i);
            }
            r = Matlab.times(NonnegativePLBFGS.H, q);
            iter_s_ks = NonnegativePLBFGS.s_ks.iterator();
            iter_y_ks = NonnegativePLBFGS.y_ks.iterator();
            iter_rou_ks = NonnegativePLBFGS.rou_ks.iterator();
            for (int i = 0; i < NonnegativePLBFGS.s_ks.size(); ++i) {
                s_k_i = iter_s_ks.next();
                y_k_i = iter_y_ks.next();
                rou_k_i = iter_rou_ks.next();
                b = rou_k_i * Matlab.innerProduct(y_k_i, r);
                InPlaceOperator.plusAssign(r, a[i] - b, s_k_i);
            }
            final Matrix HG = r;
            final Matrix PHG = HG.copy();
            I_k = Matlab.or(Matlab.lt(HG, 0.0), Matlab.gt(NonnegativePLBFGS.X, 0.0));
            I_k_com = Matlab.not(I_k);
            Matlab.logicalIndexingAssignment(PHG, I_k_com, 0.0);
            if (Matlab.innerProduct(PHG, NonnegativePLBFGS.G) <= 0.0) {
                NonnegativePLBFGS.p = Matlab.uminus(NonnegativePLBFGS.PG);
            }
            else {
                NonnegativePLBFGS.p = Matlab.uminus(PHG);
            }
            NonnegativePLBFGS.t = 1.0;
            NonnegativePLBFGS.state = 2;
            InPlaceOperator.affine(X_t, NonnegativePLBFGS.X, NonnegativePLBFGS.t, NonnegativePLBFGS.p);
            InPlaceOperator.subplusAssign(X_t);
            NonnegativePLBFGS.converge = false;
            NonnegativePLBFGS.gradientRequired = false;
            return new boolean[] { NonnegativePLBFGS.converge, NonnegativePLBFGS.gradientRequired };
        }
        else {
            if (NonnegativePLBFGS.state == 2) {
                NonnegativePLBFGS.converge = false;
                if (fval_t <= NonnegativePLBFGS.fval + NonnegativePLBFGS.alpha * Matlab.innerProduct(NonnegativePLBFGS.G, Matlab.minus(X_t, NonnegativePLBFGS.X))) {
                    NonnegativePLBFGS.gradientRequired = true;
                    NonnegativePLBFGS.state = 3;
                }
                else {
                    NonnegativePLBFGS.t *= NonnegativePLBFGS.beta;
                    NonnegativePLBFGS.gradientRequired = false;
                    InPlaceOperator.affine(X_t, NonnegativePLBFGS.X, NonnegativePLBFGS.t, NonnegativePLBFGS.p);
                    InPlaceOperator.subplusAssign(X_t);
                }
                return new boolean[] { NonnegativePLBFGS.converge, NonnegativePLBFGS.gradientRequired };
            }
            if (NonnegativePLBFGS.state == 3) {
                if (NonnegativePLBFGS.X_pre == null) {
                    NonnegativePLBFGS.X_pre = NonnegativePLBFGS.X.copy();
                }
                else {
                    InPlaceOperator.assign(NonnegativePLBFGS.X_pre, NonnegativePLBFGS.X);
                }
                if (NonnegativePLBFGS.G_pre == null) {
                    NonnegativePLBFGS.G_pre = NonnegativePLBFGS.G.copy();
                }
                else {
                    InPlaceOperator.assign(NonnegativePLBFGS.G_pre, NonnegativePLBFGS.G);
                }
                if (Math.abs(fval_t - NonnegativePLBFGS.fval) < Matlab.eps) {
                    NonnegativePLBFGS.converge = true;
                    NonnegativePLBFGS.gradientRequired = false;
                    NonnegativePLBFGS.state = 4;
                    System.out.printf("Objective function value doesn't decrease, iteration stopped!\n", new Object[0]);
                    System.out.format("Iter %d, ofv: %g, norm(PGrad): %g\n", NonnegativePLBFGS.k + 1, NonnegativePLBFGS.fval, Matlab.norm(NonnegativePLBFGS.PG, Matlab.inf));
                    return new boolean[] { NonnegativePLBFGS.converge, NonnegativePLBFGS.gradientRequired };
                }
                NonnegativePLBFGS.fval = fval_t;
                NonnegativePLBFGS.J.add(NonnegativePLBFGS.fval);
                System.out.format("Iter %d, ofv: %g, norm(PGrad): %g\n", NonnegativePLBFGS.k + 1, NonnegativePLBFGS.fval, Matlab.norm(NonnegativePLBFGS.PG, Matlab.inf));
                InPlaceOperator.assign(NonnegativePLBFGS.X, X_t);
                InPlaceOperator.assign(NonnegativePLBFGS.G, Grad_t);
                if (NonnegativePLBFGS.k >= NonnegativePLBFGS.m) {
                    NonnegativePLBFGS.s_k = NonnegativePLBFGS.s_ks.removeFirst();
                    NonnegativePLBFGS.y_k = NonnegativePLBFGS.y_ks.removeFirst();
                    NonnegativePLBFGS.rou_ks.removeFirst();
                    InPlaceOperator.minus(NonnegativePLBFGS.s_k, NonnegativePLBFGS.X, NonnegativePLBFGS.X_pre);
                    InPlaceOperator.minus(NonnegativePLBFGS.y_k, NonnegativePLBFGS.G, NonnegativePLBFGS.G_pre);
                }
                else {
                    NonnegativePLBFGS.s_k = NonnegativePLBFGS.X.minus(NonnegativePLBFGS.X_pre);
                    NonnegativePLBFGS.y_k = NonnegativePLBFGS.G.minus(NonnegativePLBFGS.G_pre);
                }
                NonnegativePLBFGS.rou_k = 1.0 / Matlab.innerProduct(NonnegativePLBFGS.y_k, NonnegativePLBFGS.s_k);
                NonnegativePLBFGS.s_ks.add(NonnegativePLBFGS.s_k);
                NonnegativePLBFGS.y_ks.add(NonnegativePLBFGS.y_k);
                NonnegativePLBFGS.rou_ks.add(NonnegativePLBFGS.rou_k);
                ++NonnegativePLBFGS.k;
                NonnegativePLBFGS.state = 1;
            }
            NonnegativePLBFGS.converge = false;
            NonnegativePLBFGS.gradientRequired = false;
            return new boolean[] { NonnegativePLBFGS.converge, NonnegativePLBFGS.gradientRequired };
        }
    }
}
