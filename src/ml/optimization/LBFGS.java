package ml.optimization;

import la.matrix.*;
import ml.utils.*;
import java.util.*;

public class LBFGS
{
    private static Matrix G;
    private static Matrix G_pre;
    private static Matrix X;
    private static Matrix X_pre;
    private static Matrix p;
    private static double fval;
    private static boolean gradientRequired;
    private static boolean converge;
    private static int state;
    private static double t;
    private static double z;
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
    private static ArrayList<Double> J;
    
    static {
        LBFGS.G = null;
        LBFGS.G_pre = null;
        LBFGS.X = null;
        LBFGS.X_pre = null;
        LBFGS.p = null;
        LBFGS.fval = 0.0;
        LBFGS.gradientRequired = false;
        LBFGS.converge = false;
        LBFGS.state = 0;
        LBFGS.t = 1.0;
        LBFGS.z = 0.0;
        LBFGS.k = 0;
        LBFGS.alpha = 0.2;
        LBFGS.beta = 0.75;
        LBFGS.m = 30;
        LBFGS.H = 0.0;
        LBFGS.s_k = null;
        LBFGS.y_k = null;
        LBFGS.s_ks = new LinkedList<Matrix>();
        LBFGS.y_ks = new LinkedList<Matrix>();
        LBFGS.rou_ks = new LinkedList<Double>();
        LBFGS.J = new ArrayList<Double>();
    }
    
    public static boolean[] run(final Matrix Grad_t, final double fval_t, final double epsilon, final Matrix X_t) {
        if (LBFGS.state == 4) {
            LBFGS.s_ks.clear();
            LBFGS.y_ks.clear();
            LBFGS.rou_ks.clear();
            LBFGS.J.clear();
            LBFGS.X_pre = null;
            LBFGS.G_pre = null;
            LBFGS.state = 0;
        }
        if (LBFGS.state == 0) {
            LBFGS.X = X_t.copy();
            if (Grad_t == null) {
                System.err.println("Gradient is required on the first call!");
                System.exit(1);
            }
            LBFGS.G = Grad_t.copy();
            LBFGS.fval = fval_t;
            if (Double.isNaN(LBFGS.fval)) {
                System.err.println("Object function value is nan!");
                System.exit(1);
            }
            System.out.format("Initial ofv: %g\n", LBFGS.fval);
            LBFGS.k = 0;
            LBFGS.state = 1;
        }
        if (LBFGS.state == 1) {
            final double norm_Grad = Matlab.norm(LBFGS.G, Matlab.inf);
            if (norm_Grad < epsilon) {
                LBFGS.converge = true;
                LBFGS.gradientRequired = false;
                LBFGS.state = 4;
                System.out.printf("L-BFGS converges with norm(Grad) %f\n", norm_Grad);
                return new boolean[] { LBFGS.converge, LBFGS.gradientRequired };
            }
            if (LBFGS.k == 0) {
                LBFGS.H = 1.0;
            }
            else {
                LBFGS.H = Matlab.innerProduct(LBFGS.s_k, LBFGS.y_k) / Matlab.innerProduct(LBFGS.y_k, LBFGS.y_k);
            }
            Matrix s_k_i = null;
            Matrix y_k_i = null;
            Double rou_k_i = null;
            Iterator<Matrix> iter_s_ks = null;
            Iterator<Matrix> iter_y_ks = null;
            Iterator<Double> iter_rou_ks = null;
            final double[] a = new double[LBFGS.m];
            double b = 0.0;
            Matrix q = null;
            Matrix r = null;
            q = LBFGS.G.copy();
            iter_s_ks = LBFGS.s_ks.descendingIterator();
            iter_y_ks = LBFGS.y_ks.descendingIterator();
            iter_rou_ks = LBFGS.rou_ks.descendingIterator();
            for (int i = LBFGS.s_ks.size() - 1; i >= 0; --i) {
                s_k_i = iter_s_ks.next();
                y_k_i = iter_y_ks.next();
                rou_k_i = iter_rou_ks.next();
                InPlaceOperator.minusAssign(q, a[i] = rou_k_i * Matlab.innerProduct(s_k_i, q), y_k_i);
            }
            r = q;
            InPlaceOperator.timesAssign(r, LBFGS.H);
            iter_s_ks = LBFGS.s_ks.iterator();
            iter_y_ks = LBFGS.y_ks.iterator();
            iter_rou_ks = LBFGS.rou_ks.iterator();
            for (int i = 0; i < LBFGS.s_ks.size(); ++i) {
                s_k_i = iter_s_ks.next();
                y_k_i = iter_y_ks.next();
                rou_k_i = iter_rou_ks.next();
                b = rou_k_i * Matlab.innerProduct(y_k_i, r);
                InPlaceOperator.plusAssign(r, a[i] - b, s_k_i);
            }
            InPlaceOperator.uminusAssign(LBFGS.p = r);
            LBFGS.t = 1.0;
            LBFGS.z = Matlab.innerProduct(LBFGS.G, LBFGS.p);
            LBFGS.state = 2;
            InPlaceOperator.affine(X_t, LBFGS.X, LBFGS.t, LBFGS.p);
            LBFGS.converge = false;
            LBFGS.gradientRequired = false;
            return new boolean[] { LBFGS.converge, LBFGS.gradientRequired };
        }
        else {
            if (LBFGS.state == 2) {
                LBFGS.converge = false;
                if (fval_t <= LBFGS.fval + LBFGS.alpha * LBFGS.t * LBFGS.z) {
                    LBFGS.gradientRequired = true;
                    LBFGS.state = 3;
                }
                else {
                    LBFGS.t *= LBFGS.beta;
                    LBFGS.gradientRequired = false;
                    InPlaceOperator.affine(X_t, LBFGS.X, LBFGS.t, LBFGS.p);
                }
                return new boolean[] { LBFGS.converge, LBFGS.gradientRequired };
            }
            if (LBFGS.state == 3) {
                if (LBFGS.X_pre == null) {
                    LBFGS.X_pre = LBFGS.X.copy();
                }
                else {
                    InPlaceOperator.assign(LBFGS.X_pre, LBFGS.X);
                }
                if (LBFGS.G_pre == null) {
                    LBFGS.G_pre = LBFGS.G.copy();
                }
                else {
                    InPlaceOperator.assign(LBFGS.G_pre, LBFGS.G);
                }
                if (Math.abs(fval_t - LBFGS.fval) < 1.0E-32) {
                    LBFGS.converge = true;
                    LBFGS.gradientRequired = false;
                    LBFGS.state = 4;
                    System.out.printf("Objective function value doesn't decrease, iteration stopped!\n", new Object[0]);
                    System.out.format("Iter %d, ofv: %g, norm(Grad): %g\n", LBFGS.k + 1, LBFGS.fval, Matlab.norm(LBFGS.G, Matlab.inf));
                    return new boolean[] { LBFGS.converge, LBFGS.gradientRequired };
                }
                LBFGS.fval = fval_t;
                LBFGS.J.add(LBFGS.fval);
                System.out.format("Iter %d, ofv: %g, norm(Grad): %g\n", LBFGS.k + 1, LBFGS.fval, Matlab.norm(LBFGS.G, Matlab.inf));
                InPlaceOperator.assign(LBFGS.X, X_t);
                InPlaceOperator.assign(LBFGS.G, Grad_t);
                if (LBFGS.k >= LBFGS.m) {
                    LBFGS.s_k = LBFGS.s_ks.removeFirst();
                    LBFGS.y_k = LBFGS.y_ks.removeFirst();
                    LBFGS.rou_ks.removeFirst();
                    InPlaceOperator.minus(LBFGS.s_k, LBFGS.X, LBFGS.X_pre);
                    InPlaceOperator.minus(LBFGS.y_k, LBFGS.G, LBFGS.G_pre);
                }
                else {
                    LBFGS.s_k = LBFGS.X.minus(LBFGS.X_pre);
                    LBFGS.y_k = LBFGS.G.minus(LBFGS.G_pre);
                }
                LBFGS.rou_k = 1.0 / Matlab.innerProduct(LBFGS.y_k, LBFGS.s_k);
                LBFGS.s_ks.add(LBFGS.s_k);
                LBFGS.y_ks.add(LBFGS.y_k);
                LBFGS.rou_ks.add(LBFGS.rou_k);
                ++LBFGS.k;
                LBFGS.state = 1;
            }
            LBFGS.converge = false;
            LBFGS.gradientRequired = false;
            return new boolean[] { LBFGS.converge, LBFGS.gradientRequired };
        }
    }
}
