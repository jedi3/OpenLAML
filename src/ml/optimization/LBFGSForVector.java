package ml.optimization;

import la.vector.*;
import la.vector.Vector;
import ml.utils.*;

import java.util.*;

public class LBFGSForVector
{
    private static Vector G;
    private static Vector G_pre;
    private static Vector X;
    private static Vector X_pre;
    private static Vector p;
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
    private static Vector s_k;
    private static Vector y_k;
    private static double rou_k;
    private static LinkedList<Vector> s_ks;
    private static LinkedList<Vector> y_ks;
    private static LinkedList<Double> rou_ks;
    private static ArrayList<Double> J;
    
    static {
        LBFGSForVector.G = null;
        LBFGSForVector.G_pre = null;
        LBFGSForVector.X = null;
        LBFGSForVector.X_pre = null;
        LBFGSForVector.p = null;
        LBFGSForVector.fval = 0.0;
        LBFGSForVector.gradientRequired = false;
        LBFGSForVector.converge = false;
        LBFGSForVector.state = 0;
        LBFGSForVector.t = 1.0;
        LBFGSForVector.z = 0.0;
        LBFGSForVector.k = 0;
        LBFGSForVector.alpha = 0.2;
        LBFGSForVector.beta = 0.75;
        LBFGSForVector.m = 30;
        LBFGSForVector.H = 0.0;
        LBFGSForVector.s_k = null;
        LBFGSForVector.y_k = null;
        LBFGSForVector.s_ks = new LinkedList<Vector>();
        LBFGSForVector.y_ks = new LinkedList<Vector>();
        LBFGSForVector.rou_ks = new LinkedList<Double>();
        LBFGSForVector.J = new ArrayList<Double>();
    }
    
    public static boolean[] run(final Vector Grad_t, final double fval_t, final double epsilon, final Vector X_t) {
        if (LBFGSForVector.state == 4) {
            LBFGSForVector.s_ks.clear();
            LBFGSForVector.y_ks.clear();
            LBFGSForVector.rou_ks.clear();
            LBFGSForVector.J.clear();
            LBFGSForVector.X_pre = null;
            LBFGSForVector.G_pre = null;
            LBFGSForVector.state = 0;
        }
        if (LBFGSForVector.state == 0) {
            LBFGSForVector.X = X_t.copy();
            if (Grad_t == null) {
                System.err.println("Gradient is required on the first call!");
                System.exit(1);
            }
            LBFGSForVector.G = Grad_t.copy();
            LBFGSForVector.fval = fval_t;
            if (Double.isNaN(LBFGSForVector.fval)) {
                System.err.println("Object function value is nan!");
                System.exit(1);
            }
            System.out.format("Initial ofv: %g\n", LBFGSForVector.fval);
            LBFGSForVector.k = 0;
            LBFGSForVector.state = 1;
        }
        if (LBFGSForVector.state == 1) {
            final double norm_Grad = Matlab.norm(LBFGSForVector.G, Matlab.inf);
            if (norm_Grad < epsilon) {
                LBFGSForVector.converge = true;
                LBFGSForVector.gradientRequired = false;
                LBFGSForVector.state = 4;
                System.out.printf("L-BFGS converges with norm(Grad) %f\n", norm_Grad);
                return new boolean[] { LBFGSForVector.converge, LBFGSForVector.gradientRequired };
            }
            if (LBFGSForVector.k == 0) {
                LBFGSForVector.H = 1.0;
            }
            else {
                LBFGSForVector.H = Matlab.innerProduct(LBFGSForVector.s_k, LBFGSForVector.y_k) / Matlab.innerProduct(LBFGSForVector.y_k, LBFGSForVector.y_k);
            }
            Vector s_k_i = null;
            Vector y_k_i = null;
            Double rou_k_i = null;
            Iterator<Vector> iter_s_ks = null;
            Iterator<Vector> iter_y_ks = null;
            Iterator<Double> iter_rou_ks = null;
            final double[] a = new double[LBFGSForVector.m];
            double b = 0.0;
            Vector q = null;
            Vector r = null;
            q = LBFGSForVector.G.copy();
            iter_s_ks = LBFGSForVector.s_ks.descendingIterator();
            iter_y_ks = LBFGSForVector.y_ks.descendingIterator();
            iter_rou_ks = LBFGSForVector.rou_ks.descendingIterator();
            for (int i = LBFGSForVector.s_ks.size() - 1; i >= 0; --i) {
                s_k_i = iter_s_ks.next();
                y_k_i = iter_y_ks.next();
                rou_k_i = iter_rou_ks.next();
                InPlaceOperator.minusAssign(q, a[i] = rou_k_i * Matlab.innerProduct(s_k_i, q), y_k_i);
            }
            r = q;
            InPlaceOperator.timesAssign(r, LBFGSForVector.H);
            iter_s_ks = LBFGSForVector.s_ks.iterator();
            iter_y_ks = LBFGSForVector.y_ks.iterator();
            iter_rou_ks = LBFGSForVector.rou_ks.iterator();
            for (int i = 0; i < LBFGSForVector.s_ks.size(); ++i) {
                s_k_i = iter_s_ks.next();
                y_k_i = iter_y_ks.next();
                rou_k_i = iter_rou_ks.next();
                b = rou_k_i * Matlab.innerProduct(y_k_i, r);
                InPlaceOperator.plusAssign(r, a[i] - b, s_k_i);
            }
            InPlaceOperator.uminusAssign(LBFGSForVector.p = r);
            LBFGSForVector.t = 1.0;
            LBFGSForVector.z = Matlab.innerProduct(LBFGSForVector.G, LBFGSForVector.p);
            LBFGSForVector.state = 2;
            InPlaceOperator.affine(X_t, LBFGSForVector.X, LBFGSForVector.t, LBFGSForVector.p);
            LBFGSForVector.converge = false;
            LBFGSForVector.gradientRequired = false;
            return new boolean[] { LBFGSForVector.converge, LBFGSForVector.gradientRequired };
        }
        else {
            if (LBFGSForVector.state == 2) {
                LBFGSForVector.converge = false;
                if (fval_t <= LBFGSForVector.fval + LBFGSForVector.alpha * LBFGSForVector.t * LBFGSForVector.z) {
                    LBFGSForVector.gradientRequired = true;
                    LBFGSForVector.state = 3;
                }
                else {
                    LBFGSForVector.t *= LBFGSForVector.beta;
                    LBFGSForVector.gradientRequired = false;
                    InPlaceOperator.affine(X_t, LBFGSForVector.X, LBFGSForVector.t, LBFGSForVector.p);
                }
                return new boolean[] { LBFGSForVector.converge, LBFGSForVector.gradientRequired };
            }
            if (LBFGSForVector.state == 3) {
                if (LBFGSForVector.X_pre == null) {
                    LBFGSForVector.X_pre = LBFGSForVector.X.copy();
                }
                else {
                    InPlaceOperator.assign(LBFGSForVector.X_pre, LBFGSForVector.X);
                }
                if (LBFGSForVector.G_pre == null) {
                    LBFGSForVector.G_pre = LBFGSForVector.G.copy();
                }
                else {
                    InPlaceOperator.assign(LBFGSForVector.G_pre, LBFGSForVector.G);
                }
                if (Math.abs(fval_t - LBFGSForVector.fval) < 1.0E-32) {
                    LBFGSForVector.converge = true;
                    LBFGSForVector.gradientRequired = false;
                    LBFGSForVector.state = 4;
                    System.out.printf("Objective function value doesn't decrease, iteration stopped!\n", new Object[0]);
                    System.out.format("Iter %d, ofv: %g, norm(Grad): %g\n", LBFGSForVector.k + 1, LBFGSForVector.fval, Matlab.norm(LBFGSForVector.G, Matlab.inf));
                    return new boolean[] { LBFGSForVector.converge, LBFGSForVector.gradientRequired };
                }
                LBFGSForVector.fval = fval_t;
                LBFGSForVector.J.add(LBFGSForVector.fval);
                System.out.format("Iter %d, ofv: %g, norm(Grad): %g\n", LBFGSForVector.k + 1, LBFGSForVector.fval, Matlab.norm(LBFGSForVector.G, Matlab.inf));
                InPlaceOperator.assign(LBFGSForVector.X, X_t);
                InPlaceOperator.assign(LBFGSForVector.G, Grad_t);
                if (LBFGSForVector.k >= LBFGSForVector.m) {
                    LBFGSForVector.s_k = LBFGSForVector.s_ks.removeFirst();
                    LBFGSForVector.y_k = LBFGSForVector.y_ks.removeFirst();
                    LBFGSForVector.rou_ks.removeFirst();
                    InPlaceOperator.minus(LBFGSForVector.s_k, LBFGSForVector.X, LBFGSForVector.X_pre);
                    InPlaceOperator.minus(LBFGSForVector.y_k, LBFGSForVector.G, LBFGSForVector.G_pre);
                }
                else {
                    LBFGSForVector.s_k = LBFGSForVector.X.minus(LBFGSForVector.X_pre);
                    LBFGSForVector.y_k = LBFGSForVector.G.minus(LBFGSForVector.G_pre);
                }
                LBFGSForVector.rou_k = 1.0 / Matlab.innerProduct(LBFGSForVector.y_k, LBFGSForVector.s_k);
                LBFGSForVector.s_ks.add(LBFGSForVector.s_k);
                LBFGSForVector.y_ks.add(LBFGSForVector.y_k);
                LBFGSForVector.rou_ks.add(LBFGSForVector.rou_k);
                ++LBFGSForVector.k;
                LBFGSForVector.state = 1;
            }
            LBFGSForVector.converge = false;
            LBFGSForVector.gradientRequired = false;
            return new boolean[] { LBFGSForVector.converge, LBFGSForVector.gradientRequired };
        }
    }
}
