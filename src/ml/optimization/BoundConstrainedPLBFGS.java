package ml.optimization;

import la.matrix.*;
import ml.utils.*;
import java.util.*;

public class BoundConstrainedPLBFGS
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
        BoundConstrainedPLBFGS.G = null;
        BoundConstrainedPLBFGS.PG = null;
        BoundConstrainedPLBFGS.G_pre = null;
        BoundConstrainedPLBFGS.X = null;
        BoundConstrainedPLBFGS.X_pre = null;
        BoundConstrainedPLBFGS.p = null;
        BoundConstrainedPLBFGS.fval = 0.0;
        BoundConstrainedPLBFGS.gradientRequired = false;
        BoundConstrainedPLBFGS.converge = false;
        BoundConstrainedPLBFGS.state = 0;
        BoundConstrainedPLBFGS.t = 1.0;
        BoundConstrainedPLBFGS.k = 0;
        BoundConstrainedPLBFGS.alpha = 0.2;
        BoundConstrainedPLBFGS.beta = 0.75;
        BoundConstrainedPLBFGS.m = 30;
        BoundConstrainedPLBFGS.H = 0.0;
        BoundConstrainedPLBFGS.s_k = null;
        BoundConstrainedPLBFGS.y_k = null;
        BoundConstrainedPLBFGS.s_ks = new LinkedList<Matrix>();
        BoundConstrainedPLBFGS.y_ks = new LinkedList<Matrix>();
        BoundConstrainedPLBFGS.rou_ks = new LinkedList<Double>();
        BoundConstrainedPLBFGS.tol = 1.0;
        BoundConstrainedPLBFGS.J = new ArrayList<Double>();
    }
    
    public static boolean[] run(final Matrix Grad_t, final double fval_t, final double l, final double u, final double epsilon, final Matrix X_t) {
        if (BoundConstrainedPLBFGS.state == 4) {
            BoundConstrainedPLBFGS.s_ks.clear();
            BoundConstrainedPLBFGS.y_ks.clear();
            BoundConstrainedPLBFGS.rou_ks.clear();
            BoundConstrainedPLBFGS.J.clear();
            BoundConstrainedPLBFGS.X_pre = null;
            BoundConstrainedPLBFGS.G_pre = null;
            BoundConstrainedPLBFGS.PG = null;
            BoundConstrainedPLBFGS.state = 0;
        }
        if (BoundConstrainedPLBFGS.state == 0) {
            BoundConstrainedPLBFGS.X = X_t.copy();
            if (Grad_t == null) {
                System.err.println("Gradient is required on the first call!");
                System.exit(1);
            }
            BoundConstrainedPLBFGS.G = Grad_t.copy();
            BoundConstrainedPLBFGS.fval = fval_t;
            if (Double.isNaN(BoundConstrainedPLBFGS.fval)) {
                System.err.println("Object function value is nan!");
                System.exit(1);
            }
            System.out.format("Initial ofv: %g\n", BoundConstrainedPLBFGS.fval);
            BoundConstrainedPLBFGS.tol = epsilon * Matlab.norm(BoundConstrainedPLBFGS.G, Matlab.inf);
            BoundConstrainedPLBFGS.k = 0;
            BoundConstrainedPLBFGS.state = 1;
        }
        if (BoundConstrainedPLBFGS.state == 1) {
            Matrix I_b = null;
            Matrix I_l = null;
            Matrix I_u = null;
            I_b = Matlab.and(Matlab.lt(l, BoundConstrainedPLBFGS.X), Matlab.lt(BoundConstrainedPLBFGS.X, u));
            I_l = Matlab.eq(BoundConstrainedPLBFGS.X, l);
            I_u = Matlab.eq(BoundConstrainedPLBFGS.X, u);
            if (BoundConstrainedPLBFGS.PG == null) {
                BoundConstrainedPLBFGS.PG = BoundConstrainedPLBFGS.G.copy();
            }
            else {
                InPlaceOperator.assign(BoundConstrainedPLBFGS.PG, BoundConstrainedPLBFGS.G);
            }
            Matlab.logicalIndexingAssignment(BoundConstrainedPLBFGS.PG, I_b, Matlab.logicalIndexing(BoundConstrainedPLBFGS.G, I_b));
            Matlab.logicalIndexingAssignment(BoundConstrainedPLBFGS.PG, I_l, Matlab.min(Matlab.logicalIndexing(BoundConstrainedPLBFGS.G, I_l), 0.0));
            Matlab.logicalIndexingAssignment(BoundConstrainedPLBFGS.PG, I_u, Matlab.max(Matlab.logicalIndexing(BoundConstrainedPLBFGS.G, I_u), 0.0));
            final double norm_PGrad = Matlab.norm(BoundConstrainedPLBFGS.PG, Matlab.inf);
            if (norm_PGrad < BoundConstrainedPLBFGS.tol) {
                BoundConstrainedPLBFGS.converge = true;
                BoundConstrainedPLBFGS.gradientRequired = false;
                BoundConstrainedPLBFGS.state = 4;
                System.out.printf("PLBFGS converges with norm(PGrad) %f\n", norm_PGrad);
                return new boolean[] { BoundConstrainedPLBFGS.converge, BoundConstrainedPLBFGS.gradientRequired };
            }
            if (BoundConstrainedPLBFGS.k == 0) {
                BoundConstrainedPLBFGS.H = 1.0;
            }
            else {
                BoundConstrainedPLBFGS.H = Matlab.innerProduct(BoundConstrainedPLBFGS.s_k, BoundConstrainedPLBFGS.y_k) / Matlab.innerProduct(BoundConstrainedPLBFGS.y_k, BoundConstrainedPLBFGS.y_k);
            }
            Matrix s_k_i = null;
            Matrix y_k_i = null;
            Double rou_k_i = null;
            Iterator<Matrix> iter_s_ks = null;
            Iterator<Matrix> iter_y_ks = null;
            Iterator<Double> iter_rou_ks = null;
            final double[] a = new double[BoundConstrainedPLBFGS.m];
            double b = 0.0;
            Matrix q = null;
            Matrix r = null;
            q = BoundConstrainedPLBFGS.G.copy();
            iter_s_ks = BoundConstrainedPLBFGS.s_ks.descendingIterator();
            iter_y_ks = BoundConstrainedPLBFGS.y_ks.descendingIterator();
            iter_rou_ks = BoundConstrainedPLBFGS.rou_ks.descendingIterator();
            for (int i = BoundConstrainedPLBFGS.s_ks.size() - 1; i >= 0; --i) {
                s_k_i = iter_s_ks.next();
                y_k_i = iter_y_ks.next();
                rou_k_i = iter_rou_ks.next();
                InPlaceOperator.minusAssign(q, a[i] = rou_k_i * Matlab.innerProduct(s_k_i, q), y_k_i);
            }
            r = Matlab.times(BoundConstrainedPLBFGS.H, q);
            iter_s_ks = BoundConstrainedPLBFGS.s_ks.iterator();
            iter_y_ks = BoundConstrainedPLBFGS.y_ks.iterator();
            iter_rou_ks = BoundConstrainedPLBFGS.rou_ks.iterator();
            for (int i = 0; i < BoundConstrainedPLBFGS.s_ks.size(); ++i) {
                s_k_i = iter_s_ks.next();
                y_k_i = iter_y_ks.next();
                rou_k_i = iter_rou_ks.next();
                b = rou_k_i * Matlab.innerProduct(y_k_i, r);
                InPlaceOperator.plusAssign(r, a[i] - b, s_k_i);
            }
            final Matrix HG = r;
            final Matrix PHG = HG.copy();
            Matlab.logicalIndexingAssignment(PHG, I_b, Matlab.logicalIndexing(HG, I_b));
            Matlab.logicalIndexingAssignment(PHG, I_l, Matlab.min(Matlab.logicalIndexing(HG, I_l), 0.0));
            Matlab.logicalIndexingAssignment(PHG, I_u, Matlab.max(Matlab.logicalIndexing(HG, I_u), 0.0));
            if (Matlab.innerProduct(PHG, BoundConstrainedPLBFGS.G) <= 0.0) {
                BoundConstrainedPLBFGS.p = Matlab.uminus(BoundConstrainedPLBFGS.PG);
            }
            else {
                BoundConstrainedPLBFGS.p = Matlab.uminus(PHG);
            }
            BoundConstrainedPLBFGS.t = 1.0;
            BoundConstrainedPLBFGS.state = 2;
            Matlab.setMatrix(X_t, project(Matlab.plus(BoundConstrainedPLBFGS.X, Matlab.times(BoundConstrainedPLBFGS.t, BoundConstrainedPLBFGS.p)), l, u));
            BoundConstrainedPLBFGS.converge = false;
            BoundConstrainedPLBFGS.gradientRequired = false;
            return new boolean[] { BoundConstrainedPLBFGS.converge, BoundConstrainedPLBFGS.gradientRequired };
        }
        else {
            if (BoundConstrainedPLBFGS.state == 2) {
                BoundConstrainedPLBFGS.converge = false;
                if (fval_t <= BoundConstrainedPLBFGS.fval + BoundConstrainedPLBFGS.alpha * BoundConstrainedPLBFGS.t * Matlab.innerProduct(BoundConstrainedPLBFGS.G, Matlab.minus(X_t, BoundConstrainedPLBFGS.X))) {
                    BoundConstrainedPLBFGS.gradientRequired = true;
                    BoundConstrainedPLBFGS.state = 3;
                }
                else {
                    BoundConstrainedPLBFGS.t *= BoundConstrainedPLBFGS.beta;
                    BoundConstrainedPLBFGS.gradientRequired = false;
                    Matlab.setMatrix(X_t, project(Matlab.plus(BoundConstrainedPLBFGS.X, Matlab.times(BoundConstrainedPLBFGS.t, BoundConstrainedPLBFGS.p)), l, u));
                }
                return new boolean[] { BoundConstrainedPLBFGS.converge, BoundConstrainedPLBFGS.gradientRequired };
            }
            if (BoundConstrainedPLBFGS.state == 3) {
                if (BoundConstrainedPLBFGS.X_pre == null) {
                    BoundConstrainedPLBFGS.X_pre = BoundConstrainedPLBFGS.X.copy();
                }
                else {
                    InPlaceOperator.assign(BoundConstrainedPLBFGS.X_pre, BoundConstrainedPLBFGS.X);
                }
                if (BoundConstrainedPLBFGS.G_pre == null) {
                    BoundConstrainedPLBFGS.G_pre = BoundConstrainedPLBFGS.G.copy();
                }
                else {
                    InPlaceOperator.assign(BoundConstrainedPLBFGS.G_pre, BoundConstrainedPLBFGS.G);
                }
                if (Math.abs(fval_t - BoundConstrainedPLBFGS.fval) < Matlab.eps) {
                    BoundConstrainedPLBFGS.converge = true;
                    BoundConstrainedPLBFGS.gradientRequired = false;
                    System.out.printf("Objective function value doesn't decrease, iteration stopped!\n", new Object[0]);
                    System.out.format("Iter %d, ofv: %g, norm(PGrad): %g\n", BoundConstrainedPLBFGS.k + 1, BoundConstrainedPLBFGS.fval, Matlab.norm(BoundConstrainedPLBFGS.PG, Matlab.inf));
                    return new boolean[] { BoundConstrainedPLBFGS.converge, BoundConstrainedPLBFGS.gradientRequired };
                }
                BoundConstrainedPLBFGS.fval = fval_t;
                BoundConstrainedPLBFGS.J.add(BoundConstrainedPLBFGS.fval);
                System.out.format("Iter %d, ofv: %g, norm(PGrad): %g\n", BoundConstrainedPLBFGS.k + 1, BoundConstrainedPLBFGS.fval, Matlab.norm(BoundConstrainedPLBFGS.PG, Matlab.inf));
                InPlaceOperator.assign(BoundConstrainedPLBFGS.X, X_t);
                InPlaceOperator.assign(BoundConstrainedPLBFGS.G, Grad_t);
                if (BoundConstrainedPLBFGS.k >= BoundConstrainedPLBFGS.m) {
                    BoundConstrainedPLBFGS.s_k = BoundConstrainedPLBFGS.s_ks.removeFirst();
                    BoundConstrainedPLBFGS.y_k = BoundConstrainedPLBFGS.y_ks.removeFirst();
                    BoundConstrainedPLBFGS.rou_ks.removeFirst();
                    InPlaceOperator.minus(BoundConstrainedPLBFGS.s_k, BoundConstrainedPLBFGS.X, BoundConstrainedPLBFGS.X_pre);
                    InPlaceOperator.minus(BoundConstrainedPLBFGS.y_k, BoundConstrainedPLBFGS.G, BoundConstrainedPLBFGS.G_pre);
                }
                else {
                    BoundConstrainedPLBFGS.s_k = BoundConstrainedPLBFGS.X.minus(BoundConstrainedPLBFGS.X_pre);
                    BoundConstrainedPLBFGS.y_k = BoundConstrainedPLBFGS.G.minus(BoundConstrainedPLBFGS.G_pre);
                }
                BoundConstrainedPLBFGS.rou_k = 1.0 / Matlab.innerProduct(BoundConstrainedPLBFGS.y_k, BoundConstrainedPLBFGS.s_k);
                BoundConstrainedPLBFGS.s_ks.add(BoundConstrainedPLBFGS.s_k);
                BoundConstrainedPLBFGS.y_ks.add(BoundConstrainedPLBFGS.y_k);
                BoundConstrainedPLBFGS.rou_ks.add(BoundConstrainedPLBFGS.rou_k);
                ++BoundConstrainedPLBFGS.k;
                BoundConstrainedPLBFGS.state = 1;
            }
            BoundConstrainedPLBFGS.converge = false;
            BoundConstrainedPLBFGS.gradientRequired = false;
            return new boolean[] { BoundConstrainedPLBFGS.converge, BoundConstrainedPLBFGS.gradientRequired };
        }
    }
    
    public static boolean[] run(final Matrix Grad_t, final double fval_t, final Matrix L, final Matrix U, final double epsilon, final Matrix X_t) {
        if (BoundConstrainedPLBFGS.state == 4) {
            BoundConstrainedPLBFGS.s_ks.clear();
            BoundConstrainedPLBFGS.y_ks.clear();
            BoundConstrainedPLBFGS.rou_ks.clear();
            BoundConstrainedPLBFGS.J.clear();
            BoundConstrainedPLBFGS.state = 0;
        }
        if (BoundConstrainedPLBFGS.state == 0) {
            BoundConstrainedPLBFGS.X = X_t.copy();
            if (Grad_t == null) {
                System.err.println("Gradient is required on the first call!");
                System.exit(1);
            }
            BoundConstrainedPLBFGS.G = Grad_t.copy();
            BoundConstrainedPLBFGS.fval = fval_t;
            if (Double.isNaN(BoundConstrainedPLBFGS.fval)) {
                System.err.println("Object function value is nan!");
                System.exit(1);
            }
            System.out.format("Initial ofv: %g\n", BoundConstrainedPLBFGS.fval);
            BoundConstrainedPLBFGS.tol = epsilon * Matlab.norm(BoundConstrainedPLBFGS.G, Matlab.inf);
            BoundConstrainedPLBFGS.state = 1;
        }
        if (BoundConstrainedPLBFGS.state == 1) {
            Matrix I_b = null;
            Matrix I_l = null;
            Matrix I_u = null;
            I_b = Matlab.and(Matlab.lt(L, BoundConstrainedPLBFGS.X), Matlab.lt(BoundConstrainedPLBFGS.X, U));
            I_l = Matlab.eq(BoundConstrainedPLBFGS.X, L);
            I_u = Matlab.eq(BoundConstrainedPLBFGS.X, U);
            if (BoundConstrainedPLBFGS.PG == null) {
                BoundConstrainedPLBFGS.PG = BoundConstrainedPLBFGS.G.copy();
            }
            else {
                InPlaceOperator.assign(BoundConstrainedPLBFGS.PG, BoundConstrainedPLBFGS.G);
            }
            Matlab.logicalIndexingAssignment(BoundConstrainedPLBFGS.PG, I_b, Matlab.logicalIndexing(BoundConstrainedPLBFGS.G, I_b));
            Matlab.logicalIndexingAssignment(BoundConstrainedPLBFGS.PG, I_l, Matlab.min(Matlab.logicalIndexing(BoundConstrainedPLBFGS.G, I_l), 0.0));
            Matlab.logicalIndexingAssignment(BoundConstrainedPLBFGS.PG, I_u, Matlab.max(Matlab.logicalIndexing(BoundConstrainedPLBFGS.G, I_u), 0.0));
            final double norm_PGrad = Matlab.norm(BoundConstrainedPLBFGS.PG, Matlab.inf);
            if (norm_PGrad < BoundConstrainedPLBFGS.tol) {
                BoundConstrainedPLBFGS.converge = true;
                BoundConstrainedPLBFGS.gradientRequired = false;
                BoundConstrainedPLBFGS.state = 4;
                System.out.printf("PLBFGS converges with norm(PGrad) %f\n", norm_PGrad);
                return new boolean[] { BoundConstrainedPLBFGS.converge, BoundConstrainedPLBFGS.gradientRequired };
            }
            if (BoundConstrainedPLBFGS.k == 0) {
                BoundConstrainedPLBFGS.H = 1.0;
            }
            else {
                BoundConstrainedPLBFGS.H = Matlab.innerProduct(BoundConstrainedPLBFGS.s_k, BoundConstrainedPLBFGS.y_k) / Matlab.innerProduct(BoundConstrainedPLBFGS.y_k, BoundConstrainedPLBFGS.y_k);
            }
            Matrix s_k_i = null;
            Matrix y_k_i = null;
            Double rou_k_i = null;
            Iterator<Matrix> iter_s_ks = null;
            Iterator<Matrix> iter_y_ks = null;
            Iterator<Double> iter_rou_ks = null;
            final double[] a = new double[BoundConstrainedPLBFGS.m];
            double b = 0.0;
            Matrix q = null;
            Matrix r = null;
            q = BoundConstrainedPLBFGS.G.copy();
            iter_s_ks = BoundConstrainedPLBFGS.s_ks.descendingIterator();
            iter_y_ks = BoundConstrainedPLBFGS.y_ks.descendingIterator();
            iter_rou_ks = BoundConstrainedPLBFGS.rou_ks.descendingIterator();
            for (int i = BoundConstrainedPLBFGS.s_ks.size() - 1; i >= 0; --i) {
                s_k_i = iter_s_ks.next();
                y_k_i = iter_y_ks.next();
                rou_k_i = iter_rou_ks.next();
                InPlaceOperator.minusAssign(q, a[i] = rou_k_i * Matlab.innerProduct(s_k_i, q), y_k_i);
            }
            r = Matlab.times(BoundConstrainedPLBFGS.H, q);
            iter_s_ks = BoundConstrainedPLBFGS.s_ks.iterator();
            iter_y_ks = BoundConstrainedPLBFGS.y_ks.iterator();
            iter_rou_ks = BoundConstrainedPLBFGS.rou_ks.iterator();
            for (int i = 0; i < BoundConstrainedPLBFGS.s_ks.size(); ++i) {
                s_k_i = iter_s_ks.next();
                y_k_i = iter_y_ks.next();
                rou_k_i = iter_rou_ks.next();
                b = rou_k_i * Matlab.innerProduct(y_k_i, r);
                InPlaceOperator.plusAssign(r, a[i] - b, s_k_i);
            }
            final Matrix HG = r;
            final Matrix PHG = HG.copy();
            Matlab.logicalIndexingAssignment(PHG, I_b, Matlab.logicalIndexing(HG, I_b));
            Matlab.logicalIndexingAssignment(PHG, I_l, Matlab.min(Matlab.logicalIndexing(HG, I_l), 0.0));
            Matlab.logicalIndexingAssignment(PHG, I_u, Matlab.max(Matlab.logicalIndexing(HG, I_u), 0.0));
            if (Matlab.innerProduct(PHG, BoundConstrainedPLBFGS.G) <= 0.0) {
                BoundConstrainedPLBFGS.p = Matlab.uminus(BoundConstrainedPLBFGS.PG);
            }
            else {
                BoundConstrainedPLBFGS.p = Matlab.uminus(PHG);
            }
            BoundConstrainedPLBFGS.t = 1.0;
            BoundConstrainedPLBFGS.state = 2;
            Matlab.setMatrix(X_t, project(Matlab.plus(BoundConstrainedPLBFGS.X, Matlab.times(BoundConstrainedPLBFGS.t, BoundConstrainedPLBFGS.p)), L, U));
            BoundConstrainedPLBFGS.converge = false;
            BoundConstrainedPLBFGS.gradientRequired = false;
            return new boolean[] { BoundConstrainedPLBFGS.converge, BoundConstrainedPLBFGS.gradientRequired };
        }
        else {
            if (BoundConstrainedPLBFGS.state == 2) {
                BoundConstrainedPLBFGS.converge = false;
                if (fval_t <= BoundConstrainedPLBFGS.fval + BoundConstrainedPLBFGS.alpha * BoundConstrainedPLBFGS.t * Matlab.innerProduct(BoundConstrainedPLBFGS.G, Matlab.minus(X_t, BoundConstrainedPLBFGS.X))) {
                    BoundConstrainedPLBFGS.gradientRequired = true;
                    BoundConstrainedPLBFGS.state = 3;
                }
                else {
                    BoundConstrainedPLBFGS.t *= BoundConstrainedPLBFGS.beta;
                    BoundConstrainedPLBFGS.gradientRequired = false;
                    Matlab.setMatrix(X_t, project(Matlab.plus(BoundConstrainedPLBFGS.X, Matlab.times(BoundConstrainedPLBFGS.t, BoundConstrainedPLBFGS.p)), L, U));
                }
                return new boolean[] { BoundConstrainedPLBFGS.converge, BoundConstrainedPLBFGS.gradientRequired };
            }
            if (BoundConstrainedPLBFGS.state == 3) {
                if (BoundConstrainedPLBFGS.X_pre == null) {
                    BoundConstrainedPLBFGS.X_pre = BoundConstrainedPLBFGS.X.copy();
                }
                else {
                    InPlaceOperator.assign(BoundConstrainedPLBFGS.X_pre, BoundConstrainedPLBFGS.X);
                }
                if (BoundConstrainedPLBFGS.G_pre == null) {
                    BoundConstrainedPLBFGS.G_pre = BoundConstrainedPLBFGS.G.copy();
                }
                else {
                    InPlaceOperator.assign(BoundConstrainedPLBFGS.G_pre, BoundConstrainedPLBFGS.G);
                }
                if (Math.abs(fval_t - BoundConstrainedPLBFGS.fval) < Matlab.eps) {
                    BoundConstrainedPLBFGS.converge = true;
                    BoundConstrainedPLBFGS.gradientRequired = false;
                    System.out.printf("Objective function value doesn't decrease, iteration stopped!\n", new Object[0]);
                    System.out.format("Iter %d, ofv: %g, norm(PGrad): %g\n", BoundConstrainedPLBFGS.k + 1, BoundConstrainedPLBFGS.fval, Matlab.norm(BoundConstrainedPLBFGS.PG, Matlab.inf));
                    return new boolean[] { BoundConstrainedPLBFGS.converge, BoundConstrainedPLBFGS.gradientRequired };
                }
                BoundConstrainedPLBFGS.fval = fval_t;
                BoundConstrainedPLBFGS.J.add(BoundConstrainedPLBFGS.fval);
                System.out.format("Iter %d, ofv: %g, norm(PGrad): %g\n", BoundConstrainedPLBFGS.k + 1, BoundConstrainedPLBFGS.fval, Matlab.norm(BoundConstrainedPLBFGS.PG, Matlab.inf));
                InPlaceOperator.assign(BoundConstrainedPLBFGS.X, X_t);
                InPlaceOperator.assign(BoundConstrainedPLBFGS.G, Grad_t);
                if (BoundConstrainedPLBFGS.k >= BoundConstrainedPLBFGS.m) {
                    BoundConstrainedPLBFGS.s_k = BoundConstrainedPLBFGS.s_ks.removeFirst();
                    BoundConstrainedPLBFGS.y_k = BoundConstrainedPLBFGS.y_ks.removeFirst();
                    BoundConstrainedPLBFGS.rou_ks.removeFirst();
                    InPlaceOperator.minus(BoundConstrainedPLBFGS.s_k, BoundConstrainedPLBFGS.X, BoundConstrainedPLBFGS.X_pre);
                    InPlaceOperator.minus(BoundConstrainedPLBFGS.y_k, BoundConstrainedPLBFGS.G, BoundConstrainedPLBFGS.G_pre);
                }
                else {
                    BoundConstrainedPLBFGS.s_k = BoundConstrainedPLBFGS.X.minus(BoundConstrainedPLBFGS.X_pre);
                    BoundConstrainedPLBFGS.y_k = BoundConstrainedPLBFGS.G.minus(BoundConstrainedPLBFGS.G_pre);
                }
                BoundConstrainedPLBFGS.rou_k = 1.0 / Matlab.innerProduct(BoundConstrainedPLBFGS.y_k, BoundConstrainedPLBFGS.s_k);
                BoundConstrainedPLBFGS.s_ks.add(BoundConstrainedPLBFGS.s_k);
                BoundConstrainedPLBFGS.y_ks.add(BoundConstrainedPLBFGS.y_k);
                BoundConstrainedPLBFGS.rou_ks.add(BoundConstrainedPLBFGS.rou_k);
                ++BoundConstrainedPLBFGS.k;
                BoundConstrainedPLBFGS.state = 1;
            }
            BoundConstrainedPLBFGS.converge = false;
            BoundConstrainedPLBFGS.gradientRequired = false;
            return new boolean[] { BoundConstrainedPLBFGS.converge, BoundConstrainedPLBFGS.gradientRequired };
        }
    }
    
    private static Matrix project(final Matrix A, final double l, final double u) {
        Matlab.logicalIndexingAssignment(A, Matlab.lt(A, l), l);
        Matlab.logicalIndexingAssignment(A, Matlab.gt(A, u), u);
        return A;
    }
    
    private static Matrix project(final Matrix A, final Matrix L, final Matrix U) {
        final Matrix I_l = Matlab.lt(A, L);
        final Matrix I_u = Matlab.gt(A, U);
        Matlab.logicalIndexingAssignment(A, I_l, Matlab.logicalIndexing(L, I_l));
        Matlab.logicalIndexingAssignment(A, I_u, Matlab.logicalIndexing(U, I_u));
        return A;
    }
}
