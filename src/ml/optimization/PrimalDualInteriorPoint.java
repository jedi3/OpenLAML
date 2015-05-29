package ml.optimization;

import la.matrix.*;
import java.util.*;
import ml.utils.*;

public class PrimalDualInteriorPoint
{
    private static Matrix x;
    private static Matrix l;
    private static Matrix v;
    private static boolean gradientRequired;
    private static boolean converge;
    private static int state;
    private static double t;
    private static int k;
    private static int n;
    private static int p;
    private static int m;
    private static double mu;
    private static double epsilon;
    private static double epsilon_feas;
    private static double alpha;
    private static double beta;
    private static double eta_t;
    private static double s;
    private static double residual;
    private static Matrix r_prim;
    private static Matrix r_dual;
    private static Matrix r_cent;
    private static Matrix Matrix;
    private static Matrix Vector;
    private static Matrix z_pd;
    private static Matrix x_nt;
    private static Matrix l_nt;
    private static Matrix v_nt;
    private static Matrix l_s;
    private static Matrix v_s;
    private static ArrayList<Double> J;
    
    static {
        PrimalDualInteriorPoint.x = null;
        PrimalDualInteriorPoint.l = null;
        PrimalDualInteriorPoint.v = null;
        PrimalDualInteriorPoint.gradientRequired = false;
        PrimalDualInteriorPoint.converge = false;
        PrimalDualInteriorPoint.state = 0;
        PrimalDualInteriorPoint.t = 1.0;
        PrimalDualInteriorPoint.k = 0;
        PrimalDualInteriorPoint.n = 0;
        PrimalDualInteriorPoint.p = 0;
        PrimalDualInteriorPoint.m = 0;
        PrimalDualInteriorPoint.mu = 1.8;
        PrimalDualInteriorPoint.epsilon = 1.0E-10;
        PrimalDualInteriorPoint.epsilon_feas = 1.0E-10;
        PrimalDualInteriorPoint.alpha = 0.1;
        PrimalDualInteriorPoint.beta = 0.98;
        PrimalDualInteriorPoint.eta_t = 1.0;
        PrimalDualInteriorPoint.s = 1.0;
        PrimalDualInteriorPoint.residual = 0.0;
        PrimalDualInteriorPoint.r_prim = null;
        PrimalDualInteriorPoint.r_dual = null;
        PrimalDualInteriorPoint.r_cent = null;
        PrimalDualInteriorPoint.Matrix = null;
        PrimalDualInteriorPoint.Vector = null;
        PrimalDualInteriorPoint.z_pd = null;
        PrimalDualInteriorPoint.x_nt = null;
        PrimalDualInteriorPoint.l_nt = null;
        PrimalDualInteriorPoint.v_nt = null;
        PrimalDualInteriorPoint.l_s = null;
        PrimalDualInteriorPoint.v_s = null;
        PrimalDualInteriorPoint.J = new ArrayList<Double>();
    }
    
    public static void main(final String[] args) {
    }
    
    public static Matrix getOptimalLambda() {
        return PrimalDualInteriorPoint.l;
    }
    
    public static Matrix getOptimalNu() {
        return PrimalDualInteriorPoint.v;
    }
    
    public static boolean[] run(final Matrix A, final Matrix b, final Matrix H_x, final Matrix F_x, final Matrix DF_x, final Matrix G_f_x, final double fval, final Matrix x) {
        return run(A, b, H_x, F_x, DF_x, G_f_x, fval, x, null, null);
    }
    
    public static boolean[] run(final Matrix A, final Matrix b, final Matrix H_x, final Matrix F_x, final Matrix DF_x, final Matrix G_f_x, final double fval, final Matrix x_s, final Matrix l_opt, final Matrix v_opt) {
        if (PrimalDualInteriorPoint.state == 5) {
            PrimalDualInteriorPoint.J.clear();
            PrimalDualInteriorPoint.state = 0;
        }
        if (PrimalDualInteriorPoint.state == 0) {
            Time.tic();
            PrimalDualInteriorPoint.n = A.getColumnDimension();
            PrimalDualInteriorPoint.p = A.getRowDimension();
            PrimalDualInteriorPoint.m = F_x.getRowDimension();
            PrimalDualInteriorPoint.x = x_s.copy();
            PrimalDualInteriorPoint.l = Matlab.rdivide(Matlab.ones(PrimalDualInteriorPoint.m, 1), PrimalDualInteriorPoint.m);
            PrimalDualInteriorPoint.v = Matlab.zeros(PrimalDualInteriorPoint.p, 1);
            PrimalDualInteriorPoint.l_s = PrimalDualInteriorPoint.l.copy();
            PrimalDualInteriorPoint.v_s = PrimalDualInteriorPoint.v.copy();
            PrimalDualInteriorPoint.eta_t = -Matlab.innerProduct(F_x, PrimalDualInteriorPoint.l);
            PrimalDualInteriorPoint.k = 0;
            PrimalDualInteriorPoint.state = 1;
        }
        if (PrimalDualInteriorPoint.state == 1) {
            double residual_prim = 0.0;
            double residual_dual = 0.0;
            PrimalDualInteriorPoint.t = PrimalDualInteriorPoint.mu * PrimalDualInteriorPoint.m / PrimalDualInteriorPoint.eta_t;
            PrimalDualInteriorPoint.r_prim = A.mtimes(PrimalDualInteriorPoint.x).minus(b);
            PrimalDualInteriorPoint.r_dual = G_f_x.plus(DF_x.transpose().mtimes(PrimalDualInteriorPoint.l)).plus(A.transpose().mtimes(PrimalDualInteriorPoint.v));
            PrimalDualInteriorPoint.r_cent = Matlab.uminus(Matlab.times(PrimalDualInteriorPoint.l, F_x)).minus(Matlab.rdivide(Matlab.ones(PrimalDualInteriorPoint.m, 1), PrimalDualInteriorPoint.t));
            PrimalDualInteriorPoint.Matrix = Matlab.vertcat(Matlab.horzcat(H_x, DF_x.transpose(), A.transpose()), Matlab.horzcat(Matlab.uminus(Matlab.mtimes(Matlab.diag(PrimalDualInteriorPoint.l), DF_x)), Matlab.uminus(Matlab.diag(F_x)), Matlab.zeros(PrimalDualInteriorPoint.m, PrimalDualInteriorPoint.p)), Matlab.horzcat(A, Matlab.zeros(PrimalDualInteriorPoint.p, PrimalDualInteriorPoint.m), Matlab.zeros(PrimalDualInteriorPoint.p, PrimalDualInteriorPoint.p)));
            PrimalDualInteriorPoint.Vector = Matlab.uminus(Matlab.vertcat(PrimalDualInteriorPoint.r_dual, PrimalDualInteriorPoint.r_cent, PrimalDualInteriorPoint.r_prim));
            PrimalDualInteriorPoint.residual = Matlab.norm(PrimalDualInteriorPoint.Vector);
            residual_prim = Matlab.norm(PrimalDualInteriorPoint.r_prim);
            residual_dual = Matlab.norm(PrimalDualInteriorPoint.r_dual);
            PrimalDualInteriorPoint.eta_t = -Matlab.innerProduct(F_x, PrimalDualInteriorPoint.l);
            if (residual_prim <= PrimalDualInteriorPoint.epsilon_feas && residual_dual <= PrimalDualInteriorPoint.epsilon_feas && PrimalDualInteriorPoint.eta_t <= PrimalDualInteriorPoint.epsilon) {
                Printer.fprintf("Terminate successfully.\n\n", new Object[0]);
                if (l_opt != null) {
                    Matlab.setMatrix(l_opt, PrimalDualInteriorPoint.l);
                }
                if (v_opt != null) {
                    Matlab.setMatrix(v_opt, PrimalDualInteriorPoint.v);
                }
                PrimalDualInteriorPoint.converge = true;
                PrimalDualInteriorPoint.gradientRequired = false;
                PrimalDualInteriorPoint.state = 5;
                System.out.printf("Primal-dual interior-point algorithm converges.\n", new Object[0]);
                return new boolean[] { PrimalDualInteriorPoint.converge, PrimalDualInteriorPoint.gradientRequired };
            }
            PrimalDualInteriorPoint.z_pd = Matlab.mldivide(PrimalDualInteriorPoint.Matrix, PrimalDualInteriorPoint.Vector);
            PrimalDualInteriorPoint.x_nt = Matlab.getRows(PrimalDualInteriorPoint.z_pd, 0, PrimalDualInteriorPoint.n - 1);
            PrimalDualInteriorPoint.l_nt = Matlab.getRows(PrimalDualInteriorPoint.z_pd, PrimalDualInteriorPoint.n, PrimalDualInteriorPoint.n + PrimalDualInteriorPoint.m - 1);
            PrimalDualInteriorPoint.v_nt = Matlab.getRows(PrimalDualInteriorPoint.z_pd, PrimalDualInteriorPoint.n + PrimalDualInteriorPoint.m, PrimalDualInteriorPoint.n + PrimalDualInteriorPoint.m + PrimalDualInteriorPoint.p - 1);
            PrimalDualInteriorPoint.s = 1.0;
            while (true) {
                InPlaceOperator.affine(PrimalDualInteriorPoint.l_s, PrimalDualInteriorPoint.s, PrimalDualInteriorPoint.l_nt, '+', PrimalDualInteriorPoint.l);
                if (Matlab.sumAll(Matlab.lt(PrimalDualInteriorPoint.l_s, 0.0)) <= 0.0) {
                    break;
                }
                PrimalDualInteriorPoint.s *= PrimalDualInteriorPoint.beta;
            }
            PrimalDualInteriorPoint.state = 2;
            InPlaceOperator.affine(x_s, PrimalDualInteriorPoint.s, PrimalDualInteriorPoint.x_nt, '+', PrimalDualInteriorPoint.x);
            PrimalDualInteriorPoint.converge = false;
            PrimalDualInteriorPoint.gradientRequired = false;
            return new boolean[] { PrimalDualInteriorPoint.converge, PrimalDualInteriorPoint.gradientRequired };
        }
        else {
            if (PrimalDualInteriorPoint.state != 2) {
                if (PrimalDualInteriorPoint.state == 3) {
                    Matrix r_prim_s = null;
                    Matrix r_dual_s = null;
                    Matrix r_cent_s = null;
                    double residual_s = 0.0;
                    InPlaceOperator.affine(PrimalDualInteriorPoint.l_s, PrimalDualInteriorPoint.s, PrimalDualInteriorPoint.l_nt, '+', PrimalDualInteriorPoint.l);
                    InPlaceOperator.affine(PrimalDualInteriorPoint.v_s, PrimalDualInteriorPoint.s, PrimalDualInteriorPoint.v_nt, '+', PrimalDualInteriorPoint.v);
                    r_prim_s = A.mtimes(x_s).minus(b);
                    r_dual_s = G_f_x.plus(DF_x.transpose().mtimes(PrimalDualInteriorPoint.l_s)).plus(A.transpose().mtimes(PrimalDualInteriorPoint.v_s));
                    r_cent_s = Matlab.uminus(Matlab.times(PrimalDualInteriorPoint.l_s, F_x)).minus(Matlab.rdivide(Matlab.ones(PrimalDualInteriorPoint.m, 1), PrimalDualInteriorPoint.t));
                    residual_s = Matlab.norm(Matlab.vertcat(r_dual_s, r_cent_s, r_prim_s));
                    if (residual_s > (1.0 - PrimalDualInteriorPoint.alpha * PrimalDualInteriorPoint.s) * PrimalDualInteriorPoint.residual) {
                        PrimalDualInteriorPoint.s *= PrimalDualInteriorPoint.beta;
                        PrimalDualInteriorPoint.converge = false;
                        PrimalDualInteriorPoint.gradientRequired = true;
                        InPlaceOperator.affine(x_s, PrimalDualInteriorPoint.s, PrimalDualInteriorPoint.x_nt, '+', PrimalDualInteriorPoint.x);
                        return new boolean[] { PrimalDualInteriorPoint.converge, PrimalDualInteriorPoint.gradientRequired };
                    }
                    InPlaceOperator.assign(PrimalDualInteriorPoint.l, PrimalDualInteriorPoint.l_s);
                    InPlaceOperator.assign(PrimalDualInteriorPoint.v, PrimalDualInteriorPoint.v_s);
                    InPlaceOperator.assign(PrimalDualInteriorPoint.x, x_s);
                    PrimalDualInteriorPoint.state = 4;
                }
                if (PrimalDualInteriorPoint.state == 4) {
                    ++PrimalDualInteriorPoint.k;
                    PrimalDualInteriorPoint.state = 1;
                }
                PrimalDualInteriorPoint.converge = false;
                PrimalDualInteriorPoint.gradientRequired = true;
                return new boolean[] { PrimalDualInteriorPoint.converge, PrimalDualInteriorPoint.gradientRequired };
            }
            if (Matlab.sumAll(Matlab.gt(F_x, 0.0)) > 0.0) {
                InPlaceOperator.affine(x_s, PrimalDualInteriorPoint.s *= PrimalDualInteriorPoint.beta, PrimalDualInteriorPoint.x_nt, '+', PrimalDualInteriorPoint.x);
                PrimalDualInteriorPoint.converge = false;
                PrimalDualInteriorPoint.gradientRequired = false;
                return new boolean[] { PrimalDualInteriorPoint.converge, PrimalDualInteriorPoint.gradientRequired };
            }
            PrimalDualInteriorPoint.state = 3;
            PrimalDualInteriorPoint.converge = false;
            PrimalDualInteriorPoint.gradientRequired = true;
            return new boolean[] { PrimalDualInteriorPoint.converge, PrimalDualInteriorPoint.gradientRequired };
        }
    }
}
