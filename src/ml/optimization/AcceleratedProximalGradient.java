package ml.optimization;

import la.matrix.*;
import java.util.*;
import la.vector.*;
import ml.utils.*;

public class AcceleratedProximalGradient
{
    public static ProximalMapping prox;
    private static Matrix Grad_Y_k;
    private static Matrix X;
    private static Matrix X_pre;
    private static Matrix Y;
    private static Matrix G_Y_k;
    private static double gval_Y_k;
    private static double hval_Y_k;
    private static double fval_Y_k;
    private static boolean gradientRequired;
    private static boolean converge;
    private static int state;
    private static double t;
    private static double beta;
    private static int k;
    private static double xi;
    public static int type;
    private static Matrix T;
    private static ArrayList<Double> J;
    
    static {
        AcceleratedProximalGradient.prox = null;
        AcceleratedProximalGradient.Grad_Y_k = null;
        AcceleratedProximalGradient.X = null;
        AcceleratedProximalGradient.X_pre = null;
        AcceleratedProximalGradient.Y = null;
        AcceleratedProximalGradient.G_Y_k = null;
        AcceleratedProximalGradient.gval_Y_k = 0.0;
        AcceleratedProximalGradient.hval_Y_k = 0.0;
        AcceleratedProximalGradient.fval_Y_k = 0.0;
        AcceleratedProximalGradient.gradientRequired = false;
        AcceleratedProximalGradient.converge = false;
        AcceleratedProximalGradient.state = 0;
        AcceleratedProximalGradient.t = 1.0;
        AcceleratedProximalGradient.beta = 0.95;
        AcceleratedProximalGradient.k = 1;
        AcceleratedProximalGradient.xi = 1.0;
        AcceleratedProximalGradient.type = 0;
        AcceleratedProximalGradient.T = null;
        AcceleratedProximalGradient.J = new ArrayList<Double>();
    }
    
    public static void main(final String[] args) {
        final int n = 10;
        final Matrix t = Matlab.rand(n);
        final Matrix C = Matlab.minus(t.mtimes(t.transpose()), Matlab.times(0.1, Matlab.eye(n)));
        final Matrix y = Matlab.times(3.0, Matlab.minus(0.5, Matlab.rand(n, 1)));
        final double epsilon = 1.0E-4;
        final double gamma = 0.01;
        AcceleratedProximalGradient.prox = new ProxPlus();
        AcceleratedProximalGradient.type = 0;
        final long start = System.currentTimeMillis();
        final Matrix x0 = Matlab.rdivide(Matlab.ones(n, 1), n);
        final Matrix x = x0.copy();
        Matrix r_x = null;
        double f_x = 0.0;
        double phi_x = 0.0;
        double gval = 0.0;
        double hval = 0.0;
        double fval = 0.0;
        r_x = C.mtimes(x).minus(y);
        f_x = Matlab.norm(r_x);
        phi_x = Matlab.norm(x);
        gval = f_x + gamma * phi_x;
        hval = 0.0;
        fval = gval + hval;
        Matrix Grad_f_x = null;
        Matrix Grad_phi_x = null;
        Matrix Grad = null;
        Grad_f_x = Matlab.rdivide(C.transpose().mtimes(r_x), f_x);
        Grad_phi_x = Matlab.rdivide(x, phi_x);
        Grad = Matlab.plus(Grad_f_x, Matlab.times(gamma, Grad_phi_x));
        boolean[] flags = null;
        int k = 0;
        final int maxIter = 10000;
        hval = 0.0;
        while (true) {
            flags = run(Grad, gval, hval, epsilon, x);
            if (flags[0]) {
                break;
            }
            if (Matlab.sum(Matlab.sum(Matlab.isnan(x))) > 0.0) {
                int a = 1;
                ++a;
            }
            InPlaceOperator.affine(r_x, C, x, '-', y);
            f_x = Matlab.norm(r_x);
            phi_x = Matlab.norm(x);
            gval = f_x + gamma * phi_x;
            hval = 0.0;
            fval = gval + hval;
            if (!flags[1]) {
                continue;
            }
            if (++k > maxIter) {
                break;
            }
            Grad_f_x = Matlab.rdivide(C.transpose().mtimes(r_x), f_x);
            if (phi_x != 0.0) {
                Grad_phi_x = Matlab.rdivide(x, phi_x);
            }
            else {
                Grad_phi_x = Matlab.times(0.0, Grad_phi_x);
            }
            Grad = Matlab.plus(Grad_f_x, Matlab.times(gamma, Grad_phi_x));
        }
        final Matrix x_accelerated_proximal_gradient = x;
        final double f_accelerated_proximal_gradient = fval;
        Printer.fprintf("fval_accelerated_proximal_gradient: %g\n\n", f_accelerated_proximal_gradient);
        Printer.fprintf("x_accelerated_proximal_gradient:\n", new Object[0]);
        Printer.display(x_accelerated_proximal_gradient.transpose());
        final double elapsedTime = (System.currentTimeMillis() - start) / 1000.0;
        Printer.fprintf("Elapsed time: %.3f seconds\n", elapsedTime);
    }
    
    public static boolean[] run(final Matrix Grad_t, final double gval_t, final double hval_t, final double epsilon, final Matrix X_t) {
        if (AcceleratedProximalGradient.state == 4) {
            AcceleratedProximalGradient.J.clear();
            AcceleratedProximalGradient.X = null;
            AcceleratedProximalGradient.X_pre = null;
            AcceleratedProximalGradient.Grad_Y_k = null;
            AcceleratedProximalGradient.G_Y_k = null;
            AcceleratedProximalGradient.T = null;
            AcceleratedProximalGradient.t = 1.0;
            AcceleratedProximalGradient.k = 1;
            AcceleratedProximalGradient.state = 0;
        }
        if (AcceleratedProximalGradient.state == 0) {
            AcceleratedProximalGradient.X = X_t.copy();
            AcceleratedProximalGradient.Y = X_t.copy();
            AcceleratedProximalGradient.T = X_t.copy();
            AcceleratedProximalGradient.gval_Y_k = gval_t;
            AcceleratedProximalGradient.hval_Y_k = hval_t;
            AcceleratedProximalGradient.fval_Y_k = AcceleratedProximalGradient.gval_Y_k + AcceleratedProximalGradient.hval_Y_k;
            if (Double.isNaN(AcceleratedProximalGradient.fval_Y_k)) {
                System.err.println("Object function value is nan!");
                System.exit(1);
            }
            System.out.format("Initial ofv: %g\n", AcceleratedProximalGradient.fval_Y_k);
            AcceleratedProximalGradient.k = 1;
            AcceleratedProximalGradient.xi = 1.0;
            AcceleratedProximalGradient.t = 1.0;
            AcceleratedProximalGradient.state = 1;
        }
        if (AcceleratedProximalGradient.state == 1) {
            if (Grad_t == null) {
                System.err.println("Gradient is required!");
                System.exit(1);
            }
            if (AcceleratedProximalGradient.Grad_Y_k == null) {
                AcceleratedProximalGradient.Grad_Y_k = Grad_t.copy();
            }
            else {
                InPlaceOperator.assign(AcceleratedProximalGradient.Grad_Y_k, Grad_t);
            }
            AcceleratedProximalGradient.gval_Y_k = gval_t;
            AcceleratedProximalGradient.hval_Y_k = hval_t;
            AcceleratedProximalGradient.state = 2;
            InPlaceOperator.affine(AcceleratedProximalGradient.T, AcceleratedProximalGradient.Y, -AcceleratedProximalGradient.t, AcceleratedProximalGradient.Grad_Y_k);
            AcceleratedProximalGradient.prox.compute(X_t, AcceleratedProximalGradient.t, AcceleratedProximalGradient.T);
            if (AcceleratedProximalGradient.G_Y_k == null) {
                AcceleratedProximalGradient.G_Y_k = Matlab.rdivide(Matlab.minus(AcceleratedProximalGradient.Y, X_t), AcceleratedProximalGradient.t);
            }
            else {
                InPlaceOperator.affine(AcceleratedProximalGradient.G_Y_k, 1.0 / AcceleratedProximalGradient.t, AcceleratedProximalGradient.Y, -1.0 / AcceleratedProximalGradient.t, X_t);
            }
            AcceleratedProximalGradient.converge = false;
            AcceleratedProximalGradient.gradientRequired = false;
            return new boolean[] { AcceleratedProximalGradient.converge, AcceleratedProximalGradient.gradientRequired };
        }
        if (AcceleratedProximalGradient.state == 2) {
            AcceleratedProximalGradient.converge = false;
            if (gval_t > AcceleratedProximalGradient.gval_Y_k - AcceleratedProximalGradient.t * Matlab.innerProduct(AcceleratedProximalGradient.Grad_Y_k, AcceleratedProximalGradient.G_Y_k) + AcceleratedProximalGradient.t / 2.0 * Matlab.innerProduct(AcceleratedProximalGradient.G_Y_k, AcceleratedProximalGradient.G_Y_k) + Matlab.eps) {
                AcceleratedProximalGradient.t *= AcceleratedProximalGradient.beta;
                AcceleratedProximalGradient.gradientRequired = false;
                InPlaceOperator.affine(AcceleratedProximalGradient.T, AcceleratedProximalGradient.Y, -AcceleratedProximalGradient.t, AcceleratedProximalGradient.Grad_Y_k);
                AcceleratedProximalGradient.prox.compute(X_t, AcceleratedProximalGradient.t, AcceleratedProximalGradient.T);
                InPlaceOperator.affine(AcceleratedProximalGradient.G_Y_k, 1.0 / AcceleratedProximalGradient.t, AcceleratedProximalGradient.Y, -1.0 / AcceleratedProximalGradient.t, X_t);
                return new boolean[] { AcceleratedProximalGradient.converge, AcceleratedProximalGradient.gradientRequired };
            }
            AcceleratedProximalGradient.gradientRequired = true;
            AcceleratedProximalGradient.state = 3;
        }
        if (AcceleratedProximalGradient.state == 3) {
            final double norm_G_Y = Matlab.norm(AcceleratedProximalGradient.G_Y_k);
            if (norm_G_Y < epsilon) {
                AcceleratedProximalGradient.converge = true;
                AcceleratedProximalGradient.gradientRequired = false;
                AcceleratedProximalGradient.state = 4;
                System.out.printf("Accelerated proximal gradient method converges with norm(G_Y_k) %f\n", norm_G_Y);
                return new boolean[] { AcceleratedProximalGradient.converge, AcceleratedProximalGradient.gradientRequired };
            }
            AcceleratedProximalGradient.fval_Y_k = AcceleratedProximalGradient.gval_Y_k + AcceleratedProximalGradient.hval_Y_k;
            AcceleratedProximalGradient.J.add(AcceleratedProximalGradient.fval_Y_k);
            System.out.format("Iter %d, ofv: %g, norm(G_Y_k): %g\n", AcceleratedProximalGradient.k, AcceleratedProximalGradient.fval_Y_k, Matlab.norm(AcceleratedProximalGradient.G_Y_k));
            if (AcceleratedProximalGradient.X_pre == null) {
                AcceleratedProximalGradient.X_pre = AcceleratedProximalGradient.X.copy();
            }
            else {
                InPlaceOperator.assign(AcceleratedProximalGradient.X_pre, AcceleratedProximalGradient.X);
            }
            InPlaceOperator.assign(AcceleratedProximalGradient.X, X_t);
            if (AcceleratedProximalGradient.type == 0) {
                final double s = AcceleratedProximalGradient.k / (AcceleratedProximalGradient.k + 3);
                InPlaceOperator.affine(AcceleratedProximalGradient.Y, 1.0 + s, AcceleratedProximalGradient.X, -s, AcceleratedProximalGradient.X_pre);
            }
            else if (AcceleratedProximalGradient.type == 1) {
                final double u = 2.0 * (AcceleratedProximalGradient.xi - 1.0) / (1.0 + Math.sqrt(1.0 + 4.0 * AcceleratedProximalGradient.xi * AcceleratedProximalGradient.xi));
                InPlaceOperator.affine(AcceleratedProximalGradient.Y, 1.0 + u, AcceleratedProximalGradient.X, -u, AcceleratedProximalGradient.X_pre);
                AcceleratedProximalGradient.xi = (1.0 + Math.sqrt(1.0 + 4.0 * AcceleratedProximalGradient.xi * AcceleratedProximalGradient.xi)) / 2.0;
            }
            InPlaceOperator.assign(X_t, AcceleratedProximalGradient.Y);
            ++AcceleratedProximalGradient.k;
            AcceleratedProximalGradient.state = 1;
        }
        AcceleratedProximalGradient.converge = false;
        AcceleratedProximalGradient.gradientRequired = true;
        return new boolean[] { AcceleratedProximalGradient.converge, AcceleratedProximalGradient.gradientRequired };
    }
}
