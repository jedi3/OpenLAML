package ml.optimization;

import la.matrix.*;
import java.util.*;
import la.vector.*;
import ml.utils.*;

public class AcceleratedGradientDescent
{
    private static ProximalMapping prox;
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
    private static int type;
    private static ArrayList<Double> J;
    
    static {
        AcceleratedGradientDescent.prox = new Prox();
        AcceleratedGradientDescent.Grad_Y_k = null;
        AcceleratedGradientDescent.X = null;
        AcceleratedGradientDescent.X_pre = null;
        AcceleratedGradientDescent.Y = null;
        AcceleratedGradientDescent.G_Y_k = null;
        AcceleratedGradientDescent.gval_Y_k = 0.0;
        AcceleratedGradientDescent.hval_Y_k = 0.0;
        AcceleratedGradientDescent.fval_Y_k = 0.0;
        AcceleratedGradientDescent.gradientRequired = false;
        AcceleratedGradientDescent.converge = false;
        AcceleratedGradientDescent.state = 0;
        AcceleratedGradientDescent.t = 1.0;
        AcceleratedGradientDescent.beta = 0.95;
        AcceleratedGradientDescent.k = 1;
        AcceleratedGradientDescent.xi = 1.0;
        AcceleratedGradientDescent.type = 0;
        AcceleratedGradientDescent.J = new ArrayList<Double>();
    }
    
    public static void main(final String[] args) {
        final int n = 10;
        final Matrix t = Matlab.rand(n);
        final Matrix C = Matlab.minus(t.mtimes(t.transpose()), Matlab.times(0.1, Matlab.eye(n)));
        final Matrix y = Matlab.times(3.0, Matlab.minus(0.5, Matlab.rand(n, 1)));
        final double epsilon = 1.0E-4;
        final double gamma = 0.01;
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
    
    public static boolean[] run(final Matrix Grad_t, final double gval_t, final double epsilon, final Matrix X_t) {
        if (AcceleratedGradientDescent.state == 4) {
            AcceleratedGradientDescent.J.clear();
            AcceleratedGradientDescent.X_pre = null;
            AcceleratedGradientDescent.t = 1.0;
            AcceleratedGradientDescent.k = 1;
            AcceleratedGradientDescent.state = 0;
        }
        if (AcceleratedGradientDescent.state == 0) {
            AcceleratedGradientDescent.X = X_t.copy();
            AcceleratedGradientDescent.Y = X_t.copy();
            AcceleratedGradientDescent.gval_Y_k = gval_t;
            AcceleratedGradientDescent.hval_Y_k = 0.0;
            AcceleratedGradientDescent.fval_Y_k = AcceleratedGradientDescent.gval_Y_k + AcceleratedGradientDescent.hval_Y_k;
            if (Double.isNaN(AcceleratedGradientDescent.fval_Y_k)) {
                System.err.println("Object function value is nan!");
                System.exit(1);
            }
            System.out.format("Initial ofv: %g\n", AcceleratedGradientDescent.fval_Y_k);
            AcceleratedGradientDescent.k = 1;
            AcceleratedGradientDescent.xi = 1.0;
            AcceleratedGradientDescent.t = 1.0;
            AcceleratedGradientDescent.state = 1;
        }
        if (AcceleratedGradientDescent.state == 1) {
            if (Grad_t == null) {
                System.err.println("Gradient is required!");
                System.exit(1);
            }
            AcceleratedGradientDescent.Grad_Y_k = Grad_t.copy();
            AcceleratedGradientDescent.gval_Y_k = gval_t;
            AcceleratedGradientDescent.hval_Y_k = 0.0;
            AcceleratedGradientDescent.state = 2;
            AcceleratedGradientDescent.prox.compute(X_t, AcceleratedGradientDescent.t, Matlab.minus(AcceleratedGradientDescent.Y, Matlab.times(AcceleratedGradientDescent.t, AcceleratedGradientDescent.Grad_Y_k)));
            AcceleratedGradientDescent.G_Y_k = Matlab.rdivide(Matlab.minus(AcceleratedGradientDescent.Y, X_t), AcceleratedGradientDescent.t);
            AcceleratedGradientDescent.converge = false;
            AcceleratedGradientDescent.gradientRequired = false;
            return new boolean[] { AcceleratedGradientDescent.converge, AcceleratedGradientDescent.gradientRequired };
        }
        if (AcceleratedGradientDescent.state == 2) {
            AcceleratedGradientDescent.converge = false;
            if (gval_t > AcceleratedGradientDescent.gval_Y_k - AcceleratedGradientDescent.t * Matlab.innerProduct(AcceleratedGradientDescent.Grad_Y_k, AcceleratedGradientDescent.G_Y_k) + AcceleratedGradientDescent.t / 2.0 * Matlab.innerProduct(AcceleratedGradientDescent.G_Y_k, AcceleratedGradientDescent.G_Y_k) + Matlab.eps) {
                AcceleratedGradientDescent.t *= AcceleratedGradientDescent.beta;
                AcceleratedGradientDescent.gradientRequired = false;
                Matlab.setMatrix(X_t, AcceleratedGradientDescent.prox.compute(AcceleratedGradientDescent.t, Matlab.minus(AcceleratedGradientDescent.Y, Matlab.times(AcceleratedGradientDescent.t, AcceleratedGradientDescent.Grad_Y_k))));
                AcceleratedGradientDescent.G_Y_k = Matlab.rdivide(Matlab.minus(AcceleratedGradientDescent.Y, X_t), AcceleratedGradientDescent.t);
                return new boolean[] { AcceleratedGradientDescent.converge, AcceleratedGradientDescent.gradientRequired };
            }
            AcceleratedGradientDescent.gradientRequired = true;
            AcceleratedGradientDescent.state = 3;
        }
        if (AcceleratedGradientDescent.state == 3) {
            final double norm_G_Y = Matlab.norm(AcceleratedGradientDescent.G_Y_k);
            if (norm_G_Y < epsilon) {
                AcceleratedGradientDescent.converge = true;
                AcceleratedGradientDescent.gradientRequired = false;
                AcceleratedGradientDescent.state = 4;
                System.out.printf("Accelerated gradient descent method converges with norm(G_Y_k) %f\n", norm_G_Y);
                return new boolean[] { AcceleratedGradientDescent.converge, AcceleratedGradientDescent.gradientRequired };
            }
            AcceleratedGradientDescent.fval_Y_k = AcceleratedGradientDescent.gval_Y_k + AcceleratedGradientDescent.hval_Y_k;
            AcceleratedGradientDescent.J.add(AcceleratedGradientDescent.fval_Y_k);
            System.out.format("Iter %d, ofv: %g, norm(G_Y_k): %g\n", AcceleratedGradientDescent.k, AcceleratedGradientDescent.fval_Y_k, Matlab.norm(AcceleratedGradientDescent.G_Y_k));
            if (AcceleratedGradientDescent.X_pre == null) {
                AcceleratedGradientDescent.X_pre = AcceleratedGradientDescent.X.copy();
            }
            else {
                InPlaceOperator.assign(AcceleratedGradientDescent.X_pre, AcceleratedGradientDescent.X);
            }
            InPlaceOperator.assign(AcceleratedGradientDescent.X, X_t);
            if (AcceleratedGradientDescent.type == 0) {
                final double s = AcceleratedGradientDescent.k / (AcceleratedGradientDescent.k + 3);
                InPlaceOperator.affine(AcceleratedGradientDescent.Y, 1.0 + s, AcceleratedGradientDescent.X, -s, AcceleratedGradientDescent.X_pre);
            }
            else if (AcceleratedGradientDescent.type == 1) {
                final double u = 2.0 * (AcceleratedGradientDescent.xi - 1.0) / (1.0 + Math.sqrt(1.0 + 4.0 * AcceleratedGradientDescent.xi * AcceleratedGradientDescent.xi));
                InPlaceOperator.affine(AcceleratedGradientDescent.Y, 1.0 + u, AcceleratedGradientDescent.X, -u, AcceleratedGradientDescent.X_pre);
                AcceleratedGradientDescent.xi = (1.0 + Math.sqrt(1.0 + 4.0 * AcceleratedGradientDescent.xi * AcceleratedGradientDescent.xi)) / 2.0;
            }
            InPlaceOperator.assign(X_t, AcceleratedGradientDescent.Y);
            ++AcceleratedGradientDescent.k;
            AcceleratedGradientDescent.state = 1;
        }
        AcceleratedGradientDescent.converge = false;
        AcceleratedGradientDescent.gradientRequired = true;
        return new boolean[] { AcceleratedGradientDescent.converge, AcceleratedGradientDescent.gradientRequired };
    }
}
