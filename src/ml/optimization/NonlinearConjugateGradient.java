package ml.optimization;

import la.matrix.*;
import java.util.*;
import ml.utils.*;

public class NonlinearConjugateGradient
{
    private static Matrix G;
    private static Matrix G_pre;
    private static Matrix X;
    private static Matrix p;
    private static double fval;
    private static boolean gradientRequired;
    private static boolean converge;
    private static int state;
    private static double t;
    private static double z;
    private static int k;
    private static double alpha;
    private static double rou;
    private static int formula;
    private static ArrayList<Double> J;
    
    static {
        NonlinearConjugateGradient.G = null;
        NonlinearConjugateGradient.G_pre = null;
        NonlinearConjugateGradient.X = null;
        NonlinearConjugateGradient.p = null;
        NonlinearConjugateGradient.fval = 0.0;
        NonlinearConjugateGradient.gradientRequired = false;
        NonlinearConjugateGradient.converge = false;
        NonlinearConjugateGradient.state = 0;
        NonlinearConjugateGradient.t = 1.0;
        NonlinearConjugateGradient.z = 0.0;
        NonlinearConjugateGradient.k = 0;
        NonlinearConjugateGradient.alpha = 0.05;
        NonlinearConjugateGradient.rou = 0.9;
        NonlinearConjugateGradient.formula = 4;
        NonlinearConjugateGradient.J = new ArrayList<Double>();
    }
    
    public static boolean[] run(final Matrix Grad_t, final double fval_t, final double epsilon, final Matrix X_t) {
        if (NonlinearConjugateGradient.state == 4) {
            NonlinearConjugateGradient.G_pre = null;
            NonlinearConjugateGradient.J.clear();
            NonlinearConjugateGradient.k = 0;
            NonlinearConjugateGradient.state = 0;
        }
        if (NonlinearConjugateGradient.state == 0) {
            NonlinearConjugateGradient.X = X_t.copy();
            if (Grad_t == null) {
                System.err.println("Gradient is required on the first call!");
                System.exit(1);
            }
            NonlinearConjugateGradient.G = Grad_t.copy();
            NonlinearConjugateGradient.fval = fval_t;
            if (Double.isNaN(NonlinearConjugateGradient.fval)) {
                System.err.println("Object function value is nan!");
                System.exit(1);
            }
            System.out.format("Initial ofv: %g\n", NonlinearConjugateGradient.fval);
            NonlinearConjugateGradient.p = Matlab.uminus(NonlinearConjugateGradient.G);
            NonlinearConjugateGradient.state = 1;
        }
        if (NonlinearConjugateGradient.state == 1) {
            final double norm_Grad = Matlab.norm(NonlinearConjugateGradient.G);
            if (norm_Grad < epsilon) {
                NonlinearConjugateGradient.converge = true;
                NonlinearConjugateGradient.gradientRequired = false;
                NonlinearConjugateGradient.state = 4;
                System.out.printf("CG converges with norm(Grad) %f\n", norm_Grad);
                return new boolean[] { NonlinearConjugateGradient.converge, NonlinearConjugateGradient.gradientRequired };
            }
            NonlinearConjugateGradient.t = 1.0;
            NonlinearConjugateGradient.z = Matlab.innerProduct(NonlinearConjugateGradient.G, NonlinearConjugateGradient.p);
            NonlinearConjugateGradient.state = 2;
            Matlab.setMatrix(X_t, Matlab.plus(NonlinearConjugateGradient.X, Matlab.times(NonlinearConjugateGradient.t, NonlinearConjugateGradient.p)));
            NonlinearConjugateGradient.converge = false;
            NonlinearConjugateGradient.gradientRequired = false;
            return new boolean[] { NonlinearConjugateGradient.converge, NonlinearConjugateGradient.gradientRequired };
        }
        else {
            if (NonlinearConjugateGradient.state == 2) {
                NonlinearConjugateGradient.converge = false;
                if (fval_t <= NonlinearConjugateGradient.fval + NonlinearConjugateGradient.alpha * NonlinearConjugateGradient.t * NonlinearConjugateGradient.z) {
                    NonlinearConjugateGradient.gradientRequired = true;
                    NonlinearConjugateGradient.state = 3;
                }
                else {
                    NonlinearConjugateGradient.t *= NonlinearConjugateGradient.rou;
                    NonlinearConjugateGradient.gradientRequired = false;
                    InPlaceOperator.affine(X_t, NonlinearConjugateGradient.X, NonlinearConjugateGradient.t, NonlinearConjugateGradient.p);
                }
                return new boolean[] { NonlinearConjugateGradient.converge, NonlinearConjugateGradient.gradientRequired };
            }
            if (NonlinearConjugateGradient.state == 3) {
                if (NonlinearConjugateGradient.G_pre == null) {
                    NonlinearConjugateGradient.G_pre = NonlinearConjugateGradient.G.copy();
                }
                else {
                    InPlaceOperator.assign(NonlinearConjugateGradient.G_pre, NonlinearConjugateGradient.G);
                }
                NonlinearConjugateGradient.fval = fval_t;
                NonlinearConjugateGradient.J.add(NonlinearConjugateGradient.fval);
                System.out.format("Iter %d, ofv: %g, norm(Grad): %g\n", NonlinearConjugateGradient.k + 1, NonlinearConjugateGradient.fval, Matlab.norm(NonlinearConjugateGradient.G));
                InPlaceOperator.assign(NonlinearConjugateGradient.X, X_t);
                InPlaceOperator.assign(NonlinearConjugateGradient.G, Grad_t);
                Matrix y_k = null;
                y_k = Matlab.minus(NonlinearConjugateGradient.G, NonlinearConjugateGradient.G_pre);
                double beta = 0.0;
                switch (NonlinearConjugateGradient.formula) {
                    case 1: {
                        beta = Matlab.innerProduct(NonlinearConjugateGradient.G, NonlinearConjugateGradient.G) / Matlab.innerProduct(NonlinearConjugateGradient.G_pre, NonlinearConjugateGradient.G);
                        break;
                    }
                    case 2: {
                        beta = Matlab.innerProduct(NonlinearConjugateGradient.G, y_k) / Matlab.innerProduct(NonlinearConjugateGradient.G_pre, NonlinearConjugateGradient.G_pre);
                        break;
                    }
                    case 3: {
                        beta = Math.max(Matlab.innerProduct(NonlinearConjugateGradient.G, y_k) / Matlab.innerProduct(NonlinearConjugateGradient.G_pre, NonlinearConjugateGradient.G_pre), 0.0);
                        break;
                    }
                    case 4: {
                        beta = Matlab.innerProduct(NonlinearConjugateGradient.G, y_k) / Matlab.innerProduct(y_k, NonlinearConjugateGradient.p);
                        break;
                    }
                    case 5: {
                        beta = Matlab.innerProduct(NonlinearConjugateGradient.G, NonlinearConjugateGradient.G) / Matlab.innerProduct(y_k, NonlinearConjugateGradient.p);
                        break;
                    }
                    default: {
                        beta = Matlab.innerProduct(NonlinearConjugateGradient.G, y_k) / Matlab.innerProduct(y_k, NonlinearConjugateGradient.p);
                        break;
                    }
                }
                InPlaceOperator.affine(NonlinearConjugateGradient.p, beta, NonlinearConjugateGradient.p, '-', NonlinearConjugateGradient.G);
                ++NonlinearConjugateGradient.k;
                NonlinearConjugateGradient.state = 1;
            }
            NonlinearConjugateGradient.converge = false;
            NonlinearConjugateGradient.gradientRequired = false;
            return new boolean[] { NonlinearConjugateGradient.converge, NonlinearConjugateGradient.gradientRequired };
        }
    }
}
