package ml.recovery;

import la.matrix.*;
import ml.optimization.*;
import ml.utils.*;

public class RobustPCA
{
    double lambda;
    Matrix D;
    Matrix A;
    Matrix E;
    
    public static void main(final String[] args) {
        final int m = 8;
        final int r = m / 4;
        final Matrix L = Matlab.randn(m, r);
        final Matrix R = Matlab.randn(m, r);
        final Matrix A_star = Matlab.mtimes(L, R.transpose());
        Matrix E_star = Matlab.zeros(Matlab.size(A_star));
        final int[] indices = Matlab.randperm(m * m);
        final int nz = m * m / 20;
        final int[] nz_indices = new int[nz];
        for (int i = 0; i < nz; ++i) {
            nz_indices[i] = indices[i] - 1;
        }
        final Matrix E_vec = Matlab.vec(E_star);
        Matlab.setSubMatrix(E_vec, nz_indices, new int[1], Matlab.minus(Matlab.rand(nz, 1), 0.5).times(100.0));
        E_star = Matlab.reshape(E_vec, Matlab.size(E_star));
        final Matrix D = A_star.plus(E_star);
        final double lambda = 1.0 * Math.pow(m, -0.5);
        final RobustPCA robustPCA = new RobustPCA(lambda);
        robustPCA.feedData(D);
        Time.tic();
        robustPCA.run();
        Printer.fprintf("Elapsed time: %.2f seconds.%n", Time.toc());
        final Matrix A_hat = robustPCA.GetLowRankEstimation();
        final Matrix E_hat = robustPCA.GetErrorMatrix();
        Printer.fprintf("A*:\n", new Object[0]);
        Printer.disp(A_star, 4);
        Printer.fprintf("A^:\n", new Object[0]);
        Printer.disp(A_hat, 4);
        Printer.fprintf("E*:\n", new Object[0]);
        Printer.disp(E_star, 4);
        Printer.fprintf("E^:\n", new Object[0]);
        Printer.disp(E_hat, 4);
        Printer.fprintf("rank(A*): %d\n", Matlab.rank(A_star));
        Printer.fprintf("rank(A^): %d\n", Matlab.rank(A_hat));
        Printer.fprintf("||A* - A^||_F: %.4f\n", Matlab.norm(A_star.minus(A_hat), "fro"));
        Printer.fprintf("||E* - E^||_F: %.4f\n", Matlab.norm(E_star.minus(E_hat), "fro"));
    }
    
    public RobustPCA(final double lambda) {
        this.lambda = lambda;
    }
    
    public void feedData(final Matrix D) {
        this.D = D;
    }
    
    public void run() {
        final Matrix[] res = run(this.D, this.lambda);
        this.A = res[0];
        this.E = res[1];
    }
    
    public Matrix GetLowRankEstimation() {
        return this.A;
    }
    
    public Matrix GetErrorMatrix() {
        return this.E;
    }
    
    public static Matrix[] run(final Matrix D, final double lambda) {
        final Matrix Y = Matlab.rdivide(D, J(D, lambda));
        final Matrix E = Matlab.zeros(Matlab.size(D));
        Matrix A = Matlab.minus(D, E);
        double mu = 1.25 / Matlab.norm(D, 2);
        final double rou = 1.6;
        int k = 0;
        final double norm_D = Matlab.norm(D, "fro");
        final double e1 = 1.0E-7;
        final double e2 = 1.0E-6;
        double c1 = 0.0;
        double c2 = 0.0;
        double mu_old = 0.0;
        Matrix E_old = null;
        Matrix[] SVD = null;
        while (true) {
            if (k > 0) {
                c1 = Matlab.norm(D.minus(A).minus(E), "fro") / norm_D;
                c2 = mu_old * Matlab.norm(E.minus(E_old), "fro") / norm_D;
                if (c1 <= e1 && c2 <= e2) {
                    break;
                }
            }
            E_old = E;
            mu_old = mu;
            ShrinkageOperator.shrinkage(E, Matlab.plus(Matlab.minus(D, A), Matlab.rdivide(Y, mu)), lambda / mu);
            SVD = Matlab.svd(Matlab.plus(Matlab.minus(D, E), Matlab.rdivide(Y, mu)));
            A = SVD[0].mtimes(ShrinkageOperator.shrinkage(SVD[1], 1.0 / mu)).mtimes(SVD[2].transpose());
            InPlaceOperator.plusAssign(Y, mu, D.minus(A).minus(E));
            if (Matlab.norm(E.minus(E_old), "fro") * mu / norm_D < e2) {
                mu *= rou;
            }
            ++k;
        }
        final Matrix[] res = { A, E };
        return res;
    }
    
    private static double J(final Matrix Y, final double lambda) {
        return Math.max(Matlab.norm(Y, 2), Matlab.max(Matlab.max(Matlab.abs(Y))[0])[0] / lambda);
    }
}
