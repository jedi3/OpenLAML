package ml.recovery;

import la.io.*;
import la.io.IO;
import la.matrix.*;
import ml.utils.*;
import ml.optimization.*;

public class MatrixCompletion
{
    Matrix D;
    Matrix Omega;
    Matrix A;
    Matrix E;
    
    public static void main(final String[] args) {
        final int m = 6;
        final int r = 1;
        final int p = (int)Math.round(m * m * 0.3);
        final Matrix L = Matlab.randn(m, r);
        final Matrix R = Matlab.randn(m, r);
        final Matrix A_star = Matlab.mtimes(L, R.transpose());
        int[] indices = Matlab.randperm(m * m);
        ArrayOperator.minusAssign(indices, 1);
        indices = Matlab.linearIndexing(indices, ArrayOperator.colon(0, p - 1));
        Matrix Omega = Matlab.zeros(Matlab.size(A_star));
        Matlab.linearIndexingAssignment(Omega, indices, 1.0);
        Matrix D = Matlab.zeros(Matlab.size(A_star));
        Matlab.linearIndexingAssignment(D, indices, Matlab.linearIndexing(A_star, indices));
        final Matrix E_star = D.minus(A_star);
        Matlab.logicalIndexingAssignment(E_star, Omega, 0.0);
        D = IO.loadMatrix("D.txt");
        Omega = IO.loadMatrix("Omega.txt");
        final MatrixCompletion matrixCompletion = new MatrixCompletion();
        matrixCompletion.feedData(D);
        matrixCompletion.feedIndices(Omega);
        Time.tic();
        matrixCompletion.run();
        Printer.fprintf("Elapsed time: %.2f seconds.%n", Time.toc());
        final Matrix A_hat = matrixCompletion.GetLowRankEstimation();
        Printer.fprintf("A*:\n", new Object[0]);
        Printer.disp(A_star, 4);
        Printer.fprintf("A^:\n", new Object[0]);
        Printer.disp(A_hat, 4);
        Printer.fprintf("D:\n", new Object[0]);
        Printer.disp(D, 4);
        Printer.fprintf("rank(A*): %d\n", Matlab.rank(A_star));
        Printer.fprintf("rank(A^): %d\n", Matlab.rank(A_hat));
        Printer.fprintf("||A* - A^||_F: %.4f\n", Matlab.norm(A_star.minus(A_hat), "fro"));
    }
    
    public void feedData(final Matrix D) {
        this.D = D;
    }
    
    public void feedIndices(final Matrix Omega) {
        this.Omega = Omega;
    }
    
    public void feedIndices(final int[] indices) {
        Matlab.linearIndexingAssignment(this.Omega = new SparseMatrix(Matlab.size(this.D, 1), Matlab.size(this.D, 2)), indices, 1.0);
    }
    
    public void run() {
        final Matrix[] res = matrixCompletion(this.D, this.Omega);
        this.A = res[0];
        this.E = res[1];
    }
    
    public Matrix GetLowRankEstimation() {
        return this.A;
    }
    
    public Matrix GetErrorMatrix() {
        return this.E;
    }
    
    public static Matrix[] matrixCompletion(final Matrix D, final Matrix Omega) {
        final Matrix Y = Matlab.zeros(Matlab.size(D));
        final Matrix E = Matlab.zeros(Matlab.size(D));
        Matrix A = Matlab.minus(D, E);
        final int m = Matlab.size(D, 1);
        final int n = Matlab.size(D, 2);
        double mu = 1.0 / Matlab.norm(D, 2);
        final double rou_s = Matlab.sumAll(Matlab.gt(Omega, 0.0)) / (m * n);
        final double rou = 1.2172 + 1.8588 * rou_s;
        int k = 0;
        final double norm_D = Matlab.norm(D, "fro");
        final double e1 = 1.0E-7;
        final double e2 = 1.0E-6;
        double c1 = 0.0;
        double c2 = 0.0;
        double mu_old = 0.0;
        final Matrix E_old = E.copy();
        Matrix[] SVD = null;
        while (true) {
            if (k > 1) {
                c1 = Matlab.norm(D.minus(A).minus(E), "fro") / norm_D;
                c2 = mu_old * Matlab.norm(E.minus(E_old), "fro") / norm_D;
                if (c1 <= e1 && c2 <= e2) {
                    break;
                }
            }
            InPlaceOperator.assign(E_old, E);
            mu_old = mu;
            SVD = Matlab.svd(Matlab.plus(Matlab.minus(D, E), Matlab.rdivide(Y, mu)));
            A = SVD[0].mtimes(ShrinkageOperator.shrinkage(SVD[1], 1.0 / mu)).mtimes(SVD[2].transpose());
            InPlaceOperator.minus(E, D, A);
            Matlab.logicalIndexingAssignment(E, Omega, 0.0);
            InPlaceOperator.plusAssign(Y, mu, D.minus(A).minus(E));
            if (Matlab.norm(E.minus(E_old), "fro") * mu / norm_D < e2) {
                mu *= rou;
            }
            ++k;
        }
        final Matrix[] res = { A, E };
        return res;
    }
    
    public static Matrix[] matrixCompletion(final Matrix D, final int[] indices) {
        final Matrix Omega = new SparseMatrix(Matlab.size(D, 1), Matlab.size(D, 2));
        Matlab.linearIndexingAssignment(Omega, indices, 1.0);
        return matrixCompletion(D, Omega);
    }
}
