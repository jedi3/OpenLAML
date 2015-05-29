package ml.regression;

import ml.options.*;

import java.io.*;
import java.util.*;

import la.matrix.*;
import ml.utils.*;
import la.vector.*;
import la.vector.Vector;

public class LASSO extends Regression
{
    private double lambda;
    private boolean calc_OV;
    private boolean verbose;
    
    public static void main(final String[] args) {
        final double[][] data = { { 1.0, 2.0, 3.0, 2.0 }, { 4.0, 2.0, 3.0, 6.0 }, { 5.0, 1.0, 4.0, 1.0 } };
        final double[][] depVars = { { 3.0, 2.0 }, { 2.0, 3.0 }, { 1.0, 4.0 } };
        final Options options = new Options();
        options.maxIter = 600;
        options.lambda = 0.1;
        options.verbose = false;
        options.calc_OV = false;
        options.epsilon = 1.0E-5;
        final Regression LASSO = new LASSO(options);
        LASSO.feedData(data);
        LASSO.feedDependentVariables(depVars);
        Time.tic();
        LASSO.train();
        Printer.fprintf("Elapsed time: %.3f seconds\n\n", Time.toc());
        Printer.fprintf("Projection matrix:\n", new Object[0]);
        Printer.display(LASSO.W);
        final Matrix Yt = LASSO.predict(data);
        Printer.fprintf("Predicted dependent variables:\n", new Object[0]);
        Printer.display(Yt);
    }
    
    public LASSO() {
        this.lambda = 1.0;
        this.calc_OV = false;
        this.verbose = false;
    }
    
    public LASSO(final double epsilon) {
        super(epsilon);
        this.lambda = 1.0;
        this.calc_OV = false;
        this.verbose = false;
    }
    
    public LASSO(final int maxIter, final double epsilon) {
        super(maxIter, epsilon);
        this.lambda = 1.0;
        this.calc_OV = false;
        this.verbose = false;
    }
    
    public LASSO(final double lambda, final int maxIter, final double epsilon) {
        super(maxIter, epsilon);
        this.lambda = lambda;
        this.calc_OV = false;
        this.verbose = false;
    }
    
    public LASSO(final Options options) {
        super(options);
        this.lambda = options.lambda;
        this.calc_OV = options.calc_OV;
        this.verbose = options.verbose;
    }
    
    @Override
    public void train() {
        this.W = this.train(this.X, this.Y);
    }
    
    @Override
    public void loadModel(final String filePath) {
        try {
            final ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath));
            this.W = (Matrix)ois.readObject();
            ois.close();
            System.out.println("Model loaded.");
        }
        catch (FileNotFoundException e) {
            e.printStackTrace();
            System.exit(1);
        }
        catch (IOException e2) {
            e2.printStackTrace();
        }
        catch (ClassNotFoundException e3) {
            e3.printStackTrace();
        }
    }
    
    @Override
    public void saveModel(final String filePath) {
        final File parentFile = new File(filePath).getParentFile();
        if (parentFile != null && !parentFile.exists()) {
            parentFile.mkdirs();
        }
        try {
            final ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath));
            oos.writeObject(this.W);
            oos.close();
            System.out.println("Model saved.");
        }
        catch (FileNotFoundException e) {
            e.printStackTrace();
            System.exit(1);
        }
        catch (IOException e2) {
            e2.printStackTrace();
        }
    }
    
    public static Matrix train(final Matrix X, final Matrix Y, final Options options) {
        final int p = Matlab.size(X, 2);
        final int ny = Matlab.size(Y, 2);
        final double epsilon = options.epsilon;
        final int maxIter = options.maxIter;
        final double lambda = options.lambda;
        final boolean calc_OV = options.calc_OV;
        final boolean verbose = options.verbose;
        final Matrix XNX = Matlab.horzcat(X, Matlab.uminus(X));
        final Matrix H_G = XNX.transpose().mtimes(XNX);
        final double[] Q = new double[Matlab.size(H_G, 1)];
        for (int i = 0; i < Q.length; ++i) {
            Q[i] = H_G.getEntry(i, i);
        }
        final Matrix XNXTY = XNX.transpose().mtimes(Y);
        Matrix A = Matlab.mldivide(Matlab.plus(X.transpose().mtimes(X), Matlab.times(lambda, Matlab.eye(p))), X.transpose().mtimes(Y));
        final Matrix AA = Matlab.vertcat(Matlab.subplus(A), Matlab.subplus(Matlab.uminus(A)));
        final Matrix C = Matlab.plus(Matlab.uminus(XNXTY), lambda);
        final Matrix Grad = Matlab.plus(C, Matlab.mtimes(H_G, AA));
        final double tol = epsilon * Matlab.norm(Grad);
        final Matrix PGrad = Matlab.zeros(Matlab.size(Grad));
        final ArrayList<Double> J = new ArrayList<Double>();
        double fval = 0.0;
        if (calc_OV) {
            fval = Matlab.sum(Matlab.sum(Matlab.pow(Matlab.minus(Y, Matlab.mtimes(X, A)), 2.0))) / 2.0 + lambda * Matlab.sum(Matlab.sum(Matlab.abs(A)));
            J.add(fval);
        }
        final Matrix I_k = Grad.copy();
        Matrix I_k_com = null;
        double d = 0.0;
        int k = 0;
        Vector SFPlusCi = null;
        final Matrix S = H_G;
        Vector[] SRows = null;
        if (H_G instanceof DenseMatrix) {
            SRows = Matlab.denseMatrix2DenseRowVectors(S);
        }
        else {
            SRows = Matlab.sparseMatrix2SparseRowVectors(S);
        }
        Vector[] CRows = null;
        if (C instanceof DenseMatrix) {
            CRows = Matlab.denseMatrix2DenseRowVectors(C);
        }
        else {
            CRows = Matlab.sparseMatrix2SparseRowVectors(C);
        }
        final double[][] FData = ((DenseMatrix)AA).getData();
        double[] FRow = null;
        double[] pr = null;
        final int K = 2 * p;
        while (true) {
            InPlaceOperator.or(I_k, Matlab.lt(Grad, 0.0), Matlab.gt(AA, 0.0));
            I_k_com = Matlab.not(I_k);
            InPlaceOperator.assign(PGrad, Grad);
            Matlab.logicalIndexingAssignment(PGrad, I_k_com, 0.0);
            d = Matlab.norm(PGrad, Matlab.inf);
            if (d < tol) {
                if (verbose) {
                    System.out.println("Converge successfully!");
                    break;
                }
                break;
            }
            else {
                for (int j = 0; j < K; ++j) {
                    SFPlusCi = SRows[j].operate(AA);
                    InPlaceOperator.plusAssign(SFPlusCi, CRows[j]);
                    InPlaceOperator.timesAssign(SFPlusCi, 1.0 / Q[j]);
                    pr = ((DenseVector)SFPlusCi).getPr();
                    FRow = FData[j];
                    for (int l = 0; l < AA.getColumnDimension(); ++l) {
                        FRow[l] = Math.max(FRow[l] - pr[l], 0.0);
                    }
                }
                InPlaceOperator.plus(Grad, C, Matlab.mtimes(H_G, AA));
                if (++k > maxIter) {
                    if (verbose) {
                        System.out.println("Maximal iterations");
                        break;
                    }
                    break;
                }
                else {
                    if (calc_OV) {
                        fval = Matlab.sum(Matlab.sum(Matlab.pow(Matlab.minus(Y, Matlab.mtimes(XNX, AA)), 2.0))) / 2.0 + lambda * Matlab.sum(Matlab.sum(Matlab.abs(AA)));
                        J.add(fval);
                    }
                    if (k % 10 != 0 || !verbose) {
                        continue;
                    }
                    if (calc_OV) {
                        System.out.format("Iter %d - ||PGrad||: %f, ofv: %f\n", k, d, J.get(J.size() - 1));
                    }
                    else {
                        System.out.format("Iter %d - ||PGrad||: %f\n", k, d);
                    }
                }
            }
        }
        A = Matlab.minus(AA.getSubMatrix(0, p - 1, 0, ny - 1), AA.getSubMatrix(p, 2 * p - 1, 0, ny - 1));
        return A;
    }
    
    @Override
    public Matrix train(final Matrix X, final Matrix Y) {
        final int p = Matlab.size(X, 2);
        final int ny = Matlab.size(Y, 2);
        final Matrix XNX = Matlab.horzcat(X, Matlab.uminus(X));
        final Matrix H_G = XNX.transpose().mtimes(XNX);
        final double[] Q = new double[Matlab.size(H_G, 1)];
        for (int i = 0; i < Q.length; ++i) {
            Q[i] = H_G.getEntry(i, i);
        }
        final Matrix XNXTY = XNX.transpose().mtimes(Y);
        Matrix A = Matlab.mldivide(Matlab.plus(X.transpose().mtimes(X), Matlab.times(this.lambda, Matlab.eye(p))), X.transpose().mtimes(Y));
        final Matrix AA = Matlab.vertcat(Matlab.subplus(A), Matlab.subplus(Matlab.uminus(A)));
        final Matrix C = Matlab.plus(Matlab.uminus(XNXTY), this.lambda);
        final Matrix Grad = Matlab.plus(C, Matlab.mtimes(H_G, AA));
        final double tol = this.epsilon * Matlab.norm(Grad);
        final Matrix PGrad = Matlab.zeros(Matlab.size(Grad));
        final ArrayList<Double> J = new ArrayList<Double>();
        double fval = 0.0;
        if (this.calc_OV) {
            fval = Matlab.sumAll(Matlab.pow(Matlab.minus(Y, Matlab.mtimes(X, A)), 2.0)) / 2.0 + this.lambda * Matlab.sum(Matlab.sum(Matlab.abs(A)));
            J.add(fval);
        }
        final Matrix I_k = Grad.copy();
        Matrix I_k_com = null;
        double d = 0.0;
        int k = 0;
        Vector SFPlusCi = null;
        final Matrix S = H_G;
        Vector[] SRows = null;
        if (H_G instanceof DenseMatrix) {
            SRows = Matlab.denseMatrix2DenseRowVectors(S);
        }
        else {
            SRows = Matlab.sparseMatrix2SparseRowVectors(S);
        }
        Vector[] CRows = null;
        if (C instanceof DenseMatrix) {
            CRows = Matlab.denseMatrix2DenseRowVectors(C);
        }
        else {
            CRows = Matlab.sparseMatrix2SparseRowVectors(C);
        }
        final double[][] FData = ((DenseMatrix)AA).getData();
        double[] FRow = null;
        double[] pr = null;
        final int K = 2 * p;
        while (true) {
            InPlaceOperator.or(I_k, Matlab.lt(Grad, 0.0), Matlab.gt(AA, 0.0));
            I_k_com = Matlab.not(I_k);
            InPlaceOperator.assign(PGrad, Grad);
            Matlab.logicalIndexingAssignment(PGrad, I_k_com, 0.0);
            d = Matlab.norm(PGrad, Matlab.inf);
            if (d < tol) {
                if (this.verbose) {
                    System.out.println("Converge successfully!");
                    break;
                }
                break;
            }
            else {
                for (int j = 0; j < K; ++j) {
                    SFPlusCi = SRows[j].operate(AA);
                    InPlaceOperator.plusAssign(SFPlusCi, CRows[j]);
                    InPlaceOperator.timesAssign(SFPlusCi, 1.0 / Q[j]);
                    pr = ((DenseVector)SFPlusCi).getPr();
                    FRow = FData[j];
                    for (int l = 0; l < AA.getColumnDimension(); ++l) {
                        FRow[l] = Math.max(FRow[l] - pr[l], 0.0);
                    }
                }
                InPlaceOperator.plus(Grad, C, Matlab.mtimes(H_G, AA));
                if (++k > this.maxIter) {
                    if (this.verbose) {
                        System.out.println("Maximal iterations");
                        break;
                    }
                    break;
                }
                else {
                    if (this.calc_OV) {
                        fval = Matlab.sum(Matlab.sum(Matlab.pow(Matlab.minus(Y, Matlab.mtimes(XNX, AA)), 2.0))) / 2.0 + this.lambda * Matlab.sum(Matlab.sum(Matlab.abs(AA)));
                        J.add(fval);
                    }
                    if (k % 10 != 0 || !this.verbose) {
                        continue;
                    }
                    if (this.calc_OV) {
                        System.out.format("Iter %d - ||PGrad||: %f, ofv: %f\n", k, d, J.get(J.size() - 1));
                    }
                    else {
                        System.out.format("Iter %d - ||PGrad||: %f\n", k, d);
                    }
                }
            }
        }
        A = Matlab.minus(AA.getSubMatrix(0, p - 1, 0, ny - 1), AA.getSubMatrix(p, 2 * p - 1, 0, ny - 1));
        return A;
    }
    
    @Override
    public Matrix train(final Matrix X, final Matrix Y, final Matrix W0) {
        this.p = W0.getRowDimension();
        this.ny = W0.getColumnDimension();
        final Matrix XNX = Matlab.horzcat(X, Matlab.uminus(X));
        final Matrix H_G = XNX.transpose().mtimes(XNX);
        final double[] Q = new double[Matlab.size(H_G, 1)];
        for (int i = 0; i < Q.length; ++i) {
            Q[i] = H_G.getEntry(i, i);
        }
        final Matrix XNXTY = XNX.transpose().mtimes(Y);
        Matrix A = W0.copy();
        final Matrix AA = Matlab.vertcat(Matlab.subplus(A), Matlab.subplus(Matlab.uminus(A)));
        final Matrix C = Matlab.plus(Matlab.uminus(XNXTY), this.lambda);
        final Matrix Grad = Matlab.plus(C, Matlab.mtimes(H_G, AA));
        final double tol = this.epsilon * Matlab.norm(Grad);
        final Matrix PGrad = Matlab.zeros(Matlab.size(Grad));
        final ArrayList<Double> J = new ArrayList<Double>();
        double fval = 0.0;
        if (this.calc_OV) {
            fval = Matlab.sumAll(Matlab.pow(Matlab.minus(Y, Matlab.mtimes(X, A)), 2.0)) / 2.0 + this.lambda * Matlab.sum(Matlab.sum(Matlab.abs(A)));
            J.add(fval);
        }
        final Matrix I_k = Grad.copy();
        Matrix I_k_com = null;
        double d = 0.0;
        int k = 0;
        Vector SFPlusCi = null;
        final Matrix S = H_G;
        Vector[] SRows = null;
        if (H_G instanceof DenseMatrix) {
            SRows = Matlab.denseMatrix2DenseRowVectors(S);
        }
        else {
            SRows = Matlab.sparseMatrix2SparseRowVectors(S);
        }
        Vector[] CRows = null;
        if (C instanceof DenseMatrix) {
            CRows = Matlab.denseMatrix2DenseRowVectors(C);
        }
        else {
            CRows = Matlab.sparseMatrix2SparseRowVectors(C);
        }
        final double[][] FData = ((DenseMatrix)AA).getData();
        double[] FRow = null;
        double[] pr = null;
        final int K = 2 * this.p;
        while (true) {
            InPlaceOperator.or(I_k, Matlab.lt(Grad, 0.0), Matlab.gt(AA, 0.0));
            I_k_com = Matlab.not(I_k);
            InPlaceOperator.assign(PGrad, Grad);
            Matlab.logicalIndexingAssignment(PGrad, I_k_com, 0.0);
            d = Matlab.norm(PGrad, Matlab.inf);
            if (d < tol) {
                if (this.verbose) {
                    System.out.println("LASSO converges successfully!");
                    break;
                }
                break;
            }
            else {
                for (int j = 0; j < K; ++j) {
                    SFPlusCi = SRows[j].operate(AA);
                    InPlaceOperator.plusAssign(SFPlusCi, CRows[j]);
                    InPlaceOperator.timesAssign(SFPlusCi, 1.0 / Q[j]);
                    pr = ((DenseVector)SFPlusCi).getPr();
                    FRow = FData[j];
                    for (int l = 0; l < AA.getColumnDimension(); ++l) {
                        FRow[l] = Math.max(FRow[l] - pr[l], 0.0);
                    }
                }
                InPlaceOperator.plus(Grad, C, Matlab.mtimes(H_G, AA));
                if (++k > this.maxIter) {
                    if (this.verbose) {
                        System.out.println("Maximal iterations");
                        break;
                    }
                    break;
                }
                else {
                    if (this.calc_OV) {
                        fval = Matlab.sum(Matlab.sum(Matlab.pow(Matlab.minus(Y, Matlab.mtimes(XNX, AA)), 2.0))) / 2.0 + this.lambda * Matlab.sum(Matlab.sum(Matlab.abs(AA)));
                        J.add(fval);
                    }
                    if (k % 10 != 0 || !this.verbose) {
                        continue;
                    }
                    if (this.calc_OV) {
                        System.out.format("Iter %d - ||PGrad||: %f, ofv: %f\n", k, d, J.get(J.size() - 1));
                    }
                    else {
                        System.out.format("Iter %d - ||PGrad||: %f\n", k, d);
                    }
                }
            }
        }
        A = Matlab.minus(AA.getSubMatrix(0, this.p - 1, 0, this.ny - 1), AA.getSubMatrix(this.p, 2 * this.p - 1, 0, this.ny - 1));
        return A;
    }
    
    @Override
    public void train(final Matrix W0) {
        this.W = this.train(this.X, this.Y, W0);
    }
}
