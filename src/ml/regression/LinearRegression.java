package ml.regression;

import ml.options.*;
import la.matrix.*;
import ml.utils.*;
import java.io.*;

public class LinearRegression extends Regression
{
    private double lambda;
    private boolean calc_OV;
    private boolean verbose;
    public double[] B;
    
    public static void main(final String[] args) {
        final double[][] data = { { 1.0, 2.0, 3.0, 2.0 }, { 4.0, 2.0, 3.0, 6.0 }, { 5.0, 1.0, 4.0, 1.0 } };
        final double[][] depVars = { { 3.0, 2.0 }, { 2.0, 3.0 }, { 1.0, 4.0 } };
        final Options options = new Options();
        options.maxIter = 600;
        options.lambda = 0.1;
        options.verbose = false;
        options.calc_OV = false;
        options.epsilon = 1.0E-5;
        final Regression LR = new LinearRegression(options);
        LR.feedData(data);
        LR.feedDependentVariables(depVars);
        Time.tic();
        LR.train();
        Printer.fprintf("Elapsed time: %.3f seconds\n\n", Time.toc());
        Printer.fprintf("Projection matrix:\n", new Object[0]);
        Printer.display(LR.W);
        Printer.fprintf("Bias vector:\n", new Object[0]);
        Printer.display(((LinearRegression)LR).B);
        final Matrix Yt = LR.predict(data);
        Printer.fprintf("Predicted dependent variables:\n", new Object[0]);
        Printer.display(Yt);
    }
    
    public LinearRegression() {
    }
    
    public LinearRegression(final double epsilon) {
        super(epsilon);
    }
    
    public LinearRegression(final int maxIter, final double epsilon) {
        super(maxIter, epsilon);
    }
    
    public LinearRegression(final Options options) {
        super(options);
        this.lambda = options.lambda;
        this.calc_OV = options.calc_OV;
        this.verbose = options.verbose;
    }
    
    private double train(final Matrix X, final double[] y, final double[] w0, final double b0) {
        double b = b0;
        final double[] y_hat = new double[this.n];
        final double[] e = new double[this.n];
        double[] OFVs = null;
        final boolean debug = false;
        final int blockSize = 10;
        if (this.calc_OV && this.verbose) {
            OFVs = ArrayOperator.allocate1DArray(this.maxIter + 1, 0.0);
            double ofv = 0.0;
            ofv = this.computeOFV(y, w0, b);
            OFVs[0] = ofv;
            Printer.fprintf("Iter %d: %.10g\n", 0, ofv);
        }
        int cnt = 0;
        double ofv_old = 0.0;
        double ofv_new = 0.0;
        if (X instanceof SparseMatrix) {
            final int[] ic = ((SparseMatrix)X).getIc();
            final int[] ir = ((SparseMatrix)X).getIr();
            final int[] jc = ((SparseMatrix)X).getJc();
            final int[] jr = ((SparseMatrix)X).getJr();
            final double[] pr = ((SparseMatrix)X).getPr();
            final int[] valCSRIndices = ((SparseMatrix)X).getValCSRIndices();
            for (int r = 0; r < this.n; ++r) {
                double s = b;
                for (int k = jr[r]; k < jr[r + 1]; ++k) {
                    final int j = ic[k];
                    s += w0[j] * pr[valCSRIndices[k]];
                }
                y_hat[r] = s;
                e[r] = y[r] - s;
            }
            do {
                ofv_old = 0.0;
                if (debug) {
                    ofv_old = this.computeOFV(y, w0, b);
                    Printer.printf("f(b): %f\n", ofv_old);
                }
                final double b_new = (b * this.n + ArrayOperator.sum(e)) / (this.n + this.lambda);
                for (int i = 0; i < this.n; ++i) {
                    final double[] array = e;
                    final int n = i;
                    array[n] -= b_new - b;
                }
                b = b_new;
                if (debug) {
                    ofv_new = this.computeOFV(y, w0, b);
                    Printer.printf("b updated: %f\n", ofv_new);
                    if (ofv_old < ofv_new) {
                        Printer.errf("Error when updating b\n", new Object[0]);
                    }
                }
                for (int l = 0; l < this.p; ++l) {
                    ofv_old = 0.0;
                    double v1 = 0.0;
                    double v2 = 0.0;
                    for (int m = jc[l]; m < jc[l + 1]; ++m) {
                        final int i2 = ir[m];
                        final double hj;
                        final double xj = hj = pr[m];
                        v1 += hj * hj;
                        v2 += hj * e[i2];
                    }
                    final double wj_new = (w0[l] * v1 + v2) / (v1 + this.lambda);
                    if (Double.isInfinite(wj_new)) {
                        int a = 1;
                        ++a;
                    }
                    for (int k2 = jc[l]; k2 < jc[l + 1]; ++k2) {
                        final int i3 = ir[k2];
                        final double xj2 = pr[k2];
                        final double[] array2 = e;
                        final int n2 = i3;
                        array2[n2] -= (wj_new - w0[l]) * xj2;
                    }
                    w0[l] = wj_new;
                    if (debug) {
                        ofv_new = this.computeOFV(y, w0, b);
                        Printer.printf("w[%d] updated: %f\n", l, ofv_new);
                        if (ofv_old < ofv_new) {
                            Printer.errf("Error when updating w[%d]\n", l);
                        }
                    }
                }
                ++cnt;
                if (this.verbose) {
                    if (this.calc_OV) {
                        final double ofv2 = this.computeOFV(y, w0, b);
                        OFVs[cnt] = ofv2;
                        if (cnt % blockSize == 0) {
                            Printer.fprintf(".Iter %d: %.8g\n", cnt, ofv2);
                        }
                        else {
                            Printer.fprintf(".", new Object[0]);
                        }
                    }
                    else if (cnt % blockSize == 0) {
                        Printer.fprintf(".Iter %d\n", cnt);
                    }
                    else {
                        Printer.fprintf(".", new Object[0]);
                    }
                }
            } while (cnt < this.maxIter);
        }
        else if (X instanceof DenseMatrix) {
            final double[][] data = X.getData();
            for (int r2 = 0; r2 < this.n; ++r2) {
                double s2 = b;
                s2 += ArrayOperator.innerProduct(w0, data[r2]);
                y_hat[r2] = s2;
                e[r2] = y[r2] - s2;
            }
            do {
                ofv_old = 0.0;
                if (debug) {
                    ofv_old = this.computeOFV(y, w0, b);
                    Printer.printf("f(b): %f\n", ofv_old);
                }
                final double b_new2 = (b * this.n + ArrayOperator.sum(e)) / (this.n + this.lambda);
                for (int i4 = 0; i4 < this.n; ++i4) {
                    final double[] array3 = e;
                    final int n3 = i4;
                    array3[n3] -= b_new2 - b;
                }
                b = b_new2;
                if (debug) {
                    ofv_new = this.computeOFV(y, w0, b);
                    Printer.printf("b updated: %f\n", ofv_new);
                    if (ofv_old < ofv_new) {
                        Printer.errf("Error when updating b\n", new Object[0]);
                    }
                }
                for (int j2 = 0; j2 < this.p; ++j2) {
                    ofv_old = 0.0;
                    double v3 = 0.0;
                    double v4 = 0.0;
                    for (int i = 0; i < this.n; ++i) {
                        final double hj2;
                        final double xj3 = hj2 = data[i][j2];
                        v3 += hj2 * hj2;
                        v4 += hj2 * e[i];
                    }
                    final double wj_new2 = (w0[j2] * v3 + v4) / (v3 + this.lambda);
                    if (Double.isInfinite(wj_new2)) {
                        int a2 = 1;
                        ++a2;
                    }
                    for (int i5 = 0; i5 < this.n; ++i5) {
                        final double xj4 = data[i5][j2];
                        final double[] array4 = e;
                        final int n4 = i5;
                        array4[n4] -= (wj_new2 - w0[j2]) * xj4;
                    }
                    w0[j2] = wj_new2;
                    if (debug) {
                        ofv_new = this.computeOFV(y, w0, b);
                        Printer.printf("w[%d] updated: %f\n", j2, ofv_new);
                        if (ofv_old < ofv_new) {
                            Printer.errf("Error when updating w[%d]\n", j2);
                        }
                    }
                }
                ++cnt;
                if (this.verbose) {
                    if (this.calc_OV) {
                        final double ofv3 = this.computeOFV(y, w0, b);
                        OFVs[cnt] = ofv3;
                        if (cnt % blockSize == 0) {
                            Printer.fprintf(".Iter %d: %.8g\n", cnt, ofv3);
                        }
                        else {
                            Printer.fprintf(".", new Object[0]);
                        }
                    }
                    else if (cnt % blockSize == 0) {
                        Printer.fprintf(".Iter %d\n", cnt);
                    }
                    else {
                        Printer.fprintf(".", new Object[0]);
                    }
                }
            } while (cnt < this.maxIter);
        }
        return b;
    }
    
    @Override
    public void train() {
        final double[][] ws = ArrayOperator.allocate2DArray(this.ny, this.p, 0.0);
        this.B = ArrayOperator.allocate1DArray(this.ny, 0.0);
        for (int k = 0; k < this.ny; ++k) {
            this.B[k] = this.train(this.X, Matlab.full(this.Y.getColumnVector(k)).getPr(), ws[k], this.B[k]);
        }
        this.W = new DenseMatrix(ws).transpose();
    }
    
    @Override
    public void train(final Matrix W0) {
        final double[][] ws = W0.transpose().getData();
        this.B = ArrayOperator.allocate1DArray(this.ny, 0.0);
        for (int k = 0; k < this.ny; ++k) {
            this.B[k] = this.train(this.X, Matlab.full(this.Y.getColumnVector(k)).getPr(), ws[k], this.B[k]);
        }
        this.W = new DenseMatrix(ws).transpose();
    }
    
    @Override
    public Matrix train(final Matrix X, final Matrix Y) {
        final String Method = "Linear Regression";
        System.out.printf("Training %s...\n", Method);
        final double[][] ws = ArrayOperator.allocate2DArray(this.ny, this.p, 0.0);
        this.B = ArrayOperator.allocate1DArray(this.ny, 0.0);
        for (int k = 0; k < this.ny; ++k) {
            this.B[k] = this.train(X, Matlab.full(Y.getColumnVector(k)).getPr(), ws[k], this.B[k]);
        }
        return this.W = new DenseMatrix(ws).transpose();
    }
    
    private double computeOFV(final double[] y, final double[] w, final double b) {
        double ofv = 0.0;
        ofv += this.lambda * b * b;
        ofv += this.lambda * ArrayOperator.innerProduct(w, w);
        final int[] ic = ((SparseMatrix)this.X).getIc();
        final int[] jr = ((SparseMatrix)this.X).getJr();
        final double[] pr = ((SparseMatrix)this.X).getPr();
        final int[] valCSRIndices = ((SparseMatrix)this.X).getValCSRIndices();
        for (int r = 0; r < this.n; ++r) {
            double s = b;
            for (int k = jr[r]; k < jr[r + 1]; ++k) {
                final int j = ic[k];
                s += w[j] * pr[valCSRIndices[k]];
                if (Double.isNaN(s)) {
                    int a = 1;
                    ++a;
                }
            }
            final double e = y[r] - s;
            ofv += e * e;
        }
        return ofv;
    }
    
    @Override
    public Matrix train(final Matrix X, final Matrix Y, final Matrix W0) {
        final double[][] ws = W0.transpose().getData();
        this.B = ArrayOperator.allocate1DArray(this.ny, 0.0);
        for (int k = 0; k < this.ny; ++k) {
            this.B[k] = this.train(X, Matlab.full(Y.getColumnVector(k)).getPr(), ws[k], this.B[k]);
        }
        return this.W = new DenseMatrix(ws).transpose();
    }
    
    @Override
    public void loadModel(final String filePath) {
        try {
            final ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath));
            this.W = (Matrix)ois.readObject();
            this.B = (double[])ois.readObject();
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
            oos.writeObject(this.B);
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
}
