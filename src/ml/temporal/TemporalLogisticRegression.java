package ml.temporal;

import ml.options.*;
import la.vector.*;
import ml.optimization.*;
import ml.utils.*;
import la.matrix.*;
import java.io.*;

public class TemporalLogisticRegression
{
    public int regularizationType;
    protected Matrix X;
    protected Matrix T;
    protected Matrix Y;
    public Matrix W;
    public double[] b;
    public double rho;
    public Options options;
    private int nClass;
    protected int nFeature;
    protected int nExample;
    
    public static void main(final String[] args) {
    }
    
    public TemporalLogisticRegression(final double lambda) {
        this.regularizationType = 1;
        this.options = new Options();
        this.nClass = 2;
        this.options.lambda = lambda;
    }
    
    public TemporalLogisticRegression(final Options options) {
        this.regularizationType = 1;
        this.options = new Options();
        this.nClass = 2;
        this.options = options;
    }
    
    public TemporalLogisticRegression() {
        this.regularizationType = 1;
        this.options = new Options();
        this.nClass = 2;
    }
    
    public void initialize(final double rho0) {
        this.rho = rho0;
    }
    
    public void initialize(final double... params) {
        if (params.length == 3) {
            this.rho = params[1];
        }
        else {
            this.rho = params[0];
        }
    }
    
    public void feedData(final Matrix X) {
        this.X = X;
        this.nExample = X.getRowDimension();
        this.nFeature = X.getColumnDimension();
    }
    
    public void feedTime(final Matrix T) {
        this.T = T;
    }
    
    public void feedScore(final int[] labels) {
        final int n = labels.length;
        this.Y = new DenseMatrix(n, 2, 0.0);
        for (int i = 0; i < n; ++i) {
            if (labels[i] == 1) {
                this.Y.setEntry(i, 0, 1.0);
            }
            else if (labels[i] == 0) {
                this.Y.setEntry(i, 1, 1.0);
            }
        }
    }
    
    public void feedScore(final double[] scores) {
        final int n = scores.length;
        this.Y = new DenseMatrix(n, 2, 0.0);
        for (int i = 0; i < n; ++i) {
            if (scores[i] == 1.0) {
                this.Y.setEntry(i, 0, 1.0);
            }
            else if (scores[i] == 0.0) {
                this.Y.setEntry(i, 1, 1.0);
            }
        }
    }
    
    public void feedScore(final Matrix V) {
        final int n = V.getRowDimension();
        this.Y = new DenseMatrix(n, 2, 0.0);
        for (int i = 0; i < n; ++i) {
            if (V.getEntry(i, 0) == 1.0) {
                this.Y.setEntry(i, 0, 1.0);
            }
            else if (V.getEntry(i, 0) == 0.0) {
                this.Y.setEntry(i, 1, 1.0);
            }
        }
    }
    
    public void train() {
        final double lambda = this.options.lambda;
        final double epsilon = this.options.epsilon;
        final int maxIter = this.options.maxIter;
        final DenseMatrix W = (DenseMatrix)Matlab.ones(this.nFeature + 1, this.nClass);
        final Matrix A = new DenseMatrix(this.nExample, this.nClass);
        final Matrix V = A.copy();
        final Matrix G = W.copy();
        final Matrix XW = new DenseMatrix(this.nExample, this.nClass);
        final Matrix VMinusY = new DenseMatrix(this.nExample, this.nClass);
        final Matrix YLogV = new DenseMatrix(this.nExample, this.nClass);
        final Matrix VPlusEps = new DenseMatrix(this.nExample, this.nClass);
        final Matrix LogV = new DenseMatrix(this.nExample, this.nClass);
        final Matrix C = Matlab.speye(this.nExample);
        for (int i = 0; i < this.nExample; ++i) {
            C.setEntry(i, i, (this.Y.getEntry(i, 0) == 0.0) ? 5 : 1);
        }
        double fval = 0.0;
        double hval = 0.0;
        double fval_pre = 0.0;
        final Matrix Grad4Rho = new DenseMatrix(0.0);
        final Matrix Rho = new DenseMatrix(this.rho);
        int cnt = 0;
        do {
            this.computeActivation(A, this.X, W, this.T, this.rho);
            InPlaceOperator.sigmoid(V, A);
            InPlaceOperator.minus(VMinusY, V, this.Y);
            InPlaceOperator.mtimes(VMinusY, C, VMinusY);
            this.computeGradient(G, this.X, VMinusY, this.rho);
            InPlaceOperator.timesAssign(G, 1.0 / this.nExample);
            InPlaceOperator.plus(VPlusEps, V, Matlab.eps);
            InPlaceOperator.log(LogV, VPlusEps);
            InPlaceOperator.times(YLogV, this.Y, LogV);
            InPlaceOperator.mtimes(YLogV, C, YLogV);
            fval = -Matlab.sum(Matlab.sum(YLogV)) / this.nExample;
            boolean[] flags = null;
            AcceleratedProximalGradient.type = 1;
            switch (this.regularizationType) {
                case 1: {
                    AcceleratedProximalGradient.prox = new ProxL1(lambda);
                    break;
                }
                case 2: {
                    AcceleratedProximalGradient.prox = new ProxL2Square(lambda);
                    break;
                }
                case 3: {
                    AcceleratedProximalGradient.prox = new ProxL2(lambda);
                    break;
                }
                case 4: {
                    AcceleratedProximalGradient.prox = new ProxLInfinity(lambda);
                    break;
                }
            }
            while (true) {
                if (this.regularizationType == 0) {
                    flags = LBFGS.run(G, fval, epsilon, W);
                }
                else {
                    flags = AcceleratedProximalGradient.run(G, fval, hval, epsilon, W);
                }
                if (flags[0]) {
                    break;
                }
                this.computeActivation(A, this.X, W, this.T, this.rho);
                InPlaceOperator.sigmoid(V, A);
                InPlaceOperator.plus(VPlusEps, V, Matlab.eps);
                InPlaceOperator.log(LogV, VPlusEps);
                InPlaceOperator.times(YLogV, this.Y, LogV);
                InPlaceOperator.mtimes(YLogV, C, YLogV);
                fval = -Matlab.sum(Matlab.sum(YLogV)) / this.nExample;
                switch (this.regularizationType) {
                    case 1: {
                        hval = lambda * Matlab.sumAll(Matlab.abs(W));
                        break;
                    }
                    case 2: {
                        final double norm = Matlab.norm(W, "fro");
                        hval = lambda * norm * norm;
                        break;
                    }
                    case 3: {
                        hval = lambda * Matlab.norm(W, "fro");
                        break;
                    }
                    case 4: {
                        hval = lambda * Matlab.max(Matlab.max(Matlab.abs(W))[0])[0];
                        break;
                    }
                }
                if (!flags[1]) {
                    continue;
                }
                InPlaceOperator.minus(VMinusY, V, this.Y);
                InPlaceOperator.mtimes(VMinusY, C, VMinusY);
                this.computeGradient(G, this.X, VMinusY, this.rho);
                InPlaceOperator.timesAssign(G, 1.0 / this.nExample);
            }
            Rho.setEntry(0, 0, this.rho);
            this.computeXW(XW, this.X, W, this.T);
            InPlaceOperator.minus(VMinusY, V, this.Y);
            InPlaceOperator.mtimes(VMinusY, C, VMinusY);
            Grad4Rho.setEntry(0, 0, Matlab.innerProduct(VMinusY, XW) / this.nExample);
            hval = 0.0;
            InPlaceOperator.assign(A, XW);
            InPlaceOperator.timesAssign(A, this.rho);
            InPlaceOperator.sigmoid(V, A);
            InPlaceOperator.plus(VPlusEps, V, Matlab.eps);
            InPlaceOperator.log(LogV, VPlusEps);
            InPlaceOperator.times(YLogV, this.Y, LogV);
            InPlaceOperator.mtimes(YLogV, C, YLogV);
            double gval = -Matlab.sum(Matlab.sum(YLogV)) / this.nExample;
            AcceleratedProximalGradient.prox = new Prox();
            final double l = -10.0;
            final double u = 10.0;
            while (true) {
                flags = BoundConstrainedPLBFGS.run(Grad4Rho, gval, l, u, epsilon, Rho);
                this.rho = Rho.getEntry(0, 0);
                if (flags[0]) {
                    break;
                }
                InPlaceOperator.assign(A, XW);
                InPlaceOperator.timesAssign(A, this.rho);
                InPlaceOperator.sigmoid(V, A);
                InPlaceOperator.plus(VPlusEps, V, Matlab.eps);
                InPlaceOperator.log(LogV, VPlusEps);
                InPlaceOperator.times(YLogV, this.Y, LogV);
                InPlaceOperator.mtimes(YLogV, C, YLogV);
                gval = -Matlab.sum(Matlab.sum(YLogV)) / this.nExample;
                if (!flags[1]) {
                    continue;
                }
                InPlaceOperator.minus(VMinusY, V, this.Y);
                InPlaceOperator.mtimes(VMinusY, C, VMinusY);
                Grad4Rho.setEntry(0, 0, Matlab.innerProduct(VMinusY, XW) / this.nExample);
            }
            ++cnt;
            fval = gval + lambda * Matlab.norm(W, 1);
            Printer.fprintf("Iter %d - fval: %.4f\n", cnt, fval);
            if (cnt > 1 && Math.abs(fval_pre - fval) < Matlab.eps) {
                fval_pre = fval;
            }
        } while (cnt < maxIter);
        final double[][] WData = W.getData();
        final double[][] thisWData = new double[this.nFeature][];
        for (int feaIdx = 0; feaIdx < this.nFeature; ++feaIdx) {
            thisWData[feaIdx] = WData[feaIdx];
        }
        this.W = new DenseMatrix(thisWData);
        this.b = WData[this.nFeature];
    }
    
    public Matrix predict(final Matrix Xt, final Matrix Tt) {
        final DenseMatrix ActivationMatrix = (DenseMatrix)Xt.mtimes(this.W);
        final double[][] Activation = ActivationMatrix.getData();
        for (int i = 0; i < Xt.getRowDimension(); ++i) {
            ArrayOperator.plusAssign(Activation[i], this.b);
            final double[] array = Activation[i];
            final int n = 1;
            array[n] += Tt.getEntry(i, 0);
            ArrayOperator.timesAssign(Activation[i], this.rho);
        }
        return Matlab.sigmoid(ActivationMatrix).getColumnMatrix(0);
    }
    
    private void computeGradient(final Matrix res, final Matrix A, final Matrix B, final double rho) {
        if (!(res instanceof SparseMatrix) && res instanceof DenseMatrix) {
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            final int NB = B.getColumnDimension();
            final int N = A.getRowDimension();
            final int M = A.getColumnDimension();
            if (A instanceof DenseMatrix) {
                final double[][] AData = ((DenseMatrix)A).getData();
                if (B instanceof DenseMatrix) {
                    final double[][] BData = ((DenseMatrix)B).getData();
                    double[] BRow = null;
                    double A_ki = 0.0;
                    for (int i = 0; i < M; ++i) {
                        resRow = resData[i];
                        InPlaceOperator.clear(resRow);
                        for (int k = 0; k < N; ++k) {
                            BRow = BData[k];
                            A_ki = AData[k][i];
                            for (int j = 0; j < NB; ++j) {
                                final double[] array = resRow;
                                final int n = j;
                                array[n] += A_ki * BRow[j];
                            }
                        }
                    }
                    resRow = resData[M];
                    InPlaceOperator.clear(resRow);
                    for (int l = 0; l < N; ++l) {
                        BRow = BData[l];
                        for (int m = 0; m < NB; ++m) {
                            final double[] array2 = resRow;
                            final int n2 = m;
                            array2[n2] += BRow[m];
                        }
                    }
                }
                else if (B instanceof SparseMatrix) {
                    int[] ir = null;
                    int[] jc = null;
                    double[] pr = null;
                    ir = ((SparseMatrix)B).getIr();
                    jc = ((SparseMatrix)B).getJc();
                    pr = ((SparseMatrix)B).getPr();
                    int r = -1;
                    double s = 0.0;
                    final double[] columnA = new double[A.getRowDimension()];
                    for (int i2 = 0; i2 < M; ++i2) {
                        for (int t = 0; t < N; ++t) {
                            columnA[t] = AData[t][i2];
                        }
                        resRow = resData[i2];
                        for (int j2 = 0; j2 < NB; ++j2) {
                            s = 0.0;
                            for (int k2 = jc[j2]; k2 < jc[j2 + 1]; ++k2) {
                                r = ir[k2];
                                s += columnA[r] * pr[k2];
                            }
                            resRow[j2] = s;
                        }
                    }
                    resRow = resData[M];
                    for (int j3 = 0; j3 < NB; ++j3) {
                        s = 0.0;
                        for (int k3 = jc[j3]; k3 < jc[j3 + 1]; ++k3) {
                            s += pr[k3];
                        }
                        resRow[j3] = s;
                    }
                }
            }
            else if (A instanceof SparseMatrix) {
                if (B instanceof DenseMatrix) {
                    final int[] ir2 = ((SparseMatrix)A).getIr();
                    final int[] jc2 = ((SparseMatrix)A).getJc();
                    final double[] pr2 = ((SparseMatrix)A).getPr();
                    final double[][] BData2 = ((DenseMatrix)B).getData();
                    double[] BRow2 = null;
                    int c = -1;
                    double s2 = 0.0;
                    for (int i2 = 0; i2 < M; ++i2) {
                        resRow = resData[i2];
                        for (int j2 = 0; j2 < NB; ++j2) {
                            s2 = 0.0;
                            for (int k2 = jc2[i2]; k2 < jc2[i2 + 1]; ++k2) {
                                c = ir2[k2];
                                s2 += pr2[k2] * BData2[c][j2];
                            }
                            resRow[j2] = s2;
                        }
                    }
                    resRow = resData[M];
                    InPlaceOperator.clear(resRow);
                    for (int k4 = 0; k4 < N; ++k4) {
                        BRow2 = BData2[k4];
                        for (int j2 = 0; j2 < NB; ++j2) {
                            final double[] array3 = resRow;
                            final int n3 = j2;
                            array3[n3] += BRow2[j2];
                        }
                    }
                }
                else if (B instanceof SparseMatrix) {
                    final int[] ir3 = ((SparseMatrix)A).getIr();
                    final int[] jc3 = ((SparseMatrix)A).getJc();
                    final double[] pr3 = ((SparseMatrix)A).getPr();
                    final int[] ir4 = ((SparseMatrix)B).getIr();
                    final int[] jc4 = ((SparseMatrix)B).getJc();
                    final double[] pr4 = ((SparseMatrix)B).getPr();
                    int rr = -1;
                    int cl = -1;
                    double s3 = 0.0;
                    int kl = 0;
                    int kr = 0;
                    for (int i3 = 0; i3 < M; ++i3) {
                        resRow = resData[i3];
                        for (int j4 = 0; j4 < NB; ++j4) {
                            s3 = 0.0;
                            kl = jc3[i3];
                            kr = jc4[j4];
                            while (kl < jc3[i3 + 1] && kr < jc4[j4 + 1]) {
                                cl = ir3[kl];
                                rr = ir4[kr];
                                if (cl < rr) {
                                    ++kl;
                                }
                                else if (cl > rr) {
                                    ++kr;
                                }
                                else {
                                    s3 += pr3[kl] * pr4[kr];
                                    ++kl;
                                    ++kr;
                                }
                            }
                            resRow[j4] = s3;
                        }
                    }
                    resRow = resData[M];
                    for (int j5 = 0; j5 < NB; ++j5) {
                        s3 = 0.0;
                        for (int k5 = jc4[j5]; k5 < jc4[j5 + 1]; ++k5) {
                            s3 += pr4[k5];
                        }
                        resRow[j5] = s3;
                    }
                }
            }
            InPlaceOperator.timesAssign(res, rho);
        }
    }
    
    private void computeActivation(final Matrix A, final Matrix X, final Matrix W, final Matrix T, final double rho) {
        final double[][] AData = ((DenseMatrix)A).getData();
        double[] ARow = null;
        final double[][] WData = ((DenseMatrix)W).getData();
        final double[] WColumn = new double[W.getRowDimension()];
        double[] WRow = null;
        if (X instanceof DenseMatrix) {
            final double[][] XData = ((DenseMatrix)X).getData();
            double[] XRow = null;
            double s = 0.0;
            for (int j = 0; j < this.nClass; ++j) {
                for (int r = 0; r < W.getRowDimension(); ++r) {
                    WColumn[r] = WData[r][j];
                }
                for (int i = 0; i < this.nExample; ++i) {
                    XRow = XData[i];
                    s = 0.0;
                    for (int k = 0; k < this.nFeature; ++k) {
                        s += XRow[k] * WColumn[k];
                    }
                    AData[i][j] = s + WColumn[this.nFeature];
                }
            }
        }
        else if (X instanceof SparseMatrix) {
            final int[] ic = ((SparseMatrix)X).getIc();
            final int[] jr = ((SparseMatrix)X).getJr();
            final int[] valCSRIndices = ((SparseMatrix)X).getValCSRIndices();
            final double[] pr = ((SparseMatrix)X).getPr();
            int feaIdx = -1;
            double v = 0.0;
            for (int l = 0; l < this.nExample; ++l) {
                ARow = AData[l];
                InPlaceOperator.clear(ARow);
                for (int m = jr[l]; m < jr[l + 1]; ++m) {
                    feaIdx = ic[m];
                    WRow = WData[feaIdx];
                    v = pr[valCSRIndices[m]];
                    for (int j2 = 0; j2 < this.nClass; ++j2) {
                        final double[] array = ARow;
                        final int n = j2;
                        array[n] += v * WRow[j2];
                    }
                }
                WRow = WData[this.nFeature];
                for (int j3 = 0; j3 < this.nClass; ++j3) {
                    final double[] array2 = ARow;
                    final int n2 = j3;
                    array2[n2] += WRow[j3];
                }
            }
        }
        for (int i2 = 0; i2 < this.nExample; ++i2) {
            final double[] array3 = AData[i2];
            final int n3 = 1;
            array3[n3] += T.getEntry(i2, 0);
        }
        InPlaceOperator.timesAssign(A, rho);
    }
    
    private void computeXW(final Matrix A, final Matrix X, final Matrix W, final Matrix T) {
        final double[][] AData = ((DenseMatrix)A).getData();
        double[] ARow = null;
        final double[][] WData = ((DenseMatrix)W).getData();
        final double[] WColumn = new double[W.getRowDimension()];
        double[] WRow = null;
        if (X instanceof DenseMatrix) {
            final double[][] XData = ((DenseMatrix)X).getData();
            double[] XRow = null;
            double s = 0.0;
            for (int j = 0; j < this.nClass; ++j) {
                for (int r = 0; r < W.getRowDimension(); ++r) {
                    WColumn[r] = WData[r][j];
                }
                for (int i = 0; i < this.nExample; ++i) {
                    XRow = XData[i];
                    s = 0.0;
                    for (int k = 0; k < this.nFeature; ++k) {
                        s += XRow[k] * WColumn[k];
                    }
                    AData[i][j] = s + WColumn[this.nFeature];
                }
            }
        }
        else if (X instanceof SparseMatrix) {
            final int[] ic = ((SparseMatrix)X).getIc();
            final int[] jr = ((SparseMatrix)X).getJr();
            final int[] valCSRIndices = ((SparseMatrix)X).getValCSRIndices();
            final double[] pr = ((SparseMatrix)X).getPr();
            int feaIdx = -1;
            double v = 0.0;
            for (int l = 0; l < this.nExample; ++l) {
                ARow = AData[l];
                InPlaceOperator.clear(ARow);
                for (int m = jr[l]; m < jr[l + 1]; ++m) {
                    feaIdx = ic[m];
                    WRow = WData[feaIdx];
                    v = pr[valCSRIndices[m]];
                    for (int j2 = 0; j2 < this.nClass; ++j2) {
                        final double[] array = ARow;
                        final int n = j2;
                        array[n] += v * WRow[j2];
                    }
                }
                WRow = WData[this.nFeature];
                for (int j3 = 0; j3 < this.nClass; ++j3) {
                    final double[] array2 = ARow;
                    final int n2 = j3;
                    array2[n2] += WRow[j3];
                }
            }
        }
        for (int i2 = 0; i2 < this.nExample; ++i2) {
            final double[] array3 = AData[i2];
            final int n3 = 1;
            array3[n3] += T.getEntry(i2, 0);
        }
    }
    
    public void loadModel(final String filePath) {
        try {
            final ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath));
            this.W = (Matrix)ois.readObject();
            this.b = (double[])ois.readObject();
            this.rho = ois.readDouble();
            this.options.lambda = ois.readDouble();
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
    
    public void saveModel(final String filePath) {
        final File parentFile = new File(filePath).getParentFile();
        if (parentFile != null && !parentFile.exists()) {
            parentFile.mkdirs();
        }
        try {
            final ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath));
            oos.writeObject(this.W);
            oos.writeObject(this.b);
            oos.writeDouble(this.rho);
            oos.writeDouble(this.options.lambda);
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
