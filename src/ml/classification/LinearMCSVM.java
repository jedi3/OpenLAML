package ml.classification;

import la.io.*;
import java.io.*;
import ml.utils.*;
import la.matrix.*;

public class LinearMCSVM extends Classifier
{
    private static final long serialVersionUID = -4808466628014511429L;
    double C;
    double eps;
    
    public static void main(final String[] args) throws IOException, InvalidInputDataException {
        double C = 1.0;
        double eps = 1.0E-4;
        Classifier linearMCSVM = new LinearMCSVM(C, eps);
        final double[][] data = { { 3.5, 4.4, 1.3, 2.3 }, { 5.3, 2.2, 0.5, 4.5 }, { 0.2, 0.3, 4.1, -3.1 }, { -1.2, 0.4, 3.2, 1.6 } };
        final int[] labels = { 1, 2, 3, 4 };
        linearMCSVM.feedData(data);
        linearMCSVM.feedLabels(labels);
        linearMCSVM.train();
        Printer.fprintf("W:%n", new Object[0]);
        Printer.printMatrix(linearMCSVM.W);
        Printer.fprintf("b:%n", new Object[0]);
        Printer.printVector(linearMCSVM.b);
        int[] pred_labels = linearMCSVM.predict(data);
        Classifier.getAccuracy(pred_labels, labels);
        final long start = System.currentTimeMillis();
        final String filePath = "heart_scale";
        C = 1.0;
        eps = 0.01;
        linearMCSVM = new LinearMCSVM(C, eps);
        final DataSet dataSet = DataSet.readDataSetFromFile(filePath);
        linearMCSVM.feedData(dataSet.X);
        linearMCSVM.feedLabels(dataSet.Y);
        linearMCSVM.train();
        final Matrix XTest = dataSet.X;
        pred_labels = linearMCSVM.predict(XTest);
        Classifier.getAccuracy(pred_labels, linearMCSVM.labels);
        System.out.format("Elapsed time: %.2f seconds.%n", (System.currentTimeMillis() - start) / 1000.0f);
    }
    
    public LinearMCSVM() {
        this.C = 1.0;
        this.eps = 0.01;
    }
    
    public LinearMCSVM(final double C, final double eps) {
        this.C = C;
        this.eps = eps;
    }
    
    @Override
    public void loadModel(final String filePath) {
        System.out.println("Loading model...");
        try {
            final ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath));
            this.W = (Matrix)ois.readObject();
            this.b = (double[])ois.readObject();
            this.IDLabelMap = (int[])ois.readObject();
            this.nClass = this.IDLabelMap.length;
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
            oos.writeObject(this.b);
            oos.writeObject(this.IDLabelMap);
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
    
    @Override
    public void train() {
        double[] pr_CSR = null;
        if (this.X instanceof SparseMatrix) {
            final double[] pr = ((SparseMatrix)this.X).getPr();
            final int[] valCSRIndices = ((SparseMatrix)this.X).getValCSRIndices();
            final int nnz = ((SparseMatrix)this.X).getNNZ();
            pr_CSR = ArrayOperator.allocate1DArray(nnz);
            for (int k = 0; k < nnz; ++k) {
                pr_CSR[k] = pr[valCSRIndices[k]];
            }
        }
        final double[][] Ws = new double[this.nClass][];
        for (int c = 0; c < this.nClass; ++c) {
            Ws[c] = ArrayOperator.allocateVector(this.nFeature + 1, 0.0);
        }
        final double[] Q = this.computeQ(this.X, pr_CSR);
        final double[][] Alpha = new DenseMatrix(this.nExample, this.nClass, 0.0).getData();
        final int Np = this.nExample * this.nClass;
        double M = Double.NEGATIVE_INFINITY;
        double m = Double.POSITIVE_INFINITY;
        double Grad = 0.0;
        double alpha_old = 0.0;
        double alpha_new = 0.0;
        double PGrad = 0.0;
        final int C = this.nClass;
        final int[] y = this.labelIDs;
        double delta = 0.0;
        int cnt = 1;
        do {
            M = Double.NEGATIVE_INFINITY;
            m = Double.POSITIVE_INFINITY;
            for (int i = 0; i < Np; ++i) {
                final int q = i % C;
                final int j = (i - q) / C;
                final int p = y[j];
                if (q != p) {
                    Grad = this.computeGradient(this.X, j, pr_CSR, Ws[p], Ws[q]);
                    alpha_old = Alpha[j][q];
                    if (alpha_old == 0.0) {
                        PGrad = Math.min(Grad, 0.0);
                    }
                    else if (alpha_old == C) {
                        PGrad = Math.max(Grad, 0.0);
                    }
                    else {
                        PGrad = Grad;
                    }
                    M = Math.max(M, PGrad);
                    m = Math.min(m, PGrad);
                    if (PGrad != 0.0) {
                        alpha_new = Math.min(Math.max(alpha_old - Grad / Q[j], 0.0), C);
                        Alpha[j][q] = alpha_new;
                        delta = alpha_new - alpha_old;
                        this.updateW(Ws[p], Ws[q], delta, this.X, j, pr_CSR);
                    }
                }
            }
            if (cnt % 20 == 0) {
                Printer.fprintf(".", new Object[0]);
            }
            if (cnt % 400 == 0) {
                Printer.fprintf("%n", new Object[0]);
            }
            ++cnt;
        } while (Math.abs(M - m) > this.eps);
        Printer.fprintf("%n", new Object[0]);
        final double[][] weights = new double[this.nClass][];
        this.b = new double[this.nClass];
        for (int c2 = 0; c2 < this.nClass; ++c2) {
            weights[c2] = new double[this.nFeature];
            System.arraycopy(Ws[c2], 0, weights[c2], 0, this.nFeature);
            this.b[c2] = Ws[c2][this.nFeature];
        }
        this.W = new DenseMatrix(weights).transpose();
    }
    
    private double[] computeQ(final Matrix X, final double[] pr_CSR) {
        final int l = X.getRowDimension();
        final double[] Q = new double[l];
        double s = 0.0;
        double v = 0.0;
        final int M = X.getRowDimension();
        final int N = X.getColumnDimension();
        if (X instanceof DenseMatrix) {
            final double[][] XData = ((DenseMatrix)X).getData();
            double[] XRow = null;
            for (int i = 0; i < M; ++i) {
                XRow = XData[i];
                s = 1.0;
                for (int j = 0; j < N; ++j) {
                    v = XRow[j];
                    s += v * v;
                }
                Q[i] = 2.0 * s;
            }
        }
        else if (X instanceof SparseMatrix) {
            final int[] jr = ((SparseMatrix)X).getJr();
            for (int k = 0; k < M; ++k) {
                s = 1.0;
                for (int m = jr[k]; m < jr[k + 1]; ++m) {
                    v = pr_CSR[m];
                    s += v * v;
                }
                Q[k] = 2.0 * s;
            }
        }
        return Q;
    }
    
    private void updateW(final double[] Wp, final double[] Wq, final double delta, final Matrix X, final int i, final double[] pr_CSR) {
        final int N = X.getColumnDimension();
        double v = 0.0;
        if (X instanceof DenseMatrix) {
            final double[][] XData = ((DenseMatrix)X).getData();
            double[] XRow = null;
            XRow = XData[i];
            for (int j = 0; j < N; ++j) {
                v = delta * XRow[j];
                final int n = j;
                Wp[n] += v;
                final int n2 = j;
                Wq[n2] -= v;
            }
            final int n3 = N;
            Wp[n3] += delta;
            final int n4 = N;
            Wq[n4] -= delta;
        }
        else if (X instanceof SparseMatrix) {
            final int[] ic = ((SparseMatrix)X).getIc();
            final int[] jr = ((SparseMatrix)X).getJr();
            int idx = 0;
            for (int k = jr[i]; k < jr[i + 1]; ++k) {
                idx = ic[k];
                v = delta * pr_CSR[k];
                final int n5 = idx;
                Wp[n5] += v;
                final int n6 = idx;
                Wq[n6] -= v;
            }
            final int n7 = N;
            Wp[n7] += delta;
            final int n8 = N;
            Wq[n8] -= delta;
        }
    }
    
    private double computeGradient(final Matrix X, final int i, final double[] pr_CSR, final double[] Wp, final double[] Wq) {
        final int N = X.getColumnDimension();
        double res = 0.0;
        double s = -1.0;
        if (X instanceof DenseMatrix) {
            final double[][] XData = ((DenseMatrix)X).getData();
            double[] XRow = null;
            XRow = XData[i];
            for (int j = 0; j < N; ++j) {
                s += (Wp[j] - Wq[j]) * XRow[j];
            }
            res = s + Wp[N] - Wq[N];
        }
        else if (X instanceof SparseMatrix) {
            final int[] ic = ((SparseMatrix)X).getIc();
            final int[] jr = ((SparseMatrix)X).getJr();
            int idx = 0;
            for (int k = jr[i]; k < jr[i + 1]; ++k) {
                idx = ic[k];
                s += (Wp[idx] - Wq[idx]) * pr_CSR[k];
            }
            res = s + Wp[N] - Wq[N];
        }
        return res;
    }
    
    @Override
    public Matrix predictLabelScoreMatrix(final Matrix Xt) {
        final int n = Xt.getRowDimension();
        Matrix ScoreMatrix = null;
        final Matrix Bias = new DenseMatrix(n, 1, 1.0).mtimes(new DenseMatrix(this.b, 2));
        ScoreMatrix = Xt.mtimes(this.W).plus(Bias);
        return ScoreMatrix;
    }
}
