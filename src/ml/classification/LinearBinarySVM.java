package ml.classification;

import la.io.*;
import java.io.*;
import ml.utils.*;
import la.matrix.*;

public class LinearBinarySVM extends Classifier
{
    private static final long serialVersionUID = -2374085637018946130L;
    double C;
    double eps;
    
    public static void main(final String[] args) throws IOException, InvalidInputDataException {
        double C = 1.0;
        double eps = 1.0E-4;
        Classifier linearBinarySVM = new LinearBinarySVM(C, eps);
        int[] pred_labels = null;
        final double[][] data = { { 3.5, 4.4, 1.3, 2.3 }, { 5.3, 2.2, 0.5, 4.5 }, { 0.2, 0.3, 4.1, -3.1 }, { -1.2, 0.4, 3.2, 1.6 } };
        final int[] labels = { 1, 1, -1, -1 };
        linearBinarySVM.feedData(data);
        linearBinarySVM.feedLabels(labels);
        linearBinarySVM.train();
        Printer.fprintf("W:%n", new Object[0]);
        Printer.printMatrix(linearBinarySVM.W);
        Printer.fprintf("b:%n", new Object[0]);
        Printer.printVector(linearBinarySVM.b);
        pred_labels = linearBinarySVM.predict(data);
        Classifier.getAccuracy(pred_labels, labels);
        final long start = System.currentTimeMillis();
        final String trainDataFilePath = "heart_scale";
        C = 1.0;
        eps = 0.01;
        linearBinarySVM = new LinearBinarySVM(C, eps);
        final DataSet dataSet = DataSet.readDataSetFromFile(trainDataFilePath);
        linearBinarySVM.feedData(dataSet.X);
        linearBinarySVM.feedLabels(dataSet.Y);
        linearBinarySVM.train();
        final Matrix XTest = dataSet.X;
        pred_labels = linearBinarySVM.predict(XTest);
        Classifier.getAccuracy(pred_labels, linearBinarySVM.labels);
        System.out.format("Elapsed time: %.2f seconds.%n", (System.currentTimeMillis() - start) / 1000.0f);
    }
    
    public LinearBinarySVM() {
        this.C = 1.0;
        this.eps = 0.01;
    }
    
    public LinearBinarySVM(final double C, final double eps) {
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
        final double[] Y = new double[this.nExample];
        for (int i = 0; i < this.nExample; ++i) {
            Y[i] = -2.0 * (this.labelIDs[i] - 0.5);
        }
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
        final double[] W = new double[this.nFeature + 1];
        final double[] Q = this.computeQ(this.X, pr_CSR);
        final double[] alphas = ArrayOperator.allocateVector(this.nExample, 0.0);
        double M = Double.NEGATIVE_INFINITY;
        double m = Double.POSITIVE_INFINITY;
        double Grad = 0.0;
        double alpha_old = 0.0;
        double alpha_new = 0.0;
        double PGrad = 0.0;
        int cnt = 1;
        do {
            M = Double.NEGATIVE_INFINITY;
            m = Double.POSITIVE_INFINITY;
            for (int j = 0; j < this.nExample; ++j) {
                Grad = Y[j] * this.innerProduct(W, this.X, j, pr_CSR) - 1.0;
                alpha_old = alphas[j];
                if (alpha_old == 0.0) {
                    PGrad = Math.min(Grad, 0.0);
                }
                else if (alpha_old == this.C) {
                    PGrad = Math.max(Grad, 0.0);
                }
                else {
                    PGrad = Grad;
                }
                M = Math.max(M, PGrad);
                m = Math.min(m, PGrad);
                if (PGrad != 0.0) {
                    alpha_new = Math.min(Math.max(alpha_old - Grad / Q[j], 0.0), this.C);
                    this.updateW(W, Y[j] * (alpha_new - alpha_old), this.X, j, pr_CSR);
                    alphas[j] = alpha_new;
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
        final double[] weights = new double[this.X.getColumnDimension()];
        System.arraycopy(W, 0, weights, 0, this.X.getColumnDimension());
        this.W = new DenseMatrix(weights, 1);
        (this.b = new double[1])[0] = W[this.X.getColumnDimension()];
    }
    
    private double innerProduct(final double[] W, final Matrix X, final int i, final double[] pr_CSR) {
        final int N = X.getColumnDimension();
        double res = 0.0;
        double s = 0.0;
        if (X instanceof DenseMatrix) {
            final double[][] XData = ((DenseMatrix)X).getData();
            double[] XRow = null;
            XRow = XData[i];
            for (int j = 0; j < N; ++j) {
                s += W[j] * XRow[j];
            }
            res = s + W[N];
        }
        else if (X instanceof SparseMatrix) {
            final int[] ic = ((SparseMatrix)X).getIc();
            final int[] jr = ((SparseMatrix)X).getJr();
            for (int k = jr[i]; k < jr[i + 1]; ++k) {
                s += W[ic[k]] * pr_CSR[k];
            }
            res = s + W[N];
        }
        return res;
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
                Q[i] = s;
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
                Q[k] = s;
            }
        }
        return Q;
    }
    
    private void updateW(final double[] W, final double v, final Matrix X, final int i, final double[] pr_CSR) {
        final int N = X.getColumnDimension();
        if (X instanceof DenseMatrix) {
            final double[][] XData = ((DenseMatrix)X).getData();
            double[] XRow = null;
            XRow = XData[i];
            for (int j = 0; j < N; ++j) {
                final int n = j;
                W[n] += v * XRow[j];
            }
            final int n2 = N;
            W[n2] += v;
        }
        else if (X instanceof SparseMatrix) {
            final int[] ic = ((SparseMatrix)X).getIc();
            final int[] jr = ((SparseMatrix)X).getJr();
            for (int k = jr[i]; k < jr[i + 1]; ++k) {
                final int n3 = ic[k];
                W[n3] += v * pr_CSR[k];
            }
            final int n4 = N;
            W[n4] += v;
        }
    }
    
    @Override
    public Matrix predictLabelScoreMatrix(final Matrix Xt) {
        final int n = Xt.getRowDimension();
        final double[][] ScoreData = ((DenseMatrix)Xt.mtimes(this.W).plus(this.b[0])).getData();
        final DenseMatrix ScoreMatrix = new DenseMatrix(n, 2);
        final double[][] scores = ScoreMatrix.getData();
        for (int i = 0; i < n; ++i) {
            scores[i][0] = ScoreData[i][0];
            scores[i][1] = -ScoreData[i][0];
        }
        return ScoreMatrix;
    }
}
