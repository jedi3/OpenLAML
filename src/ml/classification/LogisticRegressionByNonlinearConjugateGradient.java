package ml.classification;

import la.io.*;
import java.io.*;
import la.vector.*;
import ml.optimization.*;
import la.matrix.*;
import ml.utils.*;

public class LogisticRegressionByNonlinearConjugateGradient extends Classifier
{
    private static final long serialVersionUID = -1812982662269422441L;
    
    public static void main(final String[] args) {
        final double[][] data = { { 3.5, 5.3, 0.2, -1.2 }, { 4.4, 2.2, 0.3, 0.4 }, { 1.3, 0.5, 4.1, 3.2 } };
        final int[] labels = { 1, 2, 3 };
        Classifier logReg = new LogisticRegressionByNonlinearConjugateGradient();
        logReg.epsilon = 1.0E-5;
        logReg.feedData(data);
        logReg.feedLabels(labels);
        long start = System.currentTimeMillis();
        logReg.train();
        System.out.format("Elapsed time: %.3f seconds.%n", (System.currentTimeMillis() - start) / 1000.0f);
        Printer.fprintf("W:%n", new Object[0]);
        Printer.printMatrix(logReg.W);
        Printer.fprintf("b:%n", new Object[0]);
        Printer.printVector(logReg.b);
        final double[][] dataTest = data;
        Printer.fprintf("Ground truth:%n", new Object[0]);
        Printer.printMatrix(logReg.Y);
        Printer.fprintf("Predicted probability matrix:%n", new Object[0]);
        final Matrix Prob_pred = logReg.predictLabelScoreMatrix(dataTest);
        Printer.disp(Prob_pred);
        Printer.fprintf("Predicted label matrix:%n", new Object[0]);
        final Matrix Y_pred = logReg.predictLabelMatrix(dataTest);
        Printer.printMatrix(Y_pred);
        int[] pred_labels = logReg.predict(dataTest);
        Classifier.getAccuracy(pred_labels, labels);
        start = System.currentTimeMillis();
        final String filePath = "heart_scale";
        logReg = new LogisticRegressionByNonlinearConjugateGradient();
        DataSet dataSet = null;
        try {
            dataSet = DataSet.readDataSetFromFile(filePath);
        }
        catch (IOException e) {
            e.printStackTrace();
        }
        catch (InvalidInputDataException e2) {
            e2.printStackTrace();
        }
        logReg.feedData(dataSet.X);
        logReg.feedLabels(dataSet.Y);
        logReg.train();
        final Matrix XTest = dataSet.X;
        pred_labels = logReg.predict(XTest);
        Classifier.getAccuracy(pred_labels, logReg.labels);
        System.out.format("Elapsed time: %.2f seconds.%n", (System.currentTimeMillis() - start) / 1000.0f);
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
        Matrix A = null;
        Matrix V = null;
        Matrix G = null;
        double fval = 0.0;
        final DenseMatrix W = new DenseMatrix(this.nFeature + 1, this.nClass);
        A = new DenseMatrix(this.nExample, this.nClass);
        this.computeActivation(A, this.X, W);
        V = Matlab.sigmoid(A);
        G = W.copy();
        final Matrix VMinusY = new DenseMatrix(this.nExample, this.nClass);
        InPlaceOperator.minus(VMinusY, V, this.Y);
        this.computeGradient(G, this.X, VMinusY);
        InPlaceOperator.timesAssign(G, 1.0 / this.nExample);
        final Matrix YLogV = new DenseMatrix(this.nExample, this.nClass);
        final Matrix VPlusEps = new DenseMatrix(this.nExample, this.nClass);
        final Matrix LogV = new DenseMatrix(this.nExample, this.nClass);
        InPlaceOperator.plus(VPlusEps, V, Matlab.eps);
        InPlaceOperator.log(LogV, VPlusEps);
        InPlaceOperator.times(YLogV, this.Y, LogV);
        fval = -Matlab.sum(Matlab.sum(YLogV)) / this.nExample;
        boolean[] flags = null;
        while (true) {
            flags = NonlinearConjugateGradient.run(G, fval, this.epsilon, W);
            Printer.disp("W:");
            Printer.disp(W);
            if (flags[0]) {
                break;
            }
            this.computeActivation(A, this.X, W);
            V = Matlab.sigmoid(A);
            InPlaceOperator.plus(VPlusEps, V, Matlab.eps);
            InPlaceOperator.log(LogV, VPlusEps);
            InPlaceOperator.times(YLogV, this.Y, LogV);
            fval = -Matlab.sum(Matlab.sum(YLogV)) / this.nExample;
            if (!flags[1]) {
                continue;
            }
            InPlaceOperator.minus(VMinusY, V, this.Y);
            this.computeGradient(G, this.X, VMinusY);
            InPlaceOperator.timesAssign(G, 1.0 / this.nExample);
        }
        final double[][] WData = W.getData();
        final double[][] thisWData = new double[this.nFeature][];
        for (int feaIdx = 0; feaIdx < this.nFeature; ++feaIdx) {
            thisWData[feaIdx] = WData[feaIdx];
        }
        this.W = new DenseMatrix(thisWData);
        this.b = WData[this.nFeature];
    }
    
    private void computeGradient(final Matrix res, final Matrix A, final Matrix B) {
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
        }
    }
    
    private Matrix getXT(final Matrix X) {
        if (X instanceof DenseMatrix) {
            final double[][] resData = ArrayOperator.allocate2DArray(this.nFeature + 1, this.nExample, 0.0);
            double[] resRow = null;
            final double[][] XData = ((DenseMatrix)X).getData();
            for (int feaIdx = 0; feaIdx < this.nFeature; ++feaIdx) {
                resRow = resData[feaIdx];
                for (int sampleIdx = 0; sampleIdx < this.nExample; ++sampleIdx) {
                    resRow[sampleIdx] = XData[sampleIdx][feaIdx];
                }
            }
            resRow = resData[this.nFeature];
            InPlaceOperator.assign(resRow, 1.0);
            return new DenseMatrix(resData);
        }
        if (X instanceof SparseMatrix) {
            final SparseMatrix res = (SparseMatrix)X.transpose();
            res.appendAnEmptyRow();
            for (int sampleIdx2 = 0; sampleIdx2 < this.nExample; ++sampleIdx2) {
                res.setEntry(this.nFeature, sampleIdx2, 1.0);
            }
            return res;
        }
        return null;
    }
    
    private void computeActivation(final Matrix A, final Matrix X, final Matrix W) {
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
    }
    
    @Override
    public Matrix predictLabelScoreMatrix(final Matrix Xt) {
        final DenseMatrix ScoreMatrix = (DenseMatrix)Xt.mtimes(this.W);
        final double[][] scoreData = ScoreMatrix.getData();
        for (int i = 0; i < Xt.getRowDimension(); ++i) {
            ArrayOperator.plusAssign(scoreData[i], this.b);
        }
        return Matlab.sigmoid(ScoreMatrix);
    }
}
