package ml.temporal;

import la.matrix.*;
import ml.utils.*;

public abstract class SurvivalScoreModel
{
    public Matrix X;
    public Matrix T;
    public Matrix Y;
    public Matrix W;
    public int n;
    public int p;
    
    public void feedData(final Matrix X) {
        this.X = X;
        this.n = X.getRowDimension();
        this.p = X.getColumnDimension();
    }
    
    public void feedTime(final Matrix T) {
        this.T = T;
    }
    
    public void feedScore(final Matrix Y) {
        this.Y = Y;
    }
    
    public void feedScore(final double[] scores) {
        this.Y = new DenseMatrix(scores, 1);
    }
    
    public void feedScore(final int[] labels) {
        final double[] scores = new double[labels.length];
        for (int i = 0; i < labels.length; ++i) {
            scores[i] = labels[i];
        }
        this.Y = new DenseMatrix(scores, 1);
    }
    
    public abstract void initialize(final double... p0);
    
    public abstract void train();
    
    public abstract Matrix predict(final Matrix p0, final Matrix p1);
    
    public abstract void loadModel(final String p0);
    
    public abstract void saveModel(final String p0);
    
    public static double[][] computeROC(final int[] scoreArray, final Matrix Yhat, final int numROCPoints) {
        final double[] labels = new double[scoreArray.length];
        for (int i = 0; i < scoreArray.length; ++i) {
            labels[i] = scoreArray[i];
        }
        final Matrix Yt = new DenseMatrix(labels, 1);
        return computeROC(Yt, Yhat, numROCPoints);
    }
    
    public static double[][] computeROC(final double[] labels, final Matrix Yhat, final int numROCPoints) {
        final Matrix Yt = new DenseMatrix(labels, 1);
        return computeROC(Yt, Yhat, numROCPoints);
    }
    
    public static double[][] computeROC(final Matrix Yt, final Matrix Yhat, final int numROCPoints) {
        final double[] Y = Matlab.full(Yt.getColumnVector(0)).getPr();
        final int n = Y.length;
        final double[] predY = Matlab.full(Yhat.getColumnVector(0)).getPr();
        double threshold = 0.0;
        final double min = min(predY);
        final double max = max(predY);
        final double d = (max - min) / (numROCPoints - 1);
        final double[] FPRs = ArrayOperator.allocate1DArray(numROCPoints);
        final double[] TPRs = ArrayOperator.allocate1DArray(numROCPoints);
        for (int k = 0; k < numROCPoints; ++k) {
            threshold = min + k * d;
            double FP = 0.0;
            double TP = 0.0;
            double N = 0.0;
            double P = 0.0;
            for (int i = 0; i < n; ++i) {
                if (Y[i] == 1.0) {
                    ++P;
                    if (predY[i] >= threshold) {
                        ++TP;
                    }
                }
                else {
                    ++N;
                    if (predY[i] >= threshold) {
                        ++FP;
                    }
                }
            }
            final double FPR = FP / N;
            final double TPR = TP / P;
            FPRs[k] = FPR;
            TPRs[k] = TPR;
        }
        return new double[][] { FPRs, TPRs };
    }
    
    public static double min(final double[] V) {
        double res = V[0];
        for (int i = 1; i < V.length; ++i) {
            if (res > V[i]) {
                res = V[i];
            }
        }
        return res;
    }
    
    public static double max(final double[] V) {
        double res = V[0];
        for (int i = 1; i < V.length; ++i) {
            if (res < V[i]) {
                res = V[i];
            }
        }
        return res;
    }
}
