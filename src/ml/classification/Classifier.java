package ml.classification;

import java.io.*;
import java.util.*;

import la.matrix.*;
import la.vector.*;
import la.vector.Vector;
import ml.utils.*;

public abstract class Classifier implements Serializable
{
    private static final long serialVersionUID = 2859398998295434078L;
    protected int nClass;
    protected int nFeature;
    protected int nExample;
    protected Matrix X;
    protected Matrix Y;
    protected int[] labelIDs;
    protected int[] labels;
    protected Matrix W;
    protected double[] b;
    protected double epsilon;
    protected int[] IDLabelMap;
    
    public Classifier() {
        this.nClass = 0;
        this.nFeature = 0;
        this.nExample = 0;
        this.X = null;
        this.W = null;
        this.epsilon = 1.0E-4;
    }
    
    public abstract void loadModel(final String p0);
    
    public abstract void saveModel(final String p0);
    
    public void feedData(final Matrix X) {
        this.X = X;
        this.nFeature = X.getColumnDimension();
        this.nExample = X.getRowDimension();
    }
    
    public void feedData(final double[][] data) {
        this.feedData(new DenseMatrix(data));
    }
    
    public static int calcNumClass(final int[] labels) {
        final TreeMap<Integer, Integer> IDLabelMap = new TreeMap<Integer, Integer>();
        int ID = 0;
        int label = -1;
        for (int i = 0; i < labels.length; ++i) {
            label = labels[i];
            if (!IDLabelMap.containsValue(label)) {
                IDLabelMap.put(ID++, label);
            }
        }
        final int nClass = IDLabelMap.size();
        return nClass;
    }
    
    public static int[] getIDLabelMap(final int[] labels) {
        final TreeMap<Integer, Integer> IDLabelMap = new TreeMap<Integer, Integer>();
        int ID = 0;
        int label = -1;
        for (int i = 0; i < labels.length; ++i) {
            label = labels[i];
            if (!IDLabelMap.containsValue(label)) {
                IDLabelMap.put(ID++, label);
            }
        }
        final int nClass = IDLabelMap.size();
        final int[] IDLabelArray = new int[nClass];
        for (final int idx : IDLabelMap.keySet()) {
            IDLabelArray[idx] = IDLabelMap.get(idx);
        }
        return IDLabelArray;
    }
    
    public static TreeMap<Integer, Integer> getLabelIDMap(final int[] labels) {
        final TreeMap<Integer, Integer> labelIDMap = new TreeMap<Integer, Integer>();
        int ID = 0;
        int label = -1;
        for (int i = 0; i < labels.length; ++i) {
            label = labels[i];
            if (!labelIDMap.containsKey(label)) {
                labelIDMap.put(label, ID++);
            }
        }
        return labelIDMap;
    }
    
    public void feedLabels(final int[] labels) {
        this.nClass = calcNumClass(labels);
        this.IDLabelMap = getIDLabelMap(labels);
        final TreeMap<Integer, Integer> labelIDMap = getLabelIDMap(labels);
        final int[] labelIDs = new int[labels.length];
        for (int i = 0; i < labels.length; ++i) {
            labelIDs[i] = labelIDMap.get(labels[i]);
        }
        final int[] labelIndices = labelIDs;
        this.Y = labelIndexArray2LabelMatrix(labelIndices, this.nClass);
        this.labels = labels;
        this.labelIDs = labelIndices;
    }
    
    public void feedLabels(final Matrix Y) {
        this.Y = Y;
        this.nClass = Y.getColumnDimension();
        if (this.nExample != Y.getRowDimension()) {
            System.err.println("Number of labels error!");
            System.exit(1);
        }
        final int[] labelIndices = labelScoreMatrix2LabelIndexArray(Y);
        this.labels = labelIndices;
        this.IDLabelMap = getIDLabelMap(this.labels);
        this.labelIDs = labelIndices;
    }
    
    public void feedLabels(final double[][] labels) {
        this.feedLabels(new DenseMatrix(labels));
    }
    
    public abstract void train();
    
    public int[] predict(final Matrix Xt) {
        final Matrix Yt = this.predictLabelScoreMatrix(Xt);
        final int[] labelIndices = labelScoreMatrix2LabelIndexArray(Yt);
        final int[] labels = new int[labelIndices.length];
        for (int i = 0; i < labelIndices.length; ++i) {
            labels[i] = this.IDLabelMap[labelIndices[i]];
        }
        return labels;
    }
    
    public int[] predict(final double[][] Xt) {
        return this.predict(new DenseMatrix(Xt));
    }
    
    public abstract Matrix predictLabelScoreMatrix(final Matrix p0);
    
    public Matrix predictLabelScoreMatrix(final double[][] Xt) {
        return this.predictLabelScoreMatrix(new DenseMatrix(Xt));
    }
    
    public Matrix predictLabelMatrix(final Matrix Xt) {
        final Matrix Yt = this.predictLabelScoreMatrix(Xt);
        final int[] labelIndices = labelScoreMatrix2LabelIndexArray(Yt);
        return labelIndexArray2LabelMatrix(labelIndices, this.nClass);
    }
    
    public Matrix predictLabelMatrix(final double[][] Xt) {
        return this.predictLabelMatrix(new DenseMatrix(Xt));
    }
    
    public static double getAccuracy(final int[] pre_labels, final int[] labels) {
        if (pre_labels.length != labels.length) {
            System.err.println("Number of predicted labels and number of true labels mismatch.");
            System.exit(1);
        }
        final int N = labels.length;
        int cnt_correct = 0;
        for (int i = 0; i < N; ++i) {
            if (pre_labels[i] == labels[i]) {
                ++cnt_correct;
            }
        }
        final double accuracy = cnt_correct / N;
        System.out.println(String.format("Accuracy: %.2f%%\n", accuracy * 100.0));
        return accuracy;
    }
    
    public Matrix getProjectionMatrix() {
        return this.W;
    }
    
    public Matrix getTrainingLabelMatrix() {
        return this.Y;
    }
    
    public static int[] labelScoreMatrix2LabelIndexArray(final Matrix Y) {
        final int[] labelIndices = new int[Y.getRowDimension()];
        if (Y instanceof SparseMatrix) {
            final int[] ic = ((SparseMatrix)Y).getIc();
            final int[] jr = ((SparseMatrix)Y).getJr();
            final int[] valCSRIndices = ((SparseMatrix)Y).getValCSRIndices();
            final double[] pr = ((SparseMatrix)Y).getPr();
            for (int i = 0; i < Y.getRowDimension(); ++i) {
                double max = Double.NEGATIVE_INFINITY;
                labelIndices[i] = 0;
                for (int k = jr[i]; k < jr[i + 1]; ++k) {
                    if (max < pr[valCSRIndices[k]]) {
                        max = pr[valCSRIndices[k]];
                        labelIndices[i] = ic[k];
                    }
                }
            }
        }
        else {
            final double[][] YData = ((DenseMatrix)Y).getData();
            for (int j = 0; j < Y.getRowDimension(); ++j) {
                double max2 = Double.NEGATIVE_INFINITY;
                labelIndices[j] = 0;
                for (int l = 0; l < Y.getColumnDimension(); ++l) {
                    if (max2 < YData[j][l]) {
                        max2 = YData[j][l];
                        labelIndices[j] = l;
                    }
                }
            }
        }
        return labelIndices;
    }
    
    public static Matrix labelIndexArray2LabelMatrix(final int[] labelIndices, final int nClass) {
        final int[] rIndices = new int[labelIndices.length];
        final int[] cIndices = new int[labelIndices.length];
        final double[] values = new double[labelIndices.length];
        for (int i = 0; i < labelIndices.length; ++i) {
            cIndices[rIndices[i] = i] = labelIndices[i];
            values[i] = 1.0;
        }
        return new SparseMatrix(rIndices, cIndices, values, labelIndices.length, nClass, labelIndices.length);
    }
    
    public static Vector[] getVectors(final Matrix X) {
        final int l = X.getRowDimension();
        final int d = X.getColumnDimension();
        final Vector[] res = new Vector[l];
        if (X instanceof SparseMatrix) {
            final int[] ic = ((SparseMatrix)X).getIc();
            final int[] jr = ((SparseMatrix)X).getJr();
            final int[] valCSRIndices = ((SparseMatrix)X).getValCSRIndices();
            final double[] pr_X = ((SparseMatrix)X).getPr();
            int k = 0;
            int idx = 0;
            int nnz = 0;
            for (int i = 0; i < l; ++i) {
                nnz = jr[i + 1] - jr[i] + 1;
                final int[] ir = new int[nnz];
                final double[] pr = new double[nnz];
                for (k = jr[i], idx = 0; k < jr[i + 1]; ++k, ++idx) {
                    ir[idx] = ic[k];
                    pr[idx] = pr_X[valCSRIndices[k]];
                }
                ir[idx] = d;
                pr[idx] = 1.0;
                res[i] = new SparseVector(ir, pr, nnz, d + 1);
            }
        }
        else {
            final double[][] data = ((DenseMatrix)X).getData();
            final double[] dataVector = new double[d + 1];
            for (int j = 0; j < l; ++j) {
                System.arraycopy(data[j], 0, dataVector, 0, d);
                dataVector[d] = 1.0;
                res[j] = new DenseVector(dataVector);
            }
        }
        return res;
    }
    
    private static Matrix addBias(final Matrix X) {
        final int l = X.getRowDimension();
        final int d = X.getColumnDimension();
        final Vector[] XVs = new Vector[l];
        Matrix res = null;
        if (X instanceof SparseMatrix) {
            final int[] ic = ((SparseMatrix)X).getIc();
            final int[] jr = ((SparseMatrix)X).getJr();
            final int[] valCSRIndices = ((SparseMatrix)X).getValCSRIndices();
            final double[] pr_X = ((SparseMatrix)X).getPr();
            int k = 0;
            int idx = 0;
            int nnz = 0;
            for (int i = 0; i < l; ++i) {
                nnz = jr[i + 1] - jr[i] + 1;
                final int[] ir = new int[nnz];
                final double[] pr = new double[nnz];
                for (k = jr[i], idx = 0; k < jr[i + 1]; ++k, ++idx) {
                    ir[idx] = ic[k];
                    pr[idx] = pr_X[valCSRIndices[k]];
                }
                ir[idx] = d;
                pr[idx] = 1.0;
                XVs[i] = new SparseVector(ir, pr, nnz, d + 1);
            }
            res = Matlab.sparseRowVectors2SparseMatrix(XVs);
        }
        else {
            final double[][] data = ((DenseMatrix)X).getData();
            final double[] dataVector = new double[d + 1];
            final double[][] resData = new double[l][];
            for (int j = 0; j < l; ++j) {
                System.arraycopy(data[j], 0, dataVector, 0, d);
                dataVector[d] = 1.0;
                resData[j] = dataVector;
            }
            res = new DenseMatrix(resData);
        }
        return res;
    }
}
