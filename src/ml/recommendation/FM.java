package ml.recommendation;

import ml.recommendation.util.*;
import ml.recommendation.util.Utility;

import java.util.*;

import la.matrix.*;
import la.vector.*;

import java.io.*;

import la.io.*;
import ml.utils.*;
import ml.utils.IO;

public class FM
{
    public static String Method;
    private static String AppDirPath;
    private static int MaxIter;
    private static boolean calcOFV;
    private static int p;
    private static int K;
    private static int n;
    private static double b;
    private static DenseVector W;
    private static DenseVector[] V;
    private static int[] y;
    private static double[] y_hat;
    private static double[] e;
    private static double[] q;
    private static double[][] Q;
    private static Matrix X;
    private static double lambda;
    static TreeMap<Integer, Integer> TestIdx2TrainIdxUserMap;
    static TreeMap<Integer, Integer> TestIdx2TrainIdxItemMap;
    public static HashMap<Integer, LinkedList<Integer>> TestUser2EventIndexSetMap;
    public static int[] TestUserIndices;
    
    static {
        FM.Method = "FM";
        FM.AppDirPath = "";
        FM.MaxIter = 20;
        FM.calcOFV = false;
        FM.K = 8;
        FM.TestIdx2TrainIdxUserMap = new TreeMap<Integer, Integer>();
        FM.TestIdx2TrainIdxItemMap = new TreeMap<Integer, Integer>();
        FM.TestUser2EventIndexSetMap = new HashMap<Integer, LinkedList<Integer>>();
        FM.TestUserIndices = null;
    }
    
    public static void main(final String[] args) {
        final String appPath = FM.class.getProtectionDomain().getCodeSource().getLocation().getPath();
        FM.AppDirPath = new File(appPath).getParent();
        double lambda = 0.01;
        int maxIter = 50;
        FM.calcOFV = true;
        final int k = 8;
        String attribute = "";
        String value = "";
        for (int i = 0; i < args.length; ++i) {
            if (args[i].charAt(0) != '-') {
                System.err.println("Wrong options.");
                Utility.exit(1);
            }
            if (++i >= args.length) {
                Utility.exit(1);
            }
            attribute = args[i - 1];
            value = args[i];
            if (attribute.equals("-MaxIter")) {
                maxIter = Integer.parseInt(value);
            }
            else if (attribute.equals("-lambda")) {
                lambda = Double.parseDouble(value);
            }
        }
        System.out.println("Running FM...");
        Printer.fprintf("lambda = %f\n", lambda);
        final String trainFilePath = String.valueOf(FM.AppDirPath) + File.separator + "Train.libfm.txt";
        final String testFilePath = String.valueOf(FM.AppDirPath) + File.separator + "Test.libfm.txt";
        final String outputFilePath = String.valueOf(FM.AppDirPath) + File.separator + "FM-YijPredOnTest.txt";
        final int idxStart = 0;
        feedTrainingData(trainFilePath, idxStart);
        allocateResource(k);
        feedParams(maxIter, lambda);
        initialize();
        train();
        final DataSet testData = loadData(testFilePath, 0);
        final double[] Yij_pred = predict(testData.X);
        IO.save(Yij_pred, outputFilePath);
        Utility.loadMap(FM.TestIdx2TrainIdxUserMap, String.valueOf(FM.AppDirPath) + File.separator + "TestIdx2TrainIdxUserMap.txt");
        Utility.loadMap(FM.TestIdx2TrainIdxItemMap, String.valueOf(FM.AppDirPath) + File.separator + "TestIdx2TrainIdxItemMap.txt");
        FM.TestUserIndices = Utility.loadTestUserEventRelation(String.valueOf(FM.AppDirPath) + File.separator + "Test-Events.txt", FM.TestUser2EventIndexSetMap);
        double[] measures = null;
        measures = Utility.predict(testData.Y, Yij_pred, FM.TestUser2EventIndexSetMap);
        Utility.saveMeasures(FM.AppDirPath, Printer.sprintf("%s-Measures", FM.Method), measures);
        measures = Utility.predictColdStart(testData.Y, Yij_pred, FM.TestUserIndices, FM.TestUser2EventIndexSetMap, FM.TestIdx2TrainIdxUserMap, FM.TestIdx2TrainIdxItemMap);
        Utility.saveMeasures(FM.AppDirPath, Printer.sprintf("%s-ColdStart-Measures", FM.Method), measures);
        System.out.println("\nMission complete.");
    }
    
    public static void allocateResource(final int K) {
        FM.W = new DenseVector(FM.p, 0.0);
        FM.V = new DenseVector[K];
        for (int k = 0; k < K; ++k) {
            FM.V[k] = new DenseVector(FM.p, 0.0);
        }
        FM.e = ArrayOperator.allocate1DArray(FM.n, 0.0);
        FM.Q = ArrayOperator.allocate2DArray(K, FM.n);
        FM.y_hat = ArrayOperator.allocate1DArray(FM.n, 0.0);
    }
    
    public static void initialize() {
        FM.b = 0.0;
        FM.W.clear();
        final Random generator = new Random();
        final double sigma = 1.0E-4;
        for (int k = 0; k < FM.K; ++k) {
            final double[] pr = FM.V[k].getPr();
            for (int j = 1; j < FM.p; ++j) {
                pr[j] = generator.nextGaussian() * sigma;
            }
        }
        ArrayOperator.assign(FM.y_hat, 0.0);
        ArrayOperator.assign(FM.e, 0.0);
    }
    
    static void feedParams(final int MaxIter, final double lambda) {
        FM.MaxIter = MaxIter;
        FM.lambda = lambda * FM.n;
    }
    
    private static void predict(final String outputFilePath) {
        final int[] ic = ((SparseMatrix)FM.X).getIc();
        final int[] jr = ((SparseMatrix)FM.X).getJr();
        final double[] pr = ((SparseMatrix)FM.X).getPr();
        final int[] valCSRIndices = ((SparseMatrix)FM.X).getValCSRIndices();
        final double[] w = FM.W.getPr();
        for (int r = 0; r < FM.n; ++r) {
            double s = FM.b;
            for (int k = jr[r]; k < jr[r + 1]; ++k) {
                final int j = ic[k];
                s += w[j] * pr[valCSRIndices[k]];
            }
            double A = 0.0;
            double B = 0.0;
            for (int f = 0; f < FM.K; ++f) {
                final double[] v = FM.V[f].getPr();
                double a = 0.0;
                for (int i = jr[r]; i < jr[r + 1]; ++i) {
                    final int l = ic[i];
                    final double vj = v[l];
                    final double xj = pr[valCSRIndices[i]];
                    a += vj * xj;
                    B += vj * vj * xj * xj;
                }
                A += a * a;
            }
            s += (A - B) / 2.0;
            FM.y_hat[r] = s;
        }
        la.io.IO.saveVector(outputFilePath, new DenseVector(FM.y_hat));
    }
    
    static double[] predict(final Matrix X, final String outputFilePath) {
        final double[] y_pred = predict(X);
        la.io.IO.saveVector(outputFilePath, new DenseVector(y_pred));
        return y_pred;
    }
    
    static double[] predict(final Matrix X) {
        final int n = X.getRowDimension();
        final double[] y_hat = ArrayOperator.allocate1DArray(n);
        final int[] ic = ((SparseMatrix)X).getIc();
        final int[] jr = ((SparseMatrix)X).getJr();
        final double[] pr = ((SparseMatrix)X).getPr();
        final int[] valCSRIndices = ((SparseMatrix)X).getValCSRIndices();
        final double[] w = FM.W.getPr();
        for (int r = 0; r < n; ++r) {
            double s = FM.b;
            for (int k = jr[r]; k < jr[r + 1]; ++k) {
                final int j = ic[k];
                s += w[j] * pr[valCSRIndices[k]];
            }
            double A = 0.0;
            double B = 0.0;
            for (int f = 0; f < FM.K; ++f) {
                final double[] v = FM.V[f].getPr();
                double a = 0.0;
                for (int i = jr[r]; i < jr[r + 1]; ++i) {
                    final int l = ic[i];
                    final double vj = v[l];
                    final double xj = pr[valCSRIndices[i]];
                    a += vj * xj;
                    B += vj * vj * xj * xj;
                }
                A += a * a;
            }
            s += (A - B) / 2.0;
            y_hat[r] = s;
        }
        return y_hat;
    }
    
    static void feedTrainingData(final DataSet dataSet) {
        FM.X = dataSet.X;
        FM.y = dataSet.Y;
        FM.n = FM.X.getRowDimension();
        FM.p = FM.X.getColumnDimension();
    }
    
    static void feedTrainingData(final String trainFilePath) {
        feedTrainingData(trainFilePath, 0);
    }
    
    static void feedTrainingData(final String trainFilePath, final int idxStart) {
        feedTrainingData(loadData(trainFilePath, idxStart));
    }
    
    public static DataSet loadData(final String filePath, final int idxStart) {
        DataSet.IdxStart = idxStart;
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
        return dataSet;
    }
    
    static void train() {
        System.out.printf("Training %s...\n", FM.Method);
        double[] OFVs = null;
        final boolean debug = false;
        if (FM.calcOFV) {
            OFVs = ArrayOperator.allocate1DArray(FM.MaxIter + 1, 0.0);
            double ofv = 0.0;
            for (int i = 0; i < FM.n; ++i) {
                ofv += FM.y[i] * FM.y[i];
            }
            OFVs[0] = ofv;
            Printer.fprintf("Iter %d: %.10g\n", 0, ofv);
        }
        int cnt = 0;
        final double[] w = FM.W.getPr();
        double ofv_old = 0.0;
        double ofv_new = 0.0;
        final int[] ic = ((SparseMatrix)FM.X).getIc();
        final int[] ir = ((SparseMatrix)FM.X).getIr();
        final int[] jc = ((SparseMatrix)FM.X).getJc();
        final int[] jr = ((SparseMatrix)FM.X).getJr();
        final double[] pr = ((SparseMatrix)FM.X).getPr();
        final int[] valCSRIndices = ((SparseMatrix)FM.X).getValCSRIndices();
        for (int r = 0; r < FM.n; ++r) {
            double s = FM.b;
            for (int k = jr[r]; k < jr[r + 1]; ++k) {
                final int j = ic[k];
                s += w[j] * pr[valCSRIndices[k]];
            }
            double A = 0.0;
            double B = 0.0;
            for (int f = 0; f < FM.K; ++f) {
                final double[] v = FM.V[f].getPr();
                double a = 0.0;
                for (int l = jr[r]; l < jr[r + 1]; ++l) {
                    final int m = ic[l];
                    final double vj = v[m];
                    final double xj = pr[valCSRIndices[l]];
                    a += vj * xj;
                    B += vj * vj * xj * xj;
                }
                A += a * a;
            }
            s += (A - B) / 2.0;
            FM.y_hat[r] = s;
            FM.e[r] = FM.y[r] - s;
        }
        for (int f2 = 0; f2 < FM.K; ++f2) {
            final double[] v2 = FM.V[f2].getPr();
            FM.q = FM.Q[f2];
            for (int r2 = 0; r2 < FM.n; ++r2) {
                int s2 = 0;
                for (int k2 = jr[r2]; k2 < jr[r2 + 1]; ++k2) {
                    final int j2 = ic[k2];
                    s2 += (int)(v2[j2] * pr[valCSRIndices[k2]]);
                }
                FM.q[r2] = s2;
            }
        }
        do {
            ofv_old = 0.0;
            if (debug) {
                ofv_old = computOFV();
                Printer.printf("f(b): %f\n", ofv_old);
            }
            final double b_new = (FM.b * FM.n + ArrayOperator.sum(FM.e)) / (FM.n + FM.lambda);
            for (int i2 = 0; i2 < FM.n; ++i2) {
                final double[] e = FM.e;
                final int n = i2;
                e[n] -= b_new - FM.b;
            }
            FM.b = b_new;
            if (debug) {
                ofv_new = computOFV();
                Printer.printf("b updated: %f\n", ofv_new);
                if (ofv_old < ofv_new) {
                    Printer.errf("Error when updating b\n", new Object[0]);
                }
            }
            for (int j3 = 0; j3 < FM.p; ++j3) {
                ofv_old = 0.0;
                if (debug) {
                    ofv_old = computOFV();
                    Printer.printf("f(w[%d]): %f\n", j3, ofv_old);
                }
                double v3 = 0.0;
                double v4 = 0.0;
                for (int k3 = jc[j3]; k3 < jc[j3 + 1]; ++k3) {
                    final int i3 = ir[k3];
                    final double hj;
                    final double xj2 = hj = pr[k3];
                    v3 += hj * hj;
                    v4 += hj * FM.e[i3];
                }
                final double wj_new = (w[j3] * v3 + v4) / (v3 + FM.lambda);
                for (int k4 = jc[j3]; k4 < jc[j3 + 1]; ++k4) {
                    final int i4 = ir[k4];
                    final double xj3 = pr[k4];
                    final double[] e2 = FM.e;
                    final int n2 = i4;
                    e2[n2] -= (wj_new - w[j3]) * xj3;
                }
                w[j3] = wj_new;
                if (debug) {
                    ofv_new = computOFV();
                    Printer.printf("w[%d] updated: %f\n", j3, ofv_new);
                    if (ofv_old < ofv_new) {
                        Printer.errf("Error when updating w[%d]\n", j3);
                    }
                }
            }
            for (int f3 = 0; f3 < FM.K; ++f3) {
                final double[] v5 = FM.V[f3].getPr();
                FM.q = FM.Q[f3];
                for (int j = 0; j < FM.p; ++j) {
                    ofv_old = 0.0;
                    if (debug) {
                        ofv_old = computOFV();
                        Printer.printf("f(V[%d, %d]): %f\n", j, f3, ofv_old);
                    }
                    double v6 = 0.0;
                    double v7 = 0.0;
                    for (int k4 = jc[j]; k4 < jc[j + 1]; ++k4) {
                        final int i4 = ir[k4];
                        final double xj3 = pr[k4];
                        final double hj2 = xj3 * (FM.q[i4] - v5[j] * xj3);
                        v6 += hj2 * hj2;
                        v7 += hj2 * FM.e[i4];
                    }
                    final double vj_new = (v5[j] * v6 + v7) / (v6 + FM.lambda);
                    for (int l = jc[j]; l < jc[j + 1]; ++l) {
                        final int i5 = ir[l];
                        final double xj4 = pr[l];
                        final double hj3 = xj4 * (FM.q[i5] - v5[j] * xj4);
                        final double[] e3 = FM.e;
                        final int n3 = i5;
                        e3[n3] -= (vj_new - v5[j]) * hj3;
                    }
                    for (int l = jc[j]; l < jc[j + 1]; ++l) {
                        final int i5 = ir[l];
                        final double xj4 = pr[l];
                        final double[] q = FM.q;
                        final int n4 = i5;
                        q[n4] += (vj_new - v5[j]) * xj4;
                    }
                    v5[j] = vj_new;
                    if (debug) {
                        ofv_new = computOFV();
                        Printer.printf("V[%d, %d] updated: %f\n", j, f3, ofv_new);
                        if (ofv_old < ofv_new) {
                            Printer.errf("Error when updating V[%d,%d]\n", j, f3);
                        }
                    }
                }
            }
            ++cnt;
            if (FM.calcOFV) {
                double ofv2 = 0.0;
                ofv2 = computOFV();
                OFVs[cnt] = ofv2;
                if (cnt % 10 == 0) {
                    Printer.fprintf(".Iter %d: %.8g\n", cnt, ofv2);
                }
                else {
                    Printer.fprintf(".", new Object[0]);
                }
            }
            else if (cnt % 10 == 0) {
                Printer.fprintf(".Iter %d\n", cnt);
            }
            else {
                Printer.fprintf(".", new Object[0]);
            }
        } while (cnt < FM.MaxIter);
    }
    
    private static double computOFV() {
        double ofv = 0.0;
        ofv += FM.lambda * FM.b * FM.b;
        ofv += FM.lambda * Matlab.innerProduct(FM.W, FM.W);
        for (int f = 0; f < FM.K; ++f) {
            ofv += FM.lambda * Matlab.innerProduct(FM.V[f], FM.V[f]);
        }
        final int[] ic = ((SparseMatrix)FM.X).getIc();
        final int[] jr = ((SparseMatrix)FM.X).getJr();
        final double[] pr = ((SparseMatrix)FM.X).getPr();
        final int[] valCSRIndices = ((SparseMatrix)FM.X).getValCSRIndices();
        final double[] w = FM.W.getPr();
        for (int r = 0; r < FM.n; ++r) {
            double s = FM.b;
            for (int k = jr[r]; k < jr[r + 1]; ++k) {
                final int j = ic[k];
                s += w[j] * pr[valCSRIndices[k]];
            }
            double A = 0.0;
            double B = 0.0;
            for (int f2 = 0; f2 < FM.K; ++f2) {
                final double[] v = FM.V[f2].getPr();
                double a = 0.0;
                for (int i = jr[r]; i < jr[r + 1]; ++i) {
                    final int l = ic[i];
                    final double vj = v[l];
                    final double xj = pr[valCSRIndices[i]];
                    a += vj * xj;
                    B += vj * vj * xj * xj;
                }
                A += a * a;
            }
            s += (A - B) / 2.0;
            final double e = FM.y[r] - s;
            ofv += e * e;
        }
        return ofv;
    }
}
