package la.decomposition;

import la.matrix.*;
import la.vector.*;
import la.vector.Vector;

import java.util.*;

import ml.utils.*;

public class QRDecomposition
{
    private Matrix Q;
    private Matrix R;
    private Matrix P;
    
    public static void main(final String[] args) {
        final int m = 4;
        final int n = 3;
        Matrix A = Matlab.hilb(m, n);
        Printer.fprintf("When A is full:\n", new Object[0]);
        Printer.fprintf("A:\n", new Object[0]);
        Printer.printMatrix(A);
        long start = 0L;
        start = System.currentTimeMillis();
        Matrix[] QRP = decompose(A);
        Matrix Q = QRP[0];
        Matrix R = QRP[1];
        Matrix P = QRP[2];
        Printer.fprintf("Q:\n", new Object[0]);
        Printer.printMatrix(Q);
        Printer.fprintf("R:\n", new Object[0]);
        Printer.printMatrix(R);
        Printer.fprintf("P:\n", new Object[0]);
        Printer.printMatrix(P);
        Printer.fprintf("AP:\n", new Object[0]);
        Printer.printMatrix(A.mtimes(P));
        Printer.fprintf("QR:\n", new Object[0]);
        Printer.printMatrix(Q.mtimes(R));
        Printer.fprintf("Q'Q:\n", new Object[0]);
        Printer.printMatrix(Q.transpose().mtimes(Q));
        Printer.fprintf("Elapsed time: %.2f seconds.\n", (System.currentTimeMillis() - start) / 1000.0f);
        Printer.fprintf("**********************************\n", new Object[0]);
        A = Matlab.sparse(Matlab.hilb(m, n));
        Printer.fprintf("When A is sparse:\n", new Object[0]);
        Printer.fprintf("A:\n", new Object[0]);
        Printer.printMatrix(A);
        start = System.currentTimeMillis();
        QRP = decompose(A);
        Q = QRP[0];
        R = QRP[1];
        P = QRP[2];
        Printer.fprintf("Q:\n", new Object[0]);
        Printer.printMatrix(Q);
        Printer.fprintf("R:\n", new Object[0]);
        Printer.printMatrix(R);
        Printer.fprintf("P:\n", new Object[0]);
        Printer.printMatrix(P);
        Printer.fprintf("AP:\n", new Object[0]);
        Printer.printMatrix(A.mtimes(P));
        Printer.fprintf("QR:\n", new Object[0]);
        Printer.printMatrix(Q.mtimes(R));
        Printer.fprintf("Q'Q:\n", new Object[0]);
        Printer.printMatrix(Q.transpose().mtimes(Q));
        Printer.fprintf("Elapsed time: %.2f seconds.\n", (System.currentTimeMillis() - start) / 1000.0f);
        final QRDecomposition QRDecomp = new QRDecomposition(A);
        final Vector b = new DenseVector(new double[] { 2.0, 3.0, 4.0, 9.0 });
        final Vector x = QRDecomp.solve(b);
        Printer.fprintf("Solution for Ax = b:\n", new Object[0]);
        Printer.printVector(x);
        Printer.fprintf("b = \n", new Object[0]);
        Printer.printVector(b);
        Printer.fprintf("Ax = \n", new Object[0]);
        Printer.printVector(A.operate(x));
    }
    
    public Matrix getQ() {
        return this.Q;
    }
    
    public Matrix getR() {
        return this.R;
    }
    
    public Matrix getP() {
        return this.P;
    }
    
    public QRDecomposition(final Matrix A) {
        final Matrix[] QRP = this.run(A);
        this.Q = QRP[0];
        this.R = QRP[1];
        this.P = QRP[2];
    }
    
    public QRDecomposition(final double[][] A) {
        final Matrix[] QRP = this.run(new DenseMatrix(A));
        this.Q = QRP[0];
        this.R = QRP[1];
        this.P = QRP[2];
    }
    
    private Matrix[] run(final Matrix A) {
        return decompose(A);
    }
    
    public Vector solve(final double[] b) {
        return this.solve(new DenseVector(b));
    }
    
    public Vector solve(final Vector b) {
        final double[] d = Matlab.full(this.Q.transpose().operate(b)).getPr();
        int rank = 0;
        final int m = this.R.getRowDimension();
        final int n = this.R.getColumnDimension();
        for (int len = Math.min(m, n), i = 0; i < len; ++i) {
            if (this.R.getEntry(i, i) == 0.0) {
                rank = i;
                break;
            }
            ++rank;
        }
        final double[] y = ArrayOperator.allocate1DArray(n, 0.0);
        if (this.R instanceof DenseMatrix) {
            final double[][] RData = ((DenseMatrix)this.R).getData();
            double[] RData_i = null;
            double v = 0.0;
            for (int j = rank - 1; j > -1; --j) {
                RData_i = RData[j];
                v = d[j];
                for (int k = n - 1; k > j; --k) {
                    v -= RData_i[k] * y[k];
                }
                y[j] = v / RData_i[j];
            }
        }
        else if (this.R instanceof SparseMatrix) {
            final Vector[] RVs = Matlab.sparseMatrix2SparseRowVectors(this.R);
            Vector RRow_i = null;
            double v = 0.0;
            for (int j = rank - 1; j > -1; --j) {
                RRow_i = RVs[j];
                v = d[j];
                final int[] ir = ((SparseVector)RRow_i).getIr();
                final double[] pr = ((SparseVector)RRow_i).getPr();
                final int nnz = ((SparseVector)RRow_i).getNNZ();
                int idx = -1;
                int l = nnz - 1;
                while (true) {
                    idx = ir[l];
                    if (idx <= j) {
                        break;
                    }
                    v -= pr[l] * y[idx];
                    --l;
                }
                y[j] = v / RRow_i.get(j);
            }
        }
        final Vector x = this.P.operate(new DenseVector(y));
        return x;
    }
    
    public Matrix solve(final double[][] B) {
        return this.solve(new DenseMatrix(B));
    }
    
    public Matrix solve(final Matrix B) {
        final double[][] D = Matlab.full(this.Q.transpose().mtimes(B)).getData();
        double[] DRow_i = null;
        int rank = 0;
        final int m = this.R.getRowDimension();
        final int n = this.R.getColumnDimension();
        for (int len = Math.min(m, n), i = 0; i < len; ++i) {
            if (this.R.getEntry(i, i) == 0.0) {
                rank = i;
                break;
            }
            ++rank;
        }
        final double[][] Y = ArrayOperator.allocate2DArray(n, B.getColumnDimension(), 0.0);
        double[] YRow_i = null;
        if (this.R instanceof DenseMatrix) {
            final double[][] RData = ((DenseMatrix)this.R).getData();
            double[] RData_i = null;
            double v = 0.0;
            for (int j = rank - 1; j > -1; --j) {
                RData_i = RData[j];
                DRow_i = D[j];
                YRow_i = Y[j];
                for (int k = 0; k < B.getColumnDimension(); ++k) {
                    v = DRow_i[k];
                    for (int l = n - 1; l > j; --l) {
                        v -= RData_i[l] * Y[l][k];
                    }
                    YRow_i[k] = v / RData_i[j];
                }
            }
        }
        else if (this.R instanceof SparseMatrix) {
            final Vector[] RVs = Matlab.sparseMatrix2SparseRowVectors(this.R);
            Vector RRow_i = null;
            double v = 0.0;
            for (int j = rank - 1; j > -1; --j) {
                RRow_i = RVs[j];
                DRow_i = D[j];
                YRow_i = Y[j];
                for (int k = 0; k < B.getColumnDimension(); ++k) {
                    v = DRow_i[k];
                    final int[] ir = ((SparseVector)RRow_i).getIr();
                    final double[] pr = ((SparseVector)RRow_i).getPr();
                    final int nnz = ((SparseVector)RRow_i).getNNZ();
                    int idx = -1;
                    int k2 = nnz - 1;
                    while (true) {
                        idx = ir[k2];
                        if (idx <= j) {
                            break;
                        }
                        v -= pr[k2] * Y[idx][k];
                        --k2;
                    }
                    YRow_i[k] = v / RRow_i.get(j);
                }
            }
        }
        final Matrix X = this.P.mtimes(new DenseMatrix(Y));
        return X;
    }
    
    public static Matrix[] decompose(Matrix A) {
        A = A.copy();
        final int m = A.getRowDimension();
        final int n = A.getColumnDimension();
        final Matrix[] QRP = new Matrix[3];
        final double[] d = ArrayOperator.allocateVector(n, 0.0);
        final Vector[] PVs = Matlab.sparseMatrix2SparseColumnVectors(new SparseMatrix(n, n));
        for (int i = 0; i < n; ++i) {
            PVs[i].set(i, 1.0);
        }
        if (A instanceof DenseMatrix) {
            final double[][] AData = ((DenseMatrix)A).getData();
            final double[] c = ArrayOperator.allocateVector(n, 0.0);
            for (int j = 0; j < n; ++j) {
                if (j >= m) {
                    break;
                }
                for (int jj = j; jj < n; ++jj) {
                    double s = 0.0;
                    for (int k = j; k < m; ++k) {
                        s += Math.pow(AData[k][jj], 2.0);
                    }
                    c[jj] = s;
                }
                int l = j;
                double maxVal = c[j];
                for (int k2 = j + 1; k2 < n; ++k2) {
                    if (maxVal < c[k2]) {
                        l = k2;
                        maxVal = c[k2];
                    }
                }
                if (maxVal == 0.0) {
                    System.out.println("Rank(A) < n.");
                    QRP[0] = computeQ(A);
                    QRP[1] = computeR(A, d);
                    QRP[2] = Matlab.sparseRowVectors2SparseMatrix(PVs);
                    return QRP;
                }
                if (l != j) {
                    double temp = 0.0;
                    for (int k3 = 0; k3 < m; ++k3) {
                        temp = AData[k3][l];
                        AData[k3][l] = AData[k3][j];
                        AData[k3][j] = temp;
                    }
                    temp = c[l];
                    c[l] = c[j];
                    c[j] = temp;
                    final Vector V = PVs[l];
                    PVs[l] = PVs[j];
                    PVs[j] = V;
                }
                double s2 = Math.sqrt(c[j]);
                d[j] = ((AData[j][j] > 0.0) ? (-s2) : s2);
                final double r = Math.sqrt(s2 * (s2 + Math.abs(AData[j][j])));
                final double[] array = AData[j];
                final int n2 = j;
                array[n2] -= d[j];
                for (int k4 = j; k4 < m; ++k4) {
                    final double[] array2 = AData[k4];
                    final int n3 = j;
                    array2[n3] /= r;
                }
                for (int k4 = j + 1; k4 < n; ++k4) {
                    s2 = 0.0;
                    for (int t = j; t < m; ++t) {
                        s2 += AData[t][j] * AData[t][k4];
                    }
                    for (int t = j; t < m; ++t) {
                        final double[] array3 = AData[t];
                        final int n4 = k4;
                        array3[n4] -= s2 * AData[t][j];
                    }
                }
            }
        }
        else if (A instanceof SparseMatrix) {
            final Vector[] AVs = Matlab.sparseMatrix2SparseColumnVectors(A);
            final double[] c = ArrayOperator.allocateVector(n, 0.0);
            for (int j = 0; j < n && j < m; ++j) {
                for (int jj = j; jj < n; ++jj) {
                    final SparseVector A_j = (SparseVector)AVs[jj];
                    final double[] pr = A_j.getPr();
                    final int[] ir = A_j.getIr();
                    final int nnz = A_j.getNNZ();
                    double s3 = 0.0;
                    int idx = -1;
                    for (int k5 = 0; k5 < nnz; ++k5) {
                        idx = ir[k5];
                        if (idx >= j) {
                            s3 += Math.pow(pr[k5], 2.0);
                        }
                    }
                    c[jj] = s3;
                }
                int l = j;
                double maxVal = c[j];
                for (int k2 = j + 1; k2 < n; ++k2) {
                    if (maxVal < c[k2]) {
                        l = k2;
                        maxVal = c[k2];
                    }
                }
                if (maxVal == 0.0) {
                    System.out.println("Rank(A) < n.");
                    QRP[0] = computeQ(A);
                    QRP[1] = computeR(A, d);
                    QRP[2] = Matlab.sparseRowVectors2SparseMatrix(PVs);
                    return QRP;
                }
                if (l != j) {
                    double temp = 0.0;
                    Vector V = null;
                    V = AVs[l];
                    AVs[l] = AVs[j];
                    AVs[j] = V;
                    temp = c[l];
                    c[l] = c[j];
                    c[j] = temp;
                    V = PVs[l];
                    PVs[l] = PVs[j];
                    PVs[j] = V;
                }
                double s2 = Math.sqrt(c[j]);
                final SparseVector A_j2 = (SparseVector)AVs[j];
                final double Ajj = A_j2.get(j);
                d[j] = ((Ajj > 0.0) ? (-s2) : s2);
                final double r2 = Math.sqrt(s2 * (s2 + Math.abs(Ajj)));
                A_j2.set(j, Ajj - d[j]);
                final int[] ir2 = A_j2.getIr();
                final double[] pr2 = A_j2.getPr();
                final int nnz2 = A_j2.getNNZ();
                int idx2 = 0;
                for (int k6 = 0; k6 < nnz2; ++k6) {
                    idx2 = ir2[k6];
                    if (idx2 >= j) {
                        final double[] array4 = pr2;
                        final int n5 = k6;
                        array4[n5] /= r2;
                    }
                }
                for (int k6 = j + 1; k6 < n; ++k6) {
                    final SparseVector A_k = (SparseVector)AVs[k6];
                    s2 = 0.0;
                    final int[] ir3 = A_k.getIr();
                    final double[] pr3 = A_k.getPr();
                    final int nnz3 = A_k.getNNZ();
                    if (nnz2 != 0 && nnz3 != 0) {
                        int k7 = 0;
                        int k8 = 0;
                        int r3 = 0;
                        int r4 = 0;
                        double v = 0.0;
                        while (k7 < nnz2 && k8 < nnz3) {
                            r3 = ir2[k7];
                            r4 = ir3[k8];
                            if (r3 < r4) {
                                ++k7;
                            }
                            else if (r3 == r4) {
                                v = pr2[k7] * pr3[k8];
                                ++k7;
                                ++k8;
                                if (r3 < j) {
                                    continue;
                                }
                                s2 += v;
                            }
                            else {
                                ++k8;
                            }
                        }
                    }
                    for (int t2 = j; t2 < m; ++t2) {
                        A_k.set(t2, A_k.get(t2) - s2 * A_j2.get(t2));
                    }
                }
            }
            A = Matlab.sparseColumnVectors2SparseMatrix(AVs);
        }
        QRP[0] = computeQ(A);
        QRP[1] = computeR(A, d);
        QRP[2] = Matlab.sparseColumnVectors2SparseMatrix(PVs);
        return QRP;
    }
    
    private static Matrix computeQ(final Matrix A) {
        final int m = A.getRowDimension();
        final int n = A.getColumnDimension();
        final double[][] Q = new DenseMatrix(m, m, 0.0).getData();
        double s = 0.0;
        double[] y = null;
        for (int i = 0; i < m; ++i) {
            y = Q[i];
            y[i] = 1.0;
            for (int j = 0; j < n; ++j) {
                s = 0.0;
                for (int k = j; k < m; ++k) {
                    s += A.getEntry(k, j) * y[k];
                }
                for (int k = j; k < m; ++k) {
                    final double[] array = y;
                    final int n2 = k;
                    array[n2] -= A.getEntry(k, j) * s;
                }
            }
        }
        return new DenseMatrix(Q);
    }
    
    private static Matrix computeR(final Matrix A, final double[] d) {
        final int m = A.getRowDimension();
        final int n = A.getColumnDimension();
        Matrix R = null;
        if (A instanceof DenseMatrix) {
            final double[][] AData = ((DenseMatrix)A).getData();
            for (int i = 0; i < m; ++i) {
                final double[] A_i = AData[i];
                if (i < n) {
                    A_i[i] = d[i];
                }
                for (int len = Math.min(i, n), j = 0; j < len; ++j) {
                    A_i[j] = 0.0;
                }
            }
            R = A;
        }
        else if (A instanceof SparseMatrix) {
            final TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();
            for (int i = 0; i < m; ++i) {
                if (i < n) {
                    map.put(Pair.of(i, i), d[i]);
                }
                for (int k = i + 1; k < n; ++k) {
                    map.put(Pair.of(i, k), A.getEntry(i, k));
                }
            }
            R = SparseMatrix.createSparseMatrix(map, m, n);
        }
        return R;
    }
}
