package la.decomposition;

import la.matrix.*;
import java.util.*;
import ml.utils.*;

/**
 * A Java implementation for the eigenvalue decomposition of a symmetric real matrix.
 * <br><br>
 * The input matrix is first reduced to tridiagonal matrix and then is diagonalized by implicit symmetric shifted QR algorithm.
 * @version 1.0 Dec. 19th, 2013
 * @author Mingjie Qian
 */
public class EigenValueDecomposition
{
	/**
	 * Tolerance.
	 */
    public static double tol;
    
    /**
     * Maximum number of iterations.
     */
    public static int maxIter;
    
    /**
     * Eigenvectors.
     */
    private Matrix V;
    
    /**
     * A sparse diagonal matrix D with its diagonal being all eigenvalues in decreasing order (absolute value).
     */
    private Matrix D;
    
    static {
        EigenValueDecomposition.tol = 1.0E-16;
    }
    
    public static void main(final String[] args) {
        final int m = 4;
        final int n = 4;
        final Matrix A = Matlab.hilb(m, n);
        Printer.fprintf("A:\n", new Object[0]);
        Printer.disp(A);
        long start = 0L;
        start = System.currentTimeMillis();
        final Matrix[] VD = decompose(A);
        System.out.format("Elapsed time: %.4f seconds.\n", (System.currentTimeMillis() - start) / 1000.0);
        Printer.fprintf("*****************************************\n", new Object[0]);
        final Matrix V = VD[0];
        final Matrix D = VD[1];
        Printer.fprintf("V:\n", new Object[0]);
        Printer.printMatrix(V);
        Printer.fprintf("D:\n", new Object[0]);
        Printer.printMatrix(D);
        Printer.fprintf("VDV':\n", new Object[0]);
        Printer.disp(V.mtimes(D).mtimes(V.transpose()));
        Printer.fprintf("A:\n", new Object[0]);
        Printer.printMatrix(A);
        Printer.fprintf("V'V:\n", new Object[0]);
        Printer.printMatrix(V.transpose().mtimes(V));
    }
    
    public Matrix getV() {
        return this.V;
    }
    
    public Matrix getD() {
        return this.D;
    }
    
    /**
     * Construct this eigenvalue decomposition instance from a real symmetric matrix.
     * @param A a real symmetric matrix
     */
    public EigenValueDecomposition(final Matrix A) {
        final Matrix[] VD = decompose(A, true);
        this.V = VD[0];
        this.D = VD[1];
    }
    
    /**
     * Construct this eigenvalue decomposition instance from a real symmetric matrix.
     * @param A a real symmetric matrix
     * @param tol tolerance
     */
    public EigenValueDecomposition(final Matrix A, final double tol) {
        EigenValueDecomposition.tol = tol;
        final Matrix[] VD = decompose(A, true);
        this.V = VD[0];
        this.D = VD[1];
    }
    
    /**
     * Construct this eigenvalue decomposition instance from a real symmetric matrix.
     * @param A a real symmetric matrix
     * @param computeV if V is to be computed
     */
    public EigenValueDecomposition(final Matrix A, final boolean computeV) {
        final Matrix[] VD = decompose(A, computeV);
        this.V = VD[0];
        this.D = VD[1];
    }
    
    /**
     * Do eigenvalue decomposition for a real symmetric matrix, i.e. AV = VD.
     * @param A a real symmetric matrix
     * @return a Matrix array [V, D]
     */
    public static Matrix[] decompose(final Matrix A) {
        return decompose(A, true);
    }
    
    /**
     * Do eigenvalue decomposition for a real symmetric matrix, i.e. AV = VD.
     * @param A a real symmetric matrix
     * @param computeV if V is to be computed
     * @return a Matrix array [V, D]
     */
    public static Matrix[] decompose(final Matrix A, final boolean computeV) {
        final int m = A.getRowDimension();
        final int n = A.getColumnDimension();
        if (m != n) {
            System.err.println("Input should be a square matrix.");
            System.exit(1);
        }
        EigenValueDecomposition.maxIter = 30 * n * n;
        final Matrix[] QT = tridiagonalize(A, computeV);
        final Matrix Q = QT[0];
        final Matrix T = QT[1];
        final Matrix[] VD = diagonalizeTD(T, computeV);
        final Matrix V = VD[0];
        final Matrix D = VD[1];
        final Matrix[] res = { computeV ? Q.mtimes(V) : null, D };
        return res;
    }
    
    /**
     * Only eigenvalues of a symmetric real matrix are computed.
     * @param A a symmetric real matrix
     * @return a 1D double array containing the eigenvalues in decreasing order (absolute value)
     */
    public static double[] computeEigenvalues(final Matrix A) {
        final SparseMatrix S = (SparseMatrix)decompose(A, false)[1];
        final int m = S.getRowDimension();
        final int n = S.getColumnDimension();
        final int len = (m >= n) ? n : m;
        final double[] s = ArrayOperator.allocateVector(len, 0.0);
        for (int i = 0; i < len; ++i) {
            s[i] = S.getEntry(i, i);
        }
        return s;
    }
    
    /**
     * Tridiagonalize a real symmetric matrix A, i.e. A = Q * T * Q' such that Q is an orthogonal matrix and T is a tridiagonal matrix.
     * <br>
     * A = QTQ' &lt;=> Q'AQ = T
     * @param A a real symmetric matrix
     * @return a <code>Matrix</code> array [Q, T]
     */
    private static Matrix[] tridiagonalize(final Matrix A) {
        return tridiagonalize(A, true);
    }
    
    /**
     * Tridiagonalize a real symmetric matrix A, i.e. A = Q * T * Q' such that Q is an orthogonal matrix and T is a tridiagonal matrix.
     * <br>
     * A = QTQ' &lt;=> Q'AQ = T
     * @param A a real symmetric matrix
     * @param computeV if V is to be computed
     * @return a <code>Matrix</code> array [Q, T]
     */
    private static Matrix[] tridiagonalize(Matrix A, final boolean computeV) {
        A = Matlab.full(A).copy();
        final int m = A.getRowDimension();
        final int n = A.getColumnDimension();
        Matrix[] QT = new Matrix[2];
        final double[] a = ArrayOperator.allocateVector(n, 0.0);
        final double[] b = ArrayOperator.allocateVector(n, 0.0);
        final double[][] AData = ((DenseMatrix)A).getData();
        double c = 0.0;
        double s = 0.0;
        double r = 0.0;
        for (int j = 0; j < n - 2; ++j) {
            a[j] = AData[j][j];
            c = 0.0;
            for (int i = j + 1; i < m; ++i) {
                c += Math.pow(AData[i][j], 2.0);
            }
            if (c != 0.0) {
                s = Math.sqrt(c);
                b[j] = ((AData[j + 1][j] > 0.0) ? (-s) : s);
                r = Math.sqrt(s * (s + Math.abs(AData[j + 1][j])));
                final double[] array = AData[j + 1];
                final int n2 = j;
                array[n2] -= b[j];
                for (int k = j + 1; k < m; ++k) {
                    final double[] array2 = AData[k];
                    final int n3 = j;
                    array2[n3] /= r;
                }
                final double[] w = new double[n - j - 1];
                final double[] u = new double[n - j - 1];
                final double[] v = new double[n - j - 1];
                for (int l = j + 1, t = 0; l < m; ++l, ++t) {
                    u[t] = AData[l][j];
                }
                for (int l = j + 1, t = 0; l < m; ++l, ++t) {
                    final double[] ARow_i = AData[l];
                    s = 0.0;
                    for (int k2 = j + 1, l2 = 0; k2 < n; ++k2, ++l2) {
                        s += ARow_i[k2] * u[l2];
                    }
                    v[t] = s;
                }
                c = ArrayOperator.innerProduct(u, v) / 2.0;
                for (int l = j + 1, t = 0; l < m; ++l, ++t) {
                    w[t] = v[t] - c * u[t];
                }
                for (int l = j + 1, t = 0; l < m; ++l, ++t) {
                    final double[] ARow_i = AData[l];
                    for (int k2 = j + 1, l2 = 0; k2 < n; ++k2, ++l2) {
                        ARow_i[k2] -= u[t] * w[l2] + w[t] * u[l2];
                    }
                }
            }
        }
        a[n - 2] = AData[n - 2][n - 2];
        a[n - 1] = AData[n - 1][n - 1];
        b[n - 2] = AData[n - 1][n - 2];
        QT = unpack(A, a, b, computeV);
        return QT;
    }
    
    /**
     * Unpack Q and T from the result of tridiagonalization.
     * @param A tridiagonalization result
     * @param a diagonal
     * @param b superdiagonal
     * @param computeV if V is to be computed
     * @return a <code>Matrix</code> array [Q, T]
     */
    private static Matrix[] unpack(final Matrix A, final double[] a, final double[] b, final boolean computeV) {
        final Matrix[] QT = new Matrix[3];
        final int m = A.getRowDimension();
        final int n = A.getColumnDimension();
        DenseMatrix Q = null;
        if (computeV) {
            Q = new DenseMatrix(m, m, 0.0);
            final double[][] QData = Q.getData();
            double s = 0.0;
            double[] y = null;
            for (int i = 0; i < m; ++i) {
                y = QData[i];
                y[i] = 1.0;
                for (int j = 0; j < n - 2; ++j) {
                    s = 0.0;
                    for (int k = j + 1; k < m; ++k) {
                        s += A.getEntry(k, j) * y[k];
                    }
                    for (int k = j + 1; k < m; ++k) {
                        final double[] array = y;
                        final int n2 = k;
                        array[n2] -= A.getEntry(k, j) * s;
                    }
                }
            }
        }
        final TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();
        for (int l = 0; l < m; ++l) {
            if (l < n) {
                map.put(Pair.of(l, l), a[l]);
            }
            if (l < n - 1) {
                map.put(Pair.of(l, l + 1), b[l]);
                map.put(Pair.of(l + 1, l), b[l]);
            }
        }
        final Matrix T = SparseMatrix.createSparseMatrix(map, m, n);
        QT[0] = Q;
        QT[1] = T;
        return QT;
    }
    
    /**
     * Do eigenvalue decomposition for a real symmetric tridiagonal matrix, i.e. T = VDV'.
     * @param T a real symmetric tridiagonal matrix
     * @return a <code>Matrix</code> array [V, D]
     */
    private static Matrix[] diagonalizeTD(final Matrix T) {
        return diagonalizeTD(T, true);
    }
    
    /**
     * Do eigenvalue decomposition for a real symmetric tridiagonal matrix, i.e. T = VDV'.
     * @param T a real symmetric tridiagonal matrix
     * @param computeV if V is to be computed
     * @return a <code>Matrix</code> array [V, D]
     */
    private static Matrix[] diagonalizeTD(final Matrix T, final boolean computeV) {
        final int m = T.getRowDimension();
        final int n = T.getColumnDimension();
        final int len = (m >= n) ? n : m;
        int idx = 0;
        final double[] s = ArrayOperator.allocateVector(len, 0.0);
        final double[] e = ArrayOperator.allocateVector(len, 0.0);
        for (int i = 0; i < len - 1; ++i) {
            s[i] = T.getEntry(i, i);
            e[i] = T.getEntry(i, i + 1);
        }
        s[len - 1] = T.getEntry(len - 1, len - 1);
        double[][] Vt = null;
        if (computeV) {
            Vt = Matlab.eye(n, n).getData();
        }
        final double[] mu = ArrayOperator.allocate1DArray(len, 0.0);
        int i_start = 0;
        int i_end = len - 1;
        int ind = 1;
        while (true) {
            for (idx = len - 2; idx >= 0 && e[idx] == 0.0; --idx) {}
            i_end = idx + 1;
            while (idx >= 0 && e[idx] != 0.0) {
                --idx;
            }
            i_start = idx + 1;
            if (i_start == i_end) {
                break;
            }
            boolean set2Zero = false;
            mu[i_start] = Math.abs(s[i_start]);
            for (int j = i_start; j < i_end; ++j) {
                mu[j + 1] = Math.abs(s[j + 1]) * mu[j] / (mu[j] + Math.abs(e[j]));
                if (Math.abs(e[j]) <= mu[j] * EigenValueDecomposition.tol) {
                    e[j] = 0.0;
                    set2Zero = true;
                }
            }
            if (set2Zero) {
                continue;
            }
            implicitSymmetricShiftedQR(s, e, Vt, i_start, i_end, computeV);
            if (ind == EigenValueDecomposition.maxIter) {
                break;
            }
            ++ind;
        }
        quickSort(s, Vt, 0, len - 1, "descend", computeV);
        final Matrix[] VD = { computeV ? new DenseMatrix(Vt).transpose() : null, buildD(s, m, n) };
        return VD;
    }
    
    /**
     * Sort the eigenvalues in a specified order. If computeV is true, eigenvectors will also be sorted.
     * @param s a 1D double array containing the eigenvalues
     * @param Vt eigenvectors
     * @param start start index (inclusive)
     * @param end end index (inclusive)
     * @param order a String either "descend" or "ascend"
     * @param computeV if V is to be computed
     */
    private static void quickSort(final double[] s, final double[][] Vt, final int start, final int end, final String order, final boolean computeV) {
        int i = start;
        int j = end;
        final double temp = s[i];
        final double[] tempV = (double[])(computeV ? Vt[i] : null);
        do {
            if (order.equals("ascend")) {
                while (Math.abs(s[j]) > Math.abs(temp)) {
                    if (j <= i) {
                        break;
                    }
                    --j;
                }
            }
            else if (order.equals("descend")) {
                while (Math.abs(s[j]) < Math.abs(temp) && j > i) {
                    --j;
                }
            }
            if (j > i) {
                s[i] = s[j];
                if (computeV) {
                    Vt[i] = Vt[j];
                }
                ++i;
            }
            if (order.equals("ascend")) {
                while (Math.abs(s[i]) < Math.abs(temp)) {
                    if (j <= i) {
                        break;
                    }
                    ++i;
                }
            }
            else if (order.equals("descend")) {
                while (Math.abs(s[i]) > Math.abs(temp) && j > i) {
                    ++i;
                }
            }
            if (j > i) {
                s[j] = s[i];
                if (computeV) {
                    Vt[j] = Vt[i];
                }
                --j;
            }
        } while (i != j);
        s[i] = temp;
        if (computeV) {
            Vt[i] = tempV;
        }
        ++i;
        --j;
        if (start < j) {
            quickSort(s, Vt, start, j, order, computeV);
        }
        if (i < end) {
            quickSort(s, Vt, i, end, order, computeV);
        }
    }
    
    /**
     * Build the diagonal matrix containing all eigenvalues.
     * @param s eigenvalues
     * @param m number of rows
     * @param n number of columns
     * @return a diagonal matrix containing all eigenvalues
     */
    private static Matrix buildD(final double[] s, final int m, final int n) {
        final TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();
        for (int i = 0; i < m; ++i) {
            if (i < n) {
                map.put(Pair.of(i, i), s[i]);
            }
        }
        return SparseMatrix.createSparseMatrix(map, m, n);
    }
    
    /**
     * Implicit symmetric shifted QR algorithm on B_hat which is the bottommost unreduced submatrix of B begin from i_start (inclusive) to i_end (inclusive).
     * @param s diagonal elements
     * @param e superdiagonal elements
     * @param Vt transposition of eigenvector matrix
     * @param i_start start index of B_hat (inclusive)
     * @param i_end end index of B_hat (inclusive)
     * @param computeV if V is to be computed
     */
    private static void implicitSymmetricShiftedQR(final double[] s, final double[] e, final double[][] Vt, final int i_start, final int i_end, final boolean computeV) {
        double d = 0.0;
        d = (s[i_end - 1] - s[i_end]) / 2.0;
        final double c = e[i_end - 1] * e[i_end - 1];
        double shift = Math.sqrt(d * d + c);
        shift = ((d > 0.0) ? shift : (-shift));
        shift = c / (d + shift);
        double f = s[i_start] - s[i_end] + shift;
        double g = e[i_start];
        double cs = 0.0;
        double sn = 0.0;
        double r = 0.0;
        double h = 0.0;
        for (int i = i_start; i < i_end; ++i) {
            if (f == 0.0) {
                cs = 0.0;
                sn = 1.0;
                r = g;
            }
            else if (Math.abs(f) > Math.abs(g)) {
                final double t = g / f;
                final double tt = Math.sqrt(1.0 + t * t);
                cs = 1.0 / tt;
                sn = t * cs;
                r = f * tt;
            }
            else {
                final double t = f / g;
                final double tt = Math.sqrt(1.0 + t * t);
                sn = 1.0 / tt;
                cs = t * sn;
                r = g * tt;
            }
            if (computeV) {
                update(cs, sn, Vt[i], Vt[i + 1]);
            }
            if (i != i_start) {
                e[i - 1] = r;
            }
            f = cs * s[i] + sn * e[i];
            h = -sn * s[i] + cs * e[i];
            g = cs * e[i] + sn * s[i + 1];
            s[i + 1] = -sn * e[i] + cs * s[i + 1];
            r = cs * f + sn * g;
            s[i] = r;
            e[i] = -sn * f + cs * g;
            s[i + 1] = -sn * h + cs * s[i + 1];
            h = sn * e[i + 1];
            final int n = i + 1;
            e[n] *= cs;
            if (i < i_end - 1) {
                f = e[i];
                g = h;
            }
        }
    }
    
    /**
     * Update two 1D double arrays V1 and V2 by Givens rotation parameterized by cs and sn, i.e. [V1 V2] * |cs -sn| or |cs sn| * |V1'| |sn cs| |-sn cs| |V2'|
     * @param cs cos(theta)
     * @param sn sin(theta)
     * @param V1 a 1D double arrays
     * @param V2 a 1D double arrays
     */
    private static void update(final double cs, final double sn, final double[] V1, final double[] V2) {
        for (int i = 0; i < V1.length; ++i) {
            final double temp = V1[i];
            V1[i] = cs * temp + sn * V2[i];
            V2[i] = -sn * temp + cs * V2[i];
        }
    }
}
