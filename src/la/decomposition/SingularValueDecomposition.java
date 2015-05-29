package la.decomposition;

import la.matrix.*;
import java.util.*;
import ml.utils.*;

public class SingularValueDecomposition
{
    public static double tol;
    public static int maxIter;
    private Matrix U;
    private Matrix S;
    private Matrix V;
    
    static {
        SingularValueDecomposition.tol = 1.0E-12;
    }
    
    public static void main(final String[] args) {
        final int m = 6;
        final int n = 4;
        final Matrix A = Matlab.hilb(m, n);
        Printer.disp("A:");
        Printer.printMatrix(A);
        long start = 0L;
        start = System.currentTimeMillis();
        final boolean computeUV = true;
        final Matrix[] USV = decompose(A, computeUV);
        System.out.format("Elapsed time: %.4f seconds.\n", (System.currentTimeMillis() - start) / 1000.0);
        Printer.fprintf("*****************************************\n", new Object[0]);
        final Matrix U = USV[0];
        final Matrix S = USV[1];
        final Matrix V = USV[2];
        if (computeUV) {
            Printer.fprintf("USV':\n", new Object[0]);
            Printer.disp(U.mtimes(S).mtimes(V.transpose()));
            Printer.fprintf("A:\n", new Object[0]);
            Printer.printMatrix(A);
            Printer.fprintf("U'U:\n", new Object[0]);
            Printer.printMatrix(U.transpose().mtimes(U));
            Printer.fprintf("V'V:\n", new Object[0]);
            Printer.printMatrix(V.transpose().mtimes(V));
            Printer.fprintf("U:\n", new Object[0]);
            Printer.printMatrix(U);
            Printer.fprintf("V:\n", new Object[0]);
            Printer.printMatrix(V);
        }
        Printer.fprintf("S:\n", new Object[0]);
        Printer.printMatrix(S);
        Printer.fprintf("rank(A): %d\n", rank(A));
    }
    
    public Matrix getU() {
        return this.U;
    }
    
    public Matrix getS() {
        return this.S;
    }
    
    public Matrix getV() {
        return this.V;
    }
    
    public SingularValueDecomposition(final Matrix A) {
        final Matrix[] USV = decompose(A, true);
        this.U = USV[0];
        this.S = USV[1];
        this.V = USV[2];
    }
    
    public SingularValueDecomposition(final Matrix A, final boolean computeUV) {
        final Matrix[] USV = decompose(A, computeUV);
        this.U = USV[0];
        this.S = USV[1];
        this.V = USV[2];
    }
    
    public static Matrix[] decompose(final Matrix A) {
        return decompose(A, true);
    }
    
    public static Matrix[] decompose(final Matrix A, final boolean computeUV) {
        final int n = A.getColumnDimension();
        SingularValueDecomposition.maxIter = 3 * n * n;
        final Matrix[] UBV = bidiagonalize(A, computeUV);
        final Matrix B = UBV[1];
        final Matrix[] USV = diagonalizeBD(B, computeUV);
        final Matrix[] res = new Matrix[3];
        if (computeUV) {
            res[0] = UBV[0].mtimes(USV[0]);
        }
        else {
            res[0] = null;
        }
        res[1] = USV[1];
        if (computeUV) {
            res[2] = UBV[2].mtimes(USV[2]);
        }
        else {
            res[2] = null;
        }
        return res;
    }
    
    public static double[] computeSingularValues(final Matrix A) {
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
    
    public static int rank(final Matrix A) {
        int r = 0;
        final double[] s = computeSingularValues(A);
        final int m = A.getRowDimension();
        final int n = A.getColumnDimension();
        final double t = (m >= n) ? (m * Math.pow(2.0, -52.0)) : (n * Math.pow(2.0, -52.0));
        for (int i = 0; i < s.length; ++i) {
            if (s[i] > t) {
                ++r;
            }
        }
        return r;
    }
    
    private static Matrix[] diagonalizeBD(final Matrix B) {
        return diagonalizeBD(B, true);
    }
    
    private static Matrix[] diagonalizeBD(final Matrix B, final boolean computeUV) {
        final int m = B.getRowDimension();
        final int n = B.getColumnDimension();
        final int len = (m >= n) ? n : m;
        int idx = 0;
        final double[] s = ArrayOperator.allocateVector(len, 0.0);
        final double[] e = ArrayOperator.allocateVector(len, 0.0);
        for (int i = 0; i < len - 1; ++i) {
            s[i] = B.getEntry(i, i);
            e[i] = B.getEntry(i, i + 1);
        }
        s[len - 1] = B.getEntry(len - 1, len - 1);
        double[][] Ut = null;
        if (computeUV) {
            Ut = Matlab.eye(m, m).getData();
        }
        double[][] Vt = null;
        if (computeUV) {
            Vt = Matlab.eye(n, n).getData();
        }
        final double[] mu = ArrayOperator.allocate1DArray(len, 0.0);
        double sigma_min = 0.0;
        double sigma_max = 0.0;
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
                if (Math.abs(e[j]) <= mu[j] * SingularValueDecomposition.tol) {
                    e[j] = 0.0;
                    set2Zero = true;
                }
            }
            if (set2Zero) {
                continue;
            }
            sigma_min = Math.abs(mu[i_start]);
            for (int j = i_start; j <= i_end; ++j) {
                if (sigma_min > Math.abs(mu[j])) {
                    sigma_min = Math.abs(mu[j]);
                }
            }
            sigma_max = Math.abs(s[i_start]);
            for (int j = i_start; j <= i_end; ++j) {
                if (sigma_max < Math.abs(s[j])) {
                    sigma_max = Math.abs(s[j]);
                }
            }
            for (int j = i_start; j < i_end; ++j) {
                if (sigma_max < Math.abs(e[j])) {
                    sigma_max = Math.abs(e[j]);
                }
            }
            if (n * sigma_max < sigma_min * Math.max(Double.MIN_VALUE / SingularValueDecomposition.tol, 0.01)) {
                implicitZeroShiftQR(s, e, Ut, Vt, i_start, i_end, computeUV);
            }
            else {
                implicitZeroShiftQR(s, e, Ut, Vt, i_start, i_end, computeUV);
            }
            if (ind == SingularValueDecomposition.maxIter) {
                break;
            }
            ++ind;
        }
        for (int k = 0; k < len; ++k) {
            if (s[k] < 0.0) {
                if (computeUV) {
                    ArrayOperator.timesAssign(Ut[k], -1.0);
                }
                final double[] array = s;
                final int n2 = k;
                array[n2] *= -1.0;
            }
        }
        quickSort(s, Ut, Vt, 0, len - 1, "descend", computeUV);
        final Matrix[] USV = new Matrix[3];
        if (computeUV) {
            USV[0] = new DenseMatrix(Ut).transpose();
        }
        else {
            USV[0] = null;
        }
        USV[1] = buildS(s, m, n);
        if (computeUV) {
            USV[2] = new DenseMatrix(Vt).transpose();
        }
        else {
            USV[2] = null;
        }
        return USV;
    }
    
    private static void quickSort(final double[] s, final double[][] Ut, final double[][] Vt, final int start, final int end, final String order, final boolean computeUV) {
        int i = start;
        int j = end;
        final double temp = s[i];
        final double[] tempU = (double[])(computeUV ? Ut[i] : null);
        final double[] tempV = (double[])(computeUV ? Vt[i] : null);
        do {
            if (order.equals("ascend")) {
                while (s[j] > temp) {
                    if (j <= i) {
                        break;
                    }
                    --j;
                }
            }
            else if (order.equals("descend")) {
                while (s[j] < temp && j > i) {
                    --j;
                }
            }
            if (j > i) {
                s[i] = s[j];
                if (computeUV) {
                    Ut[i] = Ut[j];
                    Vt[i] = Vt[j];
                }
                ++i;
            }
            if (order.equals("ascend")) {
                while (s[i] < temp) {
                    if (j <= i) {
                        break;
                    }
                    ++i;
                }
            }
            else if (order.equals("descend")) {
                while (s[i] > temp && j > i) {
                    ++i;
                }
            }
            if (j > i) {
                s[j] = s[i];
                if (computeUV) {
                    Ut[j] = Ut[i];
                    Vt[j] = Vt[i];
                }
                --j;
            }
        } while (i != j);
        s[i] = temp;
        if (computeUV) {
            Ut[i] = tempU;
            Vt[i] = tempV;
        }
        ++i;
        --j;
        if (start < j) {
            quickSort(s, Ut, Vt, start, j, order, computeUV);
        }
        if (i < end) {
            quickSort(s, Ut, Vt, i, end, order, computeUV);
        }
    }
    
    private static void standardShiftedQR(final double[] s, final double[] e, final double[][] Ut, final double[][] Vt, final int i_start, final int i_end, final boolean computeUV) {
        double d = 0.0;
        if (i_end >= 2) {
            d = ((s[i_end - 1] + s[i_end]) * (s[i_end - 1] - s[i_end]) + (e[i_end - 2] + e[i_end - 1]) * (e[i_end - 2] - e[i_end - 1])) / 2.0;
        }
        else {
            d = ((s[i_end - 1] + s[i_end]) * (s[i_end - 1] - s[i_end]) - e[i_end - 1] * e[i_end - 1]) / 2.0;
        }
        double c = s[i_end - 1] * e[i_end - 1];
        c *= c;
        double shift = Math.sqrt(d * d + c);
        shift = ((d > 0.0) ? shift : (-shift));
        shift = c / (d + shift);
        double f = (s[i_start] + s[i_end]) * (s[i_start] - s[i_end]) - e[i_end - 1] * e[i_end - 1] + shift;
        double g = s[i_start] * e[i_start];
        double cs = 0.0;
        double sn = 0.0;
        double r = 0.0;
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
            if (computeUV) {
                update(cs, sn, Vt[i], Vt[i + 1]);
            }
            if (i != i_start) {
                e[i - 1] = r;
            }
            f = cs * s[i] + sn * e[i];
            e[i] = cs * e[i] - sn * s[i];
            g = sn * s[i + 1];
            final int n = i + 1;
            s[n] *= cs;
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
            if (computeUV) {
                update(cs, sn, Ut[i], Ut[i + 1]);
            }
            s[i] = r;
            f = cs * e[i] + sn * s[i + 1];
            s[i + 1] = -sn * e[i] + cs * s[i + 1];
            g = sn * e[i + 1];
            final int n2 = i + 1;
            e[n2] *= cs;
        }
        e[i_end - 1] = f;
    }
    
    private static void implicitZeroShiftQR(final double[] s, final double[] e, final double[][] Ut, final double[][] Vt, final int i_start, final int i_end, final boolean computeUV) {
        double oldcs = 1.0;
        double oldsn = 0.0;
        double f = s[i_start];
        double g = e[i_start];
        double h = 0.0;
        double cs = 0.0;
        double sn = 0.0;
        double r = 0.0;
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
            if (computeUV) {
                update(cs, sn, Vt[i], Vt[i + 1]);
            }
            if (i != i_start) {
                e[i - 1] = oldsn * r;
            }
            f = oldcs * r;
            g = s[i + 1] * sn;
            h = s[i + 1] * cs;
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
            if (computeUV) {
                update(cs, sn, Ut[i], Ut[i + 1]);
            }
            s[i] = r;
            f = h;
            g = e[i + 1];
            oldcs = cs;
            oldsn = sn;
        }
        e[i_end - 1] = h * sn;
        s[i_end] = h * cs;
    }
    
    private static Matrix buildS(final double[] s, final int m, final int n) {
        final TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();
        for (int i = 0; i < m; ++i) {
            if (i < n) {
                map.put(Pair.of(i, i), s[i]);
            }
        }
        return SparseMatrix.createSparseMatrix(map, m, n);
    }
    
    private static Matrix buildB(final double[] s, final double[] e, final int m, final int n) {
        final TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();
        for (int i = 0; i < m; ++i) {
            if (i < n) {
                map.put(Pair.of(i, i), s[i]);
            }
            if (i < n - 1) {
                map.put(Pair.of(i, i + 1), e[i]);
            }
        }
        return SparseMatrix.createSparseMatrix(map, m, n);
    }
    
    private static void update(final double cs, final double sn, final double[] V1, final double[] V2) {
        for (int i = 0; i < V1.length; ++i) {
            final double temp = V1[i];
            V1[i] = cs * temp + sn * V2[i];
            V2[i] = -sn * temp + cs * V2[i];
        }
    }
    
    private static Matrix[] bidiagonalize(final Matrix A) {
        return bidiagonalize(A, true);
    }
    
    private static Matrix[] bidiagonalize(Matrix A, final boolean computeUV) {
        A = Matlab.full(A).copy();
        final int m = A.getRowDimension();
        final int n = A.getColumnDimension();
        Matrix[] UBV = new Matrix[3];
        final double[] d = ArrayOperator.allocateVector(n, 0.0);
        final double[] e = ArrayOperator.allocateVector(n, 0.0);
        if (A instanceof DenseMatrix) {
            final double[][] AData = ((DenseMatrix)A).getData();
            double c = 0.0;
            double s = 0.0;
            double r = 0.0;
            for (int j = 0; j < n; ++j) {
                if (j >= m) {
                    break;
                }
                c = 0.0;
                for (int i = j; i < m; ++i) {
                    c += Math.pow(AData[i][j], 2.0);
                }
                if (c != 0.0) {
                    s = Math.sqrt(c);
                    d[j] = ((AData[j][j] > 0.0) ? (-s) : s);
                    r = Math.sqrt(s * (s + Math.abs(AData[j][j])));
                    final double[] array = AData[j];
                    final int n2 = j;
                    array[n2] -= d[j];
                    for (int k = j; k < m; ++k) {
                        final double[] array2 = AData[k];
                        final int n3 = j;
                        array2[n3] /= r;
                    }
                    for (int k = j + 1; k < n; ++k) {
                        s = 0.0;
                        for (int t = j; t < m; ++t) {
                            s += AData[t][j] * AData[t][k];
                        }
                        for (int t = j; t < m; ++t) {
                            final double[] array3 = AData[t];
                            final int n4 = k;
                            array3[n4] -= s * AData[t][j];
                        }
                    }
                }
                if (j < n - 1) {
                    c = 0.0;
                    final double[] ARow_j = AData[j];
                    for (int l = j + 1; l < n; ++l) {
                        c += Math.pow(ARow_j[l], 2.0);
                    }
                    if (c != 0.0) {
                        s = Math.sqrt(c);
                        e[j + 1] = ((ARow_j[j + 1] > 0.0) ? (-s) : s);
                        r = Math.sqrt(s * (s + Math.abs(ARow_j[j + 1])));
                        final double[] array4 = ARow_j;
                        final int n5 = j + 1;
                        array4[n5] -= e[j + 1];
                        for (int l = j + 1; l < n; ++l) {
                            final double[] array5 = ARow_j;
                            final int n6 = l;
                            array5[n6] /= r;
                        }
                        double[] ARow_k = null;
                        for (int k2 = j + 1; k2 < m; ++k2) {
                            ARow_k = AData[k2];
                            s = 0.0;
                            for (int t2 = j + 1; t2 < n; ++t2) {
                                s += ARow_j[t2] * ARow_k[t2];
                            }
                            for (int t2 = j + 1; t2 < n; ++t2) {
                                final double[] array6 = ARow_k;
                                final int n7 = t2;
                                array6[n7] -= s * ARow_j[t2];
                            }
                        }
                    }
                }
            }
        }
        else {
            final boolean b = A instanceof SparseMatrix;
        }
        UBV = unpack(A, d, e, computeUV);
        return UBV;
    }
    
    private static Matrix[] unpack(final Matrix A, final double[] d, final double[] e, final boolean computeUV) {
        final Matrix[] UBV = new Matrix[3];
        final int m = A.getRowDimension();
        final int n = A.getColumnDimension();
        DenseMatrix U = null;
        if (computeUV) {
            U = new DenseMatrix(m, m, 0.0);
            final double[][] UData = U.getData();
            double s = 0.0;
            double[] y = null;
            for (int i = 0; i < m; ++i) {
                y = UData[i];
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
        }
        final TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();
        for (int l = 0; l < m; ++l) {
            if (l < n) {
                map.put(Pair.of(l, l), d[l]);
            }
            if (l < n - 1) {
                map.put(Pair.of(l, l + 1), e[l + 1]);
            }
        }
        final Matrix B = SparseMatrix.createSparseMatrix(map, m, n);
        DenseMatrix V = null;
        if (computeUV) {
            V = new DenseMatrix(n, n, 0.0);
            final double[][] VData = V.getData();
            double s2 = 0.0;
            double[] y2 = null;
            for (int i2 = 0; i2 < n; ++i2) {
                y2 = VData[i2];
                y2[i2] = 1.0;
                for (int j2 = 0; j2 < n - 1; ++j2) {
                    if (j2 == n - 2) {
                        int a = 0;
                        a += a;
                    }
                    s2 = 0.0;
                    for (int k2 = j2 + 1; k2 < n; ++k2) {
                        s2 += A.getEntry(j2, k2) * y2[k2];
                    }
                    for (int k2 = j2 + 1; k2 < n; ++k2) {
                        final double[] array2 = y2;
                        final int n3 = k2;
                        array2[n3] -= A.getEntry(j2, k2) * s2;
                    }
                }
            }
        }
        UBV[0] = U;
        UBV[1] = B;
        UBV[2] = V;
        return UBV;
    }
}
