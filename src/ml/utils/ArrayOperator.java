package ml.utils;

import la.matrix.*;

public class ArrayOperator
{
    public static void main(final String[] args) {
        final double[] A = { 1.0, 1.0, 1.0, 1.0, 1.0 };
        Printer.printVector(A);
        Printer.println(sort(A, "descend"));
        Printer.printVector(A);
    }
    
    public static double[][] inv(final double[][] A) {
        final int M = A.length;
        final int N = A[0].length;
        if (M != N) {
            System.err.println("The input 2D array should be square.");
            System.exit(1);
        }
        return Matlab.full(Matlab.mldivide(new DenseMatrix(A), Matlab.speye(M))).getData();
    }
    
    public static int[] sort(final double[] V, final String order) {
        final int len = V.length;
        final int[] indices = colon(0, 1, len - 1);
        final int start = 0;
        final int end = len - 1;
        quickSort(V, indices, start, end, order);
        return indices;
    }
    
    public static int[] sort(final double[] V) {
        return sort(V, "ascend");
    }
    
    public static void quickSort(final double[] values, final int[] indices, final int start, final int end, final String order) {
        int i = start;
        int j = end;
        final double temp = values[i];
        final int tempV = indices[i];
        do {
            if (order.equals("ascend")) {
                while (values[j] >= temp) {
                    if (j <= i) {
                        break;
                    }
                    --j;
                }
            }
            else if (order.equals("descend")) {
                while (values[j] <= temp && j > i) {
                    --j;
                }
            }
            if (j > i) {
                values[i] = values[j];
                indices[i] = indices[j];
                ++i;
            }
            if (order.equals("ascend")) {
                while (values[i] <= temp) {
                    if (i >= j) {
                        break;
                    }
                    ++i;
                }
            }
            else if (order.equals("descend")) {
                while (values[i] >= temp && i < j) {
                    ++i;
                }
            }
            if (j > i) {
                values[j] = values[i];
                indices[j] = indices[i];
                --j;
            }
        } while (i != j);
        values[i] = temp;
        indices[i] = tempV;
        ++i;
        --j;
        if (start < j) {
            quickSort(values, indices, start, j, order);
        }
        if (i < end) {
            quickSort(values, indices, i, end, order);
        }
    }
    
    public static void quickSort(final double[] values, final double[] indices, final int start, final int end, final String order) {
        int i = start;
        int j = end;
        final double temp = values[i];
        final double tempV = indices[i];
        do {
            if (order.equals("ascend")) {
                while (values[j] >= temp) {
                    if (j <= i) {
                        break;
                    }
                    --j;
                }
            }
            else if (order.equals("descend")) {
                while (values[j] <= temp && j > i) {
                    --j;
                }
            }
            if (j > i) {
                values[i] = values[j];
                indices[i] = indices[j];
                ++i;
            }
            if (order.equals("ascend")) {
                while (values[i] <= temp) {
                    if (i >= j) {
                        break;
                    }
                    ++i;
                }
            }
            else if (order.equals("descend")) {
                while (values[i] >= temp && i < j) {
                    ++i;
                }
            }
            if (j > i) {
                values[j] = values[i];
                indices[j] = indices[i];
                --j;
            }
        } while (i != j);
        values[i] = temp;
        indices[i] = tempV;
        ++i;
        --j;
        if (start < j) {
            quickSort(values, indices, start, j, order);
        }
        if (i < end) {
            quickSort(values, indices, i, end, order);
        }
    }
    
    public static int fix(final double x) {
        if (x > 0.0) {
            return (int)Math.floor(x);
        }
        return (int)Math.ceil(x);
    }
    
    public static int[] colon(final int begin, final int d, final int end) {
        final int m = fix((end - begin) / d);
        if (m < 0) {
            System.err.println("Difference error!");
            System.exit(1);
        }
        final int[] res = new int[m + 1];
        for (int i = 0; i <= m; ++i) {
            res[i] = begin + i * d;
        }
        return res;
    }
    
    public static int[] colon(final int begin, final int end) {
        return colon(begin, 1, end);
    }
    
    public static double[] colon(final double begin, final double d, final double end) {
        final int m = fix((end - begin) / d);
        if (m < 0) {
            System.err.println("Difference error!");
            System.exit(1);
        }
        final double[] res = new double[m + 1];
        for (int i = 0; i <= m; ++i) {
            res[i] = begin + i * d;
        }
        return res;
    }
    
    public static double[] colon(final double begin, final double end) {
        return colon(begin, 1.0, end);
    }
    
    public static int argmax(final double[] V) {
        int maxIdx = 0;
        double maxVal = V[0];
        for (int i = 1; i < V.length; ++i) {
            if (maxVal < V[i]) {
                maxVal = V[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }
    
    public static int argmax(final double[] V, final int begin, final int end) {
        int maxIdx = begin;
        double maxVal = V[begin];
        for (int i = begin + 1; i < end; ++i) {
            if (maxVal < V[i]) {
                maxVal = V[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }
    
    public static int argmin(final double[] V) {
        int maxIdx = 0;
        double maxVal = V[0];
        for (int i = 1; i < V.length; ++i) {
            if (maxVal > V[i]) {
                maxVal = V[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }
    
    public static int argmin(final double[] V, final int begin, final int end) {
        int maxIdx = begin;
        double maxVal = V[begin];
        for (int i = begin + 1; i < end; ++i) {
            if (maxVal > V[i]) {
                maxVal = V[i];
                maxIdx = i;
            }
        }
        return maxIdx;
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
    
    public static double min(final double[] V, final int begin, final int end) {
        double res = V[0];
        for (int i = begin + 1; i < end; ++i) {
            if (res > V[i]) {
                res = V[i];
            }
        }
        return res;
    }
    
    public static double max(final double[] V, final int begin, final int end) {
        double res = V[begin];
        for (int i = begin + 1; i < end; ++i) {
            if (res < V[i]) {
                res = V[i];
            }
        }
        return res;
    }
    
    public static void assignVector(final double[] V, final double v) {
        for (int i = 0; i < V.length; ++i) {
            V[i] = v;
        }
    }
    
    public static void assign(final double[] V, final double v) {
        assignVector(V, v);
    }
    
    public static void assignIntegerVector(final int[] V, final int v) {
        for (int i = 0; i < V.length; ++i) {
            V[i] = v;
        }
    }
    
    public static void assign(final int[] V, final int v) {
        assign(V, v);
    }
    
    public static void clearVector(final double[] V) {
        assignVector(V, 0.0);
    }
    
    public static void clear(final double[] V) {
        clearVector(V);
    }
    
    public static void clearMatrix(final double[][] M) {
        for (int i = 0; i < M.length; ++i) {
            assignVector(M[i], 0.0);
        }
    }
    
    public static void clear(final double[][] M) {
        clearMatrix(M);
    }
    
    public static double[] allocate1DArray(final int n) {
        return allocateVector(n, 0.0);
    }
    
    public static double[] allocate1DArray(final int n, final double v) {
        return allocateVector(n, v);
    }
    
    public static double[] allocateVector(final int n) {
        return allocateVector(n, 0.0);
    }
    
    public static double[] allocateVector(final int n, final double v) {
        final double[] res = new double[n];
        assignVector(res, v);
        return res;
    }
    
    public static double[][] allocate2DArray(final int m, final int n, final double v) {
        final double[][] res = new double[m][];
        for (int i = 0; i < m; ++i) {
            res[i] = new double[n];
            for (int j = 0; j < n; ++j) {
                res[i][j] = v;
            }
        }
        return res;
    }
    
    public static double[][] allocate2DArray(final int m, final int n) {
        return allocate2DArray(m, n, 0.0);
    }
    
    public static int[] allocateIntegerVector(final int n) {
        return allocateIntegerVector(n, 0);
    }
    
    public static int[] allocateIntegerVector(final int n, final int v) {
        final int[] res = new int[n];
        assignIntegerVector(res, v);
        return res;
    }
    
    public static double[][] allocateMatrix(final int nRows, final int nCols) {
        final double[][] res = new double[nRows][];
        for (int i = 0; i < nRows; ++i) {
            res[i] = allocateVector(nCols);
        }
        return res;
    }
    
    public static void divideAssign(final double[] V, final double v) {
        for (int i = 0; i < V.length; ++i) {
            final int n = i;
            V[n] /= v;
        }
    }
    
    public static void divideAssign(final double[] V1, final double[] V2) {
        for (int i = 0; i < V1.length; ++i) {
            final int n = i;
            V1[n] /= V2[i];
        }
    }
    
    public static void divideAssign(final double[][] res, final double v) {
        for (int i = 0; i < res.length; ++i) {
            final double[] row = res[i];
            for (int j = 0; j < row.length; ++j) {
                final double[] array = row;
                final int n = j;
                array[n] /= v;
            }
        }
    }
    
    public static void timesAssign(final double[] V, final double v) {
        for (int i = 0; i < V.length; ++i) {
            final int n = i;
            V[n] *= v;
        }
    }
    
    public static void timesAssign(final double[] V1, final double[] V2) {
        for (int i = 0; i < V1.length; ++i) {
            final int n = i;
            V1[n] *= V2[i];
        }
    }
    
    public static void timesAssign(final double[][] res, final double v) {
        for (int i = 0; i < res.length; ++i) {
            final double[] row = res[i];
            for (int j = 0; j < row.length; ++j) {
                final double[] array = row;
                final int n = j;
                array[n] *= v;
            }
        }
    }
    
    public static double sum(final double[] V) {
        double res = 0.0;
        for (int i = 0; i < V.length; ++i) {
            res += V[i];
        }
        return res;
    }
    
    public static double mean(final double[] V) {
        return sum(V) / V.length;
    }
    
    public static double std(final double[] V, final int flag) {
        final int n = V.length;
        if (n == 1) {
            return 0.0;
        }
        final double mean = mean(V);
        double res = 0.0;
        for (final double v : V) {
            final double diff = v - mean;
            res += diff * diff;
        }
        if (flag == 0) {
            res /= n - 1;
        }
        else if (flag == 1) {
            res /= n;
        }
        res = Math.sqrt(res);
        return res;
    }
    
    public static void sum2one(final double[] V) {
        divideAssign(V, sum(V));
    }
    
    public static void plusAssign(final double[] V, final double v) {
        for (int i = 0; i < V.length; ++i) {
            final int n = i;
            V[n] += v;
        }
    }
    
    public static void plusAssign(final double[] res, final double a, final double[] V) {
        for (int i = 0; i < res.length; ++i) {
            final int n = i;
            res[n] += a * V[i];
        }
    }
    
    public static void plusAssign(final double[] V1, final double[] V2) {
        for (int i = 0; i < V1.length; ++i) {
            final int n = i;
            V1[n] += V2[i];
        }
    }
    
    public static void plusAssign(final double[][] res, final double v) {
        for (int i = 0; i < res.length; ++i) {
            final double[] row = res[i];
            for (int j = 0; j < row.length; ++j) {
                final double[] array = row;
                final int n = j;
                array[n] += v;
            }
        }
    }
    
    public static void minusAssign(final int[] V, final int v) {
        for (int i = 0; i < V.length; ++i) {
            final int n = i;
            V[n] -= v;
        }
    }
    
    public static void minusAssign(final double[] V, final double v) {
        for (int i = 0; i < V.length; ++i) {
            final int n = i;
            V[n] -= v;
        }
    }
    
    public static void minusAssign(final double[] res, final double a, final double[] V) {
        for (int i = 0; i < res.length; ++i) {
            final int n = i;
            res[n] -= a * V[i];
        }
    }
    
    public static void minusAssign(final double[] V1, final double[] V2) {
        for (int i = 0; i < V1.length; ++i) {
            final int n = i;
            V1[n] -= V2[i];
        }
    }
    
    public static void minusAssign(final double[][] res, final double v) {
        for (int i = 0; i < res.length; ++i) {
            final double[] row = res[i];
            for (int j = 0; j < row.length; ++j) {
                final double[] array = row;
                final int n = j;
                array[n] -= v;
            }
        }
    }
    
    public static void assignVector(final double[] V1, final double[] V2) {
        System.arraycopy(V2, 0, V1, 0, V1.length);
    }
    
    public static void assign(final double[][] res, final double[][] A) {
        for (int i = 0; i < res.length; ++i) {
            assignVector(res[i], A[i]);
        }
    }
    
    public static void assign(final double[][] res, final double v) {
        for (int i = 0; i < res.length; ++i) {
            assignVector(res[i], v);
        }
    }
    
    public static double[] operate(final double[][] A, final double[] V) {
        final double[] res = new double[A.length];
        double s = 0.0;
        for (int i = 0; i < res.length; ++i) {
            s = 0.0;
            final double[] A_i = A[i];
            for (int j = 0; j < V.length; ++j) {
                s += A_i[j] * V[j];
            }
            res[i] = s;
        }
        return res;
    }
    
    public static void operate(final double[] V1, final double[][] A, final double[] V2) {
        double s = 0.0;
        for (int i = 0; i < V1.length; ++i) {
            final double[] ARow = A[i];
            s = 0.0;
            for (int j = 0; j < V2.length; ++j) {
                s += ARow[j] * V2[j];
            }
            V1[i] = s;
        }
    }
    
    public static double[] operate(final double[] V, final double[][] A) {
        final double[] res = new double[A[0].length];
        double s = 0.0;
        for (int j = 0; j < res.length; ++j) {
            s = 0.0;
            for (int i = 0; i < V.length; ++i) {
                s += V[i] * A[i][j];
            }
            res[j] = s;
        }
        return res;
    }
    
    public static void operate(final double[] V1, final double[] V2, final double[][] A) {
        double s = 0.0;
        for (int j = 0; j < V1.length; ++j) {
            s = 0.0;
            for (int i = 0; i < V2.length; ++i) {
                s += V2[i] * A[i][j];
            }
            V1[j] = s;
        }
    }
    
    public static double innerProduct(final double[] V1, final double[] V2) {
        if (V1 == null || V2 == null) {
            return 0.0;
        }
        double res = 0.0;
        for (int i = 0; i < V1.length; ++i) {
            res += V1[i] * V2[i];
        }
        return res;
    }
    
    public static double innerProduct(final double[] V1, final double[] V2, final int from, final int to) {
        if (V1 == null || V2 == null) {
            return 0.0;
        }
        double res = 0.0;
        for (int i = from; i < to; ++i) {
            res += V1[i] * V2[i];
        }
        return res;
    }
    
    public static double[] times(final double[] V1, final double[] V2) {
        final double[] res = new double[V1.length];
        for (int i = 0; i < V1.length; ++i) {
            res[i] = V1[i] * V2[i];
        }
        return res;
    }
    
    public static double[] plus(final double[] V1, final double[] V2) {
        final double[] res = new double[V1.length];
        for (int i = 0; i < V1.length; ++i) {
            res[i] = V1[i] + V2[i];
        }
        return res;
    }
    
    public static double[] minus(final double[] V1, final double[] V2) {
        final double[] res = new double[V1.length];
        for (int i = 0; i < V1.length; ++i) {
            res[i] = V1[i] - V2[i];
        }
        return res;
    }
}
