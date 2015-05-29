package la.decomposition;

import la.matrix.*;
import la.vector.*;
import ml.utils.*;

public class LUDecomposition
{
    private Matrix L;
    private Matrix U;
    private Matrix P;
    private int numRowExchange;
    
    public static void main(final String[] args) {
        final double[][] data = { { 1.0, -2.0, 3.0 }, { 2.0, -5.0, 12.0 }, { 0.0, 2.0, -10.0 } };
        Matrix A = new DenseMatrix(data);
        Printer.fprintf("A:\n", new Object[0]);
        Printer.printMatrix(A);
        Matrix[] LUP = decompose(A);
        Matrix L = LUP[0];
        Matrix U = LUP[1];
        Matrix P = LUP[2];
        Printer.fprintf("L:\n", new Object[0]);
        Printer.printMatrix(L);
        Printer.fprintf("U:\n", new Object[0]);
        Printer.printMatrix(U);
        Printer.fprintf("P:\n", new Object[0]);
        Printer.printMatrix(P);
        Printer.fprintf("PA:\n", new Object[0]);
        Printer.printMatrix(P.mtimes(A));
        Printer.fprintf("LU:\n", new Object[0]);
        Printer.printMatrix(L.mtimes(U));
        long start = 0L;
        start = System.currentTimeMillis();
        LUDecomposition LUDecomp = new LUDecomposition(A);
        Vector b = new DenseVector(new double[] { 2.0, 3.0, 4.0 });
        Vector x = LUDecomp.solve(b);
        Printer.fprintf("Solution for Ax = b:\n", new Object[0]);
        Printer.printVector(x);
        Printer.fprintf("b = \n", new Object[0]);
        Printer.printVector(b);
        Printer.fprintf("Ax = \n", new Object[0]);
        Printer.printVector(A.operate(x));
        Printer.fprintf("A^{-1}:\n", new Object[0]);
        Printer.printMatrix(LUDecomp.inverse());
        Printer.fprintf("det(A) = %.2f\n", LUDecomp.det());
        System.out.format("Elapsed time: %.2f seconds.\n", (System.currentTimeMillis() - start) / 1000.0f);
        Printer.fprintf("**********************************\n", new Object[0]);
        A = Matlab.sparse(A);
        Printer.fprintf("A:\n", new Object[0]);
        Printer.printMatrix(A);
        LUP = decompose(A);
        L = LUP[0];
        U = LUP[1];
        P = LUP[2];
        Printer.fprintf("L:\n", new Object[0]);
        Printer.printMatrix(L);
        Printer.fprintf("U:\n", new Object[0]);
        Printer.printMatrix(U);
        Printer.fprintf("P:\n", new Object[0]);
        Printer.printMatrix(P);
        Printer.fprintf("PA:\n", new Object[0]);
        Printer.printMatrix(P.mtimes(A));
        Printer.fprintf("LU:\n", new Object[0]);
        Printer.printMatrix(L.mtimes(U));
        start = System.currentTimeMillis();
        LUDecomp = new LUDecomposition(Matlab.sparse(A));
        b = new DenseVector(new double[] { 2.0, 3.0, 4.0 });
        x = LUDecomp.solve(b);
        Printer.fprintf("Solution for Ax = b:\n", new Object[0]);
        Printer.printVector(x);
        Printer.fprintf("Ax = \n", new Object[0]);
        Printer.printVector(A.operate(x));
        Printer.fprintf("b = \n", new Object[0]);
        Printer.printVector(b);
        final Matrix B = new DenseMatrix(new double[][] { { 2.0, 4.0 }, { 3.0, 3.0 }, { 4.0, 2.0 } });
        final Matrix X = LUDecomp.solve(B);
        Printer.fprintf("Solution for AX = B:\n", new Object[0]);
        Printer.printMatrix(X);
        Printer.fprintf("AX = \n", new Object[0]);
        Printer.printMatrix(A.mtimes(X));
        Printer.fprintf("B = \n", new Object[0]);
        Printer.printMatrix(B);
        Printer.fprintf("A^{-1}:\n", new Object[0]);
        Printer.printMatrix(LUDecomp.inverse());
        Printer.fprintf("det(A) = %.2f\n", LUDecomp.det());
        System.out.format("Elapsed time: %.2f seconds.\n", (System.currentTimeMillis() - start) / 1000.0f);
    }
    
    public Matrix getL() {
        return this.L;
    }
    
    public Matrix getU() {
        return this.U;
    }
    
    public Matrix getP() {
        return this.P;
    }
    
    public LUDecomposition(final Matrix A) {
        final Matrix[] LUP = this.run(A);
        this.L = LUP[0];
        this.U = LUP[1];
        this.P = LUP[2];
    }
    
    private Matrix[] run(final Matrix A) {
        final int n = A.getRowDimension();
        if (n != A.getColumnDimension()) {
            System.err.println("A should be a square matrix.");
            System.exit(1);
        }
        this.numRowExchange = 0;
        final Matrix[] LUP = new Matrix[3];
        if (A instanceof DenseMatrix) {
            final double[][] L = new DenseMatrix(n, n, 0.0).getData();
            final double[][] AData = ((DenseMatrix)A.copy()).getData();
            final double[][] P = new DenseMatrix(n, n, 0.0).getData();
            for (int i = 0; i < n; ++i) {
                P[i][i] = 1.0;
            }
            for (int i = 0; i < n; ++i) {
                double maxVal = AData[i][i];
                int j = i;
                for (int k = i + 1; k < n; ++k) {
                    if (maxVal < AData[k][i]) {
                        j = k;
                        maxVal = AData[k][i];
                    }
                }
                if (maxVal == 0.0) {
                    System.err.println("Matrix A is singular.");
                    LUP[0] = null;
                    LUP[2] = (LUP[1] = null);
                    return LUP;
                }
                if (j != i) {
                    double[] temp = null;
                    temp = AData[i];
                    AData[i] = AData[j];
                    AData[j] = temp;
                    temp = L[i];
                    L[i] = L[j];
                    L[j] = temp;
                    temp = P[i];
                    P[i] = P[j];
                    P[j] = temp;
                    ++this.numRowExchange;
                }
                L[i][i] = 1.0;
                final double[] A_i = AData[i];
                double L_ki = 0.0;
                for (int l = i + 1; l < n; ++l) {
                    L_ki = AData[l][i] / maxVal;
                    L[l][i] = L_ki;
                    final double[] A_k = AData[l];
                    A_k[i] = 0.0;
                    for (int m = i + 1; m < n; ++m) {
                        final double[] array = A_k;
                        final int n2 = m;
                        array[n2] -= L_ki * A_i[m];
                    }
                }
            }
            LUP[0] = new DenseMatrix(L);
            LUP[1] = new DenseMatrix(AData);
            LUP[2] = new DenseMatrix(P);
        }
        else if (A instanceof SparseMatrix) {
            final Vector[] LVs = Matlab.sparseMatrix2SparseRowVectors(new SparseMatrix(n, n));
            final Vector[] AVs = Matlab.sparseMatrix2SparseRowVectors(A);
            final Vector[] PVs = Matlab.sparseMatrix2SparseRowVectors(new SparseMatrix(n, n));
            for (int i = 0; i < n; ++i) {
                PVs[i].set(i, 1.0);
            }
            for (int i = 0; i < n; ++i) {
                double maxVal = AVs[i].get(i);
                int j = i;
                for (int k = i + 1; k < n; ++k) {
                    final double v = AVs[k].get(i);
                    if (maxVal < v) {
                        j = k;
                        maxVal = v;
                    }
                }
                if (maxVal == 0.0) {
                    System.err.println("Matrix A is singular.");
                    LUP[0] = null;
                    LUP[2] = (LUP[1] = null);
                    return LUP;
                }
                if (j != i) {
                    Vector temp2 = null;
                    temp2 = AVs[i];
                    AVs[i] = AVs[j];
                    AVs[j] = temp2;
                    temp2 = LVs[i];
                    LVs[i] = LVs[j];
                    LVs[j] = temp2;
                    temp2 = PVs[i];
                    PVs[i] = PVs[j];
                    PVs[j] = temp2;
                    ++this.numRowExchange;
                }
                LVs[i].set(i, 1.0);
                final Vector A_i2 = AVs[i];
                double L_ki = 0.0;
                for (int l = i + 1; l < n; ++l) {
                    L_ki = AVs[l].get(i) / maxVal;
                    LVs[l].set(i, L_ki);
                    final Vector A_k2 = AVs[l];
                    A_k2.set(i, 0.0);
                    for (int m = i + 1; m < n; ++m) {
                        A_k2.set(m, A_k2.get(m) - L_ki * A_i2.get(m));
                    }
                }
            }
            LUP[0] = Matlab.sparseRowVectors2SparseMatrix(LVs);
            LUP[1] = Matlab.sparseRowVectors2SparseMatrix(AVs);
            LUP[2] = Matlab.sparseRowVectors2SparseMatrix(PVs);
        }
        return LUP;
    }
    
    public static Matrix[] decompose(final Matrix A) {
        final int n = A.getRowDimension();
        if (n != A.getColumnDimension()) {
            System.err.println("A should be a square matrix.");
            System.exit(1);
        }
        final Matrix[] LUP = new Matrix[3];
        if (A instanceof DenseMatrix) {
            final double[][] L = new DenseMatrix(n, n, 0.0).getData();
            final double[][] AData = ((DenseMatrix)A.copy()).getData();
            final double[][] P = new DenseMatrix(n, n, 0.0).getData();
            for (int i = 0; i < n; ++i) {
                P[i][i] = 1.0;
            }
            for (int i = 0; i < n; ++i) {
                double maxVal = AData[i][i];
                int j = i;
                for (int k = i + 1; k < n; ++k) {
                    if (maxVal < AData[k][i]) {
                        j = k;
                        maxVal = AData[k][i];
                    }
                }
                if (maxVal == 0.0) {
                    System.err.println("Matrix A is singular.");
                    LUP[0] = null;
                    LUP[2] = (LUP[1] = null);
                    return LUP;
                }
                if (j != i) {
                    double[] temp = null;
                    temp = AData[i];
                    AData[i] = AData[j];
                    AData[j] = temp;
                    temp = L[i];
                    L[i] = L[j];
                    L[j] = temp;
                    temp = P[i];
                    P[i] = P[j];
                    P[j] = temp;
                }
                L[i][i] = 1.0;
                final double[] A_i = AData[i];
                double L_ki = 0.0;
                for (int l = i + 1; l < n; ++l) {
                    L_ki = AData[l][i] / maxVal;
                    L[l][i] = L_ki;
                    final double[] A_k = AData[l];
                    A_k[i] = 0.0;
                    for (int m = i + 1; m < n; ++m) {
                        final double[] array = A_k;
                        final int n2 = m;
                        array[n2] -= L_ki * A_i[m];
                    }
                }
            }
            LUP[0] = new DenseMatrix(L);
            LUP[1] = new DenseMatrix(AData);
            LUP[2] = new DenseMatrix(P);
        }
        else if (A instanceof SparseMatrix) {
            final Vector[] LVs = Matlab.sparseMatrix2SparseRowVectors(new SparseMatrix(n, n));
            final Vector[] AVs = Matlab.sparseMatrix2SparseRowVectors(A);
            final Vector[] PVs = Matlab.sparseMatrix2SparseRowVectors(new SparseMatrix(n, n));
            for (int i = 0; i < n; ++i) {
                PVs[i].set(i, 1.0);
            }
            for (int i = 0; i < n; ++i) {
                double maxVal = AVs[i].get(i);
                int j = i;
                for (int k = i + 1; k < n; ++k) {
                    final double v = AVs[k].get(i);
                    if (maxVal < v) {
                        j = k;
                        maxVal = v;
                    }
                }
                if (maxVal == 0.0) {
                    System.err.println("Matrix A is singular.");
                    LUP[0] = null;
                    LUP[2] = (LUP[1] = null);
                    return LUP;
                }
                if (j != i) {
                    Vector temp2 = null;
                    temp2 = AVs[i];
                    AVs[i] = AVs[j];
                    AVs[j] = temp2;
                    temp2 = LVs[i];
                    LVs[i] = LVs[j];
                    LVs[j] = temp2;
                    temp2 = PVs[i];
                    PVs[i] = PVs[j];
                    PVs[j] = temp2;
                }
                LVs[i].set(i, 1.0);
                final Vector A_i2 = AVs[i];
                double L_ki = 0.0;
                for (int l = i + 1; l < n; ++l) {
                    L_ki = AVs[l].get(i) / maxVal;
                    LVs[l].set(i, L_ki);
                    final Vector A_k2 = AVs[l];
                    A_k2.set(i, 0.0);
                    for (int m = i + 1; m < n; ++m) {
                        A_k2.set(m, A_k2.get(m) - L_ki * A_i2.get(m));
                    }
                }
            }
            LUP[0] = Matlab.sparseRowVectors2SparseMatrix(LVs);
            LUP[1] = Matlab.sparseRowVectors2SparseMatrix(AVs);
            LUP[2] = Matlab.sparseRowVectors2SparseMatrix(PVs);
        }
        return LUP;
    }
    
    private static void swap(final double[] V1, final double[] V2, final int start, final int end) {
        double temp = 0.0;
        for (int i = start; i < end; ++i) {
            temp = V1[i];
            V1[i] = V2[i];
            V2[i] = temp;
        }
    }
    
    public Vector solve(final double[] b) {
        return this.solve(new DenseVector(b));
    }
    
    public Vector solve(final Vector b) {
        Vector res = null;
        if (this.L instanceof DenseMatrix) {
            final double[] d = Matlab.full(this.P.operate(b)).getPr();
            final int n = this.L.getColumnDimension();
            final double[][] LData = Matlab.full(this.L).getData();
            double[] LData_i = null;
            final double[] y = new double[n];
            double v = 0.0;
            for (int i = 0; i < n; ++i) {
                v = d[i];
                LData_i = LData[i];
                for (int k = 0; k < i; ++k) {
                    v -= LData_i[k] * y[k];
                }
                y[i] = v;
            }
            final double[][] UData = Matlab.full(this.U).getData();
            double[] UData_i = null;
            final double[] x = new double[n];
            v = 0.0;
            for (int j = n - 1; j > -1; --j) {
                UData_i = UData[j];
                v = y[j];
                for (int l = n - 1; l > j; --l) {
                    v -= UData_i[l] * x[l];
                }
                x[j] = v / UData_i[j];
            }
            res = new DenseVector(x);
        }
        else if (this.L instanceof SparseMatrix) {
            final double[] d = Matlab.full(this.P.operate(b)).getPr();
            final int n = this.L.getColumnDimension();
            final Vector[] LVs = Matlab.sparseMatrix2SparseRowVectors(this.L);
            Vector LRow_i = null;
            final double[] y = new double[n];
            double v = 0.0;
            for (int i = 0; i < n; ++i) {
                v = d[i];
                LRow_i = LVs[i];
                final int[] ir = ((SparseVector)LRow_i).getIr();
                final double[] pr = ((SparseVector)LRow_i).getPr();
                final int nnz = ((SparseVector)LRow_i).getNNZ();
                int idx = -1;
                for (int m = 0; m < nnz; ++m) {
                    idx = ir[m];
                    if (idx >= i) {
                        break;
                    }
                    v -= pr[m] * y[idx];
                }
                y[i] = v;
            }
            final Vector[] UVs = Matlab.sparseMatrix2SparseRowVectors(this.U);
            Vector URow_i = null;
            final double[] x = new double[n];
            v = 0.0;
            for (int j = n - 1; j > -1; --j) {
                URow_i = UVs[j];
                v = y[j];
                final int[] ir2 = ((SparseVector)URow_i).getIr();
                final double[] pr2 = ((SparseVector)URow_i).getPr();
                final int nnz2 = ((SparseVector)URow_i).getNNZ();
                int idx2 = -1;
                int k2 = nnz2 - 1;
                while (true) {
                    idx2 = ir2[k2];
                    if (idx2 <= j) {
                        break;
                    }
                    v -= pr2[k2] * x[idx2];
                    --k2;
                }
                x[j] = v / URow_i.get(j);
            }
            res = new DenseVector(x);
        }
        return res;
    }
    
    public Matrix solve(final double[][] B) {
        return this.solve(new DenseMatrix(B));
    }
    
    public Matrix solve(final Matrix B) {
        Matrix res = null;
        if (this.L instanceof DenseMatrix) {
            final double[][] D = Matlab.full(this.P.mtimes(B)).getData();
            double[] DRow_i = null;
            final int n = this.L.getColumnDimension();
            final double[][] LData = Matlab.full(this.L).getData();
            double[] LRow_i = null;
            final double[][] Y = ArrayOperator.allocate2DArray(n, B.getColumnDimension(), 0.0);
            double[] YRow_i = null;
            double v = 0.0;
            for (int i = 0; i < n; ++i) {
                LRow_i = LData[i];
                DRow_i = D[i];
                YRow_i = Y[i];
                for (int j = 0; j < B.getColumnDimension(); ++j) {
                    v = DRow_i[j];
                    for (int k = 0; k < i; ++k) {
                        v -= LRow_i[k] * Y[k][j];
                    }
                    YRow_i[j] = v;
                }
            }
            final double[][] UData = Matlab.full(this.U).getData();
            double[] URow_i = null;
            final double[][] X = ArrayOperator.allocate2DArray(n, B.getColumnDimension(), 0.0);
            double[] XRow_i = null;
            for (int l = n - 1; l > -1; --l) {
                URow_i = UData[l];
                YRow_i = Y[l];
                XRow_i = X[l];
                for (int m = 0; m < B.getColumnDimension(); ++m) {
                    v = YRow_i[m];
                    for (int k2 = n - 1; k2 > l; --k2) {
                        v -= URow_i[k2] * X[k2][m];
                    }
                    XRow_i[m] = v / URow_i[l];
                }
            }
            res = new DenseMatrix(X);
        }
        else if (this.L instanceof SparseMatrix) {
            final double[][] D = Matlab.full(this.P.mtimes(B)).getData();
            double[] DRow_i = null;
            final int n = this.L.getColumnDimension();
            final Vector[] LVs = Matlab.sparseMatrix2SparseRowVectors(this.L);
            Vector LRow_i2 = null;
            final double[][] Y = ArrayOperator.allocate2DArray(n, B.getColumnDimension(), 0.0);
            double[] YRow_i = null;
            double v = 0.0;
            for (int i = 0; i < n; ++i) {
                LRow_i2 = LVs[i];
                final int[] ir = ((SparseVector)LRow_i2).getIr();
                final double[] pr = ((SparseVector)LRow_i2).getPr();
                final int nnz = ((SparseVector)LRow_i2).getNNZ();
                int idx = -1;
                DRow_i = D[i];
                YRow_i = Y[i];
                for (int m = 0; m < B.getColumnDimension(); ++m) {
                    v = DRow_i[m];
                    for (int k2 = 0; k2 < nnz; ++k2) {
                        idx = ir[k2];
                        if (idx >= i) {
                            break;
                        }
                        v -= pr[k2] * Y[idx][m];
                    }
                    YRow_i[m] = v;
                }
            }
            final Vector[] UVs = Matlab.sparseMatrix2SparseRowVectors(this.U);
            Vector URow_i2 = null;
            final double[][] X = ArrayOperator.allocate2DArray(n, B.getColumnDimension(), 0.0);
            double[] XRow_i = null;
            for (int l = n - 1; l > -1; --l) {
                URow_i2 = UVs[l];
                final int[] ir2 = ((SparseVector)URow_i2).getIr();
                final double[] pr2 = ((SparseVector)URow_i2).getPr();
                final int nnz2 = ((SparseVector)URow_i2).getNNZ();
                int idx2 = -1;
                YRow_i = Y[l];
                XRow_i = X[l];
                for (int j2 = 0; j2 < B.getColumnDimension(); ++j2) {
                    v = YRow_i[j2];
                    int k3 = nnz2 - 1;
                    while (true) {
                        idx2 = ir2[k3];
                        if (idx2 <= l) {
                            break;
                        }
                        v -= pr2[k3] * X[idx2][j2];
                        --k3;
                    }
                    XRow_i[j2] = v / URow_i2.get(l);
                }
            }
            res = new DenseMatrix(X);
        }
        return res;
    }
    
    public Matrix inverse() {
        if (this.U == null) {
            return null;
        }
        final int n = this.L.getColumnDimension();
        final double[][] AInverseTransposeData = new double[n][];
        final double[][] eye = new double[n][];
        for (int i = 0; i < n; ++i) {
            (eye[i] = ArrayOperator.allocateVector(n, 0.0))[i] = 1.0;
        }
        for (int i = 0; i < n; ++i) {
            AInverseTransposeData[i] = Matlab.full(this.solve(eye[i])).getPr();
        }
        return new DenseMatrix(AInverseTransposeData).transpose();
    }
    
    public double det() {
        if (this.U == null) {
            return 0.0;
        }
        double s = 1.0;
        for (int k = 0; k < this.U.getColumnDimension(); ++k) {
            s *= this.U.getEntry(k, k);
            if (s == 0.0) {
                break;
            }
        }
        return (this.numRowExchange % 2 == 0) ? s : (-s);
    }
}
