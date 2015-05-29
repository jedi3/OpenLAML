package ml.utils;

import la.matrix.*;
import la.vector.*;
import la.vector.Vector;
import ml.random.*;
import la.decomposition.*;

import java.util.*;

public class Matlab
{
    public static double eps;
    public static double inf;
    
    static {
        Matlab.eps = Double.MIN_VALUE;
        Matlab.inf = Double.POSITIVE_INFINITY;
    }
    
    public static void main(final String[] args) {
        final double[] Vec = { 4.0, 2.0, 3.0, 6.0, 1.0, 8.0, 5.0, 9.0, 7.0 };
        double start = 0.0;
        start = Time.tic();
        Printer.disp(max(Vec));
        System.out.format("Elapsed time: %.9f seconds.%n", Time.toc(start));
        start = Time.tic();
        double max = Vec[0];
        for (int i = 1; i < Vec.length; ++i) {
            if (max < Vec[i]) {
                max = Vec[i];
            }
        }
        Printer.disp(max);
        System.out.format("Elapsed time: %.9f seconds.%n", Time.toc(start));
        final double[][] data = { { 10.0, -5.0, 0.0, 3.0 }, { 2.0, 0.0, 1.0, 2.0 }, { 1.0, 6.0, 0.0, 5.0 } };
        Matrix A = new DenseMatrix(data);
        Printer.disp(sigmoid(A));
        Time.tic();
        for (int j = 0; j < 0; ++j) {
            A = new DenseMatrix(1000, 1000);
        }
        System.out.format("Elapsed time: %.9f seconds.%n", Time.toc());
        final int m = 4;
        final int n = 3;
        A = hilb(m, n);
        final int[] rIndices = { 0, 1, 3, 1, 2, 2, 3, 2, 3 };
        final int[] cIndices = { 0, 0, 0, 1, 1, 2, 2, 3, 3 };
        final double[] values = { 10.0, 3.2, 3.0, 9.0, 7.0, 8.0, 8.0, 7.0, 7.0 };
        final int numRows = 4;
        final int numColumns = 4;
        final int nzmax = rIndices.length;
        A = new SparseMatrix(rIndices, cIndices, values, numRows, numColumns, nzmax);
        Printer.fprintf("A:%n", new Object[0]);
        Printer.printMatrix(A);
        Printer.fprintf("sum(A):%n", new Object[0]);
        Printer.disp(sum(A));
        Printer.fprintf("sum(A, 2):%n", new Object[0]);
        Printer.disp(sum(A, 2));
        Printer.disp("mean(A):");
        Printer.disp(mean(A, 1));
        Printer.disp("std(A):");
        Printer.disp(std(A, 0, 1));
        Printer.fprintf("max(A):%n", new Object[0]);
        Printer.disp(max(A)[0]);
        Printer.fprintf("max(A, 2):%n", new Object[0]);
        Printer.disp(max(A, 2)[0]);
        Printer.fprintf("min(A):%n", new Object[0]);
        Printer.disp(min(A)[0]);
        Printer.fprintf("min(A, 2):%n", new Object[0]);
        Printer.disp(min(A, 2)[0]);
        A = full(A);
        Printer.fprintf("A:%n", new Object[0]);
        Printer.disp(A);
        Printer.fprintf("sum(A):%n", new Object[0]);
        Printer.disp(sum(A));
        Printer.fprintf("sum(A, 2):%n", new Object[0]);
        Printer.disp(sum(A, 2));
        Printer.fprintf("max(A):%n", new Object[0]);
        Printer.disp(max(A)[0]);
        Printer.fprintf("max(A, 2):%n", new Object[0]);
        Printer.disp(max(A, 2)[0]);
        Printer.fprintf("min(A):%n", new Object[0]);
        Printer.disp(min(A)[0]);
        Printer.fprintf("min(A, 2):%n", new Object[0]);
        Printer.disp(min(A, 2)[0]);
        Printer.fprintf("A'A:%n", new Object[0]);
        Printer.disp(A.transpose().mtimes(A));
        final Matrix[] VD = EigenValueDecomposition.decompose(A.transpose().mtimes(A));
        final Matrix V = VD[0];
        final Matrix D = VD[1];
        Printer.fprintf("V:%n", new Object[0]);
        Printer.printMatrix(V);
        Printer.fprintf("D:%n", new Object[0]);
        Printer.printMatrix(D);
        Printer.fprintf("VDV':%n", new Object[0]);
        Printer.disp(V.mtimes(D).mtimes(V.transpose()));
        Printer.fprintf("A'A:%n", new Object[0]);
        Printer.printMatrix(A.transpose().mtimes(A));
        Printer.fprintf("V'V:%n", new Object[0]);
        Printer.printMatrix(V.transpose().mtimes(V));
        Printer.fprintf("norm(A, 2):%n", new Object[0]);
        Printer.disp(norm(A, 2));
        Printer.fprintf("rank(A):%n", new Object[0]);
        Printer.disp(rank(A));
        final Vector V2 = new SparseVector(3);
        V2.set(1, 1.0);
        V2.set(2, -1.0);
        Printer.disp("V1:");
        Printer.disp(V2);
        final Vector V3 = new SparseVector(3);
        V3.set(0, -1.0);
        V3.set(2, 1.0);
        Printer.disp("V2:");
        Printer.disp(V3);
        Printer.fprintf("max(V1, -1)%n", new Object[0]);
        Printer.disp(max(V2, -1.0));
        Printer.fprintf("max(V1, 1)%n", new Object[0]);
        Printer.disp(max(V2, 1.0));
        Printer.fprintf("max(V1, 0)%n", new Object[0]);
        Printer.disp(max(V2, 0.0));
        Printer.fprintf("max(V1, V2)%n", new Object[0]);
        Printer.disp(max(V2, V3));
        Printer.fprintf("min(V1, -1)%n", new Object[0]);
        Printer.disp(min(V2, -1.0));
        Printer.fprintf("min(V1, 1)%n", new Object[0]);
        Printer.disp(min(V2, 1.0));
        Printer.fprintf("min(V1, 0)%n", new Object[0]);
        Printer.disp(min(V2, 0.0));
        Printer.fprintf("min(V1, V2)%n", new Object[0]);
        Printer.disp(min(V2, V3));
        A = new SparseMatrix(rIndices, cIndices, values, numRows, numColumns, nzmax);
        Printer.disp("A:");
        Printer.printMatrix(A);
        Vector[] Vs = sparseMatrix2SparseRowVectors(A);
        Matrix S = sparseRowVectors2SparseMatrix(Vs);
        Printer.disp("S:");
        Printer.printMatrix(S);
        Vs = sparseMatrix2SparseColumnVectors(S);
        S = sparseColumnVectors2SparseMatrix(Vs);
        Printer.disp("S:");
        Printer.printMatrix(S);
        Matrix B = sparse(new DenseMatrix(size(A), 2.0).times(A.transpose()));
        Printer.disp("B:");
        Printer.printMatrix(B);
        Printer.disp("max(A, 5)");
        Printer.printMatrix(max(A, 5.0));
        Printer.disp("max(A, -2)");
        Printer.printMatrix(max(A, -2.0));
        Printer.disp("max(A, B)");
        Printer.printMatrix(max(A, B));
        Printer.disp("A:");
        Printer.printMatrix(A);
        Printer.disp("B:");
        Printer.printMatrix(B);
        Printer.disp("min(A, 5)");
        Printer.printMatrix(min(A, 5.0));
        Printer.disp("min(A, -2)");
        Printer.printMatrix(min(A, -2.0));
        Printer.disp("min(A, B)");
        Printer.printMatrix(min(A, B));
        Printer.disp("A:");
        Printer.printMatrix(A);
        final Matrix[] sortRes = sort(A, 1, "ascend");
        Printer.disp("Sorted values:");
        Printer.printMatrix(sortRes[0]);
        Printer.disp("Sorted indices:");
        Printer.disp(sortRes[1]);
        final Vector V4 = new SparseVector(8);
        V4.set(1, 6.0);
        V4.set(2, -2.0);
        V4.set(4, 9.0);
        V4.set(6, 8.0);
        Printer.disp("V3:");
        Printer.disp(V4);
        final double[] IX = sort(V4, "ascend");
        Printer.disp("Sorted V3:");
        Printer.disp(V4);
        Printer.disp("Sorted indices:");
        Printer.disp(IX);
        Printer.disp("A:");
        Printer.printMatrix(A);
        Printer.disp("repmat(A, 2, 3):");
        Printer.printMatrix(repmat(A, 2, 3));
        Printer.disp("vec(A)");
        Printer.disp(vec(A));
        Printer.disp("reshape(vec(A), 4, 4)");
        Printer.printMatrix(reshape(vec(A), 4, 4));
        A = full(A);
        Printer.disp("full(A)");
        Printer.printMatrix(A);
        Printer.disp("repmat(A, 2, 3):");
        Printer.disp(repmat(A, 2, 3));
        Printer.disp("vec(A)");
        Printer.disp(vec(A));
        Printer.disp("reshape(vec(A), 4, 4)");
        Printer.disp(reshape(vec(A), 4, 4));
        B = new DenseMatrix(new double[][] { { 3.0, 2.0 }, { 0.0, 2.0 } });
        Printer.disp("sparse(A)");
        Printer.printMatrix(sparse(A));
        Printer.disp("sparse(B)");
        Printer.printMatrix(sparse(B));
        Printer.printMatrix(kron(full(A), full(B)));
        Printer.printMatrix(kron(full(A), sparse(B)));
        Printer.printMatrix(kron(sparse(A), full(B)));
        Printer.printMatrix(kron(sparse(A), sparse(B)));
    }
    
    public static DenseVector std(final Matrix X, final int flag, final int dim) {
        if (dim != 1 && dim != 2) {
            Printer.err("dim should be 1 or 2.");
            Utility.exit(1);
        }
        final int M = X.getRowDimension();
        final int N = X.getColumnDimension();
        final DenseVector mean = mean(X, dim);
        Matrix meanMat = null;
        Matrix temp = null;
        if (dim == 1) {
            meanMat = rowVector2RowMatrix(mean);
            temp = repmat(meanMat, M, 1);
        }
        else {
            meanMat = columnVector2ColumnMatrix(mean);
            temp = repmat(meanMat, 1, N);
        }
        InPlaceOperator.minusAssign(temp, X);
        InPlaceOperator.timesAssign(temp, temp);
        final int num = size(X, dim);
        final double[] res = sum(temp, dim).getPr();
        if (flag == 0) {
            if (num == 1) {
                InPlaceOperator.clear(res);
            }
            else {
                ArrayOperator.divideAssign(res, num - 1);
            }
        }
        else if (flag == 1 && num != 1) {
            ArrayOperator.divideAssign(res, num);
        }
        for (int k = 0; k < res.length; ++k) {
            res[k] = Math.sqrt(res[k]);
        }
        return new DenseVector(res);
    }
    
    public static DenseVector std(final Matrix X, final int flag) {
        return std(X, flag, 1);
    }
    
    public static DenseVector std(final Matrix X) {
        return std(X, 0);
    }
    
    public static void setSubMatrix(final Matrix A, final int[] selectedRows, final int[] selectedColumns, final Matrix B) {
        for (int i = 0; i < selectedRows.length; ++i) {
            for (int j = 0; j < selectedColumns.length; ++j) {
                final int r = selectedRows[i];
                final int c = selectedColumns[j];
                A.setEntry(r, c, B.getEntry(i, j));
            }
        }
    }
    
    public static Matrix reshape(final Matrix A, final int[] size) {
        if (size.length != 2) {
            System.err.println("Input vector should have two elements!");
        }
        final int M = size[0];
        final int N = size[1];
        if (M * N != A.getRowDimension() * A.getColumnDimension()) {
            System.err.println("Wrong shape!");
            Utility.exit(1);
        }
        Matrix res = null;
        if (A instanceof DenseMatrix) {
            final double[][] AData = ((DenseMatrix)A).getData();
            final double[][] resData = new double[M][];
            double[] resRow = null;
            for (int i = 0, shiftI = 0; i < M; ++i, ++shiftI) {
                resData[i] = new double[N];
                resRow = resData[i];
                for (int j = 0, shiftJ = shiftI; j < N; ++j, shiftJ += M) {
                    final int r = shiftJ % A.getRowDimension();
                    final int c = shiftJ / A.getRowDimension();
                    resRow[j] = AData[r][c];
                }
            }
            res = new DenseMatrix(resData);
        }
        else if (A instanceof SparseMatrix) {
            final int[] ir = ((SparseMatrix)A).getIr();
            final int[] jc = ((SparseMatrix)A).getJc();
            final double[] pr = ((SparseMatrix)A).getPr();
            final int nnz = ((SparseMatrix)A).getNNZ();
            final int[] resIr = new int[nnz];
            final int[] resJc = new int[N + 1];
            final double[] resPr = new double[nnz];
            System.arraycopy(pr, 0, resPr, 0, nnz);
            int lastColIdx = -1;
            int currentColIdx = 0;
            int idx = 0;
            for (int k = 0, shiftJ2 = 0; k < A.getColumnDimension(); ++k, shiftJ2 += A.getRowDimension()) {
                for (int l = jc[k]; l < jc[k + 1]; ++l) {
                    idx = ir[l] + shiftJ2;
                    currentColIdx = idx / M;
                    resIr[l] = idx % M;
                    while (lastColIdx < currentColIdx) {
                        resJc[lastColIdx + 1] = l;
                        ++lastColIdx;
                    }
                }
            }
            resJc[N] = nnz;
            res = SparseMatrix.createSparseMatrixByCSCArrays(resIr, resJc, resPr, M, N, nnz);
        }
        return res;
    }
    
    public static Matrix reshape(final Matrix A, final int M, final int N) {
        return reshape(A, new int[] { M, N });
    }
    
    public static Matrix reshape(final Vector V, final int[] size) {
        if (size.length != 2) {
            System.err.println("Input vector should have two elements!");
            Utility.exit(1);
        }
        final int dim = V.getDim();
        if (size[0] * size[1] != dim) {
            System.err.println("Wrong shape!");
        }
        Matrix res = null;
        final int M = size[0];
        final int N = size[1];
        if (V instanceof DenseVector) {
            final double[][] resData = new double[M][];
            double[] resRow = null;
            final double[] pr = ((DenseVector)V).getPr();
            for (int i = 0, shiftI = 0; i < M; ++i, ++shiftI) {
                resData[i] = new double[N];
                resRow = resData[i];
                for (int j = 0, shiftJ = shiftI; j < N; ++j, shiftJ += M) {
                    resRow[j] = pr[shiftJ];
                }
            }
            res = new DenseMatrix(resData);
        }
        else if (V instanceof SparseVector) {
            final int[] ir = ((SparseVector)V).getIr();
            final double[] pr2 = ((SparseVector)V).getPr();
            final int nnz = ((SparseVector)V).getNNZ();
            final int[] resIr = new int[nnz];
            final int[] resJc = new int[N + 1];
            final double[] resPr = new double[nnz];
            System.arraycopy(pr2, 0, resPr, 0, nnz);
            int lastColIdx = -1;
            int currentColIdx = 0;
            int idx = 0;
            for (int k = 0; k < nnz; ++k) {
                idx = ir[k];
                currentColIdx = idx / M;
                resIr[k] = idx % M;
                while (lastColIdx < currentColIdx) {
                    resJc[lastColIdx + 1] = k;
                    ++lastColIdx;
                }
            }
            resJc[N] = nnz;
            res = SparseMatrix.createSparseMatrixByCSCArrays(resIr, resJc, resPr, M, N, nnz);
        }
        return res;
    }
    
    public static Matrix reshape(final Vector V, final int M, final int N) {
        final int dim = V.getDim();
        if (M * N != dim) {
            System.err.println("Wrong shape!");
        }
        Matrix res = null;
        if (V instanceof DenseVector) {
            final double[][] resData = new double[M][];
            double[] resRow = null;
            final double[] pr = ((DenseVector)V).getPr();
            for (int i = 0, shiftI = 0; i < M; ++i, ++shiftI) {
                resData[i] = new double[N];
                resRow = resData[i];
                for (int j = 0, shiftJ = shiftI; j < N; ++j, shiftJ += M) {
                    resRow[j] = pr[shiftJ];
                }
            }
            res = new DenseMatrix(resData);
        }
        else if (V instanceof SparseVector) {
            final int[] ir = ((SparseVector)V).getIr();
            final double[] pr2 = ((SparseVector)V).getPr();
            final int nnz = ((SparseVector)V).getNNZ();
            final int[] resIr = new int[nnz];
            final int[] resJc = new int[N + 1];
            final double[] resPr = new double[nnz];
            System.arraycopy(pr2, 0, resPr, 0, nnz);
            int lastColIdx = -1;
            int currentColIdx = 0;
            int idx = 0;
            for (int k = 0; k < nnz; ++k) {
                idx = ir[k];
                currentColIdx = idx / M;
                resIr[k] = idx % M;
                while (lastColIdx < currentColIdx) {
                    resJc[lastColIdx + 1] = k;
                    ++lastColIdx;
                }
            }
            resJc[N] = nnz;
            res = SparseMatrix.createSparseMatrixByCSCArrays(resIr, resJc, resPr, M, N, nnz);
        }
        return res;
    }
    
    public static Matrix vec(final Matrix A) {
        final int M = A.getRowDimension();
        final int N = A.getColumnDimension();
        if (N == 1) {
            return A;
        }
        Matrix res = null;
        final int dim = M * N;
        if (A instanceof DenseMatrix) {
            final double[][] resData = new double[dim][];
            final double[][] AData = ((DenseMatrix)A).getData();
            for (int j = 0, shift = 0; j < N; ++j, shift += M) {
                for (int i = 0, shiftI = shift; i < M; ++i, ++shiftI) {
                    resData[shiftI] = new double[] { AData[i][j] };
                }
            }
            res = new DenseMatrix(resData);
        }
        else if (A instanceof SparseMatrix) {
            final int[] ir = ((SparseMatrix)A).getIr();
            final int[] jc = ((SparseMatrix)A).getJc();
            final double[] pr = ((SparseMatrix)A).getPr();
            final int nnz = ((SparseMatrix)A).getNNZ();
            final int[] resIr = new int[nnz];
            final int[] resJc = { 0, nnz };
            final double[] resPr = new double[nnz];
            System.arraycopy(pr, 0, resPr, 0, nnz);
            int cnt = 0;
            for (int k = 0, shift2 = 0; k < N; ++k, shift2 += M) {
                for (int l = jc[k]; l < jc[k + 1]; ++l) {
                    resIr[cnt++] = ir[l] + shift2;
                }
            }
            res = SparseMatrix.createSparseMatrixByCSCArrays(resIr, resJc, resPr, dim, 1, nnz);
        }
        return res;
    }
    
    public static Matrix[] svd(final Matrix A) {
        final SingularValueDecomposition svdImpl = new SingularValueDecomposition(A);
        final Matrix U = svdImpl.getU();
        final Matrix S = svdImpl.getS();
        final Matrix V = svdImpl.getV();
        final Matrix[] res = { U, S, V };
        return res;
    }
    
    public static Matrix mvnrnd(final Matrix MU, final Matrix SIGMA, final int cases) {
        return MultivariateGaussianDistribution.mvnrnd(MU, SIGMA, cases);
    }
    
    public static Matrix mvnrnd(final double[] MU, final double[][] SIGMA, final int cases) {
        return mvnrnd(new DenseMatrix(MU, 2), new DenseMatrix(SIGMA), cases);
    }
    
    public static Matrix mvnrnd(final double[] MU, final double[] SIGMA, final int cases) {
        return mvnrnd(new DenseMatrix(MU, 2), diag(SIGMA), cases);
    }
    
    public static Matrix repmat(final Matrix A, final int M, final int N) {
        Matrix res = null;
        final int nRow = M * A.getRowDimension();
        final int nCol = N * A.getColumnDimension();
        if (A instanceof DenseMatrix) {
            final double[][] resData = ArrayOperator.allocate2DArray(nRow, nCol);
            double[] resRow = null;
            final double[][] AData = ((DenseMatrix)A).getData();
            double[] ARow = null;
            for (int i = 0; i < nRow; ++i) {
                resRow = resData[i];
                final int r = i % A.getRowDimension();
                ARow = AData[r];
                for (int k = 0, shift = 0; k < N; ++k, shift += A.getColumnDimension()) {
                    System.arraycopy(ARow, 0, resRow, shift, A.getColumnDimension());
                }
            }
            res = new DenseMatrix(resData);
        }
        else if (A instanceof SparseMatrix) {
            final int[] ir = ((SparseMatrix)A).getIr();
            final int[] jc = ((SparseMatrix)A).getJc();
            final double[] pr = ((SparseMatrix)A).getPr();
            final int nnz = ((SparseMatrix)A).getNNZ();
            final int resNNZ = nnz * N * M;
            final int[] resIr = new int[resNNZ];
            final int[] resJc = new int[nCol + 1];
            final double[] resPr = new double[resNNZ];
            final int[] nnzPerColumn = new int[A.getColumnDimension()];
            for (int j = 0; j < A.getColumnDimension(); ++j) {
                nnzPerColumn[j] = M * (jc[j + 1] - jc[j]);
            }
            resJc[0] = 0;
            for (int c = 0; c < nCol; ++c) {
                final int l = c % A.getColumnDimension();
                resJc[c + 1] = resJc[c] + nnzPerColumn[l];
            }
            int j = 0;
            int shiftA = 0;
            while (j < A.getColumnDimension()) {
                final int numNNZACol_j = jc[j + 1] - jc[j];
                final int[] irACol_j = new int[numNNZACol_j];
                for (int m = 0, shift2 = shiftA * M; m < N; ++m, shift2 += nnz * M) {
                    System.arraycopy(ir, shiftA, irACol_j, 0, numNNZACol_j);
                    for (int i2 = 0, shift3 = shift2; i2 < M; ++i2, shift3 += numNNZACol_j) {
                        System.arraycopy(irACol_j, 0, resIr, shift3, numNNZACol_j);
                        if (i2 < M - 1) {
                            for (int t = 0; t < numNNZACol_j; ++t) {
                                final int[] array = irACol_j;
                                final int n = t;
                                array[n] += A.getRowDimension();
                            }
                        }
                        System.arraycopy(pr, shiftA, resPr, shift3, numNNZACol_j);
                    }
                }
                shiftA += numNNZACol_j;
                ++j;
            }
            res = SparseMatrix.createSparseMatrixByCSCArrays(resIr, resJc, resPr, nRow, nCol, resNNZ);
        }
        return res;
    }
    
    public static Matrix repmat(final Matrix A, final int[] size) {
        return repmat(A, size[0], size[1]);
    }
    
    public static DenseVector mean(final Matrix X, final int dim) {
        final int N = size(X, dim);
        final double[] S = sum(X, dim).getPr();
        ArrayOperator.divideAssign(S, N);
        return new DenseVector(S);
    }
    
    public static Matrix[] eigs(final Matrix A, final int K, final String sigma) {
        final EigenValueDecomposition eigImpl = new EigenValueDecomposition(A, 1.0E-6);
        final Matrix eigV = eigImpl.getV();
        final Matrix eigD = eigImpl.getD();
        final int N = A.getRowDimension();
        final Matrix[] res = new Matrix[2];
        final Vector eigenValueVector = new DenseVector(K);
        Matrix eigenVectors = null;
        if (sigma.equals("lm")) {
            for (int k = 0; k < K; ++k) {
                eigenValueVector.set(k, eigD.getEntry(k, k));
            }
            eigenVectors = eigV.getSubMatrix(0, N - 1, 0, K - 1);
        }
        else if (sigma.equals("sm")) {
            for (int k = 0; k < K; ++k) {
                eigenValueVector.set(k, eigD.getEntry(N - 1 - k, N - 1 - k));
            }
            final double[][] eigenVectorsData = ArrayOperator.allocate2DArray(N, K);
            final double[][] eigVData = ((DenseMatrix)eigV).getData();
            double[] eigenVectorsRow = null;
            double[] eigVRow = null;
            for (int i = 0; i < N; ++i) {
                eigenVectorsRow = eigenVectorsData[i];
                eigVRow = eigVData[i];
                int j = N - 1;
                for (int l = 0; l < K; ++l) {
                    eigenVectorsRow[l] = eigVRow[j];
                    --j;
                }
            }
            eigenVectors = new DenseMatrix(eigenVectorsData);
        }
        else {
            System.err.println("sigma should be either \"lm\" or \"sm\"");
            System.exit(-1);
        }
        res[0] = eigenVectors;
        res[1] = diag(eigenValueVector);
        return res;
    }
    
    public static Matrix diag(final double[] V) {
        final int d = V.length;
        final Matrix res = new SparseMatrix(d, d);
        for (int i = 0; i < d; ++i) {
            res.setEntry(i, i, V[i]);
        }
        return res;
    }
    
    public static DenseMatrix log(final Matrix A) {
        final int nRow = A.getRowDimension();
        final int nCol = A.getColumnDimension();
        final double[][] resData = ArrayOperator.allocate2DArray(nRow, nCol);
        double[] resRow = null;
        if (A instanceof DenseMatrix) {
            final double[][] AData = ((DenseMatrix)A).getData();
            double[] ARow = null;
            for (int i = 0; i < nRow; ++i) {
                resRow = resData[i];
                ARow = AData[i];
                for (int j = 0; j < nCol; ++j) {
                    resRow[j] = Math.log(ARow[j]);
                }
            }
        }
        else if (A instanceof SparseMatrix) {
            final int[] ic = ((SparseMatrix)A).getIc();
            final int[] jr = ((SparseMatrix)A).getJr();
            final int[] valCSRIndices = ((SparseMatrix)A).getValCSRIndices();
            final double[] pr = ((SparseMatrix)A).getPr();
            for (int k = 0; k < nRow; ++k) {
                resRow = resData[k];
                if (jr[k + 1] == jr[k]) {
                    InPlaceOperator.assign(resRow, Double.NEGATIVE_INFINITY);
                }
                else {
                    int lastIdx = -1;
                    int currentIdx = 0;
                    for (int l = jr[k]; l < jr[k + 1]; ++l) {
                        currentIdx = ic[l];
                        for (int m = lastIdx + 1; m < currentIdx; ++m) {
                            resRow[m] = Double.NEGATIVE_INFINITY;
                        }
                        resRow[currentIdx] = Math.log(pr[valCSRIndices[l]]);
                        lastIdx = currentIdx;
                    }
                    for (int j2 = lastIdx + 1; j2 < nCol; ++j2) {
                        resRow[j2] = Double.NEGATIVE_INFINITY;
                    }
                }
            }
        }
        return new DenseMatrix(resData);
    }
    
    public static Matrix getTFIDF(final Matrix docTermCountMatrix) {
        final int NTerm = docTermCountMatrix.getRowDimension();
        final int NDoc = docTermCountMatrix.getColumnDimension();
        final double[] tfVector = new double[NTerm];
        for (int i = 0; i < docTermCountMatrix.getRowDimension(); ++i) {
            tfVector[i] = 0.0;
            for (int j = 0; j < docTermCountMatrix.getColumnDimension(); ++j) {
                final double[] array = tfVector;
                final int n = i;
                array[n] += ((docTermCountMatrix.getEntry(i, j) > 0.0) ? 1 : 0);
            }
        }
        final Matrix res = docTermCountMatrix.copy();
        for (int k = 0; k < docTermCountMatrix.getRowDimension(); ++k) {
            for (int l = 0; l < docTermCountMatrix.getColumnDimension(); ++l) {
                if (res.getEntry(k, l) > 0.0) {
                    res.setEntry(k, l, res.getEntry(k, l) * ((tfVector[k] > 0.0) ? Math.log(NDoc / tfVector[k]) : 0.0));
                }
            }
        }
        return res;
    }
    
    public static Matrix normalizeByColumns(final Matrix A) {
        final double[] AA = full(sqrt(sum(A.times(A)))).getPr();
        final Matrix res = A.copy();
        final int M = A.getRowDimension();
        final int N = A.getColumnDimension();
        if (res instanceof DenseMatrix) {
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            for (int i = 0; i < M; ++i) {
                resRow = resData[i];
                for (int j = 0; j < N; ++j) {
                    final double[] array = resRow;
                    final int n = j;
                    array[n] /= AA[j];
                }
            }
        }
        else if (res instanceof SparseMatrix) {
            final int[] jc = ((SparseMatrix)res).getJc();
            final double[] pr = ((SparseMatrix)res).getPr();
            double v = 0.0;
            for (int k = 0; k < N; ++k) {
                v = AA[k];
                for (int l = jc[k]; l < jc[k + 1]; ++l) {
                    final double[] array2 = pr;
                    final int n2 = l;
                    array2[n2] /= v;
                }
            }
        }
        return res;
    }
    
    public static int[] randperm(final int n) {
        final int[] res = new int[n];
        final Set<Integer> leftSet = new TreeSet<Integer>();
        for (int i = 0; i < n; ++i) {
            leftSet.add(i);
        }
        final Random generator = new Random();
        for (int j = 0; j < n; ++j) {
            final double[] uniformDist = ArrayOperator.allocateVector(n - j, 1.0 / (n - j));
            final double rndRealScalor = generator.nextDouble();
            double sum = 0.0;
            int k = 0;
            int l = 0;
            while (k < n) {
                if (leftSet.contains(k)) {
                    sum += uniformDist[l];
                    if (rndRealScalor <= sum) {
                        res[j] = k + 1;
                        leftSet.remove(k);
                        break;
                    }
                    ++l;
                }
                ++k;
            }
        }
        return res;
    }
    
    public static int[] find(final Vector V) {
        int[] indices = null;
        if (V instanceof DenseVector) {
            final ArrayList<Integer> idxList = new ArrayList<Integer>();
            final double[] pr = ((DenseVector)V).getPr();
            double v = 0.0;
            for (int k = 0; k < V.getDim(); ++k) {
                v = pr[k];
                if (v != 0.0) {
                    idxList.add(k);
                }
            }
            final int nnz = idxList.size();
            indices = new int[nnz];
            final Iterator<Integer> idxIter = idxList.iterator();
            int cnt = 0;
            while (idxIter.hasNext()) {
                indices[cnt++] = idxIter.next();
            }
        }
        else if (V instanceof SparseVector) {
            ((SparseVector)V).clean();
            final int nnz2 = ((SparseVector)V).getNNZ();
            final int[] ir = ((SparseVector)V).getIr();
            indices = new int[nnz2];
            System.arraycopy(ir, 0, indices, 0, nnz2);
        }
        return indices;
    }
    
    public static FindResult find(final Matrix A) {
        int[] rows = null;
        int[] cols = null;
        double[] vals = null;
        if (A instanceof SparseMatrix) {
            ((SparseMatrix)A).clean();
            final int nnz = ((SparseMatrix)A).getNNZ();
            rows = new int[nnz];
            cols = new int[nnz];
            vals = new double[nnz];
            final int[] ir = ((SparseMatrix)A).getIr();
            final int[] jc = ((SparseMatrix)A).getJc();
            final double[] pr = ((SparseMatrix)A).getPr();
            int cnt = 0;
            for (int j = 0; j < A.getColumnDimension(); ++j) {
                for (int k = jc[j]; k < jc[j + 1]; ++k) {
                    rows[cnt] = ir[k];
                    cols[cnt] = j;
                    vals[cnt] = pr[k];
                    ++cnt;
                }
            }
        }
        else if (A instanceof DenseMatrix) {
            final int M = A.getRowDimension();
            final int N = A.getColumnDimension();
            final ArrayList<Integer> rowIdxList = new ArrayList<Integer>();
            final ArrayList<Integer> colIdxList = new ArrayList<Integer>();
            final ArrayList<Double> valList = new ArrayList<Double>();
            final double[][] AData = ((DenseMatrix)A).getData();
            double[] ARow = null;
            double v = 0.0;
            for (int i = 0; i < M; ++i) {
                ARow = AData[i];
                for (int l = 0; l < N; ++l) {
                    v = ARow[l];
                    if (v != 0.0) {
                        rowIdxList.add(i);
                        colIdxList.add(l);
                        valList.add(v);
                    }
                }
            }
            final int nnz2 = valList.size();
            rows = new int[nnz2];
            cols = new int[nnz2];
            vals = new double[nnz2];
            final Iterator<Integer> rowIdxIter = rowIdxList.iterator();
            final Iterator<Integer> colIdxIter = colIdxList.iterator();
            final Iterator<Double> valIter = valList.iterator();
            int cnt2 = 0;
            while (valIter.hasNext()) {
                rows[cnt2] = rowIdxIter.next();
                cols[cnt2] = colIdxIter.next();
                vals[cnt2] = valIter.next();
                ++cnt2;
            }
        }
        return new FindResult(rows, cols, vals);
    }
    
    public static Matrix exp(final Matrix A) {
        final int M = A.getRowDimension();
        final int N = A.getColumnDimension();
        final Matrix res = new DenseMatrix(M, N, 1.0);
        final double[][] resData = ((DenseMatrix)res).getData();
        double[] resRow = null;
        if (A instanceof DenseMatrix) {
            for (int i = 0; i < M; ++i) {
                resRow = resData[i];
                for (int j = 0; j < N; ++j) {
                    resRow[j] = Math.exp(resRow[j]);
                }
            }
        }
        else if (A instanceof SparseMatrix) {
            final double[] pr = ((SparseMatrix)A).getPr();
            final int[] ir = ((SparseMatrix)A).getIr();
            final int[] jc = ((SparseMatrix)A).getJc();
            for (int k = 0; k < N; ++k) {
                for (int l = jc[k]; l < jc[k + 1]; ++l) {
                    resData[ir[l]][k] = Math.exp(pr[l]);
                }
            }
        }
        return res;
    }
    
    public static Matrix getRows(final Matrix A, final int startRow, final int endRow) {
        return A.getRows(startRow, endRow);
    }
    
    public static Matrix getRows(final Matrix A, final int... selectedRows) {
        return A.getRows(selectedRows);
    }
    
    public static Matrix getColumns(final Matrix A, final int startColumn, final int endColumn) {
        Matrix res = null;
        final int nRow = A.getRowDimension();
        final int nCol = endColumn - startColumn + 1;
        if (A instanceof DenseMatrix) {
            final double[][] AData = ((DenseMatrix)A).getData();
            double[] ARow = null;
            final double[][] resData = new double[nRow][nCol];
            double[] resRow = null;
            for (int r = 0; r < nRow; ++r) {
                ARow = AData[r];
                resRow = new double[nCol];
                for (int c = startColumn, j = 0; c <= endColumn; ++c, ++j) {
                    resRow[j] = ARow[c];
                }
                resData[r] = resRow;
            }
            res = new DenseMatrix(resData);
        }
        else if (A instanceof SparseMatrix) {
            final Vector[] vectors = sparseMatrix2SparseColumnVectors(A);
            final Vector[] resVectors = new Vector[nCol];
            for (int c2 = startColumn, i = 0; c2 <= endColumn; ++c2, ++i) {
                resVectors[i] = vectors[c2];
            }
            res = sparseColumnVectors2SparseMatrix(resVectors);
        }
        return res;
    }
    
    public static Matrix getColumns(final Matrix A, final int... selectedColumns) {
        Matrix res = null;
        final int nRow = A.getRowDimension();
        final int nCol = selectedColumns.length;
        if (A instanceof DenseMatrix) {
            final double[][] AData = ((DenseMatrix)A).getData();
            double[] ARow = null;
            final double[][] resData = new double[nRow][nCol];
            double[] resRow = null;
            for (int r = 0; r < nRow; ++r) {
                ARow = AData[r];
                resRow = new double[nCol];
                for (int j = 0; j < nCol; ++j) {
                    resRow[j] = ARow[selectedColumns[j]];
                }
                resData[r] = resRow;
            }
            res = new DenseMatrix(resData);
        }
        else if (A instanceof SparseMatrix) {
            final Vector[] vectors = sparseMatrix2SparseColumnVectors(A);
            final Vector[] resVectors = new Vector[nCol];
            for (int i = 0; i < nCol; ++i) {
                resVectors[i] = vectors[selectedColumns[i]];
            }
            res = sparseColumnVectors2SparseMatrix(resVectors);
        }
        return res;
    }
    
    public static Matrix vertcat(final Matrix... As) {
        final int nM = As.length;
        int nRow = 0;
        int nCol = 0;
        for (int i = 0; i < nM; ++i) {
            if (As[i] != null) {
                nRow += As[i].getRowDimension();
                nCol = As[i].getColumnDimension();
            }
        }
        for (int i = 1; i < nM; ++i) {
            if (As[i] != null && nCol != As[i].getColumnDimension()) {
                System.err.println("Any matrix in the argument list should either be empty matrix or have the same number of columns to the others!");
            }
        }
        if (nRow == 0 || nCol == 0) {
            return null;
        }
        Matrix res = null;
        final double[][] resData = new double[nRow][];
        double[] resRow = null;
        int idx = 0;
        for (int j = 0; j < nM; ++j) {
            if (j > 0 && As[j - 1] != null) {
                idx += As[j - 1].getRowDimension();
            }
            if (As[j] != null) {
                if (As[j] instanceof DenseMatrix) {
                    final DenseMatrix A = (DenseMatrix)As[j];
                    final double[][] AData = A.getData();
                    for (int r = 0; r < A.getRowDimension(); ++r) {
                        resData[idx + r] = AData[r].clone();
                    }
                }
                else if (As[j] instanceof SparseMatrix) {
                    final SparseMatrix A2 = (SparseMatrix)As[j];
                    final double[] pr = A2.getPr();
                    final int[] ic = A2.getIc();
                    final int[] jr = A2.getJr();
                    final int[] valCSRIndices = A2.getValCSRIndices();
                    for (int r2 = 0; r2 < A2.getRowDimension(); ++r2) {
                        resRow = ArrayOperator.allocate1DArray(nCol, 0.0);
                        for (int k = jr[r2]; k < jr[r2 + 1]; ++k) {
                            resRow[ic[k]] = pr[valCSRIndices[k]];
                        }
                        resData[idx + r2] = resRow;
                    }
                }
            }
        }
        res = new DenseMatrix(resData);
        return res;
    }
    
    public static Matrix horzcat(final Matrix... As) {
        final int nM = As.length;
        int nCol = 0;
        int nRow = 0;
        for (int i = 0; i < nM; ++i) {
            if (As[i] != null) {
                nCol += As[i].getColumnDimension();
                nRow = As[i].getRowDimension();
            }
        }
        for (int i = 1; i < nM; ++i) {
            if (As[i] != null && nRow != As[i].getRowDimension()) {
                System.err.println("Any matrix in the argument list should either be empty matrix or have the same number of rows to the others!");
            }
        }
        if (nRow == 0 || nCol == 0) {
            return null;
        }
        final Matrix res = new DenseMatrix(nRow, nCol, 0.0);
        final double[][] resData = ((DenseMatrix)res).getData();
        double[] resRow = null;
        int idx = 0;
        for (int r = 0; r < nRow; ++r) {
            resRow = resData[r];
            idx = 0;
            for (int j = 0; j < nM; ++j) {
                if (j > 0 && As[j - 1] != null) {
                    idx += As[j - 1].getColumnDimension();
                }
                if (As[j] != null) {
                    if (As[j] instanceof DenseMatrix) {
                        final DenseMatrix A = (DenseMatrix)As[j];
                        System.arraycopy(A.getData()[r], 0, resRow, idx, A.getColumnDimension());
                    }
                    else if (As[j] instanceof SparseMatrix) {
                        final SparseMatrix A2 = (SparseMatrix)As[j];
                        final double[] pr = A2.getPr();
                        final int[] ic = A2.getIc();
                        final int[] jr = A2.getJr();
                        final int[] valCSRIndices = A2.getValCSRIndices();
                        for (int k = jr[r]; k < jr[r + 1]; ++k) {
                            resRow[idx + ic[k]] = pr[valCSRIndices[k]];
                        }
                    }
                }
            }
        }
        return res;
    }
    
    public static Matrix cat(final int dim, final Matrix... As) {
        Matrix res = null;
        if (dim == 1) {
            res = vertcat(As);
        }
        else if (dim == 2) {
            res = horzcat(As);
        }
        else {
            System.err.println("Specified dimension can only be either 1 or 2 currently!");
        }
        return res;
    }
    
    public static Matrix kron(final Matrix A, final Matrix B) {
        Matrix res = null;
        final int nRowLeft = A.getRowDimension();
        final int nColLeft = A.getColumnDimension();
        final int nRowRight = B.getRowDimension();
        final int nColRight = B.getColumnDimension();
        if (A instanceof DenseMatrix && B instanceof DenseMatrix) {
            res = new DenseMatrix(nRowLeft * nRowRight, nColLeft * nColRight, 0.0);
            final double[][] resData = res.getData();
            final double[][] BData = B.getData();
            for (int i = 0, rShift = 0; i < nRowLeft; ++i, rShift += nRowRight) {
                for (int j = 0, cShift = 0; j < nColLeft; ++j, cShift += nColRight) {
                    final double A_ij = A.getEntry(i, j);
                    if (A_ij != 0.0) {
                        for (int p = 0; p < nRowRight; ++p) {
                            final int r = rShift + p;
                            final double[] BRow = BData[p];
                            final double[] resRow = resData[r];
                            for (int q = 0; q < nColRight; ++q) {
                                if (BRow[q] != 0.0) {
                                    final int c = cShift + q;
                                    resRow[c] = A_ij * BRow[q];
                                }
                            }
                        }
                    }
                }
            }
        }
        else if (A instanceof DenseMatrix && B instanceof SparseMatrix) {
            final TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();
            final int[] ir2 = ((SparseMatrix)B).getIr();
            final int[] jc2 = ((SparseMatrix)B).getJc();
            final double[] pr2 = ((SparseMatrix)B).getPr();
            for (int k = 0, rShift2 = 0; k < nRowLeft; ++k, rShift2 += nRowRight) {
                for (int l = 0, cShift2 = 0; l < nColLeft; ++l, cShift2 += nColRight) {
                    final double A_ij2 = A.getEntry(k, l);
                    if (A_ij2 != 0.0) {
                        for (int j2 = 0, c2 = cShift2; j2 < nColRight; ++j2, ++c2) {
                            for (int k2 = jc2[j2]; k2 < jc2[j2 + 1]; ++k2) {
                                if (pr2[k2] != 0.0) {
                                    final int r2 = rShift2 + ir2[k2];
                                    map.put(Pair.of(r2, c2), A_ij2 * pr2[k2]);
                                }
                            }
                        }
                    }
                }
            }
            res = SparseMatrix.createSparseMatrix(map, nRowLeft * nRowRight, nColLeft * nColRight);
        }
        else if (A instanceof SparseMatrix && B instanceof DenseMatrix) {
            final TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();
            final int[] ir2 = ((SparseMatrix)A).getIr();
            final int[] jc2 = ((SparseMatrix)A).getJc();
            final double[] pr2 = ((SparseMatrix)A).getPr();
            final double[][] BData2 = B.getData();
            for (int j3 = 0, cShift3 = 0; j3 < nColLeft; ++j3, cShift3 += nColRight) {
                for (int k3 = jc2[j3]; k3 < jc2[j3 + 1]; ++k3) {
                    if (pr2[k3] != 0.0) {
                        final int rShift3 = ir2[k3] * nRowRight;
                        for (int m = 0; m < nRowRight; ++m) {
                            final double[] BRow = BData2[m];
                            final int r3 = rShift3 + m;
                            for (int j4 = 0; j4 < nColRight; ++j4) {
                                if (BRow[j4] != 0.0) {
                                    final int c = cShift3 + j4;
                                    map.put(Pair.of(r3, c), pr2[k3] * BRow[j4]);
                                }
                            }
                        }
                    }
                }
            }
            res = SparseMatrix.createSparseMatrix(map, nRowLeft * nRowRight, nColLeft * nColRight);
        }
        else if (A instanceof SparseMatrix && B instanceof SparseMatrix) {
            final TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();
            final int[] ir2 = ((SparseMatrix)A).getIr();
            final int[] jc2 = ((SparseMatrix)A).getJc();
            final double[] pr2 = ((SparseMatrix)A).getPr();
            final int[] ir3 = ((SparseMatrix)B).getIr();
            final int[] jc3 = ((SparseMatrix)B).getJc();
            final double[] pr3 = ((SparseMatrix)B).getPr();
            for (int j5 = 0, cShift4 = 0; j5 < nColLeft; ++j5, cShift4 += nColRight) {
                for (int k4 = jc2[j5]; k4 < jc2[j5 + 1]; ++k4) {
                    if (pr2[k4] != 0.0) {
                        final int rShift4 = ir2[k4] * nRowRight;
                        for (int j6 = 0, c3 = cShift4; j6 < nColRight; ++j6, ++c3) {
                            for (int k5 = jc3[j6]; k5 < jc3[j6 + 1]; ++k5) {
                                if (pr3[k5] != 0.0) {
                                    final int r4 = rShift4 + ir3[k5];
                                    map.put(Pair.of(r4, c3), pr2[k4] * pr3[k5]);
                                }
                            }
                        }
                    }
                }
            }
            res = SparseMatrix.createSparseMatrix(map, nRowLeft * nRowRight, nColLeft * nColRight);
        }
        return res;
    }
    
    public static double sumAll(final Matrix A) {
        return sum(sum(A));
    }
    
    public static Matrix diag(final Matrix A) {
        final int nRow = A.getRowDimension();
        final int nCol = A.getColumnDimension();
        Matrix res = null;
        if (nRow == 1) {
            res = new SparseMatrix(nCol, nCol);
            for (int i = 0; i < nCol; ++i) {
                res.setEntry(i, i, A.getEntry(0, i));
            }
        }
        else if (nCol == 1) {
            res = new SparseMatrix(nRow, nRow);
            for (int i = 0; i < nRow; ++i) {
                res.setEntry(i, i, A.getEntry(i, 0));
            }
        }
        else if (nRow == nCol) {
            res = new DenseMatrix(nRow, 1);
            for (int i = 0; i < nRow; ++i) {
                res.setEntry(i, 0, A.getEntry(i, i));
            }
        }
        return res;
    }
    
    public static SparseMatrix diag(final Vector V) {
        final int dim = V.getDim();
        final SparseMatrix res = new SparseMatrix(dim, dim);
        for (int i = 0; i < dim; ++i) {
            res.setEntry(i, i, V.get(i));
        }
        return res;
    }
    
    public static Matrix rdivide(final Matrix A, final double v) {
        final int nRow = A.getRowDimension();
        final int nCol = A.getColumnDimension();
        final Matrix res = A.copy();
        if (res instanceof DenseMatrix) {
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            for (int i = 0; i < nRow; ++i) {
                resRow = resData[i];
                for (int j = 0; j < nCol; ++j) {
                    final double[] array = resRow;
                    final int n = j;
                    array[n] /= v;
                }
            }
        }
        else if (res instanceof SparseMatrix) {
            final double[] pr = ((SparseMatrix)res).getPr();
            for (int k = 0; k < pr.length; ++k) {
                final double[] array2 = pr;
                final int n2 = k;
                array2[n2] /= v;
            }
        }
        return res;
    }
    
    public static Matrix rand(final int nRow, final int nCol) {
        final Random generator = new Random();
        final Matrix res = new DenseMatrix(nRow, nCol);
        for (int i = 0; i < nRow; ++i) {
            for (int j = 0; j < nCol; ++j) {
                res.setEntry(i, j, generator.nextDouble());
            }
        }
        return res;
    }
    
    public static Matrix rand(final int n) {
        return rand(n, n);
    }
    
    public static Matrix randn(final int nRow, final int nCol) {
        final Random generator = new Random();
        final Matrix res = new DenseMatrix(nRow, nCol);
        for (int i = 0; i < nRow; ++i) {
            for (int j = 0; j < nCol; ++j) {
                res.setEntry(i, j, generator.nextGaussian());
            }
        }
        return res;
    }
    
    public static Matrix randn(final int n) {
        return randn(n, n);
    }
    
    public static Matrix sign(final Matrix A) {
        final int nRow = A.getRowDimension();
        final int nCol = A.getColumnDimension();
        final Matrix res = A.copy();
        double v = 0.0;
        if (res instanceof DenseMatrix) {
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            for (int i = 0; i < nRow; ++i) {
                resRow = resData[i];
                for (int j = 0; j < nCol; ++j) {
                    v = resRow[j];
                    if (v > 0.0) {
                        resRow[j] = 1.0;
                    }
                    else if (v < 0.0) {
                        resRow[j] = -1.0;
                    }
                }
            }
        }
        else if (res instanceof SparseMatrix) {
            final double[] pr = ((SparseMatrix)res).getPr();
            for (int k = 0; k < pr.length; ++k) {
                v = pr[k];
                if (v > 0.0) {
                    pr[k] = 1.0;
                }
                else if (v < 0.0) {
                    pr[k] = -1.0;
                }
            }
        }
        return res;
    }
    
    @Deprecated
    public static Matrix l2DistanceSquare0(final Matrix X, final Matrix Y) {
        final int nX = X.getColumnDimension();
        final int nY = Y.getColumnDimension();
        Matrix dist = null;
        final Matrix part1 = columnVector2ColumnMatrix(sum(times(X, X), 1)).mtimes(ones(1, nY));
        final Matrix part2 = ones(nX, 1).mtimes(rowVector2RowMatrix(sum(times(Y, Y), 1)));
        final Matrix part3 = X.transpose().mtimes(Y).times(2.0);
        dist = part1.plus(part2).minus(part3);
        final Matrix I = lt(dist, 0.0);
        logicalIndexingAssignment(dist, I, 0.0);
        return dist;
    }
    
    public static Matrix l2DistanceSquare(final Matrix X, final Matrix Y) {
        final int nX = X.getRowDimension();
        final int nY = Y.getRowDimension();
        Matrix dist = null;
        final double[] XX = sum(times(X, X), 2).getPr();
        final double[] YY = sum(times(Y, Y), 2).getPr();
        dist = full(X.mtimes(Y.transpose()).times(-2.0));
        final double[][] resData = ((DenseMatrix)dist).getData();
        double[] resRow = null;
        double s = 0.0;
        double v = 0.0;
        for (int i = 0; i < nX; ++i) {
            resRow = resData[i];
            s = XX[i];
            for (int j = 0; j < nY; ++j) {
                v = resRow[j] + s + YY[j];
                if (v >= 0.0) {
                    resRow[j] = v;
                }
                else {
                    resRow[j] = 0.0;
                }
            }
        }
        return dist;
    }
    
    public static Vector l2DistanceSquare(final Vector V, final Matrix Y) {
        final int nY = Y.getRowDimension();
        Vector dist = null;
        final double XX = sum(times(V, V));
        final double[] YY = sum(times(Y, Y), 2).getPr();
        dist = full(Y.operate(V).times(-2.0));
        final double[] pr = ((DenseVector)dist).getPr();
        double v = 0.0;
        for (int j = 0; j < nY; ++j) {
            v = pr[j] + XX + YY[j];
            if (v >= 0.0) {
                pr[j] = v;
            }
            else {
                pr[j] = 0.0;
            }
        }
        return dist;
    }
    
    public static Matrix l2DistanceSquareByColumns(final Matrix X, final Matrix Y) {
        final int nX = X.getColumnDimension();
        final int nY = Y.getColumnDimension();
        Matrix dist = null;
        final double[] XX = sum(times(X, X)).getPr();
        final double[] YY = sum(times(Y, Y)).getPr();
        dist = full(X.transpose().mtimes(Y).times(-2.0));
        final double[][] resData = ((DenseMatrix)dist).getData();
        double[] resRow = null;
        double s = 0.0;
        double v = 0.0;
        for (int i = 0; i < nX; ++i) {
            resRow = resData[i];
            s = XX[i];
            for (int j = 0; j < nY; ++j) {
                v = resRow[j] + s + YY[j];
                if (v >= 0.0) {
                    resRow[j] = v;
                }
                else {
                    resRow[j] = 0.0;
                }
            }
        }
        return dist;
    }
    
    public static Matrix l2DistanceSquare(final Vector[] X, final Vector[] Y) {
        final int nX = X.length;
        final int nY = Y.length;
        Matrix dist = null;
        final double[] XX = new double[nX];
        Vector V = null;
        for (int i = 0; i < nX; ++i) {
            V = X[i];
            XX[i] = sum(V.times(V));
        }
        final double[] YY = new double[nY];
        for (int j = 0; j < nY; ++j) {
            V = Y[j];
            YY[j] = sum(V.times(V));
        }
        final double[][] resData = ArrayOperator.allocate2DArray(nX, nY, 0.0);
        double[] resRow = null;
        double s = 0.0;
        double v = 0.0;
        for (int k = 0; k < nX; ++k) {
            resRow = resData[k];
            V = X[k];
            s = XX[k];
            for (int l = 0; l < nY; ++l) {
                v = s + YY[l] - 2.0 * innerProduct(V, Y[l]);
                if (v >= 0.0) {
                    resRow[l] = v;
                }
                else {
                    resRow[l] = 0.0;
                }
            }
        }
        dist = new DenseMatrix(resData);
        return dist;
    }
    
    public static Matrix l2DistanceByColumns(final Matrix X, final Matrix Y) {
        return sqrt(l2DistanceSquareByColumns(X, Y));
    }
    
    public static Matrix l2Distance(final Matrix X, final Matrix Y) {
        return sqrt(l2DistanceSquare(X, Y));
    }
    
    public static Vector l2Distance(final Vector V, final Matrix Y) {
        return sqrt(l2DistanceSquare(V, Y));
    }
    
    public static Matrix l2Distance(final Vector[] X, final Vector[] Y) {
        return sqrt(l2DistanceSquare(X, Y));
    }
    
    public static Vector dotDivide(final double v, final Vector V) {
        final int dim = V.getDim();
        final DenseVector res = (DenseVector)full(V).copy();
        final double[] pr = res.getPr();
        for (int k = 0; k < dim; ++k) {
            pr[k] = v / pr[k];
        }
        return res;
    }
    
    public static Vector sqrt(final Vector V) {
        final int dim = V.getDim();
        final Vector res = V.copy();
        if (res instanceof DenseVector) {
            final double[] pr = ((DenseVector)res).getPr();
            for (int k = 0; k < dim; ++k) {
                pr[k] = Math.sqrt(pr[k]);
            }
        }
        else if (res instanceof SparseVector) {
            final double[] pr = ((SparseVector)res).getPr();
            for (int nnz = ((SparseVector)res).getNNZ(), i = 0; i < nnz; ++i) {
                pr[i] = Math.sqrt(pr[i]);
            }
        }
        return res;
    }
    
    public static Matrix sqrt(final Matrix A) {
        final int nRow = A.getRowDimension();
        final int nCol = A.getColumnDimension();
        final Matrix res = A.copy();
        if (res instanceof DenseMatrix) {
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            for (int i = 0; i < nRow; ++i) {
                resRow = resData[i];
                for (int j = 0; j < nCol; ++j) {
                    resRow[j] = Math.sqrt(resRow[j]);
                }
            }
        }
        else if (res instanceof SparseMatrix) {
            final double[] pr = ((SparseMatrix)res).getPr();
            for (int k = 0; k < pr.length; ++k) {
                pr[k] = Math.sqrt(pr[k]);
            }
        }
        return res;
    }
    
    public static Matrix rowVector2RowMatrix(final Vector V) {
        Matrix res = null;
        if (V instanceof DenseVector) {
            res = denseRowVectors2DenseMatrix(new Vector[] { V });
        }
        else if (V instanceof SparseVector) {
            res = sparseRowVectors2SparseMatrix(new Vector[] { V });
        }
        return res;
    }
    
    public static Matrix columnVector2ColumnMatrix(final Vector V) {
        Matrix res = null;
        if (V instanceof DenseVector) {
            res = denseColumnVectors2DenseMatrix(new Vector[] { V });
        }
        else if (V instanceof SparseVector) {
            res = sparseColumnVectors2SparseMatrix(new Vector[] { V });
        }
        return res;
    }
    
    public static Matrix denseRowVectors2DenseMatrix(final Vector[] Vs) {
        final int M = Vs.length;
        final double[][] resData = new double[M][];
        for (int i = 0; i < M; ++i) {
            resData[i] = ((DenseVector)Vs[i]).getPr().clone();
        }
        return new DenseMatrix(resData);
    }
    
    public static Matrix denseColumnVectors2DenseMatrix(final Vector[] Vs) {
        final int N = Vs.length;
        final int M = Vs[0].getDim();
        final double[][] resData = new double[M][];
        for (int i = 0; i < M; ++i) {
            resData[i] = new double[N];
        }
        for (int j = 0; j < N; ++j) {
            final double[] column = ((DenseVector)Vs[j]).getPr().clone();
            for (int k = 0; k < M; ++k) {
                resData[k][j] = column[k];
            }
        }
        return new DenseMatrix(resData);
    }
    
    public static Vector[] denseMatrix2DenseRowVectors(final Matrix A) {
        final int M = A.getRowDimension();
        final Vector[] res = new Vector[M];
        if (A instanceof DenseMatrix) {
            final double[][] AData = ((DenseMatrix)A).getData();
            for (int i = 0; i < M; ++i) {
                res[i] = new DenseVector(AData[i]);
            }
        }
        else {
            System.err.println("The input matrix should be a dense matrix.");
            Utility.exit(1);
        }
        return res;
    }
    
    public static Vector[] denseMatrix2DenseColumnVectors(final Matrix A) {
        final int N = A.getColumnDimension();
        final int M = A.getRowDimension();
        final Vector[] res = new Vector[N];
        if (A instanceof DenseMatrix) {
            final double[][] AData = ((DenseMatrix)A).getData();
            for (int j = 0; j < N; ++j) {
                final double[] column = new double[M];
                for (int i = 0; i < M; ++i) {
                    column[i] = AData[i][j];
                }
                res[j] = new DenseVector(column);
            }
        }
        else {
            System.err.println("The input matrix should be a dense matrix.");
            Utility.exit(1);
        }
        return res;
    }
    
    public static Matrix zeros(final int nRow, final int nCol) {
        if (nRow == 0 || nCol == 0) {
            return null;
        }
        return new DenseMatrix(nRow, nCol, 0.0);
    }
    
    public static Matrix zeros(final int[] size) {
        if (size.length != 2) {
            System.err.println("Input vector should have two elements!");
        }
        return zeros(size[0], size[1]);
    }
    
    public static Matrix zeros(final int n) {
        return zeros(n, n);
    }
    
    public static Matrix ones(final int nRow, final int nCol) {
        if (nRow == 0 || nCol == 0) {
            return null;
        }
        return new DenseMatrix(nRow, nCol, 1.0);
    }
    
    public static Matrix ones(final int[] size) {
        if (size.length != 2) {
            System.err.println("Input vector should have two elements!");
        }
        return ones(size[0], size[1]);
    }
    
    public static Matrix ones(final int n) {
        return ones(n, n);
    }
    
    public static double det(final Matrix A) {
        final int M = A.getRowDimension();
        final int N = A.getColumnDimension();
        if (M != N) {
            System.err.println("Input should be a square matrix.");
            Utility.exit(1);
        }
        return new LUDecomposition(A).det();
    }
    
    public static Matrix inv(final Matrix A) {
        final int M = A.getRowDimension();
        final int N = A.getColumnDimension();
        if (M != N) {
            System.err.println("Input should be a square matrix.");
            Utility.exit(1);
        }
        final LUDecomposition LUDecomp = new LUDecomposition(A);
        if (LUDecomp.det() == 0.0) {
            System.err.println("The input matrix is not invertible.");
            Utility.exit(1);
        }
        return LUDecomp.inverse();
    }
    
    public static double[] sort(final Vector V) {
        return sort(V, "ascend");
    }
    
    public static double[] sort(final Vector V, final String order) {
        double[] indices = null;
        final int dim = V.getDim();
        if (V instanceof DenseVector) {
            final double[] pr = ((DenseVector)V).getPr();
            indices = new double[dim];
            for (int k = 0; k < dim; ++k) {
                indices[k] = k;
            }
            ArrayOperator.quickSort(pr, indices, 0, dim - 1, order);
        }
        else if (V instanceof SparseVector) {
            final double[] pr = ((SparseVector)V).getPr();
            final int[] ir = ((SparseVector)V).getIr();
            final int nnz = ((SparseVector)V).getNNZ();
            indices = new double[dim];
            int insertionPos = nnz;
            if (order.equalsIgnoreCase("ascend")) {
                for (int i = 0; i < nnz; ++i) {
                    if (pr[i] >= 0.0) {
                        --insertionPos;
                    }
                }
            }
            else if (order.equalsIgnoreCase("descend")) {
                for (int i = 0; i < nnz; ++i) {
                    if (pr[i] <= 0.0) {
                        --insertionPos;
                    }
                }
            }
            int lastIdx = -1;
            int currentIdx = 0;
            int cnt = insertionPos;
            for (int j = 0; j < nnz; ++j) {
                currentIdx = ir[j];
                for (int idx = lastIdx + 1; idx < currentIdx; ++idx) {
                    indices[cnt++] = idx;
                }
                lastIdx = currentIdx;
            }
            for (int idx2 = lastIdx + 1; idx2 < dim; ++idx2) {
                indices[cnt++] = idx2;
            }
            ArrayOperator.quickSort(pr, ir, 0, nnz - 1, order);
            for (int j = 0; j < insertionPos; ++j) {
                indices[j] = ir[j];
            }
            for (int j = insertionPos; j < nnz; ++j) {
                indices[j + dim - nnz] = ir[j];
            }
            for (int j = 0; j < nnz; ++j) {
                if (j < insertionPos) {
                    ir[j] = j;
                }
                else {
                    ir[j] = j + dim - nnz;
                }
            }
        }
        return indices;
    }
    
    @Deprecated
    public static double[] sort1(final Vector V, final String order) {
        double[] indices = null;
        final int dim = V.getDim();
        if (V instanceof DenseVector) {
            final double[] pr = ((DenseVector)V).getPr();
            indices = new double[dim];
            for (int k = 0; k < dim; ++k) {
                indices[k] = k;
            }
            ArrayOperator.quickSort(pr, indices, 0, dim - 1, order);
        }
        else if (V instanceof SparseVector) {
            final double[] pr = ((SparseVector)V).getPr();
            final int[] ir = ((SparseVector)V).getIr();
            final int nnz = ((SparseVector)V).getNNZ();
            final int[] ir_ori = ir.clone();
            ArrayOperator.quickSort(pr, ir, 0, nnz - 1, order);
            int insertionPos = nnz;
            if (order.equalsIgnoreCase("ascend")) {
                for (int i = 0; i < nnz; ++i) {
                    if (pr[i] >= 0.0) {
                        insertionPos = i;
                        break;
                    }
                }
            }
            else if (order.equalsIgnoreCase("descend")) {
                for (int i = 0; i < nnz; ++i) {
                    if (pr[i] <= 0.0) {
                        insertionPos = i;
                        break;
                    }
                }
            }
            indices = new double[dim];
            for (int i = 0; i < insertionPos; ++i) {
                indices[i] = ir[i];
            }
            int lastIdx = -1;
            int currentIdx = 0;
            int cnt = insertionPos;
            for (int j = 0; j < nnz; ++j) {
                currentIdx = ir_ori[j];
                for (int idx = lastIdx + 1; idx < currentIdx; ++idx) {
                    indices[cnt++] = idx;
                }
                lastIdx = currentIdx;
            }
            for (int idx2 = lastIdx + 1; idx2 < dim; ++idx2) {
                indices[cnt++] = idx2;
            }
            for (int j = insertionPos; j < nnz; ++j) {
                indices[j + dim - nnz] = ir[j];
            }
            for (int j = 0; j < nnz; ++j) {
                if (j < insertionPos) {
                    ir[j] = j;
                }
                else {
                    ir[j] = j + dim - nnz;
                }
            }
        }
        return indices;
    }
    
    @Deprecated
    public static double[] sort0(final Vector V, final String order) {
        double[] indices = null;
        final int dim = V.getDim();
        if (V instanceof DenseVector) {
            final double[] pr = ((DenseVector)V).getPr();
            indices = new double[dim];
            for (int k = 0; k < dim; ++k) {
                indices[k] = k;
            }
            ArrayOperator.quickSort(pr, indices, 0, dim - 1, order);
        }
        else if (V instanceof SparseVector) {
            final double[] pr = ((SparseVector)V).getPr();
            final int[] ir = ((SparseVector)V).getIr();
            final int nnz = ((SparseVector)V).getNNZ();
            final int[] ir_ori = ir.clone();
            if (order.equalsIgnoreCase("ascend")) {
                ArrayOperator.quickSort(pr, ir, 0, nnz - 1, order);
                int numNegatives = nnz;
                for (int i = 0; i < nnz; ++i) {
                    if (pr[i] >= 0.0) {
                        numNegatives = i;
                        break;
                    }
                }
                indices = new double[dim];
                for (int i = 0; i < numNegatives; ++i) {
                    indices[i] = ir[i];
                }
                int lastIdx = -1;
                int currentIdx = 0;
                int cnt = numNegatives;
                for (int j = 0; j < nnz; ++j) {
                    currentIdx = ir_ori[j];
                    for (int idx = lastIdx + 1; idx < currentIdx; ++idx) {
                        indices[cnt++] = idx;
                    }
                    lastIdx = currentIdx;
                }
                for (int idx2 = lastIdx + 1; idx2 < dim; ++idx2) {
                    indices[cnt++] = idx2;
                }
                for (int j = numNegatives; j < nnz; ++j) {
                    indices[j + dim - nnz] = ir[j];
                }
                for (int j = 0; j < nnz; ++j) {
                    if (j < numNegatives) {
                        ir[j] = j;
                    }
                    else {
                        ir[j] = j + dim - nnz;
                    }
                }
            }
            else if (order.equalsIgnoreCase("descend")) {
                ArrayOperator.quickSort(pr, ir, 0, nnz - 1, order);
                int numPositives = nnz;
                for (int i = 0; i < nnz; ++i) {
                    if (pr[i] <= 0.0) {
                        numPositives = i;
                        break;
                    }
                }
                indices = new double[dim];
                for (int i = 0; i < numPositives; ++i) {
                    indices[i] = ir[i];
                }
                int lastIdx = -1;
                int currentIdx = 0;
                int cnt = numPositives;
                for (int j = 0; j < nnz; ++j) {
                    currentIdx = ir_ori[j];
                    for (int idx = lastIdx + 1; idx < currentIdx; ++idx) {
                        indices[cnt++] = idx;
                    }
                    lastIdx = currentIdx;
                }
                for (int idx2 = lastIdx + 1; idx2 < dim; ++idx2) {
                    indices[cnt++] = idx2;
                }
                for (int j = numPositives; j < nnz; ++j) {
                    indices[j + dim - nnz] = ir[j];
                }
                for (int j = 0; j < nnz; ++j) {
                    if (j < numPositives) {
                        ir[j] = j;
                    }
                    else {
                        ir[j] = j + dim - nnz;
                    }
                }
            }
        }
        return indices;
    }
    
    public static Matrix[] sort(final Matrix A, final int dim, final String order) {
        if (A == null) {
            return null;
        }
        if (dim != 1 && dim != 2) {
            System.err.println("Dimension should be either 1 or 2.");
            Utility.exit(1);
        }
        if (!order.equalsIgnoreCase("ascend") && !order.equalsIgnoreCase("descend")) {
            System.err.println("Order should be either \"ascend\" or \"descend\".");
            Utility.exit(1);
        }
        final Matrix[] res = new Matrix[2];
        final int M = A.getRowDimension();
        final int N = A.getColumnDimension();
        Matrix sortedValues = null;
        Matrix sortedIndices = null;
        double[][] sortedIndexData = null;
        if (A instanceof DenseMatrix) {
            sortedValues = A.copy();
            sortedIndices = null;
            final double[][] data = ((DenseMatrix)sortedValues).getData();
            if (dim == 2) {
                sortedIndexData = new double[M][];
                double[] values = null;
                double[] indices = null;
                for (int i = 0; i < M; ++i) {
                    values = data[i];
                    indices = new double[N];
                    for (int j = 0; j < N; ++j) {
                        indices[j] = j;
                    }
                    ArrayOperator.quickSort(values, indices, 0, N - 1, order);
                    sortedIndexData[i] = indices;
                }
                sortedIndices = new DenseMatrix(sortedIndexData);
            }
            else if (dim == 1) {
                final Matrix[] res2 = sort(A.transpose(), 2, order);
                sortedValues = res2[0].transpose();
                sortedIndices = res2[1].transpose();
            }
        }
        else if (A instanceof SparseMatrix) {
            if (dim == 2) {
                final Vector[] rowVectors = sparseMatrix2SparseRowVectors(A);
                sortedIndexData = new double[M][];
                for (int k = 0; k < M; ++k) {
                    sortedIndexData[k] = sort(rowVectors[k], order);
                }
                sortedIndices = new DenseMatrix(sortedIndexData);
                sortedValues = sparseRowVectors2SparseMatrix(rowVectors);
            }
            else if (dim == 1) {
                final Matrix[] res3 = sort(A.transpose(), 2, order);
                sortedValues = res3[0].transpose();
                sortedIndices = res3[1].transpose();
            }
        }
        res[0] = sortedValues;
        res[1] = sortedIndices;
        return res;
    }
    
    public static Matrix[] sort(final Matrix A, final int dim) {
        return sort(A, dim, "ascend");
    }
    
    public static Matrix[] sort(final Matrix A, final String order) {
        return sort(A, 1, order);
    }
    
    public static Matrix[] sort(final Matrix A) {
        return sort(A, "ascend");
    }
    
    public static int[] size(final Matrix A) {
        return new int[] { A.getRowDimension(), A.getColumnDimension() };
    }
    
    public static int size(final Matrix A, final int dim) {
        if (dim == 1) {
            return A.getRowDimension();
        }
        if (dim == 2) {
            return A.getColumnDimension();
        }
        System.err.println("Dim error!");
        return 0;
    }
    
    public static Matrix max(final Matrix A, final double v) {
        if (A == null) {
            return null;
        }
        Matrix res = null;
        final int M = A.getRowDimension();
        final int N = A.getColumnDimension();
        if (A instanceof DenseMatrix) {
            res = A.copy();
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            for (int i = 0; i < M; ++i) {
                resRow = resData[i];
                for (int j = 0; j < N; ++j) {
                    if (resRow[j] < v) {
                        resRow[j] = v;
                    }
                }
            }
        }
        else if (A instanceof SparseMatrix) {
            if (v == 0.0) {
                return max(A, new SparseMatrix(M, N));
            }
            res = new DenseMatrix(M, N, v);
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            final int[] ic = ((SparseMatrix)A).getIc();
            final int[] jr = ((SparseMatrix)A).getJr();
            final int[] valCSRIndices = ((SparseMatrix)A).getValCSRIndices();
            final double[] pr = ((SparseMatrix)A).getPr();
            for (int k = 0; k < M; ++k) {
                resRow = resData[k];
                if (jr[k] == jr[k + 1]) {
                    if (v < 0.0) {
                        ArrayOperator.assignVector(resRow, 0.0);
                    }
                }
                else {
                    int lastColumnIdx = -1;
                    int currentColumnIdx = 0;
                    for (int l = jr[k]; l < jr[k + 1]; ++l) {
                        currentColumnIdx = ic[l];
                        if (v < 0.0) {
                            for (int c = lastColumnIdx + 1; c < currentColumnIdx; ++c) {
                                resRow[c] = 0.0;
                            }
                        }
                        resRow[currentColumnIdx] = Math.max(pr[valCSRIndices[l]], v);
                        lastColumnIdx = currentColumnIdx;
                    }
                    for (int c2 = lastColumnIdx + 1; c2 < N; ++c2) {
                        if (v < 0.0) {
                            resRow[c2] = 0.0;
                        }
                    }
                }
            }
        }
        return res;
    }
    
    public static Matrix max(final double v, final Matrix A) {
        return max(A, v);
    }
    
    public static Matrix max(final Matrix A, final Matrix B) {
        Matrix res = null;
        final int M = A.getRowDimension();
        final int N = A.getColumnDimension();
        double v = 0.0;
        if (A instanceof DenseMatrix) {
            res = A.copy();
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            if (B instanceof DenseMatrix) {
                final double[][] BData = ((DenseMatrix)B).getData();
                double[] BRow = null;
                for (int i = 0; i < M; ++i) {
                    resRow = resData[i];
                    BRow = BData[i];
                    resRow = resData[i];
                    for (int j = 0; j < N; ++j) {
                        v = BRow[j];
                        if (resRow[j] < v) {
                            resRow[j] = v;
                        }
                    }
                }
            }
            else if (B instanceof SparseMatrix) {
                final int[] ic = ((SparseMatrix)B).getIc();
                final int[] jr = ((SparseMatrix)B).getJr();
                final int[] valCSRIndices = ((SparseMatrix)B).getValCSRIndices();
                final double[] pr = ((SparseMatrix)B).getPr();
                for (int k = 0; k < M; ++k) {
                    resRow = resData[k];
                    if (jr[k] == jr[k + 1]) {
                        for (int l = 0; l < N; ++l) {
                            if (resRow[l] < 0.0) {
                                resRow[l] = 0.0;
                            }
                        }
                    }
                    else {
                        int lastColumnIdx = -1;
                        int currentColumnIdx = 0;
                        for (int m = jr[k]; m < jr[k + 1]; ++m) {
                            currentColumnIdx = ic[m];
                            for (int c = lastColumnIdx + 1; c < currentColumnIdx; ++c) {
                                if (resRow[c] < 0.0) {
                                    resRow[c] = 0.0;
                                }
                            }
                            resRow[currentColumnIdx] = Math.max(pr[valCSRIndices[m]], resRow[currentColumnIdx]);
                            lastColumnIdx = currentColumnIdx;
                        }
                        for (int c2 = lastColumnIdx + 1; c2 < N; ++c2) {
                            if (resRow[c2] < 0.0) {
                                resRow[c2] = 0.0;
                            }
                        }
                    }
                }
            }
        }
        else if (A instanceof SparseMatrix) {
            if (B instanceof DenseMatrix) {
                return max(B, A);
            }
            if (B instanceof SparseMatrix) {
                res = new SparseMatrix(M, N);
                int[] ir1 = null;
                int[] jc1 = null;
                double[] pr2 = null;
                ir1 = ((SparseMatrix)A).getIr();
                jc1 = ((SparseMatrix)A).getJc();
                pr2 = ((SparseMatrix)A).getPr();
                int[] ir2 = null;
                int[] jc2 = null;
                double[] pr3 = null;
                ir2 = ((SparseMatrix)B).getIr();
                jc2 = ((SparseMatrix)B).getJc();
                pr3 = ((SparseMatrix)B).getPr();
                int k2 = 0;
                int k3 = 0;
                int r1 = -1;
                int r2 = -1;
                int i2 = -1;
                for (int j2 = 0; j2 < N; ++j2) {
                    k2 = jc1[j2];
                    k3 = jc2[j2];
                    if (k2 != jc1[j2 + 1] || k3 != jc2[j2 + 1]) {
                        while (k2 < jc1[j2 + 1] || k3 < jc2[j2 + 1]) {
                            if (k3 == jc2[j2 + 1]) {
                                i2 = ir1[k2];
                                v = pr2[k2];
                                if (v < 0.0) {
                                    v = 0.0;
                                }
                                ++k2;
                            }
                            else if (k2 == jc1[j2 + 1]) {
                                i2 = ir2[k3];
                                v = pr3[k3];
                                if (v < 0.0) {
                                    v = 0.0;
                                }
                                ++k3;
                            }
                            else {
                                r1 = ir1[k2];
                                r2 = ir2[k3];
                                if (r1 < r2) {
                                    i2 = r1;
                                    v = pr2[k2];
                                    if (v < 0.0) {
                                        v = 0.0;
                                    }
                                    ++k2;
                                }
                                else if (r1 == r2) {
                                    i2 = r1;
                                    v = Math.max(pr2[k2], pr3[k3]);
                                    ++k2;
                                    ++k3;
                                }
                                else {
                                    i2 = r2;
                                    v = pr3[k3];
                                    if (v < 0.0) {
                                        v = 0.0;
                                    }
                                    ++k3;
                                }
                            }
                            if (v != 0.0) {
                                res.setEntry(i2, j2, v);
                            }
                        }
                    }
                }
            }
        }
        return res;
    }
    
    public static Matrix min(final Matrix A, final double v) {
        if (A == null) {
            return null;
        }
        Matrix res = null;
        final int M = A.getRowDimension();
        final int N = A.getColumnDimension();
        if (A instanceof DenseMatrix) {
            res = A.copy();
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            for (int i = 0; i < M; ++i) {
                resRow = resData[i];
                for (int j = 0; j < N; ++j) {
                    if (resRow[j] > v) {
                        resRow[j] = v;
                    }
                }
            }
        }
        else if (A instanceof SparseMatrix) {
            if (v == 0.0) {
                return min(A, new SparseMatrix(M, N));
            }
            res = new DenseMatrix(M, N, v);
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            final int[] ic = ((SparseMatrix)A).getIc();
            final int[] jr = ((SparseMatrix)A).getJr();
            final int[] valCSRIndices = ((SparseMatrix)A).getValCSRIndices();
            final double[] pr = ((SparseMatrix)A).getPr();
            for (int k = 0; k < M; ++k) {
                resRow = resData[k];
                if (jr[k] == jr[k + 1]) {
                    if (v > 0.0) {
                        ArrayOperator.assignVector(resRow, 0.0);
                    }
                }
                else {
                    int lastColumnIdx = -1;
                    int currentColumnIdx = 0;
                    for (int l = jr[k]; l < jr[k + 1]; ++l) {
                        currentColumnIdx = ic[l];
                        if (v > 0.0) {
                            for (int c = lastColumnIdx + 1; c < currentColumnIdx; ++c) {
                                resRow[c] = 0.0;
                            }
                        }
                        resRow[currentColumnIdx] = Math.min(pr[valCSRIndices[l]], v);
                        lastColumnIdx = currentColumnIdx;
                    }
                    for (int c2 = lastColumnIdx + 1; c2 < N; ++c2) {
                        if (v > 0.0) {
                            resRow[c2] = 0.0;
                        }
                    }
                }
            }
        }
        return res;
    }
    
    public static Matrix min(final double v, final Matrix A) {
        return min(A, v);
    }
    
    public static Matrix min(final Matrix A, final Matrix B) {
        Matrix res = null;
        final int M = A.getRowDimension();
        final int N = A.getColumnDimension();
        double v = 0.0;
        if (A instanceof DenseMatrix) {
            res = A.copy();
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            if (B instanceof DenseMatrix) {
                final double[][] BData = ((DenseMatrix)B).getData();
                double[] BRow = null;
                for (int i = 0; i < M; ++i) {
                    resRow = resData[i];
                    BRow = BData[i];
                    resRow = resData[i];
                    for (int j = 0; j < N; ++j) {
                        v = BRow[j];
                        if (resRow[j] > v) {
                            resRow[j] = v;
                        }
                    }
                }
            }
            else if (B instanceof SparseMatrix) {
                final int[] ic = ((SparseMatrix)B).getIc();
                final int[] jr = ((SparseMatrix)B).getJr();
                final int[] valCSRIndices = ((SparseMatrix)B).getValCSRIndices();
                final double[] pr = ((SparseMatrix)B).getPr();
                for (int k = 0; k < M; ++k) {
                    resRow = resData[k];
                    if (jr[k] == jr[k + 1]) {
                        for (int l = 0; l < N; ++l) {
                            if (resRow[l] > 0.0) {
                                resRow[l] = 0.0;
                            }
                        }
                    }
                    else {
                        int lastColumnIdx = -1;
                        int currentColumnIdx = 0;
                        for (int m = jr[k]; m < jr[k + 1]; ++m) {
                            currentColumnIdx = ic[m];
                            for (int c = lastColumnIdx + 1; c < currentColumnIdx; ++c) {
                                if (resRow[c] > 0.0) {
                                    resRow[c] = 0.0;
                                }
                            }
                            resRow[currentColumnIdx] = Math.min(pr[valCSRIndices[m]], resRow[currentColumnIdx]);
                            lastColumnIdx = currentColumnIdx;
                        }
                        for (int c2 = lastColumnIdx + 1; c2 < N; ++c2) {
                            if (resRow[c2] > 0.0) {
                                resRow[c2] = 0.0;
                            }
                        }
                    }
                }
            }
        }
        else if (A instanceof SparseMatrix) {
            if (B instanceof DenseMatrix) {
                return min(B, A);
            }
            if (B instanceof SparseMatrix) {
                res = new SparseMatrix(M, N);
                int[] ir1 = null;
                int[] jc1 = null;
                double[] pr2 = null;
                ir1 = ((SparseMatrix)A).getIr();
                jc1 = ((SparseMatrix)A).getJc();
                pr2 = ((SparseMatrix)A).getPr();
                int[] ir2 = null;
                int[] jc2 = null;
                double[] pr3 = null;
                ir2 = ((SparseMatrix)B).getIr();
                jc2 = ((SparseMatrix)B).getJc();
                pr3 = ((SparseMatrix)B).getPr();
                int k2 = 0;
                int k3 = 0;
                int r1 = -1;
                int r2 = -1;
                int i2 = -1;
                for (int j2 = 0; j2 < N; ++j2) {
                    k2 = jc1[j2];
                    k3 = jc2[j2];
                    if (k2 != jc1[j2 + 1] || k3 != jc2[j2 + 1]) {
                        while (k2 < jc1[j2 + 1] || k3 < jc2[j2 + 1]) {
                            if (k3 == jc2[j2 + 1]) {
                                i2 = ir1[k2];
                                v = pr2[k2];
                                if (v > 0.0) {
                                    v = 0.0;
                                }
                                ++k2;
                            }
                            else if (k2 == jc1[j2 + 1]) {
                                i2 = ir2[k3];
                                v = pr3[k3];
                                if (v > 0.0) {
                                    v = 0.0;
                                }
                                ++k3;
                            }
                            else {
                                r1 = ir1[k2];
                                r2 = ir2[k3];
                                if (r1 < r2) {
                                    i2 = r1;
                                    v = pr2[k2];
                                    if (v > 0.0) {
                                        v = 0.0;
                                    }
                                    ++k2;
                                }
                                else if (r1 == r2) {
                                    i2 = r1;
                                    v = Math.min(pr2[k2], pr3[k3]);
                                    ++k2;
                                    ++k3;
                                }
                                else {
                                    i2 = r2;
                                    v = pr3[k3];
                                    if (v > 0.0) {
                                        v = 0.0;
                                    }
                                    ++k3;
                                }
                            }
                            if (v != 0.0) {
                                res.setEntry(i2, j2, v);
                            }
                        }
                    }
                }
            }
        }
        return res;
    }
    
    public static Vector max(final Vector V, final double v) {
        Vector res = null;
        final int dim = V.getDim();
        if (V instanceof DenseVector) {
            res = V.copy();
            final double[] pr = ((DenseVector)res).getPr();
            for (int k = 0; k < dim; ++k) {
                if (pr[k] < v) {
                    pr[k] = v;
                }
            }
        }
        else if (V instanceof SparseVector) {
            if (v != 0.0) {
                res = new DenseVector(dim, v);
                final double[] prRes = ((DenseVector)res).getPr();
                final int[] ir = ((SparseVector)V).getIr();
                final double[] pr2 = ((SparseVector)V).getPr();
                final int nnz = ((SparseVector)V).getNNZ();
                int lastIdx = -1;
                int currentIdx = -1;
                for (int i = 0; i < nnz; ++i) {
                    currentIdx = ir[i];
                    for (int idx = lastIdx + 1; idx < currentIdx; ++idx) {
                        if (v < 0.0) {
                            prRes[idx] = 0.0;
                        }
                    }
                    if (v < pr2[i]) {
                        prRes[currentIdx] = pr2[i];
                    }
                    lastIdx = currentIdx;
                }
                for (int idx2 = lastIdx + 1; idx2 < dim; ++idx2) {
                    if (v < 0.0) {
                        prRes[idx2] = 0.0;
                    }
                }
            }
            else {
                res = V.copy();
                final double[] pr = ((SparseVector)res).getPr();
                for (int nnz2 = ((SparseVector)res).getNNZ(), j = 0; j < nnz2; ++j) {
                    if (pr[j] < 0.0) {
                        pr[j] = 0.0;
                    }
                }
                ((SparseVector)res).clean();
            }
        }
        return res;
    }
    
    public static Vector max(final double v, final Vector V) {
        return max(V, v);
    }
    
    public static Vector max(final Vector U, final Vector V) {
        Vector res = null;
        final int dim = U.getDim();
        double v = 0.0;
        if (U instanceof DenseVector) {
            res = U.copy();
            final double[] prRes = ((DenseVector)res).getPr();
            if (V instanceof DenseVector) {
                final double[] prV = ((DenseVector)V).getPr();
                for (int k = 0; k < dim; ++k) {
                    v = prV[k];
                    if (prRes[k] < v) {
                        prRes[k] = v;
                    }
                }
            }
            else if (V instanceof SparseVector) {
                final int[] ir = ((SparseVector)V).getIr();
                final double[] pr = ((SparseVector)V).getPr();
                final int nnz = ((SparseVector)V).getNNZ();
                int lastIdx = -1;
                int currentIdx = -1;
                for (int i = 0; i < nnz; ++i) {
                    currentIdx = ir[i];
                    for (int idx = lastIdx + 1; idx < currentIdx; ++idx) {
                        if (prRes[idx] < 0.0) {
                            prRes[idx] = 0.0;
                        }
                    }
                    v = pr[i];
                    if (prRes[currentIdx] < v) {
                        prRes[currentIdx] = v;
                    }
                    lastIdx = currentIdx;
                }
                for (int idx2 = lastIdx + 1; idx2 < dim; ++idx2) {
                    if (prRes[idx2] < 0.0) {
                        prRes[idx2] = 0.0;
                    }
                }
            }
        }
        else if (U instanceof SparseVector) {
            if (V instanceof DenseVector) {
                return max(V, U);
            }
            if (V instanceof SparseVector) {
                res = new SparseVector(dim);
                final int[] ir2 = ((SparseVector)V).getIr();
                final double[] pr2 = ((SparseVector)V).getPr();
                final int nnz2 = ((SparseVector)V).getNNZ();
                final int[] ir3 = ((SparseVector)U).getIr();
                final double[] pr3 = ((SparseVector)U).getPr();
                final int nnz3 = ((SparseVector)U).getNNZ();
                if (nnz2 != 0 || nnz3 != 0) {
                    int k2 = 0;
                    int k3 = 0;
                    int r1 = 0;
                    int r2 = 0;
                    int j = -1;
                    while (k2 < nnz2 || k3 < nnz3) {
                        if (k3 == nnz3) {
                            j = ir2[k2];
                            v = pr2[k2];
                            if (v < 0.0) {
                                v = 0.0;
                            }
                            ++k2;
                        }
                        else if (k2 == nnz2) {
                            j = ir3[k3];
                            v = pr3[k3];
                            if (v < 0.0) {
                                v = 0.0;
                            }
                            ++k3;
                        }
                        else {
                            r1 = ir2[k2];
                            r2 = ir3[k3];
                            if (r1 < r2) {
                                j = r1;
                                v = pr2[k2];
                                if (v < 0.0) {
                                    v = 0.0;
                                }
                                ++k2;
                            }
                            else if (r1 == r2) {
                                j = r1;
                                v = Math.max(pr2[k2], pr3[k3]);
                                ++k2;
                                ++k3;
                            }
                            else {
                                j = r2;
                                v = pr3[k3];
                                if (v < 0.0) {
                                    v = 0.0;
                                }
                                ++k3;
                            }
                        }
                        if (v != 0.0) {
                            res.set(j, v);
                        }
                    }
                }
            }
        }
        return res;
    }
    
    public static Vector min(final Vector V, final double v) {
        Vector res = null;
        final int dim = V.getDim();
        if (V instanceof DenseVector) {
            res = V.copy();
            final double[] pr = ((DenseVector)res).getPr();
            for (int k = 0; k < dim; ++k) {
                if (pr[k] > v) {
                    pr[k] = v;
                }
            }
        }
        else if (V instanceof SparseVector) {
            if (v != 0.0) {
                res = new DenseVector(dim, v);
                final double[] prRes = ((DenseVector)res).getPr();
                final int[] ir = ((SparseVector)V).getIr();
                final double[] pr2 = ((SparseVector)V).getPr();
                final int nnz = ((SparseVector)V).getNNZ();
                int lastIdx = -1;
                int currentIdx = -1;
                for (int i = 0; i < nnz; ++i) {
                    currentIdx = ir[i];
                    for (int idx = lastIdx + 1; idx < currentIdx; ++idx) {
                        if (v > 0.0) {
                            prRes[idx] = 0.0;
                        }
                    }
                    if (v > pr2[i]) {
                        prRes[currentIdx] = pr2[i];
                    }
                    lastIdx = currentIdx;
                }
                for (int idx2 = lastIdx + 1; idx2 < dim; ++idx2) {
                    if (v > 0.0) {
                        prRes[idx2] = 0.0;
                    }
                }
            }
            else {
                res = V.copy();
                final double[] pr = ((SparseVector)res).getPr();
                for (int nnz2 = ((SparseVector)res).getNNZ(), j = 0; j < nnz2; ++j) {
                    if (pr[j] > 0.0) {
                        pr[j] = 0.0;
                    }
                }
                ((SparseVector)res).clean();
            }
        }
        return res;
    }
    
    public static Vector min(final double v, final Vector V) {
        return min(V, v);
    }
    
    public static Vector min(final Vector U, final Vector V) {
        Vector res = null;
        final int dim = U.getDim();
        double v = 0.0;
        if (U instanceof DenseVector) {
            res = U.copy();
            final double[] prRes = ((DenseVector)res).getPr();
            if (V instanceof DenseVector) {
                final double[] prV = ((DenseVector)V).getPr();
                for (int k = 0; k < dim; ++k) {
                    v = prV[k];
                    if (prRes[k] > v) {
                        prRes[k] = v;
                    }
                }
            }
            else if (V instanceof SparseVector) {
                final int[] ir = ((SparseVector)V).getIr();
                final double[] pr = ((SparseVector)V).getPr();
                final int nnz = ((SparseVector)V).getNNZ();
                int lastIdx = -1;
                int currentIdx = -1;
                for (int i = 0; i < nnz; ++i) {
                    currentIdx = ir[i];
                    for (int idx = lastIdx + 1; idx < currentIdx; ++idx) {
                        if (prRes[idx] > 0.0) {
                            prRes[idx] = 0.0;
                        }
                    }
                    v = pr[i];
                    if (prRes[currentIdx] > v) {
                        prRes[currentIdx] = v;
                    }
                    lastIdx = currentIdx;
                }
                for (int idx2 = lastIdx + 1; idx2 < dim; ++idx2) {
                    if (prRes[idx2] > 0.0) {
                        prRes[idx2] = 0.0;
                    }
                }
            }
        }
        else if (U instanceof SparseVector) {
            if (V instanceof DenseVector) {
                return max(V, U);
            }
            if (V instanceof SparseVector) {
                res = new SparseVector(dim);
                final int[] ir2 = ((SparseVector)V).getIr();
                final double[] pr2 = ((SparseVector)V).getPr();
                final int nnz2 = ((SparseVector)V).getNNZ();
                final int[] ir3 = ((SparseVector)U).getIr();
                final double[] pr3 = ((SparseVector)U).getPr();
                final int nnz3 = ((SparseVector)U).getNNZ();
                if (nnz2 != 0 || nnz3 != 0) {
                    int k2 = 0;
                    int k3 = 0;
                    int r1 = 0;
                    int r2 = 0;
                    int j = -1;
                    while (k2 < nnz2 || k3 < nnz3) {
                        if (k3 == nnz3) {
                            j = ir2[k2];
                            v = pr2[k2];
                            if (v > 0.0) {
                                v = 0.0;
                            }
                            ++k2;
                        }
                        else if (k2 == nnz2) {
                            j = ir3[k3];
                            v = pr3[k3];
                            if (v > 0.0) {
                                v = 0.0;
                            }
                            ++k3;
                        }
                        else {
                            r1 = ir2[k2];
                            r2 = ir3[k3];
                            if (r1 < r2) {
                                j = r1;
                                v = pr2[k2];
                                if (v > 0.0) {
                                    v = 0.0;
                                }
                                ++k2;
                            }
                            else if (r1 == r2) {
                                j = r1;
                                v = Math.min(pr2[k2], pr3[k3]);
                                ++k2;
                                ++k3;
                            }
                            else {
                                j = r2;
                                v = pr3[k3];
                                if (v > 0.0) {
                                    v = 0.0;
                                }
                                ++k3;
                            }
                        }
                        if (v != 0.0) {
                            res.set(j, v);
                        }
                    }
                }
            }
        }
        return res;
    }
    
    public static Matrix logicalIndexing(final Matrix A, final Matrix B) {
        final int nA = A.getColumnDimension();
        final int dA = A.getRowDimension();
        final int nB = B.getColumnDimension();
        final int dB = B.getRowDimension();
        if (nA != nB || dA != dB) {
            System.err.println("The input matrices should have same size!");
            System.exit(1);
        }
        final ArrayList<Double> vals = new ArrayList<Double>();
        for (int j = 0; j < nA; ++j) {
            for (int i = 0; i < dA; ++i) {
                final double b = B.getEntry(i, j);
                if (b == 1.0) {
                    vals.add(A.getEntry(i, j));
                }
                else if (b != 0.0) {
                    System.err.println("Elements of the logical matrix should be either 1 or 0!");
                }
            }
        }
        final Double[] Data = new Double[vals.size()];
        vals.toArray(Data);
        final double[] data = new double[vals.size()];
        for (int k = 0; k < vals.size(); ++k) {
            data[k] = Data[k];
        }
        if (data.length != 0) {
            return new DenseMatrix(data, 1);
        }
        return null;
    }
    
    public static int[] linearIndexing(final int[] V, final int[] indices) {
        if (indices == null || indices.length == 0) {
            return null;
        }
        final int[] res = new int[indices.length];
        for (int i = 0; i < indices.length; ++i) {
            res[i] = V[indices[i]];
        }
        return res;
    }
    
    public static double[] linearIndexing(final double[] V, final int[] indices) {
        if (indices == null || indices.length == 0) {
            return null;
        }
        final double[] res = new double[indices.length];
        for (int i = 0; i < indices.length; ++i) {
            res[i] = V[indices[i]];
        }
        return res;
    }
    
    public static Matrix linearIndexing(final Matrix A, final int[] indices) {
        if (indices == null || indices.length == 0) {
            return null;
        }
        Matrix res = null;
        if (A instanceof DenseMatrix) {
            res = new DenseMatrix(indices.length, 1);
        }
        else {
            res = new SparseMatrix(indices.length, 1);
        }
        final int nRow = A.getRowDimension();
        int r = -1;
        int c = -1;
        int index = -1;
        for (int i = 0; i < indices.length; ++i) {
            index = indices[i];
            r = index % nRow;
            c = index / nRow;
            res.setEntry(i, 0, A.getEntry(r, c));
        }
        return res;
    }
    
    public static void linearIndexingAssignment(final Matrix A, final int[] idx, final Matrix V) {
        if (V == null) {
            return;
        }
        final int nV = V.getColumnDimension();
        final int dV = V.getRowDimension();
        if (nV != 1) {
            System.err.println("Assignment matrix should be a column matrix!");
        }
        if (idx.length != dV) {
            System.err.println("Assignment with different number of elements!");
        }
        final int nRow = A.getRowDimension();
        int r = -1;
        int c = -1;
        int index = -1;
        for (int i = 0; i < idx.length; ++i) {
            index = idx[i];
            r = index % nRow;
            c = index / nRow;
            A.setEntry(r, c, V.getEntry(i, 0));
        }
    }
    
    public static void linearIndexingAssignment(final Matrix A, final int[] idx, final double v) {
        final int nRow = A.getRowDimension();
        int r = -1;
        int c = -1;
        int index = -1;
        for (int i = 0; i < idx.length; ++i) {
            index = idx[i];
            r = index % nRow;
            c = index / nRow;
            A.setEntry(r, c, v);
        }
    }
    
    public static void logicalIndexingAssignment(final Matrix A, final Matrix B, final double v) {
        final int nA = A.getColumnDimension();
        final int dA = A.getRowDimension();
        final int nB = B.getColumnDimension();
        final int dB = B.getRowDimension();
        if (nA != nB || dA != dB) {
            System.err.println("The input matrices for logical indexing should have same size!");
            System.exit(1);
        }
        if (B instanceof SparseMatrix) {
            final int[] ir = ((SparseMatrix)B).getIr();
            final int[] jc = ((SparseMatrix)B).getJc();
            final double[] pr = ((SparseMatrix)B).getPr();
            for (int j = 0; j < nB; ++j) {
                for (int k = jc[j]; k < jc[j + 1]; ++k) {
                    final double b = pr[k];
                    if (b == 1.0) {
                        A.setEntry(ir[k], j, v);
                    }
                    else if (b != 0.0) {
                        Printer.err("Elements of the logical matrix should be either 1 or 0!");
                    }
                }
            }
        }
        else if (B instanceof DenseMatrix) {
            final double[][] BData = ((DenseMatrix)B).getData();
            double[] BRow = null;
            for (int i = 0; i < dA; ++i) {
                BRow = BData[i];
                for (int j = 0; j < nA; ++j) {
                    final double b = BRow[j];
                    if (b == 1.0) {
                        A.setEntry(i, j, v);
                    }
                    else if (b != 0.0) {
                        System.err.println("Elements of the logical matrix should be either 1 or 0!");
                    }
                }
            }
        }
    }
    
    public static void logicalIndexingAssignment(final Matrix A, final Matrix B, final Matrix V) {
        final int nA = A.getColumnDimension();
        final int dA = A.getRowDimension();
        final int nB = B.getColumnDimension();
        final int dB = B.getRowDimension();
        if (nA != nB || dA != dB) {
            System.err.println("The input matrices for logical indexing should have same size!");
            System.exit(1);
        }
        if (V == null) {
            return;
        }
        final int nV = V.getColumnDimension();
        final int dV = V.getRowDimension();
        if (nV != 1) {
            System.err.println("Assignment matrix should be a column matrix!");
            Utility.exit(1);
        }
        int cnt = 0;
        if (B instanceof SparseMatrix) {
            final int[] ir = ((SparseMatrix)B).getIr();
            final int[] jc = ((SparseMatrix)B).getJc();
            final double[] pr = ((SparseMatrix)B).getPr();
            for (int j = 0; j < nB; ++j) {
                for (int k = jc[j]; k < jc[j + 1]; ++k) {
                    final double b = pr[k];
                    if (b == 1.0) {
                        A.setEntry(ir[k], j, V.getEntry(cnt++, 0));
                    }
                    else if (b != 0.0) {
                        Printer.err("Elements of the logical matrix should be either 1 or 0!");
                    }
                }
            }
        }
        else if (B instanceof DenseMatrix) {
            final double[][] BData = ((DenseMatrix)B).getData();
            for (int i = 0; i < nA; ++i) {
                for (int l = 0; l < dA; ++l) {
                    final double b = BData[l][i];
                    if (b == 1.0) {
                        A.setEntry(l, i, V.getEntry(cnt++, 0));
                    }
                    else if (b != 0.0) {
                        System.err.println("Elements of the logical matrix should be either 1 or 0!");
                    }
                }
            }
        }
        if (cnt != dV) {
            System.err.println("Assignment with different number of elements!");
        }
    }
    
    public static Matrix subplus(final Matrix A) {
        final Matrix res = A.copy();
        final int M = res.getRowDimension();
        final int N = res.getColumnDimension();
        if (res instanceof DenseMatrix) {
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            for (int i = 0; i < M; ++i) {
                resRow = resData[i];
                for (int j = 0; j < N; ++j) {
                    if (resRow[j] < 0.0) {
                        resRow[j] = 0.0;
                    }
                }
            }
        }
        else if (res instanceof SparseMatrix) {
            final double[] pr = ((SparseMatrix)res).getPr();
            for (int nnz = ((SparseMatrix)res).getNNZ(), k = 0; k < nnz; ++k) {
                if (pr[k] < 0.0) {
                    pr[k] = 0.0;
                }
            }
            ((SparseMatrix)res).clean();
        }
        return res;
    }
    
    public static int fix(final double x) {
        if (x > 0.0) {
            return (int)Math.floor(x);
        }
        return (int)Math.ceil(x);
    }
    
    public static Matrix isnan(final Matrix A) {
        final Matrix res = A.copy();
        final int M = res.getRowDimension();
        final int N = res.getColumnDimension();
        if (res instanceof DenseMatrix) {
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            for (int i = 0; i < M; ++i) {
                resRow = resData[i];
                for (int j = 0; j < N; ++j) {
                    if (Double.isNaN(resRow[j])) {
                        resRow[j] = 1.0;
                    }
                    else {
                        resRow[j] = 0.0;
                    }
                }
            }
        }
        else if (res instanceof SparseMatrix) {
            final double[] pr = ((SparseMatrix)res).getPr();
            for (int nnz = ((SparseMatrix)res).getNNZ(), k = 0; k < nnz; ++k) {
                if (Double.isNaN(pr[k])) {
                    pr[k] = 1.0;
                }
                else {
                    pr[k] = 0.0;
                }
            }
            ((SparseMatrix)res).clean();
        }
        return res;
    }
    
    public static Matrix isinf(final Matrix A) {
        final Matrix res = A.copy();
        final int M = res.getRowDimension();
        final int N = res.getColumnDimension();
        if (res instanceof DenseMatrix) {
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            for (int i = 0; i < M; ++i) {
                resRow = resData[i];
                for (int j = 0; j < N; ++j) {
                    if (Double.isInfinite(resRow[j])) {
                        resRow[j] = 1.0;
                    }
                    else {
                        resRow[j] = 0.0;
                    }
                }
            }
        }
        else if (res instanceof SparseMatrix) {
            final double[] pr = ((SparseMatrix)res).getPr();
            for (int nnz = ((SparseMatrix)res).getNNZ(), k = 0; k < nnz; ++k) {
                if (Double.isInfinite(pr[k])) {
                    pr[k] = 1.0;
                }
                else {
                    pr[k] = 0.0;
                }
            }
            ((SparseMatrix)res).clean();
        }
        return res;
    }
    
    public static Matrix or(final Matrix A, final Matrix B) {
        final Matrix res = A.plus(B);
        final int M = res.getRowDimension();
        final int N = res.getColumnDimension();
        if (res instanceof DenseMatrix) {
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            for (int i = 0; i < M; ++i) {
                resRow = resData[i];
                for (int j = 0; j < N; ++j) {
                    if (resRow[j] > 1.0) {
                        resRow[j] = 1.0;
                    }
                }
            }
        }
        else if (res instanceof SparseMatrix) {
            final double[] pr = ((SparseMatrix)res).getPr();
            for (int nnz = ((SparseMatrix)res).getNNZ(), k = 0; k < nnz; ++k) {
                if (pr[k] > 1.0) {
                    pr[k] = 1.0;
                }
            }
        }
        return res;
    }
    
    public static Matrix and(final Matrix A, final Matrix B) {
        final Matrix res = A.times(B);
        return res;
    }
    
    public static Matrix not(final Matrix A) {
        final Matrix res = minus(1.0, A);
        return res;
    }
    
    public static Matrix ne(final Matrix X, final Matrix Y) {
        final Matrix res = X.minus(Y);
        final int M = res.getRowDimension();
        final int N = res.getColumnDimension();
        if (res instanceof DenseMatrix) {
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            for (int i = 0; i < M; ++i) {
                resRow = resData[i];
                for (int j = 0; j < N; ++j) {
                    if (resRow[j] != 0.0) {
                        resRow[j] = 1.0;
                    }
                }
            }
        }
        else if (res instanceof SparseMatrix) {
            final double[] pr = ((SparseMatrix)res).getPr();
            for (int nnz = ((SparseMatrix)res).getNNZ(), k = 0; k < nnz; ++k) {
                if (pr[k] != 0.0) {
                    pr[k] = 1.0;
                }
            }
        }
        return res;
    }
    
    public static Matrix ne(final Matrix X, final double x) {
        final Matrix res = X.minus(x);
        final int M = res.getRowDimension();
        final int N = res.getColumnDimension();
        final double[][] resData = ((DenseMatrix)res).getData();
        double[] resRow = null;
        for (int i = 0; i < M; ++i) {
            resRow = resData[i];
            for (int j = 0; j < N; ++j) {
                if (resRow[j] != 0.0) {
                    resRow[j] = 1.0;
                }
            }
        }
        return res;
    }
    
    public static Matrix ne(final double x, final Matrix X) {
        final Matrix res = X.minus(x);
        final int M = res.getRowDimension();
        final int N = res.getColumnDimension();
        final double[][] resData = ((DenseMatrix)res).getData();
        double[] resRow = null;
        for (int i = 0; i < M; ++i) {
            resRow = resData[i];
            for (int j = 0; j < N; ++j) {
                if (resRow[j] != 0.0) {
                    resRow[j] = 1.0;
                }
            }
        }
        return res;
    }
    
    public static Matrix eq(final Matrix X, final Matrix Y) {
        return minus(1.0, ne(X, Y));
    }
    
    public static Matrix eq(final Matrix X, final double x) {
        final Matrix res = X.minus(x);
        final int M = res.getRowDimension();
        final int N = res.getColumnDimension();
        final double[][] resData = ((DenseMatrix)res).getData();
        double[] resRow = null;
        for (int i = 0; i < M; ++i) {
            resRow = resData[i];
            for (int j = 0; j < N; ++j) {
                if (resRow[j] != 0.0) {
                    resRow[j] = 0.0;
                }
                else {
                    resRow[j] = 1.0;
                }
            }
        }
        return res;
    }
    
    public static Matrix eq(final double x, final Matrix X) {
        return eq(X, x);
    }
    
    public static Matrix ge(final Matrix X, final double x) {
        final Matrix res = X.minus(x);
        final int M = res.getRowDimension();
        final int N = res.getColumnDimension();
        final double[][] resData = ((DenseMatrix)res).getData();
        double[] resRow = null;
        for (int i = 0; i < M; ++i) {
            resRow = resData[i];
            for (int j = 0; j < N; ++j) {
                if (resRow[j] >= 0.0) {
                    resRow[j] = 1.0;
                }
                else {
                    resRow[j] = 0.0;
                }
            }
        }
        return res;
    }
    
    public static Matrix ge(final double x, final Matrix X) {
        final Matrix res = minus(x, X);
        final int M = res.getRowDimension();
        final int N = res.getColumnDimension();
        final double[][] resData = ((DenseMatrix)res).getData();
        double[] resRow = null;
        for (int i = 0; i < M; ++i) {
            resRow = resData[i];
            for (int j = 0; j < N; ++j) {
                if (resRow[j] >= 0.0) {
                    resRow[j] = 1.0;
                }
                else {
                    resRow[j] = 0.0;
                }
            }
        }
        return res;
    }
    
    public static Matrix ge(final Matrix X, final Matrix Y) {
        final Matrix res = full(X.minus(Y));
        final int M = res.getRowDimension();
        final int N = res.getColumnDimension();
        final double[][] resData = ((DenseMatrix)res).getData();
        double[] resRow = null;
        for (int i = 0; i < M; ++i) {
            resRow = resData[i];
            for (int j = 0; j < N; ++j) {
                if (resRow[j] >= 0.0) {
                    resRow[j] = 1.0;
                }
                else {
                    resRow[j] = 0.0;
                }
            }
        }
        return res;
    }
    
    public static Matrix le(final Matrix X, final double x) {
        return ge(x, X);
    }
    
    public static Matrix le(final double x, final Matrix X) {
        return ge(X, x);
    }
    
    public static Matrix le(final Matrix X, final Matrix Y) {
        return ge(Y, X);
    }
    
    public static Matrix gt(final Matrix X, final double x) {
        final Matrix res = X.minus(x);
        final int M = res.getRowDimension();
        final int N = res.getColumnDimension();
        final double[][] resData = ((DenseMatrix)res).getData();
        double[] resRow = null;
        for (int i = 0; i < M; ++i) {
            resRow = resData[i];
            for (int j = 0; j < N; ++j) {
                if (resRow[j] > 0.0) {
                    resRow[j] = 1.0;
                }
                else {
                    resRow[j] = 0.0;
                }
            }
        }
        return res;
    }
    
    public static Matrix gt(final double x, final Matrix X) {
        final Matrix res = minus(x, X);
        final int M = res.getRowDimension();
        final int N = res.getColumnDimension();
        final double[][] resData = ((DenseMatrix)res).getData();
        double[] resRow = null;
        for (int i = 0; i < M; ++i) {
            resRow = resData[i];
            for (int j = 0; j < N; ++j) {
                if (resRow[j] > 0.0) {
                    resRow[j] = 1.0;
                }
                else {
                    resRow[j] = 0.0;
                }
            }
        }
        return res;
    }
    
    public static Matrix gt(final Matrix X, final Matrix Y) {
        final Matrix res = full(X.minus(Y));
        final int M = res.getRowDimension();
        final int N = res.getColumnDimension();
        final double[][] resData = ((DenseMatrix)res).getData();
        double[] resRow = null;
        for (int i = 0; i < M; ++i) {
            resRow = resData[i];
            for (int j = 0; j < N; ++j) {
                if (resRow[j] > 0.0) {
                    resRow[j] = 1.0;
                }
                else {
                    resRow[j] = 0.0;
                }
            }
        }
        return res;
    }
    
    public static Matrix lt(final Matrix X, final double x) {
        return gt(x, X);
    }
    
    public static Matrix lt(final double x, final Matrix X) {
        return gt(X, x);
    }
    
    public static Matrix lt(final Matrix X, final Matrix Y) {
        return gt(Y, X);
    }
    
    public static Matrix abs(final Matrix A) {
        final int nRow = A.getRowDimension();
        final int nCol = A.getColumnDimension();
        final Matrix res = A.copy();
        if (res instanceof DenseMatrix) {
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            for (int i = 0; i < nRow; ++i) {
                resRow = resData[i];
                for (int j = 0; j < nCol; ++j) {
                    resRow[j] = Math.abs(resRow[j]);
                }
            }
        }
        else if (res instanceof SparseMatrix) {
            final double[] pr = ((SparseMatrix)res).getPr();
            for (int k = 0; k < pr.length; ++k) {
                pr[k] = Math.abs(pr[k]);
            }
        }
        return res;
    }
    
    public static Vector abs(final Vector V) {
        final Vector res = V.copy();
        if (res instanceof DenseVector) {
            final double[] pr = ((DenseVector)res).getPr();
            for (int k = 0; k < res.getDim(); ++k) {
                pr[k] = Math.abs(pr[k]);
            }
        }
        else if (res instanceof SparseVector) {
            final double[] pr = ((SparseVector)res).getPr();
            for (int nnz = ((SparseVector)res).getNNZ(), i = 0; i < nnz; ++i) {
                pr[i] = Math.abs(pr[i]);
            }
        }
        return res;
    }
    
    public static Matrix pow(final Matrix A, final double p) {
        final int nRow = A.getRowDimension();
        final int nCol = A.getColumnDimension();
        final Matrix res = A.copy();
        if (res instanceof DenseMatrix) {
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            for (int i = 0; i < nRow; ++i) {
                resRow = resData[i];
                for (int j = 0; j < nCol; ++j) {
                    resRow[j] = Math.pow(resRow[j], p);
                }
            }
        }
        else if (res instanceof SparseMatrix) {
            final double[] pr = ((SparseMatrix)res).getPr();
            for (int k = 0; k < pr.length; ++k) {
                pr[k] = Math.pow(pr[k], p);
            }
        }
        return res;
    }
    
    public static Vector pow(final Vector V, final double p) {
        final Vector res = V.copy();
        if (res instanceof DenseVector) {
            final double[] pr = ((DenseVector)res).getPr();
            for (int k = 0; k < res.getDim(); ++k) {
                pr[k] = Math.pow(pr[k], p);
            }
        }
        else if (res instanceof SparseVector) {
            final double[] pr = ((SparseVector)res).getPr();
            for (int nnz = ((SparseVector)res).getNNZ(), i = 0; i < nnz; ++i) {
                pr[i] = Math.pow(pr[i], p);
            }
        }
        return res;
    }
    
    public static double[] max(final Vector V) {
        final double[] res = new double[2];
        double maxVal = Double.NEGATIVE_INFINITY;
        double maxIdx = -1.0;
        double v = 0.0;
        if (V instanceof DenseVector) {
            final double[] pr = ((DenseVector)V).getPr();
            for (int k = 0; k < pr.length; ++k) {
                v = pr[k];
                if (maxVal < v) {
                    maxVal = v;
                    maxIdx = k;
                }
            }
        }
        else if (V instanceof SparseVector) {
            final int[] ir = ((SparseVector)V).getIr();
            final double[] pr2 = ((SparseVector)V).getPr();
            final int nnz = ((SparseVector)V).getNNZ();
            final int dim = V.getDim();
            if (nnz == 0) {
                maxVal = 0.0;
                maxIdx = 0.0;
            }
            else {
                int lastIdx = -1;
                int currentIdx = 0;
                for (int i = 0; i < nnz; ++i) {
                    currentIdx = ir[i];
                    final int j = lastIdx + 1;
                    if (j < currentIdx && maxVal < 0.0) {
                        maxVal = 0.0;
                        maxIdx = j;
                    }
                    v = pr2[i];
                    if (maxVal < v) {
                        maxVal = v;
                        maxIdx = currentIdx;
                    }
                    lastIdx = currentIdx;
                }
                final int l = lastIdx + 1;
                if (l < dim && maxVal < 0.0) {
                    maxVal = 0.0;
                    maxIdx = l;
                }
            }
        }
        res[0] = maxVal;
        res[1] = maxIdx;
        return res;
    }
    
    public static Vector[] max(final Matrix A) {
        return max(A, 1);
    }
    
    public static double[][] max(final double[][] A, final int dim) {
        final double[][] res = new double[2][];
        final int M = A.length;
        final int N = A[0].length;
        double maxVal = 0.0;
        int maxIdx = -1;
        double v = 0.0;
        double[] maxValues = null;
        maxValues = ArrayOperator.allocate1DArray(N, Double.NEGATIVE_INFINITY);
        double[] maxIndices = null;
        maxIndices = ArrayOperator.allocate1DArray(N, 0.0);
        double[] ARow = null;
        if (dim == 1) {
            maxValues = ArrayOperator.allocate1DArray(N, Double.NEGATIVE_INFINITY);
            maxIndices = ArrayOperator.allocate1DArray(N, 0.0);
            for (int i = 0; i < M; ++i) {
                ARow = A[i];
                for (int j = 0; j < N; ++j) {
                    v = ARow[j];
                    if (maxValues[j] < v) {
                        maxValues[j] = v;
                        maxIndices[j] = i;
                    }
                }
            }
        }
        else if (dim == 2) {
            maxValues = ArrayOperator.allocate1DArray(M, Double.NEGATIVE_INFINITY);
            maxIndices = ArrayOperator.allocate1DArray(M, 0.0);
            for (int i = 0; i < M; ++i) {
                ARow = A[i];
                maxVal = ARow[0];
                maxIdx = 0;
                for (int j = 1; j < N; ++j) {
                    v = ARow[j];
                    if (maxVal < v) {
                        maxVal = v;
                        maxIdx = j;
                    }
                }
                maxValues[i] = maxVal;
                maxIndices[i] = maxIdx;
            }
        }
        res[0] = maxValues;
        res[1] = maxIndices;
        return res;
    }
    
    public static Vector[] max(final Matrix A, final int dim) {
        final Vector[] res = new DenseVector[2];
        final int M = A.getRowDimension();
        final int N = A.getColumnDimension();
        double maxVal = 0.0;
        int maxIdx = -1;
        double v = 0.0;
        double[] maxValues = null;
        maxValues = ArrayOperator.allocate1DArray(N, Double.NEGATIVE_INFINITY);
        double[] maxIndices = null;
        maxIndices = ArrayOperator.allocate1DArray(N, 0.0);
        if (A instanceof DenseMatrix) {
            final double[][] AData = ((DenseMatrix)A).getData();
            double[] ARow = null;
            if (dim == 1) {
                maxValues = ArrayOperator.allocate1DArray(N, Double.NEGATIVE_INFINITY);
                maxIndices = ArrayOperator.allocate1DArray(N, 0.0);
                for (int i = 0; i < M; ++i) {
                    ARow = AData[i];
                    for (int j = 0; j < N; ++j) {
                        v = ARow[j];
                        if (maxValues[j] < v) {
                            maxValues[j] = v;
                            maxIndices[j] = i;
                        }
                    }
                }
            }
            else if (dim == 2) {
                maxValues = ArrayOperator.allocate1DArray(M, Double.NEGATIVE_INFINITY);
                maxIndices = ArrayOperator.allocate1DArray(M, 0.0);
                for (int i = 0; i < M; ++i) {
                    ARow = AData[i];
                    maxVal = ARow[0];
                    maxIdx = 0;
                    for (int j = 1; j < N; ++j) {
                        v = ARow[j];
                        if (maxVal < v) {
                            maxVal = v;
                            maxIdx = j;
                        }
                    }
                    maxValues[i] = maxVal;
                    maxIndices[i] = maxIdx;
                }
            }
        }
        else if (A instanceof SparseMatrix) {
            final double[] pr = ((SparseMatrix)A).getPr();
            if (dim == 1) {
                maxValues = ArrayOperator.allocate1DArray(N, Double.NEGATIVE_INFINITY);
                maxIndices = ArrayOperator.allocate1DArray(N, 0.0);
                final int[] ir = ((SparseMatrix)A).getIr();
                final int[] jc = ((SparseMatrix)A).getJc();
                for (int j = 0; j < N; ++j) {
                    if (jc[j] == jc[j + 1]) {
                        maxIndices[j] = (maxValues[j] = 0.0);
                    }
                    else {
                        maxVal = Double.NEGATIVE_INFINITY;
                        maxIdx = -1;
                        int lastRowIdx = -1;
                        int currentRowIdx = 0;
                        for (int k = jc[j]; k < jc[j + 1]; ++k) {
                            currentRowIdx = ir[k];
                            final int r = lastRowIdx + 1;
                            if (r < currentRowIdx && maxVal < 0.0) {
                                maxVal = 0.0;
                                maxIdx = r;
                            }
                            v = pr[k];
                            if (maxVal < v) {
                                maxVal = v;
                                maxIdx = ir[k];
                            }
                            lastRowIdx = currentRowIdx;
                        }
                        final int r2 = lastRowIdx + 1;
                        if (r2 < M && maxVal < 0.0) {
                            maxVal = 0.0;
                            maxIdx = r2;
                        }
                        maxValues[j] = maxVal;
                        maxIndices[j] = maxIdx;
                    }
                }
            }
            else if (dim == 2) {
                maxValues = ArrayOperator.allocate1DArray(M, Double.NEGATIVE_INFINITY);
                maxIndices = ArrayOperator.allocate1DArray(M, 0.0);
                final int[] ic = ((SparseMatrix)A).getIc();
                final int[] jr = ((SparseMatrix)A).getJr();
                final int[] valCSRIndices = ((SparseMatrix)A).getValCSRIndices();
                for (int l = 0; l < M; ++l) {
                    if (jr[l] == jr[l + 1]) {
                        maxIndices[l] = (maxValues[l] = 0.0);
                    }
                    else {
                        maxVal = Double.NEGATIVE_INFINITY;
                        maxIdx = -1;
                        int lastColumnIdx = -1;
                        int currentColumnIdx = 0;
                        for (int m = jr[l]; m < jr[l + 1]; ++m) {
                            currentColumnIdx = ic[m];
                            final int c = lastColumnIdx + 1;
                            if (c < currentColumnIdx && maxVal < 0.0) {
                                maxVal = 0.0;
                                maxIdx = c;
                            }
                            v = pr[valCSRIndices[m]];
                            if (maxVal < v) {
                                maxVal = v;
                                maxIdx = ic[m];
                            }
                            lastColumnIdx = currentColumnIdx;
                        }
                        final int c2 = lastColumnIdx + 1;
                        if (c2 < N && maxVal < 0.0) {
                            maxVal = 0.0;
                            maxIdx = c2;
                        }
                        maxValues[l] = maxVal;
                        maxIndices[l] = maxIdx;
                    }
                }
            }
        }
        res[0] = new DenseVector(maxValues);
        res[1] = new DenseVector(maxIndices);
        return res;
    }
    
    public static double[] min(final Vector V) {
        final double[] res = new double[2];
        double minVal = Double.POSITIVE_INFINITY;
        double minIdx = -1.0;
        double v = 0.0;
        if (V instanceof DenseVector) {
            final double[] pr = ((DenseVector)V).getPr();
            for (int k = 0; k < pr.length; ++k) {
                v = pr[k];
                if (minVal > v) {
                    minVal = v;
                    minIdx = k;
                }
            }
        }
        else if (V instanceof SparseVector) {
            final int[] ir = ((SparseVector)V).getIr();
            final double[] pr2 = ((SparseVector)V).getPr();
            final int nnz = ((SparseVector)V).getNNZ();
            final int dim = V.getDim();
            if (nnz == 0) {
                minVal = 0.0;
                minIdx = 0.0;
            }
            else {
                int lastIdx = -1;
                int currentIdx = 0;
                for (int i = 0; i < nnz; ++i) {
                    currentIdx = ir[i];
                    final int j = lastIdx + 1;
                    if (j < currentIdx && minVal > 0.0) {
                        minVal = 0.0;
                        minIdx = j;
                    }
                    v = pr2[i];
                    if (minVal > v) {
                        minVal = v;
                        minIdx = currentIdx;
                    }
                    lastIdx = currentIdx;
                }
                final int l = lastIdx + 1;
                if (l < dim && minVal > 0.0) {
                    minVal = 0.0;
                    minIdx = l;
                }
            }
        }
        res[0] = minVal;
        res[1] = minIdx;
        return res;
    }
    
    public static Vector[] min(final Matrix A) {
        return min(A, 1);
    }
    
    public static Vector[] min(final Matrix A, final int dim) {
        final Vector[] res = new DenseVector[2];
        final int M = A.getRowDimension();
        final int N = A.getColumnDimension();
        double minVal = 0.0;
        int minIdx = -1;
        double v = 0.0;
        double[] minValues = null;
        minValues = ArrayOperator.allocate1DArray(N, Double.POSITIVE_INFINITY);
        double[] minIndices = null;
        minIndices = ArrayOperator.allocate1DArray(N, 0.0);
        if (A instanceof DenseMatrix) {
            final double[][] AData = ((DenseMatrix)A).getData();
            double[] ARow = null;
            if (dim == 1) {
                minValues = ArrayOperator.allocate1DArray(N, Double.POSITIVE_INFINITY);
                minIndices = ArrayOperator.allocate1DArray(N, 0.0);
                for (int i = 0; i < M; ++i) {
                    ARow = AData[i];
                    for (int j = 0; j < N; ++j) {
                        v = ARow[j];
                        if (minValues[j] > v) {
                            minValues[j] = v;
                            minIndices[j] = i;
                        }
                    }
                }
            }
            else if (dim == 2) {
                minValues = ArrayOperator.allocate1DArray(M, Double.POSITIVE_INFINITY);
                minIndices = ArrayOperator.allocate1DArray(M, 0.0);
                for (int i = 0; i < M; ++i) {
                    ARow = AData[i];
                    minVal = ARow[0];
                    minIdx = 0;
                    for (int j = 1; j < N; ++j) {
                        v = ARow[j];
                        if (minVal > v) {
                            minVal = v;
                            minIdx = j;
                        }
                    }
                    minValues[i] = minVal;
                    minIndices[i] = minIdx;
                }
            }
        }
        else if (A instanceof SparseMatrix) {
            final double[] pr = ((SparseMatrix)A).getPr();
            if (dim == 1) {
                minValues = ArrayOperator.allocate1DArray(N, Double.POSITIVE_INFINITY);
                minIndices = ArrayOperator.allocate1DArray(N, 0.0);
                final int[] ir = ((SparseMatrix)A).getIr();
                final int[] jc = ((SparseMatrix)A).getJc();
                for (int j = 0; j < N; ++j) {
                    if (jc[j] == jc[j + 1]) {
                        minIndices[j] = (minValues[j] = 0.0);
                    }
                    else {
                        minVal = Double.POSITIVE_INFINITY;
                        minIdx = -1;
                        int lastRowIdx = -1;
                        int currentRowIdx = 0;
                        for (int k = jc[j]; k < jc[j + 1]; ++k) {
                            currentRowIdx = ir[k];
                            final int r = lastRowIdx + 1;
                            if (r < currentRowIdx && minVal > 0.0) {
                                minVal = 0.0;
                                minIdx = r;
                            }
                            v = pr[k];
                            if (minVal > v) {
                                minVal = v;
                                minIdx = ir[k];
                            }
                            lastRowIdx = currentRowIdx;
                        }
                        final int r2 = lastRowIdx + 1;
                        if (r2 < M && minVal > 0.0) {
                            minVal = 0.0;
                            minIdx = r2;
                        }
                        minValues[j] = minVal;
                        minIndices[j] = minIdx;
                    }
                }
            }
            else if (dim == 2) {
                minValues = ArrayOperator.allocate1DArray(M, Double.POSITIVE_INFINITY);
                minIndices = ArrayOperator.allocate1DArray(M, 0.0);
                final int[] ic = ((SparseMatrix)A).getIc();
                final int[] jr = ((SparseMatrix)A).getJr();
                final int[] valCSRIndices = ((SparseMatrix)A).getValCSRIndices();
                for (int l = 0; l < M; ++l) {
                    if (jr[l] == jr[l + 1]) {
                        minIndices[l] = (minValues[l] = 0.0);
                    }
                    else {
                        minVal = Double.POSITIVE_INFINITY;
                        minIdx = -1;
                        int lastColumnIdx = -1;
                        int currentColumnIdx = 0;
                        for (int m = jr[l]; m < jr[l + 1]; ++m) {
                            currentColumnIdx = ic[m];
                            final int c = lastColumnIdx + 1;
                            if (c < currentColumnIdx && minVal > 0.0) {
                                minVal = 0.0;
                                minIdx = c;
                            }
                            v = pr[valCSRIndices[m]];
                            if (minVal > v) {
                                minVal = v;
                                minIdx = ic[m];
                            }
                            lastColumnIdx = currentColumnIdx;
                        }
                        final int c2 = lastColumnIdx + 1;
                        if (c2 < N && minVal > 0.0) {
                            minVal = 0.0;
                            minIdx = c2;
                        }
                        minValues[l] = minVal;
                        minIndices[l] = minIdx;
                    }
                }
            }
        }
        res[0] = new DenseVector(minValues);
        res[1] = new DenseVector(minIndices);
        return res;
    }
    
    public static DenseVector sum(final Matrix A) {
        return sum(A, 1);
    }
    
    public static DenseVector sum(final Matrix A, final int dim) {
        double[] sumValues = null;
        final int M = A.getRowDimension();
        final int N = A.getColumnDimension();
        double s = 0.0;
        if (A instanceof DenseMatrix) {
            final double[][] AData = ((DenseMatrix)A).getData();
            double[] ARow = null;
            if (dim == 1) {
                sumValues = ArrayOperator.allocate1DArray(N, 0.0);
                for (int i = 0; i < M; ++i) {
                    ARow = AData[i];
                    for (int j = 0; j < N; ++j) {
                        s = ARow[j];
                        if (s != 0.0) {
                            final double[] array = sumValues;
                            final int n = j;
                            array[n] += s;
                        }
                    }
                }
            }
            else if (dim == 2) {
                sumValues = ArrayOperator.allocate1DArray(M, 0.0);
                for (int i = 0; i < M; ++i) {
                    ARow = AData[i];
                    s = 0.0;
                    for (int j = 0; j < N; ++j) {
                        s += ARow[j];
                    }
                    sumValues[i] = s;
                }
            }
        }
        else if (A instanceof SparseMatrix) {
            final double[] pr = ((SparseMatrix)A).getPr();
            if (dim == 1) {
                sumValues = ArrayOperator.allocate1DArray(N, 0.0);
                final int[] jc = ((SparseMatrix)A).getJc();
                for (int k = 0; k < N; ++k) {
                    if (jc[k] == jc[k + 1]) {
                        sumValues[k] = 0.0;
                    }
                    else {
                        s = 0.0;
                        for (int l = jc[k]; l < jc[k + 1]; ++l) {
                            s += pr[l];
                        }
                        sumValues[k] = s;
                    }
                }
            }
            else if (dim == 2) {
                sumValues = ArrayOperator.allocate1DArray(M, 0.0);
                final int[] jr = ((SparseMatrix)A).getJr();
                final int[] valCSRIndices = ((SparseMatrix)A).getValCSRIndices();
                for (int m = 0; m < M; ++m) {
                    if (jr[m] == jr[m + 1]) {
                        sumValues[m] = 0.0;
                    }
                    else {
                        s = 0.0;
                        for (int k2 = jr[m]; k2 < jr[m + 1]; ++k2) {
                            s += pr[valCSRIndices[k2]];
                        }
                        sumValues[m] = s;
                    }
                }
            }
        }
        return new DenseVector(sumValues);
    }
    
    public static double sum(final Vector V) {
        double res = 0.0;
        if (V instanceof DenseVector) {
            final double[] pr = ((DenseVector)V).getPr();
            for (int k = 0; k < pr.length; ++k) {
                res += pr[k];
            }
        }
        else if (V instanceof SparseVector) {
            final double[] pr = ((SparseVector)V).getPr();
            for (int nnz = ((SparseVector)V).getNNZ(), i = 0; i < nnz; ++i) {
                res += pr[i];
            }
        }
        return res;
    }
    
    public static double norm(final Matrix A) {
        return norm(A, 2);
    }
    
    public static double norm(final Matrix A, final double type) {
        double res = 0.0;
        final int nRow = A.getRowDimension();
        final int nCol = A.getColumnDimension();
        if (nRow == 1) {
            if (Double.isInfinite(type)) {
                return max(max(abs(A), 2)[0])[0];
            }
            if (type > 0.0) {
                return Math.pow(sum(sum(pow(abs(A), type))), 1.0 / type);
            }
            System.err.printf("Error norm type: %f\n", type);
            System.exit(1);
        }
        if (nCol == 1) {
            if (Double.isInfinite(type)) {
                return max(max(abs(A), 1)[0])[0];
            }
            if (type > 0.0) {
                return Math.pow(sum(sum(pow(abs(A), type))), 1.0 / type);
            }
            System.err.printf("Error norm type: %f\n", type);
            System.exit(1);
        }
        if (type == 2.0) {
            final double eigenvalue = EigenValueDecomposition.computeEigenvalues(A.transpose().mtimes(A))[0];
            res = ((eigenvalue <= 0.0) ? 0.0 : Math.sqrt(eigenvalue));
        }
        else if (Double.isInfinite(type)) {
            res = max(sum(abs(A), 2))[0];
        }
        else if (type == 1.0) {
            res = max(sum(abs(A), 1))[0];
        }
        else {
            System.err.printf("Sorry, %f-norm of a matrix is not supported currently.\n", type);
        }
        return res;
    }
    
    public static double norm(final Matrix A, final int type) {
        return norm(A, (double)type);
    }
    
    public static double norm(final Matrix A, final String type) {
        double res = 0.0;
        if (type.compareToIgnoreCase("fro") == 0) {
            res = Math.sqrt(innerProduct(A, A));
        }
        else if (type.equals("inf")) {
            res = norm(A, Matlab.inf);
        }
        else if (type.equals("nuclear")) {
            res = ArrayOperator.sum(SingularValueDecomposition.computeSingularValues(A));
        }
        else {
            System.err.println(String.format("Norm %s unimplemented!\n", type));
        }
        return res;
    }
    
    public static double norm(final Vector V, final double p) {
        if (p == 1.0) {
            return sum(abs(V));
        }
        if (p == 2.0) {
            return Math.sqrt(innerProduct(V, V));
        }
        if (Double.isInfinite(p)) {
            return max(abs(V))[0];
        }
        if (p > 0.0) {
            return Math.pow(sum(pow(abs(V), p)), 1.0 / p);
        }
        System.err.println("Wrong argument for p");
        System.exit(1);
        return -1.0;
    }
    
    public static double norm(final Vector V) {
        return norm(V, 2.0);
    }
    
    public static double norm(final double[] V, final double p) {
        return norm(new DenseVector(V), p);
    }
    
    public static double norm(final double[] V) {
        return norm(V, 2.0);
    }
    
    public static double innerProduct(final Vector V1, final Vector V2) {
        if (V1.getDim() != V2.getDim()) {
            System.err.println("Dimension doesn't match.");
            System.exit(1);
        }
        double res = 0.0;
        double v = 0.0;
        if (V1 instanceof DenseVector) {
            final double[] pr1 = ((DenseVector)V1).getPr();
            if (V2 instanceof DenseVector) {
                if (V1 == V2) {
                    for (int k = 0; k < pr1.length; ++k) {
                        v = pr1[k];
                        res += v * v;
                    }
                    return res;
                }
                final double[] pr2 = ((DenseVector)V2).getPr();
                for (int i = 0; i < pr1.length; ++i) {
                    res += pr1[i] * pr2[i];
                }
            }
            else if (V2 instanceof SparseVector) {
                final int[] ir = ((SparseVector)V2).getIr();
                final double[] pr3 = ((SparseVector)V2).getPr();
                for (int nnz = ((SparseVector)V2).getNNZ(), j = 0; j < nnz; ++j) {
                    res += pr1[ir[j]] * pr3[j];
                }
            }
        }
        else if (V1 instanceof SparseVector) {
            if (V2 instanceof DenseVector) {
                return innerProduct(V2, V1);
            }
            if (V2 instanceof SparseVector) {
                final int[] ir2 = ((SparseVector)V1).getIr();
                final double[] pr4 = ((SparseVector)V1).getPr();
                final int nnz2 = ((SparseVector)V1).getNNZ();
                if (V1 == V2) {
                    for (int l = 0; l < nnz2; ++l) {
                        v = pr4[l];
                        res += v * v;
                    }
                    return res;
                }
                final int[] ir3 = ((SparseVector)V2).getIr();
                final double[] pr5 = ((SparseVector)V2).getPr();
                final int nnz3 = ((SparseVector)V2).getNNZ();
                int k2 = 0;
                int k3 = 0;
                int i2 = 0;
                int i3 = 0;
                while (k2 < nnz2) {
                    if (k3 >= nnz3) {
                        break;
                    }
                    i2 = ir2[k2];
                    i3 = ir3[k3];
                    if (i2 < i3) {
                        ++k2;
                    }
                    else if (i2 > i3) {
                        ++k3;
                    }
                    else {
                        res += pr4[k2] * pr5[k3];
                        ++k2;
                        ++k3;
                    }
                }
            }
        }
        return res;
    }
    
    public static double innerProduct(final Matrix A, final Matrix B) {
        double s = 0.0;
        final int M = A.getRowDimension();
        final int N = A.getColumnDimension();
        if (A instanceof DenseMatrix) {
            final double[][] AData = ((DenseMatrix)A).getData();
            if (B instanceof DenseMatrix) {
                final double[][] BData = ((DenseMatrix)B).getData();
                double[] BRow = null;
                double[] ARow = null;
                for (int i = 0; i < M; ++i) {
                    ARow = AData[i];
                    BRow = BData[i];
                    for (int j = 0; j < N; ++j) {
                        s += ARow[j] * BRow[j];
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
                for (int j = 0; j < B.getColumnDimension(); ++j) {
                    for (int k = jc[j]; k < jc[j + 1]; ++k) {
                        r = ir[k];
                        s += AData[r][j] * pr[k];
                    }
                }
            }
        }
        else if (A instanceof SparseMatrix) {
            if (B instanceof DenseMatrix) {
                return innerProduct(B, A);
            }
            if (B instanceof SparseMatrix) {
                int[] ir2 = null;
                int[] jc2 = null;
                double[] pr2 = null;
                ir2 = ((SparseMatrix)A).getIr();
                jc2 = ((SparseMatrix)A).getJc();
                pr2 = ((SparseMatrix)A).getPr();
                int[] ir3 = null;
                int[] jc3 = null;
                double[] pr3 = null;
                ir3 = ((SparseMatrix)B).getIr();
                jc3 = ((SparseMatrix)B).getJc();
                pr3 = ((SparseMatrix)B).getPr();
                int k2 = 0;
                int k3 = 0;
                int r2 = -1;
                int r3 = -1;
                double v = 0.0;
                for (int l = 0; l < N; ++l) {
                    k2 = jc2[l];
                    k3 = jc3[l];
                    if (k2 != jc2[l + 1]) {
                        if (k3 != jc3[l + 1]) {
                            while (k2 < jc2[l + 1] && k3 < jc3[l + 1]) {
                                r2 = ir2[k2];
                                r3 = ir3[k3];
                                if (r2 < r3) {
                                    ++k2;
                                }
                                else if (r2 == r3) {
                                    v = pr2[k2] * pr3[k3];
                                    ++k2;
                                    ++k3;
                                    if (v == 0.0) {
                                        continue;
                                    }
                                    s += v;
                                }
                                else {
                                    ++k3;
                                }
                            }
                        }
                    }
                }
            }
        }
        return s;
    }
    
    public static Matrix sigmoid(final Matrix A) {
        final double[][] data = full(A.copy()).getData();
        final int M = A.getRowDimension();
        final int N = A.getColumnDimension();
        double[] row_i = null;
        double old = 0.0;
        double current = 0.0;
        double max = 0.0;
        double sum = 0.0;
        double v = 0.0;
        for (int i = 0; i < M; ++i) {
            row_i = data[i];
            old = row_i[0];
            current = 0.0;
            max = old;
            for (int j = 1; j < N; ++j) {
                current = row_i[j];
                if (max < current) {
                    max = current;
                }
                old = current;
            }
            sum = 0.0;
            for (int j = 0; j < N; ++j) {
                v = Math.exp(row_i[j] - max);
                sum += v;
                row_i[j] = v;
            }
            for (int j = 0; j < N; ++j) {
                final double[] array = row_i;
                final int n = j;
                array[n] /= sum;
            }
        }
        return new DenseMatrix(data);
    }
    
    public static double max(final double[] V) {
        double max = V[0];
        for (int i = 1; i < V.length; ++i) {
            if (max < V[i]) {
                max = V[i];
            }
        }
        return max;
    }
    
    public static double max(final double[] V, final int start, final int end) {
        if (start == end) {
            return V[start];
        }
        if (start == end - 1) {
            return Math.max(V[start], V[end]);
        }
        final int middle = (start + end) / 2;
        final double leftMax = max(V, start, middle);
        final double rightMax = max(V, middle + 1, end);
        return Math.max(leftMax, rightMax);
    }
    
    public static Matrix mldivide(final Matrix A, final Matrix B) {
        return new QRDecomposition(A).solve(B);
    }
    
    public static Matrix mrdivide(final Matrix A, final Matrix B) {
        return mldivide(B.transpose(), A.transpose()).transpose();
    }
    
    public static int rank(final Matrix A) {
        return SingularValueDecomposition.rank(A);
    }
    
    public static DenseMatrix eye(final int m, final int n) {
        final double[][] res = ArrayOperator.allocate2DArray(m, n, 0.0);
        for (int len = (m >= n) ? n : m, i = 0; i < len; ++i) {
            res[i][i] = 1.0;
        }
        return new DenseMatrix(res);
    }
    
    public static DenseMatrix eye(final int n) {
        return eye(n, n);
    }
    
    public static Matrix eye(final int... size) {
        if (size.length != 2) {
            System.err.println("Input size vector should have two elements!");
        }
        return eye(size[0], size[1]);
    }
    
    public static SparseMatrix speye(final int m, final int n) {
        final TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();
        for (int len = (m >= n) ? n : m, i = 0; i < len; ++i) {
            map.put(Pair.of(i, i), 1.0);
        }
        return SparseMatrix.createSparseMatrix(map, m, n);
    }
    
    public static SparseMatrix speye(final int n) {
        return speye(n, n);
    }
    
    public static Matrix speye(final int... size) {
        if (size.length != 2) {
            System.err.println("Input size vector should have two elements!");
        }
        return speye(size[0], size[1]);
    }
    
    public static Matrix hilb(final int m, final int n) {
        final DenseMatrix A = new DenseMatrix(m, n);
        final double[][] data = A.getData();
        double[] A_i = null;
        for (int i = 0; i < m; ++i) {
            A_i = data[i];
            for (int j = 0; j < n; ++j) {
                A_i[j] = 1.0 / (i + j + 1);
            }
        }
        return A;
    }
    
    public static Vector times(final Vector V1, final Vector V2) {
        return V1.times(V2);
    }
    
    public static Vector plus(final Vector V1, final Vector V2) {
        return V1.plus(V2);
    }
    
    public static Vector minus(final Vector V1, final Vector V2) {
        return V1.minus(V2);
    }
    
    public static Matrix times(final Matrix X, final Matrix Y) {
        final int nX = X.getColumnDimension();
        final int dX = X.getRowDimension();
        final int nY = Y.getColumnDimension();
        final int dY = Y.getRowDimension();
        if (dX == 1 && nX == 1) {
            return times(X.getEntry(0, 0), Y);
        }
        if (dY == 1 && nY == 1) {
            return times(X, Y.getEntry(0, 0));
        }
        if (nX != nY || dX != dY) {
            System.err.println("The operands for Hadmada product should be of same shapes!");
        }
        return X.times(Y);
    }
    
    public static Matrix times(final Matrix A, final double v) {
        return A.times(v);
    }
    
    public static Matrix times(final double v, final Matrix A) {
        return A.times(v);
    }
    
    public static Matrix mtimes(final Matrix M1, final Matrix M2) {
        return M1.mtimes(M2);
    }
    
    public static Matrix plus(final Matrix M1, final Matrix M2) {
        return M1.plus(M2);
    }
    
    public static Matrix plus(final Matrix A, final double v) {
        return A.plus(v);
    }
    
    public static Matrix plus(final double v, final Matrix A) {
        return A.plus(v);
    }
    
    public static Matrix minus(final Matrix M1, final Matrix M2) {
        return M1.minus(M2);
    }
    
    public static Matrix minus(final Matrix A, final double v) {
        return A.minus(v);
    }
    
    public static Matrix minus(final double v, final Matrix A) {
        return uminus(A).plus(v);
    }
    
    public static void setMatrix(final Matrix X, final Matrix Y) {
        InPlaceOperator.assign(X, Y);
    }
    
    public static Matrix uminus(final Matrix A) {
        if (A == null) {
            return null;
        }
        return A.times(-1.0);
    }
    
    public static DenseVector full(final Vector V) {
        if (V instanceof SparseVector) {
            final int dim = V.getDim();
            final int[] ir = ((SparseVector)V).getIr();
            final double[] pr = ((SparseVector)V).getPr();
            final double[] values = ArrayOperator.allocateVector(dim, 0.0);
            for (int k = 0; k < ((SparseVector)V).getNNZ(); ++k) {
                values[ir[k]] = pr[k];
            }
            return new DenseVector(values);
        }
        return (DenseVector)V;
    }
    
    public static SparseVector sparse(final Vector V) {
        if (V instanceof DenseVector) {
            final double[] values = ((DenseVector)V).getPr();
            final TreeMap<Integer, Double> map = new TreeMap<Integer, Double>();
            for (int k = 0; k < values.length; ++k) {
                if (values[k] != 0.0) {
                    map.put(k, values[k]);
                }
            }
            final int nnz = map.size();
            final int[] ir = new int[nnz];
            final double[] pr = new double[nnz];
            final int dim = values.length;
            int ind = 0;
            for (final Map.Entry<Integer, Double> entry : map.entrySet()) {
                ir[ind] = entry.getKey();
                pr[ind] = entry.getValue();
                ++ind;
            }
            return new SparseVector(ir, pr, nnz, dim);
        }
        return (SparseVector)V;
    }
    
    public static DenseMatrix full(final Matrix S) {
        if (S instanceof SparseMatrix) {
            final int M = S.getRowDimension();
            final int N = S.getColumnDimension();
            final double[][] data = new double[M][];
            final int[] ic = ((SparseMatrix)S).getIc();
            final int[] jr = ((SparseMatrix)S).getJr();
            final int[] valCSRIndices = ((SparseMatrix)S).getValCSRIndices();
            final double[] pr = ((SparseMatrix)S).getPr();
            for (int i = 0; i < M; ++i) {
                final double[] rowData = ArrayOperator.allocateVector(N, 0.0);
                for (int k = jr[i]; k < jr[i + 1]; ++k) {
                    rowData[ic[k]] = pr[valCSRIndices[k]];
                }
                data[i] = rowData;
            }
            return new DenseMatrix(data);
        }
        return (DenseMatrix)S;
    }
    
    public static SparseMatrix sparse(final Matrix A) {
        if (A instanceof DenseMatrix) {
            int rIdx = 0;
            int cIdx = 0;
            int nzmax = 0;
            double value = 0.0;
            final TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();
            final int numRows = A.getRowDimension();
            final int numColumns = A.getColumnDimension();
            final double[][] data = ((DenseMatrix)A).getData();
            for (int j = 0; j < numColumns; ++j) {
                cIdx = j;
                for (int i = 0; i < numRows; ++i) {
                    rIdx = i;
                    value = data[i][j];
                    if (value != 0.0) {
                        map.put(Pair.of(cIdx, rIdx), value);
                        ++nzmax;
                    }
                }
            }
            final int[] ir = new int[nzmax];
            final int[] jc = new int[numColumns + 1];
            final double[] pr = new double[nzmax];
            int k = 0;
            jc[0] = 0;
            int currentColumn = 0;
            for (final Map.Entry<Pair<Integer, Integer>, Double> entry : map.entrySet()) {
                rIdx = entry.getKey().second;
                cIdx = entry.getKey().first;
                pr[k] = entry.getValue();
                ir[k] = rIdx;
                if (currentColumn < cIdx) {
                    jc[currentColumn + 1] = k;
                    ++currentColumn;
                }
                ++k;
            }
            while (currentColumn < numColumns) {
                jc[currentColumn + 1] = k;
                ++currentColumn;
            }
            jc[numColumns] = k;
            return SparseMatrix.createSparseMatrixByCSCArrays(ir, jc, pr, numRows, numColumns, nzmax);
        }
        return (SparseMatrix)A;
    }
    
    public static Vector[] sparseMatrix2SparseRowVectors(final Matrix S) {
        if (!(S instanceof SparseMatrix)) {
            System.err.println("SparseMatrix input is expected.");
            System.exit(1);
        }
        final int M = S.getRowDimension();
        final int N = S.getColumnDimension();
        final Vector[] Vs = new Vector[M];
        final int[] ic = ((SparseMatrix)S).getIc();
        final int[] jr = ((SparseMatrix)S).getJr();
        final double[] pr = ((SparseMatrix)S).getPr();
        final int[] valCSRIndices = ((SparseMatrix)S).getValCSRIndices();
        int[] indices = null;
        double[] values = null;
        int nnz = 0;
        final int dim = N;
        for (int r = 0; r < M; ++r) {
            nnz = jr[r + 1] - jr[r];
            indices = new int[nnz];
            values = new double[nnz];
            int idx = 0;
            for (int k = jr[r]; k < jr[r + 1]; ++k) {
                indices[idx] = ic[k];
                values[idx] = pr[valCSRIndices[k]];
                ++idx;
            }
            Vs[r] = new SparseVector(indices, values, nnz, dim);
        }
        return Vs;
    }
    
    public static Vector[] sparseMatrix2SparseColumnVectors(final Matrix S) {
        if (!(S instanceof SparseMatrix)) {
            System.err.println("SparseMatrix input is expected.");
            System.exit(1);
        }
        final int M = S.getRowDimension();
        final int N = S.getColumnDimension();
        final Vector[] Vs = new Vector[N];
        final int[] ir = ((SparseMatrix)S).getIr();
        final int[] jc = ((SparseMatrix)S).getJc();
        final double[] pr = ((SparseMatrix)S).getPr();
        int[] indices = null;
        double[] values = null;
        int nnz = 0;
        final int dim = M;
        for (int c = 0; c < N; ++c) {
            nnz = jc[c + 1] - jc[c];
            indices = new int[nnz];
            values = new double[nnz];
            int idx = 0;
            for (int k = jc[c]; k < jc[c + 1]; ++k) {
                indices[idx] = ir[k];
                values[idx] = pr[k];
                ++idx;
            }
            Vs[c] = new SparseVector(indices, values, nnz, dim);
        }
        return Vs;
    }
    
    public static Matrix sparseRowVectors2SparseMatrix(final Vector[] Vs) {
        int nnz = 0;
        final int numColumns = (Vs.length > 0) ? Vs[0].getDim() : 0;
        for (int i = 0; i < Vs.length; ++i) {
            if (!(Vs[i] instanceof SparseVector)) {
                Printer.fprintf("Vs[%d] should be a sparse vector.%n", i);
                System.exit(1);
            }
            nnz += ((SparseVector)Vs[i]).getNNZ();
            if (numColumns != Vs[i].getDim()) {
                Printer.fprintf("Vs[%d]'s dimension doesn't match.%n", i);
                System.exit(1);
            }
        }
        final int numRows = Vs.length;
        final int nzmax = nnz;
        final int[] ic = new int[nzmax];
        final int[] jr = new int[numRows + 1];
        final double[] pr = new double[nzmax];
        int rIdx = -1;
        int cIdx = -1;
        int cnt = 0;
        jr[0] = 0;
        int currentRow = 0;
        for (int j = 0; j < numRows; ++j) {
            final int[] indices = ((SparseVector)Vs[j]).getIr();
            final double[] values = ((SparseVector)Vs[j]).getPr();
            nnz = ((SparseVector)Vs[j]).getNNZ();
            for (int k = 0; k < nnz; ++k) {
                cIdx = indices[k];
                rIdx = j;
                pr[cnt] = values[k];
                ic[cnt] = cIdx;
                while (currentRow < rIdx) {
                    jr[currentRow + 1] = cnt;
                    ++currentRow;
                }
                ++cnt;
            }
        }
        while (currentRow < numRows) {
            jr[currentRow + 1] = cnt;
            ++currentRow;
        }
        return SparseMatrix.createSparseMatrixByCSRArrays(ic, jr, pr, numRows, numColumns, nzmax);
    }
    
    public static Matrix sparseColumnVectors2SparseMatrix(final Vector[] Vs) {
        int nnz = 0;
        final int numRows = (Vs.length > 0) ? Vs[0].getDim() : 0;
        for (int i = 0; i < Vs.length; ++i) {
            if (!(Vs[i] instanceof SparseVector)) {
                Printer.fprintf("Vs[%d] should be a sparse vector.%n", i);
                System.exit(1);
            }
            nnz += ((SparseVector)Vs[i]).getNNZ();
            if (numRows != Vs[i].getDim()) {
                Printer.fprintf("Vs[%d]'s dimension doesn't match.%n", i);
                System.exit(1);
            }
        }
        final int numColumns = Vs.length;
        final int nzmax = nnz;
        final int[] ir = new int[nzmax];
        final int[] jc = new int[numColumns + 1];
        final double[] pr = new double[nzmax];
        int rIdx = -1;
        int cIdx = -1;
        int k = 0;
        jc[0] = 0;
        int currentColumn = 0;
        for (int c = 0; c < numColumns; ++c) {
            final int[] indices = ((SparseVector)Vs[c]).getIr();
            final double[] values = ((SparseVector)Vs[c]).getPr();
            nnz = ((SparseVector)Vs[c]).getNNZ();
            for (int r = 0; r < nnz; ++r) {
                rIdx = indices[r];
                cIdx = c;
                pr[k] = values[r];
                ir[k] = rIdx;
                while (currentColumn < cIdx) {
                    jc[currentColumn + 1] = k;
                    ++currentColumn;
                }
                ++k;
            }
        }
        while (currentColumn < numColumns) {
            jc[currentColumn + 1] = k;
            ++currentColumn;
        }
        return SparseMatrix.createSparseMatrixByCSCArrays(ir, jc, pr, numRows, numColumns, nzmax);
    }
}
