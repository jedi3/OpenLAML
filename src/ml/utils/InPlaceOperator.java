package ml.utils;

import la.matrix.*;
import la.vector.*;
import la.vector.Vector;

import java.util.*;

public class InPlaceOperator
{
    public static void main(final String[] args) {
        final int M = 5;
        final int P = 5;
        final int N = 5;
        Matrix A = Matlab.hilb(M, P);
        A.setEntry(2, 0, 0.0);
        A.setEntry(2, 1, 0.0);
        A.setEntry(2, 2, 0.0);
        Matrix B = Matlab.hilb(P, N);
        B.setEntry(1, 0, 0.0);
        B.setEntry(1, 1, 0.0);
        B.setEntry(1, 2, 0.0);
        B.setEntry(1, 4, 0.0);
        Matrix res = new DenseMatrix(M, N);
        Printer.fprintf("A * B:%n", new Object[0]);
        Printer.disp(A.mtimes(B));
        Printer.fprintf("dense * dense:%n", new Object[0]);
        mtimes(res, A, B);
        Printer.disp(res);
        Printer.fprintf("dense * sparse:%n", new Object[0]);
        mtimes(res, A, Matlab.sparse(B));
        Printer.disp(res);
        Printer.fprintf("sparse * dense:%n", new Object[0]);
        mtimes(res, Matlab.sparse(A), B);
        Printer.disp(res);
        Printer.fprintf("sparse * sparse:%n", new Object[0]);
        mtimes(res, Matlab.sparse(A), Matlab.sparse(B));
        Printer.disp(res);
        final Matrix ATB = A.transpose().mtimes(B);
        Printer.fprintf("A' * B:%n", new Object[0]);
        Printer.disp(ATB);
        Printer.fprintf("dense' * dense:%n", new Object[0]);
        mtimes(res, A, 'T', B);
        Printer.disp(Matlab.norm(res.minus(ATB)));
        Printer.fprintf("dense' * sparse:%n", new Object[0]);
        mtimes(res, A, 'T', Matlab.sparse(B));
        Printer.disp(Matlab.norm(res.minus(ATB)));
        Printer.fprintf("sparse' * dense:%n", new Object[0]);
        mtimes(res, Matlab.sparse(A), 'T', B);
        Printer.disp(Matlab.norm(res.minus(ATB)));
        Printer.fprintf("sparse' * sparse:%n", new Object[0]);
        mtimes(res, Matlab.sparse(A), 'T', Matlab.sparse(B));
        Printer.disp(Matlab.norm(res.minus(ATB)));
        Printer.fprintf("A .* B:%n", new Object[0]);
        Printer.disp(A.times(B));
        Printer.fprintf("dense .* dense:%n", new Object[0]);
        times(res, A, B);
        Printer.disp(res);
        Printer.fprintf("dense .* sparse:%n", new Object[0]);
        times(res, A, Matlab.sparse(B));
        Printer.disp(res);
        Printer.fprintf("sparse .* dense:%n", new Object[0]);
        times(res, Matlab.sparse(A), B);
        Printer.disp(res);
        Printer.fprintf("sparse .* sparse:%n", new Object[0]);
        times(res, Matlab.sparse(A), Matlab.sparse(B));
        Printer.disp(res);
        Printer.disp(A.times(B));
        res = A.copy();
        timesAssign(res, B);
        Printer.disp(res);
        res = A.copy();
        timesAssign(res, Matlab.sparse(B));
        Printer.disp(res);
        final int[] rIndices = { 0, 1, 3, 1, 2, 2, 3, 2, 3 };
        final int[] cIndices = { 0, 0, 0, 1, 1, 2, 2, 3, 3 };
        final double[] values = { 10.0, 3.2, 3.0, 9.0, 7.0, 8.0, 8.0, 7.0, 7.0 };
        final int numRows = 5;
        final int numColumns = 5;
        final int nzmax = rIndices.length;
        A = new SparseMatrix(rIndices, cIndices, values, numRows, numColumns, nzmax);
        Printer.fprintf("A:%n", new Object[0]);
        Printer.printMatrix(A);
        Printer.fprintf("A':%n", new Object[0]);
        Printer.printMatrix(A.transpose());
        Printer.printMatrix(A.plus(A.transpose()));
        plus(res, Matlab.full(A), A.transpose());
        Printer.disp(res);
        Printer.printMatrix(A.minus(A.transpose()));
        minus(res, A, A.transpose());
        Printer.disp(res);
        B = A.transpose();
        final double a = 0.5;
        final double b = -1.5;
        Printer.fprintf("res = a * A + b * B%n", new Object[0]);
        Printer.printMatrix(A.times(a).plus(B.times(b)));
        affine(res, a, A, b, B);
        Printer.disp(res);
        final double c = 2.3;
        Printer.disp("res = A * B + c");
        res = A.mtimes(B).plus(c);
        Printer.disp(res);
        affine(res, A, B, c);
        Printer.disp(res);
        Printer.disp("*************************************");
        final int dim = 5;
        final Vector resV = new DenseVector(dim, 0.0);
        final Vector V = new SparseVector(5);
        final Vector U = new SparseVector(5);
        V.set(2, 3.5);
        V.set(4, 2.5);
        U.set(0, 0.5);
        U.set(2, 2.5);
        U.set(3, 1.5);
        double r = 0.0;
        affine(resV, a, V, b, U);
        Printer.disp(resV);
        r = Matlab.norm(resV.minus(V.times(a).plus(U.times(b))));
        Printer.disp(r);
        Printer.disp("U:");
        Printer.disp(U);
        Printer.disp("a:");
        Printer.disp(a);
        Printer.disp("V:");
        Printer.disp(V);
        Printer.disp("U -= a * V for sparse V");
        final Vector Ut = U.copy();
        minusAssign(Ut, a, V);
        Printer.disp("U:");
        Printer.disp(Ut);
        Printer.disp("U -= a * V for dense V");
        minusAssign(U, a, Matlab.full(V));
        Printer.disp("U:");
        Printer.disp(U);
    }
    
    public static void operate(final Vector res, final Matrix A, final Vector V) {
        final int dim = V.getDim();
        final int M = A.getRowDimension();
        final int N = A.getColumnDimension();
        if (N != dim) {
            Printer.err("Dimension doesn't match.");
            Utility.exit(1);
        }
        if (res instanceof DenseVector) {
            final double[] resPr = ((DenseVector)res).getPr();
            if (A instanceof DenseMatrix) {
                final double[][] data = ((DenseMatrix)A).getData();
                if (V instanceof DenseVector) {
                    ArrayOperator.operate(resPr, data, ((DenseVector)V).getPr());
                }
                else if (V instanceof SparseVector) {
                    final int[] ir = ((SparseVector)V).getIr();
                    final double[] pr = ((SparseVector)V).getPr();
                    final int nnz = ((SparseVector)V).getNNZ();
                    int idx = 0;
                    double[] row_i = null;
                    for (int i = 0; i < M; ++i) {
                        row_i = data[i];
                        double s = 0.0;
                        for (int k = 0; k < nnz; ++k) {
                            idx = ir[k];
                            s += row_i[idx] * pr[k];
                        }
                        resPr[i] = s;
                    }
                }
            }
            else if (A instanceof SparseMatrix) {
                final int[] ic = ((SparseMatrix)A).getIc();
                final int[] jr = ((SparseMatrix)A).getJr();
                final int[] valCSRIndices = ((SparseMatrix)A).getValCSRIndices();
                final double[] pr2 = ((SparseMatrix)A).getPr();
                if (V instanceof DenseVector) {
                    final double[] VPr = ((DenseVector)V).getPr();
                    double s2 = 0.0;
                    int c = 0;
                    for (int r = 0; r < M; ++r) {
                        s2 = 0.0;
                        for (int k = jr[r]; k < jr[r + 1]; ++k) {
                            c = ic[k];
                            s2 += pr2[valCSRIndices[k]] * VPr[c];
                        }
                        resPr[r] = s2;
                    }
                }
                else if (V instanceof SparseVector) {
                    final int[] ir2 = ((SparseVector)V).getIr();
                    final double[] VPr2 = ((SparseVector)V).getPr();
                    final int nnz2 = ((SparseVector)V).getNNZ();
                    double s = 0.0;
                    int kl = 0;
                    int kr = 0;
                    int cl = 0;
                    int rr = 0;
                    for (int j = 0; j < M; ++j) {
                        kl = jr[j];
                        kr = 0;
                        s = 0.0;
                        while (kl < jr[j + 1] && kr < nnz2) {
                            cl = ic[kl];
                            rr = ir2[kr];
                            if (cl < rr) {
                                ++kl;
                            }
                            else if (cl > rr) {
                                ++kr;
                            }
                            else {
                                s += pr2[valCSRIndices[kl]] * VPr2[kr];
                                ++kl;
                                ++kr;
                            }
                        }
                        resPr[j] = s;
                    }
                }
            }
        }
        else if (res instanceof SparseVector) {
            Printer.err("Sparse vector is not supported for res.");
            Utility.exit(1);
        }
    }
    
    public static void operate(final Vector res, final Vector V, final Matrix A) {
        final int dim = V.getDim();
        final int M = A.getRowDimension();
        final int N = A.getColumnDimension();
        if (M != dim) {
            Printer.err("Dimension doesn't match.");
            Utility.exit(1);
        }
        if (res instanceof DenseVector) {
            final double[] resPr = ((DenseVector)res).getPr();
            if (A instanceof DenseMatrix) {
                clear(resPr);
                final double[][] AData = ((DenseMatrix)A).getData();
                double[] ARow = null;
                if (V instanceof DenseVector) {
                    final double[] pr = ((DenseVector)V).getPr();
                    double v = 0.0;
                    for (int i = 0; i < M; ++i) {
                        ARow = AData[i];
                        v = pr[i];
                        for (int j = 0; j < N; ++j) {
                            final double[] array = resPr;
                            final int n = j;
                            array[n] += v * ARow[j];
                        }
                    }
                }
                else if (V instanceof SparseVector) {
                    final int[] ir = ((SparseVector)V).getIr();
                    final double[] pr2 = ((SparseVector)V).getPr();
                    final int nnz = ((SparseVector)V).getNNZ();
                    double v2 = 0.0;
                    for (int k = 0; k < nnz; ++k) {
                        final int l = ir[k];
                        ARow = AData[l];
                        v2 = pr2[k];
                        for (int m = 0; m < N; ++m) {
                            final double[] array2 = resPr;
                            final int n2 = m;
                            array2[n2] += v2 * ARow[m];
                        }
                    }
                }
            }
            else if (A instanceof SparseMatrix) {
                final int[] ir2 = ((SparseMatrix)A).getIr();
                final int[] jc = ((SparseMatrix)A).getJc();
                final double[] pr = ((SparseMatrix)A).getPr();
                if (V instanceof DenseVector) {
                    clear(resPr);
                    final double[] VPr = ((DenseVector)V).getPr();
                    for (int j2 = 0; j2 < N; ++j2) {
                        for (int k2 = jc[j2]; k2 < jc[j2 + 1]; ++k2) {
                            final double[] array3 = resPr;
                            final int n3 = j2;
                            array3[n3] += VPr[ir2[k2]] * pr[k2];
                        }
                    }
                }
                else if (V instanceof SparseVector) {
                    final int[] VIr = ((SparseVector)V).getIr();
                    final double[] VPr2 = ((SparseVector)V).getPr();
                    final int nnz2 = ((SparseVector)V).getNNZ();
                    double s = 0.0;
                    int k3 = 0;
                    int k4 = 0;
                    int c = 0;
                    int r = 0;
                    for (int j3 = 0; j3 < N; ++j3) {
                        k3 = 0;
                        k4 = jc[j3];
                        s = 0.0;
                        while (k4 < jc[j3 + 1] && k3 < nnz2) {
                            c = VIr[k3];
                            r = ir2[k4];
                            if (r < c) {
                                ++k4;
                            }
                            else if (r > c) {
                                ++k3;
                            }
                            else {
                                s += VPr2[k3] * pr[k4];
                                ++k3;
                                ++k4;
                            }
                        }
                        resPr[j3] = s;
                    }
                }
            }
        }
        else if (res instanceof SparseVector) {
            Printer.err("Sparse vector is not supported for res.");
            Utility.exit(1);
        }
    }
    
    public static void abs(final Matrix res, final Matrix A) {
        final int nRow = A.getRowDimension();
        final int nCol = A.getColumnDimension();
        if (res instanceof DenseMatrix) {
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            if (A instanceof DenseMatrix) {
                final double[][] AData = ((DenseMatrix)A).getData();
                double[] ARow = null;
                for (int i = 0; i < nRow; ++i) {
                    resRow = resData[i];
                    ARow = AData[i];
                    for (int j = 0; j < nCol; ++j) {
                        resRow[j] = Math.abs(ARow[j]);
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
                    clear(resRow);
                    for (int l = jr[k]; l < jr[k + 1]; ++l) {
                        resRow[ic[l]] = Math.abs(pr[valCSRIndices[l]]);
                    }
                }
            }
        }
        else {
            ((SparseMatrix)res).assignSparseMatrix((SparseMatrix)Matlab.abs(A));
        }
    }
    
    public static void subplus(final Matrix res, final Matrix A) {
        assign(res, A);
        subplusAssign(res);
    }
    
    public static void subplus(final double[] res, final double[] V) {
        for (int i = 0; i < res.length; ++i) {
            final double v = V[i];
            res[i] = ((v >= 0.0) ? v : 0.0);
        }
    }
    
    public static void subplusAssign(final double[] res) {
        subplus(res, res);
    }
    
    public static void subplusAssign(final Matrix res) {
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
    }
    
    public static void or(final Matrix res, final Matrix A, final Matrix B) {
        double[][] resData = null;
        if (res instanceof DenseMatrix) {
            resData = ((DenseMatrix)res).getData();
        }
        else {
            System.err.println("res should be a dense matrix.");
            System.exit(1);
        }
        double[][] AData = null;
        if (A instanceof DenseMatrix) {
            AData = ((DenseMatrix)A).getData();
        }
        else {
            System.err.println("A should be a dense matrix.");
            System.exit(1);
        }
        double[][] BData = null;
        if (B instanceof DenseMatrix) {
            BData = ((DenseMatrix)B).getData();
        }
        else {
            System.err.println("B should be a dense matrix.");
            System.exit(1);
        }
        final int M = res.getRowDimension();
        final int N = res.getColumnDimension();
        double[] resRow = null;
        double[] ARow = null;
        double[] BRow = null;
        for (int i = 0; i < M; ++i) {
            resRow = resData[i];
            ARow = AData[i];
            BRow = BData[i];
            for (int j = 0; j < N; ++j) {
                resRow[j] = ((ARow[j] + BRow[j] >= 1.0) ? 1 : 0);
            }
        }
    }
    
    public static void affine(final double[] res, final double a, final double[] U, final double b, final double[] V) {
        for (int i = 0; i < res.length; ++i) {
            res[i] = a * U[i] + b * V[i];
        }
    }
    
    public static void affine(final Vector res, final double a, final Vector V, final double b, final Vector U) {
        if (b == 0.0) {
            times(res, a, V);
            return;
        }
        if (b == 1.0) {
            affine(res, a, V, '+', U);
            return;
        }
        if (b == -1.0) {
            affine(res, a, V, '-', U);
            return;
        }
        if (a == 0.0) {
            times(res, b, U);
            return;
        }
        if (a == 1.0) {
            affine(res, b, U, '+', V);
            return;
        }
        if (a == -1.0) {
            affine(res, b, U, '-', V);
            return;
        }
        final int dim = res.getDim();
        if (!(res instanceof SparseVector) && res instanceof DenseVector) {
            final double[] resData = ((DenseVector)res).getPr();
            if (V instanceof DenseVector) {
                final double[] VData = ((DenseVector)V).getPr();
                if (U instanceof DenseVector) {
                    final double[] UData = ((DenseVector)U).getPr();
                    for (int i = 0; i < dim; ++i) {
                        resData[i] = a * VData[i] + b * UData[i];
                    }
                }
                else if (U instanceof SparseVector) {
                    final int[] ir = ((SparseVector)U).getIr();
                    final double[] pr = ((SparseVector)U).getPr();
                    final int nnz = ((SparseVector)U).getNNZ();
                    int lastIdx = -1;
                    int currentIdx = 0;
                    for (int k = 0; k < nnz; ++k) {
                        currentIdx = ir[k];
                        for (int j = lastIdx + 1; j < currentIdx; ++j) {
                            resData[j] = a * VData[j];
                        }
                        resData[currentIdx] = a * VData[currentIdx] + b * pr[k];
                        lastIdx = currentIdx;
                    }
                    for (int l = lastIdx + 1; l < dim; ++l) {
                        resData[l] = a * VData[l];
                    }
                }
            }
            else if (V instanceof SparseVector) {
                final int[] ir2 = ((SparseVector)V).getIr();
                final double[] pr2 = ((SparseVector)V).getPr();
                final int nnz2 = ((SparseVector)V).getNNZ();
                if (U instanceof DenseVector) {
                    final double[] UData2 = ((DenseVector)U).getPr();
                    int lastIdx = -1;
                    int currentIdx = 0;
                    for (int k = 0; k < nnz2; ++k) {
                        currentIdx = ir2[k];
                        for (int j = lastIdx + 1; j < currentIdx; ++j) {
                            resData[j] = b * UData2[j];
                        }
                        resData[currentIdx] = a * pr2[k] + b * UData2[currentIdx];
                        lastIdx = currentIdx;
                    }
                    for (int l = lastIdx + 1; l < dim; ++l) {
                        resData[l] = b * UData2[l];
                    }
                }
                else if (U instanceof SparseVector) {
                    final int[] ir3 = ((SparseVector)U).getIr();
                    final double[] pr3 = ((SparseVector)U).getPr();
                    final int nnz3 = ((SparseVector)U).getNNZ();
                    clear(resData);
                    if (nnz2 != 0 || nnz3 != 0) {
                        int k2 = 0;
                        int k3 = 0;
                        int r1 = 0;
                        int r2 = 0;
                        double v = 0.0;
                        int m = -1;
                        while (k2 < nnz2 || k3 < nnz3) {
                            if (k3 == nnz3) {
                                m = ir2[k2];
                                v = a * pr2[k2];
                                ++k2;
                            }
                            else if (k2 == nnz2) {
                                m = ir3[k3];
                                v = b * pr3[k3];
                                ++k3;
                            }
                            else {
                                r1 = ir2[k2];
                                r2 = ir3[k3];
                                if (r1 < r2) {
                                    m = r1;
                                    v = a * pr2[k2];
                                    ++k2;
                                }
                                else if (r1 == r2) {
                                    m = r1;
                                    v = a * pr2[k2] + b * pr3[k3];
                                    ++k2;
                                    ++k3;
                                }
                                else {
                                    m = r2;
                                    v = b * pr3[k3];
                                    ++k3;
                                }
                            }
                            if (v != 0.0) {
                                resData[m] = v;
                            }
                        }
                    }
                }
            }
        }
    }
    
    public static void affine(final Vector res, final double a, final Vector V, final char operator, final Vector U) {
        if (operator == '+') {
            if (a == 0.0) {
                assign(res, U);
                return;
            }
            if (a == 1.0) {
                plus(res, V, U);
                return;
            }
            if (a == -1.0) {
                minus(res, U, V);
                return;
            }
            final int dim = res.getDim();
            if (!(res instanceof SparseVector) && res instanceof DenseVector) {
                final double[] resData = ((DenseVector)res).getPr();
                if (V instanceof DenseVector) {
                    final double[] VData = ((DenseVector)V).getPr();
                    if (U instanceof DenseVector) {
                        final double[] UData = ((DenseVector)U).getPr();
                        for (int i = 0; i < dim; ++i) {
                            resData[i] = a * VData[i] + UData[i];
                        }
                    }
                    else if (U instanceof SparseVector) {
                        final int[] ir = ((SparseVector)U).getIr();
                        final double[] pr = ((SparseVector)U).getPr();
                        final int nnz = ((SparseVector)U).getNNZ();
                        int lastIdx = -1;
                        int currentIdx = 0;
                        for (int k = 0; k < nnz; ++k) {
                            currentIdx = ir[k];
                            for (int j = lastIdx + 1; j < currentIdx; ++j) {
                                resData[j] = a * VData[j];
                            }
                            resData[currentIdx] = a * VData[currentIdx] + pr[k];
                            lastIdx = currentIdx;
                        }
                        for (int l = lastIdx + 1; l < dim; ++l) {
                            resData[l] = a * VData[l];
                        }
                    }
                }
                else if (V instanceof SparseVector) {
                    final int[] ir2 = ((SparseVector)V).getIr();
                    final double[] pr2 = ((SparseVector)V).getPr();
                    final int nnz2 = ((SparseVector)V).getNNZ();
                    if (U instanceof DenseVector) {
                        final double[] UData2 = ((DenseVector)U).getPr();
                        int lastIdx = -1;
                        int currentIdx = 0;
                        for (int k = 0; k < nnz2; ++k) {
                            currentIdx = ir2[k];
                            for (int j = lastIdx + 1; j < currentIdx; ++j) {
                                resData[j] = UData2[j];
                            }
                            resData[currentIdx] = a * pr2[k] + UData2[currentIdx];
                            lastIdx = currentIdx;
                        }
                        for (int l = lastIdx + 1; l < dim; ++l) {
                            resData[l] = UData2[l];
                        }
                    }
                    else if (U instanceof SparseVector) {
                        final int[] ir3 = ((SparseVector)U).getIr();
                        final double[] pr3 = ((SparseVector)U).getPr();
                        final int nnz3 = ((SparseVector)U).getNNZ();
                        clear(resData);
                        if (nnz2 != 0 || nnz3 != 0) {
                            int k2 = 0;
                            int k3 = 0;
                            int r1 = 0;
                            int r2 = 0;
                            double v = 0.0;
                            int m = -1;
                            while (k2 < nnz2 || k3 < nnz3) {
                                if (k3 == nnz3) {
                                    m = ir2[k2];
                                    v = a * pr2[k2];
                                    ++k2;
                                }
                                else if (k2 == nnz2) {
                                    m = ir3[k3];
                                    v = pr3[k3];
                                    ++k3;
                                }
                                else {
                                    r1 = ir2[k2];
                                    r2 = ir3[k3];
                                    if (r1 < r2) {
                                        m = r1;
                                        v = a * pr2[k2];
                                        ++k2;
                                    }
                                    else if (r1 == r2) {
                                        m = r1;
                                        v = a * pr2[k2] + pr3[k3];
                                        ++k2;
                                        ++k3;
                                    }
                                    else {
                                        m = r2;
                                        v = pr3[k3];
                                        ++k3;
                                    }
                                }
                                if (v != 0.0) {
                                    resData[m] = v;
                                }
                            }
                        }
                    }
                }
            }
        }
        else if (operator == '-') {
            if (a == 0.0) {
                uminus(res, U);
                return;
            }
            if (a == 1.0) {
                minus(res, V, U);
                return;
            }
            final int dim = res.getDim();
            if (!(res instanceof SparseVector) && res instanceof DenseVector) {
                final double[] resData = ((DenseVector)res).getPr();
                if (V instanceof DenseVector) {
                    final double[] VData = ((DenseVector)V).getPr();
                    if (U instanceof DenseVector) {
                        final double[] UData = ((DenseVector)U).getPr();
                        for (int i = 0; i < dim; ++i) {
                            resData[i] = a * VData[i] - UData[i];
                        }
                    }
                    else if (U instanceof SparseVector) {
                        final int[] ir = ((SparseVector)U).getIr();
                        final double[] pr = ((SparseVector)U).getPr();
                        final int nnz = ((SparseVector)U).getNNZ();
                        int lastIdx = -1;
                        int currentIdx = 0;
                        for (int k = 0; k < nnz; ++k) {
                            currentIdx = ir[k];
                            for (int j = lastIdx + 1; j < currentIdx; ++j) {
                                resData[j] = a * VData[j];
                            }
                            resData[currentIdx] = a * VData[currentIdx] - pr[k];
                            lastIdx = currentIdx;
                        }
                        for (int l = lastIdx + 1; l < dim; ++l) {
                            resData[l] = a * VData[l];
                        }
                    }
                }
                else if (V instanceof SparseVector) {
                    final int[] ir2 = ((SparseVector)V).getIr();
                    final double[] pr2 = ((SparseVector)V).getPr();
                    final int nnz2 = ((SparseVector)V).getNNZ();
                    if (U instanceof DenseVector) {
                        final double[] UData2 = ((DenseVector)U).getPr();
                        int lastIdx = -1;
                        int currentIdx = 0;
                        for (int k = 0; k < nnz2; ++k) {
                            currentIdx = ir2[k];
                            for (int j = lastIdx + 1; j < currentIdx; ++j) {
                                resData[j] = -UData2[j];
                            }
                            resData[currentIdx] = a * pr2[k] - UData2[currentIdx];
                            lastIdx = currentIdx;
                        }
                        for (int l = lastIdx + 1; l < dim; ++l) {
                            resData[l] = -UData2[l];
                        }
                    }
                    else if (U instanceof SparseVector) {
                        final int[] ir3 = ((SparseVector)U).getIr();
                        final double[] pr3 = ((SparseVector)U).getPr();
                        final int nnz3 = ((SparseVector)U).getNNZ();
                        clear(resData);
                        if (nnz2 != 0 || nnz3 != 0) {
                            int k2 = 0;
                            int k3 = 0;
                            int r1 = 0;
                            int r2 = 0;
                            double v = 0.0;
                            int m = -1;
                            while (k2 < nnz2 || k3 < nnz3) {
                                if (k3 == nnz3) {
                                    m = ir2[k2];
                                    v = a * pr2[k2];
                                    ++k2;
                                }
                                else if (k2 == nnz2) {
                                    m = ir3[k3];
                                    v = -pr3[k3];
                                    ++k3;
                                }
                                else {
                                    r1 = ir2[k2];
                                    r2 = ir3[k3];
                                    if (r1 < r2) {
                                        m = r1;
                                        v = a * pr2[k2];
                                        ++k2;
                                    }
                                    else if (r1 == r2) {
                                        m = r1;
                                        v = a * pr2[k2] - pr3[k3];
                                        ++k2;
                                        ++k3;
                                    }
                                    else {
                                        m = r2;
                                        v = -pr3[k3];
                                        ++k3;
                                    }
                                }
                                if (v != 0.0) {
                                    resData[m] = v;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    public static void affine(final Vector res, final Vector V, final double b, final Vector U) {
        affine(res, b, U, '+', V);
    }
    
    public static void affine(final double[] res, final double[] U, final double a, final double[] V) {
        for (int i = 0; i < res.length; ++i) {
            res[i] = U[i] + a * V[i];
        }
    }
    
    public static void uminusAssign(final Vector res) {
        final int dim = res.getDim();
        if (res instanceof SparseVector) {
            final double[] pr = ((SparseVector)res).getPr();
            for (int nnz = ((SparseVector)res).getNNZ(), k = 0; k < nnz; ++k) {
                pr[k] = -pr[k];
            }
        }
        else if (res instanceof DenseVector) {
            final double[] resData = ((DenseVector)res).getPr();
            for (int i = 0; i < dim; ++i) {
                resData[i] = -resData[i];
            }
        }
    }
    
    public static void uminus(final Vector res, final Vector V) {
        final int dim = res.getDim();
        if (!(res instanceof SparseVector) && res instanceof DenseVector) {
            final double[] resData = ((DenseVector)res).getPr();
            if (V instanceof DenseVector) {
                final double[] VData = ((DenseVector)V).getPr();
                for (int i = 0; i < dim; ++i) {
                    resData[i] = -VData[i];
                }
            }
            else if (V instanceof SparseVector) {
                final int[] ir = ((SparseVector)V).getIr();
                final double[] pr = ((SparseVector)V).getPr();
                final int nnz = ((SparseVector)V).getNNZ();
                int lastIdx = -1;
                int currentIdx = 0;
                for (int k = 0; k < nnz; ++k) {
                    currentIdx = ir[k];
                    for (int j = lastIdx + 1; j < currentIdx; ++j) {
                        resData[j] = 0.0;
                    }
                    resData[currentIdx] = -pr[k];
                    lastIdx = currentIdx;
                }
                for (int l = lastIdx + 1; l < dim; ++l) {
                    resData[l] = 0.0;
                }
            }
        }
    }
    
    public static void times(final Vector res, final Vector V, final Vector U) {
        final int dim = res.getDim();
        if (!(res instanceof SparseVector) && res instanceof DenseVector) {
            final double[] resData = ((DenseVector)res).getPr();
            if (V instanceof DenseVector) {
                final double[] VData = ((DenseVector)V).getPr();
                if (U instanceof DenseVector) {
                    final double[] UData = ((DenseVector)U).getPr();
                    for (int i = 0; i < dim; ++i) {
                        resData[i] = VData[i] * UData[i];
                    }
                }
                else if (U instanceof SparseVector) {
                    final int[] ir = ((SparseVector)U).getIr();
                    final double[] pr = ((SparseVector)U).getPr();
                    final int nnz = ((SparseVector)U).getNNZ();
                    int idx = -1;
                    res.clear();
                    for (int k = 0; k < nnz; ++k) {
                        idx = ir[k];
                        resData[idx] = VData[idx] * pr[k];
                    }
                }
            }
            else if (V instanceof SparseVector) {
                final int[] ir2 = ((SparseVector)V).getIr();
                final double[] pr2 = ((SparseVector)V).getPr();
                final int nnz2 = ((SparseVector)V).getNNZ();
                if (U instanceof DenseVector) {
                    final double[] UData2 = ((DenseVector)U).getPr();
                    int lastIdx = -1;
                    int currentIdx = 0;
                    for (int j = 0; j < nnz2; ++j) {
                        currentIdx = ir2[j];
                        for (int l = lastIdx + 1; l < currentIdx; ++l) {
                            resData[l] = 0.0;
                        }
                        resData[currentIdx] = pr2[j] * UData2[currentIdx];
                        lastIdx = currentIdx;
                    }
                    for (int m = lastIdx + 1; m < dim; ++m) {
                        resData[m] = 0.0;
                    }
                }
                else if (U instanceof SparseVector) {
                    final int[] ir3 = ((SparseVector)U).getIr();
                    final double[] pr3 = ((SparseVector)U).getPr();
                    final int nnz3 = ((SparseVector)U).getNNZ();
                    res.clear();
                    if (nnz2 != 0 && nnz3 != 0) {
                        int k2 = 0;
                        int k3 = 0;
                        int r1 = 0;
                        int r2 = 0;
                        double v = 0.0;
                        int i2 = -1;
                        while (k2 < nnz2 && k3 < nnz3) {
                            r1 = ir2[k2];
                            r2 = ir3[k3];
                            if (r1 < r2) {
                                ++k2;
                            }
                            else if (r1 == r2) {
                                i2 = r1;
                                v = pr2[k2] * pr3[k3];
                                ++k2;
                                ++k3;
                                if (v == 0.0) {
                                    continue;
                                }
                                resData[i2] = v;
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
    
    public static void times(final Vector res, final double v, final Vector V) {
        if (v == 0.0) {
            res.clear();
            return;
        }
        if (v == 1.0) {
            assign(res, V);
            return;
        }
        if (v == -1.0) {
            uminus(res, V);
            return;
        }
        final int dim = res.getDim();
        if (!(res instanceof SparseVector) && res instanceof DenseVector) {
            final double[] resData = ((DenseVector)res).getPr();
            if (V instanceof DenseVector) {
                final double[] VData = ((DenseVector)V).getPr();
                for (int i = 0; i < dim; ++i) {
                    resData[i] = v * VData[i];
                }
            }
            else if (V instanceof SparseVector) {
                final int[] ir = ((SparseVector)V).getIr();
                final double[] pr = ((SparseVector)V).getPr();
                final int nnz = ((SparseVector)V).getNNZ();
                int lastIdx = -1;
                int currentIdx = 0;
                for (int k = 0; k < nnz; ++k) {
                    currentIdx = ir[k];
                    for (int j = lastIdx + 1; j < currentIdx; ++j) {
                        resData[j] = 0.0;
                    }
                    resData[currentIdx] = v * pr[k];
                    lastIdx = currentIdx;
                }
                for (int l = lastIdx + 1; l < dim; ++l) {
                    resData[l] = 0.0;
                }
            }
        }
    }
    
    public static void timesAssign(final Vector res, final Vector V) {
        final int dim = res.getDim();
        if (!(res instanceof SparseVector) && res instanceof DenseVector) {
            final double[] resData = ((DenseVector)res).getPr();
            if (V instanceof DenseVector) {
                final double[] VData = ((DenseVector)V).getPr();
                for (int i = 0; i < dim; ++i) {
                    final double[] array = resData;
                    final int n = i;
                    array[n] *= VData[i];
                }
            }
            else if (V instanceof SparseVector) {
                final int[] ir = ((SparseVector)V).getIr();
                final double[] pr = ((SparseVector)V).getPr();
                final int nnz = ((SparseVector)V).getNNZ();
                int lastIdx = -1;
                int currentIdx = 0;
                for (int k = 0; k < nnz; ++k) {
                    currentIdx = ir[k];
                    for (int j = lastIdx + 1; j < currentIdx; ++j) {
                        resData[j] = 0.0;
                    }
                    final double[] array2 = resData;
                    final int n2 = currentIdx;
                    array2[n2] *= pr[k];
                    lastIdx = currentIdx;
                }
                for (int l = lastIdx + 1; l < dim; ++l) {
                    resData[l] = 0.0;
                }
            }
        }
    }
    
    public static void timesAssign(final Vector res, final double v) {
        final int dim = res.getDim();
        if (res instanceof SparseVector) {
            if (v == 0.0) {
                res.clear();
                return;
            }
            final double[] pr = ((SparseVector)res).getPr();
            for (int nnz = ((SparseVector)res).getNNZ(), k = 0; k < nnz; ++k) {
                final double[] array = pr;
                final int n = k;
                array[n] *= v;
            }
        }
        else if (res instanceof DenseVector) {
            final double[] resData = ((DenseVector)res).getPr();
            for (int i = 0; i < dim; ++i) {
                final double[] array2 = resData;
                final int n2 = i;
                array2[n2] *= v;
            }
        }
    }
    
    public static void assign(final Vector res, final Vector V) {
        final int dim = res.getDim();
        if (res instanceof SparseVector) {
            ((SparseVector)res).assignSparseVector((SparseVector)V);
        }
        else if (res instanceof DenseVector) {
            final double[] resData = ((DenseVector)res).getPr();
            if (V instanceof DenseVector) {
                final double[] VData = ((DenseVector)V).getPr();
                System.arraycopy(VData, 0, resData, 0, dim);
            }
            else if (V instanceof SparseVector) {
                final int[] ir = ((SparseVector)V).getIr();
                final double[] pr = ((SparseVector)V).getPr();
                final int nnz = ((SparseVector)V).getNNZ();
                int lastIdx = -1;
                int currentIdx = 0;
                for (int k = 0; k < nnz; ++k) {
                    currentIdx = ir[k];
                    for (int i = lastIdx + 1; i < currentIdx; ++i) {
                        resData[i] = 0.0;
                    }
                    resData[currentIdx] = pr[k];
                    lastIdx = currentIdx;
                }
                for (int j = lastIdx + 1; j < dim; ++j) {
                    resData[j] = 0.0;
                }
            }
        }
    }
    
    public static void minus(final Vector res, final Vector V, final Vector U) {
        final int dim = res.getDim();
        if (!(res instanceof SparseVector) && res instanceof DenseVector) {
            final double[] resData = ((DenseVector)res).getPr();
            if (V instanceof DenseVector) {
                final double[] VData = ((DenseVector)V).getPr();
                if (U instanceof DenseVector) {
                    final double[] UData = ((DenseVector)U).getPr();
                    for (int i = 0; i < dim; ++i) {
                        resData[i] = VData[i] - UData[i];
                    }
                }
                else if (U instanceof SparseVector) {
                    final int[] ir = ((SparseVector)U).getIr();
                    final double[] pr = ((SparseVector)U).getPr();
                    final int nnz = ((SparseVector)U).getNNZ();
                    System.arraycopy(VData, 0, resData, 0, dim);
                    for (int k = 0; k < nnz; ++k) {
                        final double[] array = resData;
                        final int n = ir[k];
                        array[n] -= pr[k];
                    }
                }
            }
            else if (V instanceof SparseVector) {
                final int[] ir2 = ((SparseVector)V).getIr();
                final double[] pr2 = ((SparseVector)V).getPr();
                final int nnz2 = ((SparseVector)V).getNNZ();
                if (U instanceof DenseVector) {
                    final double[] UData2 = ((DenseVector)U).getPr();
                    int lastIdx = -1;
                    int currentIdx = 0;
                    for (int j = 0; j < nnz2; ++j) {
                        currentIdx = ir2[j];
                        for (int l = lastIdx + 1; l < currentIdx; ++l) {
                            resData[l] = -UData2[l];
                        }
                        resData[currentIdx] = pr2[j] - UData2[currentIdx];
                        lastIdx = currentIdx;
                    }
                    for (int m = lastIdx + 1; m < dim; ++m) {
                        resData[m] = -UData2[m];
                    }
                }
                else if (U instanceof SparseVector) {
                    final int[] ir3 = ((SparseVector)U).getIr();
                    final double[] pr3 = ((SparseVector)U).getPr();
                    final int nnz3 = ((SparseVector)U).getNNZ();
                    clear(resData);
                    if (nnz2 != 0 || nnz3 != 0) {
                        int k2 = 0;
                        int k3 = 0;
                        int r1 = 0;
                        int r2 = 0;
                        double v = 0.0;
                        int i2 = -1;
                        while (k2 < nnz2 || k3 < nnz3) {
                            if (k3 == nnz3) {
                                i2 = ir2[k2];
                                v = pr2[k2];
                                ++k2;
                            }
                            else if (k2 == nnz2) {
                                i2 = ir3[k3];
                                v = -pr3[k3];
                                ++k3;
                            }
                            else {
                                r1 = ir2[k2];
                                r2 = ir3[k3];
                                if (r1 < r2) {
                                    i2 = r1;
                                    v = pr2[k2];
                                    ++k2;
                                }
                                else if (r1 == r2) {
                                    i2 = r1;
                                    v = pr2[k2] - pr3[k3];
                                    ++k2;
                                    ++k3;
                                }
                                else {
                                    i2 = r2;
                                    v = -pr3[k3];
                                    ++k3;
                                }
                            }
                            if (v != 0.0) {
                                resData[i2] = v;
                            }
                        }
                    }
                }
            }
        }
    }
    
    public static void minus(final double[] res, final double[] V1, final double[] V2) {
        for (int i = 0; i < res.length; ++i) {
            res[i] = V1[i] - V2[i];
        }
    }
    
    public static void minus(final Vector res, final Vector V, final double v) {
        final int dim = res.getDim();
        final double minusv = -v;
        if (!(res instanceof SparseVector) && res instanceof DenseVector) {
            final double[] resData = ((DenseVector)res).getPr();
            if (V instanceof DenseVector) {
                final double[] VData = ((DenseVector)V).getPr();
                for (int i = 0; i < dim; ++i) {
                    resData[i] = VData[i] - v;
                }
            }
            else if (V instanceof SparseVector) {
                final int[] ir = ((SparseVector)V).getIr();
                final double[] pr = ((SparseVector)V).getPr();
                final int nnz = ((SparseVector)V).getNNZ();
                int lastIdx = -1;
                int currentIdx = 0;
                for (int k = 0; k < nnz; ++k) {
                    currentIdx = ir[k];
                    for (int j = lastIdx + 1; j < currentIdx; ++j) {
                        resData[j] = minusv;
                    }
                    resData[currentIdx] = pr[k] - v;
                    lastIdx = currentIdx;
                }
                for (int l = lastIdx + 1; l < dim; ++l) {
                    resData[l] = minusv;
                }
            }
        }
    }
    
    public static void minus(final Vector res, final double v, final Vector V) {
        final int dim = res.getDim();
        if (!(res instanceof SparseVector) && res instanceof DenseVector) {
            final double[] resData = ((DenseVector)res).getPr();
            if (V instanceof DenseVector) {
                final double[] VData = ((DenseVector)V).getPr();
                for (int i = 0; i < dim; ++i) {
                    resData[i] = v - VData[i];
                }
            }
            else if (V instanceof SparseVector) {
                final int[] ir = ((SparseVector)V).getIr();
                final double[] pr = ((SparseVector)V).getPr();
                final int nnz = ((SparseVector)V).getNNZ();
                int lastIdx = -1;
                int currentIdx = 0;
                for (int k = 0; k < nnz; ++k) {
                    currentIdx = ir[k];
                    for (int j = lastIdx + 1; j < currentIdx; ++j) {
                        resData[j] = v;
                    }
                    resData[currentIdx] = v - pr[k];
                    lastIdx = currentIdx;
                }
                for (int l = lastIdx + 1; l < dim; ++l) {
                    resData[l] = v;
                }
            }
        }
    }
    
    public static void minusAssign(final Vector res, final double v) {
        final int dim = res.getDim();
        if (!(res instanceof SparseVector) && res instanceof DenseVector) {
            final double[] resData = ((DenseVector)res).getPr();
            for (int i = 0; i < dim; ++i) {
                final double[] array = resData;
                final int n = i;
                array[n] -= v;
            }
        }
    }
    
    public static void minusAssign(final Vector res, final Vector V) {
        final int dim = res.getDim();
        if (res instanceof SparseVector) {
            ((SparseVector)res).assignSparseVector((SparseVector)res.minus(V));
        }
        else if (res instanceof DenseVector) {
            final double[] resData = ((DenseVector)res).getPr();
            if (V instanceof DenseVector) {
                final double[] VData = ((DenseVector)V).getPr();
                for (int i = 0; i < dim; ++i) {
                    final double[] array = resData;
                    final int n = i;
                    array[n] -= VData[i];
                }
            }
            else if (V instanceof SparseVector) {
                final int[] ir = ((SparseVector)V).getIr();
                final double[] pr = ((SparseVector)V).getPr();
                for (int nnz = ((SparseVector)V).getNNZ(), k = 0; k < nnz; ++k) {
                    final double[] array2 = resData;
                    final int n2 = ir[k];
                    array2[n2] -= pr[k];
                }
            }
        }
    }
    
    public static void minusAssign(final Vector res, final double a, final Vector V) {
        if (a == 0.0) {
            return;
        }
        if (a == 1.0) {
            minusAssign(res, V);
            return;
        }
        if (a == -1.0) {
            plusAssign(res, V);
            return;
        }
        final int dim = res.getDim();
        if (res instanceof SparseVector) {
            plusAssign(res, -a, V);
        }
        else if (res instanceof DenseVector) {
            final double[] resData = ((DenseVector)res).getPr();
            if (V instanceof DenseVector) {
                final double[] VData = ((DenseVector)V).getPr();
                for (int i = 0; i < dim; ++i) {
                    final double[] array = resData;
                    final int n = i;
                    array[n] -= a * VData[i];
                }
            }
            else if (V instanceof SparseVector) {
                final int[] ir = ((SparseVector)V).getIr();
                final double[] pr = ((SparseVector)V).getPr();
                for (int nnz = ((SparseVector)V).getNNZ(), k = 0; k < nnz; ++k) {
                    final double[] array2 = resData;
                    final int n2 = ir[k];
                    array2[n2] -= a * pr[k];
                }
            }
        }
    }
    
    public static void plus(final Vector res, final Vector V, final Vector U) {
        final int dim = res.getDim();
        if (res instanceof SparseVector) {
            ((SparseVector)res).assignSparseVector((SparseVector)V.plus(U));
        }
        else if (res instanceof DenseVector) {
            final double[] resData = ((DenseVector)res).getPr();
            if (V instanceof DenseVector) {
                final double[] VData = ((DenseVector)V).getPr();
                if (U instanceof DenseVector) {
                    final double[] UData = ((DenseVector)U).getPr();
                    for (int i = 0; i < dim; ++i) {
                        resData[i] = VData[i] + UData[i];
                    }
                }
                else if (U instanceof SparseVector) {
                    final int[] ir = ((SparseVector)U).getIr();
                    final double[] pr = ((SparseVector)U).getPr();
                    final int nnz = ((SparseVector)U).getNNZ();
                    System.arraycopy(VData, 0, resData, 0, dim);
                    for (int k = 0; k < nnz; ++k) {
                        final double[] array = resData;
                        final int n = ir[k];
                        array[n] += pr[k];
                    }
                }
            }
            else if (V instanceof SparseVector) {
                final int[] ir2 = ((SparseVector)V).getIr();
                final double[] pr2 = ((SparseVector)V).getPr();
                final int nnz2 = ((SparseVector)V).getNNZ();
                if (U instanceof DenseVector) {
                    final double[] UData2 = ((DenseVector)U).getPr();
                    int lastIdx = -1;
                    int currentIdx = 0;
                    for (int j = 0; j < nnz2; ++j) {
                        currentIdx = ir2[j];
                        for (int l = lastIdx + 1; l < currentIdx; ++l) {
                            resData[l] = UData2[l];
                        }
                        resData[currentIdx] = pr2[j] + UData2[currentIdx];
                        lastIdx = currentIdx;
                    }
                    for (int m = lastIdx + 1; m < dim; ++m) {
                        resData[m] = UData2[m];
                    }
                }
                else if (U instanceof SparseVector) {
                    final int[] ir3 = ((SparseVector)U).getIr();
                    final double[] pr3 = ((SparseVector)U).getPr();
                    final int nnz3 = ((SparseVector)U).getNNZ();
                    clear(resData);
                    if (nnz2 != 0 || nnz3 != 0) {
                        int k2 = 0;
                        int k3 = 0;
                        int r1 = 0;
                        int r2 = 0;
                        double v = 0.0;
                        int i2 = -1;
                        while (k2 < nnz2 || k3 < nnz3) {
                            if (k3 == nnz3) {
                                i2 = ir2[k2];
                                v = pr2[k2];
                                ++k2;
                            }
                            else if (k2 == nnz2) {
                                i2 = ir3[k3];
                                v = pr3[k3];
                                ++k3;
                            }
                            else {
                                r1 = ir2[k2];
                                r2 = ir3[k3];
                                if (r1 < r2) {
                                    i2 = r1;
                                    v = pr2[k2];
                                    ++k2;
                                }
                                else if (r1 == r2) {
                                    i2 = r1;
                                    v = pr2[k2] + pr3[k3];
                                    ++k2;
                                    ++k3;
                                }
                                else {
                                    i2 = r2;
                                    v = pr3[k3];
                                    ++k3;
                                }
                            }
                            if (v != 0.0) {
                                resData[i2] = v;
                            }
                        }
                    }
                }
            }
        }
    }
    
    public static void plus(final double[] res, final double[] V1, final double[] V2) {
        for (int i = 0; i < res.length; ++i) {
            res[i] = V1[i] + V2[i];
        }
    }
    
    public static void plus(final Vector res, final Vector V, final double v) {
        final int dim = res.getDim();
        if (!(res instanceof SparseVector) && res instanceof DenseVector) {
            final double[] resData = ((DenseVector)res).getPr();
            if (V instanceof DenseVector) {
                final double[] VData = ((DenseVector)V).getPr();
                for (int i = 0; i < dim; ++i) {
                    resData[i] = VData[i] + v;
                }
            }
            else if (V instanceof SparseVector) {
                final int[] ir = ((SparseVector)V).getIr();
                final double[] pr = ((SparseVector)V).getPr();
                final int nnz = ((SparseVector)V).getNNZ();
                int lastIdx = -1;
                int currentIdx = 0;
                for (int k = 0; k < nnz; ++k) {
                    currentIdx = ir[k];
                    for (int j = lastIdx + 1; j < currentIdx; ++j) {
                        resData[j] = v;
                    }
                    resData[currentIdx] = pr[k] + v;
                    lastIdx = currentIdx;
                }
                for (int l = lastIdx + 1; l < dim; ++l) {
                    resData[l] = v;
                }
            }
        }
    }
    
    public static void plus(final Vector res, final double v, final Vector V) {
        final int dim = res.getDim();
        if (!(res instanceof SparseVector) && res instanceof DenseVector) {
            final double[] resData = ((DenseVector)res).getPr();
            if (V instanceof DenseVector) {
                final double[] VData = ((DenseVector)V).getPr();
                for (int i = 0; i < dim; ++i) {
                    resData[i] = v + VData[i];
                }
            }
            else if (V instanceof SparseVector) {
                final int[] ir = ((SparseVector)V).getIr();
                final double[] pr = ((SparseVector)V).getPr();
                final int nnz = ((SparseVector)V).getNNZ();
                int lastIdx = -1;
                int currentIdx = 0;
                for (int k = 0; k < nnz; ++k) {
                    currentIdx = ir[k];
                    for (int j = lastIdx + 1; j < currentIdx; ++j) {
                        resData[j] = v;
                    }
                    resData[currentIdx] = v + pr[k];
                    lastIdx = currentIdx;
                }
                for (int l = lastIdx + 1; l < dim; ++l) {
                    resData[l] = v;
                }
            }
        }
    }
    
    public static void plusAssign(final Vector res, final double v) {
        final int dim = res.getDim();
        if (!(res instanceof SparseVector) && res instanceof DenseVector) {
            final double[] resData = ((DenseVector)res).getPr();
            for (int i = 0; i < dim; ++i) {
                final double[] array = resData;
                final int n = i;
                array[n] += v;
            }
        }
    }
    
    public static void plusAssign(final Vector res, final Vector V) {
        final int dim = res.getDim();
        if (res instanceof SparseVector) {
            ((SparseVector)res).assignSparseVector((SparseVector)res.plus(V));
        }
        else if (res instanceof DenseVector) {
            final double[] resData = ((DenseVector)res).getPr();
            if (V instanceof DenseVector) {
                final double[] VData = ((DenseVector)V).getPr();
                for (int i = 0; i < dim; ++i) {
                    final double[] array = resData;
                    final int n = i;
                    array[n] += VData[i];
                }
            }
            else if (V instanceof SparseVector) {
                final int[] ir = ((SparseVector)V).getIr();
                final double[] pr = ((SparseVector)V).getPr();
                for (int nnz = ((SparseVector)V).getNNZ(), k = 0; k < nnz; ++k) {
                    final double[] array2 = resData;
                    final int n2 = ir[k];
                    array2[n2] += pr[k];
                }
            }
        }
    }
    
    public static void plusAssign(final Vector res, final double a, final Vector V) {
        if (a == 0.0) {
            return;
        }
        if (a == 1.0) {
            plusAssign(res, V);
            return;
        }
        if (a == -1.0) {
            minusAssign(res, V);
            return;
        }
        final int dim = res.getDim();
        if (res instanceof SparseVector) {
            if (V instanceof DenseVector) {
                final ArrayList<Pair<Integer, Double>> list = new ArrayList<Pair<Integer, Double>>();
                final int[] ir1 = ((SparseVector)res).getIr();
                final double[] pr1 = ((SparseVector)res).getPr();
                final int nnz1 = ((SparseVector)res).getNNZ();
                final double[] pr2 = ((DenseVector)V).getPr();
                int idx = -1;
                int i = -1;
                double v = 0.0;
                int lastIdx = -1;
                for (int k = 0; k < nnz1; ++k) {
                    idx = ir1[k];
                    for (int r = lastIdx + 1; r < idx; ++r) {
                        i = r;
                        v = a * pr2[i];
                        if (v != 0.0) {
                            list.add(Pair.of(i, v));
                        }
                    }
                    i = idx;
                    v = pr1[k] + a * pr2[i];
                    if (v != 0.0) {
                        list.add(Pair.of(i, v));
                    }
                    lastIdx = idx;
                }
                for (int r2 = lastIdx + 1; r2 < dim; ++r2) {
                    i = r2;
                    v = a * pr2[i];
                    if (v != 0.0) {
                        list.add(Pair.of(i, v));
                    }
                }
                final int nnz2 = list.size();
                final int[] ir_res = new int[nnz2];
                final double[] pr_res = new double[nnz2];
                int j = 0;
                for (final Pair<Integer, Double> pair : list) {
                    ir_res[j] = pair.first;
                    pr_res[j] = pair.second;
                    ++j;
                }
                ((SparseVector)res).assignSparseVector(new SparseVector(ir_res, pr_res, nnz2, dim));
            }
            else if (V instanceof SparseVector) {
                final ArrayList<Pair<Integer, Double>> list = new ArrayList<Pair<Integer, Double>>();
                final int[] ir1 = ((SparseVector)res).getIr();
                final double[] pr1 = ((SparseVector)res).getPr();
                final int nnz1 = ((SparseVector)res).getNNZ();
                final int[] ir2 = ((SparseVector)V).getIr();
                final double[] pr3 = ((SparseVector)V).getPr();
                final int nnz3 = ((SparseVector)V).getNNZ();
                if (nnz1 != 0 || nnz3 != 0) {
                    int k2 = 0;
                    int k3 = 0;
                    int r3 = 0;
                    int r4 = 0;
                    double v2 = 0.0;
                    int l = -1;
                    while (k2 < nnz1 || k3 < nnz3) {
                        if (k3 == nnz3) {
                            l = ir1[k2];
                            v2 = pr1[k2];
                            ++k2;
                        }
                        else if (k2 == nnz1) {
                            l = ir2[k3];
                            v2 = a * pr3[k3];
                            ++k3;
                        }
                        else {
                            r3 = ir1[k2];
                            r4 = ir2[k3];
                            if (r3 < r4) {
                                l = r3;
                                v2 = pr1[k2];
                                ++k2;
                            }
                            else if (r3 == r4) {
                                l = r3;
                                v2 = pr1[k2] + a * pr3[k3];
                                ++k2;
                                ++k3;
                            }
                            else {
                                l = r4;
                                v2 = a * pr3[k3];
                                ++k3;
                            }
                        }
                        if (v2 != 0.0) {
                            list.add(Pair.of(l, v2));
                        }
                    }
                }
                final int nnz4 = list.size();
                final int[] ir_res2 = new int[nnz4];
                final double[] pr_res2 = new double[nnz4];
                int k = 0;
                for (final Pair<Integer, Double> pair2 : list) {
                    ir_res2[k] = pair2.first;
                    pr_res2[k] = pair2.second;
                    ++k;
                }
                ((SparseVector)res).assignSparseVector(new SparseVector(ir_res2, pr_res2, nnz4, dim));
            }
        }
        else if (res instanceof DenseVector) {
            final double[] resData = ((DenseVector)res).getPr();
            if (V instanceof DenseVector) {
                final double[] VData = ((DenseVector)V).getPr();
                for (int m = 0; m < dim; ++m) {
                    final double[] array = resData;
                    final int n = m;
                    array[n] += a * VData[m];
                }
            }
            else if (V instanceof SparseVector) {
                final int[] ir3 = ((SparseVector)V).getIr();
                final double[] pr4 = ((SparseVector)V).getPr();
                for (int nnz5 = ((SparseVector)V).getNNZ(), k4 = 0; k4 < nnz5; ++k4) {
                    final double[] array2 = resData;
                    final int n2 = ir3[k4];
                    array2[n2] += a * pr4[k4];
                }
            }
        }
    }
    
    public static void mtimes(final Matrix res, final Matrix A, final char operator, final Matrix B) {
        if (operator == ' ') {
            mtimes(res, A, B);
        }
        else if (operator == 'T') {
            if (res instanceof SparseMatrix) {
                ((SparseMatrix)res).assignSparseMatrix(Matlab.sparse(A.transpose().mtimes(B)));
            }
            else if (res instanceof DenseMatrix) {
                final double[][] resData = ((DenseMatrix)res).getData();
                final int NB = B.getColumnDimension();
                final int N = A.getRowDimension();
                final int M = A.getColumnDimension();
                if (A instanceof DenseMatrix) {
                    final double[][] AData = ((DenseMatrix)A).getData();
                    if (B instanceof DenseMatrix) {
                        final double[][] BData = ((DenseMatrix)B).getData();
                        double[] resRow = null;
                        double[] BRow = null;
                        double A_ki = 0.0;
                        for (int i = 0; i < M; ++i) {
                            resRow = resData[i];
                            clear(resRow);
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
                        for (int l = 0; l < M; ++l) {
                            for (int t = 0; t < N; ++t) {
                                columnA[t] = AData[t][l];
                            }
                            for (int m = 0; m < NB; ++m) {
                                s = 0.0;
                                for (int k2 = jc[m]; k2 < jc[m + 1]; ++k2) {
                                    r = ir[k2];
                                    s += columnA[r] * pr[k2];
                                }
                                resData[l][m] = s;
                            }
                        }
                    }
                }
                else if (A instanceof SparseMatrix) {
                    if (B instanceof DenseMatrix) {
                        final int[] ir2 = ((SparseMatrix)A).getIr();
                        final int[] jc2 = ((SparseMatrix)A).getJc();
                        final double[] pr2 = ((SparseMatrix)A).getPr();
                        final double[][] BData2 = ((DenseMatrix)B).getData();
                        int c = -1;
                        double s = 0.0;
                        for (int i2 = 0; i2 < M; ++i2) {
                            for (int j = 0; j < NB; ++j) {
                                s = 0.0;
                                for (int k3 = jc2[i2]; k3 < jc2[i2 + 1]; ++k3) {
                                    c = ir2[k3];
                                    s += pr2[k3] * BData2[c][j];
                                }
                                resData[i2][j] = s;
                            }
                        }
                    }
                    else if (B instanceof SparseMatrix) {
                        double[] resRow2 = null;
                        final int[] ir3 = ((SparseMatrix)A).getIr();
                        final int[] jc3 = ((SparseMatrix)A).getJc();
                        final double[] pr3 = ((SparseMatrix)A).getPr();
                        final int[] ir4 = ((SparseMatrix)B).getIr();
                        final int[] jc4 = ((SparseMatrix)B).getJc();
                        final double[] pr4 = ((SparseMatrix)B).getPr();
                        int rr = -1;
                        int cl = -1;
                        double s2 = 0.0;
                        int kl = 0;
                        int kr = 0;
                        for (int i3 = 0; i3 < M; ++i3) {
                            resRow2 = resData[i3];
                            for (int j2 = 0; j2 < NB; ++j2) {
                                s2 = 0.0;
                                kl = jc3[i3];
                                kr = jc4[j2];
                                while (kl < jc3[i3 + 1] && kr < jc4[j2 + 1]) {
                                    cl = ir3[kl];
                                    rr = ir4[kr];
                                    if (cl < rr) {
                                        ++kl;
                                    }
                                    else if (cl > rr) {
                                        ++kr;
                                    }
                                    else {
                                        s2 += pr3[kl] * pr4[kr];
                                        ++kl;
                                        ++kr;
                                    }
                                }
                                resRow2[j2] = s2;
                            }
                        }
                    }
                }
            }
        }
    }
    
    public static void mtimes(final Matrix res, final Matrix A, final Matrix B) {
        if (res instanceof SparseMatrix) {
            ((SparseMatrix)res).assignSparseMatrix(Matlab.sparse(A.mtimes(B)));
        }
        else if (res instanceof DenseMatrix) {
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] rowA = null;
            final int NB = B.getColumnDimension();
            final int M = A.getRowDimension();
            final int N = A.getColumnDimension();
            if (A instanceof DenseMatrix) {
                final double[][] AData = ((DenseMatrix)A).getData();
                if (B instanceof DenseMatrix) {
                    final double[][] BData = ((DenseMatrix)B).getData();
                    final double[] columnB = new double[B.getRowDimension()];
                    double s = 0.0;
                    for (int j = 0; j < NB; ++j) {
                        for (int r = 0; r < B.getRowDimension(); ++r) {
                            columnB[r] = BData[r][j];
                        }
                        for (int i = 0; i < M; ++i) {
                            rowA = AData[i];
                            s = 0.0;
                            for (int k = 0; k < N; ++k) {
                                s += rowA[k] * columnB[k];
                            }
                            resData[i][j] = s;
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
                    int r2 = -1;
                    double s2 = 0.0;
                    for (int l = 0; l < M; ++l) {
                        rowA = AData[l];
                        for (int m = 0; m < NB; ++m) {
                            s2 = 0.0;
                            for (int k2 = jc[m]; k2 < jc[m + 1]; ++k2) {
                                r2 = ir[k2];
                                s2 += rowA[r2] * pr[k2];
                            }
                            resData[l][m] = s2;
                        }
                    }
                }
            }
            else if (A instanceof SparseMatrix) {
                final int[] ic = ((SparseMatrix)A).getIc();
                final int[] jr = ((SparseMatrix)A).getJr();
                final double[] pr2 = ((SparseMatrix)A).getPr();
                final int[] valCSRIndices = ((SparseMatrix)A).getValCSRIndices();
                if (B instanceof DenseMatrix) {
                    final double[][] BData2 = ((DenseMatrix)B).getData();
                    int c = -1;
                    double s3 = 0.0;
                    for (int i2 = 0; i2 < M; ++i2) {
                        for (int j2 = 0; j2 < NB; ++j2) {
                            s3 = 0.0;
                            for (int k3 = jr[i2]; k3 < jr[i2 + 1]; ++k3) {
                                c = ic[k3];
                                s3 += pr2[valCSRIndices[k3]] * BData2[c][j2];
                            }
                            resData[i2][j2] = s3;
                        }
                    }
                }
                else if (B instanceof SparseMatrix) {
                    double[] resRow = null;
                    int[] ir2 = null;
                    int[] jc2 = null;
                    double[] pr3 = null;
                    ir2 = ((SparseMatrix)B).getIr();
                    jc2 = ((SparseMatrix)B).getJc();
                    pr3 = ((SparseMatrix)B).getPr();
                    int rr = -1;
                    int cl = -1;
                    double s4 = 0.0;
                    int kl = 0;
                    int kr = 0;
                    for (int i3 = 0; i3 < M; ++i3) {
                        resRow = resData[i3];
                        for (int j3 = 0; j3 < NB; ++j3) {
                            s4 = 0.0;
                            kl = jr[i3];
                            kr = jc2[j3];
                            while (kl < jr[i3 + 1] && kr < jc2[j3 + 1]) {
                                cl = ic[kl];
                                rr = ir2[kr];
                                if (cl < rr) {
                                    ++kl;
                                }
                                else if (cl > rr) {
                                    ++kr;
                                }
                                else {
                                    s4 += pr2[valCSRIndices[kl]] * pr3[kr];
                                    ++kl;
                                    ++kr;
                                }
                            }
                            resRow[j3] = s4;
                        }
                    }
                }
            }
        }
    }
    
    public static void times(final Matrix res, final Matrix A, final Matrix B) {
        if (res instanceof SparseMatrix) {
            ((SparseMatrix)res).assignSparseMatrix(Matlab.sparse(A.times(B)));
        }
        else if (res instanceof DenseMatrix) {
            final int M = A.getRowDimension();
            final int N = A.getColumnDimension();
            final double[][] resData = ((DenseMatrix)res).getData();
            if (A instanceof DenseMatrix) {
                final double[][] AData = ((DenseMatrix)A).getData();
                if (B instanceof DenseMatrix) {
                    final double[][] BData = ((DenseMatrix)B).getData();
                    double[] BRow = null;
                    double[] ARow = null;
                    double[] resRow = null;
                    for (int i = 0; i < M; ++i) {
                        ARow = AData[i];
                        BRow = BData[i];
                        resRow = resData[i];
                        for (int j = 0; j < N; ++j) {
                            resRow[j] = ARow[j] * BRow[j];
                        }
                    }
                }
                else if (B instanceof SparseMatrix) {
                    ArrayOperator.clearMatrix(resData);
                    int[] ir = null;
                    int[] jc = null;
                    double[] pr = null;
                    ir = ((SparseMatrix)B).getIr();
                    jc = ((SparseMatrix)B).getJc();
                    pr = ((SparseMatrix)B).getPr();
                    int r = -1;
                    for (int k = 0; k < B.getColumnDimension(); ++k) {
                        for (int l = jc[k]; l < jc[k + 1]; ++l) {
                            r = ir[l];
                            resData[r][k] = AData[r][k] * pr[l];
                        }
                    }
                }
            }
            else if (A instanceof SparseMatrix) {
                if (B instanceof DenseMatrix) {
                    times(res, B, A);
                }
                else if (B instanceof SparseMatrix) {
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
                    ArrayOperator.clearMatrix(resData);
                    int k2 = 0;
                    int k3 = 0;
                    int r2 = -1;
                    int r3 = -1;
                    int m = -1;
                    double v = 0.0;
                    for (int j2 = 0; j2 < N; ++j2) {
                        k2 = jc2[j2];
                        k3 = jc3[j2];
                        if (k2 != jc2[j2 + 1]) {
                            if (k3 != jc3[j2 + 1]) {
                                while (k2 < jc2[j2 + 1] && k3 < jc3[j2 + 1]) {
                                    r2 = ir2[k2];
                                    r3 = ir3[k3];
                                    if (r2 < r3) {
                                        ++k2;
                                    }
                                    else if (r2 == r3) {
                                        m = r2;
                                        v = pr2[k2] * pr3[k3];
                                        ++k2;
                                        ++k3;
                                        if (v == 0.0) {
                                            continue;
                                        }
                                        resData[m][j2] = v;
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
        }
    }
    
    public static void times(final double[] res, final double a, final double[] V) {
        for (int i = 0; i < res.length; ++i) {
            res[i] = a * V[i];
        }
    }
    
    public static void timesAssign(final Matrix res, final Matrix A) {
        if (res instanceof SparseMatrix) {
            ((SparseMatrix)res).assignSparseMatrix(Matlab.sparse(res.times(A)));
        }
        else if (res instanceof DenseMatrix) {
            final int M = A.getRowDimension();
            final int N = A.getColumnDimension();
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            if (A instanceof DenseMatrix) {
                final double[][] AData = ((DenseMatrix)A).getData();
                double[] ARow = null;
                for (int i = 0; i < M; ++i) {
                    ARow = AData[i];
                    resRow = resData[i];
                    for (int j = 0; j < N; ++j) {
                        final double[] array = resRow;
                        final int n = j;
                        array[n] *= ARow[j];
                    }
                }
            }
            else if (A instanceof SparseMatrix) {
                final int[] ic = ((SparseMatrix)A).getIc();
                final int[] jr = ((SparseMatrix)A).getJr();
                final int[] valCSRIndices = ((SparseMatrix)A).getValCSRIndices();
                final double[] pr = ((SparseMatrix)A).getPr();
                for (int k = 0; k < M; ++k) {
                    resRow = resData[k];
                    if (jr[k] == jr[k + 1]) {
                        ArrayOperator.clearVector(resRow);
                    }
                    else {
                        int lastColumnIdx = -1;
                        int currentColumnIdx = 0;
                        for (int l = jr[k]; l < jr[k + 1]; ++l) {
                            currentColumnIdx = ic[l];
                            for (int c = lastColumnIdx + 1; c < currentColumnIdx; ++c) {
                                resRow[c] = 0.0;
                            }
                            final double[] array2 = resRow;
                            final int n2 = currentColumnIdx;
                            array2[n2] *= pr[valCSRIndices[l]];
                            lastColumnIdx = currentColumnIdx;
                        }
                        for (int c2 = lastColumnIdx + 1; c2 < N; ++c2) {
                            resRow[c2] = 0.0;
                        }
                    }
                }
            }
        }
    }
    
    public static void timesAssign(final Matrix res, final double v) {
        if (v == 0.0) {
            res.clear();
        }
        final int M = res.getRowDimension();
        final int N = res.getColumnDimension();
        if (res instanceof SparseMatrix) {
            final double[] pr = ((SparseMatrix)res).getPr();
            for (int nnz = ((SparseMatrix)res).getNNZ(), k = 0; k < nnz; ++k) {
                final double[] array = pr;
                final int n = k;
                array[n] *= v;
            }
        }
        else if (res instanceof DenseMatrix) {
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            for (int i = 0; i < M; ++i) {
                resRow = resData[i];
                for (int j = 0; j < N; ++j) {
                    final double[] array2 = resRow;
                    final int n2 = j;
                    array2[n2] *= v;
                }
            }
        }
    }
    
    public static void times(final Matrix res, final double v, final Matrix A) {
        if (v == 1.0) {
            assign(res, A);
            return;
        }
        final int M = res.getRowDimension();
        final int N = res.getColumnDimension();
        if (M != A.getRowDimension() || N != A.getColumnDimension()) {
            System.err.println("Dimension doesn't match.");
            System.exit(1);
        }
        if (!(res instanceof SparseMatrix) && res instanceof DenseMatrix) {
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            if (A instanceof DenseMatrix) {
                final double[][] AData = ((DenseMatrix)A).getData();
                double[] ARow = null;
                for (int i = 0; i < M; ++i) {
                    resRow = resData[i];
                    ARow = AData[i];
                    for (int j = 0; j < N; ++j) {
                        resRow[j] = ARow[j] * v;
                    }
                }
            }
            else if (A instanceof SparseMatrix) {
                final int[] ic = ((SparseMatrix)A).getIc();
                final int[] jr = ((SparseMatrix)A).getJr();
                final int[] valCSRIndices = ((SparseMatrix)A).getValCSRIndices();
                final double[] pr = ((SparseMatrix)A).getPr();
                for (int k = 0; k < M; ++k) {
                    resRow = resData[k];
                    if (jr[k] == jr[k + 1]) {
                        ArrayOperator.assignVector(resRow, 0.0);
                    }
                    else {
                        int lastColumnIdx = -1;
                        int currentColumnIdx = 0;
                        for (int l = jr[k]; l < jr[k + 1]; ++l) {
                            currentColumnIdx = ic[l];
                            for (int c = lastColumnIdx + 1; c < currentColumnIdx; ++c) {
                                resRow[c] = 0.0;
                            }
                            resRow[currentColumnIdx] = pr[valCSRIndices[l]] * v;
                            lastColumnIdx = currentColumnIdx;
                        }
                        for (int c2 = lastColumnIdx + 1; c2 < N; ++c2) {
                            resRow[c2] = 0.0;
                        }
                    }
                }
            }
        }
    }
    
    public static void clear(final Matrix res) {
        res.clear();
    }
    
    public static void clear(final double[][] res) {
        ArrayOperator.clearMatrix(res);
    }
    
    public static void clear(final Vector res) {
        res.clear();
    }
    
    public static void clear(final double[] res) {
        ArrayOperator.clearVector(res);
    }
    
    public static void assign(final Matrix res, final Matrix A) {
        final int M = res.getRowDimension();
        final int N = res.getColumnDimension();
        if (M != A.getRowDimension() || N != A.getColumnDimension()) {
            System.err.println("Dimension doesn't match.");
            System.exit(1);
        }
        if (res instanceof SparseMatrix) {
            ((SparseMatrix)res).assignSparseMatrix(Matlab.sparse(A));
        }
        else if (res instanceof DenseMatrix) {
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            if (A instanceof DenseMatrix) {
                final double[][] AData = ((DenseMatrix)A).getData();
                double[] ARow = null;
                for (int i = 0; i < M; ++i) {
                    resRow = resData[i];
                    ARow = AData[i];
                    for (int j = 0; j < N; ++j) {
                        resRow[j] = ARow[j];
                    }
                }
            }
            else if (A instanceof SparseMatrix) {
                final int[] ic = ((SparseMatrix)A).getIc();
                final int[] jr = ((SparseMatrix)A).getJr();
                final int[] valCSRIndices = ((SparseMatrix)A).getValCSRIndices();
                final double[] pr = ((SparseMatrix)A).getPr();
                for (int k = 0; k < M; ++k) {
                    resRow = resData[k];
                    if (jr[k] == jr[k + 1]) {
                        ArrayOperator.assignVector(resRow, 0.0);
                    }
                    else {
                        int lastColumnIdx = -1;
                        int currentColumnIdx = 0;
                        for (int l = jr[k]; l < jr[k + 1]; ++l) {
                            currentColumnIdx = ic[l];
                            for (int c = lastColumnIdx + 1; c < currentColumnIdx; ++c) {
                                resRow[c] = 0.0;
                            }
                            resRow[currentColumnIdx] = pr[valCSRIndices[l]];
                            lastColumnIdx = currentColumnIdx;
                        }
                        for (int c2 = lastColumnIdx + 1; c2 < N; ++c2) {
                            resRow[c2] = 0.0;
                        }
                    }
                }
            }
        }
    }
    
    public static void assign(final double[] res, final double[] V) {
        System.arraycopy(V, 0, res, 0, res.length);
    }
    
    public static void assign(final double[] res, final double v) {
        for (int i = 0; i < res.length; ++i) {
            res[i] = v;
        }
    }
    
    public static void uminusAssign(final Matrix res) {
        if (res instanceof SparseMatrix) {
            final double[] pr = ((SparseMatrix)res).getPr();
            for (int nnz = ((SparseMatrix)res).getNNZ(), k = 0; k < nnz; ++k) {
                pr[k] = -pr[k];
            }
        }
        else if (res instanceof DenseMatrix) {
            final int M = res.getRowDimension();
            final int N = res.getColumnDimension();
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            for (int i = 0; i < M; ++i) {
                resRow = resData[i];
                for (int j = 0; j < N; ++j) {
                    resRow[j] = -resRow[j];
                }
            }
        }
    }
    
    public static void uminus(final Matrix res, final Matrix A) {
        final int M = res.getRowDimension();
        final int N = res.getColumnDimension();
        if (M != A.getRowDimension() || N != A.getColumnDimension()) {
            System.err.println("Dimension doesn't match.");
            System.exit(1);
        }
        if (!(res instanceof SparseMatrix) && res instanceof DenseMatrix) {
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            if (A instanceof DenseMatrix) {
                final double[][] AData = ((DenseMatrix)A).getData();
                double[] ARow = null;
                for (int i = 0; i < M; ++i) {
                    resRow = resData[i];
                    ARow = AData[i];
                    for (int j = 0; j < N; ++j) {
                        resRow[j] = -ARow[j];
                    }
                }
            }
            else if (A instanceof SparseMatrix) {
                final int[] ic = ((SparseMatrix)A).getIc();
                final int[] jr = ((SparseMatrix)A).getJr();
                final int[] valCSRIndices = ((SparseMatrix)A).getValCSRIndices();
                final double[] pr = ((SparseMatrix)A).getPr();
                for (int k = 0; k < M; ++k) {
                    resRow = resData[k];
                    if (jr[k] == jr[k + 1]) {
                        ArrayOperator.assignVector(resRow, 0.0);
                    }
                    else {
                        int lastColumnIdx = -1;
                        int currentColumnIdx = 0;
                        for (int l = jr[k]; l < jr[k + 1]; ++l) {
                            currentColumnIdx = ic[l];
                            for (int c = lastColumnIdx + 1; c < currentColumnIdx; ++c) {
                                resRow[c] = 0.0;
                            }
                            resRow[currentColumnIdx] = -pr[valCSRIndices[l]];
                            lastColumnIdx = currentColumnIdx;
                        }
                        for (int c2 = lastColumnIdx + 1; c2 < N; ++c2) {
                            resRow[c2] = 0.0;
                        }
                    }
                }
            }
        }
    }
    
    public static void uminus(final double[] res, final double[] V) {
        for (int i = 0; i < res.length; ++i) {
            res[i] = -V[i];
        }
    }
    
    public static void divide(final Matrix res, final double v) {
    }
    
    public static void rdivideAssign(final Matrix res, final double v) {
        final int nRow = res.getRowDimension();
        final int nCol = res.getColumnDimension();
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
    }
    
    public static void affine(final Matrix res, final Matrix A, final Matrix B, final double v, final Matrix C) {
        if (res instanceof SparseMatrix) {
            Printer.err("Sparse matrix for res is not supported.");
            Utility.exit(1);
        }
        else if (res instanceof DenseMatrix) {
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            final int M = A.getRowDimension();
            final int N = A.getColumnDimension();
            final int NB = B.getColumnDimension();
            if (A instanceof DenseMatrix) {
                final double[][] AData = ((DenseMatrix)A).getData();
                double[] ARow = null;
                if (B instanceof DenseMatrix) {
                    final double[][] BData = ((DenseMatrix)B).getData();
                    final double[] columnB = new double[B.getRowDimension()];
                    double s = 0.0;
                    for (int j = 0; j < NB; ++j) {
                        for (int r = 0; r < A.getRowDimension(); ++r) {
                            columnB[r] = BData[r][j];
                        }
                        for (int i = 0; i < M; ++i) {
                            ARow = AData[i];
                            s = v * C.getEntry(i, j);
                            for (int k = 0; k < N; ++k) {
                                s += ARow[k] * columnB[k];
                            }
                            resData[i][j] = s;
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
                    int r2 = -1;
                    double s2 = 0.0;
                    for (int l = 0; l < M; ++l) {
                        ARow = AData[l];
                        resRow = resData[l];
                        for (int m = 0; m < NB; ++m) {
                            s2 = v * C.getEntry(l, m);
                            for (int k2 = jc[m]; k2 < jc[m + 1]; ++k2) {
                                r2 = ir[k2];
                                s2 += ARow[r2] * pr[k2];
                            }
                            resRow[m] = s2;
                        }
                    }
                }
            }
            else if (A instanceof SparseMatrix) {
                final int[] ic = ((SparseMatrix)A).getIc();
                final int[] jr = ((SparseMatrix)A).getJr();
                final int[] valCSRIndices = ((SparseMatrix)A).getValCSRIndices();
                final double[] pr2 = ((SparseMatrix)A).getPr();
                if (B instanceof DenseMatrix) {
                    final double[][] BData2 = ((DenseMatrix)B).getData();
                    int c = -1;
                    double s2 = 0.0;
                    for (int l = 0; l < M; ++l) {
                        resRow = resData[l];
                        for (int m = 0; m < NB; ++m) {
                            s2 = v * C.getEntry(l, m);
                            for (int k2 = jr[l]; k2 < jr[l + 1]; ++k2) {
                                c = ic[k2];
                                s2 += pr2[valCSRIndices[k2]] * BData2[c][m];
                            }
                            resRow[m] = s2;
                        }
                    }
                }
                else if (B instanceof SparseMatrix) {
                    int[] ir2 = null;
                    int[] jc2 = null;
                    double[] pr3 = null;
                    ir2 = ((SparseMatrix)B).getIr();
                    jc2 = ((SparseMatrix)B).getJc();
                    pr3 = ((SparseMatrix)B).getPr();
                    int rr = -1;
                    int cl = -1;
                    double s3 = 0.0;
                    int kl = 0;
                    int kr = 0;
                    for (int i2 = 0; i2 < M; ++i2) {
                        resRow = resData[i2];
                        for (int j2 = 0; j2 < NB; ++j2) {
                            s3 = v * C.getEntry(i2, j2);
                            kl = jr[i2];
                            kr = jc2[j2];
                            while (kl < jr[i2 + 1] && kr < jc2[j2 + 1]) {
                                cl = ic[kl];
                                rr = ir2[kr];
                                if (cl < rr) {
                                    ++kl;
                                }
                                else if (cl > rr) {
                                    ++kr;
                                }
                                else {
                                    s3 += pr2[valCSRIndices[kl]] * pr3[kr];
                                    ++kl;
                                    ++kr;
                                }
                            }
                            resRow[j2] = s3;
                        }
                    }
                }
            }
        }
    }
    
    public static void affine(final Matrix res, final Matrix A, final Matrix B, final char operator, final Matrix C) {
        if (res instanceof SparseMatrix) {
            Printer.err("Sparse matrix for res is not supported.");
            Utility.exit(1);
        }
        else if (res instanceof DenseMatrix) {
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            final int M = A.getRowDimension();
            final int N = A.getColumnDimension();
            final int NB = B.getColumnDimension();
            if (A instanceof DenseMatrix) {
                final double[][] AData = ((DenseMatrix)A).getData();
                double[] ARow = null;
                if (B instanceof DenseMatrix) {
                    final double[][] BData = ((DenseMatrix)B).getData();
                    final double[] columnB = new double[B.getRowDimension()];
                    double s = 0.0;
                    for (int j = 0; j < NB; ++j) {
                        for (int r = 0; r < A.getRowDimension(); ++r) {
                            columnB[r] = BData[r][j];
                        }
                        for (int i = 0; i < M; ++i) {
                            ARow = AData[i];
                            if (operator == '+') {
                                s = C.getEntry(i, j);
                            }
                            else if (operator == '-') {
                                s = -C.getEntry(i, j);
                            }
                            for (int k = 0; k < N; ++k) {
                                s += ARow[k] * columnB[k];
                            }
                            resData[i][j] = s;
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
                    int r2 = -1;
                    double s2 = 0.0;
                    for (int l = 0; l < M; ++l) {
                        ARow = AData[l];
                        resRow = resData[l];
                        for (int m = 0; m < NB; ++m) {
                            if (operator == '+') {
                                s2 = C.getEntry(l, m);
                            }
                            else if (operator == '-') {
                                s2 = -C.getEntry(l, m);
                            }
                            for (int k2 = jc[m]; k2 < jc[m + 1]; ++k2) {
                                r2 = ir[k2];
                                s2 += ARow[r2] * pr[k2];
                            }
                            resRow[m] = s2;
                        }
                    }
                }
            }
            else if (A instanceof SparseMatrix) {
                final int[] ic = ((SparseMatrix)A).getIc();
                final int[] jr = ((SparseMatrix)A).getJr();
                final int[] valCSRIndices = ((SparseMatrix)A).getValCSRIndices();
                final double[] pr2 = ((SparseMatrix)A).getPr();
                if (B instanceof DenseMatrix) {
                    final double[][] BData2 = ((DenseMatrix)B).getData();
                    int c = -1;
                    double s2 = 0.0;
                    for (int l = 0; l < M; ++l) {
                        resRow = resData[l];
                        for (int m = 0; m < NB; ++m) {
                            if (operator == '+') {
                                s2 = C.getEntry(l, m);
                            }
                            else if (operator == '-') {
                                s2 = -C.getEntry(l, m);
                            }
                            for (int k2 = jr[l]; k2 < jr[l + 1]; ++k2) {
                                c = ic[k2];
                                s2 += pr2[valCSRIndices[k2]] * BData2[c][m];
                            }
                            resRow[m] = s2;
                        }
                    }
                }
                else if (B instanceof SparseMatrix) {
                    int[] ir2 = null;
                    int[] jc2 = null;
                    double[] pr3 = null;
                    ir2 = ((SparseMatrix)B).getIr();
                    jc2 = ((SparseMatrix)B).getJc();
                    pr3 = ((SparseMatrix)B).getPr();
                    int rr = -1;
                    int cl = -1;
                    double s3 = 0.0;
                    int kl = 0;
                    int kr = 0;
                    for (int i2 = 0; i2 < M; ++i2) {
                        resRow = resData[i2];
                        for (int j2 = 0; j2 < NB; ++j2) {
                            if (operator == '+') {
                                s3 = C.getEntry(i2, j2);
                            }
                            else if (operator == '-') {
                                s3 = -C.getEntry(i2, j2);
                            }
                            kl = jr[i2];
                            kr = jc2[j2];
                            while (kl < jr[i2 + 1] && kr < jc2[j2 + 1]) {
                                cl = ic[kl];
                                rr = ir2[kr];
                                if (cl < rr) {
                                    ++kl;
                                }
                                else if (cl > rr) {
                                    ++kr;
                                }
                                else {
                                    s3 += pr2[valCSRIndices[kl]] * pr3[kr];
                                    ++kl;
                                    ++kr;
                                }
                            }
                            resRow[j2] = s3;
                        }
                    }
                }
            }
        }
    }
    
    public static void affine(final Matrix res, final Matrix A, final Matrix B, final double v) {
        if (res instanceof SparseMatrix) {
            Printer.err("Sparse matrix for res is not supported.");
            Utility.exit(1);
        }
        else if (res instanceof DenseMatrix) {
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            final int M = A.getRowDimension();
            final int N = A.getColumnDimension();
            final int NB = B.getColumnDimension();
            if (A instanceof DenseMatrix) {
                final double[][] AData = ((DenseMatrix)A).getData();
                double[] ARow = null;
                if (B instanceof DenseMatrix) {
                    final double[][] BData = ((DenseMatrix)B).getData();
                    final double[] columnB = new double[B.getRowDimension()];
                    double s = 0.0;
                    for (int j = 0; j < NB; ++j) {
                        for (int r = 0; r < A.getRowDimension(); ++r) {
                            columnB[r] = BData[r][j];
                        }
                        for (int i = 0; i < M; ++i) {
                            ARow = AData[i];
                            s = v;
                            for (int k = 0; k < N; ++k) {
                                s += ARow[k] * columnB[k];
                            }
                            resData[i][j] = s;
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
                    int r2 = -1;
                    double s2 = 0.0;
                    for (int l = 0; l < M; ++l) {
                        ARow = AData[l];
                        resRow = resData[l];
                        for (int m = 0; m < NB; ++m) {
                            s2 = v;
                            for (int k2 = jc[m]; k2 < jc[m + 1]; ++k2) {
                                r2 = ir[k2];
                                s2 += ARow[r2] * pr[k2];
                            }
                            resRow[m] = s2;
                        }
                    }
                }
            }
            else if (A instanceof SparseMatrix) {
                final int[] ic = ((SparseMatrix)A).getIc();
                final int[] jr = ((SparseMatrix)A).getJr();
                final int[] valCSRIndices = ((SparseMatrix)A).getValCSRIndices();
                final double[] pr2 = ((SparseMatrix)A).getPr();
                if (B instanceof DenseMatrix) {
                    final double[][] BData2 = ((DenseMatrix)B).getData();
                    int c = -1;
                    double s2 = 0.0;
                    for (int l = 0; l < M; ++l) {
                        resRow = resData[l];
                        for (int m = 0; m < NB; ++m) {
                            s2 = v;
                            for (int k2 = jr[l]; k2 < jr[l + 1]; ++k2) {
                                c = ic[k2];
                                s2 += pr2[valCSRIndices[k2]] * BData2[c][m];
                            }
                            resRow[m] = s2;
                        }
                    }
                }
                else if (B instanceof SparseMatrix) {
                    int[] ir2 = null;
                    int[] jc2 = null;
                    double[] pr3 = null;
                    ir2 = ((SparseMatrix)B).getIr();
                    jc2 = ((SparseMatrix)B).getJc();
                    pr3 = ((SparseMatrix)B).getPr();
                    int rr = -1;
                    int cl = -1;
                    double s3 = 0.0;
                    int kl = 0;
                    int kr = 0;
                    for (int i2 = 0; i2 < M; ++i2) {
                        resRow = resData[i2];
                        for (int j2 = 0; j2 < NB; ++j2) {
                            s3 = v;
                            kl = jr[i2];
                            kr = jc2[j2];
                            while (kl < jr[i2 + 1] && kr < jc2[j2 + 1]) {
                                cl = ic[kl];
                                rr = ir2[kr];
                                if (cl < rr) {
                                    ++kl;
                                }
                                else if (cl > rr) {
                                    ++kr;
                                }
                                else {
                                    s3 += pr2[valCSRIndices[kl]] * pr3[kr];
                                    ++kl;
                                    ++kr;
                                }
                            }
                            resRow[j2] = s3;
                        }
                    }
                }
            }
        }
    }
    
    public static void affine(final Matrix res, final double a, final Matrix A, final double b, final Matrix B) {
        if (b == 0.0) {
            times(res, a, A);
            return;
        }
        if (b == 1.0) {
            affine(res, a, A, '+', B);
            return;
        }
        if (b == -1.0) {
            affine(res, a, A, '-', B);
            return;
        }
        if (a == 0.0) {
            times(res, b, B);
            return;
        }
        if (a == 1.0) {
            affine(res, b, B, '+', A);
            return;
        }
        if (a == -1.0) {
            affine(res, b, B, '-', A);
            return;
        }
        if (res instanceof DenseMatrix) {
            final int M = A.getRowDimension();
            final int N = A.getColumnDimension();
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            if (A instanceof DenseMatrix) {
                final double[][] AData = ((DenseMatrix)A).getData();
                double[] ARow = null;
                if (B instanceof DenseMatrix) {
                    final double[][] BData = ((DenseMatrix)B).getData();
                    double[] BRow = null;
                    for (int i = 0; i < M; ++i) {
                        ARow = AData[i];
                        BRow = BData[i];
                        resRow = resData[i];
                        for (int j = 0; j < N; ++j) {
                            resRow[j] = a * ARow[j] + b * BRow[j];
                        }
                    }
                }
                else if (B instanceof SparseMatrix) {
                    final int[] ic = ((SparseMatrix)B).getIc();
                    final int[] jr = ((SparseMatrix)B).getJr();
                    final int[] valCSRIndices = ((SparseMatrix)B).getValCSRIndices();
                    final double[] pr = ((SparseMatrix)B).getPr();
                    int k = 0;
                    for (int l = 0; l < M; ++l) {
                        ARow = AData[l];
                        resRow = resData[l];
                        times(resRow, a, ARow);
                        for (int m = jr[l]; m < jr[l + 1]; ++m) {
                            k = ic[m];
                            final double[] array = resRow;
                            final int n = k;
                            array[n] += b * pr[valCSRIndices[m]];
                        }
                    }
                }
            }
            else if (A instanceof SparseMatrix) {
                if (B instanceof DenseMatrix) {
                    final double[][] BData2 = ((DenseMatrix)A).getData();
                    double[] BRow2 = null;
                    final int[] ic = ((SparseMatrix)A).getIc();
                    final int[] jr = ((SparseMatrix)A).getJr();
                    final int[] valCSRIndices = ((SparseMatrix)A).getValCSRIndices();
                    final double[] pr = ((SparseMatrix)A).getPr();
                    int k = 0;
                    for (int l = 0; l < M; ++l) {
                        BRow2 = BData2[l];
                        resRow = resData[l];
                        times(resRow, b, BRow2);
                        for (int m = jr[l]; m < jr[l + 1]; ++m) {
                            k = ic[m];
                            final double[] array2 = resRow;
                            final int n2 = k;
                            array2[n2] += a * pr[valCSRIndices[m]];
                        }
                    }
                }
                else if (B instanceof SparseMatrix) {
                    res.clear();
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
                    double v = 0.0;
                    for (int j2 = 0; j2 < N; ++j2) {
                        k2 = jc1[j2];
                        k3 = jc2[j2];
                        if (k2 != jc1[j2 + 1] || k3 != jc2[j2 + 1]) {
                            while (k2 < jc1[j2 + 1] || k3 < jc2[j2 + 1]) {
                                if (k3 == jc2[j2 + 1]) {
                                    i2 = ir1[k2];
                                    v = a * pr2[k2];
                                    ++k2;
                                }
                                else if (k2 == jc1[j2 + 1]) {
                                    i2 = ir2[k3];
                                    v = b * pr3[k3];
                                    ++k3;
                                }
                                else {
                                    r1 = ir1[k2];
                                    r2 = ir2[k3];
                                    if (r1 < r2) {
                                        i2 = r1;
                                        v = a * pr2[k2];
                                        ++k2;
                                    }
                                    else if (r1 == r2) {
                                        i2 = r1;
                                        v = a * pr2[k2] + b * pr3[k3];
                                        ++k2;
                                        ++k3;
                                    }
                                    else {
                                        i2 = r2;
                                        v = b * pr3[k3];
                                        ++k3;
                                    }
                                }
                                if (v != 0.0) {
                                    resData[i2][j2] = v;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    public static void affine(final Matrix res, final double a, final Matrix A, final char operator, final Matrix B) {
        if (operator == '+') {
            if (a == 0.0) {
                assign(res, B);
                return;
            }
            if (a == 1.0) {
                plus(res, A, B);
                return;
            }
            if (a == -1.0) {
                minus(res, B, A);
                return;
            }
            if (res instanceof DenseMatrix) {
                final int M = A.getRowDimension();
                final int N = A.getColumnDimension();
                final double[][] resData = ((DenseMatrix)res).getData();
                double[] resRow = null;
                if (A instanceof DenseMatrix) {
                    final double[][] AData = ((DenseMatrix)A).getData();
                    double[] ARow = null;
                    if (B instanceof DenseMatrix) {
                        final double[][] BData = ((DenseMatrix)B).getData();
                        double[] BRow = null;
                        for (int i = 0; i < M; ++i) {
                            ARow = AData[i];
                            BRow = BData[i];
                            resRow = resData[i];
                            for (int j = 0; j < N; ++j) {
                                resRow[j] = a * ARow[j] + BRow[j];
                            }
                        }
                    }
                    else if (B instanceof SparseMatrix) {
                        final int[] ic = ((SparseMatrix)B).getIc();
                        final int[] jr = ((SparseMatrix)B).getJr();
                        final int[] valCSRIndices = ((SparseMatrix)B).getValCSRIndices();
                        final double[] pr = ((SparseMatrix)B).getPr();
                        int k = 0;
                        for (int l = 0; l < M; ++l) {
                            ARow = AData[l];
                            resRow = resData[l];
                            times(resRow, a, ARow);
                            for (int m = jr[l]; m < jr[l + 1]; ++m) {
                                k = ic[m];
                                final double[] array = resRow;
                                final int n = k;
                                array[n] += pr[valCSRIndices[m]];
                            }
                        }
                    }
                }
                else if (A instanceof SparseMatrix) {
                    if (B instanceof DenseMatrix) {
                        final double[][] BData2 = ((DenseMatrix)A).getData();
                        double[] BRow2 = null;
                        final int[] ic = ((SparseMatrix)A).getIc();
                        final int[] jr = ((SparseMatrix)A).getJr();
                        final int[] valCSRIndices = ((SparseMatrix)A).getValCSRIndices();
                        final double[] pr = ((SparseMatrix)A).getPr();
                        int k = 0;
                        for (int l = 0; l < M; ++l) {
                            BRow2 = BData2[l];
                            resRow = resData[l];
                            assign(resRow, BRow2);
                            for (int m = jr[l]; m < jr[l + 1]; ++m) {
                                k = ic[m];
                                final double[] array2 = resRow;
                                final int n2 = k;
                                array2[n2] += a * pr[valCSRIndices[m]];
                            }
                        }
                    }
                    else if (B instanceof SparseMatrix) {
                        res.clear();
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
                        double v = 0.0;
                        for (int j2 = 0; j2 < N; ++j2) {
                            k2 = jc1[j2];
                            k3 = jc2[j2];
                            if (k2 != jc1[j2 + 1] || k3 != jc2[j2 + 1]) {
                                while (k2 < jc1[j2 + 1] || k3 < jc2[j2 + 1]) {
                                    if (k3 == jc2[j2 + 1]) {
                                        i2 = ir1[k2];
                                        v = a * pr2[k2];
                                        ++k2;
                                    }
                                    else if (k2 == jc1[j2 + 1]) {
                                        i2 = ir2[k3];
                                        v = pr3[k3];
                                        ++k3;
                                    }
                                    else {
                                        r1 = ir1[k2];
                                        r2 = ir2[k3];
                                        if (r1 < r2) {
                                            i2 = r1;
                                            v = a * pr2[k2];
                                            ++k2;
                                        }
                                        else if (r1 == r2) {
                                            i2 = r1;
                                            v = a * pr2[k2] + pr3[k3];
                                            ++k2;
                                            ++k3;
                                        }
                                        else {
                                            i2 = r2;
                                            v = pr3[k3];
                                            ++k3;
                                        }
                                    }
                                    if (v != 0.0) {
                                        resData[i2][j2] = v;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        else if (operator == '-') {
            if (a == 0.0) {
                uminus(res, B);
                return;
            }
            if (a == 1.0) {
                minus(res, A, B);
                return;
            }
            if (res instanceof DenseMatrix) {
                final int M = A.getRowDimension();
                final int N = A.getColumnDimension();
                final double[][] resData = ((DenseMatrix)res).getData();
                double[] resRow = null;
                if (A instanceof DenseMatrix) {
                    final double[][] AData = ((DenseMatrix)A).getData();
                    double[] ARow = null;
                    if (B instanceof DenseMatrix) {
                        final double[][] BData = ((DenseMatrix)B).getData();
                        double[] BRow = null;
                        for (int i = 0; i < M; ++i) {
                            ARow = AData[i];
                            BRow = BData[i];
                            resRow = resData[i];
                            for (int j = 0; j < N; ++j) {
                                resRow[j] = a * ARow[j] - BRow[j];
                            }
                        }
                    }
                    else if (B instanceof SparseMatrix) {
                        final int[] ic = ((SparseMatrix)B).getIc();
                        final int[] jr = ((SparseMatrix)B).getJr();
                        final int[] valCSRIndices = ((SparseMatrix)B).getValCSRIndices();
                        final double[] pr = ((SparseMatrix)B).getPr();
                        int k = 0;
                        for (int l = 0; l < M; ++l) {
                            ARow = AData[l];
                            resRow = resData[l];
                            times(resRow, a, ARow);
                            for (int m = jr[l]; m < jr[l + 1]; ++m) {
                                k = ic[m];
                                final double[] array3 = resRow;
                                final int n3 = k;
                                array3[n3] -= pr[valCSRIndices[m]];
                            }
                        }
                    }
                }
                else if (A instanceof SparseMatrix) {
                    if (B instanceof DenseMatrix) {
                        final double[][] BData2 = ((DenseMatrix)A).getData();
                        double[] BRow2 = null;
                        final int[] ic = ((SparseMatrix)A).getIc();
                        final int[] jr = ((SparseMatrix)A).getJr();
                        final int[] valCSRIndices = ((SparseMatrix)A).getValCSRIndices();
                        final double[] pr = ((SparseMatrix)A).getPr();
                        int k = 0;
                        for (int l = 0; l < M; ++l) {
                            BRow2 = BData2[l];
                            resRow = resData[l];
                            uminus(resRow, BRow2);
                            for (int m = jr[l]; m < jr[l + 1]; ++m) {
                                k = ic[m];
                                final double[] array4 = resRow;
                                final int n4 = k;
                                array4[n4] += a * pr[valCSRIndices[m]];
                            }
                        }
                    }
                    else if (B instanceof SparseMatrix) {
                        res.clear();
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
                        double v = 0.0;
                        for (int j2 = 0; j2 < N; ++j2) {
                            k2 = jc1[j2];
                            k3 = jc2[j2];
                            if (k2 != jc1[j2 + 1] || k3 != jc2[j2 + 1]) {
                                while (k2 < jc1[j2 + 1] || k3 < jc2[j2 + 1]) {
                                    if (k3 == jc2[j2 + 1]) {
                                        i2 = ir1[k2];
                                        v = a * pr2[k2];
                                        ++k2;
                                    }
                                    else if (k2 == jc1[j2 + 1]) {
                                        i2 = ir2[k3];
                                        v = -pr3[k3];
                                        ++k3;
                                    }
                                    else {
                                        r1 = ir1[k2];
                                        r2 = ir2[k3];
                                        if (r1 < r2) {
                                            i2 = r1;
                                            v = a * pr2[k2];
                                            ++k2;
                                        }
                                        else if (r2 < r1) {
                                            i2 = r2;
                                            v = -pr3[k3];
                                            ++k3;
                                        }
                                        else {
                                            i2 = r1;
                                            v = a * pr2[k2] - pr3[k3];
                                            ++k2;
                                            ++k3;
                                        }
                                    }
                                    if (v != 0.0) {
                                        resData[i2][j2] = v;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    public static void affine(final Matrix res, final Matrix A, final double b, final Matrix B) {
        affine(res, b, B, '+', A);
    }
    
    public static void affine(final Matrix res, final double a, final Matrix A, final double b) {
        if (a == 1.0) {
            plus(res, A, b);
            return;
        }
        if (b == 0.0) {
            times(res, a, A);
            return;
        }
        final int M = res.getRowDimension();
        final int N = res.getColumnDimension();
        if (M != A.getRowDimension() || N != A.getColumnDimension()) {
            System.err.println("Dimension doesn't match.");
            System.exit(1);
        }
        if (!(res instanceof SparseMatrix) && res instanceof DenseMatrix) {
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            if (A instanceof DenseMatrix) {
                final double[][] AData = ((DenseMatrix)A).getData();
                double[] ARow = null;
                for (int i = 0; i < M; ++i) {
                    resRow = resData[i];
                    ARow = AData[i];
                    for (int j = 0; j < N; ++j) {
                        resRow[j] = a * ARow[j] + b;
                    }
                }
            }
            else if (A instanceof SparseMatrix) {
                final int[] ic = ((SparseMatrix)A).getIc();
                final int[] jr = ((SparseMatrix)A).getJr();
                final int[] valCSRIndices = ((SparseMatrix)A).getValCSRIndices();
                final double[] pr = ((SparseMatrix)A).getPr();
                for (int k = 0; k < M; ++k) {
                    resRow = resData[k];
                    if (jr[k] == jr[k + 1]) {
                        ArrayOperator.assignVector(resRow, b);
                    }
                    else {
                        int lastColumnIdx = -1;
                        int currentColumnIdx = 0;
                        for (int l = jr[k]; l < jr[k + 1]; ++l) {
                            currentColumnIdx = ic[l];
                            for (int c = lastColumnIdx + 1; c < currentColumnIdx; ++c) {
                                resRow[c] = b;
                            }
                            resRow[currentColumnIdx] = a * pr[valCSRIndices[l]] + b;
                            lastColumnIdx = currentColumnIdx;
                        }
                        for (int c2 = lastColumnIdx + 1; c2 < N; ++c2) {
                            resRow[c2] = b;
                        }
                    }
                }
            }
        }
    }
    
    public static void plus(final Matrix res, final Matrix A, final Matrix B) {
        if (res instanceof SparseMatrix) {
            ((SparseMatrix)res).assignSparseMatrix(Matlab.sparse(A.plus(B)));
        }
        else if (res instanceof DenseMatrix) {
            final int M = A.getRowDimension();
            final int N = A.getColumnDimension();
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            if (A instanceof DenseMatrix) {
                final double[][] AData = ((DenseMatrix)A).getData();
                double[] ARow = null;
                if (B instanceof DenseMatrix) {
                    final double[][] BData = ((DenseMatrix)B).getData();
                    double[] BRow = null;
                    for (int i = 0; i < M; ++i) {
                        ARow = AData[i];
                        BRow = BData[i];
                        resRow = resData[i];
                        for (int j = 0; j < N; ++j) {
                            resRow[j] = ARow[j] + BRow[j];
                        }
                    }
                }
                else if (B instanceof SparseMatrix) {
                    final int[] ic = ((SparseMatrix)B).getIc();
                    final int[] jr = ((SparseMatrix)B).getJr();
                    final int[] valCSRIndices = ((SparseMatrix)B).getValCSRIndices();
                    final double[] pr = ((SparseMatrix)B).getPr();
                    int k = 0;
                    for (int l = 0; l < M; ++l) {
                        ARow = AData[l];
                        resRow = resData[l];
                        assign(resRow, ARow);
                        for (int m = jr[l]; m < jr[l + 1]; ++m) {
                            k = ic[m];
                            final double[] array = resRow;
                            final int n = k;
                            array[n] += pr[valCSRIndices[m]];
                        }
                    }
                }
            }
            else if (A instanceof SparseMatrix) {
                if (B instanceof DenseMatrix) {
                    plus(res, B, A);
                }
                else if (B instanceof SparseMatrix) {
                    res.clear();
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
                    double v = 0.0;
                    for (int j2 = 0; j2 < N; ++j2) {
                        k2 = jc1[j2];
                        k3 = jc2[j2];
                        if (k2 != jc1[j2 + 1] || k3 != jc2[j2 + 1]) {
                            while (k2 < jc1[j2 + 1] || k3 < jc2[j2 + 1]) {
                                if (k3 == jc2[j2 + 1]) {
                                    i2 = ir1[k2];
                                    v = pr2[k2];
                                    ++k2;
                                }
                                else if (k2 == jc1[j2 + 1]) {
                                    i2 = ir2[k3];
                                    v = pr3[k3];
                                    ++k3;
                                }
                                else {
                                    r1 = ir1[k2];
                                    r2 = ir2[k3];
                                    if (r1 < r2) {
                                        i2 = r1;
                                        v = pr2[k2];
                                        ++k2;
                                    }
                                    else if (r1 == r2) {
                                        i2 = r1;
                                        v = pr2[k2] + pr3[k3];
                                        ++k2;
                                        ++k3;
                                    }
                                    else {
                                        i2 = r2;
                                        v = pr3[k3];
                                        ++k3;
                                    }
                                }
                                if (v != 0.0) {
                                    resData[i2][j2] = v;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    public static void plus(final Matrix res, final Matrix A, final double v) {
        if (v == 0.0) {
            assign(res, A);
            return;
        }
        final int M = res.getRowDimension();
        final int N = res.getColumnDimension();
        if (M != A.getRowDimension() || N != A.getColumnDimension()) {
            System.err.println("Dimension doesn't match.");
            System.exit(1);
        }
        if (!(res instanceof SparseMatrix) && res instanceof DenseMatrix) {
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            if (A instanceof DenseMatrix) {
                final double[][] AData = ((DenseMatrix)A).getData();
                double[] ARow = null;
                for (int i = 0; i < M; ++i) {
                    resRow = resData[i];
                    ARow = AData[i];
                    for (int j = 0; j < N; ++j) {
                        resRow[j] = ARow[j] + v;
                    }
                }
            }
            else if (A instanceof SparseMatrix) {
                final int[] ic = ((SparseMatrix)A).getIc();
                final int[] jr = ((SparseMatrix)A).getJr();
                final int[] valCSRIndices = ((SparseMatrix)A).getValCSRIndices();
                final double[] pr = ((SparseMatrix)A).getPr();
                for (int k = 0; k < M; ++k) {
                    resRow = resData[k];
                    if (jr[k] == jr[k + 1]) {
                        ArrayOperator.assignVector(resRow, v);
                    }
                    else {
                        int lastColumnIdx = -1;
                        int currentColumnIdx = 0;
                        for (int l = jr[k]; l < jr[k + 1]; ++l) {
                            currentColumnIdx = ic[l];
                            for (int c = lastColumnIdx + 1; c < currentColumnIdx; ++c) {
                                resRow[c] = v;
                            }
                            resRow[currentColumnIdx] = pr[valCSRIndices[l]] + v;
                            lastColumnIdx = currentColumnIdx;
                        }
                        for (int c2 = lastColumnIdx + 1; c2 < N; ++c2) {
                            resRow[c2] = v;
                        }
                    }
                }
            }
        }
    }
    
    public static void plusAssign(final Matrix res, final Matrix A) {
        final int M = res.getRowDimension();
        final int N = res.getColumnDimension();
        if (M != A.getRowDimension() || N != A.getColumnDimension()) {
            System.err.println("Dimension doesn't match.");
            System.exit(1);
        }
        if (!(res instanceof SparseMatrix) && res instanceof DenseMatrix) {
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            if (A instanceof DenseMatrix) {
                final double[][] AData = ((DenseMatrix)A).getData();
                double[] ARow = null;
                for (int i = 0; i < M; ++i) {
                    resRow = resData[i];
                    ARow = AData[i];
                    for (int j = 0; j < N; ++j) {
                        final double[] array = resRow;
                        final int n = j;
                        array[n] += ARow[j];
                    }
                }
            }
            else if (A instanceof SparseMatrix) {
                final int[] ic = ((SparseMatrix)A).getIc();
                final int[] jr = ((SparseMatrix)A).getJr();
                final int[] valCSRIndices = ((SparseMatrix)A).getValCSRIndices();
                final double[] pr = ((SparseMatrix)A).getPr();
                int k = 0;
                for (int l = 0; l < M; ++l) {
                    resRow = resData[l];
                    for (int m = jr[l]; m < jr[l + 1]; ++m) {
                        k = ic[m];
                        final double[] array2 = resRow;
                        final int n2 = k;
                        array2[n2] += pr[valCSRIndices[m]];
                    }
                }
            }
        }
    }
    
    public static void plusAssign(final Matrix res, final double v) {
        final int M = res.getRowDimension();
        final int N = res.getColumnDimension();
        if (!(res instanceof SparseMatrix) && res instanceof DenseMatrix) {
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            for (int i = 0; i < M; ++i) {
                resRow = resData[i];
                for (int j = 0; j < N; ++j) {
                    final double[] array = resRow;
                    final int n = j;
                    array[n] += v;
                }
            }
        }
    }
    
    public static void plusAssign(final Matrix res, final double a, final Matrix A) {
        final int M = res.getRowDimension();
        final int N = res.getColumnDimension();
        if (M != A.getRowDimension() || N != A.getColumnDimension()) {
            System.err.println("Dimension doesn't match.");
            System.exit(1);
        }
        if (a == 0.0) {
            return;
        }
        if (a == 1.0) {
            plusAssign(res, A);
            return;
        }
        if (a == -1.0) {
            minusAssign(res, A);
            return;
        }
        if (!(res instanceof SparseMatrix) && res instanceof DenseMatrix) {
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            if (A instanceof DenseMatrix) {
                final double[][] AData = ((DenseMatrix)A).getData();
                double[] ARow = null;
                for (int i = 0; i < M; ++i) {
                    resRow = resData[i];
                    ARow = AData[i];
                    for (int j = 0; j < N; ++j) {
                        final double[] array = resRow;
                        final int n = j;
                        array[n] += a * ARow[j];
                    }
                }
            }
            else if (A instanceof SparseMatrix) {
                final int[] ic = ((SparseMatrix)A).getIc();
                final int[] jr = ((SparseMatrix)A).getJr();
                final int[] valCSRIndices = ((SparseMatrix)A).getValCSRIndices();
                final double[] pr = ((SparseMatrix)A).getPr();
                int k = 0;
                for (int l = 0; l < M; ++l) {
                    resRow = resData[l];
                    for (int m = jr[l]; m < jr[l + 1]; ++m) {
                        k = ic[m];
                        final double[] array2 = resRow;
                        final int n2 = k;
                        array2[n2] += a * pr[valCSRIndices[m]];
                    }
                }
            }
        }
    }
    
    public static void minus(final Matrix res, final Matrix A, final Matrix B) {
        if (res instanceof SparseMatrix) {
            ((SparseMatrix)res).assignSparseMatrix(Matlab.sparse(A.minus(B)));
        }
        else if (res instanceof DenseMatrix) {
            final int M = A.getRowDimension();
            final int N = A.getColumnDimension();
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            if (A instanceof DenseMatrix) {
                final double[][] AData = ((DenseMatrix)A).getData();
                double[] ARow = null;
                if (B instanceof DenseMatrix) {
                    final double[][] BData = ((DenseMatrix)B).getData();
                    double[] BRow = null;
                    for (int i = 0; i < M; ++i) {
                        ARow = AData[i];
                        BRow = BData[i];
                        resRow = resData[i];
                        for (int j = 0; j < N; ++j) {
                            resRow[j] = ARow[j] - BRow[j];
                        }
                    }
                }
                else if (B instanceof SparseMatrix) {
                    final int[] ic = ((SparseMatrix)B).getIc();
                    final int[] jr = ((SparseMatrix)B).getJr();
                    final int[] valCSRIndices = ((SparseMatrix)B).getValCSRIndices();
                    final double[] pr = ((SparseMatrix)B).getPr();
                    int k = 0;
                    for (int l = 0; l < M; ++l) {
                        ARow = AData[l];
                        resRow = resData[l];
                        assign(resRow, ARow);
                        for (int m = jr[l]; m < jr[l + 1]; ++m) {
                            k = ic[m];
                            final double[] array = resRow;
                            final int n = k;
                            array[n] -= pr[valCSRIndices[m]];
                        }
                    }
                }
            }
            else if (A instanceof SparseMatrix) {
                if (B instanceof DenseMatrix) {
                    final double[][] BData2 = ((DenseMatrix)A).getData();
                    double[] BRow2 = null;
                    final int[] ic = ((SparseMatrix)A).getIc();
                    final int[] jr = ((SparseMatrix)A).getJr();
                    final int[] valCSRIndices = ((SparseMatrix)A).getValCSRIndices();
                    final double[] pr = ((SparseMatrix)A).getPr();
                    int k = 0;
                    for (int l = 0; l < M; ++l) {
                        BRow2 = BData2[l];
                        resRow = resData[l];
                        uminus(resRow, BRow2);
                        for (int m = jr[l]; m < jr[l + 1]; ++m) {
                            k = ic[m];
                            final double[] array2 = resRow;
                            final int n2 = k;
                            array2[n2] += pr[valCSRIndices[m]];
                        }
                    }
                }
                else if (B instanceof SparseMatrix) {
                    res.clear();
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
                    double v = 0.0;
                    for (int j2 = 0; j2 < N; ++j2) {
                        k2 = jc1[j2];
                        k3 = jc2[j2];
                        if (k2 != jc1[j2 + 1] || k3 != jc2[j2 + 1]) {
                            while (k2 < jc1[j2 + 1] || k3 < jc2[j2 + 1]) {
                                if (k3 == jc2[j2 + 1]) {
                                    i2 = ir1[k2];
                                    v = pr2[k2];
                                    ++k2;
                                }
                                else if (k2 == jc1[j2 + 1]) {
                                    i2 = ir2[k3];
                                    v = -pr3[k3];
                                    ++k3;
                                }
                                else {
                                    r1 = ir1[k2];
                                    r2 = ir2[k3];
                                    if (r1 < r2) {
                                        i2 = r1;
                                        v = pr2[k2];
                                        ++k2;
                                    }
                                    else if (r2 < r1) {
                                        i2 = r2;
                                        v = -pr3[k3];
                                        ++k3;
                                    }
                                    else {
                                        i2 = r1;
                                        v = pr2[k2] - pr3[k3];
                                        ++k2;
                                        ++k3;
                                    }
                                }
                                if (v != 0.0) {
                                    resData[i2][j2] = v;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    public static void minus(final Matrix res, final Matrix A, final double v) {
        final int M = res.getRowDimension();
        final int N = res.getColumnDimension();
        if (M != A.getRowDimension() || N != A.getColumnDimension()) {
            System.err.println("Dimension doesn't match.");
            System.exit(1);
        }
        if (!(res instanceof SparseMatrix) && res instanceof DenseMatrix) {
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            if (A instanceof DenseMatrix) {
                final double[][] AData = ((DenseMatrix)A).getData();
                double[] ARow = null;
                for (int i = 0; i < M; ++i) {
                    resRow = resData[i];
                    ARow = AData[i];
                    for (int j = 0; j < N; ++j) {
                        resRow[j] = ARow[j] - v;
                    }
                }
            }
            else if (A instanceof SparseMatrix) {
                final int[] ic = ((SparseMatrix)A).getIc();
                final int[] jr = ((SparseMatrix)A).getJr();
                final int[] valCSRIndices = ((SparseMatrix)A).getValCSRIndices();
                final double[] pr = ((SparseMatrix)A).getPr();
                for (int k = 0; k < M; ++k) {
                    resRow = resData[k];
                    if (jr[k] == jr[k + 1]) {
                        ArrayOperator.assignVector(resRow, -v);
                    }
                    else {
                        int lastColumnIdx = -1;
                        int currentColumnIdx = 0;
                        for (int l = jr[k]; l < jr[k + 1]; ++l) {
                            currentColumnIdx = ic[l];
                            for (int c = lastColumnIdx + 1; c < currentColumnIdx; ++c) {
                                resRow[c] = -v;
                            }
                            resRow[currentColumnIdx] = pr[valCSRIndices[l]] - v;
                            lastColumnIdx = currentColumnIdx;
                        }
                        for (int c2 = lastColumnIdx + 1; c2 < N; ++c2) {
                            resRow[c2] = -v;
                        }
                    }
                }
            }
        }
    }
    
    public static void minus(final Matrix res, final double v, final Matrix A) {
        final int M = res.getRowDimension();
        final int N = res.getColumnDimension();
        if (M != A.getRowDimension() || N != A.getColumnDimension()) {
            System.err.println("Dimension doesn't match.");
            System.exit(1);
        }
        if (!(res instanceof SparseMatrix) && res instanceof DenseMatrix) {
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            if (A instanceof DenseMatrix) {
                final double[][] AData = ((DenseMatrix)A).getData();
                double[] ARow = null;
                for (int i = 0; i < M; ++i) {
                    resRow = resData[i];
                    ARow = AData[i];
                    for (int j = 0; j < N; ++j) {
                        resRow[j] = v - ARow[j];
                    }
                }
            }
            else if (A instanceof SparseMatrix) {
                final int[] ic = ((SparseMatrix)A).getIc();
                final int[] jr = ((SparseMatrix)A).getJr();
                final int[] valCSRIndices = ((SparseMatrix)A).getValCSRIndices();
                final double[] pr = ((SparseMatrix)A).getPr();
                for (int k = 0; k < M; ++k) {
                    resRow = resData[k];
                    if (jr[k] == jr[k + 1]) {
                        ArrayOperator.assignVector(resRow, v);
                    }
                    else {
                        int lastColumnIdx = -1;
                        int currentColumnIdx = 0;
                        for (int l = jr[k]; l < jr[k + 1]; ++l) {
                            currentColumnIdx = ic[l];
                            for (int c = lastColumnIdx + 1; c < currentColumnIdx; ++c) {
                                resRow[c] = v;
                            }
                            resRow[currentColumnIdx] = v - pr[valCSRIndices[l]];
                            lastColumnIdx = currentColumnIdx;
                        }
                        for (int c2 = lastColumnIdx + 1; c2 < N; ++c2) {
                            resRow[c2] = v;
                        }
                    }
                }
            }
        }
    }
    
    public static void minusAssign(final Matrix res, final Matrix A) {
        final int M = res.getRowDimension();
        final int N = res.getColumnDimension();
        if (M != A.getRowDimension() || N != A.getColumnDimension()) {
            System.err.println("Dimension doesn't match.");
            System.exit(1);
        }
        if (!(res instanceof SparseMatrix) && res instanceof DenseMatrix) {
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            if (A instanceof DenseMatrix) {
                final double[][] AData = ((DenseMatrix)A).getData();
                double[] ARow = null;
                for (int i = 0; i < M; ++i) {
                    resRow = resData[i];
                    ARow = AData[i];
                    for (int j = 0; j < N; ++j) {
                        final double[] array = resRow;
                        final int n = j;
                        array[n] -= ARow[j];
                    }
                }
            }
            else if (A instanceof SparseMatrix) {
                final int[] ic = ((SparseMatrix)A).getIc();
                final int[] jr = ((SparseMatrix)A).getJr();
                final int[] valCSRIndices = ((SparseMatrix)A).getValCSRIndices();
                final double[] pr = ((SparseMatrix)A).getPr();
                int k = 0;
                for (int l = 0; l < M; ++l) {
                    resRow = resData[l];
                    for (int m = jr[l]; m < jr[l + 1]; ++m) {
                        k = ic[m];
                        final double[] array2 = resRow;
                        final int n2 = k;
                        array2[n2] -= pr[valCSRIndices[m]];
                    }
                }
            }
        }
    }
    
    public static void minusAssign(final Matrix res, final double v) {
        if (v == 0.0) {
            return;
        }
        final int M = res.getRowDimension();
        final int N = res.getColumnDimension();
        if (!(res instanceof SparseMatrix) && res instanceof DenseMatrix) {
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            for (int i = 0; i < M; ++i) {
                resRow = resData[i];
                for (int j = 0; j < N; ++j) {
                    final double[] array = resRow;
                    final int n = j;
                    array[n] -= v;
                }
            }
        }
    }
    
    public static void minusAssign(final Matrix res, final double a, final Matrix A) {
        final int M = res.getRowDimension();
        final int N = res.getColumnDimension();
        if (M != A.getRowDimension() || N != A.getColumnDimension()) {
            System.err.println("Dimension doesn't match.");
            System.exit(1);
        }
        if (a == 0.0) {
            return;
        }
        if (a == 1.0) {
            minusAssign(res, A);
            return;
        }
        if (a == -1.0) {
            plusAssign(res, A);
            return;
        }
        if (!(res instanceof SparseMatrix) && res instanceof DenseMatrix) {
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            if (A instanceof DenseMatrix) {
                final double[][] AData = ((DenseMatrix)A).getData();
                double[] ARow = null;
                for (int i = 0; i < M; ++i) {
                    resRow = resData[i];
                    ARow = AData[i];
                    for (int j = 0; j < N; ++j) {
                        final double[] array = resRow;
                        final int n = j;
                        array[n] -= a * ARow[j];
                    }
                }
            }
            else if (A instanceof SparseMatrix) {
                final int[] ic = ((SparseMatrix)A).getIc();
                final int[] jr = ((SparseMatrix)A).getJr();
                final int[] valCSRIndices = ((SparseMatrix)A).getValCSRIndices();
                final double[] pr = ((SparseMatrix)A).getPr();
                int k = 0;
                for (int l = 0; l < M; ++l) {
                    resRow = resData[l];
                    for (int m = jr[l]; m < jr[l + 1]; ++m) {
                        k = ic[m];
                        final double[] array2 = resRow;
                        final int n2 = k;
                        array2[n2] -= a * pr[valCSRIndices[m]];
                    }
                }
            }
        }
    }
    
    public static void log(final Matrix res, final Matrix A) {
        final int M = res.getRowDimension();
        final int N = res.getColumnDimension();
        if (M != A.getRowDimension() || N != A.getColumnDimension()) {
            System.err.println("Dimension doesn't match.");
            System.exit(1);
        }
        if (!(res instanceof SparseMatrix) && res instanceof DenseMatrix) {
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            if (A instanceof DenseMatrix) {
                final double[][] AData = ((DenseMatrix)A).getData();
                double[] ARow = null;
                for (int i = 0; i < M; ++i) {
                    resRow = resData[i];
                    ARow = AData[i];
                    for (int j = 0; j < N; ++j) {
                        resRow[j] = Math.log(ARow[j]);
                    }
                }
            }
        }
    }
    
    public static void log(final double[] res, final double[] V) {
        for (int i = 0; i < res.length; ++i) {
            res[i] = Math.log(V[i]);
        }
    }
    
    public static void logAssign(final double[] res) {
        for (int i = 0; i < res.length; ++i) {
            res[i] = Math.log(res[i]);
        }
    }
    
    public static void logAssign(final Matrix res) {
        final int M = res.getRowDimension();
        final int N = res.getColumnDimension();
        if (!(res instanceof SparseMatrix) && res instanceof DenseMatrix) {
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            for (int i = 0; i < M; ++i) {
                resRow = resData[i];
                for (int j = 0; j < N; ++j) {
                    resRow[j] = Math.log(resRow[j]);
                }
            }
        }
    }
    
    public static void exp(final Matrix res, final Matrix A) {
        final int M = res.getRowDimension();
        final int N = res.getColumnDimension();
        if (M != A.getRowDimension() || N != A.getColumnDimension()) {
            System.err.println("Dimension doesn't match.");
            System.exit(1);
        }
        if (!(res instanceof SparseMatrix) && res instanceof DenseMatrix) {
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            if (A instanceof DenseMatrix) {
                final double[][] AData = ((DenseMatrix)A).getData();
                double[] ARow = null;
                for (int i = 0; i < M; ++i) {
                    resRow = resData[i];
                    ARow = AData[i];
                    for (int j = 0; j < N; ++j) {
                        resRow[j] = Math.exp(ARow[j]);
                    }
                }
            }
        }
    }
    
    public static void expAssign(final Matrix res) {
        final int M = res.getRowDimension();
        final int N = res.getColumnDimension();
        if (res instanceof SparseMatrix) {
            Printer.err("The expAssign routine doesn't support sparse matrix.");
            Utility.exit(1);
        }
        else if (res instanceof DenseMatrix) {
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            for (int i = 0; i < M; ++i) {
                resRow = resData[i];
                for (int j = 0; j < N; ++j) {
                    resRow[j] = Math.exp(resRow[j]);
                }
            }
        }
    }
    
    public static void sigmoid(final Matrix res, final Matrix A) {
        if (res instanceof DenseMatrix) {
            assign(res, A);
            final double[][] data = ((DenseMatrix)res).getData();
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
        }
        else {
            Printer.err("Sorry, sparse matrix is not support for res.");
            Utility.exit(1);
        }
    }
}
