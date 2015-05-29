package ml.utils;

import la.matrix.*;
import la.vector.*;

public class Printer
{
    public static void printSparseMatrix(final Matrix A, final int p) {
        if (!(A instanceof SparseMatrix)) {
            System.err.println("SparseMatrix input is expected.");
            return;
        }
        if (((SparseMatrix)A).getNNZ() == 0) {
            System.out.println("Empty sparse matrix.");
            System.out.println();
            return;
        }
        final int nRow = A.getRowDimension();
        final int nCol = A.getColumnDimension();
        final String leftFormat = String.format("  %%%ds, ", String.valueOf(nRow).length() + 1);
        final String rightFormat = String.format("%%-%ds", String.valueOf(nCol).length() + 2);
        final String format = String.valueOf(leftFormat) + rightFormat + sprintf("%%%ds", 8 + p - 4);
        final SparseMatrix S = (SparseMatrix)A;
        final int[] ir = S.getIr();
        final int[] jc = S.getJc();
        final double[] pr = S.getPr();
        final int N = S.getColumnDimension();
        String valueString = "";
        int i = -1;
        for (int j = 0; j < N; ++j) {
            for (int k = jc[j]; k < jc[j + 1]; ++k) {
                System.out.print("  ");
                i = ir[k];
                final double v = pr[k];
                final int rv = (int)Math.round(v);
                if (v != rv) {
                    valueString = sprintf(sprintf("%%.%df", p), v);
                }
                else {
                    valueString = sprintf("%d", rv);
                }
                final String leftString = String.format("(%d", i + 1);
                final String rightString = String.format("%d)", j + 1);
                System.out.println(String.format(format, leftString, rightString, valueString));
            }
        }
        System.out.println();
    }
    
    public static void printSparseMatrix(final Matrix A) {
        printSparseMatrix(A, 4);
    }
    
    public static void printDenseMatrix(final Matrix A, final int p) {
        if (!(A instanceof DenseMatrix)) {
            System.err.println("DenseMatrix input is expected.");
            return;
        }
        if (((DenseMatrix)A).getData() == null) {
            System.out.println("Empty matrix.");
            return;
        }
        for (int i = 0; i < A.getRowDimension(); ++i) {
            System.out.print("  ");
            for (int j = 0; j < A.getColumnDimension(); ++j) {
                String valueString = "";
                final double v = A.getEntry(i, j);
                final int rv = (int)Math.round(v);
                if (v != rv) {
                    valueString = sprintf(sprintf("%%.%df", p), v);
                }
                else {
                    valueString = sprintf("%d", rv);
                }
                System.out.print(sprintf(sprintf("%%%ds", 8 + p - 4), valueString));
                System.out.print("  ");
            }
            System.out.println();
        }
        System.out.println();
    }
    
    public static void printDenseMatrix(final Matrix A) {
        printDenseMatrix(A, 4);
    }
    
    public static void printMatrix(final Matrix A, final int p) {
        if (A == null) {
            System.out.println("Empty matrix.");
            return;
        }
        if (!(A instanceof SparseMatrix)) {
            if (A instanceof DenseMatrix) {
                if (((DenseMatrix)A).getData() == null) {
                    System.out.println("Empty matrix.");
                    return;
                }
                for (int i = 0; i < A.getRowDimension(); ++i) {
                    System.out.print("  ");
                    for (int j = 0; j < A.getColumnDimension(); ++j) {
                        String valueString = "";
                        final double v = A.getEntry(i, j);
                        final int rv = (int)Math.round(v);
                        if (v != rv) {
                            valueString = sprintf(sprintf("%%.%df", p), v);
                        }
                        else {
                            valueString = sprintf("%d", rv);
                        }
                        System.out.print(sprintf(sprintf("%%%ds", 8 + p - 4), valueString));
                        System.out.print("  ");
                    }
                    System.out.println();
                }
                System.out.println();
            }
            return;
        }
        if (((SparseMatrix)A).getNNZ() == 0) {
            System.out.println("Empty sparse matrix.");
            return;
        }
        final SparseMatrix S = (SparseMatrix)A;
        final int[] ic = S.getIc();
        final int[] jr = S.getJr();
        final double[] pr = S.getPr();
        final int[] valCSRIndices = S.getValCSRIndices();
        final int M = S.getRowDimension();
        String valueString2 = "";
        for (int r = 0; r < M; ++r) {
            System.out.print("  ");
            int currentColumn = 0;
            int lastColumn = -1;
            for (int k = jr[r]; k < jr[r + 1]; ++k) {
                for (currentColumn = ic[k]; lastColumn < currentColumn - 1; ++lastColumn) {
                    System.out.printf(String.format("%%%ds", 8 + p - 4), " ");
                    System.out.print("  ");
                }
                lastColumn = currentColumn;
                final double v2 = pr[valCSRIndices[k]];
                final int rv2 = (int)Math.round(v2);
                if (v2 != rv2) {
                    valueString2 = sprintf(sprintf("%%.%df", p), v2);
                }
                else {
                    valueString2 = sprintf("%d", rv2);
                }
                System.out.printf(sprintf("%%%ds", 8 + p - 4), valueString2);
                System.out.print("  ");
            }
            System.out.println();
        }
        System.out.println();
    }
    
    public static void printMatrix(final Matrix A) {
        printMatrix(A, 4);
    }
    
    public static void printMatrix(final double[] V, final int p) {
        for (int i = 0; i < V.length; ++i) {
            System.out.print("  ");
            String valueString = "";
            final double v = V[i];
            final int rv = (int)Math.round(v);
            if (v != rv) {
                valueString = sprintf(sprintf("%%.%df", p), v);
            }
            else {
                valueString = sprintf("%d", rv);
            }
            System.out.print(sprintf(sprintf("%%%ds", 8 + p - 4), valueString));
            System.out.print("  ");
            System.out.println();
        }
        System.out.println();
    }
    
    public static void printMatrix(final double[] V) {
        printMatrix(V, 4);
    }
    
    public static void printVector(final double[] V, final int p) {
        for (int i = 0; i < V.length; ++i) {
            System.out.print("  ");
            String valueString = "";
            final double v = V[i];
            final int rv = (int)Math.round(v);
            if (v != rv) {
                valueString = sprintf(sprintf("%%.%df", p), v);
            }
            else {
                valueString = sprintf("%d", rv);
            }
            System.out.print(sprintf(sprintf("%%%ds", 8 + p - 4), valueString));
        }
        System.out.println();
        System.out.println();
    }
    
    public static void printVector(final double[] V) {
        printVector(V, 4);
    }
    
    public static void printVector(final Vector V, final int p) {
        if (V instanceof DenseVector) {
            printDenseVector(V, p);
        }
        else {
            printSparseVector(V, p);
        }
    }
    
    public static void printVector(final Vector V) {
        printVector(V, 4);
    }
    
    public static void printDenseVector(final Vector V, final int p) {
        if (V instanceof DenseVector) {
            final int dim = V.getDim();
            final double[] pr = ((DenseVector)V).getPr();
            for (int k = 0; k < dim; ++k) {
                System.out.print("  ");
                final double v = pr[k];
                final int rv = (int)Math.round(v);
                String valueString;
                if (v != rv) {
                    valueString = sprintf(sprintf("%%.%df", p), v);
                }
                else {
                    valueString = sprintf("%d", rv);
                }
                System.out.print(sprintf(sprintf("%%%ds", 8 + p - 4), valueString));
                System.out.println();
            }
            System.out.println();
        }
        else {
            System.err.println("The input vector should be a DenseVector instance");
            System.exit(1);
        }
    }
    
    public static void printDenseVector(final Vector V) {
        printDenseVector(V, 4);
    }
    
    public static void printSparseVector(final Vector V, final int p) {
        if (V instanceof SparseVector) {
            final int[] ir = ((SparseVector)V).getIr();
            final double[] pr = ((SparseVector)V).getPr();
            for (int nnz = ((SparseVector)V).getNNZ(), k = 0; k < nnz; ++k) {
                System.out.print("  ");
                final int idx = ir[k];
                final double v = pr[k];
                final int rv = (int)Math.round(v);
                String valueString;
                if (v != rv) {
                    valueString = sprintf(sprintf("%%.%df", p), v);
                }
                else {
                    valueString = sprintf("%d", rv);
                }
                System.out.print(sprintf(sprintf("(%d, 1)%%%ds", idx + 1, 8 + p - 4), valueString));
                System.out.println();
            }
            System.out.println();
        }
        else {
            System.err.println("The input vector should be a SparseVector instance");
            System.exit(1);
        }
    }
    
    public static void printSparseVector(final Vector V) {
        printSparseVector(V, 4);
    }
    
    public static void display(final Vector V, final int p) {
        printVector(V, p);
    }
    
    public static void display(final Vector V) {
        display(V, 4);
    }
    
    public static void display(final double[] V, final int p) {
        printVector(new DenseVector(V), p);
    }
    
    public static void display(final double[] V) {
        display(V, 4);
    }
    
    public static void display(final Matrix A, final int p) {
        if (A instanceof DenseMatrix) {
            printDenseMatrix(A, p);
        }
        else if (A instanceof SparseMatrix) {
            printSparseMatrix(A, p);
        }
    }
    
    public static void display(final Matrix A) {
        display(A, 4);
    }
    
    public static void display(final double[][] A, final int p) {
        printMatrix(new DenseMatrix(A), p);
    }
    
    public static void display(final double[][] A) {
        display(A, 4);
    }
    
    public static void disp(final Vector V, final int p) {
        display(V, p);
    }
    
    public static void disp(final Vector V) {
        display(V, 4);
    }
    
    public static void disp(final double[] V, final int p) {
        display(new DenseVector(V), p);
    }
    
    public static void disp(final double[] V) {
        display(new DenseVector(V), 4);
    }
    
    public static void disp(final Matrix A, final int p) {
        display(A, p);
    }
    
    public static void disp(final Matrix A) {
        display(A, 4);
    }
    
    public static void disp(final double[][] A, final int p) {
        display(new DenseMatrix(A), p);
    }
    
    public static void disp(final double[][] A) {
        display(new DenseMatrix(A), 4);
    }
    
    public static void disp(final double v) {
        System.out.print("  ");
        System.out.println(v);
    }
    
    public static void disp(final int[] V) {
        display(V);
    }
    
    public static void disp(final int[][] M) {
        display(M);
    }
    
    public static void display(final int[] V) {
        if (V == null) {
            System.out.println("Empty vector!");
            return;
        }
        for (int i = 0; i < V.length; ++i) {
            System.out.print("  ");
            String valueString = "";
            final double v = V[i];
            final int rv = (int)Math.round(v);
            if (v != rv) {
                valueString = String.format("%.4f", v);
            }
            else {
                valueString = String.format("%d", rv);
            }
            System.out.print(String.format("%7s", valueString));
            System.out.print("  ");
        }
        System.out.println();
    }
    
    public static void display(final int[][] M) {
        if (M == null) {
            System.out.println("Empty matrix!");
            return;
        }
        for (int i = 0; i < M.length; ++i) {
            System.out.print("  ");
            for (int j = 0; j < M[0].length; ++j) {
                String valueString = "";
                final double v = M[i][j];
                final int rv = (int)Math.round(v);
                if (v != rv) {
                    valueString = String.format("%.4f", v);
                }
                else {
                    valueString = String.format("%d", rv);
                }
                System.out.print(String.format("%7s", valueString));
                System.out.print("  ");
            }
            System.out.println();
        }
        System.out.println();
    }
    
    public static void display(final String str) {
        fprintf("%s%n", str);
    }
    
    public static void disp(final String str) {
        fprintf("%s%n", str);
    }
    
    public static String sprintf(final String format, final Object... os) {
        return String.format(format, os);
    }
    
    public static void fprintf(final String format, final Object... os) {
        System.out.format(format, os);
    }
    
    public static void printf(final String format, final Object... os) {
        System.out.format(format, os);
    }
    
    public static void print(final String content) {
        System.out.print(content);
    }
    
    public static void print(final char c) {
        System.out.print(c);
    }
    
    public static void print(final char[] s) {
        System.out.print(s);
    }
    
    public static void print(final int[] A) {
        final int n = A.length;
        for (int i = 0; i < n - 1; ++i) {
            System.out.print(A[i]);
            System.out.print(' ');
        }
        System.out.print(A[n - 1]);
    }
    
    public static void print(final Object obj) {
        System.out.print(obj);
    }
    
    public static void println(final String content) {
        System.out.println(content);
    }
    
    public static void println(final char[] s) {
        System.out.println(s);
    }
    
    public static void println(final int[] A) {
        final int n = A.length;
        for (int i = 0; i < n - 1; ++i) {
            System.out.print(A[i]);
            System.out.print(' ');
        }
        System.out.println(A[n - 1]);
    }
    
    public static void println(final Object obj) {
        System.out.println(obj);
    }
    
    public static void println() {
        System.out.println();
    }
    
    public static void err(final String input) {
        System.err.println(input);
    }
    
    public static void errf(final String format, final Object... os) {
        System.err.format(format, os);
    }
    
    public static void showSpec(final DenseVector V, final String[] spec) {
        showSpec(V, spec, 4);
    }
    
    public static void showSpec(final DenseVector V, final String[] spec, final int p) {
        if (V instanceof DenseVector) {
            final int dim = V.getDim();
            final double[] pr = V.getPr();
            for (int k = 0; k < dim; ++k) {
                print("  ");
                final double v = pr[k];
                final int rv = (int)Math.round(v);
                String valueString;
                if (v != rv) {
                    valueString = sprintf(sprintf("%%.%df", p), v);
                }
                else {
                    valueString = sprintf("%d", rv);
                }
                println(sprintf(sprintf("%%%ds  %%s", 8 + p - 4), valueString, spec[k]));
            }
            println();
        }
        else {
            System.err.println("The input vector should be a DenseVector instance");
            Utility.exit(1);
        }
    }
}
