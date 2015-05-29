package la.vector;

import ml.utils.*;
import la.matrix.*;

public class Test
{
    public static void main(final String[] args) {
        final Vector V1 = new DenseVector(new double[] { 2.0, 0.0, 4.0 });
        final Vector V2 = new DenseVector(new double[] { 1.0, 2.0, 0.0 });
        Printer.fprintf("V1 .* V2:%n", new Object[0]);
        Printer.disp(Matlab.times(V1, V2));
        Printer.fprintf("V1 .* sparse(V2):%n", new Object[0]);
        Printer.disp(Matlab.times(V1, Matlab.sparse(V2)));
        Printer.fprintf("sparse(V1) .* sparse(V2):%n", new Object[0]);
        Printer.disp(Matlab.times(Matlab.sparse(V1), Matlab.sparse(V2)));
        Printer.fprintf("V1 + V2:%n", new Object[0]);
        Printer.disp(Matlab.plus(V1, V2));
        Printer.fprintf("V1 + sparse(V2):%n", new Object[0]);
        Printer.disp(Matlab.plus(V1, Matlab.sparse(V2)));
        Printer.fprintf("sparse(V1) + sparse(V2):%n", new Object[0]);
        Printer.disp(Matlab.plus(Matlab.sparse(V1), Matlab.sparse(V2)));
        Printer.fprintf("V1 - V2:%n", new Object[0]);
        Printer.disp(Matlab.minus(V1, V2));
        Printer.fprintf("V1 - sparse(V2):%n", new Object[0]);
        Printer.disp(Matlab.minus(V1, Matlab.sparse(V2)));
        Printer.fprintf("sparse(V1) - sparse(V2):%n", new Object[0]);
        Printer.disp(Matlab.minus(Matlab.sparse(V1), Matlab.sparse(V2)));
        final int dim = 4;
        final Vector V3 = new SparseVector(dim);
        for (int i = 0; i < dim; ++i) {
            Printer.fprintf("V(%d):\t%.2f%n", i + 1, V3.get(i));
        }
        V3.set(3, 4.5);
        Printer.fprintf("V(%d):\t%.2f%n", 4, V3.get(3));
        V3.set(1, 2.3);
        Printer.fprintf("V(%d):\t%.2f%n", 2, V3.get(1));
        V3.set(1, 3.2);
        Printer.fprintf("V(%d):\t%.2f%n", 2, V3.get(1));
        V3.set(3, 2.5);
        Printer.fprintf("V(%d):\t%.2f%n", 4, V3.get(3));
        Printer.fprintf("V:%n", new Object[0]);
        Printer.disp(V3);
        for (int i = 0; i < dim; ++i) {
            Printer.fprintf("V(%d):\t%.2f%n", i + 1, V3.get(i));
        }
        Matrix A = null;
        final int[] rIndices = { 0, 1, 3, 1, 2, 2, 3, 2, 3 };
        final int[] cIndices = { 0, 0, 0, 1, 1, 2, 2, 3, 3 };
        final double[] values = { 10.0, 3.2, 3.0, 9.0, 7.0, 8.0, 8.0, 7.0, 7.0 };
        final int numRows = 4;
        final int numColumns = 4;
        final int nzmax = rIndices.length;
        A = new SparseMatrix(rIndices, cIndices, values, numRows, numColumns, nzmax);
        Printer.fprintf("A:%n", new Object[0]);
        Printer.printMatrix(A);
        Printer.fprintf("AV:%n", new Object[0]);
        Printer.disp(A.operate(V3));
        Printer.fprintf("V'A':%n", new Object[0]);
        Printer.disp(V3.operate(A.transpose()));
        Printer.disp(A.operate(Matlab.full(V3)));
        Printer.disp(Matlab.full(A).operate(V3));
        Printer.disp(Matlab.full(A).operate(Matlab.full(V3)));
        final SparseVector V4 = new SparseVector(1411);
        V4.set(67, 1.0);
        V4.set(1291, 0.7514);
        final int k = 0;
        final double xk = V4.get(k);
        System.out.println(xk);
    }
}
