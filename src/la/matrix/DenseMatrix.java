package la.matrix;

import java.io.*;
import la.vector.*;
import ml.utils.*;

public class DenseMatrix implements Matrix, Serializable
{
    private static final long serialVersionUID = 6821454132254344419L;
    private int M;
    private int N;
    private double[][] data;
    
    public static void main(final String[] args) {
        final int M = 1000;
        final int N = 1000;
        final Matrix A = new DenseMatrix(M, N, 1.0);
        final Matrix B = new DenseMatrix(M, N, 1.0);
        System.out.println("1000 x 1000 matrix multiplication test.");
        final long start = System.currentTimeMillis();
        A.mtimes(B);
        System.out.format(String.valueOf(System.getProperty("line.separator")) + "Elapsed time: %.3f seconds.\n", (System.currentTimeMillis() - start) / 1000.0f);
    }
    
    public DenseMatrix() {
        this.M = 0;
        this.N = 0;
        this.data = null;
    }
    
    public DenseMatrix(final double v) {
        this.M = 1;
        this.N = 1;
        this.data = new double[][] { { v } };
    }
    
    public DenseMatrix(final int M, final int N) {
        this.data = new double[M][];
        for (int i = 0; i < M; ++i) {
            this.data[i] = new double[N];
            for (int j = 0; j < N; ++j) {
                this.data[i][j] = 0.0;
            }
        }
        this.M = M;
        this.N = N;
    }
    
    public DenseMatrix(final int[] size) {
        if (size.length != 2) {
            System.err.println("The input integer array should have exactly two entries!");
            System.exit(1);
        }
        final int M = size[0];
        final int N = size[1];
        this.data = new double[M][];
        for (int i = 0; i < M; ++i) {
            this.data[i] = new double[N];
            for (int j = 0; j < N; ++j) {
                this.data[i][j] = 0.0;
            }
        }
        this.M = M;
        this.N = N;
    }
    
    public DenseMatrix(final double[][] data) {
        this.data = data;
        this.M = data.length;
        this.N = ((this.M > 0) ? data[0].length : 0);
    }
    
    public DenseMatrix(final double[] data, final int dim) {
        if (dim == 1) {
            this.M = data.length;
            this.N = 1;
            this.data = new double[this.M][];
            for (int i = 0; i < this.M; ++i) {
                (this.data[i] = new double[this.N])[0] = data[i];
            }
        }
        else if (dim == 2) {
            this.M = 1;
            this.N = data.length;
            (this.data = new double[this.M][])[0] = data;
        }
    }
    
    public DenseMatrix(final int M, final int N, final double v) {
        this.data = new double[M][];
        for (int i = 0; i < M; ++i) {
            this.data[i] = new double[N];
            for (int j = 0; j < N; ++j) {
                this.data[i][j] = v;
            }
        }
        this.M = M;
        this.N = N;
    }
    
    public DenseMatrix(final int[] size, final double v) {
        if (size.length != 2) {
            System.err.println("The input integer array should have exactly two entries!");
            System.exit(1);
        }
        final int M = size[0];
        final int N = size[1];
        this.data = new double[M][];
        for (int i = 0; i < M; ++i) {
            this.data[i] = new double[N];
            for (int j = 0; j < N; ++j) {
                this.data[i][j] = v;
            }
        }
        this.M = M;
        this.N = N;
    }
    
    @Override
    public double[][] getData() {
        return this.data;
    }
    
    @Override
    public int getRowDimension() {
        return this.M;
    }
    
    @Override
    public int getColumnDimension() {
        return this.N;
    }
    
    @Override
    public Matrix mtimes(final Matrix A) {
        Matrix res = null;
        final double[][] resData = new double[this.M][];
        final int NA = A.getColumnDimension();
        for (int i = 0; i < this.M; ++i) {
            resData[i] = new double[NA];
        }
        double[] rowData = null;
        if (A instanceof DenseMatrix) {
            final double[][] AData = ((DenseMatrix)A).getData();
            final double[] columnA = new double[A.getRowDimension()];
            double s = 0.0;
            for (int j = 0; j < NA; ++j) {
                for (int r = 0; r < A.getRowDimension(); ++r) {
                    columnA[r] = AData[r][j];
                }
                for (int k = 0; k < this.M; ++k) {
                    rowData = this.data[k];
                    s = 0.0;
                    for (int l = 0; l < this.N; ++l) {
                        s += rowData[l] * columnA[l];
                    }
                    resData[k][j] = s;
                }
            }
        }
        else if (A instanceof SparseMatrix) {
            int[] ir = null;
            int[] jc = null;
            double[] pr = null;
            ir = ((SparseMatrix)A).getIr();
            jc = ((SparseMatrix)A).getJc();
            pr = ((SparseMatrix)A).getPr();
            int r2 = -1;
            double s2 = 0.0;
            for (int m = 0; m < this.M; ++m) {
                rowData = this.data[m];
                for (int j2 = 0; j2 < NA; ++j2) {
                    s2 = 0.0;
                    for (int k2 = jc[j2]; k2 < jc[j2 + 1]; ++k2) {
                        r2 = ir[k2];
                        s2 += rowData[r2] * pr[k2];
                    }
                    resData[m][j2] = s2;
                }
            }
        }
        res = new DenseMatrix(resData);
        return res;
    }
    
    @Override
    public double getEntry(final int r, final int c) {
        return this.data[r][c];
    }
    
    @Override
    public void setEntry(final int r, final int c, final double v) {
        this.data[r][c] = v;
    }
    
    @Override
    public Matrix transpose() {
        final double[][] resData = new double[this.N][];
        for (int i = 0; i < this.N; ++i) {
            resData[i] = new double[this.M];
            for (int j = 0; j < this.M; ++j) {
                resData[i][j] = this.data[j][i];
            }
        }
        return new DenseMatrix(resData);
    }
    
    @Override
    public Matrix plus(final Matrix A) {
        if (A.getRowDimension() != this.M || A.getColumnDimension() != this.N) {
            System.err.println("Dimension doesn't match.");
            return null;
        }
        final DenseMatrix res = (DenseMatrix)this.copy();
        if (A instanceof DenseMatrix) {
            final double[][] AData = ((DenseMatrix)A).getData();
            double[] ARow = null;
            double[] row = null;
            for (int i = 0; i < this.M; ++i) {
                row = res.data[i];
                ARow = AData[i];
                for (int j = 0; j < this.N; ++j) {
                    final double[] array = row;
                    final int n = j;
                    array[n] += ARow[j];
                }
            }
        }
        else if (A instanceof SparseMatrix) {
            int[] ir = null;
            int[] jc = null;
            double[] pr = null;
            ir = ((SparseMatrix)A).getIr();
            jc = ((SparseMatrix)A).getJc();
            pr = ((SparseMatrix)A).getPr();
            int r = -1;
            for (int j = 0; j < A.getColumnDimension(); ++j) {
                for (int k = jc[j]; k < jc[j + 1]; ++k) {
                    r = ir[k];
                    final double[] array2 = res.data[r];
                    final int n2 = j;
                    array2[n2] += pr[k];
                }
            }
        }
        return res;
    }
    
    @Override
    public Matrix minus(final Matrix A) {
        if (A.getRowDimension() != this.M || A.getColumnDimension() != this.N) {
            System.err.println("Dimension doesn't match.");
            return null;
        }
        final DenseMatrix res = (DenseMatrix)this.copy();
        if (A instanceof DenseMatrix) {
            for (int i = 0; i < this.M; ++i) {
                for (int j = 0; j < this.N; ++j) {
                    final double[] array = res.data[i];
                    final int n = j;
                    array[n] -= ((DenseMatrix)A).data[i][j];
                }
            }
        }
        else if (A instanceof SparseMatrix) {
            int[] ir = null;
            int[] jc = null;
            double[] pr = null;
            ir = ((SparseMatrix)A).getIr();
            jc = ((SparseMatrix)A).getJc();
            pr = ((SparseMatrix)A).getPr();
            int r = -1;
            for (int k = 0; k < A.getColumnDimension(); ++k) {
                for (int l = jc[k]; l < jc[k + 1]; ++l) {
                    r = ir[l];
                    final double[] array2 = res.data[r];
                    final int n2 = k;
                    array2[n2] -= pr[l];
                }
            }
        }
        return res;
    }
    
    @Override
    public Matrix times(final Matrix A) {
        if (A.getRowDimension() != this.M || A.getColumnDimension() != this.N) {
            System.err.println("Dimension doesn't match.");
            return null;
        }
        final double[][] resData = ArrayOperator.allocate2DArray(this.M, this.N, 0.0);
        if (A instanceof DenseMatrix) {
            final double[][] AData = ((DenseMatrix)A).getData();
            double[] ARow = null;
            double[] thisRow = null;
            double[] resRow = null;
            for (int i = 0; i < this.M; ++i) {
                thisRow = this.data[i];
                ARow = AData[i];
                resRow = resData[i];
                thisRow = this.data[i];
                for (int j = 0; j < this.N; ++j) {
                    resRow[j] = thisRow[j] * ARow[j];
                }
            }
        }
        else if (A instanceof SparseMatrix) {
            int[] ir = null;
            int[] jc = null;
            double[] pr = null;
            ir = ((SparseMatrix)A).getIr();
            jc = ((SparseMatrix)A).getJc();
            pr = ((SparseMatrix)A).getPr();
            int r = -1;
            for (int k = 0; k < A.getColumnDimension(); ++k) {
                for (int l = jc[k]; l < jc[k + 1]; ++l) {
                    r = ir[l];
                    resData[r][k] = this.data[r][k] * pr[l];
                }
            }
        }
        return new DenseMatrix(resData);
    }
    
    @Override
    public Matrix times(final double v) {
        final DenseMatrix res = (DenseMatrix)this.copy();
        for (int i = 0; i < this.M; ++i) {
            for (int j = 0; j < this.N; ++j) {
                final double[] array = res.data[i];
                final int n = j;
                array[n] *= v;
            }
        }
        return res;
    }
    
    @Override
    public Matrix copy() {
        final DenseMatrix res = new DenseMatrix();
        res.M = this.M;
        res.N = this.N;
        res.data = new double[this.M][];
        for (int i = 0; i < this.M; ++i) {
            res.data[i] = this.data[i].clone();
        }
        return res;
    }
    
    public Matrix clone() {
        return this.copy();
    }
    
    @Override
    public Matrix plus(final double v) {
        final DenseMatrix res = (DenseMatrix)this.copy();
        for (int i = 0; i < this.M; ++i) {
            for (int j = 0; j < this.N; ++j) {
                final double[] array = res.data[i];
                final int n = j;
                array[n] += v;
            }
        }
        return res;
    }
    
    @Override
    public Matrix minus(final double v) {
        final DenseMatrix res = (DenseMatrix)this.copy();
        for (int i = 0; i < this.M; ++i) {
            for (int j = 0; j < this.N; ++j) {
                final double[] array = res.data[i];
                final int n = j;
                array[n] -= v;
            }
        }
        return res;
    }
    
    @Override
    public Vector operate(final Vector b) {
        if (this.N != b.getDim()) {
            System.err.println("Dimension does not match.");
            System.exit(1);
        }
        final double[] V = new double[this.M];
        if (b instanceof DenseVector) {
            ArrayOperator.operate(V, this.data, ((DenseVector)b).getPr());
        }
        else if (b instanceof SparseVector) {
            final int[] ir = ((SparseVector)b).getIr();
            final double[] pr = ((SparseVector)b).getPr();
            final int nnz = ((SparseVector)b).getNNZ();
            int idx = 0;
            double[] row_i = null;
            for (int i = 0; i < this.M; ++i) {
                row_i = this.data[i];
                double s = 0.0;
                for (int k = 0; k < nnz; ++k) {
                    idx = ir[k];
                    s += row_i[idx] * pr[k];
                }
                V[i] = s;
            }
        }
        return new DenseVector(V);
    }
    
    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder(100);
        if (this.data == null) {
            sb.append("Empty matrix." + System.lineSeparator());
            return sb.toString();
        }
        final int p = 4;
        for (int i = 0; i < this.getRowDimension(); ++i) {
            sb.append("  ");
            for (int j = 0; j < this.getColumnDimension(); ++j) {
                String valueString = "";
                final double v = this.getEntry(i, j);
                final int rv = (int)Math.round(v);
                if (v != rv) {
                    valueString = Printer.sprintf(Printer.sprintf("%%.%df", p), v);
                }
                else {
                    valueString = Printer.sprintf("%d", rv);
                }
                sb.append(Printer.sprintf(Printer.sprintf("%%%ds", 8 + p - 4), valueString));
                sb.append("  ");
            }
            sb.append(System.lineSeparator());
        }
        return sb.toString();
    }
    
    @Override
    public void clear() {
        ArrayOperator.clearMatrix(this.data);
    }
    
    @Override
    public Matrix getSubMatrix(final int startRow, final int endRow, final int startColumn, final int endColumn) {
        final int nRow = endRow - startRow + 1;
        final int nCol = endColumn - startColumn + 1;
        final double[][] resData = new double[nRow][];
        double[] resRow = null;
        double[] thisRow = null;
        for (int r = 0, i = startRow; r < nRow; ++r, ++i) {
            resRow = new double[nCol];
            thisRow = this.data[i];
            System.arraycopy(thisRow, startColumn, resRow, 0, nCol);
            resData[r] = resRow;
        }
        return new DenseMatrix(resData);
    }
    
    @Override
    public Matrix getSubMatrix(final int[] selectedRows, final int[] selectedColumns) {
        final int nRow = selectedRows.length;
        final int nCol = selectedColumns.length;
        final double[][] resData = new double[nRow][];
        double[] resRow = null;
        double[] thisRow = null;
        for (int r = 0; r < nRow; ++r) {
            resRow = new double[nCol];
            thisRow = this.data[selectedRows[r]];
            for (int c = 0; c < nCol; ++c) {
                resRow[c] = thisRow[selectedColumns[c]];
            }
            resData[r] = resRow;
        }
        return new DenseMatrix(resData);
    }
    
    @Override
    public Matrix getColumnMatrix(final int c) {
        final DenseMatrix res = new DenseMatrix(this.M, 1);
        final double[][] resData = res.data;
        for (int i = 0; i < this.M; ++i) {
            resData[i][0] = this.data[i][c];
        }
        return res;
    }
    
    @Override
    public Vector getColumnVector(final int c) {
        final DenseVector res = new DenseVector(this.M);
        final double[] pr = res.getPr();
        for (int i = 0; i < this.M; ++i) {
            pr[i] = this.data[i][c];
        }
        return res;
    }
    
    @Override
    public Matrix getRowMatrix(final int r) {
        return new DenseMatrix(this.data[r], 2);
    }
    
    @Override
    public Vector getRowVector(final int r) {
        return new DenseVector(this.data[r]);
    }
    
    @Override
    public Matrix getRows(final int startRow, final int endRow) {
        final int numRows = endRow - startRow + 1;
        final double[][] resData = new double[numRows][];
        for (int r = startRow, i = 0; r <= endRow; ++r, ++i) {
            resData[i] = this.data[r].clone();
        }
        return new DenseMatrix(resData);
    }
    
    @Override
    public Matrix getRows(final int... selectedRows) {
        final int numRows = selectedRows.length;
        final double[][] resData = new double[numRows][];
        for (int i = 0; i < numRows; ++i) {
            resData[i] = this.data[selectedRows[i]].clone();
        }
        return new DenseMatrix(resData);
    }
    
    @Override
    public Vector[] getRowVectors(final int startRow, final int endRow) {
        final int numRows = endRow - startRow + 1;
        final Vector[] res = new DenseVector[numRows];
        for (int r = startRow, i = 0; r <= endRow; ++r, ++i) {
            res[i] = new DenseVector(this.data[r]);
        }
        return res;
    }
    
    @Override
    public Vector[] getRowVectors(final int... selectedRows) {
        final int numRows = selectedRows.length;
        final Vector[] res = new DenseVector[numRows];
        for (int i = 0; i < numRows; ++i) {
            res[i] = new DenseVector(this.data[selectedRows[i]]);
        }
        return res;
    }
    
    @Override
    public Matrix getColumns(final int startColumn, final int endColumn) {
        final int nRow = this.M;
        final int nCol = endColumn - startColumn + 1;
        final double[][] resData = new double[nRow][];
        double[] resRow = null;
        double[] thisRow = null;
        for (int r = 0; r < nRow; ++r) {
            resRow = new double[nCol];
            thisRow = this.data[r];
            System.arraycopy(thisRow, startColumn, resRow, 0, nCol);
            resData[r] = resRow;
        }
        return new DenseMatrix(resData);
    }
    
    @Override
    public Matrix getColumns(final int... selectedColumns) {
        final int nRow = this.M;
        final int nCol = selectedColumns.length;
        final double[][] resData = new double[nRow][];
        double[] resRow = null;
        double[] thisRow = null;
        for (int r = 0; r < nRow; ++r) {
            resRow = new double[nCol];
            thisRow = this.data[r];
            for (int c = 0; c < nCol; ++c) {
                resRow[c] = thisRow[selectedColumns[c]];
            }
            resData[r] = resRow;
        }
        return new DenseMatrix(resData);
    }
    
    @Override
    public Vector[] getColumnVectors(final int startColumn, final int endColumn) {
        final int numColumns = endColumn - startColumn + 1;
        final Vector[] res = new DenseVector[numColumns];
        for (int c = startColumn, i = 0; c <= endColumn; ++c, ++i) {
            res[i] = this.getColumnVector(c);
        }
        return res;
    }
    
    @Override
    public Vector[] getColumnVectors(final int... selectedColumns) {
        final int numColumns = selectedColumns.length;
        final Vector[] res = new DenseVector[numColumns];
        for (int j = 0; j < numColumns; ++j) {
            res[j] = this.getColumnVector(selectedColumns[j]);
        }
        return res;
    }
    
    @Override
    public void setRowMatrix(final int r, final Matrix A) {
        if (A.getRowDimension() != 1) {
            Printer.err("Input matrix should be a row matrix.");
            Utility.exit(1);
        }
        final double[] thisRow = this.data[r];
        if (A instanceof DenseMatrix) {
            final double[] ARow = ((DenseMatrix)A).data[0];
            System.arraycopy(ARow, 0, thisRow, 0, this.N);
        }
        else if (A instanceof SparseMatrix) {
            final int[] jc = ((SparseMatrix)A).getJc();
            final double[] pr = ((SparseMatrix)A).getPr();
            for (int j = 0; j < this.N; ++j) {
                if (jc[j + 1] == jc[j]) {
                    thisRow[j] = 0.0;
                }
                else {
                    thisRow[j] = pr[jc[j]];
                }
            }
        }
    }
    
    @Override
    public void setRowVector(final int r, final Vector V) {
        final double[] thisRow = this.data[r];
        if (V instanceof DenseVector) {
            final double[] pr = ((DenseVector)V).getPr();
            System.arraycopy(pr, 0, thisRow, 0, this.N);
        }
        else if (V instanceof SparseVector) {
            final int[] ir = ((SparseVector)V).getIr();
            final double[] pr2 = ((SparseVector)V).getPr();
            final int nnz = ((SparseVector)V).getNNZ();
            int lastIdx = -1;
            int currentIdx = 0;
            for (int k = 0; k < nnz; ++k) {
                currentIdx = ir[k];
                for (int j = lastIdx + 1; j < currentIdx; ++j) {
                    thisRow[j] = 0.0;
                }
                thisRow[currentIdx] = pr2[k];
                lastIdx = currentIdx;
            }
            for (int i = lastIdx + 1; i < this.N; ++i) {
                thisRow[i] = 0.0;
            }
        }
    }
    
    @Override
    public void setColumnMatrix(final int c, final Matrix A) {
        if (A.getColumnDimension() != 1) {
            Printer.err("Input matrix should be a column matrix.");
            Utility.exit(1);
        }
        if (A instanceof DenseMatrix) {
            final double[][] AData = ((DenseMatrix)A).data;
            for (int i = 0; i < this.M; ++i) {
                this.data[i][c] = AData[i][0];
            }
        }
        else if (A instanceof SparseMatrix) {
            final int[] jc = ((SparseMatrix)A).getJc();
            if (jc[1] == 0) {
                for (int i = 0; i < this.M; ++i) {
                    this.data[i][c] = 0.0;
                }
                return;
            }
            final int[] ir = ((SparseMatrix)A).getIr();
            final double[] pr = ((SparseMatrix)A).getPr();
            int lastIdx = -1;
            int currentIdx = 0;
            for (int k = 0; k < jc[1]; ++k) {
                currentIdx = ir[k];
                for (int j = lastIdx + 1; j < currentIdx; ++j) {
                    this.data[j][c] = 0.0;
                }
                this.data[currentIdx][c] = pr[k];
                lastIdx = currentIdx;
            }
            for (int l = lastIdx + 1; l < this.M; ++l) {
                this.data[l][c] = 0.0;
            }
        }
    }
    
    @Override
    public void setColumnVector(final int c, final Vector V) {
        if (V instanceof DenseVector) {
            final double[] pr = ((DenseVector)V).getPr();
            for (int i = 0; i < this.M; ++i) {
                this.data[i][c] = pr[i];
            }
        }
        else if (V instanceof SparseVector) {
            final int[] ir = ((SparseVector)V).getIr();
            final double[] pr2 = ((SparseVector)V).getPr();
            final int nnz = ((SparseVector)V).getNNZ();
            int lastIdx = -1;
            int currentIdx = 0;
            for (int k = 0; k < nnz; ++k) {
                currentIdx = ir[k];
                for (int j = lastIdx + 1; j < currentIdx; ++j) {
                    this.data[j][c] = 0.0;
                }
                this.data[currentIdx][c] = pr2[k];
                lastIdx = currentIdx;
            }
            for (int l = lastIdx + 1; l < this.M; ++l) {
                this.data[l][c] = 0.0;
            }
        }
    }
}
