package la.vector;

import java.io.*;
import ml.utils.*;
import java.util.*;
import la.matrix.*;

public class DenseVector implements Vector, Serializable
{
    private static final long serialVersionUID = -6411390717530519480L;
    private double[] pr;
    
    public DenseVector() {
    }
    
    public DenseVector(final int dim, final double v) {
        this.pr = ArrayOperator.allocateVector(dim, v);
    }
    
    public DenseVector(final int dim) {
        this.pr = ArrayOperator.allocateVector(dim, 0.0);
    }
    
    public DenseVector(final double[] pr) {
        this.pr = pr;
    }
    
    public static DenseVector buildDenseVector(final double[] pr) {
        final DenseVector res = new DenseVector();
        res.pr = pr;
        return res;
    }
    
    public double[] getPr() {
        return this.pr;
    }
    
    @Override
    public String toString() {
        final StringBuffer sb = new StringBuffer(100);
        sb.append('[');
        for (int k = 0; k < this.pr.length; ++k) {
            sb.append(String.format("%.4f", this.pr[k]));
            if (k < this.pr.length - 1) {
                sb.append(", ");
            }
        }
        sb.append(']');
        return sb.toString();
    }
    
    public static void main(final String[] args) {
    }
    
    @Override
    public int getDim() {
        return this.pr.length;
    }
    
    @Override
    public Vector copy() {
        return new DenseVector(this.pr.clone());
    }
    
    public Vector clone() {
        return this.copy();
    }
    
    @Override
    public Vector times(final Vector V) {
        if (V.getDim() != this.pr.length) {
            System.err.println("Dimension doesn't match.");
            System.exit(1);
        }
        if (V instanceof DenseVector) {
            return new DenseVector(ArrayOperator.times(this.pr, ((DenseVector)V).getPr()));
        }
        if (V instanceof SparseVector) {
            final ArrayList<Pair<Integer, Double>> list = new ArrayList<Pair<Integer, Double>>();
            final int[] ir = ((SparseVector)V).getIr();
            final double[] pr = ((SparseVector)V).getPr();
            int idx = -1;
            double v = 0.0;
            for (int k = 0; k < ((SparseVector)V).getNNZ(); ++k) {
                idx = ir[k];
                v = this.pr[idx] * pr[k];
                if (v != 0.0) {
                    list.add(Pair.of(idx, v));
                }
            }
            final int nnz = list.size();
            final int dim = this.getDim();
            final int[] ir_res = new int[nnz];
            final double[] pr_res = new double[nnz];
            int i = 0;
            for (final Pair<Integer, Double> pair : list) {
                ir_res[i] = pair.first;
                pr_res[i] = pair.second;
                ++i;
            }
            return new SparseVector(ir_res, pr_res, nnz, dim);
        }
        return null;
    }
    
    @Override
    public Vector plus(final Vector V) {
        if (V.getDim() != this.pr.length) {
            System.err.println("Dimension doesn't match.");
            System.exit(1);
        }
        if (V instanceof DenseVector) {
            return new DenseVector(ArrayOperator.plus(this.pr, ((DenseVector)V).getPr()));
        }
        if (V instanceof SparseVector) {
            final double[] res = this.pr.clone();
            final int[] ir = ((SparseVector)V).getIr();
            final double[] pr = ((SparseVector)V).getPr();
            int idx = -1;
            for (int k = 0; k < ((SparseVector)V).getNNZ(); ++k) {
                idx = ir[k];
                res[ir[k]] = this.pr[idx] + pr[k];
            }
            return new DenseVector(res);
        }
        return null;
    }
    
    @Override
    public Vector minus(final Vector V) {
        if (V.getDim() != this.pr.length) {
            System.err.println("Dimension doesn't match.");
            System.exit(1);
        }
        if (V instanceof DenseVector) {
            return new DenseVector(ArrayOperator.minus(this.pr, ((DenseVector)V).getPr()));
        }
        if (V instanceof SparseVector) {
            final double[] res = this.pr.clone();
            final int[] ir = ((SparseVector)V).getIr();
            final double[] pr = ((SparseVector)V).getPr();
            int idx = -1;
            for (int k = 0; k < ((SparseVector)V).getNNZ(); ++k) {
                idx = ir[k];
                res[ir[k]] = this.pr[idx] - pr[k];
            }
            return new DenseVector(res);
        }
        return null;
    }
    
    @Override
    public double get(final int i) {
        return this.pr[i];
    }
    
    @Override
    public void set(final int i, final double v) {
        this.pr[i] = v;
    }
    
    @Override
    public Vector operate(final Matrix A) {
        final int dim = this.getDim();
        final int M = A.getRowDimension();
        final int N = A.getColumnDimension();
        if (M != dim) {
            System.err.println("Dimension doesn't match.");
            System.exit(1);
        }
        final double[] res = ArrayOperator.allocate1DArray(N, 0.0);
        if (A instanceof DenseMatrix) {
            final double[][] AData = ((DenseMatrix)A).getData();
            double[] ARow = null;
            double v = 0.0;
            for (int i = 0; i < M; ++i) {
                ARow = AData[i];
                v = this.pr[i];
                for (int j = 0; j < N; ++j) {
                    final double[] array = res;
                    final int n = j;
                    array[n] += v * ARow[j];
                }
            }
        }
        else if (A instanceof SparseMatrix) {
            final int[] ir = ((SparseMatrix)A).getIr();
            final int[] jc = ((SparseMatrix)A).getJc();
            final double[] pr = ((SparseMatrix)A).getPr();
            for (int k = 0; k < N; ++k) {
                for (int l = jc[k]; l < jc[k + 1]; ++l) {
                    final double[] array2 = res;
                    final int n2 = k;
                    array2[n2] += this.pr[ir[l]] * pr[l];
                }
            }
        }
        return new DenseVector(res);
    }
    
    @Override
    public void clear() {
        ArrayOperator.clearVector(this.pr);
    }
    
    @Override
    public Vector times(final double v) {
        if (v == 0.0) {
            return new DenseVector(this.getDim(), 0.0);
        }
        final double[] resData = this.pr.clone();
        for (int i = 0; i < this.pr.length; ++i) {
            final double[] array = resData;
            final int n = i;
            array[n] *= v;
        }
        return new DenseVector(resData);
    }
}
