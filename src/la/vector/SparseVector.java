package la.vector;

import java.io.*;
import ml.utils.*;
import la.matrix.*;
import java.util.*;

public class SparseVector implements Vector, Serializable
{
    private static final long serialVersionUID = 4760099385084043335L;
    private int[] ir;
    private double[] pr;
    private int nnz;
    private int dim;
    
    public static void main(final String[] args) {
    }
    
    public SparseVector(final int dim) {
        this.ir = new int[0];
        this.pr = new double[0];
        this.nnz = 0;
        this.dim = dim;
    }
    
    public SparseVector(final int[] ir, final double[] pr, final int nnz, final int dim) {
        this.ir = ir;
        this.pr = pr;
        this.nnz = nnz;
        this.dim = dim;
    }
    
    public void assignSparseVector(final SparseVector V) {
        this.ir = V.ir.clone();
        this.pr = V.pr.clone();
        this.nnz = V.nnz;
        this.dim = V.dim;
    }
    
    public int[] getIr() {
        return this.ir;
    }
    
    public double[] getPr() {
        return this.pr;
    }
    
    public int getNNZ() {
        return this.nnz;
    }
    
    @Override
    public String toString() {
        final StringBuffer sb = new StringBuffer(100);
        sb.append('[');
        for (int k = 0; k < this.nnz; ++k) {
            sb.append(String.format("%d: %.4f", this.ir[k], this.pr[k]));
            if (k < this.nnz - 1) {
                sb.append(", ");
            }
        }
        sb.append(']');
        return sb.toString();
    }
    
    @Override
    public int getDim() {
        return this.dim;
    }
    
    public void setDim(final int dim) {
        if (dim > this.dim) {
            this.dim = dim;
        }
    }
    
    @Override
    public Vector copy() {
        return new SparseVector(this.ir.clone(), this.pr.clone(), this.nnz, this.dim);
    }
    
    public Vector clone() {
        return this.copy();
    }
    
    @Override
    public Vector times(final Vector V) {
        if (V.getDim() != this.dim) {
            System.err.println("Dimension doesn't match.");
            System.exit(1);
        }
        if (V instanceof DenseVector) {
            return V.times(this);
        }
        if (V instanceof SparseVector) {
            final ArrayList<Pair<Integer, Double>> list = new ArrayList<Pair<Integer, Double>>();
            final int[] ir = ((SparseVector)V).getIr();
            final double[] pr = ((SparseVector)V).getPr();
            int nnz2 = ((SparseVector)V).getNNZ();
            if (this.nnz != 0 && nnz2 != 0) {
                int k1 = 0;
                int k2 = 0;
                int r1 = 0;
                int r2 = 0;
                double v = 0.0;
                int i = -1;
                while (k1 < this.nnz && k2 < nnz2) {
                    r1 = this.ir[k1];
                    r2 = ir[k2];
                    if (r1 < r2) {
                        ++k1;
                    }
                    else if (r1 == r2) {
                        i = r1;
                        v = this.pr[k1] * pr[k2];
                        ++k1;
                        ++k2;
                        if (v == 0.0) {
                            continue;
                        }
                        list.add(Pair.of(i, v));
                    }
                    else {
                        ++k2;
                    }
                }
            }
            nnz2 = list.size();
            final int dim = this.getDim();
            final int[] ir_res = new int[nnz2];
            final double[] pr_res = new double[nnz2];
            int j = 0;
            for (final Pair<Integer, Double> pair : list) {
                ir_res[j] = pair.first;
                pr_res[j] = pair.second;
                ++j;
            }
            return new SparseVector(ir_res, pr_res, nnz2, dim);
        }
        return null;
    }
    
    @Override
    public Vector plus(final Vector V) {
        if (V.getDim() != this.dim) {
            System.err.println("Dimension doesn't match.");
            System.exit(1);
        }
        if (V instanceof DenseVector) {
            return V.plus(this);
        }
        if (V instanceof SparseVector) {
            final ArrayList<Pair<Integer, Double>> list = new ArrayList<Pair<Integer, Double>>();
            final int[] ir = ((SparseVector)V).getIr();
            final double[] pr = ((SparseVector)V).getPr();
            int nnz2 = ((SparseVector)V).getNNZ();
            if (this.nnz != 0 || nnz2 != 0) {
                int k1 = 0;
                int k2 = 0;
                int r1 = 0;
                int r2 = 0;
                double v = 0.0;
                int i = -1;
                while (k1 < this.nnz || k2 < nnz2) {
                    if (k2 == nnz2) {
                        i = this.ir[k1];
                        v = this.pr[k1];
                        ++k1;
                    }
                    else if (k1 == this.nnz) {
                        i = ir[k2];
                        v = pr[k2];
                        ++k2;
                    }
                    else {
                        r1 = this.ir[k1];
                        r2 = ir[k2];
                        if (r1 < r2) {
                            i = r1;
                            v = this.pr[k1];
                            ++k1;
                        }
                        else if (r1 == r2) {
                            i = r1;
                            v = this.pr[k1] + pr[k2];
                            ++k1;
                            ++k2;
                        }
                        else {
                            i = r2;
                            v = pr[k2];
                            ++k2;
                        }
                    }
                    if (v != 0.0) {
                        list.add(Pair.of(i, v));
                    }
                }
            }
            nnz2 = list.size();
            final int dim = this.getDim();
            final int[] ir_res = new int[nnz2];
            final double[] pr_res = new double[nnz2];
            int j = 0;
            for (final Pair<Integer, Double> pair : list) {
                ir_res[j] = pair.first;
                pr_res[j] = pair.second;
                ++j;
            }
            return new SparseVector(ir_res, pr_res, nnz2, dim);
        }
        return null;
    }
    
    @Override
    public Vector minus(final Vector V) {
        if (V.getDim() != this.dim) {
            System.err.println("Dimension doesn't match.");
            System.exit(1);
        }
        if (V instanceof DenseVector) {
            final double[] VPr = ((DenseVector)V).getPr();
            final double[] resPr = new double[this.dim];
            for (int k = 0; k < this.dim; ++k) {
                resPr[k] = -VPr[k];
            }
            for (int k = 0; k < this.nnz; ++k) {
                final double[] array = resPr;
                final int n = this.ir[k];
                array[n] += this.pr[k];
            }
            return new DenseVector(resPr);
        }
        if (V instanceof SparseVector) {
            final ArrayList<Pair<Integer, Double>> list = new ArrayList<Pair<Integer, Double>>();
            final int[] ir = ((SparseVector)V).getIr();
            final double[] pr = ((SparseVector)V).getPr();
            int nnz2 = ((SparseVector)V).getNNZ();
            if (this.nnz != 0 || nnz2 != 0) {
                int k2 = 0;
                int k3 = 0;
                int r1 = 0;
                int r2 = 0;
                double v = 0.0;
                int i = -1;
                while (k2 < this.nnz || k3 < nnz2) {
                    if (k3 == nnz2) {
                        i = this.ir[k2];
                        v = this.pr[k2];
                        ++k2;
                    }
                    else if (k2 == this.nnz) {
                        i = ir[k3];
                        v = -pr[k3];
                        ++k3;
                    }
                    else {
                        r1 = this.ir[k2];
                        r2 = ir[k3];
                        if (r1 < r2) {
                            i = r1;
                            v = this.pr[k2];
                            ++k2;
                        }
                        else if (r1 == r2) {
                            i = r1;
                            v = this.pr[k2] - pr[k3];
                            ++k2;
                            ++k3;
                        }
                        else {
                            i = r2;
                            v = -pr[k3];
                            ++k3;
                        }
                    }
                    if (v != 0.0) {
                        list.add(Pair.of(i, v));
                    }
                }
            }
            nnz2 = list.size();
            final int dim = this.getDim();
            final int[] ir_res = new int[nnz2];
            final double[] pr_res = new double[nnz2];
            int j = 0;
            for (final Pair<Integer, Double> pair : list) {
                ir_res[j] = pair.first;
                pr_res[j] = pair.second;
                ++j;
            }
            return new SparseVector(ir_res, pr_res, nnz2, dim);
        }
        return null;
    }
    
    @Override
    public double get(final int i) {
        if (i < 0 || i >= this.dim) {
            System.err.println("Wrong index.");
            System.exit(1);
        }
        if (this.nnz == 0) {
            return 0.0;
        }
        int u = this.nnz - 1;
        int l = 0;
        int idx = -1;
        int k = 0;
        while (l <= u) {
            k = (u + l) / 2;
            idx = this.ir[k];
            if (idx == i) {
                return this.pr[k];
            }
            if (idx < i) {
                l = k + 1;
            }
            else {
                u = k - 1;
            }
        }
        return 0.0;
    }
    
    @Override
    public void set(final int i, final double v) {
        if (i < 0 || i >= this.dim) {
            System.err.println("Wrong index.");
            System.exit(1);
        }
        if (this.nnz == 0) {
            this.insertEntry(i, v, 0);
            return;
        }
        int u = this.nnz - 1;
        int l = 0;
        int idx = -1;
        int k = 0;
        int flag = 0;
        while (l <= u) {
            k = (u + l) / 2;
            idx = this.ir[k];
            if (idx == i) {
                if (v == 0.0) {
                    this.deleteEntry(k);
                }
                else {
                    this.pr[k] = v;
                }
                return;
            }
            if (idx < i) {
                l = k + 1;
                flag = 1;
            }
            else {
                u = k - 1;
                flag = 2;
            }
        }
        if (flag == 1) {
            ++k;
        }
        this.insertEntry(i, v, k);
    }
    
    private void insertEntry(final int r, final double v, final int pos) {
        if (v == 0.0) {
            return;
        }
        final int len_old = this.pr.length;
        final int new_space = (len_old < this.dim - 10) ? 10 : (this.dim - len_old);
        if (this.nnz + 1 > len_old) {
            final double[] pr_new = new double[len_old + new_space];
            System.arraycopy(this.pr, 0, pr_new, 0, pos);
            pr_new[pos] = v;
            if (pos < len_old) {
                System.arraycopy(this.pr, pos, pr_new, pos + 1, len_old - pos);
            }
            this.pr = pr_new;
        }
        else {
            for (int i = this.nnz - 1; i >= pos; --i) {
                this.pr[i + 1] = this.pr[i];
            }
            this.pr[pos] = v;
        }
        if (this.nnz + 1 > len_old) {
            final int[] ir_new = new int[len_old + new_space];
            System.arraycopy(this.ir, 0, ir_new, 0, pos);
            ir_new[pos] = r;
            if (pos < len_old) {
                System.arraycopy(this.ir, pos, ir_new, pos + 1, len_old - pos);
            }
            this.ir = ir_new;
        }
        else {
            for (int i = this.nnz - 1; i >= pos; --i) {
                this.ir[i + 1] = this.ir[i];
            }
            this.ir[pos] = r;
        }
        ++this.nnz;
    }
    
    public void clean() {
        for (int k = this.nnz - 1; k >= 0; --k) {
            if (this.pr[k] == 0.0) {
                this.deleteEntry(k);
            }
        }
    }
    
    private void deleteEntry(final int pos) {
        for (int i = pos; i < this.nnz - 1; ++i) {
            this.pr[i] = this.pr[i + 1];
            this.ir[i] = this.ir[i + 1];
        }
        --this.nnz;
    }
    
    @Override
    public Vector operate(final Matrix A) {
        final int M = A.getRowDimension();
        final int N = A.getColumnDimension();
        if (M != this.dim) {
            System.err.println("Dimension doesn't match.");
            System.exit(1);
        }
        if (A instanceof DenseMatrix) {
            final double[] res = ArrayOperator.allocate1DArray(N, 0.0);
            final double[][] AData = ((DenseMatrix)A).getData();
            double[] ARow = null;
            double v = 0.0;
            for (int k = 0; k < this.nnz; ++k) {
                final int i = this.ir[k];
                ARow = AData[i];
                v = this.pr[k];
                for (int j = 0; j < N; ++j) {
                    final double[] array = res;
                    final int n = j;
                    array[n] += v * ARow[j];
                }
            }
            return new DenseVector(res);
        }
        if (A instanceof SparseMatrix) {
            int[] ir = ((SparseMatrix)A).getIr();
            final int[] jc = ((SparseMatrix)A).getJc();
            double[] pr = ((SparseMatrix)A).getPr();
            double s = 0.0;
            int k2 = 0;
            int k3 = 0;
            int c = 0;
            int r = 0;
            final TreeMap<Integer, Double> map = new TreeMap<Integer, Double>();
            for (int l = 0; l < N; ++l) {
                k2 = 0;
                k3 = jc[l];
                s = 0.0;
                while (k3 < jc[l + 1] && k2 < this.nnz) {
                    c = this.ir[k2];
                    r = ir[k3];
                    if (r < c) {
                        ++k3;
                    }
                    else if (r > c) {
                        ++k2;
                    }
                    else {
                        s += this.pr[k2] * pr[k3];
                        ++k2;
                        ++k3;
                    }
                }
                if (s != 0.0) {
                    map.put(l, s);
                }
            }
            final int nnz = map.size();
            ir = new int[nnz];
            pr = new double[nnz];
            int ind = 0;
            for (final Map.Entry<Integer, Double> entry : map.entrySet()) {
                ir[ind] = entry.getKey();
                pr[ind] = entry.getValue();
                ++ind;
            }
            return new SparseVector(ir, pr, nnz, N);
        }
        return null;
    }
    
    @Override
    public void clear() {
        this.ir = new int[0];
        this.pr = new double[0];
        this.nnz = 0;
    }
    
    @Override
    public Vector times(final double v) {
        if (v == 0.0) {
            return new SparseVector(this.dim);
        }
        final SparseVector res = (SparseVector)this.copy();
        final double[] pr = res.pr;
        for (int k = 0; k < this.nnz; ++k) {
            final double[] array = pr;
            final int n = k;
            array[n] *= v;
        }
        return res;
    }
}
