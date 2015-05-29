package la.matrix;

import java.io.*;

import la.vector.*;
import la.vector.Vector;

import java.util.*;

import ml.utils.*;

public class SparseMatrix implements Matrix, Serializable
{
    private static final long serialVersionUID = 404718895052720649L;
    private int[] ir;
    private int[] jc;
    private int[] ic;
    private int[] jr;
    private double[] pr;
    private int[] valCSRIndices;
    private int nnz;
    private int nzmax;
    private int M;
    private int N;
    
    public static void main(final String[] args) {
        final Matrix S = new SparseMatrix(3, 3);
        Printer.printMatrix(S);
        final Matrix A = new DenseMatrix(3, 3);
        Printer.printMatrix(S.mtimes(A));
        Printer.printMatrix(A.mtimes(S));
    }
    
    private SparseMatrix() {
        this.M = 0;
        this.N = 0;
        this.nzmax = 0;
        this.nnz = 0;
    }
    
    public SparseMatrix(final int M, final int N) {
        this.M = M;
        this.N = N;
        this.nzmax = 0;
        this.nnz = 0;
        this.jc = new int[N + 1];
        for (int j = 0; j < N + 1; ++j) {
            this.jc[j] = 0;
        }
        this.jr = new int[M + 1];
        for (int i = 0; i < M + 1; ++i) {
            this.jr[i] = 0;
        }
        this.ir = new int[0];
        this.pr = new double[0];
        this.ic = new int[0];
        this.valCSRIndices = new int[0];
    }
    
    public SparseMatrix(final SparseMatrix A) {
        this.ir = A.ir;
        this.jc = A.jc;
        this.pr = A.pr;
        this.ic = A.ic;
        this.jr = A.jr;
        this.valCSRIndices = A.valCSRIndices;
        this.M = A.M;
        this.N = A.N;
        this.nzmax = A.nzmax;
        this.nnz = A.nnz;
    }
    
    public SparseMatrix(final int[] rIndices, final int[] cIndices, final double[] values, final int numRows, final int numColumns, final int nzmax) {
        final SparseMatrix temp = createSparseMatrix(rIndices, cIndices, values, numRows, numColumns, nzmax);
        this.assignSparseMatrix(temp);
    }
    
    public void assignSparseMatrix(final SparseMatrix A) {
        this.ir = A.ir.clone();
        this.jc = A.jc.clone();
        this.pr = A.pr.clone();
        this.ic = A.ic.clone();
        this.jr = A.jr.clone();
        this.valCSRIndices = A.valCSRIndices.clone();
        this.M = A.M;
        this.N = A.N;
        this.nzmax = A.nzmax;
        this.nnz = A.nnz;
    }
    
    public static SparseMatrix createSparseMatrix(final TreeMap<Pair<Integer, Integer>, Double> inputMap, final int numRows, final int numColumns) {
        final TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();
        int nzmax = 0;
        for (final Map.Entry<Pair<Integer, Integer>, Double> entry : inputMap.entrySet()) {
            if (entry.getValue() != 0.0) {
                map.put(Pair.of(entry.getKey().second, entry.getKey().first), entry.getValue());
                ++nzmax;
            }
        }
        final int[] ir = new int[nzmax];
        final int[] jc = new int[numColumns + 1];
        final double[] pr = new double[nzmax];
        int rIdx = -1;
        int cIdx = -1;
        int k = 0;
        jc[0] = 0;
        int currentColumn = 0;
        for (final Map.Entry<Pair<Integer, Integer>, Double> entry2 : map.entrySet()) {
            rIdx = entry2.getKey().second;
            cIdx = entry2.getKey().first;
            pr[k] = entry2.getValue();
            ir[k] = rIdx;
            while (currentColumn < cIdx) {
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
        return createSparseMatrixByCSCArrays(ir, jc, pr, numRows, numColumns, nzmax);
    }
    
    public static SparseMatrix createSparseMatrix(final int[] rIndices, final int[] cIndices, final double[] values, final int numRows, final int numColumns, final int nzmax) {
        int k = -1;
        final TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();
        for (k = 0; k < values.length; ++k) {
            if (values[k] != 0.0) {
                map.put(Pair.of(cIndices[k], rIndices[k]), values[k]);
            }
        }
        final int[] ir = new int[nzmax];
        final int[] jc = new int[numColumns + 1];
        final double[] pr = new double[nzmax];
        int rIdx = -1;
        int cIdx = -1;
        k = 0;
        jc[0] = 0;
        int currentColumn = 0;
        for (final Map.Entry<Pair<Integer, Integer>, Double> entry : map.entrySet()) {
            rIdx = entry.getKey().second;
            cIdx = entry.getKey().first;
            pr[k] = entry.getValue();
            ir[k] = rIdx;
            while (currentColumn < cIdx) {
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
        return createSparseMatrixByCSCArrays(ir, jc, pr, numRows, numColumns, nzmax);
    }
    
    public SparseMatrix(final int[] rIndices, final int[] cIndices, final double[] values, final int numRows, final int numColumns) {
        this(rIndices, cIndices, values, numRows, numColumns, values.length);
    }
    
    public static SparseMatrix createSparseMatrixByCSCArrays(final int[] ir, final int[] jc, final double[] pr, final int M, final int N, final int nzmax) {
        final SparseMatrix res = new SparseMatrix();
        res.ir = ir;
        res.jc = jc;
        res.pr = pr;
        res.M = M;
        res.N = N;
        res.nzmax = nzmax;
        final int[] ic = new int[pr.length];
        final int[] jr = new int[M + 1];
        final int[] valCSRIndices = new int[pr.length];
        final int[] cIndices = new int[ic.length];
        int k = 0;
        int j = 0;
        while (k < ir.length && j < N) {
            if (jc[j] <= k && k < jc[j + 1]) {
                cIndices[k] = j;
                ++k;
            }
            else {
                ++j;
            }
        }
        final TreeMap<Pair<Integer, Integer>, Integer> map = new TreeMap<Pair<Integer, Integer>, Integer>();
        for (k = 0; k < pr.length; ++k) {
            if (pr[k] != 0.0) {
                map.put(Pair.of(ir[k], cIndices[k]), k);
            }
        }
        int rIdx = -1;
        int cIdx = -1;
        int vIdx = -1;
        k = 0;
        jr[0] = 0;
        int currentRow = 0;
        for (final Map.Entry<Pair<Integer, Integer>, Integer> entry : map.entrySet()) {
            rIdx = entry.getKey().first;
            cIdx = entry.getKey().second;
            vIdx = entry.getValue();
            ic[k] = cIdx;
            valCSRIndices[k] = vIdx;
            while (currentRow < rIdx) {
                jr[currentRow + 1] = k;
                ++currentRow;
            }
            ++k;
        }
        while (currentRow < M) {
            jr[currentRow + 1] = k;
            ++currentRow;
        }
        jr[M] = k;
        res.ic = ic;
        res.jr = jr;
        res.valCSRIndices = valCSRIndices;
        res.nnz = map.size();
        return res;
    }
    
    public static SparseMatrix createSparseMatrixByCSRArrays(final int[] ic, final int[] jr, double[] pr, final int M, final int N, final int nzmax) {
        final int[] rIndices = new int[ic.length];
        int k = 0;
        int i = 0;
        while (k < ic.length && i < M) {
            if (jr[i] <= k && k < jr[i + 1]) {
                rIndices[k] = i;
                ++k;
            }
            else {
                ++i;
            }
        }
        final TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();
        for (k = 0; k < pr.length; ++k) {
            if (pr[k] != 0.0) {
                map.put(Pair.of(ic[k], rIndices[k]), pr[k]);
            }
        }
        final int[] ir = new int[nzmax];
        final int[] jc = new int[N + 1];
        pr = new double[nzmax];
        int rIdx = -1;
        int cIdx = -1;
        k = 0;
        jc[0] = 0;
        int currentColumn = 0;
        for (final Map.Entry<Pair<Integer, Integer>, Double> entry : map.entrySet()) {
            rIdx = entry.getKey().second;
            cIdx = entry.getKey().first;
            pr[k] = entry.getValue();
            ir[k] = rIdx;
            while (currentColumn < cIdx) {
                jc[currentColumn + 1] = k;
                ++currentColumn;
            }
            ++k;
        }
        while (currentColumn < N) {
            jc[currentColumn + 1] = k;
            ++currentColumn;
        }
        jc[N] = k;
        return createSparseMatrixByCSCArrays(ir, jc, pr, M, N, nzmax);
    }
    
    public int[] getIr() {
        return this.ir;
    }
    
    public int[] getJc() {
        return this.jc;
    }
    
    public int[] getIc() {
        return this.ic;
    }
    
    public int[] getJr() {
        return this.jr;
    }
    
    public double[] getPr() {
        return this.pr;
    }
    
    public int[] getValCSRIndices() {
        return this.valCSRIndices;
    }
    
    @Override
    public int getRowDimension() {
        return this.M;
    }
    
    @Override
    public int getColumnDimension() {
        return this.N;
    }
    
    public int getNZMax() {
        return this.nzmax;
    }
    
    public int getNNZ() {
        return this.nnz;
    }
    
    @Override
    public double[][] getData() {
        final double[][] data = new double[this.M][];
        for (int i = 0; i < this.M; ++i) {
            final double[] rowData = ArrayOperator.allocateVector(this.N, 0.0);
            for (int k = this.jr[i]; k < this.jr[i + 1]; ++k) {
                rowData[this.ic[k]] = this.pr[this.valCSRIndices[k]];
            }
            data[i] = rowData;
        }
        return data;
    }
    
    @Override
    public Matrix mtimes(final Matrix A) {
        Matrix res = null;
        final int NA = A.getColumnDimension();
        if (A instanceof DenseMatrix) {
            final double[][] resData = new double[this.M][];
            for (int i = 0; i < this.M; ++i) {
                resData[i] = new double[NA];
            }
            double[] resRow = null;
            final double[][] data = ((DenseMatrix)A).getData();
            int c = -1;
            double s = 0.0;
            for (int j = 0; j < this.M; ++j) {
                resRow = resData[j];
                for (int k = 0; k < NA; ++k) {
                    s = 0.0;
                    for (int l = this.jr[j]; l < this.jr[j + 1]; ++l) {
                        c = this.ic[l];
                        s += this.pr[this.valCSRIndices[l]] * data[c][k];
                    }
                    resRow[k] = s;
                }
            }
            res = new DenseMatrix(resData);
        }
        else if (A instanceof SparseMatrix) {
            int[] ir = null;
            int[] jc = null;
            double[] pr = null;
            ir = ((SparseMatrix)A).getIr();
            jc = ((SparseMatrix)A).getJc();
            pr = ((SparseMatrix)A).getPr();
            int rr = -1;
            int cl = -1;
            double s2 = 0.0;
            int kl = 0;
            int kr = 0;
            int nzmax = 0;
            final TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();
            for (int m = 0; m < this.M; ++m) {
                for (int j2 = 0; j2 < NA; ++j2) {
                    s2 = 0.0;
                    kl = this.jr[m];
                    kr = jc[j2];
                    while (kl < this.jr[m + 1] && kr < jc[j2 + 1]) {
                        cl = this.ic[kl];
                        rr = ir[kr];
                        if (cl < rr) {
                            ++kl;
                        }
                        else if (cl > rr) {
                            ++kr;
                        }
                        else {
                            s2 += this.pr[this.valCSRIndices[kl]] * pr[kr];
                            ++kl;
                            ++kr;
                        }
                    }
                    if (s2 != 0.0) {
                        ++nzmax;
                        map.put(Pair.of(j2, m), s2);
                    }
                }
            }
            final int numRows = this.M;
            final int numColumns = NA;
            ir = new int[nzmax];
            jc = new int[numColumns + 1];
            pr = new double[nzmax];
            int rIdx = -1;
            int cIdx = -1;
            int k2 = 0;
            jc[0] = 0;
            int currentColumn = 0;
            for (final Map.Entry<Pair<Integer, Integer>, Double> entry : map.entrySet()) {
                rIdx = entry.getKey().second;
                cIdx = entry.getKey().first;
                pr[k2] = entry.getValue();
                ir[k2] = rIdx;
                while (currentColumn < cIdx) {
                    jc[currentColumn + 1] = k2;
                    ++currentColumn;
                }
                ++k2;
            }
            while (currentColumn < numColumns) {
                jc[currentColumn + 1] = k2;
                ++currentColumn;
            }
            jc[numColumns] = k2;
            res = createSparseMatrixByCSCArrays(ir, jc, pr, numRows, numColumns, nzmax);
        }
        return res;
    }
    
    @Override
    public double getEntry(final int r, final int c) {
        if (r < 0 || r > this.M - 1 || c < 0 || c > this.N - 1) {
            System.err.println("Wrong index.");
            System.exit(1);
        }
        final double res = 0.0;
        int idx = -1;
        if (r <= c) {
            int u = this.jc[c + 1] - 1;
            int l = this.jc[c];
            if (u < l) {
                return 0.0;
            }
            int k = this.jc[c];
            while (l <= u) {
                k = (u + l) / 2;
                idx = this.ir[k];
                if (idx == r) {
                    return this.pr[k];
                }
                if (idx < r) {
                    l = k + 1;
                }
                else {
                    u = k - 1;
                }
            }
        }
        else {
            int u = this.jr[r + 1] - 1;
            int l = this.jr[r];
            if (u < l) {
                return 0.0;
            }
            int k = this.jr[r];
            while (l <= u) {
                k = (u + l) / 2;
                idx = this.ic[k];
                if (idx == c) {
                    return this.pr[this.valCSRIndices[k]];
                }
                if (idx < c) {
                    l = k + 1;
                }
                else {
                    u = k - 1;
                }
            }
        }
        return res;
    }
    
    @Override
    public void setEntry(final int r, final int c, final double v) {
        if (r < 0 || r > this.M - 1 || c < 0 || c > this.N - 1) {
            System.err.println("Wrong index.");
            System.exit(1);
        }
        int u = this.jc[c + 1] - 1;
        int l = this.jc[c];
        if (u < l) {
            this.insertEntry(r, c, v, this.jc[c]);
            return;
        }
        int idx = -1;
        int k = this.jc[c];
        int flag = 0;
        while (l <= u) {
            k = (u + l) / 2;
            idx = this.ir[k];
            if (idx == r) {
                if (v == 0.0) {
                    this.deleteEntry(r, c, k);
                }
                else {
                    this.pr[k] = v;
                }
                return;
            }
            if (idx < r) {
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
        this.insertEntry(r, c, v, k);
    }
    
    private void insertEntry(final int r, final int c, final double v, final int pos) {
        if (v == 0.0) {
            return;
        }
        final int len_old = this.pr.length;
        final int new_space = (len_old < this.M * this.N - 10) ? 10 : (this.M * this.N - len_old);
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
        for (int j = c + 1; j < this.N + 1; ++j) {
            final int[] jc = this.jc;
            final int n = j;
            ++jc[n];
        }
        int u = this.jr[r + 1] - 1;
        int l = this.jr[r];
        int k = this.jr[r];
        int idx = -1;
        int flag = 0;
        while (l <= u) {
            k = (u + l) / 2;
            idx = this.ic[k];
            if (idx != c) {
                if (idx < c) {
                    l = k + 1;
                    flag = 1;
                }
                else {
                    u = k - 1;
                    flag = 2;
                }
            }
        }
        if (flag == 1) {
            ++k;
        }
        if (this.nnz + 1 > len_old) {
            final int[] ic_new = new int[len_old + new_space];
            System.arraycopy(this.ic, 0, ic_new, 0, k);
            ic_new[k] = c;
            if (k < len_old) {
                System.arraycopy(this.ic, k, ic_new, k + 1, len_old - k);
            }
            this.ic = ic_new;
        }
        else {
            for (int m = this.nnz - 1; m >= k; --m) {
                this.ic[m + 1] = this.ic[m];
            }
            this.ic[k] = c;
        }
        for (int m = r + 1; m < this.M + 1; ++m) {
            final int[] jr = this.jr;
            final int n2 = m;
            ++jr[n2];
        }
        for (int m = 0; m < this.nnz; ++m) {
            if (this.valCSRIndices[m] >= pos) {
                final int[] valCSRIndices = this.valCSRIndices;
                final int n3 = m;
                ++valCSRIndices[n3];
            }
        }
        if (this.nnz + 1 > len_old) {
            final int[] valCSRIndices_new = new int[len_old + new_space];
            System.arraycopy(this.valCSRIndices, 0, valCSRIndices_new, 0, k);
            valCSRIndices_new[k] = pos;
            if (k < len_old) {
                System.arraycopy(this.valCSRIndices, k, valCSRIndices_new, k + 1, len_old - k);
            }
            this.valCSRIndices = valCSRIndices_new;
        }
        else {
            for (int m = this.nnz - 1; m >= k; --m) {
                this.valCSRIndices[m + 1] = this.valCSRIndices[m];
            }
            this.valCSRIndices[k] = pos;
        }
        ++this.nnz;
        if (this.nnz > len_old) {
            this.nzmax = len_old + new_space;
        }
    }
    
    private void deleteEntry(final int r, final int c, final int pos) {
        for (int i = pos; i < this.nnz - 1; ++i) {
            this.pr[i] = this.pr[i + 1];
            this.ir[i] = this.ir[i + 1];
        }
        for (int j = c + 1; j < this.N + 1; ++j) {
            final int[] jc = this.jc;
            final int n = j;
            --jc[n];
        }
        int u = this.jr[r + 1] - 1;
        int l = this.jr[r];
        int k = this.jr[r];
        int idx = -1;
        while (true) {
            while (l <= u) {
                k = (u + l) / 2;
                idx = this.ic[k];
                if (idx == c) {
                    for (int m = 0; m < this.valCSRIndices.length; ++m) {
                        if (this.valCSRIndices[m] > pos) {
                            final int[] valCSRIndices = this.valCSRIndices;
                            final int n2 = m;
                            --valCSRIndices[n2];
                        }
                    }
                    for (int j2 = k; j2 < this.nnz - 1; ++j2) {
                        this.ic[j2] = this.ic[j2 + 1];
                        this.valCSRIndices[j2] = this.valCSRIndices[j2 + 1];
                    }
                    for (int m = r + 1; m < this.M + 1; ++m) {
                        final int[] jr = this.jr;
                        final int n3 = m;
                        --jr[n3];
                    }
                    --this.nnz;
                    return;
                }
                if (idx < c) {
                    l = k + 1;
                }
                else {
                    u = k - 1;
                }
            }
            continue;
        }
    }
    
    @Deprecated
    private Matrix transpose0() {
        final double[] values = this.pr;
        final int[] rIndices = this.ir;
        final int[] cIndices = new int[this.ic.length];
        int k = 0;
        int j = 0;
        while (k < this.nnz && j < this.N) {
            if (this.jc[j] <= k && k < this.jc[j + 1]) {
                cIndices[k] = j;
                ++k;
            }
            else {
                ++j;
            }
        }
        return createSparseMatrix(cIndices, rIndices, values, this.N, this.M, this.nzmax);
    }
    
    @Override
    public Matrix transpose() {
        final SparseMatrix res = new SparseMatrix();
        res.M = this.N;
        res.N = this.M;
        res.nnz = this.nnz;
        res.nzmax = this.nzmax;
        res.ir = this.ic.clone();
        res.jc = this.jr.clone();
        res.ic = this.ir.clone();
        res.jr = this.jc.clone();
        final double[] pr_new = new double[this.nzmax];
        int k;
        for (k = 0, k = 0; k < this.nnz; ++k) {
            pr_new[k] = this.pr[this.valCSRIndices[k]];
        }
        res.pr = pr_new;
        final int[] valCSRIndices_new = new int[this.nzmax];
        int j = 0;
        int rIdx = -1;
        int cIdx = -1;
        k = 0;
        int k2 = 0;
        int numBeforeThisEntry = 0;
        while (k < this.nnz && j < this.N) {
            if (this.jc[j] <= k && k < this.jc[j + 1]) {
                rIdx = this.ir[k];
                cIdx = j;
                numBeforeThisEntry = this.jr[rIdx];
                for (k2 = this.jr[rIdx]; k2 < this.jr[rIdx + 1] && this.ic[k2] != cIdx; ++k2) {
                    ++numBeforeThisEntry;
                }
                valCSRIndices_new[k] = numBeforeThisEntry;
                ++k;
            }
            else {
                ++j;
            }
        }
        res.valCSRIndices = valCSRIndices_new;
        return res;
    }
    
    @Override
    public Matrix plus(final Matrix A) {
        if (A.getRowDimension() != this.M || A.getColumnDimension() != this.N) {
            System.err.println("Dimension doesn't match.");
            return null;
        }
        Matrix res = null;
        if (A instanceof DenseMatrix) {
            res = A.plus(this);
        }
        else if (A instanceof SparseMatrix) {
            int[] ir = null;
            int[] jc = null;
            double[] pr = null;
            ir = ((SparseMatrix)A).getIr();
            jc = ((SparseMatrix)A).getJc();
            pr = ((SparseMatrix)A).getPr();
            int k1 = 0;
            int k2 = 0;
            int r1 = -1;
            int r2 = -1;
            int nzmax = 0;
            int i = -1;
            double v = 0.0;
            final TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();
            for (int j = 0; j < this.N; ++j) {
                k1 = this.jc[j];
                k2 = jc[j];
                if (k1 != this.jc[j + 1] || k2 != jc[j + 1]) {
                    while (k1 < this.jc[j + 1] || k2 < jc[j + 1]) {
                        if (k2 == jc[j + 1]) {
                            i = this.ir[k1];
                            v = this.pr[k1];
                            ++k1;
                        }
                        else if (k1 == this.jc[j + 1]) {
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
                            map.put(Pair.of(j, i), v);
                            ++nzmax;
                        }
                    }
                }
            }
            final int numRows = this.M;
            final int numColumns = this.N;
            ir = new int[nzmax];
            jc = new int[numColumns + 1];
            pr = new double[nzmax];
            int rIdx = -1;
            int cIdx = -1;
            int l = 0;
            jc[0] = 0;
            int currentColumn = 0;
            for (final Map.Entry<Pair<Integer, Integer>, Double> entry : map.entrySet()) {
                rIdx = entry.getKey().second;
                cIdx = entry.getKey().first;
                pr[l] = entry.getValue();
                ir[l] = rIdx;
                while (currentColumn < cIdx) {
                    jc[currentColumn + 1] = l;
                    ++currentColumn;
                }
                ++l;
            }
            while (currentColumn < numColumns) {
                jc[currentColumn + 1] = l;
                ++currentColumn;
            }
            jc[numColumns] = l;
            res = createSparseMatrixByCSCArrays(ir, jc, pr, numRows, numColumns, nzmax);
        }
        return res;
    }
    
    @Override
    public Matrix minus(final Matrix A) {
        if (A.getRowDimension() != this.M || A.getColumnDimension() != this.N) {
            System.err.println("Dimension doesn't match.");
            return null;
        }
        Matrix res = null;
        if (A instanceof DenseMatrix) {
            res = A.copy();
            final double[][] resData = ((DenseMatrix)res).getData();
            int r = -1;
            int k = 0;
            for (int j = 0; j < this.N; ++j) {
                for (int i = 0; i < this.M; ++i) {
                    resData[i][j] = -resData[i][j];
                }
                for (k = this.jc[j]; k < this.jc[j + 1]; ++k) {
                    r = this.ir[k];
                    final double[] array = resData[r];
                    final int n = j;
                    array[n] += this.pr[k];
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
            int k2 = 0;
            int k3 = 0;
            int r2 = -1;
            int r3 = -1;
            int nzmax = 0;
            int l = -1;
            double v = 0.0;
            final TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();
            for (int m = 0; m < this.N; ++m) {
                k2 = this.jc[m];
                k3 = jc[m];
                if (k2 != this.jc[m + 1] || k3 != jc[m + 1]) {
                    while (k2 < this.jc[m + 1] || k3 < jc[m + 1]) {
                        if (k3 == jc[m + 1]) {
                            l = this.ir[k2];
                            v = this.pr[k2];
                            ++k2;
                        }
                        else if (k2 == this.jc[m + 1]) {
                            l = ir[k3];
                            v = -pr[k3];
                            ++k3;
                        }
                        else {
                            r2 = this.ir[k2];
                            r3 = ir[k3];
                            if (r2 < r3) {
                                l = r2;
                                v = this.pr[k2];
                                ++k2;
                            }
                            else if (r2 == r3) {
                                l = r2;
                                v = this.pr[k2] - pr[k3];
                                ++k2;
                                ++k3;
                            }
                            else {
                                l = r3;
                                v = -pr[k3];
                                ++k3;
                            }
                        }
                        if (v != 0.0) {
                            map.put(Pair.of(m, l), v);
                            ++nzmax;
                        }
                    }
                }
            }
            final int numRows = this.M;
            final int numColumns = this.N;
            ir = new int[nzmax];
            jc = new int[numColumns + 1];
            pr = new double[nzmax];
            int rIdx = -1;
            int cIdx = -1;
            int k4 = 0;
            jc[0] = 0;
            int currentColumn = 0;
            for (final Map.Entry<Pair<Integer, Integer>, Double> entry : map.entrySet()) {
                rIdx = entry.getKey().second;
                cIdx = entry.getKey().first;
                pr[k4] = entry.getValue();
                ir[k4] = rIdx;
                while (currentColumn < cIdx) {
                    jc[currentColumn + 1] = k4;
                    ++currentColumn;
                }
                ++k4;
            }
            while (currentColumn < numColumns) {
                jc[currentColumn + 1] = k4;
                ++currentColumn;
            }
            jc[numColumns] = k4;
            res = createSparseMatrixByCSCArrays(ir, jc, pr, numRows, numColumns, nzmax);
        }
        return res;
    }
    
    @Override
    public Matrix times(final Matrix A) {
        if (A.getRowDimension() != this.M || A.getColumnDimension() != this.N) {
            System.err.println("Dimension doesn't match.");
            return null;
        }
        Matrix res = null;
        if (A instanceof DenseMatrix) {
            res = A.times(this);
        }
        else if (A instanceof SparseMatrix) {
            int[] ir = null;
            int[] jc = null;
            double[] pr = null;
            ir = ((SparseMatrix)A).getIr();
            jc = ((SparseMatrix)A).getJc();
            pr = ((SparseMatrix)A).getPr();
            int k1 = 0;
            int k2 = 0;
            int r1 = -1;
            int r2 = -1;
            int nzmax = 0;
            int i = -1;
            double v = 0.0;
            final TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();
            for (int j = 0; j < this.N; ++j) {
                k1 = this.jc[j];
                k2 = jc[j];
                if (k1 != this.jc[j + 1]) {
                    if (k2 != jc[j + 1]) {
                        while (k1 < this.jc[j + 1] && k2 < jc[j + 1]) {
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
                                map.put(Pair.of(j, i), v);
                                ++nzmax;
                            }
                            else {
                                ++k2;
                            }
                        }
                    }
                }
            }
            final int numRows = this.M;
            final int numColumns = this.N;
            ir = new int[nzmax];
            jc = new int[numColumns + 1];
            pr = new double[nzmax];
            int rIdx = -1;
            int cIdx = -1;
            int l = 0;
            jc[0] = 0;
            int currentColumn = 0;
            for (final Map.Entry<Pair<Integer, Integer>, Double> entry : map.entrySet()) {
                rIdx = entry.getKey().second;
                cIdx = entry.getKey().first;
                pr[l] = entry.getValue();
                ir[l] = rIdx;
                while (currentColumn < cIdx) {
                    jc[currentColumn + 1] = l;
                    ++currentColumn;
                }
                ++l;
            }
            while (currentColumn < numColumns) {
                jc[currentColumn + 1] = l;
                ++currentColumn;
            }
            jc[numColumns] = l;
            res = createSparseMatrixByCSCArrays(ir, jc, pr, numRows, numColumns, nzmax);
        }
        return res;
    }
    
    @Override
    public Matrix times(final double v) {
        if (v == 0.0) {
            return new SparseMatrix(this.M, this.N);
        }
        final SparseMatrix res = (SparseMatrix)this.copy();
        for (int k = 0; k < this.nnz; ++k) {
            res.pr[k] = v * this.pr[k];
        }
        return res;
    }
    
    @Override
    public Matrix copy() {
        final SparseMatrix res = new SparseMatrix();
        res.ir = this.ir.clone();
        res.jc = this.jc.clone();
        res.pr = this.pr.clone();
        res.ic = this.ic.clone();
        res.jr = this.jr.clone();
        res.valCSRIndices = this.valCSRIndices.clone();
        res.M = this.M;
        res.N = this.N;
        res.nzmax = this.nzmax;
        res.nnz = this.nnz;
        return res;
    }
    
    public Matrix clone() {
        return this.copy();
    }
    
    @Override
    public Matrix plus(final double v) {
        final Matrix res = new DenseMatrix(this.M, this.N, v);
        final double[][] data = ((DenseMatrix)res).getData();
        for (int j = 0; j < this.N; ++j) {
            for (int k = this.jc[j]; k < this.jc[j + 1]; ++k) {
                final double[] array = data[this.ir[k]];
                final int n = j;
                array[n] += this.pr[k];
            }
        }
        return res;
    }
    
    @Override
    public Matrix minus(final double v) {
        return this.plus(-v);
    }
    
    @Override
    public Vector operate(final Vector b) {
        if (this.N != b.getDim()) {
            System.err.println("Dimension does not match.");
            System.exit(1);
        }
        Vector res = null;
        if (b instanceof DenseVector) {
            final double[] V = new double[this.M];
            final double[] pr = ((DenseVector)b).getPr();
            double s = 0.0;
            int c = 0;
            for (int r = 0; r < this.M; ++r) {
                s = 0.0;
                for (int k = this.jr[r]; k < this.jr[r + 1]; ++k) {
                    c = this.ic[k];
                    s += this.pr[this.valCSRIndices[k]] * pr[c];
                }
                V[r] = s;
            }
            res = new DenseVector(V);
        }
        else if (b instanceof SparseVector) {
            int[] ir = ((SparseVector)b).getIr();
            double[] pr = ((SparseVector)b).getPr();
            int nnz = ((SparseVector)b).getNNZ();
            double s2 = 0.0;
            int kl = 0;
            int kr = 0;
            int cl = 0;
            int rr = 0;
            final TreeMap<Integer, Double> map = new TreeMap<Integer, Double>();
            for (int i = 0; i < this.M; ++i) {
                kl = this.jr[i];
                kr = 0;
                s2 = 0.0;
                while (kl < this.jr[i + 1] && kr < nnz) {
                    cl = this.ic[kl];
                    rr = ir[kr];
                    if (cl < rr) {
                        ++kl;
                    }
                    else if (cl > rr) {
                        ++kr;
                    }
                    else {
                        s2 += this.pr[this.valCSRIndices[kl]] * pr[kr];
                        ++kl;
                        ++kr;
                    }
                }
                if (s2 != 0.0) {
                    map.put(i, s2);
                }
            }
            nnz = map.size();
            ir = new int[nnz];
            pr = new double[nnz];
            int ind = 0;
            for (final Map.Entry<Integer, Double> entry : map.entrySet()) {
                ir[ind] = entry.getKey();
                pr[ind] = entry.getValue();
                ++ind;
            }
            res = new SparseVector(ir, pr, nnz, this.M);
        }
        return res;
    }
    
    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder(100);
        if (this.getNNZ() == 0) {
            sb.append("Empty sparse matrix." + System.lineSeparator());
            return sb.toString();
        }
        final int p = 4;
        final int[] ic = this.getIc();
        final int[] jr = this.getJr();
        final double[] pr = this.getPr();
        final int[] valCSRIndices = this.getValCSRIndices();
        final int M = this.getRowDimension();
        String valueString = "";
        for (int r = 0; r < M; ++r) {
            sb.append("  ");
            int currentColumn = 0;
            int lastColumn = -1;
            for (int k = jr[r]; k < jr[r + 1]; ++k) {
                for (currentColumn = ic[k]; lastColumn < currentColumn - 1; ++lastColumn) {
                    sb.append(Printer.sprintf(String.format("%%%ds", 8 + p - 4), " "));
                    sb.append("  ");
                }
                lastColumn = currentColumn;
                final double v = pr[valCSRIndices[k]];
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
        this.nzmax = 0;
        this.nnz = 0;
        this.jc = new int[this.N + 1];
        for (int j = 0; j < this.N + 1; ++j) {
            this.jc[j] = 0;
        }
        this.jr = new int[this.M + 1];
        for (int i = 0; i < this.M + 1; ++i) {
            this.jr[i] = 0;
        }
        this.ir = new int[0];
        this.pr = new double[0];
        this.ic = new int[0];
        this.valCSRIndices = new int[0];
    }
    
    public void clean() {
        final TreeSet<Pair<Integer, Integer>> set = new TreeSet<Pair<Integer, Integer>>();
        for (int k = 0; k < this.nnz; ++k) {
            if (this.pr[k] == 0.0) {
                set.add(Pair.of(this.ir[k], this.ic[k]));
            }
        }
        for (final Pair<Integer, Integer> pair : set) {
            this.setEntry(pair.first, pair.second, 0.0);
        }
    }
    
    public void appendAnEmptyRow() {
        final int[] jr = new int[this.M + 2];
        System.arraycopy(this.jr, 0, jr, 0, this.M + 1);
        jr[this.M + 1] = this.jr[this.M];
        ++this.M;
        this.jr = jr;
    }
    
    public void appendAnEmptyColumn() {
        final int[] jc = new int[this.N + 2];
        System.arraycopy(this.jc, 0, jc, 0, this.N + 1);
        jc[this.N + 1] = this.jc[this.N];
        ++this.N;
        this.jc = jc;
    }
    
    @Override
    public Matrix getSubMatrix(final int startRow, final int endRow, final int startColumn, final int endColumn) {
        int nnz = 0;
        final int numRows = endRow - startRow + 1;
        final int numColumns = endColumn - startColumn + 1;
        int rowIdx = -1;
        for (int j = startColumn; j <= endColumn; ++j) {
            for (int k = this.jc[j]; k < this.jc[j + 1]; ++k) {
                rowIdx = this.ir[k];
                if (rowIdx >= startRow) {
                    if (rowIdx > endRow) {
                        break;
                    }
                    ++nnz;
                }
            }
        }
        final int nzmax = nnz;
        final int[] ir = new int[nzmax];
        final int[] jc = new int[numColumns + 1];
        final double[] pr = new double[nzmax];
        int rIdx = -1;
        int cIdx = -1;
        int i = 0;
        jc[0] = 0;
        int currentColumn = startColumn;
        for (int l = startColumn; l <= endColumn; ++l) {
            for (int t = this.jc[l]; t < this.jc[l + 1]; ++t) {
                rowIdx = this.ir[t];
                if (rowIdx >= startRow) {
                    if (rowIdx > endRow) {
                        break;
                    }
                    rIdx = rowIdx - startRow;
                    cIdx = l;
                    pr[i] = this.pr[t];
                    ir[i] = rIdx;
                    while (currentColumn < cIdx) {
                        jc[currentColumn + 1 - startColumn] = i;
                        ++currentColumn;
                    }
                    ++i;
                }
            }
        }
        while (currentColumn < numColumns) {
            jc[currentColumn + 1 - startColumn] = i;
            ++currentColumn;
        }
        jc[numColumns] = i;
        return createSparseMatrixByCSCArrays(ir, jc, pr, numRows, numColumns, nzmax);
    }
    
    @Override
    public Matrix getSubMatrix(final int[] selectedRows, final int[] selectedColumns) {
        final int nRow = selectedRows.length;
        final int nCol = selectedColumns.length;
        double v = 0.0;
        final SparseMatrix res = new SparseMatrix(nRow, nCol);
        for (int j = 0; j < nCol; ++j) {
            final int c = selectedColumns[j];
            for (int i = 0; i < nRow; ++i) {
                final int r = selectedRows[i];
                v = this.getEntry(r, c);
                if (v != 0.0) {
                    res.setEntry(i, j, v);
                }
            }
        }
        return res;
    }
    
    @Override
    public Matrix getColumnMatrix(final int c) {
        final int nnz = this.jc[c + 1] - this.jc[c];
        if (nnz == 0) {
            return new SparseMatrix(this.M, 1);
        }
        final int[] ir = new int[nnz];
        final int[] jc = { 0, nnz };
        final double[] pr = new double[nnz];
        for (int k = this.jc[c], i = 0; k < this.jc[c + 1]; ++k, ++i) {
            ir[i] = this.ir[k];
            pr[i] = this.pr[k];
        }
        return createSparseMatrixByCSCArrays(ir, jc, pr, this.M, 1, nnz);
    }
    
    @Override
    public Vector getColumnVector(final int c) {
        final int dim = this.M;
        final int nnz = this.jc[c + 1] - this.jc[c];
        if (nnz == 0) {
            return new SparseVector(dim);
        }
        final int[] ir = new int[nnz];
        final double[] pr = new double[nnz];
        for (int k = this.jc[c], i = 0; k < this.jc[c + 1]; ++k, ++i) {
            ir[i] = this.ir[k];
            pr[i] = this.pr[k];
        }
        return new SparseVector(ir, pr, nnz, dim);
    }
    
    @Override
    public Matrix getRowMatrix(final int r) {
        final int nnz = this.jr[r + 1] - this.jr[r];
        if (nnz == 0) {
            return new SparseMatrix(1, this.N);
        }
        final int[] ic = new int[nnz];
        final int[] jr = { 0, nnz };
        final double[] pr = new double[nnz];
        for (int k = this.jr[r], j = 0; k < this.jr[r + 1]; ++k, ++j) {
            ic[j] = this.ic[k];
            pr[j] = this.pr[this.valCSRIndices[k]];
        }
        return createSparseMatrixByCSRArrays(ic, jr, pr, 1, this.N, nnz);
    }
    
    @Override
    public Vector getRowVector(final int r) {
        final int dim = this.N;
        final int nnz = this.jr[r + 1] - this.jr[r];
        if (nnz == 0) {
            return new SparseVector(dim);
        }
        final int[] ir = new int[nnz];
        final double[] pr = new double[nnz];
        for (int k = this.jr[r], j = 0; k < this.jr[r + 1]; ++k, ++j) {
            ir[j] = this.ic[k];
            pr[j] = this.pr[this.valCSRIndices[k]];
        }
        return new SparseVector(ir, pr, nnz, dim);
    }
    
    @Override
    public Matrix getRows(final int startRow, final int endRow) {
        int nnz = 0;
        final int numRows = endRow - startRow + 1;
        final int numColumns = this.N;
        int rowIdx = -1;
        for (int j = 0; j < numColumns; ++j) {
            for (int k = this.jc[j]; k < this.jc[j + 1]; ++k) {
                rowIdx = this.ir[k];
                if (rowIdx >= startRow) {
                    if (rowIdx > endRow) {
                        break;
                    }
                    ++nnz;
                }
            }
        }
        final int nzmax = nnz;
        final int[] ir = new int[nzmax];
        final int[] jc = new int[numColumns + 1];
        final double[] pr = new double[nzmax];
        int rIdx = -1;
        int cIdx = -1;
        int i = 0;
        jc[0] = 0;
        int currentColumn = 0;
        for (int l = 0; l < numColumns; ++l) {
            for (int t = this.jc[l]; t < this.jc[l + 1]; ++t) {
                rowIdx = this.ir[t];
                if (rowIdx >= startRow) {
                    if (rowIdx > endRow) {
                        break;
                    }
                    rIdx = rowIdx - startRow;
                    cIdx = l;
                    pr[i] = this.pr[t];
                    ir[i] = rIdx;
                    while (currentColumn < cIdx) {
                        jc[currentColumn + 1] = i;
                        ++currentColumn;
                    }
                    ++i;
                }
            }
        }
        while (currentColumn < numColumns) {
            jc[currentColumn + 1] = i;
            ++currentColumn;
        }
        jc[numColumns] = i;
        return createSparseMatrixByCSCArrays(ir, jc, pr, numRows, numColumns, nzmax);
    }
    
    @Override
    public Matrix getRows(final int... selectedRows) {
        final int nRow = selectedRows.length;
        final int nCol = this.N;
        double v = 0.0;
        final SparseMatrix res = new SparseMatrix(nRow, nCol);
        for (int j = 0; j < nCol; ++j) {
            final int c = j;
            for (int i = 0; i < nRow; ++i) {
                final int r = selectedRows[i];
                v = this.getEntry(r, c);
                if (v != 0.0) {
                    res.setEntry(i, j, v);
                }
            }
        }
        return res;
    }
    
    @Override
    public Vector[] getRowVectors(final int startRow, final int endRow) {
        final int numRows = endRow - startRow + 1;
        final Vector[] res = new Vector[numRows];
        for (int r = startRow, i = 0; r <= endRow; ++r, ++i) {
            res[i] = this.getRowVector(r);
        }
        return res;
    }
    
    @Override
    public Vector[] getRowVectors(final int... selectedRows) {
        final int numRows = selectedRows.length;
        final Vector[] res = new Vector[numRows];
        for (int i = 0; i < numRows; ++i) {
            res[i] = this.getRowVector(selectedRows[i]);
        }
        return res;
    }
    
    @Override
    public Matrix getColumns(final int startColumn, final int endColumn) {
        int nnz = 0;
        final int numRows = this.M;
        final int numColumns = endColumn - startColumn + 1;
        int rowIdx = -1;
        for (int j = startColumn; j <= endColumn; ++j) {
            nnz += this.jc[j + 1] - this.jc[j];
        }
        final int nzmax = nnz;
        final int[] ir = new int[nzmax];
        final int[] jc = new int[numColumns + 1];
        final double[] pr = new double[nzmax];
        int rIdx = -1;
        int cIdx = -1;
        int k = 0;
        jc[0] = 0;
        int currentColumn = startColumn;
        for (int i = startColumn; i <= endColumn; ++i) {
            for (int t = this.jc[i]; t < this.jc[i + 1]; ++t) {
                rowIdx = (rIdx = this.ir[t]);
                cIdx = i;
                pr[k] = this.pr[t];
                ir[k] = rIdx;
                while (currentColumn < cIdx) {
                    jc[currentColumn + 1 - startColumn] = k;
                    ++currentColumn;
                }
                ++k;
            }
        }
        while (currentColumn <= endColumn) {
            jc[currentColumn + 1 - startColumn] = k;
            ++currentColumn;
        }
        return createSparseMatrixByCSCArrays(ir, jc, pr, numRows, numColumns, nzmax);
    }
    
    @Override
    public Matrix getColumns(final int... selectedColumns) {
        int nnz = 0;
        final int numRows = this.M;
        final int numColumns = selectedColumns.length;
        int rowIdx = -1;
        int j = -1;
        for (int c = 0; c < numColumns; ++c) {
            j = selectedColumns[c];
            nnz += this.jc[j + 1] - this.jc[j];
        }
        final int nzmax = nnz;
        final int[] ir = new int[nzmax];
        final int[] jc = new int[numColumns + 1];
        final double[] pr = new double[nzmax];
        int rIdx = -1;
        int k = 0;
        jc[0] = 0;
        for (int c2 = 0; c2 < numColumns; ++c2) {
            j = selectedColumns[c2];
            jc[c2 + 1] = jc[c2] + this.jc[j + 1] - this.jc[j];
            for (int t = this.jc[j]; t < this.jc[j + 1]; ++t) {
                rowIdx = (rIdx = this.ir[t]);
                pr[k] = this.pr[t];
                ir[k] = rIdx;
                ++k;
            }
        }
        return createSparseMatrixByCSCArrays(ir, jc, pr, numRows, numColumns, nzmax);
    }
    
    @Override
    public Vector[] getColumnVectors(final int startColumn, final int endColumn) {
        final int numColumns = endColumn - startColumn + 1;
        final Vector[] res = new Vector[numColumns];
        for (int c = startColumn, i = 0; c <= endColumn; ++c, ++i) {
            res[i] = this.getColumnVector(c);
        }
        return res;
    }
    
    @Override
    public Vector[] getColumnVectors(final int... selectedColumns) {
        final int numColumns = selectedColumns.length;
        final Vector[] res = new Vector[numColumns];
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
        for (int j = 0; j < this.N; ++j) {
            this.setEntry(r, j, A.getEntry(0, j));
        }
    }
    
    @Override
    public void setRowVector(final int r, final Vector V) {
        for (int j = 0; j < this.N; ++j) {
            this.setEntry(r, j, V.get(j));
        }
    }
    
    @Override
    public void setColumnMatrix(final int c, final Matrix A) {
        if (A.getColumnDimension() != 1) {
            Printer.err("Input matrix should be a column matrix.");
            Utility.exit(1);
        }
        for (int i = 0; i < this.M; ++i) {
            this.setEntry(i, c, A.getEntry(i, 0));
        }
    }
    
    @Override
    public void setColumnVector(final int c, final Vector V) {
        for (int i = 0; i < this.M; ++i) {
            this.setEntry(i, c, V.get(i));
        }
    }
}
