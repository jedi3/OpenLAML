package ml.optimization;

import la.matrix.*;
import ml.utils.*;

public class ProxL1 implements ProximalMapping
{
    private double lambda;
    
    public ProxL1(final double lambda) {
        this.lambda = lambda;
    }
    
    @Override
    public Matrix compute(double t, final Matrix X) {
        t *= this.lambda;
        final Matrix res = X.copy();
        final int M = res.getRowDimension();
        final int N = res.getColumnDimension();
        double v = 0.0;
        if (res instanceof DenseMatrix) {
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            for (int i = 0; i < M; ++i) {
                resRow = resData[i];
                for (int j = 0; j < N; ++j) {
                    v = resRow[j];
                    if (v > t) {
                        resRow[j] = v - t;
                    }
                    else if (v < -t) {
                        resRow[j] = v + t;
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
                v = pr[k];
                if (v > t) {
                    pr[k] = v - t;
                }
                else if (v < -t) {
                    pr[k] = v + t;
                }
                else {
                    pr[k] = 0.0;
                }
            }
            ((SparseMatrix)res).clean();
        }
        return res;
    }
    
    @Override
    public void compute(final Matrix res, double t, final Matrix X) {
        t *= this.lambda;
        InPlaceOperator.assign(res, X);
        final int M = res.getRowDimension();
        final int N = res.getColumnDimension();
        double v = 0.0;
        if (res instanceof DenseMatrix) {
            final double[][] resData = ((DenseMatrix)res).getData();
            double[] resRow = null;
            for (int i = 0; i < M; ++i) {
                resRow = resData[i];
                for (int j = 0; j < N; ++j) {
                    v = resRow[j];
                    if (v > t) {
                        resRow[j] = v - t;
                    }
                    else if (v < -t) {
                        resRow[j] = v + t;
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
                v = pr[k];
                if (v > t) {
                    pr[k] = v - t;
                }
                else if (v < -t) {
                    pr[k] = v + t;
                }
                else {
                    pr[k] = 0.0;
                }
            }
            ((SparseMatrix)res).clean();
        }
    }
}
