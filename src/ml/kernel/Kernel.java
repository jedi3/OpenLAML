package ml.kernel;

import la.vector.*;
import ml.utils.*;
import la.matrix.*;

public class Kernel
{
    public static void main(final String[] args) {
    }
    
    public static Matrix calcKernel(final String kernelType, final double kernelParam, final Matrix X) {
        return calcKernel(kernelType, kernelParam, X, X);
    }
    
    public static Matrix calcKernel(final String kernelType, final double kernelParam, final Vector[] A, final Vector[] B) {
        Matrix K = null;
        final int nA = A.length;
        final int nB = B.length;
        if (kernelType.equals("linear")) {
            final double[][] resData = ArrayOperator.allocate2DArray(nA, nB, 0.0);
            double[] resRow = null;
            Vector V = null;
            for (int i = 0; i < nA; ++i) {
                resRow = resData[i];
                V = A[i];
                for (int j = 0; j < nB; ++j) {
                    resRow[j] = Matlab.innerProduct(V, B[j]);
                }
            }
            K = new DenseMatrix(resData);
        }
        else if (kernelType.equals("cosine")) {
            final double[] AA = new double[nA];
            Vector V2 = null;
            for (int k = 0; k < nA; ++k) {
                V2 = A[k];
                AA[k] = Matlab.sum(V2.times(V2));
            }
            final double[] BB = new double[nB];
            for (int i = 0; i < nB; ++i) {
                V2 = B[i];
                BB[i] = Matlab.sum(V2.times(V2));
            }
            final double[][] resData2 = ArrayOperator.allocate2DArray(nA, nB, 0.0);
            double[] resRow2 = null;
            for (int l = 0; l < nA; ++l) {
                resRow2 = resData2[l];
                V2 = A[l];
                for (int m = 0; m < nB; ++m) {
                    resRow2[m] = Matlab.innerProduct(V2, B[m]) / Math.sqrt(AA[l] * BB[m]);
                }
            }
            K = new DenseMatrix(resData2);
        }
        else if (kernelType.equals("poly")) {
            final double[][] resData = ArrayOperator.allocate2DArray(nA, nB, 0.0);
            double[] resRow = null;
            Vector V = null;
            for (int i = 0; i < nA; ++i) {
                resRow = resData[i];
                V = A[i];
                for (int j = 0; j < nB; ++j) {
                    resRow[j] = Math.pow(Matlab.innerProduct(V, B[j]), kernelParam);
                }
            }
            K = new DenseMatrix(resData);
        }
        else if (kernelType.equals("rbf")) {
            K = Matlab.l2DistanceSquare(A, B);
            InPlaceOperator.timesAssign(K, -1.0 / (2.0 * Math.pow(kernelParam, 2.0)));
            InPlaceOperator.expAssign(K);
        }
        return K;
    }
    
    public static Matrix calcKernel(final String kernelType, final double kernelParam, final Matrix X1, final Matrix X2) {
        Matrix K = null;
        if (kernelType.equals("linear")) {
            K = X1.transpose().mtimes(X2);
        }
        else if (kernelType.equals("cosine")) {
            final double[] AA = Matlab.sum(Matlab.times(X1, X1), 2).getPr();
            final double[] BB = Matlab.sum(Matlab.times(X2, X2), 2).getPr();
            final Matrix AB = X1.mtimes(X2.transpose());
            final int M = AB.getRowDimension();
            final int N = AB.getColumnDimension();
            double v = 0.0;
            if (AB instanceof DenseMatrix) {
                final double[][] resData = ((DenseMatrix)AB).getData();
                double[] resRow = null;
                for (int i = 0; i < M; ++i) {
                    resRow = resData[i];
                    v = AA[i];
                    for (int j = 0; j < N; ++j) {
                        final double[] array = resRow;
                        final int n = j;
                        array[n] /= Math.sqrt(v * BB[j]);
                    }
                }
            }
            else if (AB instanceof SparseMatrix) {
                final double[] pr = ((SparseMatrix)AB).getPr();
                final int[] ir = ((SparseMatrix)AB).getIr();
                final int[] jc = ((SparseMatrix)AB).getJc();
                for (int j = 0; j < N; ++j) {
                    v = BB[j];
                    for (int k = jc[j]; k < jc[j + 1]; ++k) {
                        final double[] array2 = pr;
                        final int n2 = k;
                        array2[n2] /= Math.sqrt(AA[ir[k]] * v);
                    }
                }
            }
            K = AB;
        }
        else if (kernelType.equals("poly")) {
            K = Matlab.pow(X1.mtimes(X2.transpose()), kernelParam);
        }
        else if (kernelType.equals("rbf")) {
            K = Matlab.l2DistanceSquare(X1, X2);
            InPlaceOperator.timesAssign(K, -1.0 / (2.0 * Math.pow(kernelParam, 2.0)));
            InPlaceOperator.expAssign(K);
        }
        return K;
    }
}
