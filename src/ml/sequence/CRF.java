package ml.sequence;

import java.util.*;

import la.matrix.*;
import ml.optimization.*;
import la.vector.*;
import la.vector.Vector;
import ml.utils.*;

import java.io.*;

public class CRF
{
    Matrix[][][] Fs;
    int D;
    int[] lengths;
    int d;
    int numStates;
    int startIdx;
    int[][] Ys;
    Vector W;
    double epsilon;
    double sigma;
    int maxIter;
    
    public static void main(final String[] args) {
        final int D = 1000;
        final int n_min = 4;
        final int n_max = 6;
        final int d = 10;
        final int N = 2;
        final double sparseness = 0.2;
        final Object[] dataSequences = generateDataSequences(D, n_min, n_max, d, N, sparseness);
        final Matrix[][][] Fs = (Matrix[][][])dataSequences[0];
        final int[][] Ys = (int[][])dataSequences[1];
        final double sigma = 2.0;
        final double epsilon = 1.0E-4;
        CRF CRF = new CRF(sigma, epsilon);
        CRF.feedData(Fs);
        CRF.feedLabels(Ys);
        CRF.train();
        final String modelFilePath = "CRF-Model.dat";
        CRF.saveModel(modelFilePath);
        Printer.fprintf("CRF Parameters:\n", new Object[0]);
        Printer.display(CRF.W);
        CRF = new CRF();
        CRF.loadModel(modelFilePath);
        final int ID = new Random().nextInt(D);
        final int[] Yt = Ys[ID];
        final Matrix[][] Fst = Fs[ID];
        Printer.fprintf("True label sequence:\n", new Object[0]);
        Printer.display(Yt);
        Printer.fprintf("Predicted label sequence:\n", new Object[0]);
        Printer.display(CRF.predict(Fst));
    }
    
    public CRF() {
        this.sigma = 1.0;
        this.maxIter = 50;
        this.sigma = 1.0;
        this.epsilon = 1.0E-4;
        this.startIdx = 0;
        this.maxIter = 50;
    }
    
    public CRF(final double sigma) {
        this.sigma = 1.0;
        this.maxIter = 50;
        this.sigma = sigma;
        this.epsilon = 1.0E-4;
        this.startIdx = 0;
        this.maxIter = 50;
    }
    
    public CRF(final double sigma, final double epsilon) {
        this.sigma = 1.0;
        this.maxIter = 50;
        this.sigma = sigma;
        this.epsilon = epsilon;
        this.startIdx = 0;
        this.maxIter = 50;
    }
    
    public CRF(final double sigma, final double epsilon, final int maxIter) {
        this.sigma = 1.0;
        this.maxIter = 50;
        this.sigma = sigma;
        this.epsilon = epsilon;
        this.startIdx = 0;
        this.maxIter = maxIter;
    }
    
    public void feedData(final Matrix[][][] Fs) {
        this.Fs = Fs;
        this.D = Fs.length;
        this.d = Fs[0][0].length;
        this.numStates = Fs[0][0][0].getRowDimension();
    }
    
    public void feedLabels(final int[][] Ys) {
        this.Ys = Ys;
    }
    
    public static Object[] generateDataSequences(final int D, final int n_min, final int n_max, final int d, final int N, final double sparseness) {
        final Object[] res = new Object[2];
        final Matrix[][][] Fs = new Matrix[D][][];
        final int[][] Ys = new int[D][];
        final Random generator = new Random();
        double prob = 0.0;
        double feaVal = 0.0;
        for (int k = 0; k < D; ++k) {
            final int n_x = generator.nextInt(n_max - n_min + 1) + n_min;
            Fs[k] = new Matrix[n_x][];
            Ys[k] = new int[n_x];
            for (int i = 0; i < n_x; ++i) {
                Ys[k][i] = generator.nextInt(N);
                Fs[k][i] = new Matrix[d];
                for (int j = 0; j < d; ++j) {
                    Fs[k][i][j] = new SparseMatrix(N, N);
                    for (int previous = 0; previous < N; ++previous) {
                        for (int current = 0; current < N; ++current) {
                            prob = generator.nextDouble();
                            if (prob < sparseness) {
                                feaVal = 1.0;
                                Fs[k][i][j].setEntry(previous, current, feaVal);
                            }
                        }
                    }
                }
            }
        }
        res[0] = Fs;
        res[1] = Ys;
        return res;
    }
    
    public void train() {
        double fval = 0.0;
        final int maxSequenceLength = this.computeMaxSequenceLength();
        final Matrix[] Ms = new Matrix[maxSequenceLength];
        for (int i = 0; i < maxSequenceLength; ++i) {
            Ms[i] = new DenseMatrix(this.numStates, this.numStates);
        }
        final Vector F = this.computeGlobalFeatureVector();
        this.W = new DenseVector(this.d, 10.0);
        final Vector Grad = new DenseVector(this.d);
        fval = this.computeObjectiveFunctionValue(F, Ms, true, Grad, this.W);
        boolean[] flags = null;
        int k = 0;
        do {
            flags = LBFGSForVector.run(Grad, fval, this.epsilon, this.W);
            if (flags[0]) {
                break;
            }
            fval = this.computeObjectiveFunctionValue(F, Ms, flags[1], Grad, this.W);
        } while (!flags[1] || ++k <= this.maxIter);
    }
    
    private int computeMaxSequenceLength() {
        int maxSeqLen = 0;
        int n_x = 0;
        int[] Y = null;
        for (int k = 0; k < this.D; ++k) {
            Y = this.Ys[k];
            n_x = Y.length;
            if (maxSeqLen < n_x) {
                maxSeqLen = n_x;
            }
        }
        return maxSeqLen;
    }
    
    private Vector computeGlobalFeatureVector() {
        final Vector F = new DenseVector(this.d, 1.0);
        int n_x = 0;
        double f = 0.0;
        int[] Y = null;
        Matrix[][] Fs_k = null;
        for (int j = 0; j < this.d; ++j) {
            f = 0.0;
            for (int k = 0; k < this.D; ++k) {
                Y = this.Ys[k];
                Fs_k = this.Fs[k];
                n_x = Fs_k.length;
                for (int i = 0; i < n_x; ++i) {
                    if (i == 0) {
                        f += Fs_k[i][j].getEntry(0, Y[i]);
                    }
                    else {
                        f += Fs_k[i][j].getEntry(Y[i - 1], Y[i]);
                    }
                }
            }
            F.set(j, f);
        }
        return F;
    }
    
    private double computeObjectiveFunctionValue(final Vector F, final Matrix[] Ms, final boolean calcGrad, final Vector Grad, final Vector W) {
        double fval = 0.0;
        Vector EF = null;
        final double[] EFArr = ArrayOperator.allocateVector(this.d);
        int n_x = 0;
        Matrix[][] Fs_k = null;
        Matrix f_j_x_i = null;
        for (int k = 0; k < this.D; ++k) {
            Fs_k = this.Fs[k];
            n_x = Fs_k.length;
            for (int i = 0; i < n_x; ++i) {
                Ms[i].clear();
                for (int j = 0; j < this.d; ++j) {
                    f_j_x_i = Fs_k[i][j];
                    if (j == 0) {
                        InPlaceOperator.times(Ms[i], W.get(j), f_j_x_i);
                    }
                    else {
                        InPlaceOperator.plusAssign(Ms[i], W.get(j), f_j_x_i);
                    }
                }
                InPlaceOperator.expAssign(Ms[i]);
            }
            final Vector[] Alpha_hat = new Vector[n_x];
            Vector Alpha_hat_0 = null;
            final Vector e_start = new SparseVector(this.numStates);
            e_start.set(this.startIdx, 1.0);
            final double[] c = ArrayOperator.allocateVector(n_x);
            Alpha_hat_0 = e_start;
            for (int l = 0; l < n_x; ++l) {
                if (l == 0) {
                    Alpha_hat[l] = Alpha_hat_0.operate(Ms[l]);
                }
                else {
                    Alpha_hat[l] = Alpha_hat[l - 1].operate(Ms[l]);
                }
                c[l] = 1.0 / Matlab.sum(Alpha_hat[l]);
                InPlaceOperator.timesAssign(Alpha_hat[l], c[l]);
            }
            final Vector[] Beta_hat = new Vector[n_x];
            for (int m = n_x - 1; m >= 0; --m) {
                if (m == n_x - 1) {
                    Beta_hat[m] = new DenseVector(this.numStates, 1.0);
                }
                else {
                    Beta_hat[m] = Ms[m + 1].operate(Beta_hat[m + 1]);
                }
                InPlaceOperator.timesAssign(Beta_hat[m], c[m]);
            }
            for (int m = 0; m < n_x; ++m) {
                fval -= Math.log(c[m]);
            }
            if (calcGrad) {
                for (int j2 = 0; j2 < this.d; ++j2) {
                    for (int i2 = 0; i2 < n_x; ++i2) {
                        if (i2 == 0) {
                            final double[] array = EFArr;
                            final int n = j2;
                            array[n] += Matlab.innerProduct(Alpha_hat_0, Ms[i2].times(f_j_x_i).operate(Beta_hat[i2]));
                        }
                        else {
                            final double[] array2 = EFArr;
                            final int n2 = j2;
                            array2[n2] += Matlab.innerProduct(Alpha_hat[i2 - 1], Ms[i2].times(f_j_x_i).operate(Beta_hat[i2]));
                        }
                    }
                }
            }
        }
        fval -= Matlab.innerProduct(W, F);
        fval += this.sigma * Matlab.innerProduct(W, W);
        fval /= this.D;
        if (!calcGrad) {
            return fval;
        }
        EF = new DenseVector(EFArr);
        InPlaceOperator.times(Grad, 1.0 / this.D, Matlab.plus(Matlab.minus(EF, F), W.times(2.0 * this.sigma)));
        return fval;
    }
    
    public int[] predict(final Matrix[][] Fs) {
        final Matrix[] Ms = this.computeTransitionMatrix(Fs);
        final int n_x = Fs.length;
        final double[] b = ArrayOperator.allocateVector(n_x);
        final Vector[] Beta_tilta = new Vector[n_x];
        for (int i = n_x - 1; i >= 0; --i) {
            if (i == n_x - 1) {
                Beta_tilta[i] = new DenseVector(this.numStates, 1.0);
            }
            else {
                Beta_tilta[i] = Ms[i + 1].operate(Beta_tilta[i + 1]);
            }
            b[i] = 1.0 / Matlab.sum(Beta_tilta[i]);
            InPlaceOperator.timesAssign(Beta_tilta[i], b[i]);
        }
        final double[][] Gamma_i = ArrayOperator.allocate2DArray(this.numStates, this.numStates, 0.0);
        final double[][] Phi = ArrayOperator.allocate2DArray(n_x, this.numStates, 0.0);
        final double[][] Psi = ArrayOperator.allocate2DArray(n_x, this.numStates, 0.0);
        double[][] M_i = null;
        double[] M_i_Row = null;
        double[] Gamma_i_Row = null;
        double[] Beta_tilta_i = null;
        double[] Phi_i = null;
        double[] Phi_im1 = null;
        double[][] maxResult = null;
        for (int j = 0; j < n_x; ++j) {
            M_i = ((DenseMatrix)Ms[j]).getData();
            Beta_tilta_i = ((DenseVector)Beta_tilta[j]).getPr();
            for (int y_im1 = 0; y_im1 < this.numStates; ++y_im1) {
                M_i_Row = M_i[y_im1];
                Gamma_i_Row = Gamma_i[y_im1];
                InPlaceOperator.assign(Gamma_i_Row, M_i_Row);
                ArrayOperator.timesAssign(Gamma_i_Row, Beta_tilta_i);
                ArrayOperator.sum2one(Gamma_i_Row);
            }
            Phi_i = Phi[j];
            if (j == 0) {
                InPlaceOperator.log(Phi_i, Gamma_i[this.startIdx]);
            }
            else {
                Phi_im1 = Phi[j - 1];
                for (int y_im1 = 0; y_im1 < this.numStates; ++y_im1) {
                    Gamma_i_Row = Gamma_i[y_im1];
                    InPlaceOperator.logAssign(Gamma_i_Row);
                    ArrayOperator.plusAssign(Gamma_i_Row, Phi_im1[y_im1]);
                }
                maxResult = Matlab.max(Gamma_i, 1);
                Phi[j] = maxResult[0];
                Psi[j] = maxResult[1];
            }
        }
        final double[] phi_n_x = Phi[n_x - 1];
        final int[] YPred = ArrayOperator.allocateIntegerVector(n_x);
        for (int k = n_x - 1; k >= 0; --k) {
            if (k == n_x - 1) {
                YPred[k] = ArrayOperator.argmax(phi_n_x);
            }
            else {
                YPred[k] = (int)Psi[k + 1][YPred[k + 1]];
            }
        }
        final double p = Math.exp(phi_n_x[YPred[n_x - 1]]);
        Printer.fprintf("P*(YPred|x) = %g\n", p);
        return YPred;
    }
    
    private Matrix[] computeTransitionMatrix(final Matrix[][] Fs) {
        final int n_x = Fs.length;
        Matrix f_j_x_i = null;
        final Matrix[] Ms = new Matrix[n_x];
        for (int i = 0; i < n_x; ++i) {
            Ms[i] = new DenseMatrix(this.numStates, this.numStates, 0.0);
            for (int j = 0; j < this.d; ++j) {
                f_j_x_i = Fs[i][j];
                if (j == 0) {
                    InPlaceOperator.times(Ms[i], this.W.get(j), f_j_x_i);
                }
                else {
                    InPlaceOperator.plusAssign(Ms[i], this.W.get(j), f_j_x_i);
                }
            }
            InPlaceOperator.expAssign(Ms[i]);
        }
        return Ms;
    }
    
    private Vector computeFeatureVector(final Matrix[][] Fs, final int[] Ys) {
        final Vector F = new DenseVector(this.d);
        final int[] Y = null;
        int n_x = 0;
        double f = 0.0;
        for (int j = 0; j < this.d; ++j) {
            f = 0.0;
            n_x = Fs.length;
            for (int i = 0; i < n_x; ++i) {
                if (i == 0) {
                    f += Fs[i][j].getEntry(0, Ys[i]);
                }
                else {
                    f += Fs[i][j].getEntry(Ys[i - 1], Ys[i]);
                }
            }
            F.set(j, f);
        }
        return F;
    }
    
    private Vector[] backwardRecursion4Viterbi(final Matrix[] Ms) {
        final int n_x = Ms.length;
        final double[] b = ArrayOperator.allocateVector(n_x);
        final Vector[] Beta_tilta = new Vector[n_x];
        for (int i = n_x - 1; i >= 0; --i) {
            if (i == n_x - 1) {
                Beta_tilta[i] = new DenseVector(this.numStates, 1.0);
            }
            else {
                Beta_tilta[i] = Ms[i + 1].operate(Beta_tilta[i + 1]);
            }
            b[i] = 1.0 / Matlab.sum(Beta_tilta[i]);
            InPlaceOperator.timesAssign(Beta_tilta[i], b[i]);
        }
        return Beta_tilta;
    }
    
    public void saveModel(final String filePath) {
        final File parentFile = new File(filePath).getParentFile();
        if (parentFile != null && !parentFile.exists()) {
            parentFile.mkdirs();
        }
        try {
            final ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath));
            oos.writeObject(new CRFModel(this.numStates, this.startIdx, this.W));
            oos.close();
            System.out.println("Model saved.");
        }
        catch (FileNotFoundException e) {
            e.printStackTrace();
            System.exit(1);
        }
        catch (IOException e2) {
            e2.printStackTrace();
        }
    }
    
    public void loadModel(final String filePath) {
        System.out.println("Loading model...");
        try {
            final ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath));
            final CRFModel CRFModel = (CRFModel)ois.readObject();
            this.W = CRFModel.W;
            this.d = CRFModel.d;
            this.startIdx = CRFModel.startIdx;
            this.numStates = CRFModel.numStates;
            ois.close();
            System.out.println("Model loaded.");
        }
        catch (FileNotFoundException e) {
            e.printStackTrace();
            System.exit(1);
        }
        catch (IOException e2) {
            e2.printStackTrace();
        }
        catch (ClassNotFoundException e3) {
            e3.printStackTrace();
        }
    }
}
