package ml.sequence;

import java.util.*;
import ml.utils.*;
import java.io.*;

public class HMM
{
    int D;
    int[] lengths;
    int[][] Os;
    int[][] Qs;
    int N;
    int M;
    double[] pi;
    double[][] A;
    double[][] B;
    double epsilon;
    int maxIter;
    
    public static void main(final String[] args) {
        final int numStates = 3;
        final int numObservations = 2;
        final double epsilon = 1.0E-8;
        final int maxIter = 10;
        final double[] pi = { 0.33, 0.33, 0.34 };
        final double[][] A = { { 0.5, 0.3, 0.2 }, { 0.3, 0.5, 0.2 }, { 0.2, 0.4, 0.4 } };
        final double[][] B = { { 0.7, 0.3 }, { 0.5, 0.5 }, { 0.4, 0.6 } };
        final int D = 10000;
        final int T_min = 5;
        final int T_max = 10;
        final int[][][] data = generateDataSequences(D, T_min, T_max, pi, A, B);
        final int[][] Os = data[0];
        final int[][] Qs = data[1];
        final boolean trainHMM = true;
        if (trainHMM) {
            final HMM HMM = new HMM(numStates, numObservations, epsilon, maxIter);
            HMM.feedData(Os);
            HMM.feedLabels(Qs);
            HMM.train();
            Printer.fprintf("True Model Parameters: \n", new Object[0]);
            Printer.fprintf("Initial State Distribution: \n", new Object[0]);
            Printer.display(pi);
            Printer.fprintf("State Transition Probability Matrix: \n", new Object[0]);
            Printer.display(A);
            Printer.fprintf("Observation Probability Matrix: \n", new Object[0]);
            Printer.display(B);
            Printer.fprintf("Trained Model Parameters: \n", new Object[0]);
            Printer.fprintf("Initial State Distribution: \n", new Object[0]);
            Printer.display(HMM.pi);
            Printer.fprintf("State Transition Probability Matrix: \n", new Object[0]);
            Printer.display(HMM.A);
            Printer.fprintf("Observation Probability Matrix: \n", new Object[0]);
            Printer.display(HMM.B);
            final String HMMModelFilePath = "HMMModel.dat";
            HMM.saveModel(HMMModelFilePath);
        }
        final int ID = new Random().nextInt(D);
        final int[] O = Os[ID];
        final HMM HMMt = new HMM();
        HMMt.loadModel("HMMModel.dat");
        final int[] Q = HMMt.predict(O);
        Printer.fprintf("Observation sequence: \n", new Object[0]);
        HMMt.showObservationSequence(O);
        Printer.fprintf("True state sequence: \n", new Object[0]);
        HMMt.showStateSequence(Qs[ID]);
        Printer.fprintf("Predicted state sequence: \n", new Object[0]);
        HMMt.showStateSequence(Q);
        final double p = HMMt.evaluate(O);
        System.out.format("P(O|Theta) = %f\n", p);
    }
    
    public HMM() {
        this.N = 0;
        this.M = 0;
        this.pi = null;
        this.A = null;
        this.B = null;
        this.Os = null;
        this.Qs = null;
        this.epsilon = 0.001;
        this.maxIter = 500;
    }
    
    public HMM(final int N, final int M, final double epsilon, final int maxIter) {
        this.N = N;
        this.M = M;
        this.pi = new double[N];
        for (int i = 0; i < N; ++i) {
            this.pi[i] = 0.0;
        }
        this.A = new double[N][];
        for (int i = 0; i < N; ++i) {
            this.A[i] = new double[N];
            for (int j = 0; j < N; ++j) {
                this.A[i][j] = 0.0;
            }
        }
        this.B = new double[N][];
        for (int i = 0; i < N; ++i) {
            this.B[i] = new double[M];
            for (int k = 0; k < M; ++k) {
                this.B[i][k] = 0.0;
            }
        }
        this.Os = null;
        this.Qs = null;
        this.epsilon = epsilon;
        this.maxIter = maxIter;
    }
    
    public HMM(final int N, final int M) {
        this.N = N;
        this.M = M;
        this.pi = new double[N];
        for (int i = 0; i < N; ++i) {
            this.pi[i] = 0.0;
        }
        this.A = new double[N][];
        for (int i = 0; i < N; ++i) {
            this.A[i] = new double[N];
            for (int j = 0; j < N; ++j) {
                this.A[i][j] = 0.0;
            }
        }
        this.B = new double[N][];
        for (int i = 0; i < N; ++i) {
            this.B[i] = new double[M];
            for (int k = 0; k < M; ++k) {
                this.B[i][k] = 0.0;
            }
        }
        this.Os = null;
        this.Qs = null;
        this.epsilon = 1.0E-6;
        this.maxIter = 1000;
    }
    
    public void feedData(final int[][] Os) {
        this.Os = Os;
    }
    
    public void feedLabels(final int[][] Qs) {
        this.Qs = Qs;
    }
    
    public double evaluate2(final int[] O) {
        final int T = O.length;
        double[] alpha_t = new double[this.N];
        for (int i = 0; i < this.N; ++i) {
            alpha_t[i] = this.pi[i] * this.B[i][O[0]];
        }
        double[] alpha_t_plus_1 = new double[this.N];
        double[] temp = null;
        double sum = 0.0;
        int t = 1;
        do {
            for (int j = 0; j < this.N; ++j) {
                sum = 0.0;
                for (int k = 0; k < this.N; ++k) {
                    sum += alpha_t[k] * this.A[k][j] * this.B[j][O[t]];
                }
                alpha_t_plus_1[j] = sum;
            }
            temp = alpha_t;
            alpha_t = alpha_t_plus_1;
            alpha_t_plus_1 = temp;
        } while (++t < T);
        return ArrayOperator.sum(alpha_t);
    }
    
    public double evaluate(final int[] O) {
        final int T = O.length;
        final double[] c = ArrayOperator.allocateVector(T);
        double[] alpha_hat_t = ArrayOperator.allocateVector(this.N);
        double[] alpha_hat_t_plus_1 = ArrayOperator.allocateVector(this.N);
        double[] temp_alpha = null;
        double log_likelihood = 0.0;
        for (int t = 0; t < T; ++t) {
            if (t == 0) {
                for (int i = 0; i < this.N; ++i) {
                    alpha_hat_t[i] = this.pi[i] * this.B[i][O[0]];
                }
            }
            else {
                ArrayOperator.clearVector(alpha_hat_t_plus_1);
                for (int j = 0; j < this.N; ++j) {
                    for (int k = 0; k < this.N; ++k) {
                        final double[] array = alpha_hat_t_plus_1;
                        final int n = j;
                        array[n] += alpha_hat_t[k] * this.A[k][j] * this.B[j][O[t]];
                    }
                }
                temp_alpha = alpha_hat_t;
                alpha_hat_t = alpha_hat_t_plus_1;
                alpha_hat_t_plus_1 = temp_alpha;
            }
            ArrayOperator.timesAssign(alpha_hat_t, c[t] = 1.0 / ArrayOperator.sum(alpha_hat_t));
            log_likelihood -= Math.log(c[t]);
        }
        return Math.exp(log_likelihood);
    }
    
    public int[] predict2(final int[] O) {
        final int T = O.length;
        final int[] Q = new int[T];
        double[] delta_t = new double[this.N];
        double[] delta_t_plus_1 = new double[this.N];
        final int[][] psi = new int[T][];
        for (int t = 0; t < T; ++t) {
            psi[t] = new int[this.N];
        }
        final double[] V = new double[this.N];
        for (int i = 0; i < this.N; ++i) {
            delta_t[i] = this.pi[i] * this.B[i][O[0]];
        }
        int t2 = 1;
        int maxIdx = -1;
        double maxVal = 0.0;
        do {
            for (int j = 0; j < this.N; ++j) {
                for (int k = 0; k < this.N; ++k) {
                    V[k] = delta_t[k] * this.A[k][j];
                }
                maxIdx = ArrayOperator.argmax(V);
                maxVal = V[maxIdx];
                delta_t_plus_1[j] = maxVal * this.B[j][O[t2 + 1 - 1]];
                psi[t2 + 1 - 1][j] = maxIdx;
            }
            double[] temp = null;
            temp = delta_t;
            delta_t = delta_t_plus_1;
            delta_t_plus_1 = temp;
        } while (++t2 < T);
        int i_t = ArrayOperator.argmax(delta_t);
        Q[T - 1] = i_t;
        t2 = T;
        do {
            i_t = psi[t2 - 1][i_t];
            Q[t2 - 1 - 1] = i_t;
        } while (--t2 > 1);
        return Q;
    }
    
    public int[] predict(final int[] O) {
        final int T = O.length;
        final int[] Q = new int[T];
        double[] phi_t = ArrayOperator.allocateVector(this.N);
        double[] phi_t_plus_1 = ArrayOperator.allocateVector(this.N);
        double[] temp_phi = null;
        final int[][] psi = new int[T][];
        for (int t = 0; t < T; ++t) {
            psi[t] = new int[this.N];
        }
        final double[] V = ArrayOperator.allocateVector(this.N);
        for (int i = 0; i < this.N; ++i) {
            phi_t[i] = Math.log(this.pi[i]) + Math.log(this.B[i][O[0]]);
        }
        int t2 = 1;
        int maxIdx = -1;
        double maxVal = 0.0;
        do {
            for (int j = 0; j < this.N; ++j) {
                for (int k = 0; k < this.N; ++k) {
                    V[k] = phi_t[k] + Math.log(this.A[k][j]);
                }
                maxIdx = ArrayOperator.argmax(V);
                maxVal = V[maxIdx];
                phi_t_plus_1[j] = maxVal + Math.log(this.B[j][O[t2]]);
                psi[t2][j] = maxIdx;
            }
            temp_phi = phi_t;
            phi_t = phi_t_plus_1;
            phi_t_plus_1 = temp_phi;
        } while (++t2 < T);
        int i_t = ArrayOperator.argmax(phi_t);
        Q[T - 1] = i_t;
        t2 = T;
        do {
            i_t = psi[t2 - 1][i_t];
            Q[t2 - 1 - 1] = i_t;
        } while (--t2 > 1);
        return Q;
    }
    
    public void train() {
        final int D = this.Os.length;
        int T_n = 0;
        double log_likelihood = 0.0;
        double log_likelihood_new = 0.0;
        final double epsilon = this.epsilon;
        final int maxIter = this.maxIter;
        ArrayOperator.clearVector(this.pi);
        ArrayOperator.clearMatrix(this.A);
        ArrayOperator.clearMatrix(this.B);
        final double[] a = ArrayOperator.allocateVector(this.N);
        final double[] b = ArrayOperator.allocateVector(this.N);
        int[] Q_n = null;
        int[] O_n = null;
        if (this.Qs == null) {
            this.pi = this.initializePi();
            this.A = this.initializeA();
            this.B = this.initializeB();
        }
        else {
            for (int n = 0; n < D; ++n) {
                Q_n = this.Qs[n];
                O_n = this.Os[n];
                T_n = this.Os[n].length;
                for (int t = 0; t < T_n; ++t) {
                    if (t < T_n - 1) {
                        final double[] array = this.A[Q_n[t]];
                        final int n3 = Q_n[t + 1];
                        ++array[n3];
                        final double[] array2 = a;
                        final int n4 = Q_n[t];
                        ++array2[n4];
                        if (t == 0) {
                            final double[] pi = this.pi;
                            final int n5 = Q_n[0];
                            ++pi[n5];
                        }
                    }
                    final double[] array3 = this.B[Q_n[t]];
                    final int n6 = O_n[t];
                    ++array3[n6];
                    final double[] array4 = b;
                    final int n7 = Q_n[t];
                    ++array4[n7];
                }
            }
            ArrayOperator.divideAssign(this.pi, D);
            for (int i = 0; i < this.N; ++i) {
                ArrayOperator.divideAssign(this.A[i], a[i]);
                ArrayOperator.divideAssign(this.B[i], b[i]);
            }
        }
        int s = 0;
        double[] pi_new = ArrayOperator.allocateVector(this.N);
        double[][] A_new = ArrayOperator.allocateMatrix(this.N, this.N);
        double[][] B_new = ArrayOperator.allocateMatrix(this.N, this.M);
        double[] temp_pi = null;
        double[][] temp_A = null;
        double[][] temp_B = null;
        double[][] alpha_hat = null;
        double[][] beta_hat = null;
        double[] c_n = null;
        final double[][] xi = ArrayOperator.allocateMatrix(this.N, this.N);
        final double[] gamma = ArrayOperator.allocateVector(this.N);
        do {
            ArrayOperator.clearVector(pi_new);
            ArrayOperator.clearMatrix(A_new);
            ArrayOperator.clearMatrix(B_new);
            ArrayOperator.clearVector(a);
            ArrayOperator.clearVector(b);
            log_likelihood_new = 0.0;
            for (int n2 = 0; n2 < D; ++n2) {
                O_n = this.Os[n2];
                T_n = this.Os[n2].length;
                c_n = ArrayOperator.allocateVector(T_n);
                alpha_hat = ArrayOperator.allocateMatrix(T_n, this.N);
                beta_hat = ArrayOperator.allocateMatrix(T_n, this.N);
                for (int t2 = 0; t2 <= T_n - 1; ++t2) {
                    if (t2 == 0) {
                        for (int j = 0; j < this.N; ++j) {
                            alpha_hat[0][j] = this.pi[j] * this.B[j][O_n[0]];
                        }
                    }
                    else {
                        for (int k = 0; k < this.N; ++k) {
                            for (int l = 0; l < this.N; ++l) {
                                final double[] array5 = alpha_hat[t2];
                                final int n8 = k;
                                array5[n8] += alpha_hat[t2 - 1][l] * this.A[l][k] * this.B[k][O_n[t2]];
                            }
                        }
                    }
                    c_n[t2] = 1.0 / ArrayOperator.sum(alpha_hat[t2]);
                    ArrayOperator.timesAssign(alpha_hat[t2], c_n[t2]);
                }
                for (int t2 = T_n + 1; t2 >= 2; --t2) {
                    if (t2 == T_n + 1) {
                        for (int j = 0; j < this.N; ++j) {
                            beta_hat[t2 - 2][j] = 1.0;
                        }
                    }
                    if (t2 <= T_n) {
                        for (int j = 0; j < this.N; ++j) {
                            for (int m = 0; m < this.N; ++m) {
                                final double[] array6 = beta_hat[t2 - 2];
                                final int n9 = j;
                                array6[n9] += this.A[j][m] * this.B[m][O_n[t2 - 1]] * beta_hat[t2 - 1][m];
                            }
                        }
                    }
                    ArrayOperator.timesAssign(beta_hat[t2 - 2], c_n[t2 - 2]);
                }
                for (int t2 = 0; t2 <= T_n - 1; ++t2) {
                    if (t2 < T_n - 1) {
                        for (int j = 0; j < this.N; ++j) {
                            for (int m = 0; m < this.N; ++m) {
                                xi[j][m] = alpha_hat[t2][j] * this.A[j][m] * this.B[m][O_n[t2 + 1]] * beta_hat[t2 + 1][m];
                            }
                            ArrayOperator.plusAssign(A_new[j], xi[j]);
                            gamma[j] = ArrayOperator.sum(xi[j]);
                        }
                        if (t2 == 0) {
                            ArrayOperator.plusAssign(pi_new, gamma);
                        }
                        ArrayOperator.plusAssign(a, gamma);
                    }
                    else {
                        ArrayOperator.assignVector(gamma, alpha_hat[t2]);
                    }
                    for (int k = 0; k < this.N; ++k) {
                        final double[] array7 = B_new[k];
                        final int n10 = O_n[t2];
                        array7[n10] += gamma[k];
                    }
                    ArrayOperator.plusAssign(b, gamma);
                    log_likelihood_new += -Math.log(c_n[t2]);
                }
            }
            ArrayOperator.sum2one(pi_new);
            for (int i2 = 0; i2 < this.N; ++i2) {
                ArrayOperator.divideAssign(A_new[i2], a[i2]);
            }
            for (int j2 = 0; j2 < this.N; ++j2) {
                ArrayOperator.divideAssign(B_new[j2], b[j2]);
            }
            temp_pi = this.pi;
            this.pi = pi_new;
            pi_new = temp_pi;
            temp_A = this.A;
            this.A = A_new;
            A_new = temp_A;
            temp_B = this.B;
            this.B = B_new;
            B_new = temp_B;
            if (++s > 1 && Math.abs((log_likelihood_new - log_likelihood) / log_likelihood) < epsilon) {
                Printer.fprintf("log[P(O|Theta)] does not increase.\n\n", new Object[0]);
                break;
            }
            log_likelihood = log_likelihood_new;
            Printer.fprintf("Iter: %d, log[P(O|Theta)]: %f\n", s, log_likelihood);
        } while (s < maxIter);
    }
    
    public double[] genDiscreteDistribution(final int n) {
        Random generator = null;
        generator = new Random();
        final double[] res = ArrayOperator.allocateVector(n);
        do {
            for (int i = 0; i < n; ++i) {
                res[i] = generator.nextDouble();
            }
        } while (ArrayOperator.sum(res) == 0.0);
        ArrayOperator.divideAssign(res, ArrayOperator.sum(res));
        return res;
    }
    
    private double[][] initializeB() {
        final double[][] res = new double[this.N][];
        for (int i = 0; i < this.N; ++i) {
            res[i] = this.genDiscreteDistribution(this.M);
        }
        return res;
    }
    
    private double[][] initializeA() {
        final double[][] res = new double[this.N][];
        for (int i = 0; i < this.N; ++i) {
            res[i] = this.genDiscreteDistribution(this.N);
        }
        return res;
    }
    
    private double[] initializePi() {
        return this.genDiscreteDistribution(this.N);
    }
    
    public void saveModel(final String filePath) {
        final File parentFile = new File(filePath).getParentFile();
        if (parentFile != null && !parentFile.exists()) {
            parentFile.mkdirs();
        }
        try {
            final ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath));
            oos.writeObject(new HMMModel(this.pi, this.A, this.B));
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
            final HMMModel HMMModel = (HMMModel)ois.readObject();
            this.N = HMMModel.N;
            this.M = HMMModel.M;
            this.pi = HMMModel.pi;
            this.A = HMMModel.A;
            this.B = HMMModel.B;
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
    
    public void setQs(final int[][] Qs) {
        this.Qs = Qs;
    }
    
    public void setOs(final int[][] Os) {
        this.Os = Os;
    }
    
    public void setPi(final double[] pi) {
        this.pi = pi;
    }
    
    public void setA(final double[][] A) {
        this.A = A;
    }
    
    public void setB(final double[][] B) {
        this.B = B;
    }
    
    public double[] getPi() {
        return this.pi;
    }
    
    public double[][] getA() {
        return this.A;
    }
    
    public double[][] getB() {
        return this.B;
    }
    
    public void showStateSequence(final int[] Q) {
        for (int t = 0; t < Q.length; ++t) {
            System.out.format("%d ", Q[t]);
        }
        System.out.println();
    }
    
    public void showObservationSequence(final int[] O) {
        for (int t = 0; t < O.length; ++t) {
            System.out.format("%d ", O[t]);
        }
        System.out.println();
    }
    
    public static int[][][] generateDataSequences(final int D, final int T_min, final int T_max, final double[] pi, final double[][] A, final double[][] B) {
        final int[][][] res = new int[2][][];
        final int[][] Os = new int[D][];
        final int[][] Qs = new int[D][];
        final int N = A.length;
        final int M = B[0].length;
        double[] distribution = null;
        double sum = 0.0;
        final Random generator = new Random();
        double rndRealScalor = 0.0;
        for (int n = 0; n < D; ++n) {
            final int T_n = generator.nextInt(T_max - T_min + 1) + T_min;
            final int[] O_n = new int[T_n];
            final int[] Q_n = new int[T_n];
            for (int t = 0; t < T_n; ++t) {
                rndRealScalor = generator.nextDouble();
                if (t == 0) {
                    distribution = pi;
                }
                else {
                    distribution = A[Q_n[t - 1]];
                }
                sum = 0.0;
                for (int i = 0; i < N; ++i) {
                    sum += distribution[i];
                    if (rndRealScalor <= sum) {
                        Q_n[t] = i;
                        break;
                    }
                }
                rndRealScalor = generator.nextDouble();
                distribution = B[Q_n[t]];
                sum = 0.0;
                for (int k = 0; k < M; ++k) {
                    sum += distribution[k];
                    if (rndRealScalor <= sum) {
                        O_n[t] = k;
                        break;
                    }
                }
            }
            Os[n] = O_n;
            Qs[n] = Q_n;
        }
        res[0] = Os;
        res[1] = Qs;
        return res;
    }
}
