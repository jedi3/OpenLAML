package ml.topics;

import ml.options.*;
import java.text.*;
import java.util.*;
import la.matrix.*;

public class LdaGibbsSampler
{
    Corpus corpus;
    LDAOptions LDAOptions;
    int[][] documents;
    int V;
    int K;
    double alpha;
    double beta;
    int[][] z;
    int[][] nw;
    int[][] nd;
    int[] nwsum;
    int[] ndsum;
    double[][] thetasum;
    double[][] phisum;
    int numstats;
    private static int THIN_INTERVAL;
    private static int BURN_IN;
    private static int ITERATIONS;
    private static int SAMPLE_LAG;
    private static int dispcol;
    static String[] shades;
    static NumberFormat lnf;
    
    static {
        LdaGibbsSampler.THIN_INTERVAL = 20;
        LdaGibbsSampler.BURN_IN = 100;
        LdaGibbsSampler.ITERATIONS = 1000;
        LdaGibbsSampler.dispcol = 0;
        LdaGibbsSampler.shades = new String[] { "     ", ".    ", ":    ", ":.   ", "::   ", "::.  ", ":::  ", ":::. ", ":::: ", "::::.", ":::::" };
        LdaGibbsSampler.lnf = new DecimalFormat("00E0");
    }
    
    public static void main(final String[] args) {
        final int[][] documents = { { 1, 4, 3, 2, 3, 1, 4, 3, 2, 3, 1, 4, 3, 2, 3, 6 }, { 2, 2, 4, 2, 4, 2, 2, 2, 2, 4, 2, 2 }, { 1, 6, 5, 6, 0, 1, 6, 5, 6, 0, 1, 6, 5, 6, 0, 0 }, { 5, 6, 6, 2, 3, 3, 6, 5, 6, 2, 2, 6, 5, 6, 6, 6, 0 }, { 2, 2, 4, 4, 4, 4, 1, 5, 5, 5, 5, 5, 5, 1, 1, 1, 1, 0 }, { 5, 4, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2 } };
        final int V = 7;
        final int K = 2;
        final double alpha = 2.0;
        final double beta = 0.5;
        System.out.println("Latent Dirichlet Allocation using Gibbs Sampling.");
        final LdaGibbsSampler LDA = new LdaGibbsSampler(documents, V);
        LDA.configure(10000, 2000, 100, 10);
        LDA.gibbs(K, alpha, beta);
        final double[][] theta = LDA.getTheta();
        final double[][] phi = LDA.getPhi();
        System.out.println();
        System.out.println();
        System.out.println("Document--Topic Associations, Theta[d][k] (alpha=" + alpha + ")");
        System.out.print("d\\k\t");
        for (int m = 0; m < theta[0].length; ++m) {
            System.out.print("   " + m % 10 + "    ");
        }
        System.out.println();
        for (int m = 0; m < theta.length; ++m) {
            System.out.print(String.valueOf(m) + "\t");
            for (int k = 0; k < theta[m].length; ++k) {
                System.out.print(String.valueOf(shadeDouble(theta[m][k], 1.0)) + " ");
            }
            System.out.println();
        }
        System.out.println();
        System.out.println("Topic--Term Associations, Phi[k][w] (beta=" + beta + ")");
        System.out.print("k\\w\t");
        for (int w = 0; w < phi[0].length; ++w) {
            System.out.print("   " + w % 10 + "    ");
        }
        System.out.println();
        for (int i = 0; i < phi.length; ++i) {
            System.out.print(String.valueOf(i) + "\t");
            for (int w2 = 0; w2 < phi[i].length; ++w2) {
                System.out.print(String.valueOf(shadeDouble(phi[i][w2], 1.0)) + " ");
            }
            System.out.println();
        }
    }
    
    public LdaGibbsSampler(final int[][] documents, final int V) {
        this.documents = documents;
        this.V = V;
    }
    
    public LdaGibbsSampler(final LDAOptions LDAOptions) {
        this.documents = null;
        this.V = 0;
        this.corpus = new Corpus();
        this.LDAOptions = LDAOptions;
    }
    
    public LdaGibbsSampler() {
        this.documents = null;
        this.V = 0;
        this.corpus = new Corpus();
    }
    
    public void readCorpusFromDocTermCountArray(final ArrayList<TreeMap<Integer, Integer>> docTermCountArray) {
        this.corpus.readCorpusFromDocTermCountArray(docTermCountArray);
        this.documents = this.corpus.documents;
        this.V = this.corpus.nTerm;
    }
    
    public void readCorpusFromLDAInputFile(final String LDAInputDataFilePath) {
        this.corpus.readCorpusFromLDAInputFile(LDAInputDataFilePath);
        this.documents = this.corpus.documents;
        this.V = this.corpus.nTerm;
    }
    
    public void readCorpusFromDocTermCountFile(final String docTermCountFilePath) {
        this.corpus.readCorpusFromDocTermCountFile(docTermCountFilePath);
        this.documents = this.corpus.documents;
        this.V = this.corpus.nTerm;
    }
    
    public void readCorpusFromMatrix(final Matrix X) {
        this.corpus.readCorpusFromMatrix(X);
        this.documents = this.corpus.documents;
        this.V = this.corpus.nTerm;
    }
    
    public int[][] initialState(final int K) {
        final int M = this.documents.length;
        this.nw = new int[this.V][K];
        this.nd = new int[M][K];
        this.nwsum = new int[K];
        this.ndsum = new int[M];
        this.z = new int[M][];
        for (int m = 0; m < M; ++m) {
            final int N = this.documents[m].length;
            this.z[m] = new int[N];
            for (int n = 0; n < N; ++n) {
                final int topic = (int)(Math.random() * K);
                this.z[m][n] = topic;
                final int[] array = this.nw[this.documents[m][n]];
                final int n2 = topic;
                ++array[n2];
                final int[] array2 = this.nd[m];
                final int n3 = topic;
                ++array2[n3];
                final int[] nwsum = this.nwsum;
                final int n4 = topic;
                ++nwsum[n4];
            }
            this.ndsum[m] = N;
        }
        return this.z;
    }
    
    public void gibbs(final int K, final double alpha, final double beta) {
        this.K = K;
        this.alpha = alpha;
        this.beta = beta;
        if (LdaGibbsSampler.SAMPLE_LAG > 0) {
            this.thetasum = new double[this.documents.length][K];
            this.phisum = new double[K][this.V];
            this.numstats = 0;
        }
        this.initialState(K);
        System.out.println("Sampling " + LdaGibbsSampler.ITERATIONS + " iterations with burn-in of " + LdaGibbsSampler.BURN_IN + " (B/S=" + LdaGibbsSampler.THIN_INTERVAL + ").");
        LdaGibbsSampler.dispcol = 0;
        for (int i = 0; i < LdaGibbsSampler.ITERATIONS; ++i) {
            for (int m = 0; m < this.z.length; ++m) {
                for (int n = 0; n < this.z[m].length; ++n) {
                    final int topic = this.sampleFullConditional(m, n);
                    this.z[m][n] = topic;
                }
            }
            if (i < LdaGibbsSampler.BURN_IN && i % LdaGibbsSampler.THIN_INTERVAL == 0) {
                System.out.print("B");
                ++LdaGibbsSampler.dispcol;
            }
            if (i > LdaGibbsSampler.BURN_IN && i % LdaGibbsSampler.THIN_INTERVAL == 0) {
                System.out.print("S");
                ++LdaGibbsSampler.dispcol;
            }
            if (i > LdaGibbsSampler.BURN_IN && LdaGibbsSampler.SAMPLE_LAG > 0 && i % LdaGibbsSampler.SAMPLE_LAG == 0) {
                this.updateParams();
                System.out.print("|");
                if (i % LdaGibbsSampler.THIN_INTERVAL != 0) {
                    ++LdaGibbsSampler.dispcol;
                }
            }
            if (LdaGibbsSampler.dispcol >= 100) {
                System.out.println();
                LdaGibbsSampler.dispcol = 0;
            }
        }
        System.out.println();
    }
    
    private int sampleFullConditional(final int m, final int n) {
        int topic = this.z[m][n];
        final int[] array = this.nw[this.documents[m][n]];
        final int n2 = topic;
        --array[n2];
        final int[] array2 = this.nd[m];
        final int n3 = topic;
        --array2[n3];
        final int[] nwsum = this.nwsum;
        final int n4 = topic;
        --nwsum[n4];
        final int[] ndsum = this.ndsum;
        --ndsum[m];
        final double[] p = new double[this.K];
        for (int k = 0; k < this.K; ++k) {
            p[k] = (this.nw[this.documents[m][n]][k] + this.beta) / (this.nwsum[k] + this.V * this.beta) * (this.nd[m][k] + this.alpha) / (this.ndsum[m] + this.K * this.alpha);
        }
        for (int k = 1; k < p.length; ++k) {
            final double[] array3 = p;
            final int n5 = k;
            array3[n5] += p[k - 1];
        }
        double u;
        for (u = Math.random() * p[this.K - 1], topic = 0; topic < p.length && u >= p[topic]; ++topic) {}
        final int[] array4 = this.nw[this.documents[m][n]];
        final int n6 = topic;
        ++array4[n6];
        final int[] array5 = this.nd[m];
        final int n7 = topic;
        ++array5[n7];
        final int[] nwsum2 = this.nwsum;
        final int n8 = topic;
        ++nwsum2[n8];
        final int[] ndsum2 = this.ndsum;
        ++ndsum2[m];
        return topic;
    }
    
    private void updateParams() {
        for (int m = 0; m < this.documents.length; ++m) {
            for (int k = 0; k < this.K; ++k) {
                final double[] array = this.thetasum[m];
                final int n = k;
                array[n] += (this.nd[m][k] + this.alpha) / (this.ndsum[m] + this.K * this.alpha);
            }
        }
        for (int i = 0; i < this.K; ++i) {
            for (int t = 0; t < this.V; ++t) {
                final double[] array2 = this.phisum[i];
                final int n2 = t;
                array2[n2] += (this.nw[t][i] + this.beta) / (this.nwsum[i] + this.V * this.beta);
            }
        }
        ++this.numstats;
    }
    
    public double[][] getTheta() {
        final double[][] theta = new double[this.documents.length][this.K];
        if (LdaGibbsSampler.SAMPLE_LAG > 0) {
            for (int m = 0; m < this.documents.length; ++m) {
                for (int k = 0; k < this.K; ++k) {
                    theta[m][k] = this.thetasum[m][k] / this.numstats;
                }
            }
        }
        else {
            for (int m = 0; m < this.documents.length; ++m) {
                for (int k = 0; k < this.K; ++k) {
                    theta[m][k] = (this.nd[m][k] + this.alpha) / (this.ndsum[m] + this.K * this.alpha);
                }
            }
        }
        return theta;
    }
    
    public double[][] getPhi() {
        final double[][] phi = new double[this.K][this.V];
        if (LdaGibbsSampler.SAMPLE_LAG > 0) {
            for (int k = 0; k < this.K; ++k) {
                for (int t = 0; t < this.V; ++t) {
                    phi[k][t] = this.phisum[k][t] / this.numstats;
                }
            }
        }
        else {
            for (int k = 0; k < this.K; ++k) {
                for (int t = 0; t < this.V; ++t) {
                    phi[k][t] = (this.nw[t][k] + this.beta) / (this.nwsum[k] + this.V * this.beta);
                }
            }
        }
        return phi;
    }
    
    public static double[] hist(final double[] data, final int fmax) {
        final double[] hist = new double[data.length];
        double hmax = 0.0;
        for (int i = 0; i < data.length; ++i) {
            hmax = Math.max(data[i], hmax);
        }
        final double shrink = fmax / hmax;
        for (int j = 0; j < data.length; ++j) {
            hist[j] = shrink * data[j];
        }
        final NumberFormat nf = new DecimalFormat("00");
        String scale = "";
        for (int k = 1; k < fmax / 10 + 1; ++k) {
            scale = String.valueOf(scale) + "    .    " + k % 10;
        }
        System.out.println("x" + nf.format(hmax / fmax) + "\t0" + scale);
        for (int k = 0; k < hist.length; ++k) {
            System.out.print(String.valueOf(k) + "\t|");
            for (int l = 0; l < Math.round(hist[k]); ++l) {
                if ((l + 1) % 10 == 0) {
                    System.out.print("]");
                }
                else {
                    System.out.print("|");
                }
            }
            System.out.println();
        }
        return hist;
    }
    
    public void configure(final int iterations, final int burnIn, final int thinInterval, final int sampleLag) {
        LdaGibbsSampler.ITERATIONS = iterations;
        LdaGibbsSampler.BURN_IN = burnIn;
        LdaGibbsSampler.THIN_INTERVAL = thinInterval;
        LdaGibbsSampler.SAMPLE_LAG = sampleLag;
    }
    
    public void configure(final LDAOptions LDAOptions) {
        LdaGibbsSampler.ITERATIONS = LDAOptions.iterations;
        LdaGibbsSampler.BURN_IN = LDAOptions.burnIn;
        LdaGibbsSampler.THIN_INTERVAL = LDAOptions.thinInterval;
        LdaGibbsSampler.SAMPLE_LAG = LDAOptions.sampleLag;
    }
    
    public void run() {
        this.configure(this.LDAOptions);
        this.gibbs(this.LDAOptions.nTopic, this.LDAOptions.alpha, this.LDAOptions.beta);
    }
    
    public void run(final LDAOptions LDAOptions) {
        this.configure(LDAOptions);
        this.gibbs(LDAOptions.nTopic, LDAOptions.alpha, LDAOptions.beta);
    }
    
    public static void run(final Corpus corpus, final LDAOptions LDAOptions) {
        final int V = corpus.nTerm;
        final int[][] documents = corpus.getDocuments();
        System.out.println("Latent Dirichlet Allocation using Gibbs Sampling.");
        final LdaGibbsSampler LDA = new LdaGibbsSampler(documents, V);
        LDA.configure(500, 100, 50, 10);
        LDA.gibbs(LDAOptions.nTopic, LDAOptions.alpha, LDAOptions.beta);
        final double[][] theta = LDA.getTheta();
        final double[][] phi = LDA.getPhi();
        System.out.println();
        System.out.println();
        System.out.println("Document--Topic Associations, Theta[d][k] (alpha=" + LDAOptions.alpha + ")");
        System.out.print("d\\k\t");
        for (int m = 0; m < theta[0].length; ++m) {
            System.out.print("   " + m % 10 + "    ");
        }
        System.out.println();
        for (int m = 0; m < theta.length; ++m) {
            System.out.print(String.valueOf(m) + "\t");
            for (int k = 0; k < theta[m].length; ++k) {
                System.out.print(String.valueOf(shadeDouble(theta[m][k], 1.0)) + " ");
            }
            System.out.println();
        }
        System.out.println();
        System.out.println("Topic--Term Associations, Phi[k][w] (beta=" + LDAOptions.beta + ")");
        System.out.print("k\\w\t");
        for (int w = 0; w < phi[0].length; ++w) {
            System.out.print("   " + w % 10 + "    ");
        }
        System.out.println();
        for (int i = 0; i < phi.length; ++i) {
            System.out.print(String.valueOf(i) + "\t");
            for (int w2 = 0; w2 < phi[i].length; ++w2) {
                System.out.print(String.valueOf(shadeDouble(phi[i][w2], 1.0)) + " ");
            }
            System.out.println();
        }
    }
    
    public static String shadeDouble(final double d, final double max) {
        int a = (int)Math.floor(d * 10.0 / max + 0.5);
        if (a > 10 || a < 0) {
            String x = LdaGibbsSampler.lnf.format(d);
            a = 5 - x.length();
            for (int i = 0; i < a; ++i) {
                x = String.valueOf(x) + " ";
            }
            return "<" + x + ">";
        }
        return "[" + LdaGibbsSampler.shades[a] + "]";
    }
}
