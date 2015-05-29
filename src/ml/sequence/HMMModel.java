package ml.sequence;

import java.io.*;

class HMMModel implements Serializable
{
    private static final long serialVersionUID = -3585978995931113277L;
    public int N;
    public int M;
    public double[] pi;
    public double[][] A;
    public double[][] B;
    
    public HMMModel(final double[] pi, final double[][] A, final double[][] B) {
        this.pi = pi;
        this.A = A;
        this.B = B;
        this.N = A.length;
        this.M = B[0].length;
    }
}
