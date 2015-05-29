package ml.options;

public class KMeansOptions
{
    public int nClus;
    public int maxIter;
    public boolean verbose;
    
    public KMeansOptions() {
        this.nClus = -1;
        this.maxIter = 100;
        this.verbose = false;
    }
    
    public KMeansOptions(final int nClus, final int maxIter, final boolean verbose) {
        this.nClus = nClus;
        this.maxIter = maxIter;
        this.verbose = verbose;
    }
    
    public KMeansOptions(final int nClus, final int maxIter) {
        this.nClus = nClus;
        this.maxIter = maxIter;
        this.verbose = false;
    }
}
