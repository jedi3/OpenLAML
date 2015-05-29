package ml.options;

public class ClusteringOptions
{
    public int nClus;
    public boolean verbose;
    public int maxIter;
    
    public ClusteringOptions() {
        this.nClus = 0;
        this.verbose = false;
        this.maxIter = 100;
    }
    
    public ClusteringOptions(final int nClus) {
        if (nClus < 1) {
            System.err.println("Number of clusters less than one!");
            System.exit(1);
        }
        this.nClus = nClus;
        this.verbose = false;
        this.maxIter = 100;
    }
    
    public ClusteringOptions(final int nClus, final boolean verbose, final int maxIter) {
        this.nClus = nClus;
        this.verbose = verbose;
        this.maxIter = maxIter;
    }
    
    public ClusteringOptions(final ClusteringOptions options) {
        this.nClus = options.nClus;
        this.verbose = options.verbose;
        this.maxIter = options.maxIter;
    }
}
