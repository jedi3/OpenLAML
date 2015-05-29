package ml.options;

public class NMFOptions extends ClusteringOptions
{
    public double epsilon;
    public boolean calc_OV;
    
    public NMFOptions() {
        this.epsilon = 1.0E-6;
        this.calc_OV = false;
    }
    
    public NMFOptions(final NMFOptions NMFOptions) {
        super(NMFOptions);
        this.epsilon = NMFOptions.epsilon;
        this.calc_OV = NMFOptions.calc_OV;
    }
    
    public NMFOptions(final int nClus) {
        super(nClus);
        this.epsilon = 1.0E-6;
        this.calc_OV = false;
    }
    
    public NMFOptions(final int nClus, final boolean verbose, final int maxIter) {
        super(nClus, verbose, maxIter);
        this.epsilon = 1.0E-6;
        this.calc_OV = false;
    }
    
    public NMFOptions(final ClusteringOptions options) {
        super(options);
        this.epsilon = 1.0E-6;
        this.calc_OV = false;
    }
}
