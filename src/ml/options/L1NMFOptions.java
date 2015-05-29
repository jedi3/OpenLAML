package ml.options;

public class L1NMFOptions extends ClusteringOptions
{
    public double gamma;
    public double mu;
    public double epsilon;
    public boolean calc_OV;
    
    public L1NMFOptions() {
        this.gamma = 1.0E-4;
        this.mu = 0.1;
        this.epsilon = 1.0E-6;
        this.calc_OV = false;
    }
    
    public L1NMFOptions(final L1NMFOptions L1NMFOptions) {
        super(L1NMFOptions);
        this.gamma = L1NMFOptions.gamma;
        this.mu = L1NMFOptions.mu;
        this.epsilon = L1NMFOptions.epsilon;
        this.calc_OV = L1NMFOptions.calc_OV;
    }
    
    public L1NMFOptions(final int nClus) {
        super(nClus);
        this.gamma = 1.0E-4;
        this.mu = 0.1;
        this.epsilon = 1.0E-6;
        this.calc_OV = false;
    }
    
    public L1NMFOptions(final int nClus, final boolean verbose, final int maxIter) {
        super(nClus, verbose, maxIter);
        this.gamma = 1.0E-4;
        this.mu = 0.1;
        this.epsilon = 1.0E-6;
        this.calc_OV = false;
    }
    
    public L1NMFOptions(final ClusteringOptions options) {
        super(options);
        this.gamma = 1.0E-4;
        this.mu = 0.1;
        this.epsilon = 1.0E-6;
        this.calc_OV = false;
    }
}
