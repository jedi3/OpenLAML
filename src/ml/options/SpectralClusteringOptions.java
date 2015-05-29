package ml.options;

public class SpectralClusteringOptions extends ClusteringOptions
{
    public String graphType;
    public double graphParam;
    public String graphDistanceFunction;
    public String graphWeightType;
    public double graphWeightParam;
    
    public SpectralClusteringOptions() {
        this.graphType = "nn";
        this.graphParam = 6.0;
        this.graphDistanceFunction = "euclidean";
        this.graphWeightType = "heat";
        this.graphWeightParam = 1.0;
    }
    
    public SpectralClusteringOptions(final int nClus) {
        super(nClus);
        this.graphType = "nn";
        this.graphParam = 6.0;
        this.graphDistanceFunction = "euclidean";
        this.graphWeightType = "heat";
        this.graphWeightParam = 1.0;
    }
    
    public SpectralClusteringOptions(final int nClus, final boolean verbose, final int maxIter, final String graphType, final double graphParam, final String graphDistanceFunction, final String graphWeightType, final double graphWeightParam) {
        super(nClus, verbose, maxIter);
        this.graphType = graphType;
        this.graphParam = graphParam;
        this.graphDistanceFunction = graphDistanceFunction;
        this.graphWeightType = graphWeightType;
        this.graphWeightParam = graphWeightParam;
    }
    
    public SpectralClusteringOptions(final ClusteringOptions clusteringOptions) {
        super(clusteringOptions);
        if (clusteringOptions instanceof SpectralClusteringOptions) {
            final SpectralClusteringOptions options = (SpectralClusteringOptions)clusteringOptions;
            this.graphType = options.graphType;
            this.graphParam = options.graphParam;
            this.graphDistanceFunction = options.graphDistanceFunction;
            this.graphWeightType = options.graphWeightType;
            this.graphWeightParam = options.graphWeightParam;
        }
        else {
            this.graphType = "nn";
            this.graphParam = 6.0;
            this.graphDistanceFunction = "euclidean";
            this.graphWeightType = "heat";
            this.graphWeightParam = 1.0;
        }
    }
}
