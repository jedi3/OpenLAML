package ml.options;

public class GraphOptions
{
    public String graphType;
    public double graphParam;
    public String kernelType;
    public double kernelParam;
    public String graphDistanceFunction;
    public String graphWeightType;
    public double graphWeightParam;
    public boolean graphNormalize;
    public boolean classEdges;
    
    public GraphOptions() {
        this.graphType = "nn";
        this.kernelType = "linear";
        this.kernelParam = 1.0;
        this.graphParam = 6.0;
        this.graphDistanceFunction = "euclidean";
        this.graphWeightType = "binary";
        this.graphWeightParam = 1.0;
        this.graphNormalize = true;
        this.classEdges = false;
    }
    
    public GraphOptions(final GraphOptions graphOtions) {
        this.graphType = graphOtions.graphType;
        this.kernelType = graphOtions.kernelType;
        this.kernelParam = graphOtions.kernelParam;
        this.graphParam = graphOtions.graphParam;
        this.graphDistanceFunction = graphOtions.graphDistanceFunction;
        this.graphWeightType = graphOtions.graphWeightType;
        this.graphWeightParam = graphOtions.graphWeightParam;
        this.graphNormalize = graphOtions.graphNormalize;
        this.classEdges = graphOtions.classEdges;
    }
}
