package ml.options;

public class Options
{
    public boolean verbose;
    public int nFeature;
    public int nClass;
    public int nTopic;
    public int nTerm;
    public int nDoc;
    public int nTopTerm;
    public double epsilon;
    public int maxIter;
    public double gamma;
    public double mu;
    public double lambda;
    public boolean calc_OV;
    public int nClus;
    
    public Options(final Options o) {
        this.verbose = o.verbose;
        this.nFeature = o.nFeature;
        this.nClass = o.nClass;
        this.nTopic = o.nTopic;
        this.nTerm = o.nTerm;
        this.nDoc = o.nDoc;
        this.nClus = o.nClus;
        this.nTopTerm = o.nTopTerm;
        this.epsilon = o.epsilon;
        this.maxIter = o.maxIter;
        this.gamma = o.gamma;
        this.mu = o.mu;
        this.calc_OV = o.calc_OV;
        this.lambda = o.lambda;
    }
    
    public Options() {
        this.verbose = false;
        this.nFeature = 1;
        this.nClass = 1;
        this.nTopic = 1;
        this.nTerm = 1;
        this.nDoc = 0;
        this.nTopTerm = this.nTerm;
        this.epsilon = 1.0E-6;
        this.maxIter = 300;
        this.gamma = 1.0E-4;
        this.mu = 0.1;
        this.calc_OV = false;
        this.lambda = 1.0;
        this.nClus = 1;
    }
}
