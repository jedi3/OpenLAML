package ml.options;

public class LDAOptions
{
    public boolean hasLabel;
    public boolean verbose;
    public int nTopic;
    public int nTerm;
    public double alpha;
    public double beta;
    public int iterations;
    public int burnIn;
    public int thinInterval;
    public int sampleLag;
    
    public LDAOptions() {
        this.hasLabel = false;
        this.verbose = false;
        this.nTerm = 1;
        this.nTopic = 1;
        this.alpha = 50.0 / this.nTopic;
        this.beta = 200.0 / this.nTerm;
        this.iterations = 1000;
        this.burnIn = 500;
        this.thinInterval = 50;
        this.sampleLag = 10;
    }
}
