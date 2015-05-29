package ml.utils;

public class Time
{
    private static double t;
    
    static {
        Time.t = 0.0;
    }
    
    public static double tic() {
        return Time.t = System.currentTimeMillis() / 1000.0;
    }
    
    public static double toc() {
        return System.currentTimeMillis() / 1000.0 - Time.t;
    }
    
    public static double toc(final double TSTART) {
        return System.currentTimeMillis() / 1000.0 - TSTART;
    }
    
    public static void pause(final double n) {
        try {
            Thread.sleep((long)n * 1000L);
        }
        catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}
