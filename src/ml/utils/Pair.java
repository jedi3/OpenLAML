package ml.utils;

public class Pair<A extends Comparable<? super A>, B extends Comparable<? super B>> implements Comparable<Pair<A, B>>
{
    public A first;
    public B second;
    
    public Pair(final Pair<A, B> pair) {
        this.first = pair.first;
        this.second = pair.second;
    }
    
    public Pair(final A first, final B second) {
        this.first = first;
        this.second = second;
    }
    
    public static <A extends Comparable<? super A>, B extends Comparable<? super B>> Pair<A, B> of(final A first, final B second) {
        return new Pair<A, B>(first, second);
    }
    
    @Override
    public int compareTo(final Pair<A, B> o) {
        final int cmp = (o == null) ? 1 : this.first.compareTo(o.first);
        return (cmp == 0) ? this.second.compareTo(o.second) : cmp;
    }
    
    @Override
    public int hashCode() {
        return 31 * hashcode(this.first) + hashcode(this.second);
    }
    
    private static int hashcode(final Object o) {
        return (o == null) ? 0 : o.hashCode();
    }
    
    @Override
    public boolean equals(final Object obj) {
        return obj instanceof Pair && (this == obj || (this.equal(this.first, ((Pair)obj).first) && this.equal(this.second, ((Pair)obj).second)));
    }
    
    private boolean equal(final Object o1, final Object o2) {
        return o1 == o2 || (o1 != null && o1.equals(o2));
    }
    
    @Override
    public String toString() {
        return "(" + this.first + ", " + this.second + ')';
    }
    
    public Pair<A, B> clone() {
        return new Pair<A, B>(this);
    }
}
