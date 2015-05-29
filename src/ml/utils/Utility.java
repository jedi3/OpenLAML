package ml.utils;

import java.io.*;
import java.util.*;

public class Utility
{
    public static String readFileAsString(final String filePath) throws IOException {
        final byte[] buffer = new byte[(int)new File(filePath).length()];
        BufferedInputStream f = null;
        try {
            f = new BufferedInputStream(new FileInputStream(filePath));
            f.read(buffer);
        }
        finally {
            if (f != null) {
                try {
                    f.close();
                }
                catch (IOException ex) {}
            }
        }
        if (f != null) {
            try {
                f.close();
            }
            catch (IOException ex2) {}
        }
        return new String(buffer);
    }
    
    public static String[] splitCommand(final String command) {
        final char[] commandCharArr = command.toCharArray();
        int idx = 0;
        int beginPos = 0;
        int endPos = 0;
        char ch = ' ';
        int state = 0;
        String argument = "";
        final ArrayList<String> argList = new ArrayList<String>();
        while (idx < commandCharArr.length) {
            ch = commandCharArr[idx];
            if (ch == '\"' && state == 0) {
                beginPos = idx + 1;
                state = 1;
            }
            else if (ch == '\"' && state == 1) {
                state = 2;
            }
            else if (ch == ' ') {
                if (state == 2) {
                    endPos = idx - 1;
                    state = 0;
                }
                else {
                    if (state == 1) {
                        ++idx;
                        continue;
                    }
                    endPos = idx;
                }
                argument = command.substring(beginPos, endPos).trim();
                if (!argument.isEmpty()) {
                    argList.add(argument);
                }
                while (idx < commandCharArr.length && commandCharArr[idx] == ' ') {
                    ++idx;
                }
                if (idx == commandCharArr.length) {
                    break;
                }
                beginPos = idx;
                continue;
            }
            else if (idx == commandCharArr.length - 1) {
                endPos = idx + 1;
                argument = command.substring(beginPos, endPos).trim();
                if (!argument.isEmpty()) {
                    argList.add(argument);
                    break;
                }
                break;
            }
            ++idx;
        }
        return argList.toArray(new String[argList.size()]);
    }
    
    public static <K extends Comparable<K>, V> Map<K, V> sortByKeys(final Map<K, V> map, final String order) {
        final Comparator<K> keyComparator = new Comparator<K>() {
            @Override
            public int compare(final K k1, final K k2) {
                int compare = 0;
                if (order.compareTo("descend") == 0) {
                    compare = k2.compareTo(k1);
                }
                else if (order.compareTo("ascend") == 0) {
                    compare = k1.compareTo(k2);
                }
                else {
                    System.err.println("order should be either \"descend\" or \"ascend\"!");
                }
                if (compare == 0) {
                    return 1;
                }
                return compare;
            }
        };
        final Map<K, V> sortedByKeys = new TreeMap<K, V>(keyComparator);
        sortedByKeys.putAll((Map<? extends K, ? extends V>)map);
        return sortedByKeys;
    }
    
    public static <K, V extends Comparable<V>> Map<K, V> sortByValues(final Map<K, V> map, final String order) {
        final Comparator<K> valueComparator = new Comparator<K>() {
            @Override
            public int compare(final K k1, final K k2) {
                int compare = 0;
                if (order.compareTo("descend") == 0) {
                    compare = map.get(k2).compareTo(map.get(k1));
                }
                else if (order.compareTo("ascend") == 0) {
                    compare = map.get(k1).compareTo(map.get(k2));
                }
                else {
                    System.err.println("order should be either \"descend\" or \"ascend\"!");
                }
                if (compare == 0) {
                    return 1;
                }
                return compare;
            }
        };
        final Map<K, V> sortedByValues = new TreeMap<K, V>(valueComparator);
        sortedByValues.putAll((Map<? extends K, ? extends V>)map);
        return sortedByValues;
    }
    
    public static <K, V extends Comparable<? super V>> Map<K, V> sortByValue(final Map<K, V> map, final String order) {
        final List<Map.Entry<K, V>> list = new LinkedList<Map.Entry<K, V>>(map.entrySet());
        if (order.compareTo("ascend") == 0) {
            Collections.sort(list, new Comparator<Map.Entry<K, V>>() {
                @Override
                public int compare(final Map.Entry<K, V> o1, final Map.Entry<K, V> o2) {
                    return o1.getValue().compareTo(o2.getValue());
                }
            });
        }
        else if (order.compareTo("descend") == 0) {
            Collections.sort(list, new Comparator<Map.Entry<K, V>>() {
                @Override
                public int compare(final Map.Entry<K, V> o1, final Map.Entry<K, V> o2) {
                    return o2.getValue().compareTo(o1.getValue());
                }
            });
        }
        else {
            System.err.println("order should be either \"descend\" or \"ascend\"!");
        }
        final Map<K, V> result = new LinkedHashMap<K, V>();
        for (final Map.Entry<K, V> entry : list) {
            result.put(entry.getKey(), entry.getValue());
        }
        return result;
    }
    
    public static <K extends Comparable<? super K>, V> Map<K, V> sortByKey(final Map<K, V> map, final String order) {
        final List<Map.Entry<K, V>> list = new LinkedList<Map.Entry<K, V>>(map.entrySet());
        if (order.compareTo("ascend") == 0) {
            Collections.sort(list, new Comparator<Map.Entry<K, V>>() {
                @Override
                public int compare(final Map.Entry<K, V> o1, final Map.Entry<K, V> o2) {
                    return o1.getKey().compareTo(o2.getKey());
                }
            });
        }
        else if (order.compareTo("descend") == 0) {
            Collections.sort(list, new Comparator<Map.Entry<K, V>>() {
                @Override
                public int compare(final Map.Entry<K, V> o1, final Map.Entry<K, V> o2) {
                    return o2.getKey().compareTo(o1.getKey());
                }
            });
        }
        else {
            System.err.println("order should be either \"descend\" or \"ascend\"!");
        }
        final Map<K, V> result = new LinkedHashMap<K, V>();
        for (final Map.Entry<K, V> entry : list) {
            result.put(entry.getKey(), entry.getValue());
        }
        return result;
    }
    
    public static void exit(final int code) {
        System.exit(code);
    }
    
    public static class ArrayIndexComparator<V extends Comparable<? super V>> implements Comparator<Integer>
    {
        private final V[] array;
        
        public ArrayIndexComparator(final V[] array) {
            this.array = array;
        }
        
        public Integer[] createIndexArray() {
            final Integer[] idxVector = new Integer[this.array.length];
            for (int i = 0; i < this.array.length; ++i) {
                idxVector[i] = i;
            }
            return idxVector;
        }
        
        @Override
        public int compare(final Integer index1, final Integer index2) {
            return this.array[index2].compareTo(this.array[index1]);
        }
    }
    
    public static class keyAscendComparator<K extends Comparable<K>> implements Comparator<K>
    {
        @Override
        public int compare(final K k1, final K k2) {
            return k1.compareTo(k2);
        }
    }
    
    public static class keyDescendComparator<K extends Comparable<K>> implements Comparator<K>
    {
        @Override
        public int compare(final K k1, final K k2) {
            return k2.compareTo(k1);
        }
    }
}
