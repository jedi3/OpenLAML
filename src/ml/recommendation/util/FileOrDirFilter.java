package ml.recommendation.util;

import java.io.*;

public class FileOrDirFilter implements FileFilter
{
    String ext;
    
    public FileOrDirFilter(final String ext) {
        this.ext = ext;
    }
    
    @Override
    public boolean accept(final File pathname) {
        return pathname.isDirectory() || pathname.getName().endsWith(this.ext);
    }
}
