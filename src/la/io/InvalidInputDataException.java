package la.io;

import java.io.*;

public class InvalidInputDataException extends Exception
{
    private static final long serialVersionUID = 2945131732407207308L;
    private final int _line;
    private File _file;
    
    public InvalidInputDataException(final String message, final File file, final int line) {
        super(message);
        this._file = file;
        this._line = line;
    }
    
    public InvalidInputDataException(final String message, final int line) {
        super(message);
        this._file = null;
        this._line = line;
    }
    
    public InvalidInputDataException(final String message, final String filename, final int line) {
        this(message, new File(filename), line);
    }
    
    public InvalidInputDataException(final String message, final File file, final int lineNr, final Exception cause) {
        super(message, cause);
        this._file = file;
        this._line = lineNr;
    }
    
    public InvalidInputDataException(final String message, final int lineNr, final Exception cause) {
        super(message, cause);
        this._file = null;
        this._line = lineNr;
    }
    
    public InvalidInputDataException(final String message, final String filename, final int lineNr, final Exception cause) {
        this(message, new File(filename), lineNr, cause);
    }
    
    public File getFile() {
        return this._file;
    }
    
    public int getLine() {
        return this._line;
    }
    
    @Override
    public String toString() {
        return String.valueOf(super.toString()) + " (" + this._file + ":" + this._line + ")";
    }
}
