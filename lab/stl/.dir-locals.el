(;; Default to c++ mode for *any* file for which c mode would normally
 ;; be used within the "include" subdirectory. This is needed so that
 ;; the ".h" header files are treated as C++ headers.
 ("include" . ((c-mode . ((mode . c++) (c-basic-offset . 2)))))
 ;; Set the indentation level for all C++ files.
 (c++-mode . ((c-basic-offset . 2))))
