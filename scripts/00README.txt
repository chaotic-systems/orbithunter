convert_folder.py              Converts folder of invariant 2-tori data stored in .h5 to perseus-formatted-textfile for use with GUDHI
			       (http://gudhi.gforge.inria.fr/)

trawl.py                       Torus converging code (merged kstori.py and kstori_auto.py)

continuation.py    	       Numerical continuation code (merged kstori_continuation.py &&
			       kstori_continuation_auto.py && kstori_continuation_NM.py)

glue.py                        Combines 2-tori to find new 2-torus (kstori_glue_tests.py 
                               && kstori_glue_auto.py)

manual_subdomains.py           Script to manually chop up solutions into subdomains and attempt to converge result (frankenstein code)

persistence_diagrams.py        produces persistence diagram figures (animated or static) for folder of txt_persistence GUDHI output files

refurbish.py  	               reproduce and save data/figures to specified location

remaster.py                    Code that converts stored data to current conventions to prevent errors.

playground.py                  Miscellaneous tests. 