# dementia_classifier
Code for my masters thesis, which contains the work from the following publications:
- Masrani, V., Murray, G., Field, T. S., & Carenini, G. (2017). Domain adaptation for detecting mild cognitive impairment. In Canadian Conference on Artificial Intelligence (pp. 248–259). Springer, Cham.
- Masrani, V., Murray, G., Field, T., & Carenini, G. (2017). Detecting dementia through retrospective analysis of routine blog posts by bloggers with dementia. In BioNLP 2017 (pp. 232–237).
- Field, T. S., Masrani, V., Murray, G., & Carenini, G. (2017). Improving Diagnostic Accuracy Of Alzheimer S Disease From Speech Analysis Using Markers Of Hemispatial Neglect. Alzheimer’s & Dementia: The Journal of the Alzheimer’s Association, 13(7), P157–P158.

. To run: 
- Request access to the [DementiaBank dataset](https://dementia.talkbank.org) from either Davida Fromm: fromm@andrew.cmu.edu or Brian MacWhinney: macw@cmu.edu  
- Once you have permission, email me at vadmas@cs.ubc.ca to get a copy of the preprocessed dataset.
    -  Place data alongside run.py.
    -  Place lib within dementia_classifier/ 
- Install python requirements (may want to use a virtualenv)
- Note: May need to install the 'stopwords' and 'punkt' package for the NLTK python package. 
- Make sure sql is installed on your system and create a database with the appropriate permissions to store the processed data and results. Modify dementia_classifier/db.py with the appropriate user, password, and database name.
- Start the stanford parser with:
    ```bash
    java -Xmx4g -cp "dementia_classifier/lib/stanford/stanford-corenlp-full-2015-12-09/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 20000
    ```
- run with:
  ```python
  python run.py
  ```


## Troubleshooting
Error: "RuntimeError: Python is not installed as a framework. The Mac OS X backend will not be able to function correctly if Python is not installed as a framework. See the Python documentation for more information on installing Python as a framework on Mac OS X. Please either reinstall Python as a framework, or try one of the other backends. If you are using (Ana)Conda please install python.app and replace the use of 'python' with 'pythonw'. See 'Working with Matplotlib on OSX' in the Matplotlib FAQ for more information.
"
See: https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python

Error: "ValueError: invalid literal for int() with base 10: 'sh: dementia_classifier/lib/SCA/L2SCA/./tregex.sh: Permission denied'"

Check permissions for tregex.sh:

chmod 755 dementia_classifier/lib/SCA/L2SCA/tregex.sh
