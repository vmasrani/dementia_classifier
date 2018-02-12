# dementia_classifier
Code for my masters thesis. To run: 
- Request the missing data + lib directory from vadmas@gmail.com (too large to store on github) 
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
