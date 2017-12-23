# dementia_classifier
Code for my masters thesis. To run: 
- Request the missing data + lib directory from vadmas@gmail.com (too large to store on github) 
    -  Place data alongside run.py.
    -  Place lib within dementia_classifier/ 
- Install python requirements (may want to use a virtualenv)
- Make sure sql is installed on your system and create a database with the appropriate permissions to store the processed data and results. Modify dementia_classifier/db.py with the appropriate user, password, and database name.
- Start the stanford parser with:
    ```bash
    java -Xmx4g -cp "dementia_classifier/lib/stanford/stanford-corenlp-full-2015-12-09/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 20000
    ```
- run with:
  ```python
  python run.py
  ```

